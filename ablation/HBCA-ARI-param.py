import os
import sys
import gc
import random
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# =========================================================
# 0. 环境与路径
# =========================================================
os.environ["R_HOME"] = "/data2/liangyefeng/miniconda3/envs/graphst/lib/R"

project_root = "/data2/liangyefeng/My_GraphST_Innovation"
sys.path.insert(0, os.path.join(project_root, "src"))

from GraphST.GraphST import GraphST
import GraphST.utils as graphst_utils
from GraphST.utils import clustering

# =========================================================
# 1. patch mclust，完全沿用你原始 HBCA 代码逻辑
# =========================================================
def patched_mclust_R(adata, num_cluster, modelNames="EEE", used_obsm="emb_pca", random_seed=2020):
    import numpy as np
    import pandas as pd
    import rpy2.robjects as robjects
    from rpy2.robjects import r
    from rpy2.robjects import numpy2ri

    x = np.asarray(adata.obsm[used_obsm], dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"{used_obsm} must be 2D, but got shape {x.shape}")

    if not np.isfinite(x).all():
        bad_count = np.size(x) - np.isfinite(x).sum()
        raise ValueError(f"{used_obsm} contains NaN/Inf values, bad_count={bad_count}")

    numpy2ri.activate()
    robjects.globalenv["x_mat"] = x
    robjects.globalenv["n_cluster"] = int(num_cluster)
    robjects.globalenv["model_name"] = modelNames
    robjects.globalenv["seed"] = int(random_seed)

    r(
        """
        suppressMessages(library(mclust))
        set.seed(seed)
        x_mat <- as.matrix(x_mat)
        dimnames(x_mat) <- NULL

        res <- tryCatch(
          Mclust(x_mat, G=n_cluster, modelNames=model_name),
          error = function(e) NULL
        )

        if (is.null(res)) {
          cls <- rep(NA, nrow(x_mat))
        } else {
          cls <- res$classification
          if (is.null(cls)) {
            cls <- rep(NA, nrow(x_mat))
          }
        }
        """
    )

    mclust_res = np.array(r["cls"], dtype=object)
    cls_series = pd.Series(mclust_res, index=adata.obs_names)
    cls_series = pd.to_numeric(cls_series, errors="coerce")

    na_count = cls_series.isna().sum()
    if na_count > 0:
        raise ValueError(f"mclust returned {na_count} invalid labels.")

    adata.obs["mclust"] = cls_series.astype(int).astype("category")
    return adata


graphst_utils.mclust_R = patched_mclust_R

# =========================================================
# 2. 固定随机种子
# =========================================================
seed = 41
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("✅ 当前设备:", device)

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"

# =========================================================
# 3. 数据路径
# =========================================================
file_fold = "/data2/liangyefeng/My_GraphST_Innovation/data/Human-Breast-Cancer-Block-A/section1"
gt_dir = os.path.join(file_fold, "gt")

result_root = os.path.join(file_fold, "HBCA_GraphST_ParameterSensitivity_Ada")
os.makedirs(result_root, exist_ok=True)

# 是否保存每次参数扫描的 h5ad
SAVE_ALL_H5AD = False

# =========================================================
# 4. 读取数据
# =========================================================
adata = sc.read_visium(
    file_fold,
    count_file="section1_filtered_feature_bc_matrix.h5",
    load_images=True
)
adata.var_names_make_unique()

print("✅ adata shape:", adata.shape)

# =========================================================
# 5. 读取 GT
# =========================================================
gold_meta_path = os.path.join(gt_dir, "gold_metadata.tsv")
tissue_gt_path = os.path.join(gt_dir, "tissue_positions_list_GTs.txt")


def detect_label_col(df):
    candidates = [c for c in df.columns if any(k in c.lower() for k in [
        "label", "annotation", "region", "cluster", "layer", "ground", "gt"
    ])]
    if len(candidates) > 0:
        return candidates[0]
    return df.columns[-1]


def load_gt(obs_names):
    if os.path.exists(gold_meta_path):
        df = pd.read_csv(gold_meta_path, sep="\t")
        barcode_col = next((c for c in df.columns if "barcode" in c.lower()), None)
        label_col = detect_label_col(df)

        if barcode_col is not None:
            df = df.set_index(barcode_col)
            df.index = df.index.astype(str)
            obs_names = pd.Index(obs_names.astype(str))
            df = df.loc[obs_names]
            gt = df[label_col]
        else:
            gt = pd.Series(df[label_col].values, index=obs_names)

        return gt, gold_meta_path, label_col

    elif os.path.exists(tissue_gt_path):
        try:
            df = pd.read_csv(tissue_gt_path, sep="\t")
            if df.shape[1] == 1:
                df = pd.read_csv(tissue_gt_path, sep=",")
        except Exception:
            df = pd.read_csv(tissue_gt_path, sep=",")

        barcode_col = next((c for c in df.columns if "barcode" in c.lower()), None)
        label_col = detect_label_col(df)

        if barcode_col is not None:
            df = df.set_index(barcode_col)
            df.index = df.index.astype(str)
            obs_names = pd.Index(obs_names.astype(str))
            df = df.loc[obs_names]
            gt = df[label_col]
        else:
            gt = pd.Series(df[label_col].values, index=obs_names)

        return gt, tissue_gt_path, label_col

    else:
        raise FileNotFoundError("No GT file found.")


gt, gt_path, label_col = load_gt(adata.obs_names)
adata.obs["ground_truth"] = gt

print("✅ GT file:", gt_path)
print("✅ label column:", label_col)

adata_eval = adata[~pd.isnull(adata.obs["ground_truth"])].copy()
n_clusters = adata_eval.obs["ground_truth"].nunique()
print("✅ n_clusters from GT:", n_clusters)

# =========================================================
# 6. 参数敏感性中心点
#    以 Ada-GraphST 当前最优参数为中心做局部扫描
# =========================================================
BEST_ADA = {
    "mode": "explicit",
    "w_smooth": 0.05,
    "w_sharpen": 0.08,
    "gamma": 2.5,
    "warmup_epochs": 100,
    "use_learnable_proj": False,
}

BASELINE_CFG = {
    "mode": "baseline_default"
}

# 围绕中心点做“局部”敏感性
# 这样更符合参数敏感性分析，而不是重新做大范围搜索
SENSITIVITY_POOLS = OrderedDict({
    "w_smooth": [0.01, 0.03, 0.05, 0.07, 0.09],
    "w_sharpen": [0.02, 0.05, 0.08, 0.11, 0.14],
})

# 二维联合敏感性：只做最关键的 ws × wsh
HEATMAP_W_SMOOTH = [0.01, 0.03, 0.05, 0.07, 0.09]
HEATMAP_W_SHARPEN = [0.02, 0.05, 0.08, 0.11, 0.14]

# =========================================================
# 7. 工具函数
# =========================================================
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_graphst_once(adata_input, variant_name, variant_cfg, n_clusters, radius=50):
    print("\n" + "-" * 100)
    print(f"Running: {variant_name}")
    print("-" * 100)

    adata_run = adata_input.copy()

    if variant_cfg["mode"] == "baseline_default":
        model = GraphST(adata_run, device=device)
    else:
        model = GraphST(
            adata_run,
            device=device,
            random_seed=seed,
            epochs=600,
            dim_output=64,
            datatype="10X",
            w_smooth=variant_cfg["w_smooth"],
            w_sharpen=variant_cfg["w_sharpen"],
            gamma=variant_cfg["gamma"],
            warmup_epochs=variant_cfg.get("warmup_epochs", 200),
            use_learnable_proj=variant_cfg.get("use_learnable_proj", False),
        )

    out = model.train()
    if out is not None:
        adata_run = out

    if "emb" not in adata_run.obsm:
        raise KeyError("adata.obsm['emb'] not found after training.")

    n_pcs = min(20, adata_run.obsm["emb"].shape[1])
    adata_run.obsm["emb_pca"] = PCA(
        n_components=n_pcs,
        random_state=seed
    ).fit_transform(adata_run.obsm["emb"])

    clustering(
        adata_run,
        n_clusters,
        radius=radius,
        method="mclust",
        refinement=True
    )

    adata_eval = adata_run[~pd.isnull(adata_run.obs["ground_truth"])].copy()

    ari = metrics.adjusted_rand_score(
        adata_eval.obs["domain"],
        adata_eval.obs["ground_truth"]
    )
    nmi = normalized_mutual_info_score(
        adata_eval.obs["domain"],
        adata_eval.obs["ground_truth"]
    )

    print(f"ARI: {ari:.6f}")
    print(f"NMI: {nmi:.6f}")

    return ari, nmi, adata_run, adata_eval


def make_explicit_cfg_from_best(best_cfg, param_name=None, param_value=None, extra_updates=None):
    cfg = {
        "mode": "explicit",
        "w_smooth": best_cfg["w_smooth"],
        "w_sharpen": best_cfg["w_sharpen"],
        "gamma": best_cfg["gamma"],
        "warmup_epochs": best_cfg["warmup_epochs"],
        "use_learnable_proj": best_cfg["use_learnable_proj"],
    }

    if param_name is not None:
        cfg[param_name] = param_value

    if extra_updates is not None:
        for k, v in extra_updates.items():
            cfg[k] = v

    return cfg


def save_run_h5ad_if_needed(adata_run, save_path):
    if SAVE_ALL_H5AD:
        adata_run.write(save_path)


def annotate_points(ax, xs, ys, fmt="{:.3f}", fontsize=9):
    for x, y in zip(xs, ys):
        if pd.notna(y):
            ax.text(
                x, y + 0.003, fmt.format(y),
                ha="center", va="bottom",
                fontsize=fontsize, fontweight="bold"
            )


def plot_onefactor_curve(df, param_name, baseline_ari, baseline_nmi, best_value, out_dir):
    sub = df[df["param_name"] == param_name].copy()
    sub = sub.sort_values("param_value").reset_index(drop=True)

    x = sub["param_value"].values
    ari = sub["ARI"].values
    nmi = sub["NMI"].values

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.patch.set_facecolor("white")

    metric_info = [
        ("ARI", ari, baseline_ari, axes[0]),
        ("NMI", nmi, baseline_nmi, axes[1]),
    ]

    for metric_name, y, baseline_y, ax in metric_info:
        ax.plot(
            x, y,
            marker="o",
            linewidth=2.2,
            markersize=7
        )
        ax.axhline(baseline_y, linestyle="--", linewidth=1.5, label="Baseline")
        ax.axvline(best_value, linestyle=":", linewidth=1.5, label="Best Ada param")
        annotate_points(ax, x, y, fmt="{:.3f}", fontsize=8.5)

        ax.set_title(f"{param_name} sensitivity ({metric_name})", fontsize=12.5, fontweight="bold")
        ax.set_xlabel(param_name, fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(frameon=True)

    plt.suptitle(
        f"HBCA section1 Parameter Sensitivity: {param_name}\n"
        f"(other parameters fixed at Ada-GraphST best setting)",
        fontsize=14.5,
        fontweight="bold",
        y=1.03
    )
    plt.tight_layout()

    png = os.path.join(out_dir, f"sensitivity_{param_name}.png")
    pdf = os.path.join(out_dir, f"sensitivity_{param_name}.pdf")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()


def draw_heatmap(ax, pivot_df, metric_name, center_ws, center_wsh):
    mat = pivot_df.values.astype(float)
    im = ax.imshow(mat, aspect="auto")

    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels([f"{x:.2f}" for x in pivot_df.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels([f"{y:.2f}" for y in pivot_df.index])

    ax.set_xlabel("w_smooth", fontsize=11)
    ax.set_ylabel("w_sharpen", fontsize=11)
    ax.set_title(metric_name, fontsize=13, fontweight="bold")

    vmax = np.nanmax(mat)
    vmin = np.nanmin(mat)
    mid = (vmax + vmin) / 2.0

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            color = "white" if val >= mid else "black"
            ax.text(
                j, i, f"{val:.3f}",
                ha="center", va="center",
                fontsize=8.3, fontweight="bold",
                color=color
            )

    # 标记当前 Ada 最优点
    center_i = list(pivot_df.index).index(center_wsh)
    center_j = list(pivot_df.columns).index(center_ws)
    rect_center = Rectangle(
        (center_j - 0.5, center_i - 0.5), 1, 1,
        fill=False, edgecolor="black", linewidth=2.6
    )
    ax.add_patch(rect_center)

    # 标记热图里当前最优点
    best_flat = np.nanargmax(mat)
    best_i, best_j = np.unravel_index(best_flat, mat.shape)
    rect_best = Rectangle(
        (best_j - 0.5, best_i - 0.5), 1, 1,
        fill=False, edgecolor="red", linewidth=2.4, linestyle="--"
    )
    ax.add_patch(rect_best)

    return im


def plot_ws_wsh_heatmap(df, center_ws, center_wsh, out_dir):
    ari_pivot = pd.pivot_table(
        df,
        index="w_sharpen",
        columns="w_smooth",
        values="ARI",
        aggfunc="mean"
    ).sort_index().sort_index(axis=1)

    nmi_pivot = pd.pivot_table(
        df,
        index="w_sharpen",
        columns="w_smooth",
        values="NMI",
        aggfunc="mean"
    ).sort_index().sort_index(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    fig.patch.set_facecolor("white")

    im1 = draw_heatmap(axes[0], ari_pivot, "ARI", center_ws, center_wsh)
    im2 = draw_heatmap(axes[1], nmi_pivot, "NMI", center_ws, center_wsh)

    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label("ARI", fontsize=10.5)

    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label("NMI", fontsize=10.5)

    plt.suptitle(
        "HBCA section1 Joint Sensitivity Heatmap: w_smooth × w_sharpen\n"
        "(black box = current Ada best, red dashed = best in grid)",
        fontsize=14.5,
        fontweight="bold",
        y=1.03
    )
    plt.tight_layout()

    png = os.path.join(out_dir, "sensitivity_wsmooth_wsharpen_heatmap.png")
    pdf = os.path.join(out_dir, "sensitivity_wsmooth_wsharpen_heatmap.pdf")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()


# =========================================================
# 8. 先跑 Baseline
# =========================================================
baseline_ari, baseline_nmi, baseline_adata, baseline_eval = run_graphst_once(
    adata_input=adata,
    variant_name="Baseline",
    variant_cfg=BASELINE_CFG,
    n_clusters=n_clusters,
    radius=50
)

baseline_h5ad = os.path.join(result_root, "HBCA_section1_Baseline.h5ad")
baseline_adata.write(baseline_h5ad)

baseline_df = pd.DataFrame([{
    "Dataset": "Human-Breast-Cancer-Block-A_section1",
    "Variant": "Baseline",
    "ARI": baseline_ari,
    "NMI": baseline_nmi,
    "w_smooth": 0.0,
    "w_sharpen": 0.0,
    "gamma": np.nan,
    "warmup_epochs": np.nan,
    "use_learnable_proj": False,
    "save_h5ad": baseline_h5ad,
}])

del baseline_adata
clear_memory()

# =========================================================
# 9. 单因素参数敏感性
# =========================================================
onefactor_records = []

for param_name, pool in SENSITIVITY_POOLS.items():
    print("\n" + "=" * 100)
    print(f"📊 Running one-factor sensitivity for: {param_name}")
    print("=" * 100)

    for value in pool:
        variant_cfg = make_explicit_cfg_from_best(
            BEST_ADA,
            param_name=param_name,
            param_value=value
        )

        variant_name = f"AdaSensitivity_{param_name}_{value}"

        try:
            ari, nmi, adata_out, adata_eval_out = run_graphst_once(
                adata_input=adata,
                variant_name=variant_name,
                variant_cfg=variant_cfg,
                n_clusters=n_clusters,
                radius=50
            )

            save_h5ad = os.path.join(
                result_root,
                f"{variant_name}.h5ad"
            )
            save_run_h5ad_if_needed(adata_out, save_h5ad)

            onefactor_records.append({
                "Dataset": "Human-Breast-Cancer-Block-A_section1",
                "analysis_type": "one_factor",
                "param_name": param_name,
                "param_value": value,
                "ARI": ari,
                "NMI": nmi,
                "w_smooth": variant_cfg["w_smooth"],
                "w_sharpen": variant_cfg["w_sharpen"],
                "gamma": variant_cfg["gamma"],
                "warmup_epochs": variant_cfg["warmup_epochs"],
                "use_learnable_proj": variant_cfg["use_learnable_proj"],
                "save_h5ad": save_h5ad if SAVE_ALL_H5AD else "",
                "status": "success",
                "error": "",
            })

        except Exception as e:
            print(f"❌ Failed: {variant_name} | {e}")
            onefactor_records.append({
                "Dataset": "Human-Breast-Cancer-Block-A_section1",
                "analysis_type": "one_factor",
                "param_name": param_name,
                "param_value": value,
                "ARI": np.nan,
                "NMI": np.nan,
                "w_smooth": variant_cfg["w_smooth"],
                "w_sharpen": variant_cfg["w_sharpen"],
                "gamma": variant_cfg["gamma"],
                "warmup_epochs": variant_cfg["warmup_epochs"],
                "use_learnable_proj": variant_cfg["use_learnable_proj"],
                "save_h5ad": "",
                "status": "failed",
                "error": str(e),
            })

        try:
            del adata_out
        except:
            pass
        clear_memory()

onefactor_df = pd.DataFrame(onefactor_records)
onefactor_csv = os.path.join(result_root, "HBCA_section1_onefactor_sensitivity_results.csv")
onefactor_df.to_csv(onefactor_csv, index=False)

# =========================================================
# 10. 二维参数敏感性：w_smooth × w_sharpen
# =========================================================
heatmap_records = []

print("\n" + "=" * 100)
print("📊 Running joint sensitivity heatmap: w_smooth × w_sharpen")
print("=" * 100)

for ws in HEATMAP_W_SMOOTH:
    for wsh in HEATMAP_W_SHARPEN:
        variant_cfg = make_explicit_cfg_from_best(
            BEST_ADA,
            extra_updates={
                "w_smooth": ws,
                "w_sharpen": wsh
            }
        )

        variant_name = f"AdaGrid_ws_{ws}_wsh_{wsh}"

        try:
            ari, nmi, adata_out, adata_eval_out = run_graphst_once(
                adata_input=adata,
                variant_name=variant_name,
                variant_cfg=variant_cfg,
                n_clusters=n_clusters,
                radius=50
            )

            save_h5ad = os.path.join(
                result_root,
                f"{variant_name}.h5ad"
            )
            save_run_h5ad_if_needed(adata_out, save_h5ad)

            heatmap_records.append({
                "Dataset": "Human-Breast-Cancer-Block-A_section1",
                "analysis_type": "two_factor",
                "w_smooth": ws,
                "w_sharpen": wsh,
                "gamma": variant_cfg["gamma"],
                "warmup_epochs": variant_cfg["warmup_epochs"],
                "ARI": ari,
                "NMI": nmi,
                "save_h5ad": save_h5ad if SAVE_ALL_H5AD else "",
                "status": "success",
                "error": "",
            })

        except Exception as e:
            print(f"❌ Failed: {variant_name} | {e}")
            heatmap_records.append({
                "Dataset": "Human-Breast-Cancer-Block-A_section1",
                "analysis_type": "two_factor",
                "w_smooth": ws,
                "w_sharpen": wsh,
                "gamma": variant_cfg["gamma"],
                "warmup_epochs": variant_cfg["warmup_epochs"],
                "ARI": np.nan,
                "NMI": np.nan,
                "save_h5ad": "",
                "status": "failed",
                "error": str(e),
            })

        try:
            del adata_out
        except:
            pass
        clear_memory()

heatmap_df = pd.DataFrame(heatmap_records)
heatmap_csv = os.path.join(result_root, "HBCA_section1_wsmooth_wsharpen_heatmap_results.csv")
heatmap_df.to_csv(heatmap_csv, index=False)

# =========================================================
# 11. 画单因素敏感性曲线
# =========================================================
for param_name in SENSITIVITY_POOLS.keys():
    plot_onefactor_curve(
        df=onefactor_df[onefactor_df["status"] == "success"].copy(),
        param_name=param_name,
        baseline_ari=baseline_ari,
        baseline_nmi=baseline_nmi,
        best_value=BEST_ADA[param_name],
        out_dir=result_root
    )

# =========================================================
# 12. 画 ws × wsh 热图
# =========================================================
heatmap_ok = heatmap_df[heatmap_df["status"] == "success"].copy()
plot_ws_wsh_heatmap(
    df=heatmap_ok,
    center_ws=BEST_ADA["w_smooth"],
    center_wsh=BEST_ADA["w_sharpen"],
    out_dir=result_root
)

# =========================================================
# 13. 汇总表
# =========================================================
summary_rows = []

summary_rows.append({
    "Block": "Baseline",
    "Best_or_Center": "Baseline",
    "ARI": baseline_ari,
    "NMI": baseline_nmi,
    "w_smooth": 0.0,
    "w_sharpen": 0.0,
    "gamma": np.nan,
    "warmup_epochs": np.nan,
})

for param_name in SENSITIVITY_POOLS.keys():
    sub = onefactor_df[
        (onefactor_df["param_name"] == param_name) &
        (onefactor_df["status"] == "success")
    ].copy()

    if len(sub) > 0:
        best_ari_row = sub.loc[sub["ARI"].idxmax()]
        best_nmi_row = sub.loc[sub["NMI"].idxmax()]

        summary_rows.append({
            "Block": f"OneFactor_{param_name}",
            "Best_or_Center": "Best_ARI",
            "ARI": best_ari_row["ARI"],
            "NMI": best_ari_row["NMI"],
            "w_smooth": best_ari_row["w_smooth"],
            "w_sharpen": best_ari_row["w_sharpen"],
            "gamma": best_ari_row["gamma"],
            "warmup_epochs": best_ari_row["warmup_epochs"],
        })

        summary_rows.append({
            "Block": f"OneFactor_{param_name}",
            "Best_or_Center": "Best_NMI",
            "ARI": best_nmi_row["ARI"],
            "NMI": best_nmi_row["NMI"],
            "w_smooth": best_nmi_row["w_smooth"],
            "w_sharpen": best_nmi_row["w_sharpen"],
            "gamma": best_nmi_row["gamma"],
            "warmup_epochs": best_nmi_row["warmup_epochs"],
        })

if len(heatmap_ok) > 0:
    best_heat_ari = heatmap_ok.loc[heatmap_ok["ARI"].idxmax()]
    best_heat_nmi = heatmap_ok.loc[heatmap_ok["NMI"].idxmax()]

    summary_rows.append({
        "Block": "TwoFactor_wsmooth_wsharpen",
        "Best_or_Center": "Best_ARI",
        "ARI": best_heat_ari["ARI"],
        "NMI": best_heat_ari["NMI"],
        "w_smooth": best_heat_ari["w_smooth"],
        "w_sharpen": best_heat_ari["w_sharpen"],
        "gamma": best_heat_ari["gamma"],
        "warmup_epochs": best_heat_ari["warmup_epochs"],
    })

    summary_rows.append({
        "Block": "TwoFactor_wsmooth_wsharpen",
        "Best_or_Center": "Best_NMI",
        "ARI": best_heat_nmi["ARI"],
        "NMI": best_heat_nmi["NMI"],
        "w_smooth": best_heat_nmi["w_smooth"],
        "w_sharpen": best_heat_nmi["w_sharpen"],
        "gamma": best_heat_nmi["gamma"],
        "warmup_epochs": best_heat_nmi["warmup_epochs"],
    })

    summary_rows.append({
        "Block": "Ada_Center",
        "Best_or_Center": "Center",
        "ARI": heatmap_ok[
            (heatmap_ok["w_smooth"] == BEST_ADA["w_smooth"]) &
            (heatmap_ok["w_sharpen"] == BEST_ADA["w_sharpen"])
        ]["ARI"].iloc[0],
        "NMI": heatmap_ok[
            (heatmap_ok["w_smooth"] == BEST_ADA["w_smooth"]) &
            (heatmap_ok["w_sharpen"] == BEST_ADA["w_sharpen"])
        ]["NMI"].iloc[0],
        "w_smooth": BEST_ADA["w_smooth"],
        "w_sharpen": BEST_ADA["w_sharpen"],
        "gamma": BEST_ADA["gamma"],
        "warmup_epochs": BEST_ADA["warmup_epochs"],
    })

summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(result_root, "HBCA_section1_parameter_sensitivity_summary.csv")
summary_df.to_csv(summary_csv, index=False)

# =========================================================
# 14. 完成提示
# =========================================================
print("\n" + "=" * 100)
print("HBCA SECTION1 PARAMETER SENSITIVITY FINISHED")
print("=" * 100)
print("\n[Baseline]")
print(baseline_df[["Variant", "ARI", "NMI"]])

print("\n[One-factor best summary]")
print(summary_df.to_string(index=False))

print(f"\nSaved baseline results to: {baseline_h5ad}")
print(f"Saved one-factor csv to: {onefactor_csv}")
print(f"Saved heatmap csv to: {heatmap_csv}")
print(f"Saved summary csv to: {summary_csv}")
print(f"Saved all figures to: {result_root}")