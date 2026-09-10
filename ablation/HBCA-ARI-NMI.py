import os
import sys
import random
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import matplotlib.pyplot as plt
import matplotlib
from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment

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


#字体调节
def style_axis_font(
    ax,
    tick_fontsize=12,
    label_fontsize=14,
    title_fontsize=15,
    spine_width=1.2
):
    """
    统一加粗、放大坐标轴刻度值、坐标轴标题和边框。
    """
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
        width=spine_width
    )

    # X/Y 轴刻度值加粗
    for tick_label in ax.get_xticklabels():
        tick_label.set_fontweight("bold")

    for tick_label in ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    # X/Y 轴标题加粗
    ax.xaxis.label.set_size(label_fontsize)
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_size(label_fontsize)
    ax.yaxis.label.set_weight("bold")

    # 图标题加粗
    ax.title.set_size(title_fontsize)
    ax.title.set_weight("bold")

    # 坐标轴边框加粗
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)



# =========================================================
# 1. patch mclust，沿用你原始 HBCA 代码逻辑
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

# 画图字体
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"

# =========================================================
# 3. 数据路径
# =========================================================
file_fold = "/data2/liangyefeng/My_GraphST_Innovation/data/Human-Breast-Cancer-Block-A/section1"
gt_dir = os.path.join(file_fold, "gt")

result_root = os.path.join(file_fold, "HBCA_GraphST_Ablation")
os.makedirs(result_root, exist_ok=True)

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
#    优先用 gold_metadata.tsv，完全沿用你原始代码逻辑
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

# 去掉无标注 spot
adata_eval = adata[~pd.isnull(adata.obs["ground_truth"])].copy()

# GT 类别数作为聚类数
n_clusters = adata_eval.obs["ground_truth"].nunique()
print("✅ n_clusters from GT:", n_clusters)

# =========================================================
# 6. 消融组配置
#    你可以只改下面这些值
# =========================================================
SMOOTH_WS = 0.05
SHARPEN_WSH = 0.08

SMOOTH_WARMUP = 200
SHARPEN_WARMUP = 200

ADA_WS = 0.05
ADA_WSH = 0.08
ADA_GAMMA = 2.5
ADA_WARMUP = 100   # ← 你自己改

ABLATION_CONFIGS = OrderedDict()

# 1) Baseline：完全对齐你原始 HBCA 代码
ABLATION_CONFIGS["Baseline"] = {
    "mode": "baseline_default"
}

# 2) Smooth only
ABLATION_CONFIGS["Smooth only"] = {
    "mode": "explicit",
    "w_smooth": 0.01,
    "w_sharpen": 0.0,
    "gamma": 3.0,
    "warmup_epochs": SMOOTH_WARMUP,
    "use_learnable_proj": False,
}

# 3) Sharpen only
ABLATION_CONFIGS["Sharpen only"] = {
    "mode": "explicit",
    "w_smooth": 0.0,
    "w_sharpen": 0.3,
    "gamma": 3,
    "warmup_epochs": SHARPEN_WARMUP,
    "use_learnable_proj": False,
}

# 4) Ada-GraphST
ABLATION_CONFIGS["Ada-GraphST"] = {
    "mode": "explicit",
    "w_smooth": ADA_WS,
    "w_sharpen": ADA_WSH,
    "gamma": ADA_GAMMA,
    "warmup_epochs": ADA_WARMUP,
    "use_learnable_proj": False,
}

# =========================================================
# 7. 跑单个消融组
# =========================================================
def run_one_variant(adata_input, variant_name, variant_cfg, n_clusters, radius=50):
    print("\n" + "-" * 100)
    print(f"Running variant: {variant_name}")
    print("-" * 100)

    adata_run = adata_input.copy()

    # ---- 完全贴你的原始 baseline 写法 ----
    if variant_cfg["mode"] == "baseline_default":
        model = GraphST(adata_run, device=device)

    # ---- 其他组在你原始 GraphST 调用基础上显式加参数 ----
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

    adata_run = model.train()

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

    print(f"Variant: {variant_name}")
    print(f"ARI: {ari:.6f}")
    print(f"NMI: {nmi:.6f}")

    return ari, nmi, adata_run, adata_eval


# =========================================================
# 8. 主流程：依次跑 4 个消融组
# =========================================================
all_records = []
variant_adatas = OrderedDict()

for variant_name, variant_cfg in ABLATION_CONFIGS.items():
    try:
        ari, nmi, adata_out, adata_eval_out = run_one_variant(
            adata_input=adata,
            variant_name=variant_name,
            variant_cfg=variant_cfg,
            n_clusters=n_clusters,
            radius=50
        )

        save_h5ad = os.path.join(
            result_root,
            f"HBCA_section1_{variant_name.replace(' ', '_')}.h5ad"
        )
        adata_out.write(save_h5ad)
        variant_adatas[variant_name] = adata_out.copy()

        record = {
            "Dataset": "Human-Breast-Cancer-Block-A_section1",
            "Variant": variant_name,
            "Seed": seed,
            "N_Clusters": n_clusters,
            "N_Obs_Eval": adata_eval_out.n_obs,
            "ARI": ari,
            "NMI": nmi,
            "save_h5ad": save_h5ad,
        }

        if variant_cfg["mode"] == "baseline_default":
            record["w_smooth"] = 0.0
            record["w_sharpen"] = 0.0
            record["gamma"] = np.nan
            record["warmup_epochs"] = np.nan
            record["use_learnable_proj"] = False
            record["mode"] = "baseline_default"
        else:
            record["w_smooth"] = variant_cfg["w_smooth"]
            record["w_sharpen"] = variant_cfg["w_sharpen"]
            record["gamma"] = variant_cfg["gamma"]
            record["warmup_epochs"] = variant_cfg.get("warmup_epochs", np.nan)
            record["use_learnable_proj"] = variant_cfg.get("use_learnable_proj", False)
            record["mode"] = "explicit"

        all_records.append(record)

    except Exception as e:
        print(f"❌ Variant [{variant_name}] failed: {e}")
        all_records.append({
            "Dataset": "Human-Breast-Cancer-Block-A_section1",
            "Variant": variant_name,
            "Seed": seed,
            "N_Clusters": n_clusters,
            "N_Obs_Eval": np.nan,
            "ARI": np.nan,
            "NMI": np.nan,
            "save_h5ad": "",
            "w_smooth": np.nan,
            "w_sharpen": np.nan,
            "gamma": np.nan,
            "warmup_epochs": np.nan,
            "use_learnable_proj": False,
            "mode": variant_cfg["mode"],
            "Error": str(e),
        })

# =========================================================
# 9. 保存结果
# =========================================================
results_df = pd.DataFrame(all_records)

variant_order = ["Baseline", "Smooth only", "Sharpen only", "Ada-GraphST"]
results_df["Variant"] = pd.Categorical(results_df["Variant"], categories=variant_order, ordered=True)
results_df = results_df.sort_values("Variant").reset_index(drop=True)
variant_metric_map = {
    row["Variant"]: {
        "ARI": row["ARI"],
        "NMI": row["NMI"]
    }
    for _, row in results_df.iterrows()
}

detail_csv = os.path.join(result_root, "HBCA_section1_ablation_results.csv")
results_df.to_csv(detail_csv, index=False)

print(f"\n💾 详细结果已保存至: {detail_csv}")

# =========================================================
# 10. 画 ARI 柱状图
# =========================================================
# =========================================================
# 10. 画 ARI 柱状图
# =========================================================
plot_colors = {
    "Baseline": "#B07AA1",
    "Smooth only": "#4E79A7",
    "Sharpen only": "#F28E2B",
    "Ada-GraphST": "#76B7B2",
}

fig, ax = plt.subplots(figsize=(8, 6))

bars = ax.bar(
    results_df["Variant"].astype(str),
    results_df["ARI"],
    color=[plot_colors[v] for v in results_df["Variant"].astype(str)],
    edgecolor="black",
    linewidth=1.0
)

ax.set_ylabel("ARI", fontsize=15, fontweight="bold")
ax.set_title(
    "HBCA section1 Ablation Study (ARI)",
    fontsize=16,
    fontweight="bold"
)

ax.set_xticklabels(
    results_df["Variant"].astype(str),
    rotation=20,
    ha="right",
    fontsize=13,
    fontweight="bold"
)

ax.tick_params(axis="y", labelsize=13, width=1.2)

for label in ax.get_yticklabels():
    label.set_fontweight("bold")

ax.grid(axis="y", linestyle="--", alpha=0.35)

style_axis_font(
    ax,
    tick_fontsize=13,
    label_fontsize=15,
    title_fontsize=16,
    spine_width=1.2
)

for bar, val in zip(bars, results_df["ARI"]):
    if pd.notna(val):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold"
        )

plt.tight_layout()
ari_png = os.path.join(result_root, "HBCA_section1_Ablation_ARI_barplot.png")
ari_pdf = os.path.join(result_root, "HBCA_section1_Ablation_ARI_barplot.pdf")
plt.savefig(ari_png, dpi=300, bbox_inches="tight")
plt.savefig(ari_pdf, bbox_inches="tight")
plt.close()
# =========================================================
# 11. 画 NMI 柱状图
# =========================================================
plt.figure(figsize=(8, 6))
bars = plt.bar(
    results_df["Variant"].astype(str),
    results_df["NMI"],
    color=[plot_colors[v] for v in results_df["Variant"].astype(str)],
    edgecolor="black",
    linewidth=0.8
)
plt.ylabel("NMI", fontsize=12)
plt.title("HBCA section1 Ablation Study (NMI)", fontsize=14, fontweight="bold")
plt.xticks(rotation=20, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.35)

for bar, val in zip(bars, results_df["NMI"]):
    if pd.notna(val):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

plt.tight_layout()
nmi_png = os.path.join(result_root, "HBCA_section1_Ablation_NMI_barplot.png")
nmi_pdf = os.path.join(result_root, "HBCA_section1_Ablation_NMI_barplot.pdf")
plt.savefig(nmi_png, dpi=300, bbox_inches="tight")
plt.savefig(nmi_pdf, bbox_inches="tight")
plt.close()

# =========================================================
# 12. 可视化辅助函数
# =========================================================
def get_spatial_xy(adata):
    if "spatial" in adata.obsm and adata.obsm["spatial"].shape[1] >= 2:
        xy = np.asarray(adata.obsm["spatial"])[:, :2]
        return xy[:, 0], xy[:, 1]

    candidate_pairs = [
        ("x", "y"),
        ("X", "Y"),
        ("array_col", "array_row"),
        ("imagecol", "imagerow"),
        ("col", "row"),
    ]
    for cx, cy in candidate_pairs:
        if cx in adata.obs.columns and cy in adata.obs.columns:
            return adata.obs[cx].astype(float).values, adata.obs[cy].astype(float).values

    raise KeyError("Cannot find spatial coordinates from adata.obsm['spatial'] or obs columns.")


def build_category_palette(categories):
    n = len(categories)
    cmap = plt.get_cmap("tab20", max(n, 3))
    palette = {}
    for i, cat in enumerate(categories):
        palette[str(cat)] = cmap(i)
    return palette


def align_pred_to_ground_truth(adata, pred_col="domain", gt_col="ground_truth", out_col="domain_matched"):
    if pred_col not in adata.obs.columns:
        raise KeyError(f"{pred_col} not found in adata.obs")
    if gt_col not in adata.obs.columns:
        raise KeyError(f"{gt_col} not found in adata.obs")

    valid_mask = (~pd.isna(adata.obs[pred_col])) & (~pd.isna(adata.obs[gt_col]))
    df = adata.obs.loc[valid_mask, [pred_col, gt_col]].copy()

    pred_vals = df[pred_col].astype(str)
    gt_vals = df[gt_col].astype(str)

    contingency = pd.crosstab(gt_vals, pred_vals)
    gt_labels = contingency.index.tolist()
    pred_labels = contingency.columns.tolist()

    cost = -contingency.values
    row_ind, col_ind = linear_sum_assignment(cost)

    pred_to_gt = {}
    for r, c in zip(row_ind, col_ind):
        pred_to_gt[pred_labels[c]] = gt_labels[r]

    mapped = adata.obs[pred_col].astype(str).map(pred_to_gt)
    raw_pred = adata.obs[pred_col].astype(str)
    mapped = mapped.where(~mapped.isna(), raw_pred)

    adata.obs[out_col] = pd.Categorical(mapped)
    return adata


def plot_spatial_panel(ax, adata, color_col, title, palette, point_size=8, show_legend=False):
    x, y = get_spatial_xy(adata)

    vals = adata.obs[color_col].astype(str).fillna("NA").values
    cats = list(pd.Categorical(adata.obs[color_col].astype(str)).categories)

    for cat in cats:
        mask = vals == str(cat)
        ax.scatter(
            x[mask], y[mask],
            s=point_size,
            c=[palette.get(str(cat), (0.7, 0.7, 0.7, 1.0))],
            linewidths=0,
            label=str(cat)
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_aspect("equal")

    if show_legend:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=9,
            markerscale=2
        )


def compute_umap_for_plot(adata, seed=41):
    ad = adata.copy()
    use_rep = "emb_pca" if "emb_pca" in ad.obsm else "emb"
    if use_rep not in ad.obsm:
        raise KeyError("Neither 'emb_pca' nor 'emb' found in adata.obsm for UMAP plotting.")

    sc.pp.neighbors(ad, use_rep=use_rep, n_neighbors=15)
    sc.tl.umap(ad, random_state=seed)
    return ad


def plot_umap_panel(ax, adata, color_col, title, palette, point_size=10, show_legend=False):
    if "X_umap" not in adata.obsm:
        raise KeyError("X_umap not found in adata.obsm")

    umap_xy = np.asarray(adata.obsm["X_umap"])
    vals = adata.obs[color_col].astype(str).fillna("NA").values
    cats = list(pd.Categorical(adata.obs[color_col].astype(str)).categories)

    for cat in cats:
        mask = vals == str(cat)
        ax.scatter(
            umap_xy[mask, 0], umap_xy[mask, 1],
            s=point_size,
            c=[palette.get(str(cat), (0.7, 0.7, 0.7, 1.0))],
            linewidths=0,
            label=str(cat)
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    if show_legend:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=9,
            markerscale=2
        )

# =========================================================
# 13. Spatial clustering 对比图
# =========================================================
if "ground_truth" not in adata.obs.columns:
    raise KeyError("ground_truth not found in adata.obs, cannot plot ablation comparison.")

gt_categories = list(pd.Categorical(adata.obs["ground_truth"].astype(str)).categories)
gt_palette = build_category_palette(gt_categories)

for variant_name in list(variant_adatas.keys()):
    variant_adatas[variant_name] = align_pred_to_ground_truth(
        variant_adatas[variant_name],
        pred_col="domain",
        gt_col="ground_truth",
        out_col="domain_matched"
    )

spatial_order = ["Baseline", "Smooth only", "Sharpen only", "Ada-GraphST"]
available_spatial_order = [v for v in spatial_order if v in variant_adatas]

n_panels = 1 + len(available_spatial_order)
fig, axes = plt.subplots(1, n_panels, figsize=(4.8 * n_panels, 5.2))

if n_panels == 1:
    axes = [axes]

plot_spatial_panel(
    ax=axes[0],
    adata=adata,
    color_col="ground_truth",
    title="Ground Truth",
    palette=gt_palette,
    point_size=8,
    show_legend=(n_panels == 1)
)

for i, variant_name in enumerate(available_spatial_order, start=1):
    ari_val = variant_metric_map.get(variant_name, {}).get("ARI", np.nan)
    panel_title = variant_name
    if pd.notna(ari_val):
        panel_title = f"{variant_name}\nARI={ari_val:.3f}"

    plot_spatial_panel(
        ax=axes[i],
        adata=variant_adatas[variant_name],
        color_col="domain_matched",
        title=panel_title,
        palette=gt_palette,
        point_size=8,
        show_legend=(i == len(available_spatial_order))
    )

plt.suptitle("HBCA section1 Ablation: Spatial Clustering Comparison", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()

spatial_png = os.path.join(result_root, "HBCA_section1_Ablation_Spatial_Comparison.png")
spatial_pdf = os.path.join(result_root, "HBCA_section1_Ablation_Spatial_Comparison.pdf")
plt.savefig(spatial_png, dpi=300, bbox_inches="tight")
plt.savefig(spatial_pdf, bbox_inches="tight")
plt.close()

# =========================================================
# 14. UMAP 对比图（按 matched prediction 上色）
# =========================================================
variant_umaps = OrderedDict()
for variant_name in available_spatial_order:
    try:
        variant_umaps[variant_name] = compute_umap_for_plot(variant_adatas[variant_name], seed=seed)
    except Exception as e:
        print(f"[UMAP] {variant_name} failed: {e}")

available_umap_order = list(variant_umaps.keys())

if len(available_umap_order) > 0:
    fig, axes = plt.subplots(1, len(available_umap_order), figsize=(5.0 * len(available_umap_order), 4.8))

    if len(available_umap_order) == 1:
        axes = [axes]

    for i, variant_name in enumerate(available_umap_order):
        ari_val = variant_metric_map.get(variant_name, {}).get("ARI", np.nan)
        panel_title = variant_name
        if pd.notna(ari_val):
            panel_title = f"{variant_name}\nARI={ari_val:.3f}"

        plot_umap_panel(
            ax=axes[i],
            adata=variant_umaps[variant_name],
            color_col="domain_matched",
            title=panel_title,
            palette=gt_palette,
            point_size=10,
            show_legend=(i == len(available_umap_order) - 1)
        )

    plt.suptitle("HBCA section1 Ablation: UMAP Comparison (Matched Prediction)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    umap_pred_png = os.path.join(result_root, "HBCA_section1_Ablation_UMAP_Prediction_Comparison.png")
    umap_pred_pdf = os.path.join(result_root, "HBCA_section1_Ablation_UMAP_Prediction_Comparison.pdf")
    plt.savefig(umap_pred_png, dpi=300, bbox_inches="tight")
    plt.savefig(umap_pred_pdf, bbox_inches="tight")
    plt.close()
else:
    umap_pred_png = None
    umap_pred_pdf = None

# =========================================================
# 15. UMAP 对比图（按 Ground Truth 上色）
# =========================================================
if len(available_umap_order) > 0:
    fig, axes = plt.subplots(1, len(available_umap_order), figsize=(5.0 * len(available_umap_order), 4.8))

    if len(available_umap_order) == 1:
        axes = [axes]

    for i, variant_name in enumerate(available_umap_order):
        ari_val = variant_metric_map.get(variant_name, {}).get("ARI", np.nan)
        panel_title = variant_name
        if pd.notna(ari_val):
            panel_title = f"{variant_name}\nARI={ari_val:.3f}"

        plot_umap_panel(
            ax=axes[i],
            adata=variant_umaps[variant_name],
            color_col="ground_truth",
            title=panel_title,
            palette=gt_palette,
            point_size=10,
            show_legend=(i == len(available_umap_order) - 1)
        )

    plt.suptitle("HBCA section1 Ablation: UMAP Comparison (Ground Truth)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    umap_gt_png = os.path.join(result_root, "HBCA_section1_Ablation_UMAP_GT_Comparison.png")
    umap_gt_pdf = os.path.join(result_root, "HBCA_section1_Ablation_UMAP_GT_Comparison.pdf")
    plt.savefig(umap_gt_png, dpi=300, bbox_inches="tight")
    plt.savefig(umap_gt_pdf, bbox_inches="tight")
    plt.close()
else:
    umap_gt_png = None
    umap_gt_pdf = None

# =========================================================
# 16. 完成提示
# =========================================================
print("\n" + "=" * 100)
print("HBCA SECTION1 ABLATION FINISHED")
print("=" * 100)
print(results_df[["Variant", "ARI", "NMI", "warmup_epochs"]])

print(f"\nSaved detail results to: {detail_csv}")
print(f"Saved ARI barplot to: {ari_png}")
print(f"Saved NMI barplot to: {nmi_png}")
print(f"Saved spatial comparison to: {spatial_png}")

if umap_pred_png is not None:
    print(f"Saved UMAP prediction comparison to: {umap_pred_png}")
if umap_gt_png is not None:
    print(f"Saved UMAP GT comparison to: {umap_gt_png}")