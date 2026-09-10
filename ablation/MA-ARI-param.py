import os
import sys
import gc
from collections import OrderedDict
import warnings

import scanpy as sc
from scipy.optimize import linear_sum_assignment
warnings.filterwarnings("ignore")

os.environ["R_HOME"] = "/data2/liangyefeng/miniconda3/envs/graphst/lib/R"

project_root = "/data2/liangyefeng/My_GraphST_Innovation"
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "scripts", "Method"))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA

from GraphST.GraphST import GraphST
import GraphST.utils as graphst_utils
from GraphST.utils import clustering

from mouse_brain_MA_common import (
    reset_seed,
    safe_mclust_R,
    load_ma_dataset,
    to_counts_adata,
    compute_ari_nmi,
)

# =========================================================
# 0. 路径与全局设置
# =========================================================
RESULT_ROOT = "/data2/liangyefeng/My_GraphST_Innovation/Result_MA_ParameterSensitivity_Ada"
SEED = 41
GRAPHST_REFINEMENT = False

os.makedirs(RESULT_ROOT, exist_ok=True)

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"

# 是否保存每次扫描得到的 h5ad
SAVE_ALL_H5AD = False

# =========================================================
# 1. patch mclust
# =========================================================
def patched_mclust_R(adata, num_cluster, modelNames="EEE", used_obsm="emb_pca", random_seed=2020):
    return safe_mclust_R(
        adata,
        num_cluster=num_cluster,
        used_obsm=used_obsm,
        key_added="mclust",
        modelNames=modelNames,
        random_seed=random_seed
    )

graphst_utils.mclust_R = patched_mclust_R

# =========================================================
# 2. 固定随机种子 / device
# =========================================================
reset_seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

# =========================================================
# 3. Ada-GraphST 中心参数
#    只围绕 smooth / sharpen 做敏感性
#    gamma 固定为 2.5
# =========================================================
BEST_ADA = {
    "device": device,
    "random_seed": SEED,
    "epochs": 600,
    "dim_output": 64,
    "datatype": "Stereo",
    "warmup_epochs": 200,
    "update_interval": 30,
    "graph_update_rate": 0.3,
    "gamma": 2.5,
    "graph_reg_weight": 0.0,
    "use_learnable_proj": False,
    "w_smooth": 0.05,
    "w_sharpen": 0.05,
    "use_original_baseline": False,
}

BASELINE_CFG = {
    "use_original_baseline": True
}

# =========================================================
# 4. 参数池
#    围绕 Ada-GraphST 中心点局部扫描
# =========================================================
SENSITIVITY_POOLS = OrderedDict({
    "w_smooth": [0.01, 0.03, 0.05, 0.07, 0.09],
    "w_sharpen": [0.01, 0.03, 0.05, 0.07, 0.09],
})

HEATMAP_W_SMOOTH = [0.01, 0.03, 0.05, 0.07, 0.09]
HEATMAP_W_SHARPEN = [0.01, 0.03, 0.05, 0.07, 0.09]

print("\n========== PARAMETER SENSITIVITY CENTER ==========")
print({
    "w_smooth": BEST_ADA["w_smooth"],
    "w_sharpen": BEST_ADA["w_sharpen"],
    "gamma": BEST_ADA["gamma"],
    "warmup_epochs": BEST_ADA["warmup_epochs"],
    "update_interval": BEST_ADA["update_interval"],
    "graph_update_rate": BEST_ADA["graph_update_rate"],
    "graph_reg_weight": BEST_ADA["graph_reg_weight"],
    "use_learnable_proj": BEST_ADA["use_learnable_proj"],
})

# =========================================================
# 5. 读入数据
# =========================================================
adata_raw, label_col, n_clusters = load_ma_dataset()
adata_raw = to_counts_adata(adata_raw)

print("✅ Dataset ready")
print("✅ label_col:", label_col)
print("✅ n_clusters:", n_clusters)
print("✅ n_obs:", adata_raw.n_obs)
print("✅ n_vars:", adata_raw.n_vars)

# =========================================================
# 6. 工具函数
# =========================================================
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_one_variant(adata_input, n_clusters, variant_name, variant_params):
    print("\n" + "-" * 100)
    print(f"Running variant: {variant_name}")
    print("-" * 100)

    reset_seed(SEED)
    adata = adata_input.copy()

    if variant_params.get("use_original_baseline", False):
        print("[INFO] Using ORIGINAL GraphST baseline path")
        model = GraphST(adata, device=device)
    else:
        graphst_params = {
            k: v for k, v in variant_params.items()
            if k != "use_original_baseline"
        }
        model = GraphST(adata, **graphst_params)

    adata = model.train()

    if "emb" not in adata.obsm:
        raise KeyError("adata.obsm['emb'] not found after training.")

    n_pcs = min(20, adata.obsm["emb"].shape[1])
    adata.obsm["emb_pca"] = PCA(
        n_components=n_pcs,
        random_state=SEED
    ).fit_transform(adata.obsm["emb"])

    clustering(
        adata,
        n_clusters,
        radius=50,
        method="mclust",
        refinement=GRAPHST_REFINEMENT
    )

    ari, nmi, n_eval = compute_ari_nmi(
        adata,
        pred_col="domain",
        gt_col="ground_truth"
    )

    print(f"Variant: {variant_name}")
    print(f"ARI: {ari:.6f}")
    print(f"NMI: {nmi:.6f}")
    print(f"N eval: {n_eval}")

    return ari, nmi, n_eval, adata


def make_cfg_from_best(best_cfg, param_name=None, param_value=None, extra_updates=None):
    cfg = {
        "device": best_cfg["device"],
        "random_seed": best_cfg["random_seed"],
        "epochs": best_cfg["epochs"],
        "dim_output": best_cfg["dim_output"],
        "datatype": best_cfg["datatype"],
        "warmup_epochs": best_cfg["warmup_epochs"],
        "update_interval": best_cfg["update_interval"],
        "graph_update_rate": best_cfg["graph_update_rate"],
        "gamma": best_cfg["gamma"],
        "graph_reg_weight": best_cfg["graph_reg_weight"],
        "use_learnable_proj": best_cfg["use_learnable_proj"],
        "w_smooth": best_cfg["w_smooth"],
        "w_sharpen": best_cfg["w_sharpen"],
        "use_original_baseline": False,
    }

    if param_name is not None:
        cfg[param_name] = param_value

    if extra_updates is not None:
        for k, v in extra_updates.items():
            cfg[k] = v

    return cfg


def save_h5ad_if_needed(adata_obj, save_path):
    if SAVE_ALL_H5AD:
        adata_obj.write(save_path)


def annotate_points(ax, xs, ys, fmt="{:.3f}", fontsize=9):
    for x, y in zip(xs, ys):
        if pd.notna(y):
            ax.text(
                x, y + 0.003,
                fmt.format(y),
                ha="center",
                va="bottom",
                fontsize=fontsize,
                fontweight="bold"
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
        ax.axvline(best_value, linestyle=":", linewidth=1.5, label="Ada center")
        annotate_points(ax, x, y, fmt="{:.3f}", fontsize=8.5)

        ax.set_title(
            f"{param_name} sensitivity ({metric_name})",
            fontsize=12.5,
            fontweight="bold"
        )
        ax.set_xlabel(param_name, fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(frameon=True)

    plt.suptitle(
        f"Mouse Brain MA Parameter Sensitivity: {param_name}\n"
        f"(gamma fixed at 2.5; other parameters fixed at Ada-GraphST setting)",
        fontsize=14.5,
        fontweight="bold",
        y=1.03
    )
    plt.tight_layout()

    png = os.path.join(out_dir, f"MA_sensitivity_{param_name}.png")
    pdf = os.path.join(out_dir, f"MA_sensitivity_{param_name}.pdf")
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

    center_i = list(pivot_df.index).index(center_wsh)
    center_j = list(pivot_df.columns).index(center_ws)

    rect_center = Rectangle(
        (center_j - 0.5, center_i - 0.5), 1, 1,
        fill=False, edgecolor="black", linewidth=2.6
    )
    ax.add_patch(rect_center)

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
        "Mouse Brain MA Joint Sensitivity Heatmap: w_smooth × w_sharpen\n"
        "(gamma fixed at 2.5; black box = Ada center; red dashed = best in grid)",
        fontsize=14.5,
        fontweight="bold",
        y=1.03
    )
    plt.tight_layout()

    png = os.path.join(out_dir, "MA_sensitivity_wsmooth_wsharpen_heatmap.png")
    pdf = os.path.join(out_dir, "MA_sensitivity_wsmooth_wsharpen_heatmap.pdf")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()

# =========================================================
# 7. 先跑 Baseline
# =========================================================
baseline_ari, baseline_nmi, baseline_n_eval, baseline_adata = run_one_variant(
    adata_input=adata_raw,
    n_clusters=n_clusters,
    variant_name="Baseline",
    variant_params=BASELINE_CFG
)

baseline_h5ad = os.path.join(RESULT_ROOT, "MA_Baseline.h5ad")
baseline_adata.write(baseline_h5ad)

baseline_df = pd.DataFrame([{
    "dataset": "MA",
    "variant": "Baseline",
    "label_col": label_col,
    "N_Clusters": n_clusters,
    "N_Obs_Eval": baseline_n_eval,
    "ARI": baseline_ari,
    "NMI": baseline_nmi,
    "save_h5ad": baseline_h5ad,
    "w_smooth": 0.0,
    "w_sharpen": 0.0,
    "gamma": np.nan,
    "warmup_epochs": np.nan,
    "update_interval": np.nan,
    "graph_update_rate": np.nan,
    "graph_reg_weight": np.nan,
    "use_learnable_proj": False,
}])

del baseline_adata
clear_memory()

# =========================================================
# 8. 单因素敏感性：w_smooth / w_sharpen
# =========================================================
onefactor_records = []

for param_name, pool in SENSITIVITY_POOLS.items():
    print("\n" + "=" * 100)
    print(f"📊 Running one-factor sensitivity for: {param_name}")
    print("=" * 100)

    for value in pool:
        variant_cfg = make_cfg_from_best(
            BEST_ADA,
            param_name=param_name,
            param_value=value
        )

        variant_name = f"AdaSensitivity_{param_name}_{value}"

        try:
            ari, nmi, n_eval, adata_out = run_one_variant(
                adata_input=adata_raw,
                n_clusters=n_clusters,
                variant_name=variant_name,
                variant_params=variant_cfg
            )

            save_h5ad = os.path.join(
                RESULT_ROOT,
                f"{variant_name}.h5ad"
            )
            save_h5ad_if_needed(adata_out, save_h5ad)

            onefactor_records.append({
                "dataset": "MA",
                "analysis_type": "one_factor",
                "param_name": param_name,
                "param_value": value,
                "label_col": label_col,
                "N_Clusters": n_clusters,
                "N_Obs_Eval": n_eval,
                "ARI": ari,
                "NMI": nmi,
                "w_smooth": variant_cfg["w_smooth"],
                "w_sharpen": variant_cfg["w_sharpen"],
                "gamma": variant_cfg["gamma"],
                "warmup_epochs": variant_cfg["warmup_epochs"],
                "update_interval": variant_cfg["update_interval"],
                "graph_update_rate": variant_cfg["graph_update_rate"],
                "graph_reg_weight": variant_cfg["graph_reg_weight"],
                "use_learnable_proj": variant_cfg["use_learnable_proj"],
                "save_h5ad": save_h5ad if SAVE_ALL_H5AD else "",
                "status": "success",
                "error": "",
            })

        except Exception as e:
            print(f"❌ Failed: {variant_name} | {e}")
            onefactor_records.append({
                "dataset": "MA",
                "analysis_type": "one_factor",
                "param_name": param_name,
                "param_value": value,
                "label_col": label_col,
                "N_Clusters": n_clusters,
                "N_Obs_Eval": np.nan,
                "ARI": np.nan,
                "NMI": np.nan,
                "w_smooth": variant_cfg["w_smooth"],
                "w_sharpen": variant_cfg["w_sharpen"],
                "gamma": variant_cfg["gamma"],
                "warmup_epochs": variant_cfg["warmup_epochs"],
                "update_interval": variant_cfg["update_interval"],
                "graph_update_rate": variant_cfg["graph_update_rate"],
                "graph_reg_weight": variant_cfg["graph_reg_weight"],
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
onefactor_csv = os.path.join(RESULT_ROOT, "MA_onefactor_sensitivity_results.csv")
onefactor_df.to_csv(onefactor_csv, index=False)

# =========================================================
# 9. 二维联合敏感性：w_smooth × w_sharpen
# =========================================================
heatmap_records = []

print("\n" + "=" * 100)
print("📊 Running joint sensitivity heatmap: w_smooth × w_sharpen")
print("=" * 100)

for ws in HEATMAP_W_SMOOTH:
    for wsh in HEATMAP_W_SHARPEN:
        variant_cfg = make_cfg_from_best(
            BEST_ADA,
            extra_updates={
                "w_smooth": ws,
                "w_sharpen": wsh
            }
        )

        variant_name = f"AdaGrid_ws_{ws}_wsh_{wsh}"

        try:
            ari, nmi, n_eval, adata_out = run_one_variant(
                adata_input=adata_raw,
                n_clusters=n_clusters,
                variant_name=variant_name,
                variant_params=variant_cfg
            )

            save_h5ad = os.path.join(
                RESULT_ROOT,
                f"{variant_name}.h5ad"
            )
            save_h5ad_if_needed(adata_out, save_h5ad)

            heatmap_records.append({
                "dataset": "MA",
                "analysis_type": "two_factor",
                "label_col": label_col,
                "N_Clusters": n_clusters,
                "N_Obs_Eval": n_eval,
                "w_smooth": ws,
                "w_sharpen": wsh,
                "gamma": variant_cfg["gamma"],
                "warmup_epochs": variant_cfg["warmup_epochs"],
                "update_interval": variant_cfg["update_interval"],
                "graph_update_rate": variant_cfg["graph_update_rate"],
                "graph_reg_weight": variant_cfg["graph_reg_weight"],
                "use_learnable_proj": variant_cfg["use_learnable_proj"],
                "ARI": ari,
                "NMI": nmi,
                "save_h5ad": save_h5ad if SAVE_ALL_H5AD else "",
                "status": "success",
                "error": "",
            })

        except Exception as e:
            print(f"❌ Failed: {variant_name} | {e}")
            heatmap_records.append({
                "dataset": "MA",
                "analysis_type": "two_factor",
                "label_col": label_col,
                "N_Clusters": n_clusters,
                "N_Obs_Eval": np.nan,
                "w_smooth": ws,
                "w_sharpen": wsh,
                "gamma": variant_cfg["gamma"],
                "warmup_epochs": variant_cfg["warmup_epochs"],
                "update_interval": variant_cfg["update_interval"],
                "graph_update_rate": variant_cfg["graph_update_rate"],
                "graph_reg_weight": variant_cfg["graph_reg_weight"],
                "use_learnable_proj": variant_cfg["use_learnable_proj"],
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
heatmap_csv = os.path.join(RESULT_ROOT, "MA_wsmooth_wsharpen_heatmap_results.csv")
heatmap_df.to_csv(heatmap_csv, index=False)

# =========================================================
# 10. 画单因素曲线
# =========================================================
onefactor_ok = onefactor_df[onefactor_df["status"] == "success"].copy()

for param_name in SENSITIVITY_POOLS.keys():
    plot_onefactor_curve(
        df=onefactor_ok,
        param_name=param_name,
        baseline_ari=baseline_ari,
        baseline_nmi=baseline_nmi,
        best_value=BEST_ADA[param_name],
        out_dir=RESULT_ROOT
    )

# =========================================================
# 11. 画二维热图
# =========================================================
heatmap_ok = heatmap_df[heatmap_df["status"] == "success"].copy()

plot_ws_wsh_heatmap(
    df=heatmap_ok,
    center_ws=BEST_ADA["w_smooth"],
    center_wsh=BEST_ADA["w_sharpen"],
    out_dir=RESULT_ROOT
)

# =========================================================
# 12. 汇总表
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
    "update_interval": np.nan,
    "graph_update_rate": np.nan,
})

for param_name in SENSITIVITY_POOLS.keys():
    sub = onefactor_ok[onefactor_ok["param_name"] == param_name].copy()
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
            "update_interval": best_ari_row["update_interval"],
            "graph_update_rate": best_ari_row["graph_update_rate"],
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
            "update_interval": best_nmi_row["update_interval"],
            "graph_update_rate": best_nmi_row["graph_update_rate"],
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
        "update_interval": best_heat_ari["update_interval"],
        "graph_update_rate": best_heat_ari["graph_update_rate"],
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
        "update_interval": best_heat_nmi["update_interval"],
        "graph_update_rate": best_heat_nmi["graph_update_rate"],
    })

    center_sub = heatmap_ok[
        (np.isclose(heatmap_ok["w_smooth"], BEST_ADA["w_smooth"])) &
        (np.isclose(heatmap_ok["w_sharpen"], BEST_ADA["w_sharpen"]))
    ].copy()

    if len(center_sub) > 0:
        center_row = center_sub.iloc[0]
        summary_rows.append({
            "Block": "Ada_Center",
            "Best_or_Center": "Center",
            "ARI": center_row["ARI"],
            "NMI": center_row["NMI"],
            "w_smooth": center_row["w_smooth"],
            "w_sharpen": center_row["w_sharpen"],
            "gamma": center_row["gamma"],
            "warmup_epochs": center_row["warmup_epochs"],
            "update_interval": center_row["update_interval"],
            "graph_update_rate": center_row["graph_update_rate"],
        })

summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(RESULT_ROOT, "MA_parameter_sensitivity_summary.csv")
summary_df.to_csv(summary_csv, index=False)

# =========================================================
# 13. 完成提示
# =========================================================
print("\n" + "=" * 120)
print("ALL MA PARAMETER SENSITIVITY EXPERIMENTS FINISHED")
print("=" * 120)

print("\n[Baseline]")
print(baseline_df[["variant", "ARI", "NMI"]])

print("\n[Summary]")
print(summary_df.to_string(index=False))

print(f"\nSaved baseline h5ad to: {baseline_h5ad}")
print(f"Saved one-factor csv to: {onefactor_csv}")
print(f"Saved heatmap csv to: {heatmap_csv}")
print(f"Saved summary csv to: {summary_csv}")
print(f"Saved all figures to: {RESULT_ROOT}")