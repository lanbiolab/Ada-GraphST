import os
import sys
from collections import OrderedDict
import warnings
import scanpy as sc
from scipy.optimize import linear_sum_assignment
warnings.filterwarnings("ignore")

os.environ["R_HOME"] = "/data2/liangyefeng/miniconda3/envs/graphst/lib/R"

project_root = "/data2/liangyefeng/My_GraphST_Innovation"
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "scripts", "Clusting"))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib
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
RESULT_ROOT = "/data2/liangyefeng/My_GraphST_Innovation/Result_MA_Ablation"
SEED = 41
GRAPHST_REFINEMENT = False

os.makedirs(RESULT_ROOT, exist_ok=True)

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"


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
# 3. GraphST 公共参数
# =========================================================
common_params = {
    "device": device,
    "random_seed": SEED,
    "epochs": 600,
    "dim_output": 64,
    "datatype": "Stereo",
    "warmup_epochs": 200,
    "update_interval": 30,
    "graph_update_rate": 0.3,
    "gamma": 3.0,
    "graph_reg_weight": 0.0,
    "use_learnable_proj": False,
}


# =========================================================
# 4. 消融组配置
#    这里的参数你自己填
# =========================================================
ABLATION_CONFIGS = OrderedDict()

# 1) Baseline
# 保持你原文本代码中的 baseline 调用逻辑
ABLATION_CONFIGS["Baseline"] = {
    **common_params,
    "w_smooth": 0.0,
    "w_sharpen": 0.0,
    "gamma": 3.0,
    "warmup_epochs": 200,
    "update_interval": 30,
    "graph_update_rate": 0.3,
    "graph_reg_weight": 0.0,
    "use_learnable_proj": False,
    "use_original_baseline": True,
}

# 2) Smooth only
ABLATION_CONFIGS["Smooth only"] = {
    **common_params,
    "w_smooth": 0.01,
    "w_sharpen": 0.0,
    "gamma": 3.0,
    "warmup_epochs": 200,
    "update_interval": 20,
    "graph_update_rate": 0.3,
    "graph_reg_weight": 0.0,
    "use_learnable_proj": False,
    "use_original_baseline": False,
}

# 3) Sharpen only
ABLATION_CONFIGS["Sharpen only"] = {
    **common_params,
    "w_smooth": 0.0,
    "w_sharpen": 0.1,
    "gamma": 3,
    "warmup_epochs": 200,
    "update_interval": 20,
    "graph_update_rate": 0.3,
    "graph_reg_weight": 0.0,
    "use_learnable_proj": False,
    "use_original_baseline": False,
}

# 4) Ada-GraphST
ABLATION_CONFIGS["Ada-GraphST"] = {
    **common_params,
    "w_smooth": 0.05,
    "w_sharpen": 0.05,
    "gamma": 2.5,
    "warmup_epochs": 200,
    "update_interval": 30,
    "graph_update_rate": 0.3,
    "graph_reg_weight": 0.0,
    "use_learnable_proj": False,
    "use_original_baseline": False,
}

print("\n========== ABLATION CONFIGS ==========")
for name, cfg in ABLATION_CONFIGS.items():
    show_dict = {k: cfg.get(k) for k in [
        "w_smooth", "w_sharpen", "gamma", "warmup_epochs",
        "update_interval", "graph_update_rate", "graph_reg_weight",
        "use_learnable_proj", "use_original_baseline"
    ]}
    print(f"{name}: {show_dict}")


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
# 6. 运行单个 variant
# =========================================================
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


# =========================================================
# 7. 主循环
# =========================================================
all_records = []
variant_adatas = OrderedDict()
best_name, best_ari, best_nmi = None, -1.0, -1.0

for idx, (variant_name, variant_params) in enumerate(ABLATION_CONFIGS.items(), start=1):
    print(f"\n[{idx}/{len(ABLATION_CONFIGS)}] {variant_name}")

    try:
        ari, nmi, n_eval, adata_out = run_one_variant(
            adata_input=adata_raw,
            n_clusters=n_clusters,
            variant_name=variant_name,
            variant_params=variant_params
        )

        save_h5ad = os.path.join(
            RESULT_ROOT,
            f"MA_{variant_name.replace(' ', '_')}.h5ad"
        )
        adata_out.write(save_h5ad)

        record = {
            "dataset": "MA",
            "variant": variant_name,
            "label_col": label_col,
            "N_Clusters": n_clusters,
            "N_Obs_Eval": n_eval,
            "ARI": ari,
            "NMI": nmi,
            "save_h5ad": save_h5ad,
            "use_original_baseline": variant_params.get("use_original_baseline", False),
            "w_smooth": variant_params.get("w_smooth", np.nan),
            "w_sharpen": variant_params.get("w_sharpen", np.nan),
            "gamma": variant_params.get("gamma", np.nan),
            "warmup_epochs": variant_params.get("warmup_epochs", np.nan),
            "update_interval": variant_params.get("update_interval", np.nan),
            "graph_update_rate": variant_params.get("graph_update_rate", np.nan),
            "graph_reg_weight": variant_params.get("graph_reg_weight", np.nan),
            "use_learnable_proj": variant_params.get("use_learnable_proj", False),
        }
        all_records.append(record)
        variant_adatas[variant_name] = adata_out.copy()

        if (ari > best_ari) or (np.isclose(ari, best_ari) and nmi > best_nmi):
            best_name, best_ari, best_nmi = variant_name, ari, nmi

    except Exception as e:
        print(f"Variant [{variant_name}] failed: {e}")
        record = {
            "dataset": "MA",
            "variant": variant_name,
            "label_col": label_col,
            "N_Clusters": n_clusters,
            "N_Obs_Eval": np.nan,
            "ARI": np.nan,
            "NMI": np.nan,
            "Error": str(e),
            "use_original_baseline": variant_params.get("use_original_baseline", False),
            "w_smooth": variant_params.get("w_smooth", np.nan),
            "w_sharpen": variant_params.get("w_sharpen", np.nan),
            "gamma": variant_params.get("gamma", np.nan),
            "warmup_epochs": variant_params.get("warmup_epochs", np.nan),
            "update_interval": variant_params.get("update_interval", np.nan),
            "graph_update_rate": variant_params.get("graph_update_rate", np.nan),
            "graph_reg_weight": variant_params.get("graph_reg_weight", np.nan),
            "use_learnable_proj": variant_params.get("use_learnable_proj", False),
        }
        all_records.append(record)


# =========================================================
# 8. 保存结果
# =========================================================
all_results_df = pd.DataFrame(all_records)
variant_metric_map = {
    row["variant"]: {
        "ARI": row["ARI"],
        "NMI": row["NMI"]
    }
    for _, row in all_results_df.iterrows()
}
all_results_csv = os.path.join(RESULT_ROOT, "MA_Ablation_AllResults.csv")
all_results_df.to_csv(all_results_csv, index=False)

summary_df = all_results_df[[
    "dataset", "variant", "ARI", "NMI",
    "w_smooth", "w_sharpen", "gamma",
    "warmup_epochs", "update_interval",
    "graph_update_rate", "graph_reg_weight",
    "use_learnable_proj", "save_h5ad"
]].copy()
summary_csv = os.path.join(RESULT_ROOT, "MA_Ablation_Summary.csv")
summary_df.to_csv(summary_csv, index=False)

best_summary_df = pd.DataFrame([{
    "dataset": "MA",
    "label_col": label_col,
    "best_variant": best_name,
    "best_ARI": best_ari if best_ari >= 0 else np.nan,
    "best_NMI": best_nmi if best_nmi >= 0 else np.nan,
}])
best_summary_csv = os.path.join(RESULT_ROOT, "MA_Ablation_BestSummary.csv")
best_summary_df.to_csv(best_summary_csv, index=False)


# =========================================================
# 9. 画柱状图
#    MA 是单个数据集，因此这里直接画该数据集的消融对比
# =========================================================
plot_order = ["Baseline", "Smooth only", "Sharpen only", "Ada-GraphST"]
plot_colors = {
    "Baseline": "#B07AA1",
    "Smooth only": "#4E79A7",
    "Sharpen only": "#F28E2B",
    "Ada-GraphST": "#76B7B2",
}

plot_df = all_results_df.copy()
plot_df["variant"] = pd.Categorical(plot_df["variant"], categories=plot_order, ordered=True)
plot_df = plot_df.sort_values("variant").reset_index(drop=True)


def style_bar_axis(
    ax,
    tick_fontsize=14,
    label_fontsize=16,
    title_fontsize=17,
    spine_width=1.3
):
    """
    统一设置柱状图坐标轴字体：
    1. X/Y 轴刻度数值加大加粗
    2. Y 轴标签加大加粗
    3. 标题加大加粗
    4. 坐标轴边框加粗
    """
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
        width=spine_width,
        length=5
    )

    for label in ax.get_xticklabels():
        label.set_fontweight("bold")

    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.xaxis.label.set_size(label_fontsize)
    ax.xaxis.label.set_weight("bold")

    ax.yaxis.label.set_size(label_fontsize)
    ax.yaxis.label.set_weight("bold")

    ax.title.set_size(title_fontsize)
    ax.title.set_weight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)


# ---- ARI ----
fig, ax = plt.subplots(figsize=(8, 6))

x = np.arange(len(plot_df))
x_labels = plot_df["variant"].astype(str).tolist()

bars = ax.bar(
    x,
    plot_df["ARI"],
    color=[plot_colors[v] for v in x_labels],
    edgecolor="black",
    linewidth=1.0
)

ax.set_xticks(x)
ax.set_xticklabels(
    x_labels,
    rotation=20,
    ha="right",
    fontsize=14,
    fontweight="bold"
)

ax.set_ylabel("ARI", fontsize=16, fontweight="bold")
ax.set_title(
    "Mouse Brain MA Ablation Study (ARI)",
    fontsize=17,
    fontweight="bold"
)

ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.set_axisbelow(True)

style_bar_axis(
    ax,
    tick_fontsize=14,
    label_fontsize=16,
    title_fontsize=17,
    spine_width=1.3
)

for bar, val in zip(bars, plot_df["ARI"]):
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
ari_png = os.path.join(RESULT_ROOT, "MA_Ablation_ARI_barplot.png")
ari_pdf = os.path.join(RESULT_ROOT, "MA_Ablation_ARI_barplot.pdf")
plt.savefig(ari_png, dpi=300, bbox_inches="tight")
plt.savefig(ari_pdf, bbox_inches="tight")
plt.close()


# ---- NMI ----
fig, ax = plt.subplots(figsize=(8, 6))

x = np.arange(len(plot_df))
x_labels = plot_df["variant"].astype(str).tolist()

bars = ax.bar(
    x,
    plot_df["NMI"],
    color=[plot_colors[v] for v in x_labels],
    edgecolor="black",
    linewidth=1.0
)

ax.set_xticks(x)
ax.set_xticklabels(
    x_labels,
    rotation=20,
    ha="right",
    fontsize=14,
    fontweight="bold"
)

ax.set_ylabel("NMI", fontsize=16, fontweight="bold")
ax.set_title(
    "Mouse Brain MA Ablation Study (NMI)",
    fontsize=17,
    fontweight="bold"
)

ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.set_axisbelow(True)

style_bar_axis(
    ax,
    tick_fontsize=14,
    label_fontsize=16,
    title_fontsize=17,
    spine_width=1.3
)

for bar, val in zip(bars, plot_df["NMI"]):
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
nmi_png = os.path.join(RESULT_ROOT, "MA_Ablation_NMI_barplot.png")
nmi_pdf = os.path.join(RESULT_ROOT, "MA_Ablation_NMI_barplot.pdf")
plt.savefig(nmi_png, dpi=300, bbox_inches="tight")
plt.savefig(nmi_pdf, bbox_inches="tight")
plt.close()
# =========================================================
# 10. 额外可视化：
#     1) Spatial clustering 对比图
#     2) UMAP 对比图
# =========================================================

def get_spatial_xy(adata):
    """
    尽量鲁棒地提取空间坐标
    """
    if "spatial" in adata.obsm and adata.obsm["spatial"].shape[1] >= 2:
        xy = np.asarray(adata.obsm["spatial"])[:, :2]
        return xy[:, 0], xy[:, 1]

    candidate_pairs = [
        ("x", "y"),
        ("X", "Y"),
        ("array_col", "array_row"),
        ("imagecol", "imagerow"),
        ("x2", "x1"),   # 你 MA 数据里常见这个
        ("col", "row"),
    ]

    for cx, cy in candidate_pairs:
        if cx in adata.obs.columns and cy in adata.obs.columns:
            return adata.obs[cx].astype(float).values, adata.obs[cy].astype(float).values

    raise KeyError(
        "Cannot find spatial coordinates. "
        "Checked adata.obsm['spatial'] and common obs column pairs."
    )


def build_category_palette(categories):
    """
    从 tab20、tab20b、tab20c 中组合出前 60 种不同颜色。
    """
    n = len(categories)
    cmap_list = []
    for cmap_name in ['tab20', 'tab20b', 'tab20c']:
        cmap = plt.get_cmap(cmap_name)
        for i in range(20):
            cmap_list.append(cmap(i))
    palette = {}
    for i, cat in enumerate(categories):
        palette[str(cat)] = cmap_list[i % len(cmap_list)]
    return palette

def align_pred_to_ground_truth(adata, pred_col="domain", gt_col="ground_truth", out_col="domain_matched"):
    """
    用 Hungarian matching 把预测 cluster 尽量对齐到 GT 标签，
    这样不同 variant 的颜色更容易对比。
    """
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

    # 对没有匹配到的 cluster，保留原始名字
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


# ---------------------------------------------------------
# 10.1 准备 Ground Truth 调色板
# ---------------------------------------------------------
if "ground_truth" not in adata_raw.obs.columns:
    raise KeyError("ground_truth not found in adata_raw.obs, cannot plot ablation comparison.")

gt_categories = list(pd.Categorical(adata_raw.obs["ground_truth"].astype(str)).categories)
gt_palette = build_category_palette(gt_categories)

# 先给每个成功 variant 做 matched label
for variant_name in list(variant_adatas.keys()):
    variant_adatas[variant_name] = align_pred_to_ground_truth(
        variant_adatas[variant_name],
        pred_col="domain",
        gt_col="ground_truth",
        out_col="domain_matched"
    )


# ---------------------------------------------------------
# 10.2 Spatial clustering 对比图（右侧三列图例，显示原始簇名）
# ---------------------------------------------------------
spatial_order = ["Baseline", "Smooth only", "Sharpen only", "Ada-GraphST"]
available_spatial_order = [v for v in spatial_order if v in variant_adatas]

n_panels = 1 + len(available_spatial_order)
fig, axes = plt.subplots(1, n_panels, figsize=(4.8 * n_panels, 5.2))
if n_panels == 1:
    axes = [axes]

# 绘制 Ground Truth
plot_spatial_panel(
    ax=axes[0],
    adata=adata_raw,
    color_col="ground_truth",
    title="Ground Truth",
    palette=gt_palette,
    point_size=8,
    show_legend=False
)

# 绘制各变体
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
        show_legend=False
    )

plt.suptitle("Mouse Brain MA Ablation: Spatial Clustering Comparison", fontsize=15, fontweight="bold", y=1.02)

# ---------- 添加右侧全局图例（三列，显示原始簇名称） ----------
from matplotlib.patches import Patch

legend_handles = []
legend_labels = []

for cat in gt_categories:
    color = gt_palette.get(str(cat), (0.7, 0.7, 0.7, 1.0))
    legend_handles.append(Patch(facecolor=color, edgecolor="none"))
    legend_labels.append(str(cat))

# 先给右侧图例预留空间
# rect=[left, bottom, right, top]
# right=0.72 表示子图区域只占到整张图 72% 的宽度，右侧留给 legend
plt.tight_layout(rect=[0.00, 0.00, 0.72, 0.95])

# 图例固定放在整张 figure 的最右侧，并分成三列
fig.legend(
    handles=legend_handles,
    labels=legend_labels,
    loc="center left",
    bbox_to_anchor=(0.74, 0.50),   # 越大越靠右；可在 0.72~0.80 间微调
    bbox_transform=fig.transFigure,
    ncol=3,
    fontsize=7,
    frameon=False,
    title="Cluster",
    title_fontsize=9,
    columnspacing=1.0,
    handlelength=1.0,
    handletextpad=0.4,
    borderaxespad=0.0
)

spatial_png = os.path.join(RESULT_ROOT, "MA_Ablation_Spatial_Comparison.png")
spatial_pdf = os.path.join(RESULT_ROOT, "MA_Ablation_Spatial_Comparison.pdf")
plt.savefig(spatial_png, dpi=300, bbox_inches="tight")
plt.savefig(spatial_pdf, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# 10.3 UMAP 对比图（按 matched prediction 上色）
# ---------------------------------------------------------
variant_umaps = OrderedDict()
for variant_name in available_spatial_order:
    try:
        variant_umaps[variant_name] = compute_umap_for_plot(variant_adatas[variant_name], seed=SEED)
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

    plt.suptitle("Mouse Brain MA Ablation: UMAP Comparison (Matched Prediction)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    umap_pred_png = os.path.join(RESULT_ROOT, "MA_Ablation_UMAP_Prediction_Comparison.png")
    umap_pred_pdf = os.path.join(RESULT_ROOT, "MA_Ablation_UMAP_Prediction_Comparison.pdf")
    plt.savefig(umap_pred_png, dpi=300, bbox_inches="tight")
    plt.savefig(umap_pred_pdf, bbox_inches="tight")
    plt.close()
else:
    umap_pred_png = None
    umap_pred_pdf = None


# ---------------------------------------------------------
# 10.4 UMAP 对比图（按 Ground Truth 上色）
#     这个图能看 embedding 是否更贴近真实类别结构
# ---------------------------------------------------------
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

    plt.suptitle("Mouse Brain MA Ablation: UMAP Comparison (Ground Truth)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    umap_gt_png = os.path.join(RESULT_ROOT, "MA_Ablation_UMAP_GT_Comparison.png")
    umap_gt_pdf = os.path.join(RESULT_ROOT, "MA_Ablation_UMAP_GT_Comparison.pdf")
    plt.savefig(umap_gt_png, dpi=300, bbox_inches="tight")
    plt.savefig(umap_gt_pdf, bbox_inches="tight")
    plt.close()
else:
    umap_gt_png = None
    umap_gt_pdf = None


print("\nSaved spatial comparison to:", spatial_png)
if umap_pred_png is not None:
    print("Saved UMAP prediction comparison to:", umap_pred_png)
if umap_gt_png is not None:
    print("Saved UMAP GT comparison to:", umap_gt_png)

# =========================================================
# 11. 完成提示
# =========================================================
print("\n" + "=" * 120)
print("ALL MA ABLATION EXPERIMENTS FINISHED")
print("=" * 120)
print(best_summary_df.to_string(index=False))

print(f"\nSaved all results to: {all_results_csv}")
print(f"Saved summary to: {summary_csv}")
print(f"Saved best summary to: {best_summary_csv}")
print(f"Saved ARI barplot to: {ari_png}")
print(f"Saved NMI barplot to: {nmi_png}")