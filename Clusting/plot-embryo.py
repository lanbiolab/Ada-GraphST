# -*- coding: utf-8 -*-

import os
import logging
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
from sklearn import metrics
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")

# =========================================================
# 0. 全局画图格式配置
# =========================================================
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

plt.rcParams["font.family"] = "Arial"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


# =========================================================
# 1. 路径配置
# =========================================================
project_root = "/data2/liangyefeng/My_GraphST_Innovation"

data_dir = os.path.join(
    project_root,
    "data",
    "Mouse-embryo"
)

sample_name = "E16.5_E2S3"

save_dir = os.path.join(
    data_dir,
    "MouseEmbryo_E16.5_E2S3_Cluster_Figures"
)

individual_save_dir = os.path.join(
    save_dir,
    "individual"
)

os.makedirs(save_dir, exist_ok=True)
os.makedirs(individual_save_dir, exist_ok=True)

# =========================================================
# 全局图形字号和点大小
# =========================================================
POINT_SIZE = 2          # 横向总图点大小，避免太大
INDIVIDUAL_POINT_SIZE = 6  # 单独保存图保持你原本的大小

METHOD_TITLE_FONTSIZE = 14
GT_TITLE_FONTSIZE = 14
ARI_FONTSIZE = 12
MISSING_TEXT_FONTSIZE = 12
LEGEND_FONTSIZE = 12

# =========================================================
# 2. h5ad 文件配置
# =========================================================
methods_config = [
    {
        "title": "STAGATE",
        "path": os.path.join(
            data_dir,
            "E16.5_E2S3_STAGATE_bin3_KNN_k10.h5ad"
        ),
        "pred_candidates": ["domain", "mclust"],
        "highlight": False,
    },
    {
        "title": "SpaceFlow",
        "path": os.path.join(
            data_dir,
            "E16.5_E2S3_SpaceFlow_bin3_z50_radius50.h5ad"
        ),
        "pred_candidates": ["domain", "mclust", "spaceflow_pred"],
        "highlight": False,
    },
    {
        "title": "SpaGCN",
        "path": os.path.join(
            data_dir,
            "E16.5_E2S3_SpaGCN_bin3.h5ad"
        ),
        "pred_candidates": ["domain", "spagcn_pred"],
        "highlight": False,
    },
    {
        "title": "conST",
        "path": os.path.join(
            data_dir,
            "E16.5_E2S3_conST_bin3_k20.h5ad"
        ),
        "pred_candidates": ["domain", "mclust", "conST_mclust"],
        "highlight": False,
    },
    {
        "title": "GraphST",
        "path": os.path.join(
            data_dir,
            "E16.5_E2S3_Baseline_Internal_bin3_radius50.h5ad"
        ),
        "pred_candidates": ["domain", "mclust"],
        "highlight": False,
    },
    {
        "title": "Ada-GraphST",
        "path": os.path.join(
            data_dir,
            "E16.5_E2S3_AdaGraphST_smooth0p02_sharpen0p03_bin3_radius50.h5ad"
        ),
        "pred_candidates": ["domain", "mclust"],
        "highlight": False,   # 不标红、不加星号
    },
]

# 用 GraphST baseline 作为 GT 和空间坐标参考
reference_h5ad = os.path.join(
    data_dir,
    "E16.5_E2S3_Baseline_Internal_bin3_radius50.h5ad"
)

gt_candidates = [
    "ground_truth",
    "annotation",
    "layer_guess_reordered",
    "layer_guess",
    "original_domain"
]


# =========================================================
# 3. 匈牙利算法对齐标签
# =========================================================
def align_labels_hungarian(y_true, y_pred):
    """
    用匈牙利算法把预测聚类标签映射到 GT 标签。

    作用：
    - ARI 计算仍然用原始预测标签；
    - 画图颜色时，把预测类别尽量对齐到 GT 类别颜色；
    - 这样不同方法的颜色更容易横向比较。
    """
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)

    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)

    cost_matrix = np.zeros(
        (len(unique_true), len(unique_pred)),
        dtype=float
    )

    for i, t in enumerate(unique_true):
        for j, p in enumerate(unique_pred):
            cost_matrix[i, j] = -np.sum(
                (y_true == t) & (y_pred == p)
            )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[unique_pred[c]] = unique_true[r]

    aligned_pred = np.asarray([
        mapping.get(p, p)
        for p in y_pred
    ])

    return aligned_pred


# =========================================================
# 4. 工具函数
# =========================================================
def find_gt_col(adata):
    for c in gt_candidates:
        if c in adata.obs.columns:
            return c

    cols = list(adata.obs.columns)

    for target in gt_candidates:
        for c in cols:
            if c.lower() == target.lower():
                return c

    for target in gt_candidates:
        for c in cols:
            if target.lower() in c.lower():
                return c

    return None


def find_pred_col(adata, pred_candidates):
    cols = list(adata.obs.columns)

    for c in pred_candidates:
        if c in cols:
            return c

    # 大小写兼容
    for target in pred_candidates:
        for c in cols:
            if c.lower() == target.lower():
                return c

    # 包含匹配
    for target in pred_candidates:
        for c in cols:
            if target.lower() in c.lower():
                return c

    return None


def has_valid_spatial(adata):
    if "spatial" not in adata.obsm:
        return False

    arr = np.asarray(adata.obsm["spatial"])

    if arr.ndim != 2:
        return False

    if arr.shape[0] != adata.n_obs:
        return False

    if arr.shape[1] < 2:
        return False

    return True


def attach_spatial_from_reference_if_missing(adata, adata_ref):
    """
    如果方法结果没有 spatial，则从 reference h5ad 按 obs_names 对齐补上。

    你的这些方法都是基于同一个 binning 结果生成的，
    理论上 obs_names 是一致的，所以可以安全按 bin id 对齐。
    """
    if has_valid_spatial(adata):
        arr = np.asarray(adata.obsm["spatial"])
        adata.obsm["spatial"] = arr[:, :2].astype(float)
        return adata

    if not has_valid_spatial(adata_ref):
        return adata

    common_idx = adata.obs_names.intersection(adata_ref.obs_names)

    if len(common_idx) != adata.n_obs:
        return adata

    ref_df = pd.DataFrame(
        np.asarray(adata_ref.obsm["spatial"])[:, :2],
        index=adata_ref.obs_names,
        columns=["x", "y"]
    )

    ref_df = ref_df.loc[adata.obs_names]

    adata.obsm["spatial"] = ref_df[["x", "y"]].to_numpy(dtype=float)

    return adata


def build_palette_from_gt_categories(categories):
    """
    构建论文风格颜色。

    你的 Mouse embryo annotation 类别可能比较多，
    所以这里先给一组清晰的固定颜色；
    如果类别数更多，再自动接 scanpy 的 default_102。
    """
    categories = list(categories)

    fallback_palette = [
        "#76B7B2",  # teal
        "#4E79A7",  # deep blue
        "#59A14F",  # deep green
        "#E15759",  # deep red
        "#9C6ADE",  # purple
        "#9C755F",  # brown
        "#E377C2",  # pink
        "#F28E2B",  # orange
        "#EDC948",  # yellow
        "#B07AA1",
        "#FF9DA7",
        "#86BCB6",
        "#BAB0AC",
        "#79706E",
        "#D37295",
        "#8CD17D",
        "#B6992D",
        "#499894",
        "#A0CBE8",
        "#FFBE7D",
        "#98DF8A",
        "#FF9896",
        "#C5B0D5",
        "#C49C94",
        "#F7B6D2",
        "#DBDB8D",
        "#9EDAE5",
        "#17BECF",
        "#BCBD22",
        "#7F7F7F",
    ]

    n_categories = len(categories)

    if n_categories <= len(fallback_palette):
        return fallback_palette[:n_categories]

    extra_needed = n_categories - len(fallback_palette)

    if hasattr(sc.pl.palettes, "default_102"):
        extra_palette = list(sc.pl.palettes.default_102)
    elif hasattr(sc.pl.palettes, "default_28"):
        extra_palette = list(sc.pl.palettes.default_28)
    else:
        extra_palette = list(plt.cm.tab20.colors)

    extra_colors = []
    while len(extra_colors) < extra_needed:
        extra_colors.extend(extra_palette)

    return fallback_palette + extra_colors[:extra_needed]


def draw_panel_border(ax, linewidth=1.0, color="black"):
    """
    额外画一层矩形边框。

    这样即使 scanpy 或 matplotlib 隐藏了 spine，
    也能保证每个聚类图外面有清晰边框。
    """
    rect = Rectangle(
        (0, 0),
        1,
        1,
        transform=ax.transAxes,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        clip_on=False,
        zorder=10000
    )
    ax.add_patch(rect)


def set_axis_missing(ax, title, msg):
    """
    缺失面板样式：
    - 中间显示 Missing / Error 信息；
    - 方法名放在图下面；
    - 保留黑色边框。
    """
    ax.set_axis_on()
    ax.set_frame_on(True)

    ax.text(
        0.5,
        0.5,
        msg,
        ha="center",
        va="center",
        fontsize=MISSING_TEXT_FONTSIZE,
        color="black"
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_facecolor("white")
    ax.set_aspect("equal")

    ax.tick_params(
        axis="both",
        which="both",
        left=False,
        right=False,
        bottom=False,
        top=False,
        labelleft=False,
        labelbottom=False
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")
        spine.set_zorder(10000)

    draw_panel_border(ax, linewidth=1.0, color="black")

    # 方法名放在图下方
    ax.text(
        0.5,
        -0.16,
        title,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=METHOD_TITLE_FONTSIZE,
        fontweight="normal",
        color="black",
        clip_on=False
    )


def style_panel_axis(ax, highlight=False):
    """
    每个聚类图统一样式：
    - 去掉坐标轴刻度；
    - 白底；
    - 保留完整黑色边框；
    - 不再对 Ada-GraphST 做红色高亮。
    """
    ax.set_axis_on()
    ax.set_frame_on(True)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_facecolor("white")
    ax.set_aspect("equal")

    ax.tick_params(
        axis="both",
        which="both",
        left=False,
        right=False,
        bottom=False,
        top=False,
        labelleft=False,
        labelbottom=False
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")
        spine.set_zorder(10000)

    # 额外加一层矩形边框，保证一定显示
    draw_panel_border(ax, linewidth=1.0, color="black")


def add_panel_titles(ax, method_name, ari_val=None, highlight=False):
    """
    示例图风格排版：
    - ARI 放在图上方；
    - 方法名放在图下方；
    - 不标红、不加星号；
    - 字号和前面箱线图保持一致。
    """
    # ARI 放在图上方
    if ari_val is not None:
        ax.text(
            0.5,
            1.04,
            f"ARI={ari_val:.2f}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=ARI_FONTSIZE,
            fontweight="normal",
            color="black",
            clip_on=False
        )

    # 方法名放在图下方
    if ari_val is None and "Ground Truth" in method_name:
        title_fontsize = GT_TITLE_FONTSIZE
    else:
        title_fontsize = METHOD_TITLE_FONTSIZE

    ax.text(
        0.5,
        -0.16,
        method_name,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=title_fontsize,
        fontweight="normal",
        color="black",
        clip_on=False
    )


def plot_spatial_panel(
    adata,
    color_key,
    ax,
    point_size=POINT_SIZE,
    show_legend=False,
    highlight=False
):
    """
    统一聚类图画法：
    - 白底；
    - 实心点；
    - 无边缘线；
    - 保留黑色边框；
    - 图例字体和箱线图字号统一。
    """
    sc.pl.embedding(
        adata,
        basis="spatial",
        color=color_key,
        show=False,
        ax=ax,
        frameon=True,          # 关键：必须打开 frame
        title="",
        s=point_size,
        alpha=1.0,
        edgecolors="none",
        linewidths=0,
        legend_loc="right margin" if show_legend else None,
        legend_fontsize=LEGEND_FONTSIZE,
        legend_fontoutline=1,
        na_in_legend=False
    )

    style_panel_axis(ax, highlight=highlight)


def prepare_gt_adata(adata_ref, gt_col):
    """
    准备用于画 Ground Truth 的 adata。
    过滤掉 ground_truth 缺失的点位，并统一类别和颜色。
    """
    gt_series = adata_ref.obs[gt_col]

    valid_mask = (
        (~pd.isnull(gt_series))
        & (gt_series.astype(str).str.lower() != "nan")
        & (gt_series.astype(str).str.lower() != "none")
    )

    adata_gt = adata_ref[valid_mask].copy()

    y_true = adata_gt.obs[gt_col].astype(str)

    # 为了稳定颜色顺序，这里用自然出现顺序，而不是默认字母排序
    gt_categories = pd.Index(y_true).drop_duplicates().tolist()

    adata_gt.obs["ground_truth"] = pd.Categorical(
        y_true,
        categories=gt_categories
    )

    palette = build_palette_from_gt_categories(gt_categories)
    adata_gt.uns["ground_truth_colors"] = palette

    return adata_gt, gt_categories, palette


def get_method_plot_adata(
    method_adata,
    adata_gt,
    pred_col,
    gt_categories,
    palette
):
    """
    构建某个方法用于画图的 AnnData。

    关键点：
    - ARI 用原始 pred 算；
    - 颜色用匈牙利算法对齐到 GT 类别；
    - 坐标统一使用 adata_gt 的 spatial，保证所有方法位置完全一致。
    """
    common_idx = adata_gt.obs_names.intersection(method_adata.obs_names)

    if len(common_idx) == 0:
        raise ValueError("No common spots between GT and method adata.")

    y_true_common = (
        adata_gt.obs.loc[common_idx, "ground_truth"]
        .astype(str)
        .values
    )

    y_pred_common = (
        method_adata.obs.loc[common_idx, pred_col]
        .astype(str)
        .values
    )

    ari_val = metrics.adjusted_rand_score(
        y_true_common,
        y_pred_common
    )

    aligned_pred = align_labels_hungarian(
        y_true_common,
        y_pred_common
    )

    adata_method_plot = adata_gt[common_idx].copy()

    # 预测簇数量可能大于/小于 GT 类别数。
    # 如果有没法映射到 GT 的类别，额外加进 categories，避免变成 NaN。
    extra_categories = [
        x for x in pd.unique(aligned_pred)
        if x not in gt_categories
    ]

    final_categories = list(gt_categories) + list(extra_categories)

    if len(extra_categories) > 0:
        extra_palette = build_palette_from_gt_categories(final_categories)
        final_palette = extra_palette
    else:
        final_palette = palette

    adata_method_plot.obs["Method_Aligned"] = pd.Categorical(
        aligned_pred,
        categories=final_categories
    )

    adata_method_plot.uns["Method_Aligned_colors"] = final_palette

    return adata_method_plot, ari_val, len(common_idx)


# =========================================================
# 5. 保存单个方法图
# =========================================================
def save_individual_plot(
    adata_plot,
    color_key,
    title,
    save_prefix,
    ari_val=None,
    highlight=False,
    point_size=POINT_SIZE,
    show_legend=True
):
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(5.0, 5.2),
        dpi=400,
        facecolor="white"
    )

    ax.set_facecolor("white")

    plot_spatial_panel(
        adata=adata_plot,
        color_key=color_key,
        ax=ax,
        point_size=point_size,
        show_legend=show_legend,
        highlight=False
    )

    add_panel_titles(
        ax=ax,
        method_name=title,
        ari_val=ari_val,
        highlight=False
    )

    plt.subplots_adjust(
        left=0.03,
        right=0.84 if show_legend else 0.98,
        top=0.86,
        bottom=0.20
    )

    pdf_path = os.path.join(
        individual_save_dir,
        f"{save_prefix}.pdf"
    )

    png_path = os.path.join(
        individual_save_dir,
        f"{save_prefix}.png"
    )

    plt.savefig(
        pdf_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    print(f"      Saved individual: {png_path}")


# =========================================================
# 6. 主程序
# =========================================================
if __name__ == "__main__":

    print("=" * 100)
    print(f"Start plotting clustering maps for {sample_name}")
    print("=" * 100)

    # -----------------------------------------------------
    # 6.1 读取 reference h5ad
    # -----------------------------------------------------
    if not os.path.exists(reference_h5ad):
        raise FileNotFoundError(
            f"Reference h5ad not found: {reference_h5ad}"
        )

    adata_ref = sc.read_h5ad(reference_h5ad)

    gt_col = find_gt_col(adata_ref)

    if gt_col is None:
        raise KeyError(
            f"GT column not found in reference h5ad. "
            f"Available obs columns: {list(adata_ref.obs.columns)}"
        )

    if not has_valid_spatial(adata_ref):
        raise KeyError(
            "Reference h5ad does not contain valid adata.obsm['spatial']."
        )

    adata_ref.obsm["spatial"] = np.asarray(
        adata_ref.obsm["spatial"]
    )[:, :2].astype(float)

    print(f"Reference h5ad: {reference_h5ad}")
    print(f"GT column: {gt_col}")
    print(f"Reference shape: {adata_ref.shape}")
    print(f"Reference obs columns: {list(adata_ref.obs.columns)}")

    # -----------------------------------------------------
    # 6.2 准备 GT
    # -----------------------------------------------------
    adata_gt, gt_categories, palette = prepare_gt_adata(
        adata_ref,
        gt_col
    )

    print(f"GT valid spots: {adata_gt.n_obs}")
    print(f"GT categories ({len(gt_categories)}):")
    print(gt_categories)

    # -----------------------------------------------------
    # 6.3 横向总图：GT + 6 methods = 1 x 7
    # -----------------------------------------------------
    n_cols = 1 + len(methods_config)

    fig, axs = plt.subplots(
        1,
        n_cols,
        figsize=(9.8, 3.2),
        dpi=400,
        facecolor="white",
        gridspec_kw={
            "wspace": 0.0
        }
    )

    axs = np.ravel(axs)

    for ax in axs:
        ax.set_facecolor("white")

    # ---- Ground Truth ----
    plot_spatial_panel(
        adata=adata_gt,
        color_key="ground_truth",
        ax=axs[0],
        point_size=POINT_SIZE,
        show_legend=False,
        highlight=False
    )

    add_panel_titles(
        axs[0],
        f"Ground Truth",
        ari_val=None,
        highlight=False
    )

    # 单独保存 GT
    save_individual_plot(
        adata_plot=adata_gt,
        color_key="ground_truth",
        title=f"Ground Truth ({sample_name})",
        save_prefix=f"{sample_name}_GroundTruth",
        ari_val=None,
        highlight=False,
        point_size=INDIVIDUAL_POINT_SIZE,
        show_legend=True
    )

    # ---- 各方法 ----
    summary_records = []

    for i, method_cfg in enumerate(methods_config, start=1):
        ax = axs[i]

        title = method_cfg["title"]
        method_path = method_cfg["path"]
        pred_candidates = method_cfg["pred_candidates"]
        highlight = False

        # 只在最后一个方法图保留图例，和你原来的代码一致
        show_legend = (i == len(methods_config))

        print("-" * 80)
        print(f"[{title}]")
        print(f"Path: {method_path}")

        if not os.path.exists(method_path):
            set_axis_missing(ax, title, "Missing")
            print(f"   Missing: {method_path}")

            summary_records.append({
                "Sample": sample_name,
                "Method": title,
                "H5AD_Path": method_path,
                "Pred_Col": "",
                "Common_Spots": 0,
                "ARI": np.nan,
                "Status": "Missing",
                "Error": "h5ad not found",
            })
            continue

        try:
            adata_m = sc.read_h5ad(method_path)
        except Exception as e:
            set_axis_missing(ax, title, f"Read Error\n{e}")
            print(f"   Read Error: {e}")

            summary_records.append({
                "Sample": sample_name,
                "Method": title,
                "H5AD_Path": method_path,
                "Pred_Col": "",
                "Common_Spots": 0,
                "ARI": np.nan,
                "Status": "Read_Error",
                "Error": str(e),
            })
            continue

        adata_m = attach_spatial_from_reference_if_missing(
            adata_m,
            adata_ref
        )

        if not has_valid_spatial(adata_m):
            set_axis_missing(ax, title, "Spatial Missing")
            print(f"   Spatial Missing")

            summary_records.append({
                "Sample": sample_name,
                "Method": title,
                "H5AD_Path": method_path,
                "Pred_Col": "",
                "Common_Spots": 0,
                "ARI": np.nan,
                "Status": "Spatial_Missing",
                "Error": "adata.obsm['spatial'] missing or invalid",
            })
            continue

        pred_col = find_pred_col(
            adata_m,
            pred_candidates
        )

        if pred_col is None:
            set_axis_missing(ax, title, "Pred Col Missing")
            print(f"   Pred Col Missing | obs={list(adata_m.obs.columns)}")

            summary_records.append({
                "Sample": sample_name,
                "Method": title,
                "H5AD_Path": method_path,
                "Pred_Col": "",
                "Common_Spots": 0,
                "ARI": np.nan,
                "Status": "Pred_Col_Missing",
                "Error": f"Available obs columns: {list(adata_m.obs.columns)}",
            })
            continue

        try:
            adata_method_plot, ari_val, n_common = get_method_plot_adata(
                method_adata=adata_m,
                adata_gt=adata_gt,
                pred_col=pred_col,
                gt_categories=gt_categories,
                palette=palette
            )

            plot_spatial_panel(
                adata=adata_method_plot,
                color_key="Method_Aligned",
                ax=ax,
                point_size=POINT_SIZE,
                show_legend=show_legend,
                highlight=False
            )

            add_panel_titles(
                ax=ax,
                method_name=title,
                ari_val=ari_val,
                highlight=False
            )

            save_individual_plot(
                adata_plot=adata_method_plot,
                color_key="Method_Aligned",
                title=title,
                save_prefix=f"{sample_name}_{title.replace('-', '_').replace(' ', '_')}",
                ari_val=ari_val,
                highlight=False,
                point_size=INDIVIDUAL_POINT_SIZE,
                show_legend=True
            )

            print(
                f"   OK | common={n_common} | "
                f"pred_col={pred_col} | ARI={ari_val:.6f}"
            )

            summary_records.append({
                "Sample": sample_name,
                "Method": title,
                "H5AD_Path": method_path,
                "Pred_Col": pred_col,
                "Common_Spots": int(n_common),
                "ARI": float(ari_val),
                "Status": "OK",
                "Error": "",
            })

        except Exception as e:
            set_axis_missing(ax, title, f"Plot Error\n{e}")
            print(f"   Plot Error: {e}")

            summary_records.append({
                "Sample": sample_name,
                "Method": title,
                "H5AD_Path": method_path,
                "Pred_Col": pred_col,
                "Common_Spots": 0,
                "ARI": np.nan,
                "Status": "Plot_Error",
                "Error": str(e),
            })

    # -----------------------------------------------------
    # 6.4 保存横向总图
    # -----------------------------------------------------
    plt.subplots_adjust(
        left=0.005,
        right=0.90,
        top=0.82,
        bottom=0.26,
        wspace=0.0
    )

    save_path_pdf = os.path.join(
        save_dir,
        f"Fig_MouseEmbryo_{sample_name}_All_Methods_paper_style.pdf"
    )

    save_path_png = os.path.join(
        save_dir,
        f"Fig_MouseEmbryo_{sample_name}_All_Methods_paper_style.png"
    )

    plt.savefig(
        save_path_pdf,
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.savefig(
        save_path_png,
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    print("=" * 100)
    print("Combined figure saved:")
    print(save_path_pdf)
    print(save_path_png)

    # -----------------------------------------------------
    # 6.5 保存 ARI 和绘图状态汇总
    # -----------------------------------------------------
    summary_df = pd.DataFrame(summary_records)

    summary_csv = os.path.join(
        save_dir,
        f"Fig_MouseEmbryo_{sample_name}_All_Methods_plot_summary.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False
    )

    print("Summary CSV saved:")
    print(summary_csv)

    print("=" * 100)
    print("Plotting finished.")
    print(summary_df)
    print("=" * 100)