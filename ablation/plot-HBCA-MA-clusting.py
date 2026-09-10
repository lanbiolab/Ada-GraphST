# -*- coding: utf-8 -*-

import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.optimize import linear_sum_assignment


# =========================================================
# 0. 全局画图格式配置
# =========================================================
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"


# =========================================================
# 1. 字体参数：和前面统一
# =========================================================
ARI_FONTSIZE = 14
METHOD_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 12
LEGEND_TITLE_FONTSIZE = 14

POINT_SIZE = 8
HBCA_POINT_SIZE = 3
MA_POINT_SIZE = 8

SPINE_LINEWIDTH = 1.0


# =========================================================
# 2. 路径配置
# =========================================================
PROJECT_ROOT = "/data2/liangyefeng/My_GraphST_Innovation"

# ---------- HBCA ----------
HBCA_RESULT_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "Human-Breast-Cancer-Block-A",
    "section1",
    "HBCA_GraphST_Ablation"
)

HBCA_RESULT_CSV = os.path.join(
    HBCA_RESULT_ROOT,
    "HBCA_section1_ablation_results.csv"
)

HBCA_SPATIAL_PNG = os.path.join(
    HBCA_RESULT_ROOT,
    "HBCA_section1_Ablation_Spatial_Comparison.png"
)

HBCA_SPATIAL_PDF = os.path.join(
    HBCA_RESULT_ROOT,
    "HBCA_section1_Ablation_Spatial_Comparison.pdf"
)

# ---------- Mouse Brain MA ----------
MA_RESULT_ROOT = os.path.join(
    PROJECT_ROOT,
    "Result_MA_Ablation"
)

MA_RESULT_CSV = os.path.join(
    MA_RESULT_ROOT,
    "MA_Ablation_AllResults.csv"
)

MA_SPATIAL_PNG = os.path.join(
    MA_RESULT_ROOT,
    "MA_Ablation_Spatial_Comparison.png"
)

MA_SPATIAL_PDF = os.path.join(
    MA_RESULT_ROOT,
    "MA_Ablation_Spatial_Comparison.pdf"
)


# =========================================================
# 3. 消融组顺序
# =========================================================
VARIANT_ORDER = [
    "Baseline",
    "Smooth only",
    "Sharpen only",
    "Ada-GraphST"
]


# =========================================================
# 4. 通用工具函数
# =========================================================
def normalize_variant_name(x):
    x_raw = str(x).strip()
    x_norm = x_raw.lower().replace("_", " ")

    mapping = {
        "baseline": "Baseline",
        "smooth only": "Smooth only",
        "sharpen only": "Sharpen only",
        "ada graphst": "Ada-GraphST",
        "ada-graphst": "Ada-GraphST",
    }

    return mapping.get(x_norm, x_raw)


def get_spatial_xy(adata):
    """
    尽量鲁棒地提取空间坐标。
    """
    if "spatial" in adata.obsm:
        arr = np.asarray(adata.obsm["spatial"])
        if arr.ndim == 2 and arr.shape[1] >= 2:
            xy = arr[:, :2]
            return xy[:, 0], xy[:, 1]

    candidate_pairs = [
        ("x", "y"),
        ("X", "Y"),
        ("array_col", "array_row"),
        ("imagecol", "imagerow"),
        ("x2", "x1"),
        ("col", "row"),
    ]

    for cx, cy in candidate_pairs:
        if cx in adata.obs.columns and cy in adata.obs.columns:
            return (
                adata.obs[cx].astype(float).values,
                adata.obs[cy].astype(float).values
            )

    raise KeyError(
        "Cannot find spatial coordinates from adata.obsm['spatial'] "
        "or common obs coordinate columns."
    )


def align_pred_to_ground_truth(
    adata,
    pred_col="domain",
    gt_col="ground_truth",
    out_col="domain_matched"
):
    """
    用 Hungarian matching 把预测 cluster 尽量对齐到 GT 标签。
    """
    if pred_col not in adata.obs.columns:
        if "mclust" in adata.obs.columns:
            adata.obs[pred_col] = adata.obs["mclust"].astype("category")
        else:
            raise KeyError(f"{pred_col} not found in adata.obs")

    if gt_col not in adata.obs.columns:
        raise KeyError(f"{gt_col} not found in adata.obs")

    valid_mask = (
        (~pd.isna(adata.obs[pred_col]))
        & (~pd.isna(adata.obs[gt_col]))
    )

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

    # 没匹配到的 cluster 保留原始标签
    raw_pred = adata.obs[pred_col].astype(str)
    mapped = mapped.where(~mapped.isna(), raw_pred)

    adata.obs[out_col] = pd.Categorical(mapped)

    return adata


def attach_ground_truth_from_reference(adata, adata_ref):
    """
    如果某个 h5ad 里没有 ground_truth，则从参考 adata 按 obs_names 补上。
    """
    if "ground_truth" in adata.obs.columns:
        return adata

    if "ground_truth" not in adata_ref.obs.columns:
        return adata

    common_idx = adata.obs_names.intersection(adata_ref.obs_names)

    if len(common_idx) != adata.n_obs:
        return adata

    adata.obs["ground_truth"] = adata_ref.obs.loc[
        adata.obs_names,
        "ground_truth"
    ].values

    return adata


def load_metric_map(csv_path, variant_col):
    """
    读取每个消融组的 ARI，用于显示在空间图上方。
    """
    if not os.path.exists(csv_path):
        print(f"[Warning] 结果 CSV 不存在，ARI 将不会显示: {csv_path}")
        return {}

    df = pd.read_csv(csv_path)

    if variant_col not in df.columns or "ARI" not in df.columns:
        print(
            f"[Warning] CSV 缺少 {variant_col} 或 ARI 列，ARI 将不会显示。"
            f"当前列名: {list(df.columns)}"
        )
        return {}

    df = df.copy()
    df["Variant_norm"] = df[variant_col].apply(normalize_variant_name)
    df["ARI"] = pd.to_numeric(df["ARI"], errors="coerce")

    metric_map = {}
    for _, row in df.iterrows():
        v = row["Variant_norm"]
        ari = row["ARI"]
        if pd.notna(ari):
            metric_map[v] = float(ari)

    return metric_map


def resolve_h5ad_path(
    result_root,
    variant_name,
    filename_func,
    csv_path=None,
    variant_col=None
):
    """
    优先按固定命名找 h5ad；
    找不到时，尝试从 CSV 的 save_h5ad / Result_H5AD 列里找。
    """
    file_name = filename_func(variant_name)
    path = os.path.join(result_root, file_name)

    if os.path.exists(path):
        return path

    if csv_path is not None and os.path.exists(csv_path) and variant_col is not None:
        df = pd.read_csv(csv_path)

        if variant_col in df.columns:
            df = df.copy()
            df["Variant_norm"] = df[variant_col].apply(normalize_variant_name)

            sub = df[df["Variant_norm"] == variant_name].copy()

            for col in ["save_h5ad", "Result_H5AD", "result_h5ad", "h5ad_path"]:
                if col in sub.columns and not sub.empty:
                    candidate = str(sub[col].iloc[0])
                    if candidate and candidate != "nan" and os.path.exists(candidate):
                        return candidate

    return path


def load_variant_adatas(result_root, csv_path, variant_col, filename_func):
    """
    读取 4 个消融组对应的 h5ad。
    """
    variant_adatas = OrderedDict()

    for variant_name in VARIANT_ORDER:
        h5ad_path = resolve_h5ad_path(
            result_root=result_root,
            variant_name=variant_name,
            filename_func=filename_func,
            csv_path=csv_path,
            variant_col=variant_col
        )

        if not os.path.exists(h5ad_path):
            print(f"[Warning] 缺少 h5ad，跳过 {variant_name}: {h5ad_path}")
            continue

        adata = sc.read_h5ad(h5ad_path)

        if "domain" not in adata.obs.columns and "mclust" in adata.obs.columns:
            adata.obs["domain"] = adata.obs["mclust"].astype("category")

        variant_adatas[variant_name] = adata

        print(f"[OK] Loaded {variant_name}: {h5ad_path}")

    if len(variant_adatas) == 0:
        raise ValueError(f"没有读取到任何 h5ad，请检查目录: {result_root}")

    return variant_adatas


def plot_spatial_panel(
    ax,
    adata,
    color_col,
    top_text,
    bottom_text,
    palette,
    point_size=POINT_SIZE,
    show_legend=False
):
    """
    空间聚类 panel：
    - ARI 放在图上方；
    - 方法名 / Ground Truth 放在图下方；
    - 不使用 ax.set_title，避免标题挤在图上方。
    """
    x, y = get_spatial_xy(adata)

    vals = adata.obs[color_col].astype(str).fillna("NA").values
    cats = list(pd.Categorical(adata.obs[color_col].astype(str)).categories)

    for cat in cats:
        mask = vals == str(cat)
        ax.scatter(
            x[mask],
            y[mask],
            s=point_size,
            c=[palette.get(str(cat), (0.7, 0.7, 0.7, 1.0))],
            linewidths=0,
            label=str(cat)
        )

    # 去掉坐标刻度
    ax.set_xticks([])
    ax.set_yticks([])

    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_facecolor("white")

    # 边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_LINEWIDTH)
        spine.set_color("black")

    # ARI 放在图上面
    if top_text is not None and str(top_text).strip() != "":
        ax.text(
            0.5,
            1.06,
            top_text,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=ARI_FONTSIZE,
            fontweight="normal",
            color="black",
            clip_on=False
        )

    # 方法名 / Ground Truth 放在图下面
    if bottom_text is not None and str(bottom_text).strip() != "":
        ax.text(
            0.5,
            -0.08,
            bottom_text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=METHOD_LABEL_FONTSIZE,
            fontweight="normal",
            color="black",
            clip_on=False
        )

    if show_legend:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            markerscale=2
        )


# =========================================================
# 5. HBCA 专用颜色逻辑：保持你原来 HBCA 代码的 tab20 逻辑
# =========================================================
def build_hbca_category_palette(categories):
    n = len(categories)
    cmap = plt.get_cmap("tab20", max(n, 3))

    palette = {}
    for i, cat in enumerate(categories):
        palette[str(cat)] = cmap(i)

    return palette


def replot_hbca_spatial_comparison():
    print("\n" + "=" * 100)
    print("Replot HBCA section1 spatial comparison")
    print("=" * 100)

    metric_map = load_metric_map(
        csv_path=HBCA_RESULT_CSV,
        variant_col="Variant"
    )

    def hbca_filename_func(variant_name):
        return f"HBCA_section1_{variant_name.replace(' ', '_')}.h5ad"

    variant_adatas = load_variant_adatas(
        result_root=HBCA_RESULT_ROOT,
        csv_path=HBCA_RESULT_CSV,
        variant_col="Variant",
        filename_func=hbca_filename_func
    )

    # 用第一个成功读取的 h5ad 作为 GT 和坐标参考
    first_key = next(iter(variant_adatas))
    adata_ref = variant_adatas[first_key].copy()

    if "ground_truth" not in adata_ref.obs.columns:
        raise KeyError(
            "HBCA h5ad 中没有 ground_truth，无法画 Ground Truth panel。"
        )

    gt_categories = list(
        pd.Categorical(adata_ref.obs["ground_truth"].astype(str)).categories
    )
    gt_palette = build_hbca_category_palette(gt_categories)

    available_order = [v for v in VARIANT_ORDER if v in variant_adatas]

    # 对齐预测标签到 ground truth
    for variant_name in available_order:
        variant_adatas[variant_name] = attach_ground_truth_from_reference(
            variant_adatas[variant_name],
            adata_ref
        )

        variant_adatas[variant_name] = align_pred_to_ground_truth(
            variant_adatas[variant_name],
            pred_col="domain",
            gt_col="ground_truth",
            out_col="domain_matched"
        )

    n_panels = 1 + len(available_order)

    # HBCA 图整体紧凑，同时右侧留空间放两列图例
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(3.25 * n_panels, 4.8),
        dpi=300
    )

    fig.patch.set_facecolor("white")

    if n_panels == 1:
        axes = [axes]

    # Ground Truth：名字放下面，上面不放 ARI
    plot_spatial_panel(
        ax=axes[0],
        adata=adata_ref,
        color_col="ground_truth",
        top_text="",
        bottom_text="Ground Truth",
        palette=gt_palette,
        point_size=HBCA_POINT_SIZE,
        show_legend=False
    )

    # 各消融组：ARI 放上面，方法名放下面
    for i, variant_name in enumerate(available_order, start=1):
        ari_val = metric_map.get(variant_name, np.nan)

        if pd.notna(ari_val):
            top_text = f"ARI={ari_val:.3f}"
        else:
            top_text = ""

        plot_spatial_panel(
            ax=axes[i],
            adata=variant_adatas[variant_name],
            color_col="domain_matched",
            top_text=top_text,
            bottom_text=variant_name,
            palette=gt_palette,
            point_size=HBCA_POINT_SIZE,
            show_legend=False
        )

    # ---------- HBCA 全局图例：放在右侧，两列 ----------
    legend_handles = []
    legend_labels = []

    for cat in gt_categories:
        color = gt_palette.get(str(cat), (0.7, 0.7, 0.7, 1.0))
        legend_handles.append(
            Patch(facecolor=color, edgecolor="none")
        )
        legend_labels.append(str(cat))

    # 不添加总标题
    # rect 的 right=0.74：左侧 74% 用来放聚类图，右侧留给两列图例
    # w_pad 越小，聚类图之间越近
    plt.tight_layout(
        rect=[0.00, 0.10, 0.74, 0.94],
        w_pad=0.05
    )

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="center left",
        bbox_to_anchor=(0.755, 0.50),
        bbox_transform=fig.transFigure,
        ncol=2,                         # 关键：右侧两列
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
        columnspacing=0.9,
        handlelength=1.0,
        handletextpad=0.35,
        borderaxespad=0.0
    )

    plt.savefig(
        HBCA_SPATIAL_PNG,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.savefig(
        HBCA_SPATIAL_PDF,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close()

    print(f"[Saved] {HBCA_SPATIAL_PNG}")
    print(f"[Saved] {HBCA_SPATIAL_PDF}")


# =========================================================
# 6. MA 专用颜色逻辑：保持你原来 MA 代码的 tab20/tab20b/tab20c 逻辑
# =========================================================
def build_ma_category_palette(categories):
    cmap_list = []

    for cmap_name in ["tab20", "tab20b", "tab20c"]:
        cmap = plt.get_cmap(cmap_name)
        for i in range(20):
            cmap_list.append(cmap(i))

    palette = {}
    for i, cat in enumerate(categories):
        palette[str(cat)] = cmap_list[i % len(cmap_list)]

    return palette


def replot_ma_spatial_comparison():
    print("\n" + "=" * 100)
    print("Replot Mouse Brain MA spatial comparison")
    print("=" * 100)

    metric_map = load_metric_map(
        csv_path=MA_RESULT_CSV,
        variant_col="variant"
    )

    def ma_filename_func(variant_name):
        return f"MA_{variant_name.replace(' ', '_')}.h5ad"

    variant_adatas = load_variant_adatas(
        result_root=MA_RESULT_ROOT,
        csv_path=MA_RESULT_CSV,
        variant_col="variant",
        filename_func=ma_filename_func
    )

    # 用第一个成功读取的 h5ad 作为 GT 和坐标参考
    first_key = next(iter(variant_adatas))
    adata_ref = variant_adatas[first_key].copy()

    if "ground_truth" not in adata_ref.obs.columns:
        raise KeyError(
            "MA h5ad 中没有 ground_truth，无法画 Ground Truth panel。"
        )

    gt_categories = list(
        pd.Categorical(adata_ref.obs["ground_truth"].astype(str)).categories
    )
    gt_palette = build_ma_category_palette(gt_categories)

    available_order = [v for v in VARIANT_ORDER if v in variant_adatas]

    # 对齐预测标签到 ground truth
    for variant_name in available_order:
        variant_adatas[variant_name] = attach_ground_truth_from_reference(
            variant_adatas[variant_name],
            adata_ref
        )

        variant_adatas[variant_name] = align_pred_to_ground_truth(
            variant_adatas[variant_name],
            pred_col="domain",
            gt_col="ground_truth",
            out_col="domain_matched"
        )

    n_panels = 1 + len(available_order)

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(4.8 * n_panels, 5.2),
        dpi=300
    )

    fig.patch.set_facecolor("white")

    if n_panels == 1:
        axes = [axes]

    # Ground Truth：名字放下面，上面不放 ARI
    plot_spatial_panel(
        ax=axes[0],
        adata=adata_ref,
        color_col="ground_truth",
        top_text="",
        bottom_text="Ground Truth",
        palette=gt_palette,
        point_size=MA_POINT_SIZE,
        show_legend=False
    )

    # 各消融组：ARI 放上面，方法名放下面
    for i, variant_name in enumerate(available_order, start=1):
        ari_val = metric_map.get(variant_name, np.nan)

        if pd.notna(ari_val):
            top_text = f"ARI={ari_val:.3f}"
        else:
            top_text = ""

        plot_spatial_panel(
            ax=axes[i],
            adata=variant_adatas[variant_name],
            color_col="domain_matched",
            top_text=top_text,
            bottom_text=variant_name,
            palette=gt_palette,
            point_size=MA_POINT_SIZE,
            show_legend=False
        )

    # ---------- 右侧全局图例：保持你原来 MA 代码的三列图例逻辑 ----------
    legend_handles = []
    legend_labels = []

    for cat in gt_categories:
        color = gt_palette.get(str(cat), (0.7, 0.7, 0.7, 1.0))
        legend_handles.append(
            Patch(facecolor=color, edgecolor="none")
        )
        legend_labels.append(str(cat))

    # 不添加总标题。
    # 右侧给图例留空间，上方给 ARI 留空间，下方给方法名留空间。
    plt.tight_layout(
        rect=[0.00, 0.08, 0.72, 0.94],
        w_pad=0.8
    )

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="center left",
        bbox_to_anchor=(0.74, 0.50),
        bbox_transform=fig.transFigure,
        ncol=3,
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
        title="Cluster",
        title_fontsize=LEGEND_TITLE_FONTSIZE,
        columnspacing=1.0,
        handlelength=1.0,
        handletextpad=0.4,
        borderaxespad=0.0
    )

    plt.savefig(
        MA_SPATIAL_PNG,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.savefig(
        MA_SPATIAL_PDF,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close()

    print(f"[Saved] {MA_SPATIAL_PNG}")
    print(f"[Saved] {MA_SPATIAL_PDF}")


# =========================================================
# 7. 主程序
# =========================================================
if __name__ == "__main__":
    replot_hbca_spatial_comparison()
    replot_ma_spatial_comparison()

    print("\n全部空间聚类对比图已重新生成。")