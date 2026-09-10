# save as: plot_spatial_clustering_from_result_h5ad.py

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import silhouette_score
import scipy.sparse as sp

# =========================================================
# 0. 图形设置
# =========================================================
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"

OUT_DIR = (
    "/data2/liangyefeng/My_GraphST_Innovation/Result/"
    "Spatial_clustering_visualization"
)
os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# 1. 这里填你已经生成的 result h5ad
# =========================================================
DATASET_CONFIG = {
    "Human breast cancer": {
        "GraphST": (
            "/data2/liangyefeng/My_GraphST_Innovation/Result/"
            "HBCA_section1_section2_vertical_integration_from_aligned_h5ad/"
            "HBCA_section1_section2_baseline_result.h5ad"
        ),
        "Ada-GraphST": (
            "/data2/liangyefeng/My_GraphST_Innovation/Result/"
            "HBCA_section1_section2_vertical_integration_from_aligned_h5ad/"
            "HBCA_section1_section2_ours_result.h5ad"
        ),
    },

    "Mouse brain": {
        "GraphST": (
            "/data2/liangyefeng/My_GraphST_Innovation/data/"
            "Mouse-Brain-Section-2-Sagittal-Anterior-and-Posterior/"
            "PASTE_then_GraphST_MA_MP/"
            "MouseBrain_MA_MP_baseline_result.h5ad"
        ),
        "Ada-GraphST": (
            "/data2/liangyefeng/My_GraphST_Innovation/data/"
            "Mouse-Brain-Section-2-Sagittal-Anterior-and-Posterior/"
            "PASTE_then_GraphST_MA_MP/"
            "MouseBrain_MA_MP_ours_result.h5ad"
        ),
    },

    "Mouse Breast Cancer": {
        "GraphST": (
            "/data2/liangyefeng/My_GraphST_Innovation/Result/"
            "MouseBreastCancer_vertical_integration_h5ad/"
            "MouseBreastCancer_baseline_result.h5ad"
        ),
        "Ada-GraphST": (
            "/data2/liangyefeng/My_GraphST_Innovation/Result/"
            "MouseBreastCancer_vertical_integration_h5ad/"
            "MouseBreastCancer_ours_result.h5ad"
        ),
    },
}

METHOD_ORDER = ["GraphST", "Ada-GraphST"]


# =========================================================
# 2. 工具函数
# =========================================================
def get_cluster_key(adata):
    """
    自动识别聚类列。
    优先 domain，其次 mclust。
    """
    if "domain" in adata.obs.columns:
        return "domain"
    if "mclust" in adata.obs.columns:
        return "mclust"
    raise KeyError(
        f"找不到聚类列。当前 obs columns = {list(adata.obs.columns)}"
    )


def get_batch_key(adata):
    """
    自动识别切片/批次列。兼容单切片和多切片。
    """
    candidates = ["batch", "section", "section_name", "slice", "slices", "data"]

    for key in candidates:
        if key in adata.obs.columns:
            return key

    # 如果没有任何候选列，创建一个默认的虚拟批次以兼容单切片数据绘图
    adata.obs["batch"] = "single_slice"
    return "batch"


def get_embed_key(adata):
    """
    自动寻找降维特征矩阵的键名，用于计算 Silhouette Score。
    GraphST / Ada-GraphST 通常使用 'emb', 'latent' 或 'feat' 作为低维表征。
    """
    candidates = ["emb", "latent", "feat", "X_pca"]
    for key in candidates:
        if key in adata.obsm.keys():
            return key
    return None


def prepare_adata(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 h5ad 文件: {path}")

    adata = sc.read_h5ad(path)

    if "spatial" not in adata.obsm:
        raise KeyError(f"{path} 中找不到 adata.obsm['spatial']")

    cluster_key = get_cluster_key(adata)
    batch_key = get_batch_key(adata)
    embed_key = get_embed_key(adata)

    adata.obs[cluster_key] = adata.obs[cluster_key].astype(str).astype("category")
    adata.obs[batch_key] = adata.obs[batch_key].astype(str).astype("category")

    return adata, cluster_key, batch_key, embed_key


def make_shared_palette(adatas, cluster_keys):
    """
    保证 GraphST 和 Ada-GraphST 使用同一套颜色。
    """
    all_labels = []

    for adata, key in zip(adatas, cluster_keys):
        labels = adata.obs[key].astype(str).unique().tolist()
        all_labels.extend(labels)

    all_labels = sorted(list(set(all_labels)), key=lambda x: int(x) if x.isdigit() else x)
    n = len(all_labels)

    if n <= 20:
        base_palette = sc.pl.palettes.default_20
    elif n <= 28:
        base_palette = sc.pl.palettes.default_28
    else:
        base_palette = sc.pl.palettes.default_102

    color_map = {
        label: base_palette[i % len(base_palette)]
        for i, label in enumerate(all_labels)
    }

    return color_map


def plot_one_panel(ax, adata, cluster_key, batch_key, batch_value, color_map, title, embed_key):
    sub = adata[adata.obs[batch_key].astype(str) == str(batch_value)].copy()

    coords = np.asarray(sub.obsm["spatial"])
    labels = sub.obs[cluster_key].astype(str).values
    colors = [color_map[x] for x in labels]

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=colors,
        s=6,  # 严格保持原始尺寸。若需更小，可改为 8 或 6
        linewidths=0,
        alpha=0.95,
    )

    # pad=35 强行把子标题往上推，给分数腾出空间
    ax.set_title(title, fontsize=16, pad=35)
    ax.set_aspect("equal")
    ax.axis("off")

    # 空间转录组图一般需要翻转 y 轴
    ax.invert_yaxis()

    # ==========================================
    # 计算并标注 Silhouette Score (轮廓系数)
    # ==========================================
    try:
        if embed_key is not None and embed_key in sub.obsm:
            feat_matrix = sub.obsm[embed_key]
        else:
            feat_matrix = sub.X.toarray() if sp.issparse(sub.X) else sub.X

        unique_labels = np.unique(labels)

        if len(unique_labels) > 1 and feat_matrix is not None:
            sil_score = silhouette_score(feat_matrix, labels)

            # 放置在图框外部正上方 (y=1.05)，va='bottom' 保证文字往上方生长，绝不挡图
            ax.text(
                0.5, 1.05, f"Silhouette: {sil_score:.3f}",
                transform=ax.transAxes,
                fontsize=12,
                fontweight='bold',
                color='black',
                ha='center',
                va='bottom',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='white',
                    alpha=0.8,
                    edgecolor='lightgray'
                )
            )
            print(f"  [{title}] 轮廓系数: {sil_score:.3f}")
        else:
            print(f"  [{title}] 类别太少或无特征矩阵，跳过计算轮廓系数")
    except Exception as e:
        print(f"  [{title}] 轮廓系数计算失败: {e}")


def plot_dataset_spatial_comparison(dataset_name, graphst_path, ada_path):
    print("\n" + "=" * 80)
    print(f"正在绘制: {dataset_name}")
    print("=" * 80)

    adata_g, cluster_g, batch_g, embed_g = prepare_adata(graphst_path)
    adata_a, cluster_a, batch_a, embed_a = prepare_adata(ada_path)

    batches_g = adata_g.obs[batch_g].astype(str).unique().tolist()
    batches_a = adata_a.obs[batch_a].astype(str).unique().tolist()
    batches = [b for b in batches_g if b in batches_a]

    if len(batches) == 0:
        raise ValueError("GraphST 和 Ada-GraphST 没有共同 batch，请检查 batch 列。")

    color_map = make_shared_palette([adata_g, adata_a], [cluster_g, cluster_a])

    n_batches = len(batches)

    fig, axes = plt.subplots(
        n_batches,
        2,
        figsize=(8.5, 4.0 * n_batches),
        squeeze=False,
    )

    for i, batch in enumerate(batches):
        plot_one_panel(
            axes[i, 0], adata_g, cluster_g, batch_g, batch, color_map, f"GraphST - {batch}", embed_g
        )
        plot_one_panel(
            axes[i, 1], adata_a, cluster_a, batch_a, batch, color_map, f"Ada-GraphST - {batch}", embed_a
        )

    # 主标题也往上提一点，防止和拉高的子标题重叠
    fig.suptitle(
        f"{dataset_name}: spatial clustering visualization",
        fontsize=16,
        y=1.08,
    )

    plt.tight_layout()

    safe_name = (
        dataset_name.replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )

    png_path = os.path.join(
        OUT_DIR,
        f"{safe_name}_spatial_clustering_GraphST_vs_AdaGraphST.png",
    )
    pdf_path = os.path.join(
        OUT_DIR,
        f"{safe_name}_spatial_clustering_GraphST_vs_AdaGraphST.pdf",
    )

    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# =========================================================
# 3. 主程序
# =========================================================
if __name__ == "__main__":
    for dataset_name, cfg in DATASET_CONFIG.items():
        graphst_path = cfg["GraphST"]
        ada_path = cfg["Ada-GraphST"]

        try:
            plot_dataset_spatial_comparison(
                dataset_name=dataset_name,
                graphst_path=graphst_path,
                ada_path=ada_path,
            )
        except FileNotFoundError as e:
            print(f"⚠️ 跳过 {dataset_name}: {e}")

    print("\n全部 spatial clustering 可视化图及轮廓系数计算完成。")