import os
import sys
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib

try:
    import scanpy.external as sce
except ImportError:
    raise ImportError(
        "没有检测到 scanpy.external，请确认当前 scanpy 安装完整。"
    )

PROJECT_ROOT = "/data2/liangyefeng/My_GraphST_Innovation"
PROJECT_SRC = os.path.join(PROJECT_ROOT, "src")

DATA_ROOT = "/data2/liangyefeng/My_GraphST_Innovation/data/Human-Breast-Cancer-Block-A"
ALIGNED_H5AD = os.path.join(
    DATA_ROOT,
    "PASTE_then_GraphST_section1_section2",
    "HBCA_section1_section2_PASTE_aligned_for_GraphST.h5ad"
)

SAVE_DIR = os.path.join(DATA_ROOT, "PASTE_then_Harmony_section1_section2")
os.makedirs(SAVE_DIR, exist_ok=True)

RESULT_H5AD = os.path.join(SAVE_DIR, "HBCA_section1_section2_Harmony_result.h5ad")
RESULT_CSV = os.path.join(SAVE_DIR, "HBCA_section1_section2_Harmony_iLISI_summary.csv")
UMAP_PNG = os.path.join(SAVE_DIR, "HBCA_section1_section2_Harmony_UMAP.png")
UMAP_PDF = os.path.join(SAVE_DIR, "HBCA_section1_section2_Harmony_UMAP.pdf")

SEED = 41

# Harmony / preprocessing 参数
MIN_CELLS = 5
N_TOP_GENES = 3000
N_PCS = 50
TARGET_SUM = 1e4
HVG_FLAVOR = "seurat_v3"   # 先在 counts 上选 HVG
SCALE_MAX_VALUE = 10

if PROJECT_SRC not in sys.path:
    sys.path.append(PROJECT_SRC)

from hbca_section1_section2_integration_common import (
    load_integrated_dataset,
    to_counts_adata,
    infer_n_clusters,
    compute_ilisi_from_rep,
    mclust_R,
)

# 字体设置，避免 PDF 字体问题
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"


def seed_torch(seed=41):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def ensure_counts_layer(adata):
    """
    确保 adata.layers['counts'] 存在。
    """
    if "counts" in adata.layers:
        counts = adata.layers["counts"]
    else:
        counts = adata.X.copy()
        adata.layers["counts"] = counts

    if sp.issparse(adata.layers["counts"]):
        adata.layers["counts"] = adata.layers["counts"].tocsr()
    else:
        adata.layers["counts"] = np.asarray(adata.layers["counts"])

    return adata


def maybe_make_batch_categorical(adata):
    if "batch" not in adata.obs.columns:
        raise ValueError("adata.obs 中没有 'batch' 列，Harmony 需要 batch 信息。")

    adata.obs["batch"] = adata.obs["batch"].astype(str).astype("category")
    return adata


def preprocess_for_harmony(adata, seed=41):
    """
    Harmony 前的标准流程：
    counts -> HVG -> normalize -> log1p -> scale -> PCA
    """
    adata_proc = adata.copy()

    # 用 counts 作为输入
    if "counts" in adata_proc.layers:
        adata_proc.X = adata_proc.layers["counts"].copy()

    if sp.issparse(adata_proc.X):
        adata_proc.X = adata_proc.X.tocsr()

    sc.pp.filter_genes(
        adata_proc,
        min_cells=MIN_CELLS
    )

    # 在原始 counts 上选 HVG，更适合 seurat_v3
    sc.pp.highly_variable_genes(
        adata_proc,
        n_top_genes=N_TOP_GENES,
        batch_key="batch",
        flavor=HVG_FLAVOR,
        subset=True,
    )

    sc.pp.normalize_total(
        adata_proc,
        target_sum=TARGET_SUM
    )

    sc.pp.log1p(adata_proc)

    sc.pp.scale(
        adata_proc,
        max_value=SCALE_MAX_VALUE
    )

    n_comps = min(
        N_PCS,
        max(2, adata_proc.n_vars - 1),
        max(2, adata_proc.n_obs - 1)
    )

    sc.tl.pca(
        adata_proc,
        n_comps=n_comps,
        svd_solver="arpack",
        random_state=seed,
    )

    return adata_proc, n_comps


def run_harmony_scanpy(adata_proc):
    """
    使用 scanpy.external.pp.harmony_integrate。
    输出保存在 adata_proc.obsm['X_pca_harmony']。
    """
    try:
        sce.pp.harmony_integrate(
            adata_proc,
            key="batch",
            basis="X_pca",
            adjusted_basis="X_pca_harmony",
        )
    except TypeError:
        # 兼容部分旧版参数差异
        sce.pp.harmony_integrate(
            adata_proc,
            "batch",
            basis="X_pca",
            adjusted_basis="X_pca_harmony",
        )

    if "X_pca_harmony" not in adata_proc.obsm:
        raise ValueError("Harmony 运行后没有生成 adata.obsm['X_pca_harmony']。")

    return adata_proc


def remove_all_legends(fig):
    """
    移除 figure 中所有图例，避免显示右侧的 S1/S3 或 1~20。
    """
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    for legend in list(fig.legends):
        legend.remove()


def plot_umap_batch_domain_no_legend(
    adata,
    method_name,
    save_png,
    save_pdf,
    seed=41,
    rep_key="X_pca_harmony"
):
    """
    绘制 batch/domain 的 UMAP 图：
    1. 不显示 iLISI；
    2. 不显示 batch 图例；
    3. 不显示 domain 图例。
    """
    if rep_key not in adata.obsm:
        raise KeyError(f"❌ adata.obsm['{rep_key}'] 不存在，无法绘制 UMAP。")

    if "batch" not in adata.obs.columns:
        raise KeyError("❌ adata.obs['batch'] 不存在，无法绘制 batch UMAP。")

    if "domain" not in adata.obs.columns:
        raise KeyError("❌ adata.obs['domain'] 不存在，无法绘制 domain UMAP。")

    adata.obs["batch"] = adata.obs["batch"].astype(str).astype("category")
    adata.obs["domain"] = adata.obs["domain"].astype(str).astype("category")

    # 基于 Harmony embedding 计算 UMAP
    sc.pp.neighbors(
        adata,
        use_rep=rep_key,
        random_state=seed
    )

    sc.tl.umap(
        adata,
        random_state=seed
    )

    # 设置 domain 颜色
    n_domain = len(adata.obs["domain"].cat.categories)

    if n_domain <= 20:
        adata.uns["domain_colors"] = sc.pl.palettes.default_20[:n_domain]
    elif n_domain <= 28:
        adata.uns["domain_colors"] = sc.pl.palettes.default_28[:n_domain]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))

    # Batch UMAP
    batch_kwargs = {
        "adata": adata,
        "color": "batch",
        "title": f"{method_name} Batch",
        "show": False,
        "ax": axs[0],
        "frameon": False,
        "s": 30,
        "legend_loc": None
    }

    if len(adata.obs["batch"].cat.categories) <= 2:
        batch_kwargs["palette"] = ["#5B9BD5", "#ED7D31"]

    sc.pl.umap(**batch_kwargs)

    # Domain UMAP
    sc.pl.umap(
        adata,
        color="domain",
        title=f"{method_name} Domain",
        show=False,
        ax=axs[1],
        frameon=False,
        s=30,
        legend_loc=None
    )

    # 二次确保所有 legend 被移除
    remove_all_legends(fig)

    fig.tight_layout()

    fig.savefig(
        save_png,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        save_pdf,
        bbox_inches="tight"
    )

    plt.close(fig)


if __name__ == "__main__":
    seed_torch(SEED)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1. 读取 PASTE 对齐后的输入 h5ad
    adata = load_integrated_dataset(ALIGNED_H5AD)
    adata = to_counts_adata(adata)
    adata = ensure_counts_layer(adata)
    adata = maybe_make_batch_categorical(adata)

    n_clusters, gt_col = infer_n_clusters(adata, default_n=20)

    print("Dataset: HBCA_section1_section2")
    print(f"n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    print(f"batches={adata.obs['batch'].cat.categories.tolist()}")
    print(f"n_clusters={n_clusters}")

    # 2. Harmony 前预处理
    adata_proc, actual_npcs = preprocess_for_harmony(
        adata,
        seed=SEED
    )

    print(
        f"After preprocessing: "
        f"n_obs={adata_proc.n_obs}, "
        f"n_vars={adata_proc.n_vars}, "
        f"n_pcs={actual_npcs}"
    )

    # 3. Harmony integration
    adata_proc = run_harmony_scanpy(adata_proc)

    harmony_emb = np.asarray(
        adata_proc.obsm["X_pca_harmony"],
        dtype=np.float32
    )

    # 4. 把 Harmony 表示写回原始 adata
    # 这里只过滤了基因，没有过滤细胞/spot，因此 obs 顺序一致
    if adata.n_obs != adata_proc.n_obs:
        raise ValueError(
            f"adata.n_obs ({adata.n_obs}) 与 adata_proc.n_obs ({adata_proc.n_obs}) 不一致，"
            "无法安全写回 Harmony embedding。"
        )

    adata.obsm["X_pca_harmony"] = harmony_emb.copy()
    adata.obsm["X_emb"] = harmony_emb.copy()   # 兼容你现有公共评估函数

    # 5. 计算 iLISI
    ilisi = compute_ilisi_from_rep(
        adata,
        rep_key="X_pca_harmony",
        seed=SEED
    )

    # 6. 在 Harmony embedding 上聚类，生成 domain
    adata = mclust_R(
        adata,
        num_cluster=n_clusters,
        used_obsm="X_pca_harmony",
        key_added="domain",
        random_seed=SEED
    )

    # 7. 画 UMAP 图：不显示 iLISI，不显示右侧图例
    plot_umap_batch_domain_no_legend(
        adata,
        method_name="Harmony",
        save_png=UMAP_PNG,
        save_pdf=UMAP_PDF,
        seed=SEED,
        rep_key="X_pca_harmony"
    )

    # 8. 保存结果 h5ad
    adata.write(RESULT_H5AD)

    # 9. 保存 iLISI summary csv
    pd.DataFrame([{
        "Dataset": "HBCA_section1_section2",
        "Method": "Harmony",
        "Seed": SEED,
        "Input_H5AD": ALIGNED_H5AD,
        "Result_H5AD": RESULT_H5AD,
        "GT_Col_For_NClusters": gt_col,
        "Device": device,
        "Min_Cells": MIN_CELLS,
        "N_Top_Genes": N_TOP_GENES,
        "N_PCs": actual_npcs,
        "Target_Sum": TARGET_SUM,
        "HVG_Flavor": HVG_FLAVOR,
        "Scale_Max_Value": SCALE_MAX_VALUE,
        "N_Clusters_For_Domain": n_clusters,
        "N_Obs_Total": int(adata.n_obs),
        "N_Vars": int(adata.n_vars),
        "iLISI": float(ilisi),
    }]).to_csv(RESULT_CSV, index=False)

    print(f"Harmony | iLISI={ilisi:.6f}")
    print("saved h5ad :", RESULT_H5AD)
    print("saved csv  :", RESULT_CSV)
    print("saved umap :", UMAP_PNG)
    print("saved umap :", UMAP_PDF)