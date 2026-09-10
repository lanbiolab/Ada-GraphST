import os
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import scanpy as sc
import scipy.sparse as sp
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib

try:
    import scanpy.external as sce
except ImportError:
    raise ImportError(
        "没有检测到 scanpy.external，请确认当前 scanpy 安装完整。"
    )

try:
    from harmonypy.lisi import compute_lisi
except ImportError:
    raise ImportError(
        "没有检测到 harmonypy，请先安装：pip install harmonypy"
    )

ALIGNED_H5AD = "/data2/liangyefeng/My_GraphST_Innovation/data/7.Mouse_Breast_Cancer_Sample_1/mouse_breast_cancer_sample1_section1&2.h5ad"

SAVE_DIR = "/data2/liangyefeng/My_GraphST_Innovation/data/7.Mouse_Breast_Cancer_Sample_1/Harmony_section1_section2"
os.makedirs(SAVE_DIR, exist_ok=True)

RESULT_H5AD = os.path.join(SAVE_DIR, "mouse_breast_cancer_sample1_section1_section2_Harmony_result.h5ad")
RESULT_CSV = os.path.join(SAVE_DIR, "mouse_breast_cancer_sample1_section1_section2_Harmony_iLISI_summary.csv")
UMAP_PNG = os.path.join(SAVE_DIR, "mouse_breast_cancer_sample1_section1_section2_Harmony_UMAP.png")
UMAP_PDF = os.path.join(SAVE_DIR, "mouse_breast_cancer_sample1_section1_section2_Harmony_UMAP.pdf")

SEED = 41

MIN_CELLS = 5
N_TOP_GENES = 3000
N_PCS = 50
TARGET_SUM = 1e4
HVG_FLAVOR = "seurat_v3"
SCALE_MAX_VALUE = 10
LISI_PERPLEXITY = 30

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
        for col in ["data", "slice_name", "slices", "section"]:
            if col in adata.obs.columns:
                adata.obs["batch"] = adata.obs[col].astype(str)
                break

    if "batch" not in adata.obs.columns:
        raise ValueError("adata.obs 中没有 'batch' 列，Harmony 需要 batch 信息。")

    adata.obs["batch"] = adata.obs["batch"].astype(str).astype("category")
    return adata


def load_integrated_dataset(path):
    adata = sc.read_h5ad(path)
    adata.var_names_make_unique()
    return adata


def to_counts_adata(adata):
    adata = adata.copy()
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    return adata


def infer_n_clusters(adata, default_n=20):
    candidate_cols = [
        "ground_truth", "original_domain", "domain_gt", "gt", "label",
        "labels", "annotation", "cell_type", "region", "tissue",
        "manual_annot", "cluster"
    ]

    for col in candidate_cols:
        if col in adata.obs.columns:
            vals = pd.Series(adata.obs[col]).astype(str)
            vals = vals[~vals.isna()]
            vals = vals[~vals.isin(["nan", "NA", "None", "unknown", "undetermined"])]
            n = vals.nunique()
            if n >= 2:
                return int(n), col

    return int(default_n), "default"


def compute_ilisi_from_rep(
    adata,
    rep_key="X_pca_harmony",
    seed=41,
    batch_col="batch",
    perplexity=30
):
    if rep_key not in adata.obsm:
        raise KeyError(f"{rep_key} 不在 adata.obsm 中。")

    X = np.asarray(adata.obsm[rep_key], dtype=np.float64)
    meta = adata.obs[[batch_col]].copy()

    lisi_res = compute_lisi(
        X=X,
        metadata=meta,
        label_colnames=[batch_col],
        perplexity=perplexity,
    )

    if isinstance(lisi_res, pd.DataFrame):
        vals = lisi_res[batch_col].to_numpy(dtype=float)
    else:
        vals = np.asarray(lisi_res, dtype=float)
        if vals.ndim == 2:
            vals = vals[:, 0]

    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        raise ValueError("iLISI 计算结果为空。")

    return float(np.mean(vals))


def mclust_R(
    adata,
    num_cluster,
    used_obsm="X_pca_harmony",
    key_added="domain",
    random_seed=41
):
    X = np.asarray(adata.obsm[used_obsm], dtype=np.float64)

    if X.ndim != 2:
        raise ValueError(f"{used_obsm} 不是二维矩阵，当前 shape={X.shape}")

    finite_col_mask = np.isfinite(X).all(axis=0)
    var_col_mask = np.var(X, axis=0) > 1e-12
    keep_col_mask = finite_col_mask & var_col_mask

    if keep_col_mask.sum() == 0:
        raise ValueError(f"{used_obsm} 所有列都无效，无法聚类。")

    X_use = X[:, keep_col_mask]

    bad_row_mask = ~np.isfinite(X_use).all(axis=1)
    if bad_row_mask.any():
        n_bad = int(bad_row_mask.sum())
        raise ValueError(f"{used_obsm} 中有 {n_bad} 行存在 NaN/Inf，无法聚类。")

    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri

        numpy2ri.activate()
        ro.r("suppressMessages(library(mclust))")
        ro.globalenv["emb"] = X_use
        ro.globalenv["n_cluster"] = int(num_cluster)
        ro.globalenv["seed"] = int(random_seed)

        ro.r(
            """
            set.seed(seed)
            cls <- NULL
            model_candidates <- c("EEE", "VVV", "EEV", "VEV", "EII", "VII")
            for (mn in model_candidates) {
              res <- tryCatch(
                Mclust(emb, G=n_cluster, modelNames=mn),
                error = function(e) NULL
              )
              if (!is.null(res) && !is.null(res$classification)) {
                cls <- res$classification
                break
              }
            }
            """
        )

        cls_r = ro.r("cls")

        if cls_r is not None:
            cls_np = np.array(cls_r)
            if cls_np is not None and len(cls_np) == adata.n_obs:
                cls_np = cls_np.astype(int)
                adata.obs[key_added] = pd.Categorical(cls_np.astype(str))
                print("[mclust_R] 使用 R/mclust 成功。")
                return adata

        print("[mclust_R] R/mclust 未返回有效 classification，回退到 GaussianMixture。")

    except Exception as e:
        print(f"[mclust_R] R/mclust 失败，回退到 GaussianMixture。原因: {e}")

    gmm = GaussianMixture(
        n_components=int(num_cluster),
        covariance_type="full",
        random_state=int(random_seed),
        reg_covar=1e-5,
        n_init=5,
    )

    cls = gmm.fit_predict(X_use) + 1
    adata.obs[key_added] = pd.Categorical(cls.astype(str))

    print("[mclust_R] 已回退到 sklearn GaussianMixture。")
    return adata


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


def plot_umap_batch_domain(
    adata,
    method_name="Harmony",
    ilisi=None,
    save_png=None,
    save_pdf=None,
    seed=41
):
    """
    绘制 batch/domain 的 UMAP 图：
    1. 不显示 iLISI；
    2. 不显示 batch 图例；
    3. 不显示 domain 图例。
    """
    plot_adata = adata.copy()

    if "batch" not in plot_adata.obs.columns:
        raise KeyError("adata.obs 中不存在 'batch'，无法绘制 batch UMAP。")

    if "domain" not in plot_adata.obs.columns:
        raise KeyError("adata.obs 中不存在 'domain'，无法绘制 domain UMAP。")

    plot_adata.obs["batch"] = plot_adata.obs["batch"].astype(str).astype("category")
    plot_adata.obs["domain"] = plot_adata.obs["domain"].astype(str).astype("category")

    sc.pp.neighbors(
        plot_adata,
        use_rep="X_pca_harmony",
        random_state=seed
    )

    sc.tl.umap(
        plot_adata,
        random_state=seed
    )

    # 设置 domain 颜色
    n_domain = len(plot_adata.obs["domain"].cat.categories)

    if n_domain <= 20:
        plot_adata.uns["domain_colors"] = sc.pl.palettes.default_20[:n_domain]
    elif n_domain <= 28:
        plot_adata.uns["domain_colors"] = sc.pl.palettes.default_28[:n_domain]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Batch UMAP：不显示 iLISI，不显示 legend
    batch_kwargs = {
        "adata": plot_adata,
        "color": "batch",
        "ax": axes[0],
        "show": False,
        "title": f"{method_name} | batch",
        "frameon": False,
        "legend_loc": None,
    }

    if len(plot_adata.obs["batch"].cat.categories) <= 2:
        batch_kwargs["palette"] = ["#5B9BD5", "#ED7D31"]

    sc.pl.umap(**batch_kwargs)

    # Domain UMAP：不显示 legend
    sc.pl.umap(
        plot_adata,
        color="domain",
        ax=axes[1],
        show=False,
        title=f"{method_name} | domain",
        frameon=False,
        legend_loc=None,
    )

    # 二次确保所有 legend 被移除
    remove_all_legends(fig)

    fig.tight_layout()

    if save_png is not None:
        fig.savefig(
            save_png,
            dpi=300,
            bbox_inches="tight"
        )

    if save_pdf is not None:
        fig.savefig(
            save_pdf,
            bbox_inches="tight"
        )

    plt.close(fig)


def preprocess_for_harmony(adata, seed=41):
    adata_proc = adata.copy()

    if "counts" in adata_proc.layers:
        adata_proc.X = adata_proc.layers["counts"].copy()

    if sp.issparse(adata_proc.X):
        adata_proc.X = adata_proc.X.tocsr()

    sc.pp.filter_genes(
        adata_proc,
        min_cells=MIN_CELLS
    )

    sc.pp.highly_variable_genes(
        adata_proc,
        n_top_genes=min(N_TOP_GENES, adata_proc.n_vars),
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
    try:
        sce.pp.harmony_integrate(
            adata_proc,
            key="batch",
            basis="X_pca",
            adjusted_basis="X_pca_harmony",
        )
    except TypeError:
        sce.pp.harmony_integrate(
            adata_proc,
            "batch",
            basis="X_pca",
            adjusted_basis="X_pca_harmony",
        )

    if "X_pca_harmony" not in adata_proc.obsm:
        raise ValueError("Harmony 运行后没有生成 adata.obsm['X_pca_harmony']。")

    return adata_proc


if __name__ == "__main__":
    seed_torch(SEED)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    adata = load_integrated_dataset(ALIGNED_H5AD)
    adata = to_counts_adata(adata)
    adata = ensure_counts_layer(adata)
    adata = maybe_make_batch_categorical(adata)

    n_clusters, gt_col = infer_n_clusters(adata, default_n=20)

    print("Dataset: mouse_breast_cancer_sample1_section1_section2")
    print(f"n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    print(f"batches={adata.obs['batch'].cat.categories.tolist()}")
    print(f"n_clusters={n_clusters}")

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

    adata_proc = run_harmony_scanpy(adata_proc)

    harmony_emb = np.asarray(
        adata_proc.obsm["X_pca_harmony"],
        dtype=np.float32
    )

    if adata.n_obs != adata_proc.n_obs:
        raise ValueError(
            f"adata.n_obs ({adata.n_obs}) 与 adata_proc.n_obs ({adata_proc.n_obs}) 不一致，"
            "无法安全写回 Harmony embedding。"
        )

    adata.obsm["X_pca_harmony"] = harmony_emb.copy()
    adata.obsm["X_emb"] = harmony_emb.copy()

    ilisi = compute_ilisi_from_rep(
        adata,
        rep_key="X_pca_harmony",
        seed=SEED,
        batch_col="batch",
        perplexity=LISI_PERPLEXITY,
    )

    adata = mclust_R(
        adata,
        num_cluster=n_clusters,
        used_obsm="X_pca_harmony",
        key_added="domain",
        random_seed=SEED
    )

    plot_umap_batch_domain(
        adata,
        method_name="Harmony",
        ilisi=ilisi,
        save_png=UMAP_PNG,
        save_pdf=UMAP_PDF,
        seed=SEED
    )

    adata.write(RESULT_H5AD)

    pd.DataFrame([{
        "Dataset": "mouse_breast_cancer_sample1_section1_section2",
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