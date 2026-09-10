import os
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA

try:
    import torch
except ImportError:
    torch = None

try:
    import harmonypy as hm
except ImportError:
    raise ImportError("请先安装 harmonypy: pip install harmonypy")

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"

BATCH_PALETTE = ["#5B9BD5", "#ED7D31"]


def reset_seed(seed=41):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _normalize_index_str(idx):
    return pd.Index(idx.astype(str).str.strip())


def load_integrated_dataset(aligned_h5ad):
    if not os.path.exists(aligned_h5ad):
        raise FileNotFoundError(f"❌ 找不到对齐后的 h5ad 文件: {aligned_h5ad}")

    print(f"📥 正在读取对齐后的整合输入: {aligned_h5ad}")
    adata = sc.read_h5ad(aligned_h5ad)

    adata.var_names_make_unique()
    adata.obs_names = _normalize_index_str(adata.obs_names)
    adata.obs_names_make_unique()

    if "spatial" not in adata.obsm:
        raise KeyError("❌ adata.obsm['spatial'] 不存在。")
    if "batch" not in adata.obs.columns:
        raise KeyError("❌ adata.obs['batch'] 不存在。")

    batch_values = adata.obs["batch"].astype(str)
    if set(batch_values.unique()) == {"MA", "MP"}:
        adata.obs["batch"] = pd.Categorical(batch_values, categories=["MA", "MP"])
    else:
        adata.obs["batch"] = batch_values.astype("category")

    print(f"✅ n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    print(f"✅ batches={adata.obs['batch'].cat.categories.tolist()}")
    return adata


def to_counts_adata(adata):
    ad = adata.copy()
    if "counts" in ad.layers:
        ad.X = ad.layers["counts"].copy()
        print("✅ 使用 adata.layers['counts'] 作为输入")
    else:
        print("⚠️ 未找到 layers['counts']，直接使用 adata.X 作为输入")
    return ad


def compute_ilisi_from_rep(adata, rep_key="X_emb", seed=41):
    if rep_key not in adata.obsm:
        raise KeyError(f"❌ adata.obsm['{rep_key}'] 不存在，无法计算 iLISI")

    emb = np.asarray(adata.obsm[rep_key], dtype=np.float64)
    if emb.ndim != 2:
        raise ValueError(f"❌ {rep_key} 必须是二维矩阵，当前 shape={emb.shape}")
    if np.isnan(emb).any() or np.isinf(emb).any():
        raise ValueError(f"❌ {rep_key} 中存在 NaN / Inf")

    n_pca = min(20, emb.shape[1], emb.shape[0] - 1)
    n_pca = max(2, n_pca)

    adata.obsm["emb_pca"] = PCA(
        n_components=n_pca,
        random_state=seed
    ).fit_transform(emb)

    lisi = hm.compute_lisi(
        adata.obsm["emb_pca"],
        adata.obs[["batch"]],
        ["batch"]
    )
    return float(np.mean(lisi))


def mclust_R(adata, num_cluster, used_obsm="X_emb", key_added="domain", modelNames="EEE", random_seed=41):
    import rpy2.robjects as robjects
    from rpy2.robjects import r, numpy2ri

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
        res <- Mclust(x_mat, G=n_cluster, modelNames=model_name)
        cls <- res$classification
        """
    )

    mclust_res = np.array(r["cls"]).astype(int)
    adata.obs[key_added] = pd.Categorical(mclust_res.astype(str))
    return adata


def plot_umap_batch_domain(adata, method_name, ilisi, save_png, save_pdf, seed=41):
    if "emb_pca" not in adata.obsm:
        raise KeyError("❌ adata.obsm['emb_pca'] 不存在，无法画 UMAP")
    if "domain" not in adata.obs.columns:
        raise KeyError("❌ adata.obs['domain'] 不存在，无法画 Spatial Domains")

    sc.pp.neighbors(adata, use_rep="emb_pca", random_state=seed)
    sc.tl.umap(adata, random_state=seed)

    n_domain = adata.obs["domain"].astype(str).nunique()
    palette_domain = sc.pl.palettes.default_20 if n_domain <= 20 else sc.pl.palettes.default_28
    adata.uns["domain_colors"] = palette_domain[:n_domain]

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    sc.pl.umap(
        adata,
        color="batch",
        title=f"{method_name} (Batch Mixing)\niLISI: {ilisi:.3f}",
        show=False,
        ax=axs[0],
        frameon=False,
        s=28,
        palette=BATCH_PALETTE,
        legend_loc="right margin"
    )

    sc.pl.umap(
        adata,
        color="domain",
        title=f"{method_name} (Spatial Domains)",
        show=False,
        ax=axs[1],
        frameon=False,
        s=28,
        legend_loc="right margin"
    )

    for ax in axs:
        ax.set_facecolor("#EAEAF2")

    plt.tight_layout()
    plt.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.savefig(save_pdf, bbox_inches="tight")
    plt.close()