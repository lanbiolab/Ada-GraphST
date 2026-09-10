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
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

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


def _clean_str_series(s):
    s = s.astype(str).str.strip()
    bad = s.isin(["", "nan", "None", "NA", "NaN", "null"])
    s = s.copy()
    s[bad] = np.nan
    return s


def _infer_batch_from_obs_or_names(adata):
    """
    优先从 obs 常见列里自动寻找 batch；
    这个数据集里实际 batch 在 adata.obs['data'] 中，比如 S1 / S3。
    如果 obs 里没有，再尝试从 obs_names 中识别。
    """
    candidate_cols = [
        "batch",
        "data",
        "section",
        "section_name",
        "section_id",
        "slice",
        "slices",
        "sample",
        "sample_id",
        "orig.ident",
        "orig_ident",
        "library_id",
        "library",
        "dataset",
        "source",
        "group",
    ]

    for col in candidate_cols:
        if col in adata.obs.columns:
            vals = _clean_str_series(adata.obs[col])
            nunique = vals.dropna().nunique()
            if nunique >= 2:
                adata.obs["batch"] = pd.Categorical(vals)
                print(f"✅ 使用 adata.obs['{col}'] 作为 batch")
                print(f"   -> unique values: {vals.dropna().unique().tolist()[:10]}")
                return adata, col

    idx_lower = pd.Index(adata.obs_names.astype(str)).str.lower()
    batch = pd.Series(index=adata.obs_names, dtype=object)

    mask1 = idx_lower.str.contains(r"section[_\-\s]?1|sec[_\-\s]?1|\bs1\b", regex=True)
    mask2 = idx_lower.str.contains(r"section[_\-\s]?2|sec[_\-\s]?2|\bs2\b", regex=True)

    if mask1.any() and mask2.any():
        batch.loc[mask1] = "section1"
        batch.loc[mask2] = "section2"
        adata.obs["batch"] = pd.Categorical(batch)
        print("✅ 从 obs_names 推断 batch 成功: ['section1', 'section2']")
        return adata, "obs_names"

    preview = []
    for col in adata.obs.columns:
        try:
            vals = _clean_str_series(adata.obs[col]).dropna().unique().tolist()[:5]
            preview.append(f"{col}: {vals}")
        except Exception:
            pass

    raise KeyError(
        "❌ adata.obs['batch'] 不存在，而且也无法从常见列或 obs_names 自动推断。\n"
        f"当前 obs columns:\n{list(adata.obs.columns)}\n\n"
        f"部分列示例值:\n" + "\n".join(preview[:20])
    )


def load_integrated_dataset(aligned_h5ad):
    if not os.path.exists(aligned_h5ad):
        raise FileNotFoundError(f"❌ 找不到 h5ad 文件: {aligned_h5ad}")

    print(f"📥 正在读取整合输入: {aligned_h5ad}")
    adata = sc.read_h5ad(aligned_h5ad)

    adata.var_names_make_unique()
    adata.obs_names = _normalize_index_str(adata.obs_names)
    adata.obs_names_make_unique()

    if "spatial" not in adata.obsm:
        raise KeyError(
            f"❌ adata.obsm['spatial'] 不存在。\n"
            f"当前 obsm keys: {list(adata.obsm.keys())}"
        )

    adata, batch_source = _infer_batch_from_obs_or_names(adata)
    adata.obs["batch"] = _clean_str_series(adata.obs["batch"]).astype("category")

    print(f"✅ n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    print(f"✅ batch source={batch_source}")
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


def infer_n_clusters(adata, default_n=20):
    candidate_cols = [
        "ground_truth",
        "original_domain",
        "label",
        "annotation",
        "celltype",
        "cell_type",
        "manual_annotation",
    ]
    for col in candidate_cols:
        if col in adata.obs.columns:
            s = _clean_str_series(adata.obs[col]).dropna()
            n_clusters = s.nunique()
            if n_clusters >= 2:
                print(f"✅ 使用 {col} 推断聚类数: {n_clusters}")
                return int(n_clusters), col

    print(f"⚠️ 未找到可用 GT 列，聚类数回退为默认值: {default_n}")
    return int(default_n), None


def compute_ilisi_from_rep(adata, rep_key="X_emb", seed=41):
    if rep_key not in adata.obsm:
        raise KeyError(f"❌ adata.obsm['{rep_key}'] 不存在，无法计算 iLISI")

    emb = np.asarray(adata.obsm[rep_key], dtype=np.float64)
    if emb.ndim != 2:
        raise ValueError(f"❌ {rep_key} 必须是二维矩阵，当前 shape={emb.shape}")
    if np.isnan(emb).any() or np.isinf(emb).any():
        raise ValueError(f"❌ {rep_key} 中存在 NaN / Inf")

    n_pca = min(20, emb.shape[1], max(2, emb.shape[0] - 1))
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


def _fallback_cluster_sklearn(
    x,
    num_cluster,
    random_seed=41,
    prefer_gmm=True
):
    """
    mclust 失败时的稳健回退：
    1) GaussianMixture
    2) KMeans
    返回从 1 开始的标签，尽量与 mclust 风格一致
    """
    x = np.asarray(x, dtype=np.float64)

    if prefer_gmm:
        try:
            gmm = GaussianMixture(
                n_components=int(num_cluster),
                covariance_type="full",
                reg_covar=1e-6,
                max_iter=500,
                n_init=5,
                random_state=random_seed
            )
            labels = gmm.fit_predict(x)
            labels = labels.astype(int) + 1
            print("⚠️ mclust 失败，已回退到 sklearn GaussianMixture")
            return labels
        except Exception as e:
            print(f"⚠️ GaussianMixture 也失败: {e}")

    km = KMeans(
        n_clusters=int(num_cluster),
        random_state=random_seed,
        n_init=20
    )
    labels = km.fit_predict(x)
    labels = labels.astype(int) + 1
    print("⚠️ mclust / GaussianMixture 均失败，已回退到 sklearn KMeans")
    return labels


def mclust_R(adata, num_cluster, used_obsm="X_emb", key_added="domain", modelNames="EEE", random_seed=41):
    x = np.asarray(adata.obsm[used_obsm], dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"{used_obsm} must be 2D, but got shape {x.shape}")
    if not np.isfinite(x).all():
        bad_count = np.size(x) - np.isfinite(x).sum()
        raise ValueError(f"{used_obsm} contains NaN/Inf values, bad_count={bad_count}")

    mclust_res = None
    r_error = None

    try:
        import rpy2.robjects as robjects
        from rpy2.robjects import r, numpy2ri

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
              cls <- NULL
            } else {
              cls <- res$classification
            }
            """
        )

        cls_obj = r["cls"]

        # R NULL -> Python 里通常表现成 NULLType
        if cls_obj is None or str(type(cls_obj)).lower().find("null") >= 0:
            raise ValueError("mclust returned NULL classification")

        cls_arr = np.array(cls_obj, dtype=object)
        cls_series = pd.Series(cls_arr, index=adata.obs_names)
        cls_series = pd.to_numeric(cls_series, errors="coerce")

        if cls_series.isna().all():
            raise ValueError("mclust returned all-NA classification")

        if cls_series.isna().sum() > 0:
            raise ValueError(f"mclust returned {cls_series.isna().sum()} invalid labels")

        mclust_res = cls_series.astype(int).values
        print("✅ mclust 聚类成功")

    except Exception as e:
        r_error = e
        print(f"⚠️ mclust 失败: {e}")

    if mclust_res is None:
        mclust_res = _fallback_cluster_sklearn(
            x,
            num_cluster=num_cluster,
            random_seed=random_seed,
            prefer_gmm=True
        )

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
    if n_domain <= 20:
        palette_domain = sc.pl.palettes.default_20[:n_domain]
    elif n_domain <= 28:
        palette_domain = sc.pl.palettes.default_28[:n_domain]
    else:
        palette_domain = sc.pl.palettes.default_102[:n_domain]
    adata.uns["domain_colors"] = palette_domain

    n_batch = adata.obs["batch"].astype(str).nunique()
    if n_batch <= len(BATCH_PALETTE):
        batch_palette = BATCH_PALETTE[:n_batch]
    elif n_batch <= 20:
        batch_palette = sc.pl.palettes.default_20[:n_batch]
    elif n_batch <= 28:
        batch_palette = sc.pl.palettes.default_28[:n_batch]
    else:
        batch_palette = sc.pl.palettes.default_102[:n_batch]

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
        palette=batch_palette,
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