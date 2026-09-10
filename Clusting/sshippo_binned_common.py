import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse
from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score


def reset_seed(seed=41):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def safe_mclust_R(
    adata,
    num_cluster,
    used_obsm="X_emb",
    key_added="domain",
    modelNames="EEE",
    random_seed=41
):
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
        raise ValueError(f"mclust returned {na_count} invalid labels for {used_obsm}")

    adata.obs[key_added] = cls_series.astype(int).astype("category")
    return adata


def load_binned_h5ad(data_path, force_label_col="cluster"):
    adata = sc.read_h5ad(data_path)
    adata.var_names_make_unique()

    if "spatial" not in adata.obsm:
        if "x" in adata.obs.columns and "y" in adata.obs.columns:
            adata.obsm["spatial"] = adata.obs[["x", "y"]].to_numpy(dtype=float)
        else:
            raise ValueError("No spatial coordinates found.")

    label_col = force_label_col if force_label_col is not None else "cluster"
    adata.obs["ground_truth"] = adata.obs[label_col].copy()
    adata_eval = adata[~pd.isnull(adata.obs["ground_truth"])].copy()
    n_clusters = adata_eval.obs["ground_truth"].nunique()

    print("✅ loaded:", data_path)
    print("✅ shape:", adata.shape)
    print("✅ label_col:", label_col)
    print("✅ n_clusters:", n_clusters)
    print("✅ GT counts:")
    print(adata_eval.obs["ground_truth"].value_counts().head(20))

    return adata, label_col, n_clusters


def to_counts_adata(adata):
    adata = adata.copy()

    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
        return adata

    if adata.raw is not None:
        raw_adata = adata.raw.to_adata()
        raw_adata.obs = adata.obs.copy()
        if "spatial" in adata.obsm:
            raw_adata.obsm["spatial"] = adata.obsm["spatial"].copy()
        return raw_adata

    return adata


def compute_ari_nmi(adata, pred_col="domain", gt_col="ground_truth"):
    adata_eval = adata[~pd.isnull(adata.obs[gt_col])].copy()
    ari = metrics.adjusted_rand_score(adata_eval.obs[pred_col], adata_eval.obs[gt_col])
    nmi = normalized_mutual_info_score(adata_eval.obs[pred_col], adata_eval.obs[gt_col])
    return ari, nmi, adata_eval.n_obs