import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import torch
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


def standardize_ma_obs(adata):
    adata = adata.copy()

    if list(adata.obs.columns) == ["x1", "x2", "x3", "x4", "x5"]:
        adata.obs = adata.obs.rename(columns={
            "x1": "in_tissue",
            "x2": "array_row",
            "x3": "array_col",
            "x4": "xcoord",
            "x5": "ycoord",
        })

    return adata


def ensure_spatial_from_obs(adata):
    adata = adata.copy()
    if "xcoord" not in adata.obs.columns or "ycoord" not in adata.obs.columns:
        raise ValueError("Expected xcoord/ycoord in adata.obs after standardization.")
    adata.obsm["spatial"] = adata.obs[["xcoord", "ycoord"]].to_numpy(dtype=float)
    return adata


def to_counts_adata(adata):
    adata = adata.copy()

    for key in ["counts", "raw_counts", "count", "Count", "Counts"]:
        if key in adata.layers:
            adata.X = adata.layers[key].copy()
            return adata

    if adata.raw is not None:
        try:
            raw_adata = adata.raw.to_adata()
            raw_adata.obs = adata.obs.copy()
            if "spatial" in adata.obsm:
                raw_adata.obsm["spatial"] = adata.obsm["spatial"].copy()
            return raw_adata
        except Exception:
            pass

    return adata


def load_ma_ground_truth(dataset_dir, obs_names):
    meta_path = os.path.join(dataset_dir, "metadata.tsv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.tsv not found: {meta_path}")

    df = pd.read_csv(meta_path, sep="\t", index_col=0)

    if "ground_truth" not in df.columns:
        raise ValueError("metadata.tsv does not contain ground_truth column.")

    gt = df.reindex(obs_names)["ground_truth"]
    return gt, meta_path, "ground_truth"


def load_ma_dataset(
    dataset_dir="/data2/liangyefeng/My_GraphST_Innovation/data/Mouse-Brain-Section-2-Sagittal-Anterior-and-Posterior/MA",
    h5ad_filename="MA1.h5ad",
):
    h5ad_path = os.path.join(dataset_dir, h5ad_filename)
    adata = sc.read_h5ad(h5ad_path)
    adata.var_names_make_unique()

    adata = standardize_ma_obs(adata)
    adata = ensure_spatial_from_obs(adata)

    gt, gt_path, label_col = load_ma_ground_truth(dataset_dir, adata.obs_names)
    adata.obs["ground_truth"] = gt

    adata_eval = adata[~pd.isnull(adata.obs["ground_truth"])].copy()
    n_clusters = adata_eval.obs["ground_truth"].nunique()

    print("✅ loaded:", h5ad_path)
    print("✅ shape:", adata.shape)
    print("✅ gt source:", gt_path)
    print("✅ label_col:", label_col)
    print("✅ n_clusters:", n_clusters)
    print("✅ ground_truth counts:")
    print(adata_eval.obs["ground_truth"].value_counts().head(20))
    print("✅ obsm keys:", list(adata.obsm.keys()))
    print("✅ obs cols:", list(adata.obs.columns))
    print("✅ spatial shape:", adata.obsm["spatial"].shape)

    return adata, label_col, n_clusters


def compute_ari_nmi(adata, pred_col="domain", gt_col="ground_truth"):
    adata_eval = adata[~pd.isnull(adata.obs[gt_col])].copy()
    ari = metrics.adjusted_rand_score(adata_eval.obs[pred_col], adata_eval.obs[gt_col])
    nmi = normalized_mutual_info_score(adata_eval.obs[pred_col], adata_eval.obs[gt_col])
    return ari, nmi, adata_eval.n_obs