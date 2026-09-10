import os
import random
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy import sparse
import torch
from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score

SEED = 41
DATA_DIR = "/data2/liangyefeng/My_GraphST_Innovation/data/Mouse-embryo"

DATA_PATHS = [
    os.path.join(DATA_DIR, "E16.5_E2S1.MOSTA.h5ad"),
    os.path.join(DATA_DIR, "E16.5_E2S2.MOSTA.h5ad"),
    os.path.join(DATA_DIR, "E16.5_E2S3.MOSTA.h5ad"),
    os.path.join(DATA_DIR, "E16.5_E2S4.MOSTA.h5ad"),
]

FORCE_LABEL_COL = "annotation"
SPATIAL_BIN_SIZE = 3


def seed_everything(seed=41, use_tf=False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if use_tf:
        import tensorflow.compat.v1 as tf
        tf.set_random_seed(seed)


def detect_label_col(obs_df):
    preferred = [
        "ground_truth", "ground truth",
        "annotation", "annotations",
        "label", "labels",
        "cluster", "clusters",
        "cell_type", "celltype",
        "region", "class", "domain", "type"
    ]
    cols = list(obs_df.columns)
    lower_to_orig = {c.lower(): c for c in cols}

    for k in preferred:
        if k in lower_to_orig:
            return lower_to_orig[k]

    for k in preferred:
        for c in cols:
            if k in c.lower():
                return c
    return None


def ensure_spatial(adata):
    for key in ["spatial", "X_spatial", "spatial_stereo"]:
        if key in adata.obsm:
            arr = np.asarray(adata.obsm[key])
            if arr.ndim == 2 and arr.shape[1] >= 2:
                adata.obsm["spatial"] = arr[:, :2].astype(np.float64)
                return adata, key

    obs_cols = list(adata.obs.columns)
    lower_to_orig = {c.lower(): c for c in obs_cols}
    candidate_pairs = [
        ("x", "y"),
        ("coord_x", "coord_y"),
        ("imagecol", "imagerow"),
        ("array_col", "array_row"),
        ("pixel_x", "pixel_y"),
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
        ("col", "row"),
    ]

    for xk, yk in candidate_pairs:
        if xk in lower_to_orig and yk in lower_to_orig:
            xcol = lower_to_orig[xk]
            ycol = lower_to_orig[yk]
            adata.obsm["spatial"] = adata.obs[[xcol, ycol]].values.astype(np.float64)
            return adata, f"obs[{xcol},{ycol}]"

    raise KeyError("Cannot find spatial coordinates.")


def to_counts_adata(adata):
    if "counts" in adata.layers:
        X = adata.layers["counts"]
        var_names = adata.var_names.copy()
        x_source = "layers['counts']"
    elif "raw" in adata.layers:
        X = adata.layers["raw"]
        var_names = adata.var_names.copy()
        x_source = "layers['raw']"
    elif adata.raw is not None:
        X = adata.raw.X
        var_names = adata.raw.var_names.copy()
        x_source = "adata.raw.X"
    else:
        X = adata.X
        var_names = adata.var_names.copy()
        x_source = "adata.X"

    if sparse.issparse(X):
        X_new = X.copy()
    else:
        X_new = np.asarray(X)

    new_adata = sc.AnnData(X=X_new)
    new_adata.obs = adata.obs.copy()
    new_adata.obs_names = adata.obs_names.copy()
    new_adata.var_names = pd.Index(var_names).astype(str)
    new_adata.var_names_make_unique()
    new_adata.obs_names_make_unique()

    for k, v in adata.obsm.items():
        new_adata.obsm[k] = v.copy() if hasattr(v, "copy") else v

    return new_adata, x_source


def majority_vote(series):
    vals = pd.Series(series).dropna().astype(str)
    if len(vals) == 0:
        return np.nan
    return vals.value_counts().idxmax()


def spatial_bin_adata(adata, label_col="ground_truth", bin_size=3):
    coords = np.asarray(adata.obsm["spatial"]).astype(np.float64)
    x = coords[:, 0]
    y = coords[:, 1]

    x0 = x.min()
    y0 = y.min()

    gx = np.floor((x - x0) / bin_size).astype(int)
    gy = np.floor((y - y0) / bin_size).astype(int)

    bin_ids = pd.Series([f"{i}_{j}" for i, j in zip(gx, gy)], index=adata.obs_names, name="bin_id")
    group_codes, unique_groups = pd.factorize(bin_ids.values, sort=True)
    n_groups = len(unique_groups)

    rows = group_codes
    cols = np.arange(adata.n_obs)
    data = np.ones(adata.n_obs, dtype=np.float32)
    G = sp.csr_matrix((data, (rows, cols)), shape=(n_groups, adata.n_obs))

    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(np.asarray(X))
    else:
        X = X.tocsr()

    X_bin = G @ X

    coord_df = pd.DataFrame({
        "bin_id": bin_ids.values,
        "x": x,
        "y": y,
    })
    spatial_bin = coord_df.groupby("bin_id")[["x", "y"]].mean().loc[unique_groups].values

    obs_df = adata.obs.copy()
    obs_df["bin_id"] = bin_ids.values

    label_bin = (
        obs_df.groupby("bin_id")[label_col]
        .apply(majority_vote)
        .reindex(unique_groups)
    )

    n_spots_bin = obs_df.groupby("bin_id").size().reindex(unique_groups)

    adata_bin = sc.AnnData(X=X_bin)
    adata_bin.obs_names = pd.Index(unique_groups).astype(str)
    adata_bin.var_names = adata.var_names.copy()
    adata_bin.var_names_make_unique()
    adata_bin.obs_names_make_unique()
    adata_bin.obsm["spatial"] = spatial_bin.astype(np.float64)
    adata_bin.obs["ground_truth"] = label_bin.values
    adata_bin.obs["n_spots"] = n_spots_bin.values

    return adata_bin


def load_binned_section(data_path, force_label_col="annotation", bin_size=3):
    adata = sc.read_h5ad(data_path)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    adata, spatial_source = ensure_spatial(adata)

    label_col = force_label_col if force_label_col is not None else detect_label_col(adata.obs)
    if label_col is None:
        raise ValueError(
            "No label column detected.\n"
            f"Available obs columns: {list(adata.obs.columns)}"
        )

    adata.obs["ground_truth"] = adata.obs[label_col].copy()
    adata, x_source = to_counts_adata(adata)
    adata = spatial_bin_adata(adata, label_col="ground_truth", bin_size=bin_size)

    eval_mask = ~pd.isnull(adata.obs["ground_truth"])
    n_eval = int(eval_mask.sum())
    if n_eval == 0:
        raise ValueError("No valid ground_truth after binning.")

    n_clusters = int(adata.obs.loc[eval_mask, "ground_truth"].nunique())
    if n_clusters < 2:
        raise ValueError(f"Invalid n_clusters={n_clusters} after binning.")

    sample_name = os.path.basename(data_path).replace(".MOSTA.h5ad", "")

    info = {
        "sample_name": sample_name,
        "label_col": label_col,
        "spatial_source": spatial_source,
        "expression_source": x_source,
        "n_clusters": n_clusters,
        "n_eval": n_eval,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
    }
    return adata, info


def compute_ari_nmi(adata, pred_col="domain", gt_col="ground_truth"):
    mask = ~pd.isnull(adata.obs[gt_col])
    n_eval = int(mask.sum())
    if n_eval == 0:
        return np.nan, np.nan, 0

    y_true = adata.obs.loc[mask, gt_col]
    y_pred = adata.obs.loc[mask, pred_col]

    ari = metrics.adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    return float(ari), float(nmi), n_eval


def mclust_R(adata, num_cluster, used_obsm="X_emb", key_added="domain", modelNames="EEE", random_seed=41):
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


def graphst_mclust_R(adata, num_cluster, modelNames="EEE", used_obsm="emb_pca", random_seed=41):
    import rpy2.robjects as robjects
    from rpy2.robjects import r
    from rpy2.robjects import numpy2ri

    x = np.asarray(adata.obsm[used_obsm], dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"{used_obsm} must be 2D, but got shape {x.shape}")

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

    mclust_res = np.array(r["cls"])
    adata.obs["mclust"] = pd.Categorical(pd.Series(mclust_res, index=adata.obs_names).astype(int))
    return adata


def patch_graphst_mclust():
    import GraphST.utils as graphst_utils
    graphst_utils.mclust_R = graphst_mclust_R