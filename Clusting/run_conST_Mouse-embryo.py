import os
import sys
import random
import warnings
from types import SimpleNamespace

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import scipy.sparse as sp
from scipy import sparse
from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score

# =========================================================
# 0. 环境与路径
# =========================================================
os.environ["R_HOME"] = "/data2/liangyefeng/miniconda3/envs/graphst/lib/R"

PROJECT_ROOT = "/data2/liangyefeng/My_GraphST_Innovation"
CONST_ROOT = os.path.join(PROJECT_ROOT, "src", "conST")
DATA_DIR = "/data2/liangyefeng/My_GraphST_Innovation/data/Mouse-embryo"

RESULT_CSV = os.path.join(DATA_DIR, "conST_E16.5_E2S1_S5_batched_results.csv")

DATA_PATHS = [
    os.path.join(DATA_DIR, "E16.5_E2S1.MOSTA.h5ad"),
    os.path.join(DATA_DIR, "E16.5_E2S2.MOSTA.h5ad"),
    os.path.join(DATA_DIR, "E16.5_E2S3.MOSTA.h5ad"),
    os.path.join(DATA_DIR, "E16.5_E2S4.MOSTA.h5ad"),
    os.path.join(DATA_DIR, "E16.5_E2S11.MOSTA.h5ad"),
]

FORCE_LABEL_COL = "annotation"
FORCE_N_CLUSTERS = None

SPATIAL_BIN_SIZE = 3
SEED = 41

if CONST_ROOT not in sys.path:
    sys.path.insert(0, CONST_ROOT)

from src.graph_func import graph_construction
from src.utils_func import adata_preprocess
from src.training import conST_training

# =========================================================
# 1. 固定随机种子
# =========================================================
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

seed_torch(SEED)

# =========================================================
# 2. conST 参数
# =========================================================
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

params = SimpleNamespace(
    # graph
    k=20,
    knn_distanceType="euclidean",

    # training
    epochs=200,
    cell_feat_dim=100,
    feat_hidden1=100,
    feat_hidden2=20,
    gcn_hidden1=32,
    gcn_hidden2=8,
    p_drop=0.2,
    use_img=False,
    img_w=0.1,
    use_pretrained=False,
    using_mask=False,
    feat_w=10,
    gcn_w=0.1,
    dec_kl_w=10,
    gcn_lr=0.01,
    gcn_decay=0.01,
    dec_cluster_n=10,
    dec_interval=20,
    dec_tol=0.00,

    # contrastive
    seed=SEED,
    beta=100,
    cont_l2l=0.3,
    cont_l2c=0.1,
    cont_l2g=0.1,
    edge_drop_p1=0.1,
    edge_drop_p2=0.1,
    node_drop_p1=0.2,
    node_drop_p2=0.3,

    # runtime
    device=device,
    cell_num=None,
    save_path=None,
)

# =========================================================
# 3. mclust 聚类
# =========================================================
def mclust_R(adata, num_cluster, used_obsm="X_emb", key_added="domain", modelNames="EEE", random_seed=41):
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

# =========================================================
# 4. 工具函数
# =========================================================
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
    if "spatial" not in adata.obsm:
        raise KeyError("adata.obsm['spatial'] not found.")

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

    if label_col in obs_df.columns:
        label_bin = (
            obs_df.groupby("bin_id")[label_col]
            .apply(majority_vote)
            .reindex(unique_groups)
        )
    else:
        label_bin = pd.Series([np.nan] * n_groups, index=unique_groups)

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

# =========================================================
# 5. 主循环
# =========================================================
results = []

for data_path in DATA_PATHS:
    sample_name = os.path.basename(data_path).replace(".MOSTA.h5ad", "")
    print("\n" + "=" * 100)
    print(f"START SAMPLE: {sample_name}")
    print("=" * 100)

    try:
        seed_torch(SEED)

        adata = sc.read_h5ad(data_path)
        adata.obs_names_make_unique()
        adata.var_names_make_unique()

        adata, spatial_source = ensure_spatial(adata)

        label_col = FORCE_LABEL_COL if FORCE_LABEL_COL is not None else detect_label_col(adata.obs)
        if label_col is None:
            raise ValueError(
                "No label column detected.\n"
                f"Available obs columns: {list(adata.obs.columns)}"
            )

        adata.obs["ground_truth"] = adata.obs[label_col].copy()
        adata, x_source = to_counts_adata(adata)

        print(f"[{sample_name}] before binning: n_obs={adata.n_obs}, n_vars={adata.n_vars}")
        print(f"[{sample_name}] expression source={x_source}")
        print(f"[{sample_name}] spatial source={spatial_source}")
        print(f"[{sample_name}] label_col={label_col}")

        adata = spatial_bin_adata(
            adata,
            label_col="ground_truth",
            bin_size=SPATIAL_BIN_SIZE
        )

        adata.obs_names_make_unique()
        adata.var_names_make_unique()

        eval_mask = ~pd.isnull(adata.obs["ground_truth"])
        n_eval = int(eval_mask.sum())

        if n_eval == 0:
            if FORCE_N_CLUSTERS is None:
                raise ValueError("No valid ground_truth after binning.")
            n_clusters = int(FORCE_N_CLUSTERS)
        else:
            n_clusters = int(adata.obs.loc[eval_mask, "ground_truth"].nunique())

        if n_clusters < 2:
            raise ValueError(f"Invalid n_clusters={n_clusters} after binning.")

        params.cell_num = adata.n_obs
        params.dec_cluster_n = int(n_clusters)

        print(f"[{sample_name}] after binning: n_obs={adata.n_obs}, n_vars={adata.n_vars}")
        print(f"[{sample_name}] n_clusters={n_clusters}")
        print(f"[{sample_name}] n_eval={n_eval}")
        print(f"[{sample_name}] ground_truth counts:")
        print(adata.obs["ground_truth"].value_counts(dropna=False))

        pca_n_comps = min(
            params.cell_feat_dim,
            max(2, min(adata.n_obs - 1, adata.n_vars - 1))
        )
        print(f"[{sample_name}] pca_n_comps={pca_n_comps}")

        adata_X = adata_preprocess(
            adata,
            min_cells=5,
            pca_n_comps=pca_n_comps
        )

        graph_dict = graph_construction(
            adata.obsm["spatial"],
            adata.shape[0],
            params
        )

        seed_torch(SEED)
        conST_net = conST_training(adata_X, graph_dict, params, int(n_clusters))
        conST_net.pretraining()
        conST_net.major_training()

        conST_embedding = conST_net.get_embedding()
        if torch.is_tensor(conST_embedding):
            conST_embedding = conST_embedding.detach().cpu().numpy()
        else:
            conST_embedding = np.asarray(conST_embedding)

        if conST_embedding.shape[0] != adata.n_obs:
            raise ValueError(
                f"Embedding rows ({conST_embedding.shape[0]}) != adata.n_obs ({adata.n_obs})"
            )

        adata.obsm["X_emb"] = conST_embedding.copy()

        adata = mclust_R(
            adata,
            num_cluster=int(n_clusters),
            used_obsm="X_emb",
            key_added="domain",
            random_seed=SEED
        )

        ari, nmi, n_eval = compute_ari_nmi(
            adata,
            pred_col="domain",
            gt_col="ground_truth"
        )

        print(f"[{sample_name}] conST | ARI={ari:.6f} | NMI={nmi:.6f}")

        results.append({
            "Sample": sample_name,
            "Method": "conST",
            "Seed": SEED,
            "Bin_Size": SPATIAL_BIN_SIZE,
            "Label_Col": label_col,
            "N_Clusters": int(n_clusters),
            "N_Obs": int(adata.n_obs),
            "N_Obs_Eval": int(n_eval),
            "ARI": float(ari),
            "NMI": float(nmi),
            "Status": "OK",
            "Error": "",
        })

    except Exception as e:
        print(f"[{sample_name}] FAILED: {e}")
        results.append({
            "Sample": sample_name,
            "Method": "conST",
            "Seed": SEED,
            "Bin_Size": SPATIAL_BIN_SIZE,
            "Label_Col": FORCE_LABEL_COL if FORCE_LABEL_COL is not None else "",
            "N_Clusters": np.nan,
            "N_Obs": np.nan,
            "N_Obs_Eval": np.nan,
            "ARI": np.nan,
            "NMI": np.nan,
            "Status": "FAILED",
            "Error": str(e),
        })

# =========================================================
# 6. 保存结果
# =========================================================
results_df = pd.DataFrame(results)
results_df.to_csv(RESULT_CSV, index=False)

print("\n" + "=" * 100)
print("All done.")
print(results_df)
print(f"Saved: {RESULT_CSV}")
print("=" * 100)