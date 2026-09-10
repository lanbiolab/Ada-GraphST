import os
import sys
import re
import random
import warnings
from types import SimpleNamespace

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
import scipy.sparse as sp
from scipy import sparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture

# =========================================================
# 0. 环境与路径
# =========================================================
os.environ["R_HOME"] = "/data2/liangyefeng/miniconda3/envs/graphst/lib/R"

PROJECT_ROOT = "/data2/liangyefeng/My_GraphST_Innovation"
CONST_ROOT = os.path.join(PROJECT_ROOT, "src", "conST")

if CONST_ROOT not in sys.path:
    sys.path.insert(0, CONST_ROOT)

from src.graph_func import graph_construction
from src.utils_func import adata_preprocess
from src.training import conST_training

OUT_DIR = "/data2/liangyefeng/My_GraphST_Innovation/data/MERFISH/conST_binned_raw_results"
os.makedirs(OUT_DIR, exist_ok=True)

RESULT_CSV = os.path.join(OUT_DIR, "conST_MERFISH_binned_raw_results.csv")

DATA_PATHS = [
    "/data2/liangyefeng/My_GraphST_Innovation/data/MERFISH/fbe9c55e-518b-4bb7-9e1d-00322be93695.h5ad",
    "/data2/liangyefeng/My_GraphST_Innovation/data/MERFISH/bf8cb800-86b2-4ac0-b8a7-cc89fedaec89.h5ad",
]

DATASET_ALIAS = {
    "fbe9c55e-518b-4bb7-9e1d-00322be93695": "MERFISH_A",
    "bf8cb800-86b2-4ac0-b8a7-cc89fedaec89": "MERFISH_B",
}

FORCE_LABEL_COL = "cell_type"
FORCE_SLICE_COL = "slice"

FORCE_SPATIAL_KEY = None
FORCE_X_COL = None
FORCE_Y_COL = None

SEED = 41

# ===== Binning params =====
BIN_SIZE = 100.0
MIN_CELLS_PER_BIN = 3

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

INVALID_LABELS = {
    "", "na", "n/a", "nan", "none", "null",
    "unknown", "unassigned", "unlabeled", "unlabelled", "undefined"
}

MIN_OBS_PER_SECTION = 50


# =========================================================
# 3. 聚类：mclust 优先，失败回退 GMM
# =========================================================
def sanitize_embedding(emb, name="X_emb"):
    emb = np.asarray(emb, dtype=np.float64)
    if emb.ndim != 2:
        raise ValueError(f"{name} must be 2D, but got shape {emb.shape}")

    n_bad = np.sum(~np.isfinite(emb))
    if n_bad > 0:
        raise ValueError(f"{name} contains NaN/Inf, bad_count={int(n_bad)}")
    return emb


def mclust_or_gmm(adata, num_cluster, used_obsm="X_emb", key_added="domain", random_seed=41):
    x = sanitize_embedding(adata.obsm[used_obsm], used_obsm)

    try:
        import rpy2.robjects as robjects
        from rpy2.robjects import r
        from rpy2.robjects import numpy2ri

        numpy2ri.activate()
        robjects.globalenv["x_mat"] = x
        robjects.globalenv["n_cluster"] = int(num_cluster)
        robjects.globalenv["seed"] = int(random_seed)

        r(
            """
            suppressMessages(library(mclust))
            set.seed(seed)
            x_mat <- as.matrix(x_mat)
            dimnames(x_mat) <- NULL

            res <- tryCatch(
              Mclust(x_mat, G=n_cluster, modelNames="EEE"),
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

        cls = np.array(r["cls"], dtype=object)
        cls = pd.to_numeric(pd.Series(cls, index=adata.obs_names), errors="coerce")

        na_count = int(cls.isna().sum())
        if na_count > 0:
            raise ValueError(f"mclust returned {na_count} invalid labels")

        adata.obs[key_added] = cls.astype(int).astype(str).astype("category")
        return adata, "mclust_R"

    except Exception as e:
        print(f"[WARN] mclust failed, fallback to GaussianMixture: {e}")

        gm = GaussianMixture(
            n_components=int(num_cluster),
            covariance_type="diag",
            random_state=random_seed,
            reg_covar=1e-5,
            n_init=5,
            max_iter=200
        )
        pred = gm.fit_predict(x)
        adata.obs[key_added] = pd.Categorical(pred.astype(str))
        return adata, "GaussianMixture"


# =========================================================
# 4. 工具函数
# =========================================================
def safe_name(x):
    x = str(x)
    x = re.sub(r"[^\w\-.]+", "_", x)
    return x[:200]


def print_header(msg):
    print("\n" + "=" * 120)
    print(msg)
    print("=" * 120)


def ensure_spatial(adata, force_spatial_key=None, force_x_col=None, force_y_col=None):
    if force_spatial_key is not None:
        if force_spatial_key not in adata.obsm_keys():
            raise ValueError(f"FORCE_SPATIAL_KEY='{force_spatial_key}' 不在 adata.obsm 中")
        arr = np.asarray(adata.obsm[force_spatial_key])
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"adata.obsm['{force_spatial_key}'] 不是合法二维坐标")
        adata.obsm["spatial"] = arr[:, :2].astype(np.float64)
        return adata, "obsm:" + force_spatial_key

    for key in ["X_spatial_coords", "spatial_coords", "spatial", "X_spatial", "spatial_stereo"]:
        if key in adata.obsm:
            arr = np.asarray(adata.obsm[key])
            if arr.ndim == 2 and arr.shape[1] >= 2:
                adata.obsm["spatial"] = arr[:, :2].astype(np.float64)
                return adata, "obsm:" + key

    obs_cols = list(adata.obs.columns)
    lower_to_orig = {c.lower(): c for c in obs_cols}
    candidate_pairs = [
        ("center_x", "center_y"),
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


def clean_label_series(series):
    s = series.astype("string").copy()
    s = s.str.strip()
    valid = ~s.isna()
    valid &= ~s.str.lower().isin(INVALID_LABELS)
    return s, np.asarray(valid, dtype=bool)


def check_matrix_nonnegative(X, tag):
    if sparse.issparse(X):
        data = X.data
        if np.isnan(data).sum() > 0 or np.isinf(data).sum() > 0:
            raise ValueError(f"{tag}: 存在 NaN/Inf")
        if data.size > 0 and np.min(data) < 0:
            raise ValueError(f"{tag}: 存在负值，不能作为 counts 使用")
    else:
        arr = np.asarray(X, dtype=np.float64)
        if np.isnan(arr).sum() > 0 or np.isinf(arr).sum() > 0:
            raise ValueError(f"{tag}: 存在 NaN/Inf")
        if np.min(arr) < 0:
            raise ValueError(f"{tag}: 存在负值，不能作为 counts 使用")


def summarize_matrix_quick(X, tag):
    if sparse.issparse(X):
        data = X.data
        if data.size == 0:
            print(f"[{tag}] sparse empty")
            return
        print(
            f"[{tag}] sparse shape={X.shape}, nnz={X.nnz}, "
            f"min={float(np.min(data))}, max={float(np.max(data))}, "
            f"nan={int(np.isnan(data).sum())}, inf={int(np.isinf(data).sum())}, neg={int((data < 0).sum())}"
        )
    else:
        arr = np.asarray(X, dtype=np.float64)
        print(
            f"[{tag}] dense shape={arr.shape}, "
            f"min={float(np.nanmin(arr))}, max={float(np.nanmax(arr))}, "
            f"nan={int(np.isnan(arr).sum())}, inf={int(np.isinf(arr).sum())}, neg={int((arr < 0).sum())}"
        )


def majority_vote(series):
    vals = pd.Series(series).dropna().astype(str)
    if len(vals) == 0:
        return np.nan
    return vals.value_counts().idxmax()


def build_binned_adata_from_raw(adata, label_col="ground_truth", bin_size=100.0, min_cells_per_bin=3):
    if "spatial" not in adata.obsm:
        raise KeyError("adata.obsm['spatial'] not found.")

    if adata.raw is None:
        raise ValueError("adata.raw is None，当前脚本要求使用 adata.raw.X 作为原始 counts")

    X_raw = adata.raw.X
    var_raw = adata.raw.var.copy()

    if not sparse.issparse(X_raw):
        X_raw = sp.csr_matrix(np.asarray(X_raw))
    else:
        X_raw = X_raw.tocsr()

    check_matrix_nonnegative(X_raw, "raw_counts")

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

    X_bin = (G @ X_raw).tocsr()

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

    valid_label_count = (
        obs_df.groupby("bin_id")[label_col]
        .apply(lambda s: pd.Series(s).dropna().astype(str).shape[0])
        .reindex(unique_groups)
        .fillna(0)
        .astype(int)
    )

    majority_count = (
        obs_df.groupby("bin_id")[label_col]
        .apply(lambda s: pd.Series(s).dropna().astype(str).value_counts().iloc[0] if pd.Series(s).dropna().shape[0] > 0 else 0)
        .reindex(unique_groups)
        .fillna(0)
        .astype(int)
    )

    purity = majority_count.values / np.maximum(valid_label_count.values, 1)

    keep_mask = (n_spots_bin.values >= min_cells_per_bin) & (valid_label_count.values > 0)
    keep_idx = np.where(keep_mask)[0]

    if len(keep_idx) < 2:
        raise ValueError("binning 后保留下来的 bins 太少")

    X_bin_keep = X_bin[keep_idx].tocsr()
    spatial_keep = spatial_bin[keep_idx]

    adata_bin = sc.AnnData(X=X_bin_keep)
    adata_bin.obs_names = pd.Index(np.asarray(unique_groups)[keep_idx]).astype(str)
    adata_bin.var_names = pd.Index(var_raw.index).astype(str)
    adata_bin.var_names_make_unique()
    adata_bin.obs_names_make_unique()

    adata_bin.obsm["spatial"] = spatial_keep.astype(np.float64)
    adata_bin.obs["ground_truth"] = label_bin.values[keep_idx]
    adata_bin.obs["n_spots"] = n_spots_bin.values[keep_idx]
    adata_bin.obs["valid_label_count"] = valid_label_count.values[keep_idx]
    adata_bin.obs["majority_count"] = majority_count.values[keep_idx]
    adata_bin.obs["purity"] = purity[keep_idx]

    old_to_new = np.full(n_groups, -1, dtype=int)
    old_to_new[keep_idx] = np.arange(len(keep_idx), dtype=int)
    cell_to_bin_new = old_to_new[group_codes]

    return adata_bin, cell_to_bin_new


def compute_ari_nmi_from_arrays(pred, gt):
    pred = pd.Series(pred).astype(str)
    gt = pd.Series(gt).astype(str)

    valid = (~pred.isna()) & (~gt.isna())
    pred = pred[valid]
    gt = gt[valid]

    if len(pred) == 0:
        raise ValueError("没有可用于评估的样本")
    if gt.nunique() < 2:
        raise ValueError("GT 类别数 < 2，无法计算 ARI/NMI")

    ari = adjusted_rand_score(gt, pred)
    nmi = normalized_mutual_info_score(gt, pred)
    return float(ari), float(nmi), int(len(pred))


def get_sections(adata, slice_col=None):
    if slice_col is None:
        return [("all", np.arange(adata.n_obs, dtype=int))]

    values = adata.obs[slice_col].astype(str).to_numpy()
    sections = []

    for v in pd.unique(values):
        if str(v).lower() in {"nan", "none", "na", ""}:
            continue
        idx = np.where(values == str(v))[0]
        if len(idx) > 0:
            sections.append((str(v), idx))

    if len(sections) == 0:
        return [("all", np.arange(adata.n_obs, dtype=int))]

    return sections


# =========================================================
# 5. 主循环
# =========================================================
results = []

for data_path in DATA_PATHS:
    dataset_id = os.path.basename(data_path).replace(".h5ad", "")
    dataset_alias = DATASET_ALIAS.get(dataset_id, dataset_id[:8])

    print_header(f"LOAD DATASET: {dataset_alias} ({dataset_id})")

    try:
        seed_torch(SEED)

        adata_all = sc.read_h5ad(data_path)
        adata_all.obs_names_make_unique()
        adata_all.var_names_make_unique()

        adata_all, spatial_source = ensure_spatial(
            adata_all,
            force_spatial_key=FORCE_SPATIAL_KEY,
            force_x_col=FORCE_X_COL,
            force_y_col=FORCE_Y_COL
        )

        if FORCE_LABEL_COL not in adata_all.obs.columns:
            raise ValueError(f"{FORCE_LABEL_COL} 不在 obs.columns 中")
        if FORCE_SLICE_COL not in adata_all.obs.columns:
            raise ValueError(f"{FORCE_SLICE_COL} 不在 obs.columns 中")

        sections = get_sections(adata_all, FORCE_SLICE_COL)
        print(f"[{dataset_alias}] total sections = {len(sections)}")

        for section_name, idx in sections:
            sample_name = f"{dataset_alias}_slice{section_name}"

            print("\n" + "=" * 100)
            print(f"START SAMPLE: {sample_name}")
            print("=" * 100)

            try:
                seed_torch(SEED)

                adata_sec = adata_all[idx].copy()
                adata_sec.obs_names_make_unique()
                adata_sec.var_names_make_unique()

                if adata_sec.n_obs < MIN_OBS_PER_SECTION:
                    raise ValueError(f"n_obs={adata_sec.n_obs} < MIN_OBS_PER_SECTION={MIN_OBS_PER_SECTION}")

                adata_sec.obs["ground_truth"] = adata_sec.obs[FORCE_LABEL_COL].copy()

                print(f"[{sample_name}] before binning: n_obs={adata_sec.n_obs}, n_vars={adata_sec.n_vars}")
                print(f"[{sample_name}] spatial source={spatial_source}")
                print(f"[{sample_name}] label_col={FORCE_LABEL_COL}")
                summarize_matrix_quick(adata_sec.X, f"{sample_name} section_X_processed")
                summarize_matrix_quick(adata_sec.raw.X, f"{sample_name} section_raw_X")

                adata_bin, cell_to_bin_new = build_binned_adata_from_raw(
                    adata_sec,
                    label_col="ground_truth",
                    bin_size=BIN_SIZE,
                    min_cells_per_bin=MIN_CELLS_PER_BIN
                )

                adata_bin.obs_names_make_unique()
                adata_bin.var_names_make_unique()

                eval_mask = ~pd.isnull(adata_bin.obs["ground_truth"])
                n_bin_eval = int(eval_mask.sum())

                if n_bin_eval == 0:
                    raise ValueError("No valid ground_truth after binning.")

                n_clusters = int(adata_bin.obs.loc[eval_mask, "ground_truth"].nunique())
                if n_clusters < 2:
                    raise ValueError(f"Invalid n_clusters={n_clusters} after binning.")

                params.cell_num = adata_bin.n_obs
                params.dec_cluster_n = int(n_clusters)

                print(f"[{sample_name}] after binning: n_obs={adata_bin.n_obs}, n_vars={adata_bin.n_vars}")
                print(f"[{sample_name}] n_clusters={n_clusters}")
                print(f"[{sample_name}] n_bin_eval={n_bin_eval}")
                print(f"[{sample_name}] purity mean={adata_bin.obs['purity'].astype(float).mean():.4f}")
                print(f"[{sample_name}] ground_truth counts:")
                print(adata_bin.obs["ground_truth"].value_counts(dropna=False))
                summarize_matrix_quick(adata_bin.X, f"{sample_name} binned_raw_counts")

                pca_n_comps = min(
                    params.cell_feat_dim,
                    max(2, min(adata_bin.n_obs - 1, adata_bin.n_vars - 1))
                )
                print(f"[{sample_name}] pca_n_comps={pca_n_comps}")

                adata_X = adata_preprocess(
                    adata_bin,
                    min_cells=5,
                    pca_n_comps=pca_n_comps
                )

                graph_dict = graph_construction(
                    adata_bin.obsm["spatial"],
                    adata_bin.shape[0],
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

                if conST_embedding.shape[0] != adata_bin.n_obs:
                    raise ValueError(
                        f"Embedding rows ({conST_embedding.shape[0]}) != adata_bin.n_obs ({adata_bin.n_obs})"
                    )

                adata_bin.obsm["X_emb"] = sanitize_embedding(conST_embedding, "X_emb")

                adata_bin, cluster_method = mclust_or_gmm(
                    adata_bin,
                    num_cluster=int(n_clusters),
                    used_obsm="X_emb",
                    key_added="domain",
                    random_seed=SEED
                )

                # ===== bin-level metrics =====
                bin_ari, bin_nmi, n_bin_eval = compute_ari_nmi_from_arrays(
                    pred=adata_bin.obs["domain"].astype(str).values,
                    gt=adata_bin.obs["ground_truth"].astype(str).values
                )

                # ===== projected cell-level metrics =====
                gt_cells_raw, valid_cells_mask = clean_label_series(adata_sec.obs[FORCE_LABEL_COL])
                valid_cell_eval = (cell_to_bin_new >= 0) & valid_cells_mask
                if valid_cell_eval.sum() == 0:
                    raise ValueError("没有可用于 projected cell-level 评估的细胞")

                bin_domain = adata_bin.obs["domain"].astype(str).to_numpy()
                cell_pred = bin_domain[cell_to_bin_new[valid_cell_eval]]
                cell_gt = gt_cells_raw[valid_cell_eval].astype(str).to_numpy()

                cell_ari, cell_nmi, n_cell_eval = compute_ari_nmi_from_arrays(
                    pred=cell_pred,
                    gt=cell_gt
                )

                print(
                    f"[{sample_name}] conST-binned-raw | "
                    f"Bin-ARI={bin_ari:.6f} | Bin-NMI={bin_nmi:.6f} | "
                    f"CellProj-ARI={cell_ari:.6f} | CellProj-NMI={cell_nmi:.6f}"
                )

                out_h5ad = os.path.join(OUT_DIR, f"{sample_name}__conST_binned_raw.h5ad")
                adata_bin.write_h5ad(out_h5ad)

                results.append({
                    "Sample": sample_name,
                    "Dataset_ID": dataset_id,
                    "Dataset_Alias": dataset_alias,
                    "Section": section_name,
                    "Method": "conST_binned_raw",
                    "Seed": SEED,
                    "Bin_Size": BIN_SIZE,
                    "Min_Cells_Per_Bin": MIN_CELLS_PER_BIN,
                    "Label_Col": FORCE_LABEL_COL,
                    "Slice_Col": FORCE_SLICE_COL,
                    "Spatial_Source": spatial_source,
                    "N_Clusters": int(n_clusters),
                    "N_Bins": int(adata_bin.n_obs),
                    "N_Bin_Eval": int(n_bin_eval),
                    "N_Cell_Eval": int(n_cell_eval),
                    "Bin_ARI": float(bin_ari),
                    "Bin_NMI": float(bin_nmi),
                    "CellProj_ARI": float(cell_ari),
                    "CellProj_NMI": float(cell_nmi),
                    "Cluster_Method": cluster_method,
                    "Output_h5ad": out_h5ad,
                    "Status": "OK",
                    "Error": "",
                })

            except Exception as e:
                print(f"[{sample_name}] FAILED: {e}")
                results.append({
                    "Sample": sample_name,
                    "Dataset_ID": dataset_id,
                    "Dataset_Alias": dataset_alias,
                    "Section": section_name,
                    "Method": "conST_binned_raw",
                    "Seed": SEED,
                    "Bin_Size": BIN_SIZE,
                    "Min_Cells_Per_Bin": MIN_CELLS_PER_BIN,
                    "Label_Col": FORCE_LABEL_COL,
                    "Slice_Col": FORCE_SLICE_COL,
                    "Spatial_Source": spatial_source,
                    "N_Clusters": np.nan,
                    "N_Bins": np.nan,
                    "N_Bin_Eval": np.nan,
                    "N_Cell_Eval": np.nan,
                    "Bin_ARI": np.nan,
                    "Bin_NMI": np.nan,
                    "CellProj_ARI": np.nan,
                    "CellProj_NMI": np.nan,
                    "Cluster_Method": "",
                    "Output_h5ad": "",
                    "Status": "FAILED",
                    "Error": str(e),
                })

    except Exception as e:
        print(f"[{dataset_alias}] FAILED: {e}")
        results.append({
            "Sample": dataset_alias,
            "Dataset_ID": dataset_id,
            "Dataset_Alias": dataset_alias,
            "Section": "",
            "Method": "conST_binned_raw",
            "Seed": SEED,
            "Bin_Size": BIN_SIZE,
            "Min_Cells_Per_Bin": MIN_CELLS_PER_BIN,
            "Label_Col": FORCE_LABEL_COL,
            "Slice_Col": FORCE_SLICE_COL,
            "Spatial_Source": "",
            "N_Clusters": np.nan,
            "N_Bins": np.nan,
            "N_Bin_Eval": np.nan,
            "N_Cell_Eval": np.nan,
            "Bin_ARI": np.nan,
            "Bin_NMI": np.nan,
            "CellProj_ARI": np.nan,
            "CellProj_NMI": np.nan,
            "Cluster_Method": "",
            "Output_h5ad": "",
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