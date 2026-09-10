# -*- coding: utf-8 -*-

import os
import sys
import random
import json

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy import sparse
import torch
from sklearn import metrics
from sklearn.decomposition import PCA

# =========================================================
# 0. 环境配置
# =========================================================
os.environ["R_HOME"] = "/data2/liangyefeng/miniconda3/envs/graphst/lib/R"

project_root = "/data2/liangyefeng/My_GraphST_Innovation"
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from GraphST.GraphST import GraphST
import GraphST.utils as graphst_utils
from GraphST.utils import clustering


# =========================================================
# 1. runtime patch: 修复当前环境下 GraphST 的 mclust 调用
# =========================================================
def patched_mclust_R(
    adata,
    num_cluster,
    modelNames="EEE",
    used_obsm="emb_pca",
    random_seed=2020
):
    import numpy as np
    import rpy2.robjects as robjects
    from rpy2.robjects import r
    from rpy2.robjects import numpy2ri

    x = np.asarray(adata.obsm[used_obsm], dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"{used_obsm} must be 2D, but got shape {x.shape}")

    print(f"[patched_mclust_R] used_obsm={used_obsm}, shape={x.shape}, dtype={x.dtype}")

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
    adata.obs["mclust"] = mclust_res
    adata.obs["mclust"] = adata.obs["mclust"].astype(int)
    adata.obs["mclust"] = adata.obs["mclust"].astype("category")
    return adata


graphst_utils.mclust_R = patched_mclust_R


# =========================================================
# 2. 固定随机种子
# =========================================================
seed = 41

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)


# =========================================================
# 3. 数据集与输出配置
# =========================================================
data_root = "/data2/liangyefeng/My_GraphST_Innovation/data/Mouse-embryo"

dataset = "E16.5_E2S3"
data_path = os.path.join(data_root, f"{dataset}.MOSTA.h5ad")

output_dir = data_root
os.makedirs(output_dir, exist_ok=True)

FORCE_LABEL_COL = "annotation"
SPATIAL_BIN_SIZE = 3
radius = 50


# =========================================================
# 4. GraphST 固定公共参数
# =========================================================
common_params = {
    "device": device,
    "random_seed": seed,
    "epochs": 600,
    "dim_output": 64,
    "datatype": "Stereo",
    "warmup_epochs": 200,
    "update_interval": 20,
    "graph_update_rate": 0.3,
    "gamma": 2.5,
    "graph_reg_weight": 0.0,
    "use_learnable_proj": False,
}


# =========================================================
# 5. 本次只跑两个模型：baseline 与 Ada-GraphST
# =========================================================
candidates = [
    {
        **common_params,
        "name": "Baseline_Internal",
        "w_smooth": 0.0,
        "w_sharpen": 0.0,
    },
    {
        **common_params,
        "name": "AdaGraphST_smooth0p02_sharpen0p03",
        "w_smooth": 0.02,
        "w_sharpen": 0.03,
    },
]

print("Candidates:")
for cand in candidates:
    print(
        f"  {cand['name']}: "
        f"w_smooth={cand['w_smooth']}, "
        f"w_sharpen={cand['w_sharpen']}"
    )


# =========================================================
# 6. 工具函数
# =========================================================
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
    elif "raw" in adata.layers:
        X = adata.layers["raw"]
        var_names = adata.var_names.copy()
    elif adata.raw is not None:
        X = adata.raw.X
        var_names = adata.raw.var_names.copy()
    else:
        X = adata.X
        var_names = adata.var_names.copy()

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

    return new_adata


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

    bin_ids = pd.Series(
        [f"{i}_{j}" for i, j in zip(gx, gy)],
        index=adata.obs_names,
        name="bin_id"
    )

    group_codes, unique_groups = pd.factorize(bin_ids.values, sort=True)
    n_groups = len(unique_groups)

    rows = group_codes
    cols = np.arange(adata.n_obs)
    data = np.ones(adata.n_obs, dtype=np.float32)

    G = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(n_groups, adata.n_obs)
    )

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

    spatial_bin = (
        coord_df
        .groupby("bin_id")[["x", "y"]]
        .mean()
        .loc[unique_groups]
        .values
    )

    obs_df = adata.obs.copy()
    obs_df["bin_id"] = bin_ids.values

    label_bin = (
        obs_df
        .groupby("bin_id")[label_col]
        .apply(majority_vote)
        .reindex(unique_groups)
    )

    n_spots_bin = (
        obs_df
        .groupby("bin_id")
        .size()
        .reindex(unique_groups)
    )

    adata_bin = sc.AnnData(X=X_bin)
    adata_bin.obs_names = pd.Index(unique_groups).astype(str)
    adata_bin.var_names = adata.var_names.copy()
    adata_bin.var_names_make_unique()
    adata_bin.obs_names_make_unique()

    adata_bin.obsm["spatial"] = spatial_bin.astype(np.float64)
    adata_bin.obs["ground_truth"] = label_bin.values
    adata_bin.obs["n_spots"] = n_spots_bin.values

    return adata_bin


def load_mouse_embryo_section(data_path, force_label_col="annotation", bin_size=3):
    print(f"Loading data: {data_path}")

    adata = sc.read_h5ad(data_path)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    adata, spatial_source = ensure_spatial(adata)

    label_col = force_label_col if force_label_col is not None else detect_label_col(adata.obs)
    if label_col is None:
        raise ValueError(f"No label column detected. obs columns: {list(adata.obs.columns)}")

    if label_col not in adata.obs.columns:
        raise KeyError(
            f"Label column '{label_col}' not found. "
            f"Available obs columns: {list(adata.obs.columns)}"
        )

    adata.obs["ground_truth"] = adata.obs[label_col].copy()

    adata = to_counts_adata(adata)

    adata = spatial_bin_adata(
        adata,
        label_col="ground_truth",
        bin_size=bin_size
    )

    eval_mask = ~pd.isnull(adata.obs["ground_truth"])
    n_eval = int(eval_mask.sum())

    if n_eval == 0:
        raise ValueError("No valid ground_truth after binning.")

    n_clusters = int(adata.obs.loc[eval_mask, "ground_truth"].nunique())

    if n_clusters < 2:
        raise ValueError(f"Invalid n_clusters={n_clusters} after binning.")

    print(f"spatial source: {spatial_source}")
    print(f"label_col: {label_col}")
    print(f"shape after binning: {adata.shape}")
    print(f"n_clusters: {n_clusters}")
    print(f"n_eval: {n_eval}")
    print("GT counts:")
    print(adata.obs["ground_truth"].value_counts(dropna=False))

    return adata, label_col, n_clusters, n_eval


def sanitize_params_for_uns(candidate):
    """
    h5ad 的 uns 不能安全保存 torch.device 等对象。
    这里转成可写入 h5ad 的基础类型。
    """
    clean = {}

    for k, v in candidate.items():
        if isinstance(v, torch.device):
            clean[k] = str(v)
        elif isinstance(v, (np.integer, np.floating)):
            clean[k] = v.item()
        elif isinstance(v, (str, int, float, bool)) or v is None:
            clean[k] = v
        else:
            clean[k] = str(v)

    return clean


def prepare_adata_for_write(adata):
    """
    写 h5ad 前做轻量清理，减少 object 类型导致的写入问题。
    """
    adata = adata.copy()

    for col in adata.obs.columns:
        if adata.obs[col].dtype == "object":
            try:
                adata.obs[col] = adata.obs[col].astype("category")
            except Exception:
                adata.obs[col] = adata.obs[col].astype(str)

    adata.strings_to_categoricals()

    return adata


def run_one_candidate(
    adata_input,
    candidate,
    n_clusters,
    radius,
    seed=41
):
    cand_name = candidate["name"]

    print("\n" + "-" * 100)
    print(f"Running candidate: {cand_name}")
    print("-" * 100)

    adata = adata_input.copy()

    graphst_params = {
        k: v for k, v in candidate.items()
        if k != "name"
    }

    model = GraphST(adata, **graphst_params)
    adata = model.train()

    emb_key = None
    for k in ["emb", "GraphST", "X_emb"]:
        if k in adata.obsm:
            emb_key = k
            break

    if emb_key is None:
        raise KeyError(
            f"No embedding key found in adata.obsm. "
            f"Keys={list(adata.obsm.keys())}"
        )

    emb = np.asarray(adata.obsm[emb_key])

    n_pcs = min(20, emb.shape[0], emb.shape[1])
    if n_pcs < 2:
        raise ValueError(
            f"Cannot run PCA because n_pcs={n_pcs}, "
            f"embedding shape={emb.shape}"
        )

    adata.obsm["emb_pca"] = PCA(
        n_components=n_pcs,
        random_state=seed
    ).fit_transform(emb)

    clustering(
        adata,
        n_clusters,
        radius=radius,
        method="mclust",
        refinement=True
    )

    adata_eval = adata[~pd.isnull(adata.obs["ground_truth"])].copy()

    ari = metrics.adjusted_rand_score(
        adata_eval.obs["domain"],
        adata_eval.obs["ground_truth"]
    )

    nmi = metrics.normalized_mutual_info_score(
        adata_eval.obs["domain"],
        adata_eval.obs["ground_truth"]
    )

    adata.obs["candidate_name"] = cand_name

    adata.uns["run_dataset"] = dataset
    adata.uns["run_label_col"] = FORCE_LABEL_COL
    adata.uns["run_spatial_bin_size"] = int(SPATIAL_BIN_SIZE)
    adata.uns["run_radius"] = int(radius)
    adata.uns["run_n_clusters"] = int(n_clusters)
    adata.uns["run_embedding_key"] = emb_key
    adata.uns["run_ARI"] = float(ari)
    adata.uns["run_NMI"] = float(nmi)
    adata.uns["run_params_json"] = json.dumps(
        sanitize_params_for_uns(candidate),
        ensure_ascii=False
    )

    print(f"Candidate: {cand_name}")
    print(f"ARI: {ari:.6f}")
    print(f"NMI: {nmi:.6f}")

    return adata, ari, nmi


# =========================================================
# 7. 主流程：加载 E16.5_E2S3，分别生成 baseline 和 Ada-GraphST h5ad
# =========================================================
if __name__ == "__main__":

    print("\n" + "=" * 120)
    print(f"START DATASET: {dataset}")
    print("=" * 120)

    adata_raw, label_col, n_clusters, n_eval = load_mouse_embryo_section(
        data_path=data_path,
        force_label_col=FORCE_LABEL_COL,
        bin_size=SPATIAL_BIN_SIZE
    )

    records = []

    for candidate in candidates:
        cand_name = candidate["name"]

        try:
            adata_result, ari, nmi = run_one_candidate(
                adata_input=adata_raw,
                candidate=candidate,
                n_clusters=n_clusters,
                radius=radius,
                seed=seed
            )

            if cand_name == "Baseline_Internal":
                h5ad_name = f"{dataset}_Baseline_Internal_bin{SPATIAL_BIN_SIZE}_radius{radius}.h5ad"
            else:
                h5ad_name = f"{dataset}_AdaGraphST_smooth0p02_sharpen0p03_bin{SPATIAL_BIN_SIZE}_radius{radius}.h5ad"

            h5ad_path = os.path.join(output_dir, h5ad_name)

            adata_to_save = prepare_adata_for_write(adata_result)
            adata_to_save.write_h5ad(h5ad_path, compression="gzip")

            print(f"Saved h5ad: {h5ad_path}")

            records.append({
                "dataset": dataset,
                "name": cand_name,
                "w_smooth": candidate["w_smooth"],
                "w_sharpen": candidate["w_sharpen"],
                "gamma": candidate["gamma"],
                "warmup_epochs": candidate["warmup_epochs"],
                "update_interval": candidate["update_interval"],
                "graph_update_rate": candidate["graph_update_rate"],
                "graph_reg_weight": candidate["graph_reg_weight"],
                "use_learnable_proj": candidate["use_learnable_proj"],
                "label_col": label_col,
                "n_clusters": int(n_clusters),
                "n_obs": int(adata_raw.n_obs),
                "n_eval": int(n_eval),
                "ARI": float(ari),
                "NMI": float(nmi),
                "h5ad_path": h5ad_path,
                "status": "success",
                "Error": "",
            })

        except Exception as e:
            print(f"Candidate [{cand_name}] failed: {e}")

            records.append({
                "dataset": dataset,
                "name": cand_name,
                "w_smooth": candidate["w_smooth"],
                "w_sharpen": candidate["w_sharpen"],
                "gamma": candidate["gamma"],
                "warmup_epochs": candidate["warmup_epochs"],
                "update_interval": candidate["update_interval"],
                "graph_update_rate": candidate["graph_update_rate"],
                "graph_reg_weight": candidate["graph_reg_weight"],
                "use_learnable_proj": candidate["use_learnable_proj"],
                "label_col": label_col,
                "n_clusters": int(n_clusters),
                "n_obs": int(adata_raw.n_obs),
                "n_eval": int(n_eval),
                "ARI": np.nan,
                "NMI": np.nan,
                "h5ad_path": "",
                "status": "failed",
                "Error": str(e),
            })

    # =====================================================
    # 8. 保存结果指标 CSV
    # =====================================================
    result_df = pd.DataFrame(records)

    metrics_csv = os.path.join(
        output_dir,
        f"{dataset}_Baseline_vs_AdaGraphST_smooth0p02_sharpen0p03_metrics.csv"
    )

    result_df.to_csv(metrics_csv, index=False)

    print("\n" + "=" * 120)
    print("FINISHED")
    print("=" * 120)

    print(result_df)

    print("\nSaved metrics CSV:")
    print(metrics_csv)

    print("\nExpected h5ad outputs:")
    print(os.path.join(
        output_dir,
        f"{dataset}_Baseline_Internal_bin{SPATIAL_BIN_SIZE}_radius{radius}.h5ad"
    ))
    print(os.path.join(
        output_dir,
        f"{dataset}_AdaGraphST_smooth0p02_sharpen0p03_bin{SPATIAL_BIN_SIZE}_radius{radius}.h5ad"
    ))