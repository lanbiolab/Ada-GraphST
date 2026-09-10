import os
import sys
import random
from collections import OrderedDict

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
def patched_mclust_R(adata, num_cluster, modelNames="EEE", used_obsm="emb_pca", random_seed=2020):
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
# 3. 数据集配置
# =========================================================
data_root = "/data2/liangyefeng/My_GraphST_Innovation/data/Mouse-embryo"
result_root = os.path.join(data_root, "GraphST_CandidatePool_E16.5_E2S1_S6_SmoothSharpenOnly")
os.makedirs(result_root, exist_ok=True)

dataset_paths = [
    # os.path.join(data_root, "E16.5_E2S1.MOSTA.h5ad"),
    # os.path.join(data_root, "E16.5_E2S2.MOSTA.h5ad"),
    # os.path.join(data_root, "E16.5_E2S3.MOSTA.h5ad"),
    # os.path.join(data_root, "E16.5_E2S4.MOSTA.h5ad"),
    os.path.join(data_root, "E16.5_E2S11.MOSTA.h5ad"),
]

FORCE_LABEL_COL = "annotation"
SPATIAL_BIN_SIZE = 3
radius = 50

# =========================================================
# 4. 公共参数（固定）
# =========================================================
common_params = {
    "device": device,
    "random_seed": 41,
    "epochs": 600,
    "dim_output": 64,
    "datatype": "Stereo",
    "warmup_epochs": 200,
    "update_interval": 20,
    "graph_update_rate": 0.3,
    "gamma": 2.5,                 # 固定
    "graph_reg_weight": 0.0,      # 固定
    "use_learnable_proj": False,  # 固定
}

# =========================================================
# 5. 候选池构建：只保留 smooth × sharpen 组合
# =========================================================
def fmt_num(x):
    s = str(x)
    s = s.replace(".", "p")
    s = s.replace("-", "m")
    return s


candidate_dict = OrderedDict()

KEY_FIELDS = [
    "w_smooth",
    "w_sharpen",
    "gamma",
    "warmup_epochs",
    "update_interval",
    "graph_update_rate",
    "graph_reg_weight",
    "use_learnable_proj",
]


def add_candidate(name, **updates):
    cand = dict(common_params)
    cand.update(updates)

    key = tuple(cand.get(k) for k in KEY_FIELDS)
    if key not in candidate_dict:
        cand["name"] = name
        candidate_dict[key] = cand


# baseline
add_candidate(
    "Baseline_Internal",
    w_smooth=0.0,
    w_sharpen=0.0,
)

# 搜索池
smooth_pool = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]
sharpen_pool = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70]

# smooth-only
for ws in smooth_pool:
    add_candidate(
        f"Smooth_{fmt_num(ws)}",
        w_smooth=ws,
        w_sharpen=0.0,
    )

# sharpen-only
for hs in sharpen_pool:
    add_candidate(
        f"Sharpen_{fmt_num(hs)}",
        w_smooth=0.0,
        w_sharpen=hs,
    )

# hybrid: 只搜 smooth/sharpen 组合，不再搜 gamma / proj / 其他扰动
for ws in smooth_pool:
    for hs in sharpen_pool:
        add_candidate(
            f"Hybrid_{fmt_num(ws)}_{fmt_num(hs)}",
            w_smooth=ws,
            w_sharpen=hs,
        )

candidates = list(candidate_dict.values())
print(f"Total candidates: {len(candidates)}")

candidate_table = pd.DataFrame(candidates)
candidate_table.to_csv(
    os.path.join(result_root, "CandidatePool_Definition.csv"),
    index=False
)
print("Saved candidate pool definition.")

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


def load_mouse_embryo_section(data_path, force_label_col="annotation", bin_size=3):
    adata = sc.read_h5ad(data_path)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    adata, _ = ensure_spatial(adata)

    label_col = force_label_col if force_label_col is not None else detect_label_col(adata.obs)
    if label_col is None:
        raise ValueError(f"No label column detected. obs columns: {list(adata.obs.columns)}")

    adata.obs["ground_truth"] = adata.obs[label_col].copy()
    adata = to_counts_adata(adata)
    adata = spatial_bin_adata(adata, label_col="ground_truth", bin_size=bin_size)

    eval_mask = ~pd.isnull(adata.obs["ground_truth"])
    n_eval = int(eval_mask.sum())
    if n_eval == 0:
        raise ValueError("No valid ground_truth after binning.")

    n_clusters = int(adata.obs.loc[eval_mask, "ground_truth"].nunique())
    if n_clusters < 2:
        raise ValueError(f"Invalid n_clusters={n_clusters} after binning.")

    return adata, label_col, n_clusters, n_eval


def run_one_candidate(adata_input, candidate, n_clusters, radius, seed=41):
    cand_name = candidate["name"]
    print("\n" + "-" * 100)
    print(f"Running candidate: {cand_name}")
    print("-" * 100)

    adata = adata_input.copy()
    graphst_params = {k: v for k, v in candidate.items() if k != "name"}

    model = GraphST(adata, **graphst_params)
    adata = model.train()

    emb_key = None
    for k in ["emb", "GraphST", "X_emb"]:
        if k in adata.obsm:
            emb_key = k
            break
    if emb_key is None:
        raise KeyError(f"No embedding key found in adata.obsm. Keys={list(adata.obsm.keys())}")

    n_pcs = min(20, adata.obsm[emb_key].shape[1])
    adata.obsm["emb_pca"] = PCA(n_components=n_pcs, random_state=seed).fit_transform(adata.obsm[emb_key])

    clustering(adata, n_clusters, radius=radius, method="mclust", refinement=True)

    adata_eval = adata[~pd.isnull(adata.obs["ground_truth"])].copy()

    ari = metrics.adjusted_rand_score(adata_eval.obs["domain"], adata_eval.obs["ground_truth"])
    nmi = metrics.normalized_mutual_info_score(adata_eval.obs["domain"], adata_eval.obs["ground_truth"])

    print(f"Candidate: {cand_name}")
    print(f"ARI: {ari:.6f}")
    print(f"NMI: {nmi:.6f}")

    return ari, nmi

# =========================================================
# 7. 批量跑全部 Mouse-embryo 切片
# =========================================================
all_records = []
summary_records = []

for data_path in dataset_paths:
    dataset = os.path.basename(data_path).replace(".MOSTA.h5ad", "")

    print("\n" + "=" * 120)
    print(f"START DATASET: {dataset}")
    print("=" * 120)

    dataset_save_dir = os.path.join(result_root, dataset)
    os.makedirs(dataset_save_dir, exist_ok=True)

    try:
        adata_raw, label_col, n_clusters, n_eval = load_mouse_embryo_section(
            data_path,
            force_label_col=FORCE_LABEL_COL,
            bin_size=SPATIAL_BIN_SIZE
        )
        print(f"[{dataset}] shape after binning: {adata_raw.shape}")
        print(f"[{dataset}] label_col: {label_col}")
        print(f"[{dataset}] n_clusters: {n_clusters}")
        print(f"[{dataset}] n_eval: {n_eval}")
        print(f"[{dataset}] GT counts:")
        print(adata_raw.obs["ground_truth"].value_counts(dropna=False))
    except Exception as e:
        print(f"Failed to prepare dataset {dataset}: {e}")
        for cand in candidates:
            record = dict(cand)
            record["dataset"] = dataset
            record["ARI"] = np.nan
            record["NMI"] = np.nan
            record["Error"] = f"prepare_failed: {e}"
            all_records.append(record)

        summary_records.append({
            "dataset": dataset,
            "label_col": FORCE_LABEL_COL,
            "n_clusters": np.nan,
            "n_obs": np.nan,
            "n_eval": np.nan,
            "baseline_name": "Baseline_Internal",
            "baseline_ARI": np.nan,
            "baseline_NMI": np.nan,
            "best_name": None,
            "best_ARI": np.nan,
            "best_NMI": np.nan,
            "delta_ARI_vs_baseline": np.nan,
            "delta_NMI_vs_baseline": np.nan,
            "status": f"prepare_failed: {e}",
        })
        continue

    dataset_records = []
    best_ari = -1.0
    best_nmi = -1.0
    best_name = None
    baseline_ari = np.nan
    baseline_nmi = np.nan

    for idx, candidate in enumerate(candidates, start=1):
        print(f"\n[{dataset}] candidate {idx}/{len(candidates)}")

        try:
            ari, nmi = run_one_candidate(
                adata_input=adata_raw,
                candidate=candidate,
                n_clusters=n_clusters,
                radius=radius,
                seed=seed
            )

            record = dict(candidate)
            record["dataset"] = dataset
            record["label_col"] = label_col
            record["n_clusters"] = n_clusters
            record["n_obs"] = int(adata_raw.n_obs)
            record["n_eval"] = int(n_eval)
            record["ARI"] = ari
            record["NMI"] = nmi
            dataset_records.append(record)
            all_records.append(record)

            if candidate["name"] == "Baseline_Internal":
                baseline_ari = ari
                baseline_nmi = nmi

            if ari > best_ari:
                best_ari = ari
                best_nmi = nmi
                best_name = candidate["name"]

        except Exception as e:
            print(f"Candidate [{candidate['name']}] failed on {dataset}: {e}")
            record = dict(candidate)
            record["dataset"] = dataset
            record["label_col"] = label_col
            record["n_clusters"] = n_clusters
            record["n_obs"] = int(adata_raw.n_obs)
            record["n_eval"] = int(n_eval)
            record["ARI"] = np.nan
            record["NMI"] = np.nan
            record["Error"] = str(e)
            dataset_records.append(record)
            all_records.append(record)
            continue

    dataset_df = pd.DataFrame(dataset_records)
    dataset_csv = os.path.join(dataset_save_dir, f"{dataset}_all_candidate_results.csv")
    dataset_df.to_csv(dataset_csv, index=False)

    summary_records.append({
        "dataset": dataset,
        "label_col": label_col,
        "n_clusters": int(n_clusters),
        "n_obs": int(adata_raw.n_obs),
        "n_eval": int(n_eval),
        "baseline_name": "Baseline_Internal",
        "baseline_ARI": baseline_ari,
        "baseline_NMI": baseline_nmi,
        "best_name": best_name,
        "best_ARI": best_ari if best_ari >= 0 else np.nan,
        "best_NMI": best_nmi if best_nmi >= 0 else np.nan,
        "delta_ARI_vs_baseline": (best_ari - baseline_ari) if (best_ari >= 0 and pd.notna(baseline_ari)) else np.nan,
        "delta_NMI_vs_baseline": (best_nmi - baseline_nmi) if (best_nmi >= 0 and pd.notna(baseline_nmi)) else np.nan,
        "status": "success"
    })

    print("\n" + "-" * 100)
    print(f"BASELINE VS BEST FOR {dataset}")
    print(f"Baseline ARI: {baseline_ari:.6f}" if pd.notna(baseline_ari) else "Baseline ARI: nan")
    print(f"Baseline NMI: {baseline_nmi:.6f}" if pd.notna(baseline_nmi) else "Baseline NMI: nan")
    print(f"Best candidate: {best_name}")
    print(f"Best ARI: {best_ari:.6f}" if best_ari >= 0 else "Best ARI: nan")
    print(f"Best NMI: {best_nmi:.6f}" if best_nmi >= 0 else "Best NMI: nan")
    print("-" * 100)

# =========================================================
# 8. 保存全局 CSV
# =========================================================
all_results_df = pd.DataFrame(all_records)
all_results_csv = os.path.join(result_root, "All_MouseEmbryo_AllCandidate_LongResults.csv")
all_results_df.to_csv(all_results_csv, index=False)

ari_wide = all_results_df.pivot(index="dataset", columns="name", values="ARI").reset_index()
ari_wide_csv = os.path.join(result_root, "All_MouseEmbryo_ARI_Wide.csv")
ari_wide.to_csv(ari_wide_csv, index=False)

nmi_wide = all_results_df.pivot(index="dataset", columns="name", values="NMI").reset_index()
nmi_wide_csv = os.path.join(result_root, "All_MouseEmbryo_NMI_Wide.csv")
nmi_wide.to_csv(nmi_wide_csv, index=False)

candidate_summary_records = []
for cand_name, subdf in all_results_df.groupby("name"):
    candidate_summary_records.append({
        "name": cand_name,
        "mean_ARI": subdf["ARI"].mean(skipna=True),
        "median_ARI": subdf["ARI"].median(skipna=True),
        "std_ARI": subdf["ARI"].std(skipna=True),
        "mean_NMI": subdf["NMI"].mean(skipna=True),
        "median_NMI": subdf["NMI"].median(skipna=True),
        "std_NMI": subdf["NMI"].std(skipna=True),
        "num_valid_datasets": int(subdf["ARI"].notna().sum()),
    })

candidate_summary_df = pd.DataFrame(candidate_summary_records).sort_values(
    by=["median_ARI", "mean_ARI"],
    ascending=[False, False]
)
candidate_summary_csv = os.path.join(result_root, "All_MouseEmbryo_Candidate_Summary.csv")
candidate_summary_df.to_csv(candidate_summary_csv, index=False)

best_summary_df = pd.DataFrame(summary_records)
best_summary_csv = os.path.join(result_root, "All_MouseEmbryo_Best_Summary.csv")
best_summary_df.to_csv(best_summary_csv, index=False)

print("\n" + "=" * 120)
print("ALL DATASETS FINISHED")
print("=" * 120)
print(best_summary_df)

print(f"\nSaved candidate definition to: {os.path.join(result_root, 'CandidatePool_Definition.csv')}")
print(f"Saved long results to: {all_results_csv}")
print(f"Saved ARI wide table to: {ari_wide_csv}")
print(f"Saved NMI wide table to: {nmi_wide_csv}")
print(f"Saved candidate summary to: {candidate_summary_csv}")
print(f"Saved best summary to: {best_summary_csv}")