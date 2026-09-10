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
#    保持你原来的逻辑不变
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
# 3. 数据集配置（改成 MERFISH）
# =========================================================
data_root = "/data2/liangyefeng/My_GraphST_Innovation/data/MERFISH"
result_root = os.path.join(data_root, "GraphST_CandidatePool_MERFISH")
os.makedirs(result_root, exist_ok=True)

dataset_paths = [
    os.path.join(data_root, "fbe9c55e-518b-4bb7-9e1d-00322be93695.h5ad"),
    os.path.join(data_root, "bf8cb800-86b2-4ac0-b8a7-cc89fedaec89.h5ad"),
]

DATASET_ALIAS = {
    "fbe9c55e-518b-4bb7-9e1d-00322be93695": "MERFISH_A",
    "bf8cb800-86b2-4ac0-b8a7-cc89fedaec89": "MERFISH_B",
}

FORCE_LABEL_COL = "cell_type"
FORCE_SLICE_COL = "slice"

# 这里因为换成 MERFISH raw-count binning 了，bin_size 要大很多
SPATIAL_BIN_SIZE = 100.0

# GraphST refinement 半径；对 100 大小的 bin，先给 100 比较合理
radius = 100

MIN_CELLS_PER_BIN = 3
MIN_OBS_PER_SECTION = 50

INVALID_LABELS = {
    "", "na", "n/a", "nan", "none", "null",
    "unknown", "unassigned", "unlabeled", "unlabelled", "undefined"
}

# =========================================================
# 4. 公共参数（保留你原来的逻辑）
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
    "gamma": 3.0,
    "graph_reg_weight": 0.0,
    "use_learnable_proj": False,
}

# =========================================================
# 5. 候选池构建（完全沿用你原来的逻辑）
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

    if cand.get("w_sharpen", 0.0) == 0.0:
        cand["gamma"] = 3.0
        cand["warmup_epochs"] = 200
        cand["update_interval"] = 20
        cand["graph_update_rate"] = 0.3
        cand["graph_reg_weight"] = 0.0
        cand["use_learnable_proj"] = False

    key = tuple(cand.get(k) for k in KEY_FIELDS)
    if key not in candidate_dict:
        cand["name"] = name
        candidate_dict[key] = cand


# A. Baseline
add_candidate(
    "Baseline_Internal",
    w_smooth=0.0,
    w_sharpen=0.0,
    use_learnable_proj=False,
)

# B. Smooth-only
smooth_pool = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]
for ws in smooth_pool:
    add_candidate(
        f"Smooth_{fmt_num(ws)}",
        w_smooth=ws,
        w_sharpen=0.0,
        use_learnable_proj=False,
    )

# C. Sharpen-only
sharpen_pool = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70]
for hs in sharpen_pool:
    if hs <= 0.03:
        gamma_pool = [1.5, 2.0, 2.5]
    elif hs <= 0.10:
        gamma_pool = [1.5, 2.0, 2.5, 3.0]
    else:
        gamma_pool = [2.0, 2.5, 3.0, 4.0]

    for g in gamma_pool:
        add_candidate(
            f"Sharpen_{fmt_num(hs)}_g{fmt_num(g)}",
            w_smooth=0.0,
            w_sharpen=hs,
            gamma=g,
            use_learnable_proj=False,
        )

for hs in [0.05, 0.08, 0.10]:
    for g in [2.5, 3.0]:
        add_candidate(
            f"Sharpen_{fmt_num(hs)}_g{fmt_num(g)}_Proj",
            w_smooth=0.0,
            w_sharpen=hs,
            gamma=g,
            use_learnable_proj=True,
        )

# D. Hybrid
hybrid_smooth_pool = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12]
hybrid_sharpen_pool = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
hybrid_gamma_pool = [2.0, 2.5, 3.0]

for ws in hybrid_smooth_pool:
    for hs in hybrid_sharpen_pool:
        if ws >= 0.12 and hs >= 0.08:
            continue
        if ws >= 0.10 and hs >= 0.10:
            continue

        if hs <= 0.02:
            gamma_choices = [1.5, 2.0]
        else:
            gamma_choices = hybrid_gamma_pool

        for g in gamma_choices:
            add_candidate(
                f"Hybrid_{fmt_num(ws)}_{fmt_num(hs)}_g{fmt_num(g)}",
                w_smooth=ws,
                w_sharpen=hs,
                gamma=g,
                use_learnable_proj=False,
            )

# E. 次级扰动
anchor_configs = [
    {"ws": 0.03, "hs": 0.05, "g": 2.0},
    {"ws": 0.05, "hs": 0.05, "g": 2.5},
    {"ws": 0.05, "hs": 0.08, "g": 2.5},
    {"ws": 0.08, "hs": 0.10, "g": 2.5},
]

for a in anchor_configs:
    for wup in [100, 150, 250, 300]:
        add_candidate(
            f"Hybrid_{fmt_num(a['ws'])}_{fmt_num(a['hs'])}_g{fmt_num(a['g'])}_W{wup}",
            w_smooth=a["ws"],
            w_sharpen=a["hs"],
            gamma=a["g"],
            warmup_epochs=wup,
            use_learnable_proj=False,
        )

for a in anchor_configs:
    for upd in [5, 10, 30, 50, 100]:
        add_candidate(
            f"Hybrid_{fmt_num(a['ws'])}_{fmt_num(a['hs'])}_g{fmt_num(a['g'])}_U{upd}",
            w_smooth=a["ws"],
            w_sharpen=a["hs"],
            gamma=a["g"],
            update_interval=upd,
            use_learnable_proj=False,
        )

for a in anchor_configs:
    for ema in [0.05, 0.10, 0.20, 0.50, 0.70]:
        add_candidate(
            f"Hybrid_{fmt_num(a['ws'])}_{fmt_num(a['hs'])}_g{fmt_num(a['g'])}_E{fmt_num(ema)}",
            w_smooth=a["ws"],
            w_sharpen=a["hs"],
            gamma=a["g"],
            graph_update_rate=ema,
            use_learnable_proj=False,
        )

for a in anchor_configs:
    for reg in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
        add_candidate(
            f"Hybrid_{fmt_num(a['ws'])}_{fmt_num(a['hs'])}_g{fmt_num(a['g'])}_R{fmt_num(reg)}",
            w_smooth=a["ws"],
            w_sharpen=a["hs"],
            gamma=a["g"],
            graph_reg_weight=reg,
            use_learnable_proj=False,
        )

for a in anchor_configs:
    add_candidate(
        f"Hybrid_{fmt_num(a['ws'])}_{fmt_num(a['hs'])}_g{fmt_num(a['g'])}_Proj",
        w_smooth=a["ws"],
        w_sharpen=a["hs"],
        gamma=a["g"],
        use_learnable_proj=True,
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
    for key in ["X_spatial_coords", "spatial_coords", "spatial", "X_spatial", "spatial_stereo"]:
        if key in adata.obsm:
            arr = np.asarray(adata.obsm[key])
            if arr.ndim == 2 and arr.shape[1] >= 2:
                adata.obsm["spatial"] = arr[:, :2].astype(np.float64)
                return adata, key

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


def get_sections(adata, slice_col="slice"):
    values = adata.obs[slice_col].astype(str).to_numpy()
    sections = []
    for v in pd.unique(values):
        if str(v).lower() in {"nan", "none", "na", ""}:
            continue
        idx = np.where(values == str(v))[0]
        if len(idx) > 0:
            sections.append((str(v), idx))
    return sections


def build_binned_adata_from_raw(adata, label_col="ground_truth", bin_size=100.0, min_cells_per_bin=3):
    coords = np.asarray(adata.obsm["spatial"]).astype(np.float64)
    x = coords[:, 0]
    y = coords[:, 1]

    if adata.raw is None:
        raise ValueError("adata.raw is None，当前脚本要求使用 adata.raw.X 作为原始 counts")

    X_raw = adata.raw.X
    var_raw = adata.raw.var.copy()

    if not sparse.issparse(X_raw):
        X_raw = sp.csr_matrix(np.asarray(X_raw))
    else:
        X_raw = X_raw.tocsr()

    check_matrix_nonnegative(X_raw, "raw_counts")

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

    adata_bin = sc.AnnData(X=X_bin_keep)
    adata_bin.obs_names = pd.Index(np.asarray(unique_groups)[keep_idx]).astype(str)
    adata_bin.var_names = pd.Index(var_raw.index).astype(str)
    adata_bin.var_names_make_unique()
    adata_bin.obs_names_make_unique()

    adata_bin.obsm["spatial"] = spatial_bin[keep_idx].astype(np.float64)
    adata_bin.obs["ground_truth"] = label_bin.values[keep_idx]
    adata_bin.obs["n_spots"] = n_spots_bin.values[keep_idx]
    adata_bin.obs["valid_label_count"] = valid_label_count.values[keep_idx]
    adata_bin.obs["majority_count"] = majority_count.values[keep_idx]
    adata_bin.obs["purity"] = purity[keep_idx]

    old_to_new = np.full(n_groups, -1, dtype=int)
    old_to_new[keep_idx] = np.arange(len(keep_idx), dtype=int)
    cell_to_bin_new = old_to_new[group_codes]

    return adata_bin, cell_to_bin_new


def load_merfish_slice_binned(adata_sec, force_label_col="cell_type", bin_size=100.0):
    adata_sec.obs_names_make_unique()
    adata_sec.var_names_make_unique()

    if force_label_col not in adata_sec.obs.columns:
        raise ValueError(f"{force_label_col} 不在 obs.columns 中")

    adata_sec.obs["ground_truth"] = adata_sec.obs[force_label_col].copy()
    adata_bin, cell_to_bin_new = build_binned_adata_from_raw(
        adata_sec,
        label_col="ground_truth",
        bin_size=bin_size,
        min_cells_per_bin=MIN_CELLS_PER_BIN
    )

    eval_mask = ~pd.isnull(adata_bin.obs["ground_truth"])
    n_eval = int(eval_mask.sum())
    if n_eval == 0:
        raise ValueError("No valid ground_truth after binning.")

    n_clusters = int(adata_bin.obs.loc[eval_mask, "ground_truth"].nunique())
    if n_clusters < 2:
        raise ValueError(f"Invalid n_clusters={n_clusters} after binning.")

    return adata_bin, cell_to_bin_new, force_label_col, n_clusters, n_eval


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

    n_pcs = min(20, adata.obsm[emb_key].shape[1], max(2, adata.n_obs - 1))
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
# 7. 批量跑全部 MERFISH 6 个切片
# =========================================================
all_records = []
summary_records = []

for data_path in dataset_paths:
    dataset_id = os.path.basename(data_path).replace(".h5ad", "")
    dataset_alias = DATASET_ALIAS.get(dataset_id, dataset_id[:8])

    print("\n" + "=" * 120)
    print(f"LOAD DATASET: {dataset_alias} ({dataset_id})")
    print("=" * 120)

    try:
        adata_all = sc.read_h5ad(data_path)
        adata_all.obs_names_make_unique()
        adata_all.var_names_make_unique()
        adata_all, spatial_source = ensure_spatial(adata_all)

        if FORCE_LABEL_COL not in adata_all.obs.columns:
            raise ValueError(f"{FORCE_LABEL_COL} 不在 obs.columns 中")
        if FORCE_SLICE_COL not in adata_all.obs.columns:
            raise ValueError(f"{FORCE_SLICE_COL} 不在 obs.columns 中")

        print(f"[{dataset_alias}] shape={adata_all.shape}")
        print(f"[{dataset_alias}] spatial_source={spatial_source}")

        sections = get_sections(adata_all, FORCE_SLICE_COL)
        print(f"[{dataset_alias}] total sections={len(sections)}")

    except Exception as e:
        print(f"Failed to load dataset {dataset_alias}: {e}")
        continue

    for section_name, idx in sections:
        dataset = f"{dataset_alias}_slice{section_name}"

        print("\n" + "=" * 120)
        print(f"START DATASET: {dataset}")
        print("=" * 120)

        dataset_save_dir = os.path.join(result_root, dataset)
        os.makedirs(dataset_save_dir, exist_ok=True)

        try:
            adata_sec = adata_all[idx].copy()
            adata_sec.obs_names_make_unique()
            adata_sec.var_names_make_unique()

            if adata_sec.n_obs < MIN_OBS_PER_SECTION:
                raise ValueError(f"n_obs={adata_sec.n_obs} < MIN_OBS_PER_SECTION={MIN_OBS_PER_SECTION}")

            summarize_matrix_quick(adata_sec.X, f"{dataset} section_X_processed")
            summarize_matrix_quick(adata_sec.raw.X, f"{dataset} section_raw_X")

            adata_raw, cell_to_bin_new, label_col, n_clusters, n_eval = load_merfish_slice_binned(
                adata_sec,
                force_label_col=FORCE_LABEL_COL,
                bin_size=SPATIAL_BIN_SIZE
            )
            print(f"[{dataset}] shape after binning: {adata_raw.shape}")
            print(f"[{dataset}] label_col: {label_col}")
            print(f"[{dataset}] n_clusters: {n_clusters}")
            print(f"[{dataset}] n_eval: {n_eval}")
            print(f"[{dataset}] purity mean: {adata_raw.obs['purity'].astype(float).mean():.4f}")
            print(f"[{dataset}] GT counts:")
            print(adata_raw.obs["ground_truth"].value_counts(dropna=False))
        except Exception as e:
            print(f"Failed to prepare dataset {dataset}: {e}")
            for cand in candidates:
                record = dict(cand)
                record["dataset"] = dataset
                record["dataset_alias"] = dataset_alias
                record["section"] = section_name
                record["ARI"] = np.nan
                record["NMI"] = np.nan
                record["Error"] = f"prepare_failed: {e}"
                all_records.append(record)

            summary_records.append({
                "dataset": dataset,
                "dataset_alias": dataset_alias,
                "section": section_name,
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

        for idx_cand, candidate in enumerate(candidates, start=1):
            print(f"\n[{dataset}] candidate {idx_cand}/{len(candidates)}")

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
                record["dataset_alias"] = dataset_alias
                record["section"] = section_name
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
                record["dataset_alias"] = dataset_alias
                record["section"] = section_name
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
            "dataset_alias": dataset_alias,
            "section": section_name,
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
all_results_csv = os.path.join(result_root, "All_MERFISH_AllCandidate_LongResults.csv")
all_results_df.to_csv(all_results_csv, index=False)

ari_wide = all_results_df.pivot(index="dataset", columns="name", values="ARI").reset_index()
ari_wide_csv = os.path.join(result_root, "All_MERFISH_ARI_Wide.csv")
ari_wide.to_csv(ari_wide_csv, index=False)

nmi_wide = all_results_df.pivot(index="dataset", columns="name", values="NMI").reset_index()
nmi_wide_csv = os.path.join(result_root, "All_MERFISH_NMI_Wide.csv")
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
candidate_summary_csv = os.path.join(result_root, "All_MERFISH_Candidate_Summary.csv")
candidate_summary_df.to_csv(candidate_summary_csv, index=False)

best_summary_df = pd.DataFrame(summary_records)
best_summary_csv = os.path.join(result_root, "All_MERFISH_Best_Summary.csv")
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