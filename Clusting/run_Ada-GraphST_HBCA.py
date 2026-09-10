import os
import sys
import random
from collections import OrderedDict
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 0. 环境配置
# =========================================================
os.environ['R_HOME'] = '/data2/liangyefeng/miniconda3/envs/graphst/lib/R'

project_root = "/data2/liangyefeng/My_GraphST_Innovation"
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import numpy as np
import pandas as pd
import torch
import scanpy as sc
from sklearn import metrics
from sklearn.decomposition import PCA

from GraphST.GraphST import GraphST
import GraphST.utils as graphst_utils
from GraphST.utils import clustering


# =========================================================
# 1. runtime patch: 修复当前环境下 mclust 调用
# =========================================================
def patched_mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
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

    adata.obs["mclust"] = cls_series.astype(int).astype("category")
    return adata


graphst_utils.mclust_R = patched_mclust_R


# =========================================================
# 2. 固定随机种子
# =========================================================
seed = 41

def reset_seed(seed=41):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

reset_seed(seed)

# 如果服务器显存紧张，可改成 cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print("device:", device)


# =========================================================
# 3. 数据集配置
# 说明：
# - 只有 section1 有 GT，所以只对 section1 做评分和找最优参数
# =========================================================
data_root = "/data2/liangyefeng/My_GraphST_Innovation/data/Human-Breast-Cancer-Block-A"
result_root = "/data2/liangyefeng/My_GraphST_Innovation/Result_HBCA_Section1_CandidatePool"
os.makedirs(result_root, exist_ok=True)

dataset_list = ["section1"]

radius = 50


# =========================================================
# 4. 公共参数
# =========================================================
common_params = {
    "device": device,
    "random_seed": 41,
    "epochs": 600,
    "dim_output": 64,
    "datatype": "10X",
    "warmup_epochs": 200,
    "update_interval": 20,
    "graph_update_rate": 0.3,
    "gamma": 3.0,
    "graph_reg_weight": 0.0,
    "use_learnable_proj": False,
}


# =========================================================
# 5. 候选池构建
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

    # 统一默认值
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
candidate_dict[tuple(candidate_dict[list(candidate_dict.keys())[0]].get(k) for k in KEY_FIELDS)]["use_original_baseline"] = True

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
# 6. GT 与数据读取工具
# =========================================================
def load_gt_for_hbca(file_fold, obs_names):
    """
    对 HBCA：
    - 优先使用 tissue_positions_list_GTs.txt
    - 这个文件无 header，必须 header=None
    - 最后一列作为 GT
    """
    gt_dir = os.path.join(file_fold, "gt")
    tissue_gt_path = os.path.join(gt_dir, "tissue_positions_list_GTs.txt")
    gold_meta_path = os.path.join(gt_dir, "gold_metadata.tsv")

    # 1) 优先使用带 GTs 的文件
    if os.path.exists(tissue_gt_path):
        try:
            df = pd.read_csv(tissue_gt_path, sep="\t", header=None)
            if df.shape[1] == 1:
                df = pd.read_csv(tissue_gt_path, sep=",", header=None)
        except Exception:
            df = pd.read_csv(tissue_gt_path, sep=",", header=None)

        if df.shape[1] == 7:
            df.columns = [
                "barcode",
                "in_tissue",
                "array_row",
                "array_col",
                "pxl_row_in_fullres",
                "pxl_col_in_fullres",
                "ground_truth"
            ]
        else:
            cols = [f"col_{i}" for i in range(df.shape[1])]
            df.columns = cols
            df = df.rename(columns={cols[0]: "barcode", cols[-1]: "ground_truth"})

        print("[GT DEBUG] using:", tissue_gt_path)
        print("[GT DEBUG] shape:", df.shape)
        print("[GT DEBUG] columns:", list(df.columns))
        print("[GT DEBUG] gt unique:", df["ground_truth"].nunique(dropna=True))
        print("[GT DEBUG] head:")
        print(df.head())

        df = df.set_index("barcode")
        df = df.reindex(obs_names)
        gt = df["ground_truth"]

        return gt, tissue_gt_path, "ground_truth"

    # 2) 后备：gold_metadata.tsv
    if os.path.exists(gold_meta_path):
        df = pd.read_csv(gold_meta_path, sep="\t")

        barcode_col = next((c for c in df.columns if "barcode" in c.lower()), None)

        banned = {"id", "barcode", "array_row", "array_col", "imagerow", "imagecol", "x", "y"}
        candidate_cols = [c for c in df.columns if c.lower() not in banned]

        if len(candidate_cols) == 0:
            raise ValueError("gold_metadata.tsv has no valid label column")

        label_col = min(candidate_cols, key=lambda c: df[c].nunique(dropna=True))

        print("[GT DEBUG] using:", gold_meta_path)
        print("[GT DEBUG] columns:", list(df.columns))
        print("[GT DEBUG] chosen label col:", label_col)
        print("[GT DEBUG] label nunique:", df[label_col].nunique(dropna=True))

        if barcode_col is not None:
            df = df.set_index(barcode_col)
            df = df.reindex(obs_names)
            gt = df[label_col]
        else:
            gt = pd.Series(df[label_col].values[:len(obs_names)], index=obs_names[:len(df)])

        return gt, gold_meta_path, label_col

    raise FileNotFoundError(f"No GT file found under {gt_dir}")


def attach_ground_truth(adata, file_fold):
    gt, gt_path, label_col = load_gt_for_hbca(file_fold, adata.obs_names)
    adata.obs["ground_truth"] = gt
    return adata, gt_path, label_col


def get_n_clusters_from_gt(adata):
    adata_eval = adata[~pd.isnull(adata.obs["ground_truth"])].copy()
    n_clusters = adata_eval.obs["ground_truth"].nunique()
    if n_clusters <= 1:
        raise ValueError(f"GT unique classes <= 1, got {n_clusters}")
    return n_clusters, adata_eval.n_obs


def load_visium_adata(file_fold, dataset):
    """
    section1 的 h5 文件名不是默认的 filtered_feature_bc_matrix.h5
    """
    candidate_files = [
        f"{dataset}_filtered_feature_bc_matrix.h5",
        "filtered_feature_bc_matrix.h5",
    ]

    last_err = None
    for cf in candidate_files:
        try:
            adata = sc.read_visium(
                file_fold,
                count_file=cf,
                load_images=True
            )
            adata.var_names_make_unique()
            print(f"[load_visium_adata] loaded with count_file={cf}")
            return adata
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to load Visium data from {file_fold}. Last error: {last_err}")


# =========================================================
# 7. 跑一个候选
# =========================================================
def run_one_candidate(adata_input, candidate, file_fold, n_clusters, radius, seed=41):
    cand_name = candidate["name"]
    print("\n" + "-" * 100)
    print(f"Running candidate: {cand_name}")
    print("-" * 100)

    reset_seed(seed)

    adata = adata_input.copy()

    # 关键修复：
    # Baseline_Internal 完全走原版 GraphST 调用方式，
    # 保证它和你单独跑原版 GraphST 的逻辑一致。
    if candidate.get("use_original_baseline", False):
        print("[INFO] Using ORIGINAL GraphST baseline path")
        model = GraphST(adata, device=device)
    else:
        graphst_params = {
            k: v for k, v in candidate.items()
            if k not in ["name", "use_original_baseline"]
        }
        model = GraphST(adata, **graphst_params)

    adata = model.train()

    if 'emb' not in adata.obsm:
        raise KeyError("adata.obsm['emb'] not found after training.")

    n_pcs = min(20, adata.obsm['emb'].shape[1])
    adata.obsm['emb_pca'] = PCA(
        n_components=n_pcs,
        random_state=seed
    ).fit_transform(adata.obsm['emb'])

    clustering(adata, n_clusters, radius=radius, method='mclust', refinement=True)

    adata, gt_path, label_col = attach_ground_truth(adata, file_fold)
    adata_eval = adata[~pd.isnull(adata.obs['ground_truth'])].copy()

    ari = metrics.adjusted_rand_score(
        adata_eval.obs['domain'],
        adata_eval.obs['ground_truth']
    )
    nmi = metrics.normalized_mutual_info_score(
        adata_eval.obs['domain'],
        adata_eval.obs['ground_truth']
    )

    print(f"Candidate: {cand_name}")
    print(f"GT file: {gt_path}")
    print(f"Label column: {label_col}")
    print(f"ARI: {ari:.6f}")
    print(f"NMI: {nmi:.6f}")

    return ari, nmi


# =========================================================
# 8. 批量跑当前数据集
# =========================================================
all_records = []
summary_records = []

for dataset in dataset_list:
    print("\n" + "=" * 120)
    print(f"START DATASET: {dataset}")
    print("=" * 120)

    file_fold = os.path.join(data_root, dataset)

    dataset_save_dir = os.path.join(result_root, f"HBCA_{dataset}")
    os.makedirs(dataset_save_dir, exist_ok=True)

    try:
        adata_raw = load_visium_adata(file_fold, dataset)
        adata_raw, gt_path, label_col = attach_ground_truth(adata_raw, file_fold)
        n_clusters, n_obs_eval = get_n_clusters_from_gt(adata_raw)

        print(f"[Dataset {dataset}] GT file: {gt_path}")
        print(f"[Dataset {dataset}] label column: {label_col}")
        print(f"[Dataset {dataset}] n_clusters from GT: {n_clusters}")
        print(f"[Dataset {dataset}] n_obs_eval: {n_obs_eval}")
        print("[Dataset] GT counts:")
        print(adata_raw.obs["ground_truth"].value_counts(dropna=False).head(20))

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
            "best_name": None,
            "best_ARI": np.nan,
            "best_NMI": np.nan,
            "status": f"prepare_failed: {e}"
        })
        continue

    dataset_records = []
    best_ari = -1.0
    best_nmi = -1.0
    best_name = None

    for idx, candidate in enumerate(candidates, start=1):
        print(f"\n[{dataset}] candidate {idx}/{len(candidates)}")

        try:
            ari, nmi = run_one_candidate(
                adata_input=adata_raw,
                candidate=candidate,
                file_fold=file_fold,
                n_clusters=n_clusters,
                radius=radius,
                seed=seed
            )

            record = dict(candidate)
            record["dataset"] = dataset
            record["ARI"] = ari
            record["NMI"] = nmi
            dataset_records.append(record)
            all_records.append(record)

            if (ari > best_ari) or (np.isclose(ari, best_ari) and nmi > best_nmi):
                best_ari = ari
                best_nmi = nmi
                best_name = candidate["name"]

        except Exception as e:
            print(f"Candidate [{candidate['name']}] failed on {dataset}: {e}")
            record = dict(candidate)
            record["dataset"] = dataset
            record["ARI"] = np.nan
            record["NMI"] = np.nan
            record["Error"] = str(e)
            dataset_records.append(record)
            all_records.append(record)
            continue

    dataset_df = pd.DataFrame(dataset_records)
    dataset_df["ARI_rank"] = dataset_df["ARI"].rank(ascending=False, method="min")
    dataset_df["NMI_rank"] = dataset_df["NMI"].rank(ascending=False, method="min")

    dataset_df = dataset_df.sort_values(
        by=["ARI", "NMI"],
        ascending=[False, False]
    )

    dataset_csv = os.path.join(dataset_save_dir, f"{dataset}_all_candidate_results.csv")
    dataset_df.to_csv(dataset_csv, index=False)

    top10_csv = os.path.join(dataset_save_dir, f"{dataset}_top10_candidates.csv")
    dataset_df.head(10).to_csv(top10_csv, index=False)

    summary_records.append({
        "dataset": dataset,
        "best_name": best_name,
        "best_ARI": best_ari if best_ari >= 0 else np.nan,
        "best_NMI": best_nmi if best_nmi >= 0 else np.nan,
        "status": "success"
    })

    print("\n" + "-" * 100)
    print(f"BEST RESULT FOR {dataset}")
    print(f"Best candidate: {best_name}")
    print(f"Best ARI: {best_ari:.6f}")
    print(f"Best NMI: {best_nmi:.6f}")
    print("-" * 100)


# =========================================================
# 9. 保存全局 CSV
# =========================================================
all_results_df = pd.DataFrame(all_records)
all_results_csv = os.path.join(result_root, "HBCA_section1_AllCandidate_LongResults.csv")
all_results_df.to_csv(all_results_csv, index=False)

ari_wide = all_results_df.pivot(index="dataset", columns="name", values="ARI").reset_index()
ari_wide_csv = os.path.join(result_root, "HBCA_section1_ARI_Wide.csv")
ari_wide.to_csv(ari_wide_csv, index=False)

nmi_wide = all_results_df.pivot(index="dataset", columns="name", values="NMI").reset_index()
nmi_wide_csv = os.path.join(result_root, "HBCA_section1_NMI_Wide.csv")
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
    by=["median_ARI", "mean_ARI", "mean_NMI"],
    ascending=[False, False, False]
)
candidate_summary_csv = os.path.join(result_root, "HBCA_section1_Candidate_Summary.csv")
candidate_summary_df.to_csv(candidate_summary_csv, index=False)

best_summary_df = pd.DataFrame(summary_records)
best_summary_csv = os.path.join(result_root, "HBCA_section1_Best_Summary.csv")
best_summary_df.to_csv(best_summary_csv, index=False)

print("\n" + "=" * 120)
print("HBCA SECTION1 FINISHED")
print("=" * 120)
print(best_summary_df)

print(f"\nSaved candidate definition to: {os.path.join(result_root, 'CandidatePool_Definition.csv')}")
print(f"Saved long results to: {all_results_csv}")
print(f"Saved ARI wide table to: {ari_wide_csv}")
print(f"Saved NMI wide table to: {nmi_wide_csv}")
print(f"Saved candidate summary to: {candidate_summary_csv}")
print(f"Saved best summary to: {best_summary_csv}")