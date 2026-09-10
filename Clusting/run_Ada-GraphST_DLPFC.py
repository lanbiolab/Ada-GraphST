import os
import sys
import random
from collections import OrderedDict

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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:", device)


# =========================================================
# 3. 数据集配置
# =========================================================
data_root = "/data2/liangyefeng/My_GraphST_Innovation/data/1.DLPFC"
result_root = "/data2/liangyefeng/My_GraphST_Innovation/Result_LargeCandidatePool_HybridOnly_WithBaselineCompare"
os.makedirs(result_root, exist_ok=True)

dataset_list = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676"
]

cluster_map = {
    "151507": 7, "151508": 7, "151509": 7, "151510": 7,
    "151669": 5, "151670": 5, "151671": 5, "151672": 5,
    "151673": 7, "151674": 7, "151675": 7, "151676": 7,
}

radius = 50


# =========================================================
# 4. 公共参数（除 w_smooth 和 w_sharpen 外全部固定）
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
# 5. 候选池构建：只保留 smooth + sharpen 两两组合
#    不把 baseline 放进候选池，但每个切片会先单独跑 baseline
# =========================================================
def fmt_num(x):
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    if s == "":
        s = "0"
    s = s.replace(".", "p")
    s = s.replace("-", "m")
    return s


def candidate_name(ws, hs):
    return f"WS_{fmt_num(ws)}_HS_{fmt_num(hs)}"


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


def add_candidate(w_smooth, w_sharpen):
    # 只允许真正的 hybrid 组合
    if w_smooth <= 0 or w_sharpen <= 0:
        return

    cand = dict(common_params)
    cand["w_smooth"] = float(w_smooth)
    cand["w_sharpen"] = float(w_sharpen)
    cand["name"] = candidate_name(float(w_smooth), float(w_sharpen))

    key = tuple(cand.get(k) for k in KEY_FIELDS)
    if key not in candidate_dict:
        candidate_dict[key] = cand


# 候选池：低值更密，中高值适度展开
smooth_pool = [
    0.01, 0.02, 0.03, 0.05, 0.08,
    0.10, 0.12, 0.15, 0.20, 0.30,
    0.40, 0.50, 0.70, 0.90,
    1.10, 1.30, 1.50
]

sharpen_pool = [
    0.01, 0.02, 0.03, 0.05, 0.08,
    0.10, 0.12, 0.15, 0.20, 0.30,
    0.40, 0.50, 0.70, 0.90,
    1.10, 1.30, 1.50,
]

for ws in smooth_pool:
    for hs in sharpen_pool:
        add_candidate(ws, hs)

candidates = list(candidate_dict.values())
print(f"Total hybrid-only candidates: {len(candidates)}")   # 20 x 20 = 400

candidate_table = pd.DataFrame(candidates)
candidate_table.to_csv(
    os.path.join(result_root, "CandidatePool_Definition.csv"),
    index=False
)
print("Saved candidate pool definition.")


# =========================================================
# 6. baseline 参数（每个切片单独先跑一次）
# =========================================================
baseline_candidate = dict(common_params)
baseline_candidate["w_smooth"] = 0.0
baseline_candidate["w_sharpen"] = 0.0
baseline_candidate["name"] = "Baseline_Internal"


# =========================================================
# 7. 工具函数
# =========================================================
def attach_ground_truth(adata, file_fold):
    df_meta = pd.read_csv(os.path.join(file_fold, 'metadata.tsv'), sep='\t')

    barcode_col = None
    for c in df_meta.columns:
        if 'barcode' in c.lower():
            barcode_col = c
            break

    if barcode_col is not None:
        df_meta = df_meta.set_index(barcode_col)
        df_meta = df_meta.loc[adata.obs_names]
        adata.obs['ground_truth'] = df_meta['layer_guess']
    else:
        if 'layer_guess' not in df_meta.columns:
            raise KeyError("metadata.tsv does not contain column 'layer_guess'.")
        adata.obs['ground_truth'] = df_meta['layer_guess'].values

    return adata


def run_one_candidate(adata_input, candidate, file_fold, n_clusters, radius, seed=41):
    cand_name = candidate["name"]
    print("\n" + "-" * 100)
    print(f"Running candidate: {cand_name}")
    print("-" * 100)

    adata = adata_input.copy()
    graphst_params = {k: v for k, v in candidate.items() if k != "name"}

    model = GraphST(adata, **graphst_params)
    adata = model.train()

    if 'emb' not in adata.obsm:
        raise KeyError("adata.obsm['emb'] not found after training.")

    n_pcs = min(20, adata.obsm['emb'].shape[1])
    adata.obsm['emb_pca'] = PCA(n_components=n_pcs, random_state=seed).fit_transform(adata.obsm['emb'])

    clustering(adata, n_clusters, radius=radius, method='mclust', refinement=True)

    adata = attach_ground_truth(adata, file_fold)
    adata_eval = adata[~pd.isnull(adata.obs['ground_truth'])].copy()

    ari = metrics.adjusted_rand_score(adata_eval.obs['domain'], adata_eval.obs['ground_truth'])
    nmi = metrics.normalized_mutual_info_score(adata_eval.obs['domain'], adata_eval.obs['ground_truth'])

    print(f"Candidate: {cand_name}")
    print(f"ARI: {ari:.6f}")
    print(f"NMI: {nmi:.6f}")

    return ari, nmi


def build_record(candidate, dataset, ari, nmi, baseline_ari=None, baseline_nmi=None, error_msg=None, is_baseline=False):
    record = dict(candidate)
    record["dataset"] = dataset
    record["ARI"] = ari
    record["NMI"] = nmi
    record["Error"] = error_msg
    record["is_baseline"] = bool(is_baseline)

    if baseline_ari is None or baseline_nmi is None or pd.isna(ari) or pd.isna(nmi):
        record["baseline_ARI"] = baseline_ari
        record["baseline_NMI"] = baseline_nmi
        record["delta_ARI"] = np.nan
        record["delta_NMI"] = np.nan
        record["delta_sum"] = np.nan
        record["better_than_baseline_ARI"] = np.nan
        record["better_than_baseline_NMI"] = np.nan
        record["num_metrics_better"] = np.nan
    else:
        record["baseline_ARI"] = baseline_ari
        record["baseline_NMI"] = baseline_nmi
        record["delta_ARI"] = ari - baseline_ari
        record["delta_NMI"] = nmi - baseline_nmi
        record["delta_sum"] = record["delta_ARI"] + record["delta_NMI"]
        record["better_than_baseline_ARI"] = int(ari > baseline_ari)
        record["better_than_baseline_NMI"] = int(nmi > baseline_nmi)
        record["num_metrics_better"] = int(record["better_than_baseline_ARI"] + record["better_than_baseline_NMI"])

    return record


def select_top_vs_baseline(dataset_df, atol=1e-12):
    """
    规则：
    1) 先选 num_metrics_better 最大
    2) 再选 delta_sum 最大
    3) 如果仍然并列，则全部保留
    """
    sub = dataset_df[
        (dataset_df["is_baseline"] == False) &
        dataset_df["ARI"].notna() &
        dataset_df["NMI"].notna() &
        dataset_df["num_metrics_better"].notna()
    ].copy()

    if sub.empty:
        return sub

    max_num_metrics_better = sub["num_metrics_better"].max()
    sub = sub[sub["num_metrics_better"] == max_num_metrics_better].copy()

    max_delta_sum = sub["delta_sum"].max()
    sub = sub[np.isclose(sub["delta_sum"], max_delta_sum, atol=atol)].copy()

    sub = sub.sort_values(
        by=["num_metrics_better", "delta_sum", "delta_ARI", "delta_NMI", "name"],
        ascending=[False, False, False, False, True]
    ).reset_index(drop=True)

    return sub


# =========================================================
# 8. 批量跑全部数据集
# =========================================================
all_records = []
summary_records = []
top_vs_baseline_all = []

for dataset in dataset_list:
    print("\n" + "=" * 120)
    print(f"START DATASET: {dataset}")
    print("=" * 120)

    file_fold = os.path.join(data_root, dataset)
    n_clusters = cluster_map.get(dataset, 7)

    dataset_save_dir = os.path.join(result_root, f"DLPFC_{dataset}")
    os.makedirs(dataset_save_dir, exist_ok=True)

    try:
        adata_raw = sc.read_visium(
            file_fold,
            count_file='filtered_feature_bc_matrix.h5',
            load_images=True
        )
        adata_raw.var_names_make_unique()
    except Exception as e:
        print(f"Failed to load dataset {dataset}: {e}")

        for cand in candidates:
            record = build_record(
                candidate=cand,
                dataset=dataset,
                ari=np.nan,
                nmi=np.nan,
                baseline_ari=np.nan,
                baseline_nmi=np.nan,
                error_msg=f"load_failed: {e}",
                is_baseline=False
            )
            all_records.append(record)

        summary_records.append({
            "dataset": dataset,
            "baseline_ARI": np.nan,
            "baseline_NMI": np.nan,
            "best_name_by_ARI": None,
            "best_ARI": np.nan,
            "best_NMI_of_best_ARI": np.nan,
            "top_vs_baseline_names": None,
            "top_vs_baseline_num": 0,
            "status": f"load_failed: {e}"
        })
        continue

    dataset_records = []

    # -----------------------------------------------------
    # 8.1 先跑 baseline
    # -----------------------------------------------------
    try:
        baseline_ari, baseline_nmi = run_one_candidate(
            adata_input=adata_raw,
            candidate=baseline_candidate,
            file_fold=file_fold,
            n_clusters=n_clusters,
            radius=radius,
            seed=seed
        )

        baseline_record = build_record(
            candidate=baseline_candidate,
            dataset=dataset,
            ari=baseline_ari,
            nmi=baseline_nmi,
            baseline_ari=baseline_ari,
            baseline_nmi=baseline_nmi,
            error_msg=None,
            is_baseline=True
        )
        dataset_records.append(baseline_record)
        all_records.append(baseline_record)

    except Exception as e:
        print(f"Baseline failed on {dataset}: {e}")

        baseline_record = build_record(
            candidate=baseline_candidate,
            dataset=dataset,
            ari=np.nan,
            nmi=np.nan,
            baseline_ari=np.nan,
            baseline_nmi=np.nan,
            error_msg=f"baseline_failed: {e}",
            is_baseline=True
        )
        dataset_records.append(baseline_record)
        all_records.append(baseline_record)

        # baseline 都失败了，则该切片的 hybrid 不再比较
        for cand in candidates:
            record = build_record(
                candidate=cand,
                dataset=dataset,
                ari=np.nan,
                nmi=np.nan,
                baseline_ari=np.nan,
                baseline_nmi=np.nan,
                error_msg=f"skipped_due_to_baseline_failed: {e}",
                is_baseline=False
            )
            dataset_records.append(record)
            all_records.append(record)

        dataset_df = pd.DataFrame(dataset_records)
        dataset_csv = os.path.join(dataset_save_dir, f"{dataset}_all_candidate_results.csv")
        dataset_df.to_csv(dataset_csv, index=False)

        summary_records.append({
            "dataset": dataset,
            "baseline_ARI": np.nan,
            "baseline_NMI": np.nan,
            "best_name_by_ARI": None,
            "best_ARI": np.nan,
            "best_NMI_of_best_ARI": np.nan,
            "top_vs_baseline_names": None,
            "top_vs_baseline_num": 0,
            "status": f"baseline_failed: {e}"
        })
        continue

    # -----------------------------------------------------
    # 8.2 再跑 hybrid-only candidates
    # -----------------------------------------------------
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

            record = build_record(
                candidate=candidate,
                dataset=dataset,
                ari=ari,
                nmi=nmi,
                baseline_ari=baseline_ari,
                baseline_nmi=baseline_nmi,
                error_msg=None,
                is_baseline=False
            )
            dataset_records.append(record)
            all_records.append(record)

            if ari > best_ari:
                best_ari = ari
                best_nmi = nmi
                best_name = candidate["name"]

        except Exception as e:
            print(f"Candidate [{candidate['name']}] failed on {dataset}: {e}")
            record = build_record(
                candidate=candidate,
                dataset=dataset,
                ari=np.nan,
                nmi=np.nan,
                baseline_ari=baseline_ari,
                baseline_nmi=baseline_nmi,
                error_msg=str(e),
                is_baseline=False
            )
            dataset_records.append(record)
            all_records.append(record)
            continue

    # -----------------------------------------------------
    # 8.3 保存每个切片的完整结果
    # -----------------------------------------------------
    dataset_df = pd.DataFrame(dataset_records)
    dataset_df = dataset_df.sort_values(
        by=["is_baseline", "name"],
        ascending=[False, True]
    ).reset_index(drop=True)

    dataset_csv = os.path.join(dataset_save_dir, f"{dataset}_all_candidate_results.csv")
    dataset_df.to_csv(dataset_csv, index=False)

    # -----------------------------------------------------
    # 8.4 自动筛选“超过 baseline 最多”的参数（并列全部保留）
    # -----------------------------------------------------
    top_vs_baseline_df = select_top_vs_baseline(dataset_df)

    top_csv = os.path.join(dataset_save_dir, f"{dataset}_top_vs_baseline_candidates.csv")
    top_vs_baseline_df.to_csv(top_csv, index=False)

    if not top_vs_baseline_df.empty:
        top_names = "; ".join(top_vs_baseline_df["name"].tolist())
        top_num = len(top_vs_baseline_df)

        tmp = top_vs_baseline_df.copy()
        tmp["selected_dataset"] = dataset
        top_vs_baseline_all.append(tmp)
    else:
        top_names = None
        top_num = 0

    # -----------------------------------------------------
    # 8.5 每个切片 summary
    # -----------------------------------------------------
    summary_records.append({
        "dataset": dataset,
        "baseline_ARI": baseline_ari,
        "baseline_NMI": baseline_nmi,
        "best_name_by_ARI": best_name,
        "best_ARI": best_ari if best_ari >= 0 else np.nan,
        "best_NMI_of_best_ARI": best_nmi if best_nmi >= 0 else np.nan,
        "top_vs_baseline_names": top_names,
        "top_vs_baseline_num": top_num,
        "status": "success"
    })

    print("\n" + "-" * 100)
    print(f"BASELINE FOR {dataset}")
    print(f"Baseline ARI: {baseline_ari:.6f}")
    print(f"Baseline NMI: {baseline_nmi:.6f}")
    print("-" * 100)

    print(f"BEST RESULT BY ARI FOR {dataset}")
    print(f"Best candidate: {best_name}")
    print(f"Best ARI: {best_ari:.6f}")
    print(f"Best NMI: {best_nmi:.6f}")
    print("-" * 100)

    if not top_vs_baseline_df.empty:
        print(f"TOP VS BASELINE ({dataset})")
        print(top_vs_baseline_df[[
            "name", "ARI", "NMI", "baseline_ARI", "baseline_NMI",
            "delta_ARI", "delta_NMI", "delta_sum", "num_metrics_better"
        ]])
        print("-" * 100)


# =========================================================
# 9. 保存全局 CSV
# =========================================================
all_results_df = pd.DataFrame(all_records)
all_results_csv = os.path.join(result_root, "All_DLPFC_AllCandidate_LongResults.csv")
all_results_df.to_csv(all_results_csv, index=False)

# ARI 宽表（不含 baseline）
hybrid_df = all_results_df[all_results_df["is_baseline"] == False].copy()

ari_wide = hybrid_df.pivot(index="dataset", columns="name", values="ARI").reset_index()
ari_wide_csv = os.path.join(result_root, "All_DLPFC_ARI_Wide.csv")
ari_wide.to_csv(ari_wide_csv, index=False)

nmi_wide = hybrid_df.pivot(index="dataset", columns="name", values="NMI").reset_index()
nmi_wide_csv = os.path.join(result_root, "All_DLPFC_NMI_Wide.csv")
nmi_wide.to_csv(nmi_wide_csv, index=False)

# baseline 宽表
baseline_df = all_results_df[all_results_df["is_baseline"] == True][
    ["dataset", "ARI", "NMI"]
].rename(columns={"ARI": "baseline_ARI", "NMI": "baseline_NMI"})
baseline_csv = os.path.join(result_root, "All_DLPFC_Baseline_PerDataset.csv")
baseline_df.to_csv(baseline_csv, index=False)

# 每个参数的全局汇总（重点看超过 baseline 的次数）
candidate_summary_records = []
for cand_name, subdf in hybrid_df.groupby("name"):
    candidate_summary_records.append({
        "name": cand_name,
        "mean_ARI": subdf["ARI"].mean(skipna=True),
        "median_ARI": subdf["ARI"].median(skipna=True),
        "std_ARI": subdf["ARI"].std(skipna=True),
        "mean_NMI": subdf["NMI"].mean(skipna=True),
        "median_NMI": subdf["NMI"].median(skipna=True),
        "std_NMI": subdf["NMI"].std(skipna=True),
        "mean_delta_ARI": subdf["delta_ARI"].mean(skipna=True),
        "mean_delta_NMI": subdf["delta_NMI"].mean(skipna=True),
        "mean_delta_sum": subdf["delta_sum"].mean(skipna=True),
        "num_valid_datasets": int(subdf["ARI"].notna().sum()),
        "num_better_than_baseline_ARI": int((subdf["better_than_baseline_ARI"] == 1).sum()),
        "num_better_than_baseline_NMI": int((subdf["better_than_baseline_NMI"] == 1).sum()),
        "num_better_than_baseline_both": int((subdf["num_metrics_better"] == 2).sum()),
        "num_better_than_baseline_any": int((subdf["num_metrics_better"] >= 1).sum()),
    })

candidate_summary_df = pd.DataFrame(candidate_summary_records).sort_values(
    by=[
        "num_better_than_baseline_both",
        "num_better_than_baseline_any",
        "mean_delta_sum",
        "mean_ARI"
    ],
    ascending=[False, False, False, False]
)
candidate_summary_csv = os.path.join(result_root, "All_DLPFC_Candidate_Summary.csv")
candidate_summary_df.to_csv(candidate_summary_csv, index=False)

# 每个切片“超过 baseline 最多”的参数汇总
if len(top_vs_baseline_all) > 0:
    top_vs_baseline_all_df = pd.concat(top_vs_baseline_all, axis=0, ignore_index=True)
else:
    top_vs_baseline_all_df = pd.DataFrame()

top_vs_baseline_all_csv = os.path.join(result_root, "All_DLPFC_TopVsBaseline_PerDataset.csv")
top_vs_baseline_all_df.to_csv(top_vs_baseline_all_csv, index=False)

# 每个数据集 summary
best_summary_df = pd.DataFrame(summary_records)
best_summary_csv = os.path.join(result_root, "All_DLPFC_Best_Summary.csv")
best_summary_df.to_csv(best_summary_csv, index=False)

print("\n" + "=" * 120)
print("ALL DATASETS FINISHED")
print("=" * 120)
print(best_summary_df)

print(f"\nSaved candidate definition to: {os.path.join(result_root, 'CandidatePool_Definition.csv')}")
print(f"Saved long results to: {all_results_csv}")
print(f"Saved ARI wide table to: {ari_wide_csv}")
print(f"Saved NMI wide table to: {nmi_wide_csv}")
print(f"Saved baseline table to: {baseline_csv}")
print(f"Saved candidate summary to: {candidate_summary_csv}")
print(f"Saved per-dataset top-vs-baseline table to: {top_vs_baseline_all_csv}")
print(f"Saved best summary to: {best_summary_csv}")