import os
import sys
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")

os.environ["R_HOME"] = "/data2/liangyefeng/miniconda3/envs/graphst/lib/R"

project_root = "/data2/liangyefeng/My_GraphST_Innovation"
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "scripts", "Method"))

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from GraphST.GraphST import GraphST
import GraphST.utils as graphst_utils
from GraphST.utils import clustering

from mouse_brain_MA_common import (
    reset_seed,
    safe_mclust_R,
    load_ma_dataset,
    to_counts_adata,
    compute_ari_nmi,
)

RESULT_ROOT = "/data2/liangyefeng/My_GraphST_Innovation/Result_MA_GraphST"
SEED = 41
GRAPHST_REFINEMENT = False
RUN_PARAM_SEARCH = True   # False=只跑baseline；True=跑241候选池

os.makedirs(RESULT_ROOT, exist_ok=True)


# =========================================================
# patch mclust
# =========================================================
def patched_mclust_R(adata, num_cluster, modelNames="EEE", used_obsm="emb_pca", random_seed=2020):
    return safe_mclust_R(
        adata,
        num_cluster=num_cluster,
        used_obsm=used_obsm,
        key_added="mclust",
        modelNames=modelNames,
        random_seed=random_seed
    )

graphst_utils.mclust_R = patched_mclust_R


# =========================================================
# seed / device
# =========================================================
reset_seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)


# =========================================================
# GraphST common params
# =========================================================
common_params = {
    "device": device,
    "random_seed": SEED,
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


def fmt_num(x):
    return str(x).replace(".", "p").replace("-", "m")


# =========================================================
# candidate pool
# =========================================================
candidate_dict = OrderedDict()
KEY_FIELDS = [
    "w_smooth", "w_sharpen", "gamma", "warmup_epochs",
    "update_interval", "graph_update_rate", "graph_reg_weight",
    "use_learnable_proj"
]


def add_candidate(name, use_original_baseline=False, **updates):
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
        cand["use_original_baseline"] = use_original_baseline
        candidate_dict[key] = cand


# A. Baseline
add_candidate(
    "Baseline_Internal",
    use_original_baseline=True,
    w_smooth=0.0,
    w_sharpen=0.0,
    use_learnable_proj=False,
)

if RUN_PARAM_SEARCH:
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
    os.path.join(RESULT_ROOT, "CandidatePool_Definition.csv"),
    index=False
)
print("Saved candidate pool definition.")


# =========================================================
# load dataset FIRST
# =========================================================
adata_raw, label_col, n_clusters = load_ma_dataset()
adata_raw = to_counts_adata(adata_raw)

print("✅ Dataset ready")
print("✅ label_col:", label_col)
print("✅ n_clusters:", n_clusters)


# =========================================================
# run one candidate
# =========================================================
def run_one_candidate(adata_input, n_clusters, candidate):
    reset_seed(SEED)
    adata = adata_input.copy()

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

    if "emb" not in adata.obsm:
        raise KeyError("adata.obsm['emb'] not found after training.")

    n_pcs = min(20, adata.obsm["emb"].shape[1])
    adata.obsm["emb_pca"] = PCA(n_components=n_pcs, random_state=SEED).fit_transform(adata.obsm["emb"])

    clustering(
        adata,
        n_clusters,
        radius=50,
        method="mclust",
        refinement=GRAPHST_REFINEMENT
    )

    ari, nmi, n_eval = compute_ari_nmi(adata, pred_col="domain", gt_col="ground_truth")
    return ari, nmi, n_eval


# =========================================================
# main loop
# =========================================================
all_records = []
best_name, best_ari, best_nmi = None, -1.0, -1.0

for idx, candidate in enumerate(candidates, 1):
    print(f"\ncandidate {idx}/{len(candidates)} -> {candidate['name']}")
    try:
        ari, nmi, n_eval = run_one_candidate(adata_raw, n_clusters, candidate)

        record = dict(candidate)
        record.update({
            "dataset": "MA",
            "label_col": label_col,
            "N_Clusters": n_clusters,
            "N_Obs_Eval": n_eval,
            "ARI": ari,
            "NMI": nmi,
        })
        all_records.append(record)

        if (ari > best_ari) or (np.isclose(ari, best_ari) and nmi > best_nmi):
            best_name, best_ari, best_nmi = candidate["name"], ari, nmi

        print(f"ARI={ari:.6f} | NMI={nmi:.6f}")

    except Exception as e:
        print("failed:", e)
        record = dict(candidate)
        record.update({
            "dataset": "MA",
            "label_col": label_col,
            "N_Clusters": n_clusters,
            "ARI": np.nan,
            "NMI": np.nan,
            "Error": str(e),
        })
        all_records.append(record)

all_df = pd.DataFrame(all_records)
all_df.to_csv(os.path.join(RESULT_ROOT, "All_results.csv"), index=False)

best_df = pd.DataFrame([{
    "dataset": "MA",
    "label_col": label_col,
    "best_name": best_name,
    "best_ARI": best_ari if best_ari >= 0 else np.nan,
    "best_NMI": best_nmi if best_nmi >= 0 else np.nan,
}])
best_df.to_csv(os.path.join(RESULT_ROOT, "Best_summary.csv"), index=False)

print("\nBEST SUMMARY")
print(best_df.to_string(index=False))