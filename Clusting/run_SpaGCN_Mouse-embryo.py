import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc

project_root = "/data2/liangyefeng/My_GraphST_Innovation"
sys.path.insert(0, os.path.join(project_root, "src"))

from SpaGCN.SpaGCN_package.SpaGCN.SpaGCN import SpaGCN
from SpaGCN.SpaGCN_package.SpaGCN.util import (
    prefilter_genes, prefilter_specialgenes, search_l, search_res
)
from SpaGCN.SpaGCN_package.SpaGCN.calculate_adj import calculate_adj_matrix

from mouse_embryo_E16E2_common import (
    DATA_PATHS,
    DATA_DIR,
    FORCE_LABEL_COL,
    SPATIAL_BIN_SIZE,
    SEED,
    seed_everything,
    load_binned_section,
    compute_ari_nmi,
)

# =========================
# 手动追加新切片
# =========================
NEW_DATA_PATH = "/data2/liangyefeng/My_GraphST_Innovation/data/Mouse-embryo/E16.5_E2S11.MOSTA.h5ad"

# 去重，避免重复添加
ALL_DATA_PATHS = list(DATA_PATHS)
if NEW_DATA_PATH not in ALL_DATA_PATHS:
    ALL_DATA_PATHS.append(NEW_DATA_PATH)

# 按文件名排序，结果更整齐
ALL_DATA_PATHS = sorted(ALL_DATA_PATHS)

RESULT_CSV = os.path.join(DATA_DIR, "SpaGCN_E16.5_E2S1_S5_results.csv")

seed_everything(SEED)
results = []

for data_path in ALL_DATA_PATHS:
    sample_name = os.path.basename(data_path).replace(".MOSTA.h5ad", "")
    print("\n" + "=" * 100)
    print(f"START SAMPLE: {sample_name}")
    print("=" * 100)

    try:
        seed_everything(SEED)

        adata, info = load_binned_section(
            data_path,
            force_label_col=FORCE_LABEL_COL,
            bin_size=SPATIAL_BIN_SIZE
        )

        print(f"[{sample_name}] n_obs={adata.n_obs}, n_vars={adata.n_vars}, n_clusters={info['n_clusters']}")

        prefilter_genes(adata, min_cells=3)
        prefilter_specialgenes(adata)

        try:
            sc.pp.normalize_per_cell(adata)
        except Exception:
            sc.pp.normalize_total(adata)

        sc.pp.log1p(adata)

        x = adata.obsm["spatial"][:, 0].tolist()
        y = adata.obsm["spatial"][:, 1].tolist()
        adj = calculate_adj_matrix(x=x, y=y, histology=False)

        l = search_l(0.5, adj, start=0.01, end=1000, tol=0.01, max_run=100)
        if l is None:
            l = 1.0

        seed_everything(SEED)
        res = search_res(
            adata, adj, l, target_num=info["n_clusters"],
            start=0.7, step=0.1, tol=5e-3,
            lr=0.05, max_epochs=20,
            r_seed=SEED, t_seed=SEED, n_seed=SEED, max_run=20
        )
        if res is None:
            res = 1.0

        clf = SpaGCN()
        clf.set_l(l)

        seed_everything(SEED)
        clf.train(
            adata, adj,
            init_spa=True, init="louvain",
            res=res, tol=5e-3, lr=0.05, max_epochs=200
        )

        pred, _ = clf.predict()
        adata.obs["domain"] = pd.Categorical(np.asarray(pred).astype(int))

        ari, nmi, n_eval = compute_ari_nmi(adata, pred_col="domain", gt_col="ground_truth")
        print(f"[{sample_name}] SpaGCN | ARI={ari:.6f} | NMI={nmi:.6f}")

        results.append({
            "Sample": sample_name,
            "Method": "SpaGCN",
            "Seed": SEED,
            "Bin_Size": SPATIAL_BIN_SIZE,
            "Label_Col": info["label_col"],
            "Recommended_l": l,
            "Recommended_res": res,
            "N_Clusters": info["n_clusters"],
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
            "Method": "SpaGCN",
            "Seed": SEED,
            "Bin_Size": SPATIAL_BIN_SIZE,
            "Label_Col": FORCE_LABEL_COL,
            "Recommended_l": np.nan,
            "Recommended_res": np.nan,
            "N_Clusters": np.nan,
            "N_Obs": np.nan,
            "N_Obs_Eval": np.nan,
            "ARI": np.nan,
            "NMI": np.nan,
            "Status": "FAILED",
            "Error": str(e),
        })

results_df = pd.DataFrame(results)
results_df.to_csv(RESULT_CSV, index=False)

print("\nSaved:", RESULT_CSV)
print(results_df)