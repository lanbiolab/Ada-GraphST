import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc

os.environ["R_HOME"] = "/data2/liangyefeng/miniconda3/envs/graphst/lib/R"

project_root = "/data2/liangyefeng/My_GraphST_Innovation"
sys.path.insert(0, os.path.join(project_root, "src"))

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from STAGATE.STAGATE.Train_STAGATE import train_STAGATE
from STAGATE.STAGATE.utils import Cal_Spatial_Net

from mouse_embryo_E16E2_common import (
    DATA_PATHS,
    DATA_DIR,
    FORCE_LABEL_COL,
    SPATIAL_BIN_SIZE,
    SEED,
    seed_everything,
    load_binned_section,
    compute_ari_nmi,
    mclust_R,
)

# =========================
# 手动追加新切片
# =========================
NEW_DATA_PATH = "/data2/liangyefeng/My_GraphST_Innovation/data/Mouse-embryo/E16.5_E2S11.MOSTA.h5ad"

ALL_DATA_PATHS = list(DATA_PATHS)
if NEW_DATA_PATH not in ALL_DATA_PATHS:
    ALL_DATA_PATHS.append(NEW_DATA_PATH)

ALL_DATA_PATHS = sorted(ALL_DATA_PATHS)

RESULT_CSV = os.path.join(DATA_DIR, "STAGATE_E16.5_E2S1_S5_results.csv")

# 对 bin 后坐标更稳
GRAPH_MODEL = "KNN"
K_CUTOFF = 10

HIDDEN_DIMS = [512, 30]
ALPHA = 0
N_EPOCHS = 500
LR = 1e-4
WEIGHT_DECAY = 1e-4


def preprocess_for_stagate(adata, n_top_genes=3000):
    adata = adata.copy()
    adata.var_names_make_unique()

    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat",
        n_top_genes=min(n_top_genes, adata.n_vars)
    )
    return adata


seed_everything(SEED, use_tf=True)
results = []

for data_path in ALL_DATA_PATHS:
    sample_name = os.path.basename(data_path).replace(".MOSTA.h5ad", "")
    print("\n" + "=" * 100)
    print(f"START SAMPLE: {sample_name}")
    print("=" * 100)

    try:
        seed_everything(SEED, use_tf=True)

        adata, info = load_binned_section(
            data_path,
            force_label_col=FORCE_LABEL_COL,
            bin_size=SPATIAL_BIN_SIZE
        )

        print(f"[{sample_name}] n_obs={adata.n_obs}, n_vars={adata.n_vars}, n_clusters={info['n_clusters']}")

        adata = preprocess_for_stagate(adata, n_top_genes=3000)

        Cal_Spatial_Net(adata, k_cutoff=K_CUTOFF, model=GRAPH_MODEL, verbose=True)

        tf.reset_default_graph()
        adata = train_STAGATE(
            adata,
            hidden_dims=HIDDEN_DIMS,
            alpha=ALPHA,
            n_epochs=N_EPOCHS,
            lr=LR,
            key_added="STAGATE",
            gradient_clipping=5,
            nonlinear=True,
            weight_decay=WEIGHT_DECAY,
            verbose=True,
            random_seed=SEED,
            save_attention=False,
            save_loss=False,
            save_reconstrction=False,
        )

        adata = mclust_R(
            adata,
            num_cluster=info["n_clusters"],
            used_obsm="STAGATE",
            key_added="domain",
            random_seed=SEED
        )

        ari, nmi, n_eval = compute_ari_nmi(adata, pred_col="domain", gt_col="ground_truth")
        print(f"[{sample_name}] STAGATE | ARI={ari:.6f} | NMI={nmi:.6f}")

        results.append({
            "Sample": sample_name,
            "Method": "STAGATE",
            "Seed": SEED,
            "Bin_Size": SPATIAL_BIN_SIZE,
            "Graph_Model": GRAPH_MODEL,
            "K_Cutoff": K_CUTOFF,
            "Label_Col": info["label_col"],
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
            "Method": "STAGATE",
            "Seed": SEED,
            "Bin_Size": SPATIAL_BIN_SIZE,
            "Graph_Model": GRAPH_MODEL,
            "K_Cutoff": K_CUTOFF,
            "Label_Col": FORCE_LABEL_COL,
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