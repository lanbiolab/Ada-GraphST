import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
import networkx as nx
from sklearn.decomposition import PCA

os.environ["R_HOME"] = "/data2/liangyefeng/miniconda3/envs/graphst/lib/R"

project_root = "/data2/liangyefeng/My_GraphST_Innovation"
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "src", "SpaceFlow"))

# NetworkX 3.x compatibility patch
if not hasattr(nx, "to_scipy_sparse_matrix"):
    def _to_scipy_sparse_matrix(G, nodelist=None, dtype=None, weight="weight", format="csr"):
        arr = nx.to_scipy_sparse_array(
            G,
            nodelist=nodelist,
            dtype=dtype,
            weight=weight,
            format=format,
        )
        if hasattr(arr, "asformat"):
            return arr.asformat(format)
        return sp.csr_matrix(arr)

    nx.to_scipy_sparse_matrix = _to_scipy_sparse_matrix

from SpaceFlow.SpaceFlow import SpaceFlow

import GraphST.utils as graphst_utils
from GraphST.utils import clustering

from mouse_embryo_E16E2_common import (
    DATA_PATHS,
    DATA_DIR,
    FORCE_LABEL_COL,
    SPATIAL_BIN_SIZE,
    SEED,
    seed_everything,
    load_binned_section,
    compute_ari_nmi,
    patch_graphst_mclust,
)

patch_graphst_mclust()

# =========================
# 手动追加新切片
# =========================
NEW_DATA_PATH = "/data2/liangyefeng/My_GraphST_Innovation/data/Mouse-embryo/E16.5_E2S11.MOSTA.h5ad"

ALL_DATA_PATHS = list(DATA_PATHS)
if NEW_DATA_PATH not in ALL_DATA_PATHS:
    ALL_DATA_PATHS.append(NEW_DATA_PATH)

ALL_DATA_PATHS = sorted(ALL_DATA_PATHS)

RESULT_CSV = os.path.join(DATA_DIR, "SpaceFlow_E16.5_E2S1_S4_results.csv")

N_TOP_GENES = 3000
SPATIAL_N_NEIGHBORS = 10
Z_DIM = 50
LR = 1e-3
EPOCHS = 1000
MAX_PATIENCE = 50
MIN_STOP = 100
SPATIAL_REG_STRENGTH = 0.1
GPU_ID = 0 if torch.cuda.is_available() else None
REFINE_RADIUS = 50

seed_everything(SEED)
results = []

emb_dir = os.path.join(DATA_DIR, "spaceflow_embeddings")
os.makedirs(emb_dir, exist_ok=True)

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

        sf = SpaceFlow(adata=adata.copy())

        sf.preprocessing_data(
            n_top_genes=min(N_TOP_GENES, adata.n_vars),
            n_neighbors=SPATIAL_N_NEIGHBORS
        )

        emb_path = os.path.join(emb_dir, f"{sample_name}_embedding.tsv")

        embedding = sf.train(
            embedding_save_filepath=emb_path,
            spatial_regularization_strength=SPATIAL_REG_STRENGTH,
            z_dim=Z_DIM,
            lr=LR,
            epochs=EPOCHS,
            max_patience=MAX_PATIENCE,
            min_stop=MIN_STOP,
            random_seed=SEED,
            gpu=GPU_ID,
            regularization_acceleration=True,
            edge_subset_sz=1000000
        )

        embedding = np.asarray(embedding)

        adata.obsm["SpaceFlow"] = embedding.copy()
        adata.obsm["emb"] = embedding.copy()

        n_pcs = min(20, adata.obsm["emb"].shape[1])
        adata.obsm["emb_pca"] = PCA(
            n_components=n_pcs,
            random_state=SEED
        ).fit_transform(adata.obsm["emb"])

        clustering(
            adata,
            info["n_clusters"],
            radius=REFINE_RADIUS,
            method="mclust",
            refinement=True
        )

        ari, nmi, n_eval = compute_ari_nmi(adata, pred_col="domain", gt_col="ground_truth")
        print(f"[{sample_name}] SpaceFlow | ARI={ari:.6f} | NMI={nmi:.6f}")

        results.append({
            "Sample": sample_name,
            "Method": "SpaceFlow",
            "Seed": SEED,
            "Bin_Size": SPATIAL_BIN_SIZE,
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
            "Method": "SpaceFlow",
            "Seed": SEED,
            "Bin_Size": SPATIAL_BIN_SIZE,
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