import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ["R_HOME"] = "/data2/liangyefeng/miniconda3/envs/graphst/lib/R"

import scanpy as sc
import torch
import scipy.sparse as sp
import networkx as nx

project_root = "/data2/liangyefeng/My_GraphST_Innovation"
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "src", "SpaceFlow"))

if not hasattr(nx, "to_scipy_sparse_matrix"):
    def _to_scipy_sparse_matrix(*args, **kwargs):
        arr = nx.to_scipy_sparse_array(*args, **kwargs)
        return sp.csr_matrix(arr)
    nx.to_scipy_sparse_matrix = _to_scipy_sparse_matrix

from SpaceFlow.SpaceFlow import SpaceFlow
from sshippo_binned_common import (
    reset_seed, safe_mclust_R, load_binned_h5ad, to_counts_adata, compute_ari_nmi
)

DATA_PATH = "/data2/liangyefeng/My_GraphST_Innovation/data/mouse_hyppocampus_slideseqv2/sshippo_binned.h5ad"
RESULT_CSV = "/data2/liangyefeng/My_GraphST_Innovation/data/mouse_hyppocampus_slideseqv2/SpaceFlow_sshippo_binned_results.csv"
EMB_DIR = "/data2/liangyefeng/My_GraphST_Innovation/results_spaceflow_sshippo_binned"
FORCE_LABEL_COL = "cluster"
SEED = 41

def patched_spaceflow_preprocessing_data(self, n_top_genes=None, n_neighbors=10):
    adata = self.adata.copy()
    adata.var_names_make_unique()

    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor="seurat",
        subset=True
    )
    sc.pp.pca(adata)

    spatial_locs = adata.obsm["spatial"]
    spatial_graph = self.graph_alpha(spatial_locs, n_neighbors=n_neighbors)

    self.adata_preprocessed = adata
    self.spatial_graph = spatial_graph

SpaceFlow.preprocessing_data = patched_spaceflow_preprocessing_data

reset_seed(SEED)
os.makedirs(EMB_DIR, exist_ok=True)

adata, label_col, n_clusters = load_binned_h5ad(DATA_PATH, force_label_col=FORCE_LABEL_COL)
adata = to_counts_adata(adata)

sf = SpaceFlow(adata=adata.copy())
sf.preprocessing_data(
    n_top_genes=min(2000, adata.n_vars),
    n_neighbors=10
)

emb_path = os.path.join(EMB_DIR, "sshippo_binned_embedding.tsv")
gpu_id = 0 if torch.cuda.is_available() else None

embedding = sf.train(
    embedding_save_filepath=emb_path,
    spatial_regularization_strength=0.1,
    z_dim=32,
    lr=1e-3,
    epochs=500,
    max_patience=50,
    min_stop=100,
    random_seed=SEED,
    gpu=gpu_id,
    regularization_acceleration=True,
    edge_subset_sz=1000000
)

adata.obsm["SpaceFlow"] = embedding
adata.obsm["X_emb"] = adata.obsm["SpaceFlow"].copy()

safe_mclust_R(adata, num_cluster=n_clusters, used_obsm="X_emb", key_added="domain", random_seed=SEED)
ari, nmi, n_eval = compute_ari_nmi(adata, pred_col="domain", gt_col="ground_truth")
print(f"SpaceFlow | ARI={ari:.6f} | NMI={nmi:.6f}")

import pandas as pd
pd.DataFrame([{
    "Dataset": "sshippo_binned",
    "Method": "SpaceFlow",
    "Seed": SEED,
    "Label_Col": label_col,
    "N_Clusters": n_clusters,
    "N_Obs_Eval": n_eval,
    "ARI": ari,
    "NMI": nmi,
}]).to_csv(RESULT_CSV, index=False)

print("saved:", RESULT_CSV)