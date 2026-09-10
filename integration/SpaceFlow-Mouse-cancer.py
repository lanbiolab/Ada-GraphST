import os
import sys
import warnings
warnings.filterwarnings("ignore")

os.environ["R_HOME"] = "/data2/liangyefeng/miniconda3/envs/graphst/lib/R"

import numpy as np
import pandas as pd
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

from mouse_breast_cancer_sample1_section1_section2_integration_common import (
    reset_seed,
    load_integrated_dataset,
    to_counts_adata,
    infer_n_clusters,
    compute_ilisi_from_rep,
    mclust_R,
    plot_umap_batch_domain,
)

DATA_ROOT = "/data2/liangyefeng/My_GraphST_Innovation/data/7.Mouse_Breast_Cancer_Sample_1"
ALIGNED_H5AD = os.path.join(
    DATA_ROOT,
    "mouse_breast_cancer_sample1_section1&2.h5ad"
)

SAVE_DIR = os.path.join(DATA_ROOT, "PASTE_then_SpaceFlow_section1_section2")
os.makedirs(SAVE_DIR, exist_ok=True)

RESULT_H5AD = os.path.join(SAVE_DIR, "MouseBreastCancerSample1_section1_section2_SpaceFlow_result.h5ad")
RESULT_CSV = os.path.join(SAVE_DIR, "MouseBreastCancerSample1_section1_section2_SpaceFlow_iLISI_summary.csv")
UMAP_PNG = os.path.join(SAVE_DIR, "MouseBreastCancerSample1_section1_section2_SpaceFlow_UMAP.png")
UMAP_PDF = os.path.join(SAVE_DIR, "MouseBreastCancerSample1_section1_section2_SpaceFlow_UMAP.pdf")
EMB_DIR = os.path.join(SAVE_DIR, "embeddings")
os.makedirs(EMB_DIR, exist_ok=True)

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


if __name__ == "__main__":
    reset_seed(SEED)

    adata = load_integrated_dataset(ALIGNED_H5AD)
    adata = to_counts_adata(adata)
    n_clusters, gt_col = infer_n_clusters(adata, default_n=20)

    sf = SpaceFlow(adata=adata.copy())
    sf.preprocessing_data(
        n_top_genes=min(3000, adata.n_vars),
        n_neighbors=10
    )

    emb_path = os.path.join(EMB_DIR, "MouseBreastCancerSample1_section1_section2_embedding.tsv")
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

    embedding = np.asarray(embedding, dtype=np.float64)
    if embedding.shape[0] != adata.n_obs:
        raise ValueError(f"❌ SpaceFlow embedding 行数异常: {embedding.shape} vs n_obs={adata.n_obs}")

    adata.obsm["SpaceFlow"] = embedding
    adata.obsm["X_emb"] = embedding.copy()

    ilisi = compute_ilisi_from_rep(adata, rep_key="SpaceFlow", seed=SEED)

    adata = mclust_R(
        adata,
        num_cluster=n_clusters,
        used_obsm="SpaceFlow",
        key_added="domain",
        random_seed=SEED
    )

    plot_umap_batch_domain(
        adata,
        method_name="SpaceFlow",
        ilisi=ilisi,
        save_png=UMAP_PNG,
        save_pdf=UMAP_PDF,
        seed=SEED
    )

    adata.write(RESULT_H5AD)

    pd.DataFrame([{
        "Dataset": "MouseBreastCancerSample1_section1_section2",
        "Method": "SpaceFlow",
        "Seed": SEED,
        "Input_H5AD": ALIGNED_H5AD,
        "Result_H5AD": RESULT_H5AD,
        "GT_Col_For_NClusters": gt_col,
        "Embedding_File": emb_path,
        "N_Clusters_For_Domain": n_clusters,
        "N_Obs_Total": adata.n_obs,
        "N_Vars": adata.n_vars,
        "iLISI": ilisi,
    }]).to_csv(RESULT_CSV, index=False)

    print(f"SpaceFlow | iLISI={ilisi:.6f}")
    print("saved h5ad :", RESULT_H5AD)
    print("saved csv  :", RESULT_CSV)
    print("saved umap :", UMAP_PNG)
    print("saved umap :", UMAP_PDF)