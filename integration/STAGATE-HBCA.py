import os
import sys
import warnings
warnings.filterwarnings("ignore")

os.environ["R_HOME"] = "/data2/liangyefeng/miniconda3/envs/graphst/lib/R"

project_root = "/data2/liangyefeng/My_GraphST_Innovation"
sys.path.insert(0, os.path.join(project_root, "src"))

import pandas as pd
import scanpy as sc
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from STAGATE.STAGATE.Train_STAGATE import train_STAGATE
from STAGATE.STAGATE.utils import Cal_Spatial_Net

from hbca_section1_section2_integration_common import (
    reset_seed,
    load_integrated_dataset,
    to_counts_adata,
    infer_n_clusters,
    compute_ilisi_from_rep,
    mclust_R,
    plot_umap_batch_domain,
)

DATA_ROOT = "/data2/liangyefeng/My_GraphST_Innovation/data/Human-Breast-Cancer-Block-A"
ALIGNED_H5AD = os.path.join(
    DATA_ROOT,
    "PASTE_then_GraphST_section1_section2",
   "HBCA_section1_section2_PASTE_aligned_for_GraphST.h5ad"
)

SAVE_DIR = os.path.join(DATA_ROOT, "PASTE_then_STAGATE_section1_section2")
os.makedirs(SAVE_DIR, exist_ok=True)

RESULT_H5AD = os.path.join(SAVE_DIR, "HBCA_section1_section2_STAGATE_result.h5ad")
RESULT_CSV = os.path.join(SAVE_DIR, "HBCA_section1_section2_STAGATE_iLISI_summary.csv")
UMAP_PNG = os.path.join(SAVE_DIR, "HBCA_section1_section2_STAGATE_UMAP.png")
UMAP_PDF = os.path.join(SAVE_DIR, "HBCA_section1_section2_STAGATE_UMAP.pdf")

SEED = 41

HIDDEN_DIMS = [512, 30]
ALPHA = 0
N_EPOCHS = 500
LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAPH_MODEL = "KNN"
K_CUTOFF = 12


if __name__ == "__main__":
    reset_seed(SEED)
    tf.set_random_seed(SEED)

    adata = load_integrated_dataset(ALIGNED_H5AD)
    adata = to_counts_adata(adata)
    n_clusters, gt_col = infer_n_clusters(adata, default_n=20)

    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat",
        n_top_genes=min(3000, adata.n_vars)
    )
    adata = adata[:, adata.var["highly_variable"]].copy()

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

    ilisi = compute_ilisi_from_rep(adata, rep_key="STAGATE", seed=SEED)

    adata = mclust_R(
        adata,
        num_cluster=n_clusters,
        used_obsm="STAGATE",
        key_added="domain",
        random_seed=SEED
    )

    plot_umap_batch_domain(
        adata,
        method_name="STAGATE",
        ilisi=ilisi,
        save_png=UMAP_PNG,
        save_pdf=UMAP_PDF,
        seed=SEED
    )

    adata.write(RESULT_H5AD)

    pd.DataFrame([{
        "Dataset": "HBCA_section1_section2",
        "Method": "STAGATE",
        "Seed": SEED,
        "Input_H5AD": ALIGNED_H5AD,
        "Result_H5AD": RESULT_H5AD,
        "GT_Col_For_NClusters": gt_col,
        "Graph_Model": GRAPH_MODEL,
        "K_Cutoff": K_CUTOFF,
        "Hidden_Dims": str(HIDDEN_DIMS),
        "Alpha": ALPHA,
        "N_Epochs": N_EPOCHS,
        "LR": LR,
        "Weight_Decay": WEIGHT_DECAY,
        "N_Clusters_For_Domain": n_clusters,
        "N_Obs_Total": adata.n_obs,
        "N_Vars": adata.n_vars,
        "iLISI": ilisi,
    }]).to_csv(RESULT_CSV, index=False)

    print(f"STAGATE | iLISI={ilisi:.6f}")
    print("saved h5ad :", RESULT_H5AD)
    print("saved csv  :", RESULT_CSV)
    print("saved umap :", UMAP_PNG)
    print("saved umap :", UMAP_PDF)