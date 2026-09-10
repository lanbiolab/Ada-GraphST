import os
import sys
import warnings
warnings.filterwarnings("ignore")

os.environ["R_HOME"] = "/data2/liangyefeng/miniconda3/envs/graphst/lib/R"

project_root = "/data2/liangyefeng/My_GraphST_Innovation"
sys.path.insert(0, os.path.join(project_root, "src"))

import scanpy as sc
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from STAGATE.STAGATE.Train_STAGATE import train_STAGATE
from STAGATE.STAGATE.utils import Cal_Spatial_Net

from sshippo_binned_common import (
    reset_seed, safe_mclust_R, load_binned_h5ad, to_counts_adata, compute_ari_nmi
)

DATA_PATH = "/data2/liangyefeng/My_GraphST_Innovation/data/mouse_hyppocampus_slideseqv2/sshippo_binned.h5ad"
RESULT_CSV = "/data2/liangyefeng/My_GraphST_Innovation/data/mouse_hyppocampus_slideseqv2/STAGATE_sshippo_binned_results.csv"
FORCE_LABEL_COL = "cluster"
SEED = 41

HIDDEN_DIMS = [256, 30]
ALPHA = 0
N_EPOCHS = 500
LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAPH_MODEL = "KNN"
K_CUTOFF = 12

reset_seed(SEED)

adata, label_col, n_clusters = load_binned_h5ad(DATA_PATH, force_label_col=FORCE_LABEL_COL)
adata = to_counts_adata(adata)

sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=min(2000, adata.n_vars))
adata = adata[:, adata.var["highly_variable"]].copy()

Cal_Spatial_Net(adata, k_cutoff=K_CUTOFF, model="KNN", verbose=True)

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

safe_mclust_R(adata, num_cluster=n_clusters, used_obsm="STAGATE", key_added="domain", random_seed=SEED)
ari, nmi, n_eval = compute_ari_nmi(adata, pred_col="domain", gt_col="ground_truth")
print(f"STAGATE | ARI={ari:.6f} | NMI={nmi:.6f}")

import pandas as pd
pd.DataFrame([{
    "Dataset": "sshippo_binned",
    "Method": "STAGATE",
    "Seed": SEED,
    "Label_Col": label_col,
    "Graph_Model": GRAPH_MODEL,
    "K_Cutoff": K_CUTOFF,
    "Clustering_Head": "mclust",
    "N_Clusters": n_clusters,
    "N_Obs_Eval": n_eval,
    "ARI": ari,
    "NMI": nmi,
}]).to_csv(RESULT_CSV, index=False)

print("saved:", RESULT_CSV)