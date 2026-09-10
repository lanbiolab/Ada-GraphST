import os
import sys
import warnings
warnings.filterwarnings("ignore")

project_root = "/data2/liangyefeng/My_GraphST_Innovation"
sys.path.insert(0, os.path.join(project_root, "src"))

import pandas as pd
import scanpy as sc

from SpaGCN.SpaGCN_package.SpaGCN.SpaGCN import SpaGCN
from SpaGCN.SpaGCN_package.SpaGCN.util import (
    prefilter_genes, prefilter_specialgenes, search_l, search_res
)
from SpaGCN.SpaGCN_package.SpaGCN.calculate_adj import calculate_adj_matrix

from sshippo_binned_common import (
    reset_seed, load_binned_h5ad, to_counts_adata, compute_ari_nmi
)

DATA_PATH = "/data2/liangyefeng/My_GraphST_Innovation/data/mouse_hyppocampus_slideseqv2/sshippo_binned.h5ad"
RESULT_CSV = "/data2/liangyefeng/My_GraphST_Innovation/data/mouse_hyppocampus_slideseqv2/SpaGCN_sshippo_binned_results.csv"
FORCE_LABEL_COL = "cluster"
SEED = 41

reset_seed(SEED)

adata, label_col, n_clusters = load_binned_h5ad(DATA_PATH, force_label_col=FORCE_LABEL_COL)
adata = to_counts_adata(adata)

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
    raise RuntimeError("search_l failed.")

reset_seed(SEED)
res = search_res(
    adata, adj, l, target_num=n_clusters,
    start=0.7, step=0.1, tol=5e-3,
    lr=0.05, max_epochs=20,
    r_seed=SEED, t_seed=SEED, n_seed=SEED, max_run=20
)
if res is None:
    raise RuntimeError("search_res failed.")

clf = SpaGCN()
clf.set_l(l)

reset_seed(SEED)
clf.train(
    adata, adj,
    init_spa=True, init="louvain",
    res=res, tol=5e-3, lr=0.05, max_epochs=200
)

pred, prob = clf.predict()
adata.obs["domain"] = pd.Categorical(pred.astype(int))

ari, nmi, n_eval = compute_ari_nmi(adata, pred_col="domain", gt_col="ground_truth")
print(f"SpaGCN | ARI={ari:.6f} | NMI={nmi:.6f}")

pd.DataFrame([{
    "Dataset": "sshippo_binned",
    "Method": "SpaGCN",
    "Seed": SEED,
    "Label_Col": label_col,
    "Histology": False,
    "Recommended_l": l,
    "Recommended_res": res,
    "N_Clusters": n_clusters,
    "N_Obs_Eval": n_eval,
    "ARI": ari,
    "NMI": nmi,
}]).to_csv(RESULT_CSV, index=False)

print("saved:", RESULT_CSV)