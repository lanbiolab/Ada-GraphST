import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import scipy.sparse as sp
import networkx as nx
from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score

warnings.filterwarnings("ignore")

# networkx 3.x compatibility
if not hasattr(nx, "to_scipy_sparse_matrix"):
    def _to_scipy_sparse_matrix(*args, **kwargs):
        arr = nx.to_scipy_sparse_array(*args, **kwargs)
        return sp.csr_matrix(arr)
    nx.to_scipy_sparse_matrix = _to_scipy_sparse_matrix

os.environ["R_HOME"] = "/data2/liangyefeng/miniconda3/envs/graphst/lib/R"

project_root = "/data2/liangyefeng/My_GraphST_Innovation"
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "src", "SpaceFlow"))

from SpaceFlow.SpaceFlow import SpaceFlow
import GraphST.utils as graphst_utils
from GraphST.utils import clustering

def reset_seed(seed=41):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# patched preprocessing
def patched_spaceflow_preprocessing_data(self, n_top_genes=None, n_neighbors=10):
    adata = self.adata
    adata = adata.copy()
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

def patched_mclust_R(adata, num_cluster, modelNames="EEE", used_obsm="emb_pca", random_seed=2020):
    import numpy as np
    import pandas as pd
    import rpy2.robjects as robjects
    from rpy2.robjects import r
    from rpy2.robjects import numpy2ri

    x = np.asarray(adata.obsm[used_obsm], dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"{used_obsm} must be 2D, but got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{used_obsm} contains NaN/Inf")

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
        res <- tryCatch(Mclust(x_mat, G=n_cluster, modelNames=model_name), error=function(e) NULL)
        if (is.null(res)) {
          cls <- rep(NA, nrow(x_mat))
        } else {
          cls <- res$classification
          if (is.null(cls)) cls <- rep(NA, nrow(x_mat))
        }
        """
    )

    mclust_res = np.array(r["cls"], dtype=object)
    cls_series = pd.Series(mclust_res, index=adata.obs_names)
    cls_series = pd.to_numeric(cls_series, errors="coerce")
    if cls_series.isna().sum() > 0:
        raise ValueError("mclust returned invalid labels.")
    adata.obs["mclust"] = cls_series.astype(int).astype("category")
    return adata

graphst_utils.mclust_R = patched_mclust_R

def detect_label_col(df):
    candidates = [c for c in df.columns if any(k in c.lower() for k in [
        "label", "annotation", "region", "cluster", "layer", "ground", "gt", "class"
    ])]
    return candidates[0] if len(candidates) > 0 else df.columns[-1]

def load_gt(file_fold, obs_names):
    gt_dir = os.path.join(file_fold, "gt")
    gold_meta_path = os.path.join(gt_dir, "gold_metadata.tsv")
    tissue_gt_path = os.path.join(gt_dir, "tissue_positions_list_GTs.txt")

    if os.path.exists(gold_meta_path):
        df = pd.read_csv(gold_meta_path, sep="\t")
        barcode_col = next((c for c in df.columns if "barcode" in c.lower()), None)
        label_col = detect_label_col(df)
        if barcode_col is not None:
            df = df.set_index(barcode_col)
            df = df.loc[obs_names]
            gt = df[label_col]
        else:
            gt = pd.Series(df[label_col].values, index=obs_names)
        return gt, gold_meta_path, label_col

    if os.path.exists(tissue_gt_path):
        try:
            df = pd.read_csv(tissue_gt_path, sep="\t")
            if df.shape[1] == 1:
                df = pd.read_csv(tissue_gt_path, sep=",")
        except Exception:
            df = pd.read_csv(tissue_gt_path, sep=",")
        barcode_col = next((c for c in df.columns if "barcode" in c.lower()), None)
        label_col = detect_label_col(df)
        if barcode_col is not None:
            df = df.set_index(barcode_col)
            df = df.loc[obs_names]
            gt = df[label_col]
        else:
            gt = pd.Series(df[label_col].values, index=obs_names)
        return gt, tissue_gt_path, label_col

    raise FileNotFoundError("No GT file found.")

seed = 41
reset_seed(seed)

file_fold = "/data2/liangyefeng/My_GraphST_Innovation/data/Human-Breast-Cancer-Block-A/section1"

adata = sc.read_visium(
    file_fold,
    count_file="section1_filtered_feature_bc_matrix.h5",
    load_images=True
)
adata.var_names_make_unique()

gt, gt_path, label_col = load_gt(file_fold, adata.obs_names)
adata.obs["ground_truth"] = gt
adata_eval = adata[~pd.isnull(adata.obs["ground_truth"])].copy()
n_clusters = adata_eval.obs["ground_truth"].nunique()

print("✅ GT file:", gt_path)
print("✅ label column:", label_col)
print("✅ n_clusters:", n_clusters)

sf = SpaceFlow(adata=adata.copy())

sf.preprocessing_data(
    n_top_genes=min(3000, adata.n_vars),
    n_neighbors=10
)

emb_dir = os.path.join(project_root, "results_spaceflow_embeddings_hbca")
os.makedirs(emb_dir, exist_ok=True)
emb_path = os.path.join(emb_dir, "section1_embedding.tsv")

embedding = sf.train(
    embedding_save_filepath=emb_path,
    spatial_regularization_strength=0.1,
    z_dim=50,
    lr=1e-3,
    epochs=1000,
    max_patience=50,
    min_stop=100,
    random_seed=seed,
    gpu=0 if torch.cuda.is_available() else None,
    regularization_acceleration=True,
    edge_subset_sz=1000000
)

adata.obsm["SpaceFlow"] = embedding
adata.obsm["emb"] = adata.obsm["SpaceFlow"].copy()

clustering(
    adata,
    n_clusters,
    radius=50,
    method="mclust",
    refinement=True
)

adata_eval = adata[~pd.isnull(adata.obs["ground_truth"])].copy()

ari = metrics.adjusted_rand_score(adata_eval.obs["domain"], adata_eval.obs["ground_truth"])
nmi = normalized_mutual_info_score(adata_eval.obs["domain"], adata_eval.obs["ground_truth"])

print(f"\n🎉 SpaceFlow section1 | ARI: {ari:.6f} | NMI: {nmi:.6f}")

results_df = pd.DataFrame([{
    "Dataset": "Human-Breast-Cancer-Block-A_section1",
    "Method": "SpaceFlow",
    "Seed": seed,
    "N_Clusters": n_clusters,
    "N_Obs_Eval": adata_eval.n_obs,
    "ARI": ari,
    "NMI": nmi,
}])

out_csv = os.path.join(file_fold, "SpaceFlow_section1_results.csv")
results_df.to_csv(out_csv, index=False)
print(f"💾 Saved to: {out_csv}")