import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score

warnings.filterwarnings("ignore")

# =========================================================
# 0. 环境与路径
# =========================================================
os.environ["R_HOME"] = "/data2/liangyefeng/miniconda3/envs/graphst/lib/R"

project_root = "/data2/liangyefeng/My_GraphST_Innovation"
sys.path.insert(0, os.path.join(project_root, "src"))

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from STAGATE.STAGATE.Train_STAGATE import train_STAGATE
from STAGATE.STAGATE.utils import Cal_Spatial_Net

# =========================================================
# 1. 固定随机性
# =========================================================
def reset_seed(seed=41):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

# =========================================================
# 2. 更稳的 mclust
# =========================================================
def safe_mclust_R(adata, num_cluster, used_obsm="STAGATE", key_added="domain", modelNames="EEE", random_seed=41):
    import numpy as np
    import pandas as pd
    import rpy2.robjects as robjects
    from rpy2.robjects import r
    from rpy2.robjects import numpy2ri

    x = np.asarray(adata.obsm[used_obsm], dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"{used_obsm} must be 2D, but got shape {x.shape}")

    if not np.isfinite(x).all():
        bad_count = np.size(x) - np.isfinite(x).sum()
        raise ValueError(f"{used_obsm} contains NaN/Inf values, bad_count={bad_count}")

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

        res <- tryCatch(
          Mclust(x_mat, G=n_cluster, modelNames=model_name),
          error = function(e) NULL
        )

        if (is.null(res)) {
          cls <- rep(NA, nrow(x_mat))
        } else {
          cls <- res$classification
          if (is.null(cls)) {
            cls <- rep(NA, nrow(x_mat))
          }
        }
        """
    )

    mclust_res = np.array(r["cls"], dtype=object)
    cls_series = pd.Series(mclust_res, index=adata.obs_names)
    cls_series = pd.to_numeric(cls_series, errors="coerce")

    na_count = cls_series.isna().sum()
    if na_count > 0:
        raise ValueError(f"mclust returned {na_count} invalid labels for {used_obsm}.")

    adata.obs[key_added] = cls_series.astype(int).astype("category")
    return adata

# =========================================================
# 3. GT 读取
# =========================================================
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

# =========================================================
# 4. 配置
# =========================================================
seed = 41
reset_seed(seed)

file_fold = "/data2/liangyefeng/My_GraphST_Innovation/data/Human-Breast-Cancer-Block-A/section1"

HIDDEN_DIMS = [512, 30]
ALPHA = 0
N_EPOCHS = 500
LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAPH_MODEL = "Radius"
RAD_CUTOFF = 150
K_CUTOFF = 6

# =========================================================
# 5. 预处理
# =========================================================
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

# =========================================================
# 6. 主流程
# =========================================================
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

adata = preprocess_for_stagate(adata, n_top_genes=3000)

if GRAPH_MODEL == "Radius":
    Cal_Spatial_Net(adata, rad_cutoff=RAD_CUTOFF, model="Radius", verbose=True)
else:
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
    random_seed=seed,
    save_attention=False,
    save_loss=False,
    save_reconstrction=False,
)

safe_mclust_R(
    adata,
    num_cluster=n_clusters,
    used_obsm="STAGATE",
    key_added="domain",
    random_seed=seed
)

adata_eval = adata[~pd.isnull(adata.obs["ground_truth"])].copy()

ari = metrics.adjusted_rand_score(adata_eval.obs["domain"], adata_eval.obs["ground_truth"])
nmi = normalized_mutual_info_score(adata_eval.obs["domain"], adata_eval.obs["ground_truth"])

print(f"\n🎉 STAGATE section1 | ARI: {ari:.6f} | NMI: {nmi:.6f}")

results_df = pd.DataFrame([{
    "Dataset": "Human-Breast-Cancer-Block-A_section1",
    "Method": "STAGATE",
    "Seed": seed,
    "N_Clusters": n_clusters,
    "N_Obs_Eval": adata_eval.n_obs,
    "ARI": ari,
    "NMI": nmi,
}])

out_csv = os.path.join(file_fold, "STAGATE_HBCA_results.csv")
results_df.to_csv(out_csv, index=False)
print(f"💾 Saved to: {out_csv}")