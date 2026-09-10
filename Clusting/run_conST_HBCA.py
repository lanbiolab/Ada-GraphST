import os
import sys
import random
import warnings
from types import SimpleNamespace

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score

# =========================================================
# 0. 环境与路径
# =========================================================
os.environ["R_HOME"] = "/data2/liangyefeng/miniconda3/envs/graphst/lib/R"

PROJECT_ROOT = "/data2/liangyefeng/My_GraphST_Innovation"
CONST_ROOT = os.path.join(PROJECT_ROOT, "src", "conST")

if CONST_ROOT not in sys.path:
    sys.path.insert(0, CONST_ROOT)

from src.graph_func import graph_construction
from src.utils_func import adata_preprocess
from src.training import conST_training

# =========================================================
# 1. 固定随机种子
# =========================================================
SEED = 41

def seed_torch(seed=41):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

seed_torch(SEED)

# =========================================================
# 2. 更稳的 mclust
# =========================================================
def mclust_R(adata, num_cluster, used_obsm="X_emb", key_added="domain", modelNames="EEE", random_seed=41):
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
            gt = df.reindex(obs_names)[label_col]
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
            gt = df.reindex(obs_names)[label_col]
        else:
            gt = pd.Series(df[label_col].values, index=obs_names)
        return gt, tissue_gt_path, label_col

    raise FileNotFoundError("No GT file found.")

# =========================================================
# 4. 配置
# =========================================================
FILE_FOLD = "/data2/liangyefeng/My_GraphST_Innovation/data/Human-Breast-Cancer-Block-A/section1"
COUNT_FILE = "section1_filtered_feature_bc_matrix.h5"
OUT_CSV = os.path.join(FILE_FOLD, "conST_HBCA_results.csv")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

params = SimpleNamespace(
    # graph
    k=20,
    knn_distanceType="euclidean",

    # training
    epochs=200,
    cell_feat_dim=300,
    feat_hidden1=100,
    feat_hidden2=20,
    gcn_hidden1=32,
    gcn_hidden2=8,
    p_drop=0.2,
    use_img=False,
    img_w=0.1,
    use_pretrained=False,
    using_mask=False,
    feat_w=10,
    gcn_w=0.1,
    dec_kl_w=10,
    gcn_lr=0.01,
    gcn_decay=0.01,
    dec_cluster_n=10,   # 后面覆盖
    dec_interval=20,
    dec_tol=0.00,

    # contrastive
    seed=SEED,
    beta=100,
    cont_l2l=0.3,
    cont_l2c=0.1,
    cont_l2g=0.1,
    edge_drop_p1=0.1,
    edge_drop_p2=0.1,
    node_drop_p1=0.2,
    node_drop_p2=0.3,

    # runtime
    device=device,
    cell_num=None,
    save_path=None,
)

# =========================================================
# 5. 读取数据
# =========================================================
seed_torch(SEED)

adata = sc.read_visium(
    FILE_FOLD,
    count_file=COUNT_FILE,
    load_images=True
)
adata.var_names_make_unique()
adata.obs_names_make_unique()

if "spatial" not in adata.obsm:
    raise KeyError("adata.obsm['spatial'] not found.")

gt, gt_path, label_col = load_gt(FILE_FOLD, adata.obs_names)
adata.obs["ground_truth"] = gt

adata_eval = adata[~pd.isnull(adata.obs["ground_truth"])].copy()
n_clusters = adata_eval.obs["ground_truth"].nunique()

params.cell_num = adata.n_obs
params.dec_cluster_n = int(n_clusters)

print("GT file:", gt_path)
print("label column:", label_col)
print("n_clusters:", n_clusters)
print(f"n_obs={adata.n_obs}, n_vars={adata.n_vars}")

# =========================================================
# 6. conST 预处理 + 图构建
# =========================================================
seed_torch(SEED)

adata_X = adata_preprocess(
    adata,
    min_cells=5,
    pca_n_comps=params.cell_feat_dim
)

graph_dict = graph_construction(
    adata.obsm["spatial"],
    adata.shape[0],
    params
)

# =========================================================
# 7. 训练 conST
# =========================================================
seed_torch(SEED)

conST_net = conST_training(adata_X, graph_dict, params, int(n_clusters))
conST_net.pretraining()
conST_net.major_training()

conST_embedding = conST_net.get_embedding()
if torch.is_tensor(conST_embedding):
    conST_embedding = conST_embedding.detach().cpu().numpy()
else:
    conST_embedding = np.asarray(conST_embedding)

if conST_embedding.shape[0] != adata.n_obs:
    raise ValueError(
        f"Embedding rows ({conST_embedding.shape[0]}) != adata.n_obs ({adata.n_obs})"
    )

adata.obsm["X_emb"] = conST_embedding.copy()

# =========================================================
# 8. mclust 聚类 + 评估
# =========================================================
adata = mclust_R(
    adata,
    num_cluster=int(n_clusters),
    used_obsm="X_emb",
    key_added="domain",
    random_seed=SEED
)

adata_eval = adata[~pd.isnull(adata.obs["ground_truth"])].copy()

ari = metrics.adjusted_rand_score(
    adata_eval.obs["domain"],
    adata_eval.obs["ground_truth"]
)
nmi = normalized_mutual_info_score(
    adata_eval.obs["domain"],
    adata_eval.obs["ground_truth"]
)

print(f"\nconST section1 | ARI: {ari:.6f} | NMI: {nmi:.6f}")

# =========================================================
# 9. 保存结果 csv
# =========================================================
results_df = pd.DataFrame([{
    "Dataset": "Human-Breast-Cancer-Block-A_section1",
    "Method": "conST",
    "Seed": SEED,
    "N_Clusters": int(n_clusters),
    "N_Obs_Eval": int(adata_eval.n_obs),
    "ARI": float(ari),
    "NMI": float(nmi),
}])

results_df.to_csv(OUT_CSV, index=False)
print("Saved to:", OUT_CSV)