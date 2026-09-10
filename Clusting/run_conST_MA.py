import os
import sys
import random
import warnings
from types import SimpleNamespace

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

# =========================================================
# 0. 路径配置
# =========================================================
PROJECT_ROOT = "/data2/liangyefeng/My_GraphST_Innovation"
CONST_ROOT = os.path.join(PROJECT_ROOT, "src", "conST")
PROJECT_SRC = os.path.join(PROJECT_ROOT, "src")

RESULT_CSV = "/data2/liangyefeng/My_GraphST_Innovation/data/Mouse-Brain-Section-2-Sagittal-Anterior-and-Posterior/MA/conST_results.csv"
SEED = 41

# conST 源码路径
if CONST_ROOT not in sys.path:
    sys.path.insert(0, CONST_ROOT)

# 项目公共函数路径
if PROJECT_SRC not in sys.path:
    sys.path.append(PROJECT_SRC)

# =========================================================
# 1. 导入 conST 和公共函数
# =========================================================
from src.graph_func import graph_construction
from src.utils_func import adata_preprocess
from src.training import conST_training

from mouse_brain_MA_common import (
    load_ma_dataset,
    to_counts_adata,
    compute_ari_nmi,
)

# =========================================================
# 2. 固定随机种子
# =========================================================
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
# 3. conST 参数
# =========================================================
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
    dec_cluster_n=10,   # 后面用真实簇数覆盖
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
# 4. mclust 聚类
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
        raise ValueError(f"mclust returned {na_count} invalid labels for {used_obsm}")

    adata.obs[key_added] = cls_series.astype(int).astype("category")
    return adata

# =========================================================
# 5. 读取数据
# =========================================================
seed_torch(SEED)

adata, label_col, n_clusters = load_ma_dataset()
adata = to_counts_adata(adata)

adata.obs_names_make_unique()
adata.var_names_make_unique()

if "spatial" not in adata.obsm:
    raise KeyError("adata.obsm['spatial'] not found.")

if "ground_truth" not in adata.obs.columns:
    if label_col not in adata.obs.columns:
        raise KeyError(f"Neither 'ground_truth' nor '{label_col}' exists in adata.obs.")
    adata.obs["ground_truth"] = adata.obs[label_col].copy()

params.cell_num = adata.n_obs
params.dec_cluster_n = int(n_clusters)

print(f"Dataset: MA")
print(f"n_obs={adata.n_obs}, n_vars={adata.n_vars}")
print(f"label_col={label_col}")
print(f"n_clusters={n_clusters}")

# =========================================================
# 6. 预处理 + 图构建
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

ari, nmi, n_eval = compute_ari_nmi(
    adata,
    pred_col="domain",
    gt_col="ground_truth"
)

print(f"conST | ARI={ari:.6f} | NMI={nmi:.6f}")

# =========================================================
# 9. 保存结果 csv
# =========================================================
pd.DataFrame([{
    "Dataset": "MA",
    "Method": "conST",
    "Seed": SEED,
    "Label_Col": label_col,
    "N_Clusters": int(n_clusters),
    "N_Obs_Eval": int(n_eval),
    "ARI": float(ari),
    "NMI": float(nmi),
}]).to_csv(RESULT_CSV, index=False)

print("saved:", RESULT_CSV)