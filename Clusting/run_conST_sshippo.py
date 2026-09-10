import os
import sys
import random
import warnings
from types import SimpleNamespace

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import scanpy as sc

# =========================================================
# 0. 路径配置
# =========================================================
PROJECT_ROOT = "/data2/liangyefeng/My_GraphST_Innovation"
CONST_ROOT = os.path.join(PROJECT_ROOT, "src", "conST")
PROJECT_SRC = os.path.join(PROJECT_ROOT, "src")

DATA_PATH = "/data2/liangyefeng/My_GraphST_Innovation/data/mouse_hyppocampus_slideseqv2/sshippo_binned.h5ad"
RESULT_DIR = "/data2/liangyefeng/My_GraphST_Innovation/data/mouse_hyppocampus_slideseqv2/conST_sshippo_binned"
RESULT_CSV = os.path.join(RESULT_DIR, "conST_sshippo_binned_results.csv")
RESULT_H5AD = os.path.join(RESULT_DIR, "sshippo_binned_conST_output.h5ad")
EMB_NPY = os.path.join(RESULT_DIR, "conST_embedding.npy")
OBS_CSV = os.path.join(RESULT_DIR, "sshippo_binned_conST_obs.csv")

os.makedirs(RESULT_DIR, exist_ok=True)

# conST 自己的 src
if CONST_ROOT not in sys.path:
    sys.path.insert(0, CONST_ROOT)

# 你项目里其他方法/公共函数所在 src
if PROJECT_SRC not in sys.path:
    sys.path.append(PROJECT_SRC)

# =========================================================
# 1. 导入 conST 与公共函数
# =========================================================
from src.graph_func import graph_construction
from src.utils_func import mk_dir, adata_preprocess
from src.training import conST_training

from sshippo_binned_common import (
    load_binned_h5ad,
    to_counts_adata,
    compute_ari_nmi,
)

# =========================================================
# 2. 固定随机种子
# =========================================================
SEED = 41
FORCE_LABEL_COL = "cluster"

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
#    基本沿用你 DLPFC 脚本，只适配单个 h5ad 数据集
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
    use_img=False,          # Slide-seqV2/纯表达+空间坐标，一般不用图像
    img_w=0.1,
    use_pretrained=False,
    using_mask=False,
    feat_w=10,
    gcn_w=0.1,
    dec_kl_w=10,
    gcn_lr=0.01,
    gcn_decay=0.01,
    dec_cluster_n=10,       # 后面按数据真实簇数覆盖
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
# 4. mclust 聚类函数
# =========================================================
def mclust_R(
    adata,
    num_cluster,
    used_obsm="X_emb",
    key_added="domain",
    modelNames="EEE",
    random_seed=41,
):
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

adata, label_col, n_clusters = load_binned_h5ad(
    DATA_PATH,
    force_label_col=FORCE_LABEL_COL
)
adata = to_counts_adata(adata)

if "spatial" not in adata.obsm:
    raise KeyError("adata.obsm['spatial'] not found.")

# 兼容不同公共函数的返回格式：
# 如果 load_binned_h5ad 已经生成 ground_truth，这里直接用；
# 否则回退到 label_col
if "ground_truth" not in adata.obs.columns:
    if label_col not in adata.obs.columns:
        raise KeyError(f"Neither 'ground_truth' nor '{label_col}' exists in adata.obs.")
    adata.obs["ground_truth"] = adata.obs[label_col].copy()

params.cell_num = adata.n_obs
params.dec_cluster_n = int(n_clusters)
params.save_path = mk_dir(RESULT_DIR)

print(f"Dataset loaded: {os.path.basename(DATA_PATH)}")
print(f"n_obs={adata.n_obs}, n_vars={adata.n_vars}")
print(f"label_col={label_col}")
print(f"n_clusters={n_clusters}")

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

np.save(EMB_NPY, conST_embedding)

# =========================================================
# 8. 保存 embedding + mclust 聚类
# =========================================================
adata_out = adata.copy()
adata_out.obsm["X_emb"] = conST_embedding.copy()

adata_out = mclust_R(
    adata_out,
    num_cluster=int(n_clusters),
    used_obsm="X_emb",
    key_added="domain",
    random_seed=SEED,
)

# =========================================================
# 9. 评估
# =========================================================
ari, nmi, n_eval = compute_ari_nmi(
    adata_out,
    pred_col="domain",
    gt_col="ground_truth"
)

print(f"conST | ARI={ari:.6f} | NMI={nmi:.6f}")

# =========================================================
# 10. 保存结果
# =========================================================
save_cols = []
for c in [label_col, "ground_truth", "domain"]:
    if c in adata_out.obs.columns and c not in save_cols:
        save_cols.append(c)

adata_out.write_h5ad(RESULT_H5AD)
adata_out.obs[save_cols].to_csv(OBS_CSV)

pd.DataFrame([{
    "Dataset": "sshippo_binned",
    "Method": "conST",
    "Seed": SEED,
    "Label_Col": label_col,
    "Use_Image": False,
    "N_Clusters": int(n_clusters),
    "N_Obs": int(adata_out.n_obs),
    "N_Obs_Eval": int(n_eval),
    "ARI": float(ari),
    "NMI": float(nmi),
    "Embedding_File": EMB_NPY,
    "Output_H5AD": RESULT_H5AD,
}]).to_csv(RESULT_CSV, index=False)

print("saved embedding:", EMB_NPY)
print("saved h5ad     :", RESULT_H5AD)
print("saved obs csv  :", OBS_CSV)
print("saved results  :", RESULT_CSV)