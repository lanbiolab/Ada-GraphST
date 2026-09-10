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
from sklearn import metrics

# =========================================================
# 0. 路径配置
# =========================================================
PROJECT_ROOT = "/data2/liangyefeng/My_GraphST_Innovation"
CONST_ROOT = os.path.join(PROJECT_ROOT, "src", "conST")
DATA_ROOT = "/data2/liangyefeng/My_GraphST_Innovation/data/1.DLPFC"
SAVE_ROOT = "/data2/liangyefeng/My_GraphST_Innovation/Result_conST_DLPFC_mclust"

os.makedirs(SAVE_ROOT, exist_ok=True)

# import conST
if CONST_ROOT not in sys.path:
    sys.path.insert(0, CONST_ROOT)

from src.graph_func import graph_construction
from src.utils_func import mk_dir, adata_preprocess, load_ST_file
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
# 2. DLPFC 12 个切片
# =========================================================
datasets = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676",
]

# =========================================================
# 3. conST 参数
# 这里按官方 notebook 默认值组织，只把 seed 改成 41
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
    use_pretrained=False,      # 不使用 151673 的预训练权重，12 个切片都独立训练
    using_mask=False,
    feat_w=10,
    gcn_w=0.1,
    dec_kl_w=10,
    gcn_lr=0.01,
    gcn_decay=0.01,
    dec_cluster_n=10,          # 后面按每个切片真实簇数覆盖
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
# 5. 工具函数
# =========================================================
def get_count_file(file_fold: str, dataset: str) -> str:
    """
    优先用 {dataset}_filtered_feature_bc_matrix.h5
    否则回退到 filtered_feature_bc_matrix.h5
    """
    c1 = os.path.join(file_fold, f"{dataset}_filtered_feature_bc_matrix.h5")
    c2 = os.path.join(file_fold, "filtered_feature_bc_matrix.h5")

    if os.path.exists(c1):
        return f"{dataset}_filtered_feature_bc_matrix.h5"
    if os.path.exists(c2):
        return "filtered_feature_bc_matrix.h5"

    raise FileNotFoundError(f"No count h5 found in {file_fold}")


def load_metadata(file_fold: str, obs_names):
    """
    从 metadata.tsv 读取 ground truth，默认用 layer_guess。
    按 barcode / index 对齐到 adata.obs_names。
    """
    meta_path = os.path.join(file_fold, "metadata.tsv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.tsv not found: {meta_path}")

    df_meta = pd.read_csv(meta_path, sep="\t")

    # 找 barcode / index 列
    barcode_col = None
    for c in df_meta.columns:
        cl = c.lower()
        if "barcode" in cl or cl == "index":
            barcode_col = c
            break

    if barcode_col is not None:
        df_meta = df_meta.set_index(barcode_col)
    else:
        first_col = df_meta.columns[0]
        overlap = len(set(df_meta[first_col].astype(str)) & set(map(str, obs_names)))
        if overlap > 0:
            df_meta = df_meta.set_index(first_col)
        else:
            raise ValueError("Cannot identify barcode/index column in metadata.tsv")

    if "layer_guess" not in df_meta.columns:
        raise ValueError(f"'layer_guess' not found in metadata.tsv columns: {list(df_meta.columns)}")

    gt = df_meta.reindex(obs_names)["layer_guess"]
    return gt, df_meta


def infer_n_clusters(gt: pd.Series) -> int:
    return gt.dropna().nunique()


def compute_scores(gt: pd.Series, pred: pd.Series):
    mask = ~pd.isnull(gt)
    gt_valid = gt[mask]
    pred_valid = pred[mask]

    ari = metrics.adjusted_rand_score(gt_valid, pred_valid)
    nmi = metrics.normalized_mutual_info_score(gt_valid, pred_valid)
    return ari, nmi, int(mask.sum())


# =========================================================
# 6. 主循环
# =========================================================
results = []

for data_name in datasets:
    print("\n" + "=" * 100)
    print(f"🚀 Processing dataset: {data_name}")
    print("=" * 100)

    seed_torch(SEED)

    file_fold = os.path.join(DATA_ROOT, data_name)
    params.save_path = mk_dir(os.path.join(SAVE_ROOT, data_name, "conST"))

    # 6.1 读 Visium
    count_file = get_count_file(file_fold, data_name)
    adata_h5 = load_ST_file(file_fold, count_file=count_file, load_images=True)

    # 6.2 读 metadata / GT
    gt, df_meta_raw = load_metadata(file_fold, adata_h5.obs_names)
    n_clusters = infer_n_clusters(gt)

    params.cell_num = adata_h5.shape[0]
    params.dec_cluster_n = int(n_clusters)

    print(f"✅ n_clusters inferred from layer_guess: {n_clusters}")
    print(f"✅ n_obs: {adata_h5.n_obs}, n_vars: {adata_h5.n_vars}")

    # 6.3 表达预处理 + 图构建
    adata_X = adata_preprocess(adata_h5, min_cells=5, pca_n_comps=params.cell_feat_dim)
    graph_dict = graph_construction(adata_h5.obsm['spatial'], adata_h5.shape[0], params)

    # 6.4 训练 conST
    conST_net = conST_training(adata_X, graph_dict, params, n_clusters)

    # 所有切片独立训练，不用官方 151673 的预训练权重
    conST_net.pretraining()
    conST_net.major_training()

    # 6.5 获取 embedding
    conST_embedding = conST_net.get_embedding()
    np.save(os.path.join(params.save_path, "conST_result.npy"), conST_embedding)

    adata_conST = sc.AnnData(conST_embedding)
    adata_conST.obs_names = adata_h5.obs_names.copy()
    adata_conST.obsm["spatial"] = adata_h5.obsm["spatial"].copy()
    adata_conST.obsm["X_emb"] = conST_embedding.copy()

    # 6.6 mclust 聚类
    adata_conST = mclust_R(
        adata_conST,
        num_cluster=n_clusters,
        used_obsm="X_emb",
        key_added="conST_mclust",
        random_seed=SEED
    )

    # 6.7 评分
    gt_aligned = gt.reindex(adata_conST.obs_names)
    ari, nmi, n_eval = compute_scores(gt_aligned, adata_conST.obs["conST_mclust"])

    print(f"📊 {data_name} | mclust ARI={ari:.6f} | mclust NMI={nmi:.6f}")

    # 6.8 保存每个切片结果
    out_meta = df_meta_raw.copy().reindex(adata_conST.obs_names)
    out_meta["conST_mclust"] = adata_conST.obs["conST_mclust"].astype(str).values
    out_meta.to_csv(os.path.join(params.save_path, "metadata_with_conST_mclust.tsv"), sep="\t", index=True)

    results.append({
        "Dataset": data_name,
        "Method": "conST",
        "Seed": SEED,
        "Clustering_Head": "mclust",
        "N_Clusters": n_clusters,
        "N_Obs_Eval": n_eval,
        "ARI": ari,
        "NMI": nmi,
    })

# =========================================================
# 7. 汇总保存
# =========================================================
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(SAVE_ROOT, "conST_DLPFC_12slice_mclust_results.csv"), index=False)

print("\n" + "=" * 100)
print("✅ All done.")
print(results_df.to_string(index=False))
print("=" * 100)