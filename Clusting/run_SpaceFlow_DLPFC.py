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

if not hasattr(nx, "to_scipy_sparse_matrix"):
    def _to_scipy_sparse_matrix(*args, **kwargs):
        arr = nx.to_scipy_sparse_array(*args, **kwargs)
        return sp.csr_matrix(arr)
    nx.to_scipy_sparse_matrix = _to_scipy_sparse_matrix


# =========================================================
# 0. 环境与路径
# =========================================================
os.environ["R_HOME"] = "/data2/liangyefeng/miniconda3/envs/graphst/lib/R"

project_root = "/data2/liangyefeng/My_GraphST_Innovation"
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "src", "SpaceFlow"))

# ---- SpaceFlow ----
from SpaceFlow.SpaceFlow import SpaceFlow

# ---- GraphST统一评测链 ----
import GraphST.utils as graphst_utils
from GraphST.utils import clustering


# =========================================================
# 1. 固定随机种子
# =========================================================
def reset_seed(seed: int = 41):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =========================================================
# 2. patch SpaceFlow.preprocessing_data
# 目标：
# - 避免 cell_ranger HVG 分箱重复边界报错
# - 改用 seurat flavor
# =========================================================
def patched_spaceflow_preprocessing_data(self, n_top_genes=None, n_neighbors=10):
    adata = self.adata
    if adata is None:
        raise ValueError("No AnnData found in SpaceFlow object.")

    adata = adata.copy()
    adata.var_names_make_unique()

    # 先过滤掉极低表达基因，增强稳定性
    sc.pp.filter_genes(adata, min_cells=3)

    # SpaceFlow 原始流程：normalize -> log1p -> HVG -> PCA -> graph
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 关键修复：把 flavor='cell_ranger' 改成 'seurat'
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


# =========================================================
# 3. patch mclust：更稳，避免 NULLType
# =========================================================
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
        raise ValueError(
            f"mclust returned {na_count} invalid labels for {used_obsm}."
        )

    adata.obs["mclust"] = cls_series.astype(int).astype("category")
    return adata


graphst_utils.mclust_R = patched_mclust_R


# =========================================================
# 4. 数据与聚类数
# =========================================================
seed = 41
reset_seed(seed)

print("✅ 当前设备:", "cuda" if torch.cuda.is_available() else "cpu")

data_root = "/data2/liangyefeng/GraphST-main/Data/1.DLPFC"

datasets = sorted([
    d for d in os.listdir(data_root)
    if os.path.isdir(os.path.join(data_root, d)) and d.isdigit()
])

cluster_map = {
    "151507": 7, "151508": 7, "151509": 7, "151510": 7,
    "151669": 5, "151670": 5, "151671": 5, "151672": 5,
    "151673": 7, "151674": 7, "151675": 7, "151676": 7,
}

print("✅ 待处理数据集:", datasets)

# =========================================================
# 5. SpaceFlow参数
# =========================================================
N_TOP_GENES = 3000
SPATIAL_N_NEIGHBORS = 10

Z_DIM = 50
LR = 1e-3
EPOCHS = 1000
MAX_PATIENCE = 50
MIN_STOP = 100
SPATIAL_REG_STRENGTH = 0.1
GPU_ID = 0

REFINE_RADIUS = 50

# =========================================================
# 6. 主循环
# =========================================================
results = []

for dataset in datasets:
    print("\n" + "=" * 90)
    print(f"🚀 Processing dataset: {dataset}")
    print("=" * 90)

    reset_seed(seed)

    file_fold = os.path.join(data_root, dataset)

    # 6.1 读数据
    adata = sc.read_visium(
        file_fold,
        count_file="filtered_feature_bc_matrix.h5",
        load_images=True
    )
    adata.var_names_make_unique()

    # 6.2 初始化 SpaceFlow
    sf = SpaceFlow(adata=adata.copy())

    # 6.3 预处理 + 构图（已 patched）
    sf.preprocessing_data(
        n_top_genes=min(N_TOP_GENES, adata.n_vars),
        n_neighbors=SPATIAL_N_NEIGHBORS
    )

    # 6.4 训练，拿 embedding
    emb_dir = os.path.join(project_root, "results_spaceflow_embeddings")
    emb_path = os.path.join(emb_dir, f"{dataset}_embedding.tsv")

    embedding = sf.train(
        embedding_save_filepath=emb_path,
        spatial_regularization_strength=SPATIAL_REG_STRENGTH,
        z_dim=Z_DIM,
        lr=LR,
        epochs=EPOCHS,
        max_patience=MAX_PATIENCE,
        min_stop=MIN_STOP,
        random_seed=seed,
        gpu=GPU_ID,
        regularization_acceleration=True,
        edge_subset_sz=1000000
    )

    # 把 embedding 写回原 adata，便于统一评测
    adata.obsm["SpaceFlow"] = embedding
    adata.obsm["emb"] = adata.obsm["SpaceFlow"].copy()

    n_clusters = cluster_map.get(dataset, 7)

    clustering(
        adata,
        n_clusters,
        radius=REFINE_RADIUS,
        method="mclust",
        refinement=True
    )

    # 6.6 GT 对齐
    df_meta = pd.read_csv(os.path.join(file_fold, "metadata.tsv"), sep="\t")
    barcode_col = next((c for c in df_meta.columns if "barcode" in c.lower()), None)

    if barcode_col is not None:
        df_meta = df_meta.set_index(barcode_col)
        df_meta = df_meta.loc[adata.obs_names]
        adata.obs["ground_truth"] = df_meta["layer_guess"]
    else:
        adata.obs["ground_truth"] = df_meta["layer_guess"].values

    # 6.7 去掉没标注的点
    adata_eval = adata[~pd.isnull(adata.obs["ground_truth"])].copy()

    # 6.8 计算 ARI / NMI
    ari = metrics.adjusted_rand_score(
        adata_eval.obs["domain"],
        adata_eval.obs["ground_truth"]
    )
    nmi = normalized_mutual_info_score(
        adata_eval.obs["domain"],
        adata_eval.obs["ground_truth"]
    )

    print(f"📊 Dataset {dataset} | ARI: {ari:.6f} | NMI: {nmi:.6f}")

    results.append({
        "Dataset": dataset,
        "Method": "SpaceFlow",
        "Seed": seed,
        "HVG_Flavor": "seurat",
        "N_Clusters": n_clusters,
        "N_Obs_Eval": adata_eval.n_obs,
        "ARI": ari,
        "NMI": nmi,
    })

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# =========================================================
# 7. 保存结果
# =========================================================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Dataset").reset_index(drop=True)

print("\n" + "#" * 90)
print("🎉 SpaceFlow on DLPFC: final results")
print("#" * 90)
print(results_df.to_string(index=False))

out_csv = os.path.join(data_root, "Corrected_SpaceFlow_Baseline_Results.csv")
results_df.to_csv(out_csv, index=False)
print(f"\n💾 结果已保存至: {out_csv}")