import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score

warnings.filterwarnings("ignore")

# =========================================================
# 0. 环境与路径
# =========================================================
os.environ["R_HOME"] = "/data2/liangyefeng/miniconda3/envs/graphst/lib/R"

project_root = "/data2/liangyefeng/My_GraphST_Innovation"
sys.path.insert(0, os.path.join(project_root, "src"))

# ---- TensorFlow v1 compatibility ----
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# ---- STAGATE ----
from STAGATE.STAGATE.Train_STAGATE import train_STAGATE
from STAGATE.STAGATE.utils import Cal_Spatial_Net

# ---- 统一评测链：沿用 GraphST 的 clustering ----
import GraphST.utils as graphst_utils
from GraphST.utils import clustering

# =========================================================
# 1. patch GraphST 的 mclust，确保吃 emb_pca
# =========================================================
def patched_mclust_R(adata, num_cluster, modelNames="EEE", used_obsm="emb_pca", random_seed=2020):
    import numpy as np
    import rpy2.robjects as robjects
    from rpy2.robjects import r
    from rpy2.robjects import numpy2ri

    x = np.asarray(adata.obsm[used_obsm], dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"{used_obsm} must be 2D, but got shape {x.shape}")

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
        res <- Mclust(x_mat, G=n_cluster, modelNames=model_name)
        cls <- res$classification
        """
    )

    mclust_res = np.array(r["cls"])
    adata.obs["mclust"] = mclust_res
    adata.obs["mclust"] = adata.obs["mclust"].astype(int)
    adata.obs["mclust"] = adata.obs["mclust"].astype("category")
    return adata

graphst_utils.mclust_R = patched_mclust_R

# =========================================================
# 2. 固定随机种子
# =========================================================
seed = 41
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

# =========================================================
# 3. 数据与簇数
# =========================================================
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

print("✅ datasets:", datasets)

# =========================================================
# 4. STAGATE 参数
# 说明：
# - hidden_dims=[512, 30] 是这份代码里最常见的用法
# - alpha=0 表示不用 pre-cluster pruning
# - 图构建这里先给你用 Radius=150（Visium 常用起点）
#   若后面邻居数明显过大/过小，再调这个参数
# =========================================================
HIDDEN_DIMS = [512, 30]
ALPHA = 0
N_EPOCHS = 500
LR = 1e-4
WEIGHT_DECAY = 1e-4
RANDOM_SEED = seed

GRAPH_MODEL = "Radius"   # 可改成 "KNN"
RAD_CUTOFF = 150         # 若用 Radius
K_CUTOFF = 6             # 若用 KNN

# 统一下游评测
REFINE_RADIUS = 50

# =========================================================
# 5. 预处理
# 注意：
# - train_STAGATE 里会对 HVG 子集取 X.toarray()
# - 所以这里不要 scale 成 dense 再破坏接口
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
# 6. 主循环
# =========================================================
results = []

for dataset in datasets:
    print("\n" + "=" * 90)
    print(f"🚀 Processing dataset: {dataset}")
    print("=" * 90)

    file_fold = os.path.join(data_root, dataset)

    # 6.1 读取 Visium
    adata = sc.read_visium(
        file_fold,
        count_file="filtered_feature_bc_matrix.h5",
        load_images=True
    )
    adata.var_names_make_unique()

    # 6.2 预处理
    adata = preprocess_for_stagate(adata, n_top_genes=3000)

    # 6.3 构图
    if GRAPH_MODEL == "Radius":
        Cal_Spatial_Net(
            adata,
            rad_cutoff=RAD_CUTOFF,
            model="Radius",
            verbose=True
        )
    else:
        Cal_Spatial_Net(
            adata,
            k_cutoff=K_CUTOFF,
            model="KNN",
            verbose=True
        )

    # 6.4 训练 STAGATE
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
        random_seed=RANDOM_SEED,
        save_attention=False,
        save_loss=False,
        save_reconstrction=False,
    )

    # 6.5 统一评测链：PCA -> mclust -> refinement
    n_clusters = cluster_map.get(dataset, 7)

    adata.obsm["emb"] = adata.obsm["STAGATE"].copy()

    clustering(
        adata,
        n_clusters,
        radius=REFINE_RADIUS,
        method="mclust",
        refinement=True
    )

    # 6.6 读取 metadata.tsv 并对齐 GT
    df_meta = pd.read_csv(os.path.join(file_fold, "metadata.tsv"), sep="\t")
    barcode_col = next((c for c in df_meta.columns if "barcode" in c.lower()), None)

    if barcode_col is not None:
        df_meta = df_meta.set_index(barcode_col)
        df_meta = df_meta.loc[adata.obs_names]
        adata.obs["ground_truth"] = df_meta["layer_guess"]
    else:
        adata.obs["ground_truth"] = df_meta["layer_guess"].values

    # 6.7 去掉没有真值的 spot
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
        "Method": "STAGATE",
        "Graph_Model": GRAPH_MODEL,
        "Rad_Cutoff": RAD_CUTOFF if GRAPH_MODEL == "Radius" else None,
        "K_Cutoff": K_CUTOFF if GRAPH_MODEL == "KNN" else None,
        "N_Clusters": n_clusters,
        "N_Obs_Eval": adata_eval.n_obs,
        "ARI": ari,
        "NMI": nmi,
    })

# =========================================================
# 7. 保存
# =========================================================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Dataset").reset_index(drop=True)

print("\n" + "#" * 90)
print("🎉 STAGATE on DLPFC: final results")
print("#" * 90)
print(results_df.to_string(index=False))

out_csv = os.path.join(data_root, "Corrected_STAGATE_Baseline_Results.csv")
results_df.to_csv(out_csv, index=False)
print(f"\n💾 结果已保存至: {out_csv}")