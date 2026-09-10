import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score

warnings.filterwarnings("ignore")

# =========================================================
# 0. 环境与路径
# =========================================================
os.environ["R_HOME"] = "/data2/liangyefeng/miniconda3/envs/graphst/lib/R"

project_root = "/data2/liangyefeng/My_GraphST_Innovation"
sys.path.insert(0, os.path.join(project_root, "src"))

from GraphST.GraphST import GraphST
import GraphST.utils as graphst_utils
from GraphST.utils import clustering

# =========================================================
# 1. patch mclust，避免老问题
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
        raise ValueError(f"mclust returned {na_count} invalid labels.")

    adata.obs["mclust"] = cls_series.astype(int).astype("category")
    return adata

graphst_utils.mclust_R = patched_mclust_R

# =========================================================
# 2. 固定随机种子
# =========================================================
seed = 41
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("✅ 当前设备:", device)

# =========================================================
# 3. 数据路径
# =========================================================
file_fold = "/data2/liangyefeng/My_GraphST_Innovation/data/Human-Breast-Cancer-Block-A/section1"
gt_dir = os.path.join(file_fold, "gt")

# =========================================================
# 4. 读取数据
# =========================================================
adata = sc.read_visium(
    file_fold,
    count_file="section1_filtered_feature_bc_matrix.h5",
    load_images=True
)
adata.var_names_make_unique()

print("✅ adata shape:", adata.shape)

# =========================================================
# 5. 读取 GT
# 优先用 gold_metadata.tsv
# =========================================================
gold_meta_path = os.path.join(gt_dir, "gold_metadata.tsv")
tissue_gt_path = os.path.join(gt_dir, "tissue_positions_list_GTs.txt")

def detect_label_col(df):
    candidates = [c for c in df.columns if any(k in c.lower() for k in [
        "label", "annotation", "region", "cluster", "layer", "ground", "gt"
    ])]
    if len(candidates) > 0:
        return candidates[0]
    return df.columns[-1]

def load_gt(obs_names):
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

    elif os.path.exists(tissue_gt_path):
        # 尝试自动识别分隔符
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

    else:
        raise FileNotFoundError("No GT file found.")

gt, gt_path, label_col = load_gt(adata.obs_names)
adata.obs["ground_truth"] = gt

print("✅ GT file:", gt_path)
print("✅ label column:", label_col)

# 去掉无标注 spot
adata_eval = adata[~pd.isnull(adata.obs["ground_truth"])].copy()

# GT 类别数作为聚类数
n_clusters = adata_eval.obs["ground_truth"].nunique()
print("✅ n_clusters from GT:", n_clusters)

# =========================================================
# 6. 训练 GraphST
# =========================================================
model = GraphST(adata, device=device)
adata = model.train()

# =========================================================
# 7. 聚类
# =========================================================
radius = 50
clustering(
    adata,
    n_clusters,
    radius=radius,
    method="mclust",
    refinement=True
)

# =========================================================
# 8. 对齐 GT 再评估
# =========================================================
adata_eval = adata[~pd.isnull(adata.obs["ground_truth"])].copy()

ari = metrics.adjusted_rand_score(
    adata_eval.obs["domain"],
    adata_eval.obs["ground_truth"]
)
nmi = normalized_mutual_info_score(
    adata_eval.obs["domain"],
    adata_eval.obs["ground_truth"]
)

print(f"\n🎉 section1 | ARI: {ari:.6f} | NMI: {nmi:.6f}")

# =========================================================
# 9. 保存结果
# =========================================================
results_df = pd.DataFrame([{
    "Dataset": "Human-Breast-Cancer-Block-A_section1",
    "Method": "GraphST",
    "Seed": seed,
    "N_Clusters": n_clusters,
    "N_Obs_Eval": adata_eval.n_obs,
    "ARI": ari,
    "NMI": nmi,
}])

out_csv = os.path.join(file_fold, "GraphST_HBCA_results.csv")
results_df.to_csv(out_csv, index=False)

print(f"💾 结果已保存至: {out_csv}")