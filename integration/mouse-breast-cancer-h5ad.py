# save as: MouseBreastCancer_generate_h5ad_only.py

import os
import sys
import random
import gc
import numpy as np
import pandas as pd
import torch
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# =========================================================
# 0. 环境与配置
# =========================================================
os.environ["R_HOME"] = "/data2/liangyefeng/miniconda3/envs/graphst/lib/R"

project_root = "/data2/liangyefeng/My_GraphST_Innovation"
sys.path.insert(0, os.path.join(project_root, "src"))

from GraphST.GraphST import GraphST
import GraphST.utils as graphst_utils
from GraphST.utils import clustering

# =========================================================
# 1. 路径与参数
# =========================================================
PAIR_NAME = "MouseBreastCancer_section1_section2"

data_path = (
    "/data2/liangyefeng/My_GraphST_Innovation/data/"
    "7.Mouse_Breast_Cancer_Sample_1/"
    "mouse_breast_cancer_sample1_section1&2.h5ad"
)

save_dir = (
    "/data2/liangyefeng/My_GraphST_Innovation/Result/"
    "MouseBreastCancer_vertical_integration_h5ad"
)
os.makedirs(save_dir, exist_ok=True)

SEED = 50
EPOCHS = 600
DIM_OUTPUT = 64
N_CLUSTERS = 20
GAMMA = 2.5

# Baseline GraphST
BASE_WS = 0.0
BASE_WSH = 0.0

# Ada-GraphST
OURS_WS = 1.7
OURS_WSH = 0.7

# =========================================================
# 2. 工具函数
# =========================================================
def seed_everything(seed=50):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_batch_exists(adata):
    if "batch" not in adata.obs.columns:
        print(f"⚠️ 找不到 'batch' 列，当前 obs 列名有: {list(adata.obs.columns)}")

        if "data" in adata.obs.columns and len(adata.obs["data"].unique()) > 1:
            print("💡 检测到 'data' 列，已将其作为 batch 标签。")
            adata.obs["batch"] = adata.obs["data"].astype(str)

        else:
            print("💡 尝试从 barcode 后缀中提取 batch 信息。")
            suffixes = [str(name).split("-")[-1] for name in adata.obs_names]
            adata.obs["batch"] = suffixes
            print(f"✅ 成功从 barcode 中提取到 {len(set(suffixes))} 个 batch。")

    adata.obs["batch"] = adata.obs["batch"].astype(str).astype("category")
    return adata


def compute_emb_pca(adata):
    if "emb" not in adata.obsm:
        raise KeyError("❌ GraphST 输出中不存在 adata.obsm['emb']")

    emb = np.asarray(adata.obsm["emb"], dtype=np.float64)

    if np.isnan(emb).any() or np.isinf(emb).any():
        raise ValueError("❌ emb 中存在 NaN 或 Inf，无法继续分析。")

    adata.obsm["emb_pca"] = PCA(
        n_components=min(20, emb.shape[1]),
        random_state=SEED
    ).fit_transform(emb)

    return adata


def ensure_domain_exists(adata):
    if "domain" not in adata.obs.columns:
        if "mclust" in adata.obs.columns:
            adata.obs["domain"] = adata.obs["mclust"].astype(str).astype("category")
        else:
            raise KeyError("❌ 未找到 domain 或 mclust 列。")
    else:
        adata.obs["domain"] = adata.obs["domain"].astype(str).astype("category")

    return adata


# =========================================================
# 3. 修复 mclust，失败时自动回退到 GaussianMixture
# =========================================================
def patched_mclust_R(
    adata,
    num_cluster,
    modelNames="EEE",
    used_obsm="emb_pca",
    random_seed=2020
):
    import rpy2.robjects as robjects
    from rpy2.robjects import r, numpy2ri

    if used_obsm not in adata.obsm:
        raise KeyError(f"❌ adata.obsm['{used_obsm}'] 不存在。")

    x = np.asarray(adata.obsm[used_obsm], dtype=np.float64)

    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError(f"❌ {used_obsm} 中存在 NaN 或 Inf，无法聚类。")

    col_var = np.var(x, axis=0)
    keep = col_var > 1e-12
    x_use = x[:, keep]

    try:
        numpy2ri.activate()
        robjects.globalenv["x_mat"] = x_use
        robjects.globalenv["n_cluster"] = int(num_cluster)
        robjects.globalenv["model_name"] = modelNames
        robjects.globalenv["seed"] = int(random_seed)

        r(
            """
            suppressMessages(library(mclust))
            set.seed(seed)
            x_mat <- as.matrix(x_mat)
            dimnames(x_mat) <- NULL

            cls <- NULL
            tryCatch({
                res <- Mclust(x_mat, G=n_cluster, modelNames=model_name, verbose=FALSE)
                if (!is.null(res) && !is.null(res$classification)) {
                    cls <- as.integer(res$classification)
                }
            }, error=function(e) {
                cls <- NULL
            })
            """
        )

        cls_r = r["cls"]

        if cls_r is not None and len(cls_r) == x_use.shape[0]:
            mclust_res = np.array(cls_r).astype(int)
            adata.obs["mclust"] = pd.Categorical(mclust_res)
            return adata

        print("⚠️ mclust 返回空结果，改用 sklearn GaussianMixture。")

    except Exception as e:
        print(f"⚠️ mclust 调用失败，改用 sklearn GaussianMixture。原因: {e}")

    gmm = GaussianMixture(
        n_components=num_cluster,
        covariance_type="full",
        reg_covar=1e-5,
        random_state=random_seed
    )

    pred = gmm.fit_predict(x_use) + 1
    adata.obs["mclust"] = pd.Categorical(pred)

    return adata


graphst_utils.mclust_R = patched_mclust_R


# =========================================================
# 4. 运行单个模型并保存 h5ad
# =========================================================
def run_one_model(
    adata_input,
    ws,
    wsh,
    gamma,
    device,
    refine_flag,
    method_name,
    save_path
):
    print("\n" + "=" * 80)
    print(f"⏳ 正在运行 {method_name}")
    print("=" * 80)
    print(f"w_smooth={ws}, w_sharpen={wsh}, gamma={gamma}")

    adata_run = adata_input.copy()

    model = GraphST(
        adata_run,
        device=device,
        random_seed=SEED,
        epochs=EPOCHS,
        dim_output=DIM_OUTPUT,
        datatype="10X",
        w_smooth=ws,
        w_sharpen=wsh,
        gamma=gamma,
        use_learnable_proj=False
    )

    adata_run = model.train()

    print("   -> 计算 emb_pca")
    adata_run = compute_emb_pca(adata_run)

    print("   -> 在 integrated embedding 上进行 mclust 聚类")
    clustering(
        adata_run,
        N_CLUSTERS,
        method="mclust",
        refinement=refine_flag
    )

    adata_run = ensure_domain_exists(adata_run)

    adata_run.uns["method"] = method_name
    adata_run.uns["dataset"] = PAIR_NAME
    adata_run.uns["w_smooth"] = ws
    adata_run.uns["w_sharpen"] = wsh
    adata_run.uns["gamma"] = gamma
    adata_run.uns["n_clusters"] = N_CLUSTERS
    adata_run.uns["seed"] = SEED

    print(f"   -> 保存 h5ad: {save_path}")
    adata_run.write(save_path)

    del model
    clear_memory()

    return save_path


# =========================================================
# 5. 主流程
# =========================================================
if __name__ == "__main__":
    seed_everything(SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 80)
    print("🧬 Mouse Breast Cancer integration: generate h5ad only")
    print("=" * 80)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ 找不到数据文件: {data_path}")

    print(f"📥 正在读取数据: {data_path}")
    adata_raw = sc.read_h5ad(data_path)
    adata_raw.var_names_make_unique()
    adata_raw = ensure_batch_exists(adata_raw)

    print(f"✅ adata shape: {adata_raw.shape}")
    print(f"✅ batch categories: {list(adata_raw.obs['batch'].cat.categories)}")
    print(f"✅ obsm keys: {list(adata_raw.obsm.keys())}")

    baseline_h5ad = os.path.join(
        save_dir,
        "MouseBreastCancer_baseline_result.h5ad"
    )

    ours_h5ad = os.path.join(
        save_dir,
        "MouseBreastCancer_ours_result.h5ad"
    )

    run_one_model(
        adata_input=adata_raw,
        ws=BASE_WS,
        wsh=BASE_WSH,
        gamma=GAMMA,
        device=device,
        refine_flag=True,
        method_name="GraphST",
        save_path=baseline_h5ad
    )

    run_one_model(
        adata_input=adata_raw,
        ws=OURS_WS,
        wsh=OURS_WSH,
        gamma=GAMMA,
        device=device,
        refine_flag=False,
        method_name="Ada-GraphST",
        save_path=ours_h5ad
    )

    print("\n✅ h5ad 文件生成完成：")
    print(f"   Baseline GraphST: {baseline_h5ad}")
    print(f"   Ada-GraphST:      {ours_h5ad}")