# -*- coding: utf-8 -*-

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc

# =========================================================
# 0. 环境配置
# =========================================================
os.environ["R_HOME"] = "/data2/liangyefeng/miniconda3/envs/graphst/lib/R"

project_root = "/data2/liangyefeng/My_GraphST_Innovation"
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from STAGATE.STAGATE.Train_STAGATE import train_STAGATE
from STAGATE.STAGATE.utils import Cal_Spatial_Net

from mouse_embryo_E16E2_common import (
    DATA_DIR,
    FORCE_LABEL_COL,
    SPATIAL_BIN_SIZE,
    SEED,
    seed_everything,
    load_binned_section,
    compute_ari_nmi,
    mclust_R,
)


# =========================================================
# 1. 只跑 E16.5_E2S3
# =========================================================
DATA_PATH = os.path.join(
    DATA_DIR,
    "E16.5_E2S3.MOSTA.h5ad"
)

sample_name = os.path.basename(DATA_PATH).replace(".MOSTA.h5ad", "")


# =========================================================
# 2. STAGATE 参数
# =========================================================
# 对 bin 后坐标更稳
GRAPH_MODEL = "KNN"
K_CUTOFF = 10

HIDDEN_DIMS = [512, 30]
ALPHA = 0
N_EPOCHS = 500
LR = 1e-4
WEIGHT_DECAY = 1e-4


# =========================================================
# 3. 输出路径
# =========================================================
RESULT_CSV = os.path.join(
    DATA_DIR,
    f"{sample_name}_STAGATE_result.csv"
)

OUTPUT_H5AD = os.path.join(
    DATA_DIR,
    f"{sample_name}_STAGATE_bin{SPATIAL_BIN_SIZE}_{GRAPH_MODEL}_k{K_CUTOFF}.h5ad"
)


# =========================================================
# 4. STAGATE 预处理
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
# 5. 写 h5ad 前清理函数
# =========================================================
def _clean_dataframe_for_h5ad(df):
    """
    清理 DataFrame 中的 object 列，降低 h5ad 写入失败概率。
    """
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = df[col].astype("category")
            except Exception:
                df[col] = df[col].astype(str)

    return df


def prepare_adata_for_write(adata):
    """
    写入 h5ad 前做轻量清理，避免 object 类型字段导致 AnnData 写入失败。
    """
    adata = adata.copy()

    adata.obs = _clean_dataframe_for_h5ad(adata.obs)
    adata.var = _clean_dataframe_for_h5ad(adata.var)

    for key, value in list(adata.uns.items()):
        if isinstance(value, pd.DataFrame):
            adata.uns[key] = _clean_dataframe_for_h5ad(value)

    adata.strings_to_categoricals()

    return adata


# =========================================================
# 6. 主流程：STAGATE 训练 + mclust 聚类 + 保存 h5ad
# =========================================================
seed_everything(SEED, use_tf=True)

results = []

print("\n" + "=" * 100)
print(f"START SAMPLE: {sample_name}")
print("=" * 100)

try:
    # -----------------------------------------------------
    # 6.1 加载并 binning
    # -----------------------------------------------------
    seed_everything(SEED, use_tf=True)

    adata, info = load_binned_section(
        DATA_PATH,
        force_label_col=FORCE_LABEL_COL,
        bin_size=SPATIAL_BIN_SIZE
    )

    print(
        f"[{sample_name}] "
        f"n_obs={adata.n_obs}, "
        f"n_vars={adata.n_vars}, "
        f"n_clusters={info['n_clusters']}"
    )

    # -----------------------------------------------------
    # 6.2 STAGATE 预处理
    # -----------------------------------------------------
    adata = preprocess_for_stagate(
        adata,
        n_top_genes=3000
    )

    print(
        f"[{sample_name}] after preprocess: "
        f"n_obs={adata.n_obs}, "
        f"n_vars={adata.n_vars}"
    )

    # -----------------------------------------------------
    # 6.3 构建空间网络
    # -----------------------------------------------------
    Cal_Spatial_Net(
        adata,
        k_cutoff=K_CUTOFF,
        model=GRAPH_MODEL,
        verbose=True
    )

    # -----------------------------------------------------
    # 6.4 训练 STAGATE
    # -----------------------------------------------------
    tf.reset_default_graph()

    seed_everything(SEED, use_tf=True)

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
        random_seed=SEED,
        save_attention=False,
        save_loss=False,
        save_reconstrction=False,
    )

    if "STAGATE" not in adata.obsm:
        raise KeyError(
            f"STAGATE embedding not found in adata.obsm. "
            f"Current keys: {list(adata.obsm.keys())}"
        )

    print(
        f"[{sample_name}] STAGATE embedding shape: "
        f"{adata.obsm['STAGATE'].shape}"
    )

    # -----------------------------------------------------
    # 6.5 mclust 聚类，结果写入 adata.obs["domain"]
    # -----------------------------------------------------
    adata = mclust_R(
        adata,
        num_cluster=info["n_clusters"],
        used_obsm="STAGATE",
        key_added="domain",
        random_seed=SEED
    )

    # -----------------------------------------------------
    # 6.6 计算 ARI / NMI
    # -----------------------------------------------------
    ari, nmi, n_eval = compute_ari_nmi(
        adata,
        pred_col="domain",
        gt_col="ground_truth"
    )

    print(f"[{sample_name}] STAGATE | ARI={ari:.6f} | NMI={nmi:.6f}")

    # -----------------------------------------------------
    # 6.7 保存运行信息到 adata.uns
    # -----------------------------------------------------
    adata.uns["method"] = "STAGATE"
    adata.uns["sample_name"] = sample_name
    adata.uns["seed"] = int(SEED)
    adata.uns["bin_size"] = int(SPATIAL_BIN_SIZE)
    adata.uns["label_col"] = str(info["label_col"])
    adata.uns["graph_model"] = GRAPH_MODEL
    adata.uns["k_cutoff"] = int(K_CUTOFF)
    adata.uns["hidden_dims"] = np.asarray(HIDDEN_DIMS, dtype=np.int64)
    adata.uns["alpha"] = float(ALPHA)
    adata.uns["n_epochs"] = int(N_EPOCHS)
    adata.uns["lr"] = float(LR)
    adata.uns["weight_decay"] = float(WEIGHT_DECAY)
    adata.uns["n_clusters"] = int(info["n_clusters"])
    adata.uns["n_obs"] = int(adata.n_obs)
    adata.uns["n_obs_eval"] = int(n_eval)
    adata.uns["ARI"] = float(ari)
    adata.uns["NMI"] = float(nmi)

    adata.uns["STAGATE_params_json"] = json.dumps(
        {
            "method": "STAGATE",
            "sample": sample_name,
            "seed": int(SEED),
            "bin_size": int(SPATIAL_BIN_SIZE),
            "force_label_col": FORCE_LABEL_COL,
            "preprocess": {
                "filter_genes_min_cells": 3,
                "normalize_total_target_sum": 1e4,
                "log1p": True,
                "highly_variable_genes_flavor": "seurat",
                "n_top_genes": 3000,
            },
            "spatial_graph": {
                "model": GRAPH_MODEL,
                "k_cutoff": int(K_CUTOFF),
            },
            "train": {
                "hidden_dims": HIDDEN_DIMS,
                "alpha": ALPHA,
                "n_epochs": N_EPOCHS,
                "lr": LR,
                "key_added": "STAGATE",
                "gradient_clipping": 5,
                "nonlinear": True,
                "weight_decay": WEIGHT_DECAY,
                "save_attention": False,
                "save_loss": False,
                "save_reconstrction": False,
            },
            "clustering": {
                "method": "mclust",
                "used_obsm": "STAGATE",
                "key_added": "domain",
                "num_cluster": int(info["n_clusters"]),
            },
            "ARI": float(ari),
            "NMI": float(nmi),
            "n_obs_eval": int(n_eval),
        },
        ensure_ascii=False
    )

    # -----------------------------------------------------
    # 6.8 保存 h5ad
    # -----------------------------------------------------
    adata_to_save = prepare_adata_for_write(adata)

    adata_to_save.write_h5ad(
        OUTPUT_H5AD,
        compression="gzip"
    )

    print(f"[{sample_name}] Saved h5ad: {OUTPUT_H5AD}")

    # -----------------------------------------------------
    # 6.9 保存 CSV 记录
    # -----------------------------------------------------
    results.append({
        "Sample": sample_name,
        "Method": "STAGATE",
        "Seed": SEED,
        "Bin_Size": SPATIAL_BIN_SIZE,
        "Graph_Model": GRAPH_MODEL,
        "K_Cutoff": K_CUTOFF,
        "Hidden_Dims": str(HIDDEN_DIMS),
        "Alpha": ALPHA,
        "N_Epochs": N_EPOCHS,
        "LR": LR,
        "Weight_Decay": WEIGHT_DECAY,
        "Label_Col": info["label_col"],
        "N_Clusters": info["n_clusters"],
        "N_Obs": int(adata.n_obs),
        "N_Obs_Eval": int(n_eval),
        "ARI": float(ari),
        "NMI": float(nmi),
        "H5AD_Path": OUTPUT_H5AD,
        "Status": "OK",
        "Error": "",
    })

except Exception as e:
    print(f"[{sample_name}] FAILED: {e}")

    results.append({
        "Sample": sample_name,
        "Method": "STAGATE",
        "Seed": SEED,
        "Bin_Size": SPATIAL_BIN_SIZE,
        "Graph_Model": GRAPH_MODEL,
        "K_Cutoff": K_CUTOFF,
        "Hidden_Dims": str(HIDDEN_DIMS),
        "Alpha": ALPHA,
        "N_Epochs": N_EPOCHS,
        "LR": LR,
        "Weight_Decay": WEIGHT_DECAY,
        "Label_Col": FORCE_LABEL_COL,
        "N_Clusters": np.nan,
        "N_Obs": np.nan,
        "N_Obs_Eval": np.nan,
        "ARI": np.nan,
        "NMI": np.nan,
        "H5AD_Path": "",
        "Status": "FAILED",
        "Error": str(e),
    })


# =========================================================
# 7. 保存结果 CSV
# =========================================================
results_df = pd.DataFrame(results)
results_df.to_csv(RESULT_CSV, index=False)

print("\n" + "=" * 100)
print("FINISHED")
print("=" * 100)

print("Saved CSV:")
print(RESULT_CSV)

print("Saved h5ad:")
print(OUTPUT_H5AD)

print(results_df)