# -*- coding: utf-8 -*-

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
import networkx as nx
from sklearn.decomposition import PCA

# =========================================================
# 0. 环境与路径
# =========================================================
os.environ["R_HOME"] = "/data2/liangyefeng/miniconda3/envs/graphst/lib/R"

project_root = "/data2/liangyefeng/My_GraphST_Innovation"
src_path = os.path.join(project_root, "src")
spaceflow_path = os.path.join(project_root, "src", "SpaceFlow")

if src_path not in sys.path:
    sys.path.insert(0, src_path)

if spaceflow_path not in sys.path:
    sys.path.insert(0, spaceflow_path)

DATA_DIR = "/data2/liangyefeng/My_GraphST_Innovation/data/Mouse-embryo"

DATA_PATH = os.path.join(
    DATA_DIR,
    "E16.5_E2S3.MOSTA.h5ad"
)

sample_name = os.path.basename(DATA_PATH).replace(".MOSTA.h5ad", "")


# =========================================================
# 1. NetworkX 3.x compatibility patch
# =========================================================
if not hasattr(nx, "to_scipy_sparse_matrix"):
    def _to_scipy_sparse_matrix(
        G,
        nodelist=None,
        dtype=None,
        weight="weight",
        format="csr"
    ):
        arr = nx.to_scipy_sparse_array(
            G,
            nodelist=nodelist,
            dtype=dtype,
            weight=weight,
            format=format,
        )

        if hasattr(arr, "asformat"):
            return arr.asformat(format)

        return sp.csr_matrix(arr)

    nx.to_scipy_sparse_matrix = _to_scipy_sparse_matrix


# =========================================================
# 2. 导入模型和公共函数
# =========================================================
from SpaceFlow.SpaceFlow import SpaceFlow

from GraphST.utils import clustering

from mouse_embryo_E16E2_common import (
    FORCE_LABEL_COL,
    SPATIAL_BIN_SIZE,
    SEED,
    seed_everything,
    load_binned_section,
    compute_ari_nmi,
    patch_graphst_mclust,
)

patch_graphst_mclust()


# =========================================================
# 3. SpaceFlow 参数
# =========================================================
N_TOP_GENES = 3000
SPATIAL_N_NEIGHBORS = 10

Z_DIM = 50
LR = 1e-3
EPOCHS = 1000
MAX_PATIENCE = 50
MIN_STOP = 100
SPATIAL_REG_STRENGTH = 0.1

GPU_ID = 0 if torch.cuda.is_available() else None

REFINE_RADIUS = 50


# =========================================================
# 4. 输出路径
# =========================================================
RESULT_CSV = os.path.join(
    DATA_DIR,
    f"{sample_name}_SpaceFlow_result.csv"
)

OUTPUT_H5AD = os.path.join(
    DATA_DIR,
    f"{sample_name}_SpaceFlow_bin{SPATIAL_BIN_SIZE}_z{Z_DIM}_radius{REFINE_RADIUS}.h5ad"
)

emb_dir = os.path.join(
    DATA_DIR,
    "spaceflow_embeddings"
)

os.makedirs(emb_dir, exist_ok=True)

EMB_PATH = os.path.join(
    emb_dir,
    f"{sample_name}_embedding.tsv"
)


# =========================================================
# 5. h5ad 写入前清理函数
# =========================================================
def _clean_dataframe_for_h5ad(df):
    """
    清理 DataFrame 里的 object 列，降低 h5ad 写入失败概率。
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


def load_spaceflow_embedding_from_tsv(emb_path):
    """
    如果 SpaceFlow.train 返回 None，但 embedding 文件已保存，则从 tsv 中读取。
    这是兜底逻辑，正常情况下不会触发。
    """
    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f"SpaceFlow embedding is None and file not found: {emb_path}"
        )

    emb_df = pd.read_csv(
        emb_path,
        sep="\t",
        header=None
    )

    return emb_df.values


# =========================================================
# 6. 主流程：只跑 E16.5_E2S3
# =========================================================
seed_everything(SEED)

results = []

print("\n" + "=" * 100)
print(f"START SAMPLE: {sample_name}")
print("=" * 100)

try:
    # -----------------------------------------------------
    # 6.1 加载并 binning
    # -----------------------------------------------------
    seed_everything(SEED)

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

    print(f"[{sample_name}] label_col={info['label_col']}")
    print(f"[{sample_name}] ground_truth counts:")
    print(adata.obs["ground_truth"].value_counts(dropna=False))

    # -----------------------------------------------------
    # 6.2 初始化 SpaceFlow
    # -----------------------------------------------------
    sf = SpaceFlow(
        adata=adata.copy()
    )

    # -----------------------------------------------------
    # 6.3 SpaceFlow 预处理
    # -----------------------------------------------------
    n_top_genes = min(N_TOP_GENES, adata.n_vars)

    sf.preprocessing_data(
        n_top_genes=n_top_genes,
        n_neighbors=SPATIAL_N_NEIGHBORS
    )

    # -----------------------------------------------------
    # 6.4 训练 SpaceFlow
    # -----------------------------------------------------
    seed_everything(SEED)

    embedding = sf.train(
        embedding_save_filepath=EMB_PATH,
        spatial_regularization_strength=SPATIAL_REG_STRENGTH,
        z_dim=Z_DIM,
        lr=LR,
        epochs=EPOCHS,
        max_patience=MAX_PATIENCE,
        min_stop=MIN_STOP,
        random_seed=SEED,
        gpu=GPU_ID,
        regularization_acceleration=True,
        edge_subset_sz=1000000
    )

    if embedding is None:
        print(
            f"[{sample_name}] SpaceFlow.train returned None, "
            f"try loading embedding from: {EMB_PATH}"
        )
        embedding = load_spaceflow_embedding_from_tsv(EMB_PATH)

    embedding = np.asarray(embedding)

    if embedding.ndim != 2:
        raise ValueError(
            f"SpaceFlow embedding must be 2D, got shape={embedding.shape}"
        )

    if embedding.shape[0] != adata.n_obs:
        raise ValueError(
            f"Embedding rows ({embedding.shape[0]}) "
            f"!= adata.n_obs ({adata.n_obs})"
        )

    print(f"[{sample_name}] SpaceFlow embedding shape: {embedding.shape}")

    # -----------------------------------------------------
    # 6.5 保存 embedding 到 adata.obsm
    # -----------------------------------------------------
    adata.obsm["SpaceFlow"] = embedding.copy()
    adata.obsm["emb"] = embedding.copy()

    # -----------------------------------------------------
    # 6.6 PCA，用于 GraphST clustering 的 mclust 输入
    # -----------------------------------------------------
    n_pcs = min(
        20,
        adata.obsm["emb"].shape[0],
        adata.obsm["emb"].shape[1]
    )

    if n_pcs < 2:
        raise ValueError(
            f"Cannot run PCA because n_pcs={n_pcs}, "
            f"embedding shape={adata.obsm['emb'].shape}"
        )

    adata.obsm["emb_pca"] = PCA(
        n_components=n_pcs,
        random_state=SEED
    ).fit_transform(adata.obsm["emb"])

    print(f"[{sample_name}] emb_pca shape: {adata.obsm['emb_pca'].shape}")

    # -----------------------------------------------------
    # 6.7 mclust 聚类 + refinement
    # 聚类结果写入 adata.obs["domain"]
    # -----------------------------------------------------
    clustering(
        adata,
        info["n_clusters"],
        radius=REFINE_RADIUS,
        method="mclust",
        refinement=True
    )

    if "domain" not in adata.obs.columns:
        raise KeyError(
            f"clustering finished but adata.obs['domain'] not found. "
            f"obs columns: {list(adata.obs.columns)}"
        )

    # -----------------------------------------------------
    # 6.8 计算 ARI / NMI
    # -----------------------------------------------------
    ari, nmi, n_eval = compute_ari_nmi(
        adata,
        pred_col="domain",
        gt_col="ground_truth"
    )

    print(f"[{sample_name}] SpaceFlow | ARI={ari:.6f} | NMI={nmi:.6f}")

    # -----------------------------------------------------
    # 6.9 保存运行信息到 adata.uns
    # -----------------------------------------------------
    adata.uns["method"] = "SpaceFlow"
    adata.uns["sample_name"] = sample_name
    adata.uns["seed"] = int(SEED)
    adata.uns["bin_size"] = int(SPATIAL_BIN_SIZE)
    adata.uns["label_col"] = str(info["label_col"])
    adata.uns["n_clusters"] = int(info["n_clusters"])
    adata.uns["n_obs"] = int(adata.n_obs)
    adata.uns["n_obs_eval"] = int(n_eval)
    adata.uns["n_top_genes"] = int(n_top_genes)
    adata.uns["spatial_n_neighbors"] = int(SPATIAL_N_NEIGHBORS)
    adata.uns["z_dim"] = int(Z_DIM)
    adata.uns["lr"] = float(LR)
    adata.uns["epochs"] = int(EPOCHS)
    adata.uns["max_patience"] = int(MAX_PATIENCE)
    adata.uns["min_stop"] = int(MIN_STOP)
    adata.uns["spatial_regularization_strength"] = float(SPATIAL_REG_STRENGTH)
    adata.uns["gpu_id"] = -1 if GPU_ID is None else int(GPU_ID)
    adata.uns["refine_radius"] = int(REFINE_RADIUS)
    adata.uns["embedding_tsv_path"] = str(EMB_PATH)
    adata.uns["ARI"] = float(ari)
    adata.uns["NMI"] = float(nmi)

    adata.uns["SpaceFlow_params_json"] = json.dumps(
        {
            "method": "SpaceFlow",
            "sample": sample_name,
            "seed": int(SEED),
            "bin_size": int(SPATIAL_BIN_SIZE),
            "force_label_col": FORCE_LABEL_COL,
            "preprocessing": {
                "n_top_genes": int(n_top_genes),
                "n_neighbors": int(SPATIAL_N_NEIGHBORS),
            },
            "train": {
                "embedding_save_filepath": EMB_PATH,
                "spatial_regularization_strength": SPATIAL_REG_STRENGTH,
                "z_dim": Z_DIM,
                "lr": LR,
                "epochs": EPOCHS,
                "max_patience": MAX_PATIENCE,
                "min_stop": MIN_STOP,
                "random_seed": SEED,
                "gpu": GPU_ID,
                "regularization_acceleration": True,
                "edge_subset_sz": 1000000,
            },
            "pca": {
                "used_obsm": "emb",
                "key_added": "emb_pca",
                "n_pcs": int(n_pcs),
            },
            "clustering": {
                "method": "mclust",
                "used_obsm": "emb_pca",
                "refinement": True,
                "radius": REFINE_RADIUS,
                "num_cluster": int(info["n_clusters"]),
                "key_added": "domain",
            },
            "metrics": {
                "ARI": float(ari),
                "NMI": float(nmi),
                "n_obs_eval": int(n_eval),
            },
        },
        ensure_ascii=False
    )

    # -----------------------------------------------------
    # 6.10 保存 h5ad
    # -----------------------------------------------------
    adata_to_save = prepare_adata_for_write(adata)

    adata_to_save.write_h5ad(
        OUTPUT_H5AD,
        compression="gzip"
    )

    print(f"[{sample_name}] Saved h5ad: {OUTPUT_H5AD}")

    # -----------------------------------------------------
    # 6.11 保存 CSV 记录
    # -----------------------------------------------------
    results.append({
        "Sample": sample_name,
        "Method": "SpaceFlow",
        "Seed": SEED,
        "Bin_Size": SPATIAL_BIN_SIZE,
        "Label_Col": info["label_col"],
        "N_Top_Genes": int(n_top_genes),
        "Spatial_N_Neighbors": SPATIAL_N_NEIGHBORS,
        "Z_Dim": Z_DIM,
        "LR": LR,
        "Epochs": EPOCHS,
        "Max_Patience": MAX_PATIENCE,
        "Min_Stop": MIN_STOP,
        "Spatial_Reg_Strength": SPATIAL_REG_STRENGTH,
        "GPU_ID": GPU_ID,
        "Refine_Radius": REFINE_RADIUS,
        "N_Clusters": info["n_clusters"],
        "N_Obs": int(adata.n_obs),
        "N_Obs_Eval": int(n_eval),
        "ARI": float(ari),
        "NMI": float(nmi),
        "Embedding_TSV_Path": EMB_PATH,
        "H5AD_Path": OUTPUT_H5AD,
        "Status": "OK",
        "Error": "",
    })

except Exception as e:
    print(f"[{sample_name}] FAILED: {e}")

    results.append({
        "Sample": sample_name,
        "Method": "SpaceFlow",
        "Seed": SEED,
        "Bin_Size": SPATIAL_BIN_SIZE,
        "Label_Col": FORCE_LABEL_COL,
        "N_Top_Genes": N_TOP_GENES,
        "Spatial_N_Neighbors": SPATIAL_N_NEIGHBORS,
        "Z_Dim": Z_DIM,
        "LR": LR,
        "Epochs": EPOCHS,
        "Max_Patience": MAX_PATIENCE,
        "Min_Stop": MIN_STOP,
        "Spatial_Reg_Strength": SPATIAL_REG_STRENGTH,
        "GPU_ID": GPU_ID,
        "Refine_Radius": REFINE_RADIUS,
        "N_Clusters": np.nan,
        "N_Obs": np.nan,
        "N_Obs_Eval": np.nan,
        "ARI": np.nan,
        "NMI": np.nan,
        "Embedding_TSV_Path": EMB_PATH,
        "H5AD_Path": "",
        "Status": "FAILED",
        "Error": str(e),
    })


# =========================================================
# 7. 保存结果 CSV
# =========================================================
results_df = pd.DataFrame(results)
results_df.to_csv(
    RESULT_CSV,
    index=False
)

print("\n" + "=" * 100)
print("FINISHED")
print("=" * 100)

print("Saved CSV:")
print(RESULT_CSV)

print("Saved h5ad:")
print(OUTPUT_H5AD)

print("Saved embedding TSV:")
print(EMB_PATH)

print(results_df)