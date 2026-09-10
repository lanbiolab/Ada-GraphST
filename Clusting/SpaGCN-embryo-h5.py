# -*- coding: utf-8 -*-

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import scanpy as sc

# =========================================================
# 0. 路径配置
# =========================================================
project_root = "/data2/liangyefeng/My_GraphST_Innovation"
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from SpaGCN.SpaGCN_package.SpaGCN.SpaGCN import SpaGCN
from SpaGCN.SpaGCN_package.SpaGCN.util import (
    prefilter_genes,
    prefilter_specialgenes,
    search_l,
    search_res,
)
from SpaGCN.SpaGCN_package.SpaGCN.calculate_adj import calculate_adj_matrix

from mouse_embryo_E16E2_common import (
    FORCE_LABEL_COL,
    SPATIAL_BIN_SIZE,
    SEED,
    seed_everything,
    load_binned_section,
    compute_ari_nmi,
)


# =========================================================
# 1. 只跑 E16.5_E2S3
# =========================================================
DATA_DIR = "/data2/liangyefeng/My_GraphST_Innovation/data/Mouse-embryo"

DATA_PATH = os.path.join(DATA_DIR, "E16.5_E2S3.MOSTA.h5ad")

sample_name = os.path.basename(DATA_PATH).replace(".MOSTA.h5ad", "")

RESULT_CSV = os.path.join(
    DATA_DIR,
    f"{sample_name}_SpaGCN_result.csv"
)

OUTPUT_H5AD = os.path.join(
    DATA_DIR,
    f"{sample_name}_SpaGCN_bin{SPATIAL_BIN_SIZE}.h5ad"
)


# =========================================================
# 2. 写 h5ad 前清理函数
# =========================================================
def prepare_adata_for_write(adata):
    """
    写入 h5ad 前做轻量清理，避免 object 类型字段导致 AnnData 写入失败。
    """
    adata = adata.copy()

    for col in adata.obs.columns:
        if adata.obs[col].dtype == "object":
            try:
                adata.obs[col] = adata.obs[col].astype("category")
            except Exception:
                adata.obs[col] = adata.obs[col].astype(str)

    for col in adata.var.columns:
        if adata.var[col].dtype == "object":
            try:
                adata.var[col] = adata.var[col].astype("category")
            except Exception:
                adata.var[col] = adata.var[col].astype(str)

    adata.strings_to_categoricals()

    return adata


# =========================================================
# 3. 主流程：SpaGCN 训练 + 保存 h5ad
# =========================================================
seed_everything(SEED)

results = []

print("\n" + "=" * 100)
print(f"START SAMPLE: {sample_name}")
print("=" * 100)

try:
    # -----------------------------------------------------
    # 3.1 加载并 binning
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

    # -----------------------------------------------------
    # 3.2 SpaGCN 预处理
    # -----------------------------------------------------
    prefilter_genes(adata, min_cells=3)
    prefilter_specialgenes(adata)

    try:
        sc.pp.normalize_per_cell(adata)
    except Exception:
        sc.pp.normalize_total(adata)

    sc.pp.log1p(adata)

    # -----------------------------------------------------
    # 3.3 构建空间邻接矩阵
    # -----------------------------------------------------
    x = adata.obsm["spatial"][:, 0].tolist()
    y = adata.obsm["spatial"][:, 1].tolist()

    adj = calculate_adj_matrix(
        x=x,
        y=y,
        histology=False
    )

    # -----------------------------------------------------
    # 3.4 搜索 SpaGCN 参数 l
    # -----------------------------------------------------
    l = search_l(
        0.5,
        adj,
        start=0.01,
        end=1000,
        tol=0.01,
        max_run=100
    )

    if l is None:
        l = 1.0

    print(f"[{sample_name}] recommended l = {l}")

    # -----------------------------------------------------
    # 3.5 搜索分辨率 res
    # -----------------------------------------------------
    seed_everything(SEED)

    res = search_res(
        adata,
        adj,
        l,
        target_num=info["n_clusters"],
        start=0.7,
        step=0.1,
        tol=5e-3,
        lr=0.05,
        max_epochs=20,
        r_seed=SEED,
        t_seed=SEED,
        n_seed=SEED,
        max_run=20
    )

    if res is None:
        res = 1.0

    print(f"[{sample_name}] recommended res = {res}")

    # -----------------------------------------------------
    # 3.6 训练 SpaGCN
    # -----------------------------------------------------
    clf = SpaGCN()
    clf.set_l(l)

    seed_everything(SEED)

    clf.train(
        adata,
        adj,
        init_spa=True,
        init="louvain",
        res=res,
        tol=5e-3,
        lr=0.05,
        max_epochs=200
    )

    # -----------------------------------------------------
    # 3.7 预测空间 domain
    # -----------------------------------------------------
    pred, _ = clf.predict()

    pred = np.asarray(pred).astype(int)
    adata.obs["domain"] = pd.Categorical(pred)

    # -----------------------------------------------------
    # 3.8 计算 ARI / NMI
    # -----------------------------------------------------
    ari, nmi, n_eval = compute_ari_nmi(
        adata,
        pred_col="domain",
        gt_col="ground_truth"
    )

    print(f"[{sample_name}] SpaGCN | ARI={ari:.6f} | NMI={nmi:.6f}")

    # -----------------------------------------------------
    # 3.9 保存运行信息到 adata.uns
    # -----------------------------------------------------
    adata.uns["method"] = "SpaGCN"
    adata.uns["sample_name"] = sample_name
    adata.uns["seed"] = int(SEED)
    adata.uns["bin_size"] = int(SPATIAL_BIN_SIZE)
    adata.uns["label_col"] = str(info["label_col"])
    adata.uns["n_clusters"] = int(info["n_clusters"])
    adata.uns["n_obs"] = int(adata.n_obs)
    adata.uns["n_obs_eval"] = int(n_eval)
    adata.uns["recommended_l"] = float(l)
    adata.uns["recommended_res"] = float(res)
    adata.uns["ARI"] = float(ari)
    adata.uns["NMI"] = float(nmi)

    adata.uns["SpaGCN_params_json"] = json.dumps(
        {
            "method": "SpaGCN",
            "sample": sample_name,
            "seed": int(SEED),
            "bin_size": int(SPATIAL_BIN_SIZE),
            "force_label_col": FORCE_LABEL_COL,
            "prefilter_genes_min_cells": 3,
            "histology": False,
            "search_l": {
                "p": 0.5,
                "start": 0.01,
                "end": 1000,
                "tol": 0.01,
                "max_run": 100,
            },
            "search_res": {
                "target_num": int(info["n_clusters"]),
                "start": 0.7,
                "step": 0.1,
                "tol": 5e-3,
                "lr": 0.05,
                "max_epochs": 20,
                "max_run": 20,
            },
            "train": {
                "init_spa": True,
                "init": "louvain",
                "tol": 5e-3,
                "lr": 0.05,
                "max_epochs": 200,
            },
            "recommended_l": float(l),
            "recommended_res": float(res),
            "ARI": float(ari),
            "NMI": float(nmi),
        },
        ensure_ascii=False
    )

    # -----------------------------------------------------
    # 3.10 保存 h5ad
    # 注意：不保存 adj 到 h5ad，避免文件过大
    # -----------------------------------------------------
    adata_to_save = prepare_adata_for_write(adata)

    adata_to_save.write_h5ad(
        OUTPUT_H5AD,
        compression="gzip"
    )

    print(f"[{sample_name}] Saved h5ad: {OUTPUT_H5AD}")

    # -----------------------------------------------------
    # 3.11 保存 CSV 结果
    # -----------------------------------------------------
    results.append({
        "Sample": sample_name,
        "Method": "SpaGCN",
        "Seed": SEED,
        "Bin_Size": SPATIAL_BIN_SIZE,
        "Label_Col": info["label_col"],
        "Recommended_l": l,
        "Recommended_res": res,
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
        "Method": "SpaGCN",
        "Seed": SEED,
        "Bin_Size": SPATIAL_BIN_SIZE,
        "Label_Col": FORCE_LABEL_COL,
        "Recommended_l": np.nan,
        "Recommended_res": np.nan,
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
# 4. 保存结果 CSV
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