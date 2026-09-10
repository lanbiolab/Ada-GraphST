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
# 0. 环境与路径配置
# =========================================================
project_root = "/data2/liangyefeng/My_GraphST_Innovation"
sys.path.insert(0, os.path.join(project_root, "src"))

from SpaGCN.SpaGCN_package.SpaGCN.SpaGCN import SpaGCN
from SpaGCN.SpaGCN_package.SpaGCN.util import (
    prefilter_genes,
    prefilter_specialgenes,
    search_l,
    search_res,
    refine,
)
from SpaGCN.SpaGCN_package.SpaGCN.calculate_adj import calculate_adj_matrix

# =========================================================
# 1. 固定随机种子
# =========================================================
seed = 41
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

print("✅ Seed fixed:", seed)

# =========================================================
# 2. 数据集与聚类数映射
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

print("✅ 待处理数据集:", datasets)

# =========================================================
# 3. SpaGCN参数
# 说明：
# - 这里按 ez_mode 的思路来
# - 默认 histology=False，只用表达 + 空间坐标
# =========================================================
HISTOLOGY = False
P_VALUE = 0.5          # search_l 用
ALPHA = 1              # calculate_adj_matrix 的 alpha
BETA = 49              # histology=True 才有意义，这里保留参数位
SEARCH_RES_START = 0.7
SEARCH_RES_STEP = 0.1
SEARCH_RES_TOL = 5e-3

SEARCH_L_START = 0.01
SEARCH_L_END = 1000
SEARCH_L_TOL = 0.01

SEARCH_STAGE_EPOCHS = 20
FINAL_STAGE_EPOCHS = 200
LR = 0.05

# =========================================================
# 4. 预处理函数
# =========================================================
def preprocess_for_spagcn(adata):
    """
    按 SpaGCN ez_mode 风格做预处理：
    - filter genes
    - remove ERCC / MT-
    - normalize per cell
    - log1p
    """
    adata = adata.copy()

    prefilter_genes(adata, min_cells=3)
    prefilter_specialgenes(adata)

    try:
        sc.pp.normalize_per_cell(adata)
    except Exception:
        # 某些 scanpy 版本里旧接口可能不稳定，做个兼容
        sc.pp.normalize_total(adata)

    sc.pp.log1p(adata)
    adata.var_names_make_unique()
    return adata


def build_adj_for_spagcn(adata, histology=False):
    """
    SpaGCN 训练阶段：
    - 官方 ez_mode 在 Visium 上，训练邻接通常用 pixel 坐标
    - refine 阶段再用 array_row / array_col
    """
    x_pixel = adata.obsm["spatial"][:, 0].tolist()
    y_pixel = adata.obsm["spatial"][:, 1].tolist()

    if histology:
        sample_key = list(adata.uns["spatial"].keys())[0]
        # hires 没有的话降级到 lowres
        if "hires" in adata.uns["spatial"][sample_key]["images"]:
            img = adata.uns["spatial"][sample_key]["images"]["hires"]
        else:
            img = adata.uns["spatial"][sample_key]["images"]["lowres"]

        adj = calculate_adj_matrix(
            x=x_pixel,
            y=y_pixel,
            x_pixel=x_pixel,
            y_pixel=y_pixel,
            image=img,
            beta=BETA,
            alpha=ALPHA,
            histology=True
        )
    else:
        adj = calculate_adj_matrix(
            x=x_pixel,
            y=y_pixel,
            histology=False
        )

    return adj


def refine_spagcn_pred(adata, pred):
    """
    SpaGCN refine 阶段使用二维 array 坐标
    """
    sample_id = adata.obs_names.tolist()
    x_array = adata.obs["array_row"].tolist()
    y_array = adata.obs["array_col"].tolist()

    adj_2d = calculate_adj_matrix(
        x=x_array,
        y=y_array,
        histology=False
    )

    refined_pred = refine(
        sample_id=sample_id,
        pred=pred,
        dis=adj_2d,
        shape="hexagon"
    )
    return refined_pred


# =========================================================
# 5. 主循环
# =========================================================
results = []

for dataset in datasets:
    print("\n" + "=" * 90)
    print(f"🚀 Processing dataset: {dataset}")
    print("=" * 90)

    file_fold = os.path.join(data_root, dataset)

    # 5.1 读取数据
    adata = sc.read_visium(
        file_fold,
        count_file="filtered_feature_bc_matrix.h5",
        load_images=True
    )
    adata.var_names_make_unique()

    # 5.2 预处理
    adata = preprocess_for_spagcn(adata)

    # 5.3 邻接矩阵
    adj = build_adj_for_spagcn(adata, histology=HISTOLOGY)

    # 5.4 当前切片真实簇数
    n_clusters = cluster_map.get(dataset, 7)
    print(f"🎯 目标簇数: {n_clusters}")

    # 5.5 搜索 l
    l = search_l(
        P_VALUE,
        adj,
        start=SEARCH_L_START,
        end=SEARCH_L_END,
        tol=SEARCH_L_TOL,
        max_run=100
    )
    if l is None:
        raise RuntimeError(f"[{dataset}] search_l failed, please adjust P_VALUE or search range.")

    print(f"✅ recommended l = {l}")

    # 5.6 搜索 resolution
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    res = search_res(
        adata,
        adj,
        l,
        target_num=n_clusters,
        start=SEARCH_RES_START,
        step=SEARCH_RES_STEP,
        tol=SEARCH_RES_TOL,
        lr=LR,
        max_epochs=SEARCH_STAGE_EPOCHS,
        r_seed=seed,
        t_seed=seed,
        n_seed=seed,
        max_run=20
    )
    if res is None:
        raise RuntimeError(f"[{dataset}] search_res failed.")

    print(f"✅ recommended res = {res}")

    # 5.7 正式训练
    clf = SpaGCN()
    clf.set_l(l)

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    clf.train(
        adata,
        adj,
        init_spa=True,
        init="louvain",
        res=res,
        tol=SEARCH_RES_TOL,
        lr=LR,
        max_epochs=FINAL_STAGE_EPOCHS
    )

    pred, prob = clf.predict()
    adata.obs["spagcn_pred_raw"] = pd.Categorical(pred.astype(int))

    # 5.8 refine
    refined_pred = refine_spagcn_pred(adata, pred)
    adata.obs["domain"] = pd.Categorical(pd.Series(refined_pred, index=adata.obs_names).astype(int))

    # 5.9 读取并对齐 ground truth
    df_meta = pd.read_csv(os.path.join(file_fold, "metadata.tsv"), sep="\t")
    barcode_col = next((c for c in df_meta.columns if "barcode" in c.lower()), None)

    if barcode_col is not None:
        df_meta = df_meta.set_index(barcode_col)
        df_meta = df_meta.loc[adata.obs_names]
        adata.obs["ground_truth"] = df_meta["layer_guess"]
    else:
        adata.obs["ground_truth"] = df_meta["layer_guess"].values

    # 5.10 去掉 NA 标注点
    adata_eval = adata[~pd.isnull(adata.obs["ground_truth"])].copy()

    # 5.11 计算 ARI / NMI
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
        "Method": "SpaGCN",
        "Histology": HISTOLOGY,
        "N_Clusters": n_clusters,
        "Recommended_l": l,
        "Recommended_res": res,
        "N_Obs_Eval": adata_eval.n_obs,
        "ARI": ari,
        "NMI": nmi,
    })

# =========================================================
# 6. 保存结果
# =========================================================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Dataset").reset_index(drop=True)

print("\n" + "#" * 90)
print("🎉 SpaGCN on DLPFC: final results")
print("#" * 90)
print(results_df.to_string(index=False))

out_csv = os.path.join(data_root, "Corrected_SpaGCN_Baseline_Results.csv")
results_df.to_csv(out_csv, index=False)
print(f"\n💾 结果已保存至: {out_csv}")