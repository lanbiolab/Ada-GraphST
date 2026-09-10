# save as: MouseEmbryo_E16.5_E2S1_E2S2_GraphST_iLISI_no_PASTE.py

import os
import sys
import gc
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import torch
import scanpy as sc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib

# 计算 iLISI
try:
    import harmonypy as hm
except ImportError:
    print("❌ 请先安装 harmonypy: pip install harmonypy")
    sys.exit(1)

# =========================================================
# 0. 环境与路径配置
# =========================================================
os.environ["R_HOME"] = "/data2/liangyefeng/miniconda3/envs/graphst/lib/R"
project_root = "/data2/liangyefeng/My_GraphST_Innovation"
sys.path.insert(0, os.path.join(project_root, "src"))

from GraphST.GraphST import GraphST

# -----------------------------
# 输入路径
# -----------------------------
slice1_path = "/data2/liangyefeng/My_GraphST_Innovation/data/Mouse-embryo/E16.5_E2S1.MOSTA.h5ad"
slice2_path = "/data2/liangyefeng/My_GraphST_Innovation/data/Mouse-embryo/E16.5_E2S2.MOSTA.h5ad"

PAIR_NAME = "MouseEmbryo_E16.5_E2S1_E2S2_no_PASTE"

save_dir = "/data2/liangyefeng/My_GraphST_Innovation/Result/MouseEmbryo_E16.5_E2S1_E2S2_no_PASTE_iLISI"
os.makedirs(save_dir, exist_ok=True)

merged_input_path = os.path.join(save_dir, f"{PAIR_NAME}_merged_input.h5ad")

# -----------------------------
# 画图字体
# -----------------------------
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "Arial"

# =========================================================
# 1. 固定参数
# =========================================================
SEED = 50
EPOCHS = 600
DIM_OUTPUT = 64
GAMMA = 2.5

# 如果你之前跑 MOSTA 一直用 "10X"，就保留 10X；
# 如果你本地 GraphST 跑 MOSTA 用的是 "Stereo"，改成 "Stereo" 即可
GRAPHST_DATATYPE = "10X"

# Baseline 参数
BASE_WS = 0.0
BASE_WSH = 0.0

# Ours 参数（你自己可改成这两个切片的最优参数）
OURS_WS = 1.5
OURS_WSH = 0.75


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


def ensure_spatial_exists(adata, adata_name="adata"):
    """
    确保 obsm['spatial'] 存在。
    如果没有，则尝试从 obs 的常见坐标列构造。
    """
    if "spatial" in adata.obsm:
        spatial = np.asarray(adata.obsm["spatial"])
        if spatial.ndim != 2 or spatial.shape[1] < 2:
            raise ValueError(f"❌ {adata_name}.obsm['spatial'] 维度异常: {spatial.shape}")
        adata.obsm["spatial"] = spatial[:, :2].astype(np.float64)
        return adata

    candidate_pairs = [
        ("x", "y"),
        ("X", "Y"),
        ("xcoord", "ycoord"),
        ("x_coord", "y_coord"),
        ("array_col", "array_row"),
        ("imagecol", "imagerow"),
    ]

    found = None
    for x_col, y_col in candidate_pairs:
        if x_col in adata.obs.columns and y_col in adata.obs.columns:
            found = (x_col, y_col)
            break

    if found is None:
        raise KeyError(
            f"❌ {adata_name} 中找不到 spatial 坐标。\n"
            f"obs 列: {list(adata.obs.columns)}\n"
            f"obsm 键: {list(adata.obsm.keys())}"
        )

    x_col, y_col = found
    adata.obsm["spatial"] = adata.obs[[x_col, y_col]].to_numpy().astype(np.float64)
    return adata


def ensure_nonnegative_X(adata, adata_name="adata"):
    """
    如果 X 有负值，则优先切换到 layers['counts']。
    """
    X = adata.X
    if sp.issparse(X):
        x_min = X.min()
    else:
        x_min = np.min(X)

    if x_min < 0:
        if "counts" in adata.layers:
            print(f"⚠️ {adata_name}.X 存在负值，已自动切换到 layers['counts']")
            adata.X = adata.layers["counts"].copy()
        else:
            raise ValueError(
                f"❌ {adata_name}.X 存在负值，但没有 layers['counts'] 可用。"
            )
    return adata


def load_one_slice(path, batch_name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 文件不存在: {path}")

    adata = sc.read_h5ad(path)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    adata = ensure_spatial_exists(adata, adata_name=batch_name)
    adata = ensure_nonnegative_X(adata, adata_name=batch_name)

    adata.obs["batch"] = batch_name
    adata.obs["slice_name"] = batch_name
    adata.obs["batch"] = adata.obs["batch"].astype("category")
    adata.obs["slice_name"] = adata.obs["slice_name"].astype("category")

    return adata


def subset_common_genes(a1, a2):
    common_genes = a1.var_names.intersection(a2.var_names)
    if len(common_genes) == 0:
        raise ValueError("❌ 两个切片没有共同基因，无法继续。")

    a1 = a1[:, common_genes].copy()
    a2 = a2[:, common_genes].copy()
    return a1, a2


def concat_two_slices(a1, a2):
    adata = ad.concat(
        [a1, a2],
        join="inner",
        merge="same",
        uns_merge="same",
        label=None,
        index_unique="-"
    )

    if "batch" not in adata.obs.columns:
        raise KeyError("❌ 合并后没有 batch 列。")

    adata.obs["batch"] = adata.obs["batch"].astype(str).astype("category")
    adata.obs["slice_name"] = adata.obs["slice_name"].astype(str).astype("category")

    if "spatial" not in adata.obsm:
        raise KeyError("❌ 合并后没有 obsm['spatial']。")

    return adata


def compute_ilisi_from_emb_pca(adata):
    lisi = hm.compute_lisi(
        adata.obsm["emb_pca"],
        adata.obs[["batch"]],
        ["batch"]
    )
    return float(np.mean(lisi))


def run_one_model(adata_input, ws, wsh, gamma, device, title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    adata_run = adata_input.copy()

    model = GraphST(
        adata_run,
        device=device,
        random_seed=SEED,
        epochs=EPOCHS,
        dim_output=DIM_OUTPUT,
        datatype=GRAPHST_DATATYPE,
        w_smooth=ws,
        w_sharpen=wsh,
        gamma=gamma,
        use_learnable_proj=False
    )
    adata_run = model.train()

    adata_run.obsm["emb_pca"] = PCA(
        n_components=min(20, adata_run.obsm["emb"].shape[1]),
        random_state=SEED
    ).fit_transform(adata_run.obsm["emb"])

    sc.pp.neighbors(adata_run, use_rep="emb_pca", random_state=SEED)
    sc.tl.umap(adata_run, random_state=SEED)

    ilisi = compute_ilisi_from_emb_pca(adata_run)

    del model
    clear_memory()

    return adata_run, ilisi


def save_spatial_batch_plot(adata, out_png, out_pdf):
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    sc.pl.embedding(
        adata,
        basis="spatial",
        color="batch",
        title="Merged input (colored by batch)",
        show=False,
        ax=ax,
        s=18,
        frameon=False,
        palette=["#5B9BD5", "#ED7D31"]
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def save_umap_compare_plot(adata_base, ilisi_base, adata_ours, ilisi_ours, out_png, out_pdf):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))

    sc.pl.umap(
        adata_base,
        color="batch",
        title=f"Baseline\nBatch iLISI = {ilisi_base:.4f}",
        show=False,
        ax=axs[0],
        frameon=False,
        s=18,
        palette=["#5B9BD5", "#ED7D31"]
    )

    sc.pl.umap(
        adata_ours,
        color="batch",
        title=f"Ours\nBatch iLISI = {ilisi_ours:.4f}",
        show=False,
        ax=axs[1],
        frameon=False,
        s=18,
        palette=["#5B9BD5", "#ED7D31"]
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


# =========================================================
# 3. 主流程
# =========================================================
if __name__ == "__main__":
    seed_everything(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 80)
    print("🧬 Mouse Embryo E16.5_E2S1 + E16.5_E2S2")
    print("   不做 PASTE，直接合并后用 GraphST 计算 iLISI")
    print("=" * 80)

    # -----------------------------------------
    # A. 读取两个切片
    # -----------------------------------------
    print(f"📥 读取切片1: {slice1_path}")
    adata1 = load_one_slice(slice1_path, "E16.5_E2S1")

    print(f"📥 读取切片2: {slice2_path}")
    adata2 = load_one_slice(slice2_path, "E16.5_E2S2")

    print(f"✅ slice1 shape: {adata1.shape}")
    print(f"✅ slice2 shape: {adata2.shape}")

    # -----------------------------------------
    # B. 共同基因
    # -----------------------------------------
    adata1, adata2 = subset_common_genes(adata1, adata2)
    print(f"✅ 共同基因后: slice1={adata1.shape}, slice2={adata2.shape}")

    # -----------------------------------------
    # C. 直接合并，不做 PASTE
    # -----------------------------------------
    adata_raw = concat_two_slices(adata1, adata2)
    adata_raw.write(merged_input_path)

    print(f"✅ merged input h5ad 已保存: {merged_input_path}")
    print(f"✅ merged adata shape: {adata_raw.shape}")
    print(f"✅ batch categories: {list(adata_raw.obs['batch'].cat.categories)}")
    print(f"✅ obsm keys: {list(adata_raw.obsm.keys())}")

    merged_spatial_png = os.path.join(save_dir, f"{PAIR_NAME}_merged_spatial_batch.png")
    merged_spatial_pdf = os.path.join(save_dir, f"{PAIR_NAME}_merged_spatial_batch.pdf")
    save_spatial_batch_plot(adata_raw, merged_spatial_png, merged_spatial_pdf)

    # -----------------------------------------
    # D. Baseline
    # -----------------------------------------
    adata_base, ilisi_base = run_one_model(
        adata_input=adata_raw,
        ws=BASE_WS,
        wsh=BASE_WSH,
        gamma=GAMMA,
        device=device,
        title="⏳ [1/2] 正在运行 Baseline GraphST ..."
    )
    baseline_h5ad = os.path.join(save_dir, f"{PAIR_NAME}_baseline_result.h5ad")
    adata_base.write(baseline_h5ad)

    # -----------------------------------------
    # E. Ours
    # -----------------------------------------
    adata_ours, ilisi_ours = run_one_model(
        adata_input=adata_raw,
        ws=OURS_WS,
        wsh=OURS_WSH,
        gamma=GAMMA,
        device=device,
        title="⏳ [2/2] 正在运行 Ours GraphST ..."
    )
    ours_h5ad = os.path.join(save_dir, f"{PAIR_NAME}_ours_result.h5ad")
    adata_ours.write(ours_h5ad)

    # -----------------------------------------
    # F. summary csv
    # -----------------------------------------
    summary_df = pd.DataFrame([{
        "dataset": PAIR_NAME,
        "slice1_path": slice1_path,
        "slice2_path": slice2_path,
        "merged_input_h5ad": merged_input_path,
        "graphst_datatype": GRAPHST_DATATYPE,
        "baseline_w_smooth": BASE_WS,
        "baseline_w_sharpen": BASE_WSH,
        "baseline_gamma": GAMMA,
        "baseline_iLISI": ilisi_base,
        "ours_w_smooth": OURS_WS,
        "ours_w_sharpen": OURS_WSH,
        "ours_gamma": GAMMA,
        "ours_iLISI": ilisi_ours,
        "delta_iLISI": ilisi_ours - ilisi_base,
        "baseline_result_h5ad": baseline_h5ad,
        "ours_result_h5ad": ours_h5ad,
    }])

    summary_csv = os.path.join(save_dir, f"{PAIR_NAME}_iLISI_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    # -----------------------------------------
    # G. UMAP 对比图
    # -----------------------------------------
    umap_png = os.path.join(save_dir, f"{PAIR_NAME}_umap_batch_compare.png")
    umap_pdf = os.path.join(save_dir, f"{PAIR_NAME}_umap_batch_compare.pdf")
    save_umap_compare_plot(adata_base, ilisi_base, adata_ours, ilisi_ours, umap_png, umap_pdf)

    # -----------------------------------------
    # H. 打印结果
    # -----------------------------------------
    print("\n" + "🌟" * 20)
    print("📊 iLISI 结果（你这里只有 2 个 batch，所以越接近 2 越好）")
    print(f"   Baseline iLISI : {ilisi_base:.4f}")
    print(f"   Ours iLISI     : {ilisi_ours:.4f}")
    print(f"   Delta          : {ilisi_ours - ilisi_base:+.4f}")
    print("🌟" * 20)

    print(f"✅ merged input h5ad : {merged_input_path}")
    print(f"✅ baseline h5ad     : {baseline_h5ad}")
    print(f"✅ ours h5ad         : {ours_h5ad}")
    print(f"✅ summary csv       : {summary_csv}")
    print(f"✅ merged spatial图   : {merged_spatial_png}")
    print(f"✅ umap 对比图        : {umap_png}")

    clear_memory()