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
import matplotlib.pyplot as plt
import matplotlib

PROJECT_ROOT = "/data2/liangyefeng/My_GraphST_Innovation"
CONST_ROOT = os.path.join(PROJECT_ROOT, "src", "conST")
PROJECT_SRC = os.path.join(PROJECT_ROOT, "src")

DATA_ROOT = "/data2/liangyefeng/My_GraphST_Innovation/data/Mouse-Brain-Section-2-Sagittal-Anterior-and-Posterior"
ALIGNED_H5AD = os.path.join(
    DATA_ROOT,
    "PASTE_then_GraphST_MA_MP",
    "MA_MP_PASTE_aligned_for_GraphST.h5ad"
)

SAVE_DIR = os.path.join(DATA_ROOT, "PASTE_then_conST_MA_MP")
os.makedirs(SAVE_DIR, exist_ok=True)

RESULT_H5AD = os.path.join(SAVE_DIR, "MouseBrain_MA_MP_conST_result.h5ad")
RESULT_CSV = os.path.join(SAVE_DIR, "MouseBrain_MA_MP_conST_iLISI_summary.csv")
UMAP_PNG = os.path.join(SAVE_DIR, "MouseBrain_MA_MP_conST_UMAP.png")
UMAP_PDF = os.path.join(SAVE_DIR, "MouseBrain_MA_MP_conST_UMAP.pdf")

SEED = 41
N_CLUSTERS = 20

if CONST_ROOT not in sys.path:
    sys.path.insert(0, CONST_ROOT)
if PROJECT_SRC not in sys.path:
    sys.path.append(PROJECT_SRC)

from src.graph_func import graph_construction
from src.utils_func import adata_preprocess
from src.training import conST_training

from mouse_brain_MA_MP_integration_common import (
    load_integrated_dataset,
    to_counts_adata,
    compute_ilisi_from_rep,
    mclust_R,
)

# 字体设置，避免 PDF 字体问题
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"


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


def remove_all_legends(fig):
    """
    移除 figure 中所有图例，避免显示右侧的 S1/S3 或 1~20。
    """
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    for legend in list(fig.legends):
        legend.remove()


def plot_umap_batch_domain_no_legend(
    adata,
    method_name,
    save_png,
    save_pdf,
    seed=41,
    rep_key="X_emb"
):
    """
    绘制 batch/domain 的 UMAP 图：
    1. 不显示 iLISI；
    2. 不显示 batch 图例；
    3. 不显示 domain 图例。
    """
    if rep_key not in adata.obsm:
        raise KeyError(f"❌ adata.obsm['{rep_key}'] 不存在，无法绘制 UMAP。")

    if "batch" not in adata.obs.columns:
        raise KeyError("❌ adata.obs['batch'] 不存在，无法绘制 batch UMAP。")

    if "domain" not in adata.obs.columns:
        raise KeyError("❌ adata.obs['domain'] 不存在，无法绘制 domain UMAP。")

    adata.obs["batch"] = adata.obs["batch"].astype(str).astype("category")
    adata.obs["domain"] = adata.obs["domain"].astype(str).astype("category")

    # 基于 conST embedding 计算 UMAP
    sc.pp.neighbors(
        adata,
        use_rep=rep_key,
        random_state=seed
    )

    sc.tl.umap(
        adata,
        random_state=seed
    )

    # 设置 domain 颜色
    n_domain = len(adata.obs["domain"].cat.categories)

    if n_domain <= 20:
        adata.uns["domain_colors"] = sc.pl.palettes.default_20[:n_domain]
    elif n_domain <= 28:
        adata.uns["domain_colors"] = sc.pl.palettes.default_28[:n_domain]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))

    # Batch UMAP
    batch_kwargs = {
        "adata": adata,
        "color": "batch",
        "title": f"{method_name} Batch",
        "show": False,
        "ax": axs[0],
        "frameon": False,
        "s": 30,
        "legend_loc": None
    }

    if len(adata.obs["batch"].cat.categories) <= 2:
        batch_kwargs["palette"] = ["#5B9BD5", "#ED7D31"]

    sc.pl.umap(**batch_kwargs)

    # Domain UMAP
    sc.pl.umap(
        adata,
        color="domain",
        title=f"{method_name} Domain",
        show=False,
        ax=axs[1],
        frameon=False,
        s=30,
        legend_loc=None
    )

    # 二次确保所有 legend 被移除
    remove_all_legends(fig)

    fig.tight_layout()

    fig.savefig(
        save_png,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        save_pdf,
        bbox_inches="tight"
    )

    plt.close(fig)


seed_torch(SEED)

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
    use_pretrained=False,
    using_mask=False,
    feat_w=10,
    gcn_w=0.1,
    dec_kl_w=10,
    gcn_lr=0.01,
    gcn_decay=0.01,
    dec_cluster_n=N_CLUSTERS,
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


if __name__ == "__main__":
    seed_torch(SEED)

    adata = load_integrated_dataset(ALIGNED_H5AD)
    adata = to_counts_adata(adata)

    params.cell_num = adata.n_obs

    print("Dataset: MouseBrain_MA_MP")
    print(f"n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    print(f"batches={adata.obs['batch'].cat.categories.tolist()}")

    adata_X = adata_preprocess(
        adata,
        min_cells=5,
        pca_n_comps=params.cell_feat_dim
    )

    graph_dict = graph_construction(
        adata.obsm["spatial"],
        adata.shape[0],
        params
    )

    conST_net = conST_training(
        adata_X,
        graph_dict,
        params,
        int(params.dec_cluster_n)
    )

    conST_net.pretraining()
    conST_net.major_training()

    conST_embedding = conST_net.get_embedding()

    if torch.is_tensor(conST_embedding):
        conST_embedding = conST_embedding.detach().cpu().numpy()
    else:
        conST_embedding = np.asarray(conST_embedding)

    adata.obsm["X_emb"] = conST_embedding.copy()

    ilisi = compute_ilisi_from_rep(
        adata,
        rep_key="X_emb",
        seed=SEED
    )

    adata = mclust_R(
        adata,
        num_cluster=N_CLUSTERS,
        used_obsm="X_emb",
        key_added="domain",
        random_seed=SEED
    )

    plot_umap_batch_domain_no_legend(
        adata,
        method_name="conST",
        save_png=UMAP_PNG,
        save_pdf=UMAP_PDF,
        seed=SEED,
        rep_key="X_emb"
    )

    adata.write(RESULT_H5AD)

    pd.DataFrame([{
        "Dataset": "MouseBrain_MA_MP",
        "Method": "conST",
        "Seed": SEED,
        "Input_H5AD": ALIGNED_H5AD,
        "Result_H5AD": RESULT_H5AD,
        "Device": device,
        "K": params.k,
        "Cell_Feat_Dim": params.cell_feat_dim,
        "Epochs": params.epochs,
        "N_Clusters_For_Domain": N_CLUSTERS,
        "N_Obs_Total": int(adata.n_obs),
        "N_Vars": int(adata.n_vars),
        "iLISI": float(ilisi),
    }]).to_csv(RESULT_CSV, index=False)

    print(f"conST | iLISI={ilisi:.6f}")
    print("saved h5ad :", RESULT_H5AD)
    print("saved csv  :", RESULT_CSV)
    print("saved umap :", UMAP_PNG)
    print("saved umap :", UMAP_PDF)