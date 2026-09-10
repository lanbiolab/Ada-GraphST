import os
import sys
import warnings
warnings.filterwarnings("ignore")

project_root = "/data2/liangyefeng/My_GraphST_Innovation"
sys.path.insert(0, os.path.join(project_root, "src"))

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib

from SpaGCN.SpaGCN_package.SpaGCN.SpaGCN import SpaGCN
from SpaGCN.SpaGCN_package.SpaGCN.util import (
    prefilter_genes, prefilter_specialgenes, search_l, search_res
)
from SpaGCN.SpaGCN_package.SpaGCN.calculate_adj import calculate_adj_matrix

from hbca_section1_section2_integration_common import (
    reset_seed,
    load_integrated_dataset,
    to_counts_adata,
    infer_n_clusters,
    compute_ilisi_from_rep,
)

DATA_ROOT = "/data2/liangyefeng/My_GraphST_Innovation/data/Human-Breast-Cancer-Block-A"
ALIGNED_H5AD = os.path.join(
    DATA_ROOT,
    "PASTE_then_GraphST_section1_section2",
    "HBCA_section1_section2_PASTE_aligned_for_GraphST.h5ad"
)

SAVE_DIR = os.path.join(DATA_ROOT, "PASTE_then_SpaGCN_section1_section2")
os.makedirs(SAVE_DIR, exist_ok=True)

RESULT_H5AD = os.path.join(SAVE_DIR, "HBCA_section1_section2_SpaGCN_result.h5ad")
RESULT_CSV = os.path.join(SAVE_DIR, "HBCA_section1_section2_SpaGCN_iLISI_summary.csv")
UMAP_PNG = os.path.join(SAVE_DIR, "HBCA_section1_section2_SpaGCN_UMAP.png")
UMAP_PDF = os.path.join(SAVE_DIR, "HBCA_section1_section2_SpaGCN_UMAP.pdf")

SEED = 41
HISTOLOGY = False

# 字体设置，避免 PDF 字体问题
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


def extract_spagcn_embedding(clf, adata):
    candidate_attrs = ["embed", "embedding", "emb", "latent", "z"]
    emb = None

    for attr in candidate_attrs:
        if hasattr(clf, attr):
            value = getattr(clf, attr)
            if value is not None:
                emb = _to_numpy(value)
                print(f"✅ 使用 clf.{attr} 作为 embedding")
                break

    if emb is None:
        raise AttributeError("❌ 当前 SpaGCN 对象里没有找到可用 embedding。")

    if emb.ndim != 2 or emb.shape[0] != adata.n_obs:
        raise ValueError(f"❌ SpaGCN embedding 形状异常: {emb.shape}, n_obs={adata.n_obs}")
    if np.isnan(emb).any() or np.isinf(emb).any():
        raise ValueError("❌ SpaGCN embedding 中存在 NaN / Inf")

    adata.obsm["SpaGCN"] = emb.astype(np.float64)
    return adata


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
    rep_key="SpaGCN"
):
    """
    绘制 batch/domain 的 UMAP 图，但不显示任何图例。
    """
    if rep_key not in adata.obsm:
        raise KeyError(f"❌ adata.obsm['{rep_key}'] 不存在，无法绘制 UMAP。")

    if "batch" not in adata.obs.columns:
        raise KeyError("❌ adata.obs['batch'] 不存在，无法绘制 batch UMAP。")

    if "domain" not in adata.obs.columns:
        raise KeyError("❌ adata.obs['domain'] 不存在，无法绘制 domain UMAP。")

    # 重新基于 SpaGCN embedding 计算 UMAP，避免使用旧的 X_umap
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
    n_domain = adata.obs["domain"].nunique()
    if n_domain <= 20:
        adata.uns["domain_colors"] = sc.pl.palettes.default_20[:n_domain]
    elif n_domain <= 28:
        adata.uns["domain_colors"] = sc.pl.palettes.default_28[:n_domain]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))

    # batch UMAP
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

    if adata.obs["batch"].nunique() <= 2:
        batch_kwargs["palette"] = ["#5B9BD5", "#ED7D31"]

    sc.pl.umap(**batch_kwargs)

    # domain UMAP
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

    # 二次确保所有 legend 都被移除
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


if __name__ == "__main__":
    reset_seed(SEED)

    adata = load_integrated_dataset(ALIGNED_H5AD)
    adata = to_counts_adata(adata)
    n_clusters, gt_col = infer_n_clusters(adata, default_n=20)

    prefilter_genes(adata, min_cells=3)
    prefilter_specialgenes(adata)

    try:
        sc.pp.normalize_per_cell(adata)
    except Exception:
        sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    x = adata.obsm["spatial"][:, 0].tolist()
    y = adata.obsm["spatial"][:, 1].tolist()
    adj = calculate_adj_matrix(x=x, y=y, histology=HISTOLOGY)

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

    reset_seed(SEED)
    res = search_res(
        adata,
        adj,
        l,
        target_num=n_clusters,
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

    print(f"✅ Recommended l   = {l}")
    print(f"✅ Recommended res = {res}")

    clf = SpaGCN()
    clf.set_l(l)

    reset_seed(SEED)
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

    pred, prob = clf.predict()
    adata.obs["domain"] = pd.Categorical(pred.astype(int).astype(str))

    adata = extract_spagcn_embedding(clf, adata)
    ilisi = compute_ilisi_from_rep(adata, rep_key="SpaGCN", seed=SEED)

    plot_umap_batch_domain_no_legend(
        adata,
        method_name="SpaGCN",
        save_png=UMAP_PNG,
        save_pdf=UMAP_PDF,
        seed=SEED,
        rep_key="SpaGCN"
    )

    adata.write(RESULT_H5AD)

    pd.DataFrame([{
        "Dataset": "HBCA_section1_section2",
        "Method": "SpaGCN",
        "Seed": SEED,
        "Input_H5AD": ALIGNED_H5AD,
        "Result_H5AD": RESULT_H5AD,
        "GT_Col_For_NClusters": gt_col,
        "Histology": HISTOLOGY,
        "Recommended_l": l,
        "Recommended_res": res,
        "N_Clusters_For_Domain": n_clusters,
        "N_Obs_Total": adata.n_obs,
        "N_Vars": adata.n_vars,
        "iLISI": ilisi,
    }]).to_csv(RESULT_CSV, index=False)

    print(f"SpaGCN | iLISI={ilisi:.6f}")
    print("saved h5ad :", RESULT_H5AD)
    print("saved csv  :", RESULT_CSV)
    print("saved umap :", UMAP_PNG)
    print("saved umap :", UMAP_PDF)