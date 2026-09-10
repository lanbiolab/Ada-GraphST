import os
import sys
import gc
import json
import random
import inspect
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

try:
    import torch
except ImportError:
    torch = None

try:
    import harmonypy as hm
except ImportError:
    raise ImportError("请先安装 harmonypy: pip install harmonypy")

# =========================================================
# 0. 环境配置
# =========================================================
PROJECT_ROOT = "/data2/liangyefeng/My_GraphST_Innovation"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

os.environ["R_HOME"] = "/data2/liangyefeng/miniconda3/envs/graphst/lib/R"

from GraphST.GraphST import GraphST

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"

# =========================================================
# 1. 输出目录
# =========================================================
OUT_ROOT = os.path.join(PROJECT_ROOT, "Result_Ablation_AdaGraphST_3Datasets")
os.makedirs(OUT_ROOT, exist_ok=True)

SUMMARY_CSV = os.path.join(OUT_ROOT, "ablation_summary.csv")
SUMMARY_RAW_PNG = os.path.join(OUT_ROOT, "ablation_raw_ilisi_barplot.png")
SUMMARY_RAW_PDF = os.path.join(OUT_ROOT, "ablation_raw_ilisi_barplot.pdf")

SUMMARY_DELTA_PNG = os.path.join(OUT_ROOT, "ablation_delta_ilisi_barplot.png")
SUMMARY_DELTA_PDF = os.path.join(OUT_ROOT, "ablation_delta_ilisi_barplot.pdf")

SUMMARY_LINE_PNG = os.path.join(OUT_ROOT, "ablation_trend_lineplot.png")
SUMMARY_LINE_PDF = os.path.join(OUT_ROOT, "ablation_trend_lineplot.pdf")

GAIN_CSV = os.path.join(OUT_ROOT, "ablation_gain_summary.csv")
GAIN_PNG = os.path.join(OUT_ROOT, "adagraphst_gain_summary.png")
GAIN_PDF = os.path.join(OUT_ROOT, "adagraphst_gain_summary.pdf")

# =========================================================
# 2. 数据集配置
# =========================================================
DATASETS = {
    "Mouse Brain": {
        "path": "/data2/liangyefeng/My_GraphST_Innovation/data/Mouse-Brain-Section-2-Sagittal-Anterior-and-Posterior/PASTE_then_GraphST_MA_MP/MA_MP_PASTE_aligned_for_GraphST.h5ad",
        "default_n_clusters": 20,

        # 4) Ada-GraphST / full 参数
        "full_params": {
            "w_smooth": 1.10,
            "w_sharpen": 0.9,
            "gamma": 2.5,
            "warmup": 200,
            "interval": 20,
            "ema": 0.3,
        },

        # 2) Smooth only
        "smooth_only_params": {
            "w_smooth": 1.10,
            "w_sharpen": 0.0,
            "gamma": 2.5,
            "warmup": 200,
            "interval": 20,
            "ema": 0.3,
        },

        # 3) Sharpen only
        "sharpen_only_params": {
            "w_smooth": 0.0,
            "w_sharpen": 0.3,
            "gamma": 2.5,
            "warmup": 200,
            "interval": 20,
            "ema": 0.3,
        },
    },

    "Human Breast Cancer": {
        "path": "/data2/liangyefeng/My_GraphST_Innovation/data/Human-Breast-Cancer-Block-A/PASTE_then_GraphST_section1_section2/HBCA_section1_section2_PASTE_aligned_for_GraphST.h5ad",
        "default_n_clusters": 20,

        # 4) Ada-GraphST / full 参数
        "full_params": {
            "w_smooth": 1.4,
            "w_sharpen": 0.9,
            "gamma": 2.5,
            "warmup": 200,
            "interval": 20,
            "ema": 0.3,
        },

        # 2) Smooth only
        "smooth_only_params": {
            "w_smooth": 1.4,
            "w_sharpen": 0.0,
            "gamma": 2.5,
            "warmup": 200,
            "interval": 20,
            "ema": 0.3,
        },

        # 3) Sharpen only
        "sharpen_only_params": {
            "w_smooth": 0.0,
            "w_sharpen": 0.3,
            "gamma": 2.5,
            "warmup": 200,
            "interval": 20,
            "ema": 0.3,
        },
    },

    "Mouse Breast Cancer": {
        "path": "/data2/liangyefeng/My_GraphST_Innovation/data/7.Mouse_Breast_Cancer_Sample_1/mouse_breast_cancer_sample1_section1&2.h5ad",
        "default_n_clusters": 20,

        # 4) Ada-GraphST / full 参数
        "full_params": {
            "w_smooth": 1.7,
            "w_sharpen": 0.7,
            "gamma": 2.5,
            "warmup": 200,
            "interval": 20,
            "ema": 0.3,
        },

        # 2) Smooth only
        "smooth_only_params": {
            "w_smooth": 1.7,
            "w_sharpen": 0.0,
            "gamma": 2.5,
            "warmup": 200,
            "interval": 20,
            "ema": 0.3,
        },

        # 3) Sharpen only
        "sharpen_only_params": {
            "w_smooth": 0.0,
            "w_sharpen": 0.05,
            "gamma": 2.5,
            "warmup": 200,
            "interval": 20,
            "ema": 0.3,
        },
    },
}

# =========================================================
# 3. 训练公共超参数
# =========================================================
SEED = 50
EPOCHS = 600
DIM_OUTPUT = 64
GRAPHST_DATATYPE = "10X"
USE_LEARNABLE_PROJ = False

BATCH_PALETTE = ["#5B9BD5", "#ED7D31"]

METHOD_COLORS = {
    "Baseline": "#B07AA1",
    "Smooth only": "#4E79A7",
    "Sharpen only": "#F28E2B",
    "Ada-GraphST": "#76B7B2",
}

DATASET_COLORS = {
    "Mouse Brain": "#4E79A7",
    "Human Breast Cancer": "#E15759",
    "Mouse Breast Cancer": "#59A14F",
}

VARIANT_ORDER = ["Baseline", "Smooth only", "Sharpen only", "Ada-GraphST"]
DATASET_ORDER = list(DATASETS.keys())

# =========================================================
# 4. 消融组
# =========================================================
ABLATION_VARIANTS = {
    "Baseline": {"use_smooth": False, "use_sharpen": False},
    "Smooth only": {"use_smooth": True, "use_sharpen": False},
    "Sharpen only": {"use_smooth": False, "use_sharpen": True},
    "Ada-GraphST": {"use_smooth": True, "use_sharpen": True},
}

# =========================================================
# 5. 基础工具函数
# =========================================================
def seed_everything(seed=50):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def clear_memory():
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _normalize_index_str(idx):
    return pd.Index(idx.astype(str).str.strip())


def _clean_str_series(s):
    s = s.astype(str).str.strip()
    bad = s.isin(["", "nan", "None", "NA", "NaN", "null"])
    s = s.copy()
    s[bad] = np.nan
    return s


def _normalize_variant_name(x):
    x = str(x).strip().lower().replace("_", " ")
    mapping = {
        "baseline": "Baseline",
        "smooth only": "Smooth only",
        "sharpen only": "Sharpen only",
        "ada graphst": "Ada-GraphST",
        "ada-graphst": "Ada-GraphST",
    }
    return mapping.get(x, str(x).strip())


def _infer_batch_from_obs_or_names(adata):
    candidate_cols = [
        "batch", "data", "section", "section_name", "section_id", "slice",
        "slices", "sample", "sample_id", "orig.ident", "orig_ident",
        "library_id", "library", "dataset", "source", "group",
    ]

    for col in candidate_cols:
        if col in adata.obs.columns:
            vals = _clean_str_series(adata.obs[col])
            nunique = vals.dropna().nunique()
            if nunique >= 2:
                adata.obs["batch"] = pd.Categorical(vals)
                print(f"✅ 使用 adata.obs['{col}'] 作为 batch")
                return adata, col

    idx_lower = pd.Index(adata.obs_names.astype(str)).str.lower()
    batch = pd.Series(index=adata.obs_names, dtype=object)

    mask1 = idx_lower.str.contains(r"section[_\-\s]?1|sec[_\-\s]?1|\bs1\b", regex=True)
    mask2 = idx_lower.str.contains(r"section[_\-\s]?2|sec[_\-\s]?2|\bs2\b", regex=True)

    if mask1.any() and mask2.any():
        batch.loc[mask1] = "section1"
        batch.loc[mask2] = "section2"
        adata.obs["batch"] = pd.Categorical(batch)
        print("✅ 从 obs_names 推断 batch 成功")
        return adata, "obs_names"

    raise KeyError(f"❌ 无法自动识别 batch 列。当前 obs columns: {list(adata.obs.columns)}")


def load_integrated_dataset(h5ad_path):
    if not os.path.exists(h5ad_path):
        raise FileNotFoundError(f"❌ 找不到文件: {h5ad_path}")

    print(f"📥 正在读取整合输入: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)

    adata.var_names_make_unique()
    adata.obs_names = _normalize_index_str(adata.obs_names)
    adata.obs_names_make_unique()

    if "spatial" not in adata.obsm:
        raise KeyError(f"❌ adata.obsm['spatial'] 不存在。当前 obsm keys: {list(adata.obsm.keys())}")

    adata, batch_source = _infer_batch_from_obs_or_names(adata)
    adata.obs["batch"] = _clean_str_series(adata.obs["batch"]).astype("category")

    print(f"✅ n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    print(f"✅ batch source={batch_source}")
    print(f"✅ batches={adata.obs['batch'].cat.categories.tolist()}")

    return adata


def infer_n_clusters(adata, default_n=20):
    candidate_cols = [
        "ground_truth", "original_domain", "label", "annotation",
        "celltype", "cell_type", "manual_annotation",
    ]
    for col in candidate_cols:
        if col in adata.obs.columns:
            s = _clean_str_series(adata.obs[col]).dropna()
            n_clusters = s.nunique()
            if n_clusters >= 2:
                print(f"✅ 使用 {col} 推断聚类数: {n_clusters}")
                return int(n_clusters), col

    print(f"⚠️ 未找到可用 GT 列，聚类数回退为默认值: {default_n}")
    return int(default_n), None


def check_embedding_valid(adata, emb_key="emb"):
    if emb_key not in adata.obsm:
        raise KeyError(f"❌ GraphST 输出中不存在 adata.obsm['{emb_key}']")

    emb = np.asarray(adata.obsm[emb_key], dtype=np.float64)
    if np.isnan(emb).any() or np.isinf(emb).any():
        nan_count = int(np.isnan(emb).sum())
        inf_count = int(np.isinf(emb).sum())
        raise ValueError(f"❌ {emb_key} 中存在非法值: NaN={nan_count}, Inf={inf_count}")
    return emb


def compute_ilisi_from_emb(adata, emb_key="emb", seed=50):
    emb = check_embedding_valid(adata, emb_key=emb_key)

    n_pca = min(20, emb.shape[1], max(2, emb.shape[0] - 1))
    n_pca = max(2, n_pca)

    adata.obsm["emb_pca"] = PCA(
        n_components=n_pca,
        random_state=seed
    ).fit_transform(emb)

    lisi = hm.compute_lisi(
        adata.obsm["emb_pca"],
        adata.obs[["batch"]],
        ["batch"]
    )
    return float(np.mean(lisi))


def _fallback_cluster_sklearn(x, num_cluster, random_seed=50):
    x = np.asarray(x, dtype=np.float64)

    try:
        gmm = GaussianMixture(
            n_components=int(num_cluster),
            covariance_type="full",
            reg_covar=1e-6,
            max_iter=500,
            n_init=5,
            random_state=random_seed
        )
        labels = gmm.fit_predict(x).astype(int) + 1
        print("⚠️ mclust 失败，已回退到 sklearn GaussianMixture")
        return labels
    except Exception as e:
        print(f"⚠️ GaussianMixture 也失败: {e}")

    km = KMeans(
        n_clusters=int(num_cluster),
        random_state=random_seed,
        n_init=20
    )
    labels = km.fit_predict(x).astype(int) + 1
    print("⚠️ mclust / GaussianMixture 均失败，已回退到 sklearn KMeans")
    return labels


def cluster_for_domain(adata, num_cluster, used_obsm="emb_pca", key_added="domain", random_seed=50):
    x = np.asarray(adata.obsm[used_obsm], dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"{used_obsm} must be 2D, but got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{used_obsm} contains NaN/Inf")

    labels = None
    try:
        import rpy2.robjects as robjects
        from rpy2.robjects import r, numpy2ri

        numpy2ri.activate()
        robjects.globalenv["x_mat"] = x
        robjects.globalenv["n_cluster"] = int(num_cluster)
        robjects.globalenv["seed"] = int(random_seed)

        r(
            """
            suppressMessages(library(mclust))
            set.seed(seed)
            x_mat <- as.matrix(x_mat)
            dimnames(x_mat) <- NULL

            res <- tryCatch(
              Mclust(x_mat, G=n_cluster, modelNames="EEE"),
              error = function(e) NULL
            )

            if (is.null(res)) {
              cls <- NULL
            } else {
              cls <- res$classification
            }
            """
        )

        cls_obj = r["cls"]
        if cls_obj is None or str(type(cls_obj)).lower().find("null") >= 0:
            raise ValueError("mclust returned NULL classification")

        cls_arr = np.array(cls_obj, dtype=object)
        cls_series = pd.Series(cls_arr, index=adata.obs_names)
        cls_series = pd.to_numeric(cls_series, errors="coerce")

        if cls_series.isna().all():
            raise ValueError("mclust returned all-NA classification")
        if cls_series.isna().sum() > 0:
            raise ValueError(f"mclust returned {cls_series.isna().sum()} invalid labels")

        labels = cls_series.astype(int).values
        print("✅ mclust 聚类成功")

    except Exception as e:
        print(f"⚠️ mclust 失败: {e}")

    if labels is None:
        labels = _fallback_cluster_sklearn(x, num_cluster=num_cluster, random_seed=random_seed)

    adata.obs[key_added] = pd.Categorical(labels.astype(str))
    return adata


def plot_umap_batch_domain(adata, method_name, ilisi, save_png, save_pdf, seed=50):
    if "emb_pca" not in adata.obsm:
        raise KeyError("❌ adata.obsm['emb_pca'] 不存在，无法画 UMAP")
    if "domain" not in adata.obs.columns:
        raise KeyError("❌ adata.obs['domain'] 不存在，无法画 Spatial Domains")

    sc.pp.neighbors(adata, use_rep="emb_pca", random_state=seed)
    sc.tl.umap(adata, random_state=seed)

    n_domain = adata.obs["domain"].astype(str).nunique()
    if n_domain <= 20:
        palette_domain = sc.pl.palettes.default_20[:n_domain]
    elif n_domain <= 28:
        palette_domain = sc.pl.palettes.default_28[:n_domain]
    else:
        palette_domain = sc.pl.palettes.default_102[:n_domain]
    adata.uns["domain_colors"] = palette_domain

    n_batch = adata.obs["batch"].astype(str).nunique()
    if n_batch <= len(BATCH_PALETTE):
        batch_palette = BATCH_PALETTE[:n_batch]
    elif n_batch <= 20:
        batch_palette = sc.pl.palettes.default_20[:n_batch]
    elif n_batch <= 28:
        batch_palette = sc.pl.palettes.default_28[:n_batch]
    else:
        batch_palette = sc.pl.palettes.default_102[:n_batch]

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    sc.pl.umap(
        adata, color="batch",
        title=f"{method_name} (Batch Mixing)\niLISI: {ilisi:.3f}",
        show=False, ax=axs[0], frameon=False, s=28,
        palette=batch_palette, legend_loc="right margin"
    )

    sc.pl.umap(
        adata, color="domain",
        title=f"{method_name} (Spatial Domains)",
        show=False, ax=axs[1], frameon=False, s=28,
        legend_loc="right margin"
    )

    for ax in axs:
        ax.set_facecolor("#EAEAF2")

    plt.tight_layout()
    plt.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.savefig(save_pdf, bbox_inches="tight")
    plt.close()

# =========================================================
# 6. GraphST 参数适配
# =========================================================
def build_variant_params(dataset_cfg, variant_name, variant_cfg):
    full_params = dataset_cfg["full_params"]

    if variant_name == "Baseline":
        return {
            "w_smooth": 0.0,
            "w_sharpen": 0.0,
            "gamma": 3.0,
            "warmup": full_params["warmup"],
            "interval": full_params["interval"],
            "ema": full_params["ema"],
        }

    if variant_name == "Smooth only":
        return dataset_cfg["smooth_only_params"].copy()

    if variant_name == "Sharpen only":
        return dataset_cfg["sharpen_only_params"].copy()

    return {
        "w_smooth": full_params["w_smooth"] if variant_cfg["use_smooth"] else 0.0,
        "w_sharpen": full_params["w_sharpen"] if variant_cfg["use_sharpen"] else 0.0,
        "gamma": full_params["gamma"],
        "warmup": full_params["warmup"],
        "interval": full_params["interval"],
        "ema": full_params["ema"],
    }


def build_graphst_kwargs(adata, device, variant_params):
    sig = inspect.signature(GraphST.__init__)
    supported = set(sig.parameters.keys())

    kwargs = {
        "adata": adata,
        "device": device,
        "random_seed": SEED,
        "epochs": EPOCHS,
        "dim_output": DIM_OUTPUT,
        "datatype": GRAPHST_DATATYPE,
        "use_learnable_proj": USE_LEARNABLE_PROJ,
        "w_smooth": variant_params["w_smooth"],
        "w_sharpen": variant_params["w_sharpen"],
        "gamma": variant_params["gamma"],
        "warmup": variant_params["warmup"],
        "warmup_epochs": variant_params["warmup"],
        "interval": variant_params["interval"],
        "update_interval": variant_params["interval"],
        "ema": variant_params["ema"],
        "ema_decay": variant_params["ema"],
        "boundary_ema": variant_params["ema"],
    }

    final_kwargs = {}
    for k, v in kwargs.items():
        if k in supported:
            final_kwargs[k] = v
    return final_kwargs

# =========================================================
# 7. 跑单个消融组
# =========================================================
def run_one_ablation(dataset_name, dataset_cfg, variant_name, variant_cfg):
    print("\n" + "=" * 100)
    print(f"🚀 Dataset: {dataset_name} | Variant: {variant_name}")
    print("=" * 100)

    dataset_dir = os.path.join(OUT_ROOT, dataset_name.replace(" ", "_"))
    variant_dir = os.path.join(dataset_dir, variant_name.replace(" ", "_"))
    os.makedirs(variant_dir, exist_ok=True)

    adata_input = load_integrated_dataset(dataset_cfg["path"])
    n_clusters, gt_col = infer_n_clusters(
        adata_input,
        default_n=dataset_cfg["default_n_clusters"]
    )

    device = "cuda:0" if (torch is not None and torch.cuda.is_available()) else "cpu"
    variant_params = build_variant_params(dataset_cfg, variant_name, variant_cfg)

    adata_run = adata_input.copy()
    graphst_kwargs = build_graphst_kwargs(adata_run, device, variant_params)

    print("✅ 实际传入 GraphST 的参数：")
    print(json.dumps(
        {k: (str(v) if k == "adata" else v) for k, v in graphst_kwargs.items()},
        ensure_ascii=False, indent=2
    ))

    seed_everything(SEED)
    model = GraphST(**graphst_kwargs)

    out = model.train()
    if out is not None:
        adata_run = out

    ilisi = compute_ilisi_from_emb(adata_run, emb_key="emb", seed=SEED)
    adata_run = cluster_for_domain(
        adata_run,
        num_cluster=n_clusters,
        used_obsm="emb_pca",
        key_added="domain",
        random_seed=SEED
    )

    result_h5ad = os.path.join(variant_dir, f"{variant_name.replace(' ', '_')}.h5ad")
    result_png = os.path.join(variant_dir, f"{variant_name.replace(' ', '_')}_UMAP.png")
    result_pdf = os.path.join(variant_dir, f"{variant_name.replace(' ', '_')}_UMAP.pdf")

    plot_umap_batch_domain(
        adata_run,
        method_name=variant_name,
        ilisi=ilisi,
        save_png=result_png,
        save_pdf=result_pdf,
        seed=SEED
    )

    adata_run.write(result_h5ad)

    row = {
        "Dataset": dataset_name,
        "Variant": variant_name,
        "iLISI": float(ilisi),
        "Input_H5AD": dataset_cfg["path"],
        "Result_H5AD": result_h5ad,
        "UMAP_PNG": result_png,
        "UMAP_PDF": result_pdf,
        "Seed": SEED,
        "Device": device,
        "N_Obs": int(adata_run.n_obs),
        "N_Vars": int(adata_run.n_vars),
        "N_Clusters_For_Domain": int(n_clusters),
        "GT_Col_For_NClusters": gt_col,
        "Passed_GraphST_Kwargs": json.dumps(
            {k: (str(v) if k == "adata" else v) for k, v in graphst_kwargs.items()},
            ensure_ascii=False
        ),
        "Ablation_Params": json.dumps(variant_params, ensure_ascii=False),
    }

    print(f"✅ {dataset_name} | {variant_name} | iLISI={ilisi:.6f}")
    print(f"✅ 保存: {result_h5ad}")
    print(f"✅ 保存: {result_png}")
    print(f"✅ 保存: {result_pdf}")

    del model
    clear_memory()

    return row

#字体
def style_axis_ticks(
    ax,
    x_tick_fontsize=12,
    y_tick_fontsize=12,
    label_fontsize=13,
    title_fontsize=15,
    spine_width=1.2
):
    """
    统一加粗和放大 X/Y 轴刻度文字。
    tick label 指坐标轴上的数值或类别名称。
    """
    ax.tick_params(
        axis="x",
        which="major",
        labelsize=x_tick_fontsize,
        width=spine_width,
        length=5
    )
    ax.tick_params(
        axis="y",
        which="major",
        labelsize=y_tick_fontsize,
        width=spine_width,
        length=5
    )

    # X 轴刻度文字加粗
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")

    # Y 轴刻度数值加粗
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    # X/Y 轴标题加粗
    ax.xaxis.label.set_size(label_fontsize)
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_size(label_fontsize)
    ax.yaxis.label.set_weight("bold")

    # 子图标题加粗
    ax.title.set_size(title_fontsize)
    ax.title.set_weight("bold")

    # 坐标轴边框加粗
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)


# =========================================================
# 8. 绘图辅助
# =========================================================
def prepare_summary_df(summary_df):
    df = summary_df.copy()
    df = df.dropna(subset=["iLISI"]).copy()

    df["Dataset"] = df["Dataset"].astype(str).str.strip()
    df["Variant"] = df["Variant"].apply(_normalize_variant_name)

    df = df[df["Variant"].isin(VARIANT_ORDER)].copy()
    df = df[df["Dataset"].isin(DATASET_ORDER)].copy()

    df["Dataset"] = pd.Categorical(df["Dataset"], categories=DATASET_ORDER, ordered=True)
    df["Variant"] = pd.Categorical(df["Variant"], categories=VARIANT_ORDER, ordered=True)

    df = df.sort_values(["Dataset", "Variant"]).reset_index(drop=True)

    print("\n========== 规范化后的消融结果 ==========")
    print(df[["Dataset", "Variant", "iLISI"]].to_string(index=False))

    return df


def get_complete_dataset_names(df):
    complete = []
    for dataset_name in DATASET_ORDER:
        sub = df[df["Dataset"] == dataset_name].copy()
        variants = sub["Variant"].astype(str).tolist()
        if all(v in variants for v in VARIANT_ORDER):
            complete.append(dataset_name)
        else:
            missing = [v for v in VARIANT_ORDER if v not in variants]
            print(f"⚠️ 数据集 {dataset_name} 缺少以下分组，绘图时将跳过: {missing}")
    return complete


def compute_gain_summary(df):
    rows = []
    complete_datasets = get_complete_dataset_names(df)

    for dataset_name in complete_datasets:
        sub = df[df["Dataset"] == dataset_name].copy()

        baseline_row = sub[sub["Variant"] == "Baseline"]
        ada_row = sub[sub["Variant"] == "Ada-GraphST"]

        if baseline_row.empty or ada_row.empty:
            continue

        baseline = float(baseline_row["iLISI"].iloc[0])
        ada = float(ada_row["iLISI"].iloc[0])

        delta = ada - baseline
        rel_improve_pct = (delta / baseline) * 100 if baseline != 0 else np.nan

        ideal = 2.0
        gap_baseline = ideal - baseline
        gap_ada = ideal - ada
        gap_reduction_pct = ((gap_baseline - gap_ada) / gap_baseline * 100) if gap_baseline != 0 else np.nan

        rows.append({
            "Dataset": dataset_name,
            "Baseline_iLISI": baseline,
            "Ada_iLISI": ada,
            "Delta_iLISI": delta,
            "Relative_Improvement_pct": rel_improve_pct,
            "Gap_Reduction_pct": gap_reduction_pct,
        })

    return pd.DataFrame(rows)


def plot_ablation_summary(summary_df, save_png, save_pdf):
    complete_datasets = get_complete_dataset_names(summary_df)
    if len(complete_datasets) == 0:
        print("⚠️ 没有完整数据集，跳过原始 iLISI 柱状图。")
        return

    fig, axes = plt.subplots(1, len(complete_datasets), figsize=(7 * len(complete_datasets), 6.5))
    fig.patch.set_facecolor("white")

    if len(complete_datasets) == 1:
        axes = [axes]

    for idx, (ax, dataset_name) in enumerate(zip(axes, complete_datasets), start=1):
        df_sub = summary_df[summary_df["Dataset"] == dataset_name].copy()
        df_sub["Variant"] = pd.Categorical(df_sub["Variant"], categories=VARIANT_ORDER, ordered=True)
        df_sub = df_sub.sort_values("Variant").reset_index(drop=True)

        methods = df_sub["Variant"].astype(str).tolist()
        scores = df_sub["iLISI"].tolist()
        colors = [METHOD_COLORS.get(m, "#999999") for m in methods]

        bars = ax.bar(
            range(len(methods)),
            scores,
            color=colors,
            edgecolor="black",
            linewidth=0.8
        )

        ax.set_title(f"({idx}) {dataset_name}", fontsize=15, fontweight="bold")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(
            methods,
            rotation=25,
            ha="right",
            fontsize=12,
            fontweight="bold"
        )
        ax.set_ylabel("iLISI", fontsize=13, fontweight="bold")
        ax.set_facecolor("#F5F5F5")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        style_axis_ticks(
            ax,
            x_tick_fontsize=12,
            y_tick_fontsize=12,
            label_fontsize=13,
            title_fontsize=15,
            spine_width=1.2
        )

        ymax = max(scores) if len(scores) > 0 else 1.0
        ax.set_ylim(0, max(2.1, ymax * 1.18))

        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold"
            )

    fig.suptitle("Raw iLISI Across Ablation Variants", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.savefig(save_pdf, bbox_inches="tight")
    plt.close()


def plot_delta_summary(summary_df, save_png, save_pdf):
    complete_datasets = get_complete_dataset_names(summary_df)
    if len(complete_datasets) == 0:
        print("⚠️ 没有完整数据集，跳过 ΔiLISI 柱状图。")
        return

    delta_records = []
    for dataset_name in complete_datasets:
        sub = summary_df[summary_df["Dataset"] == dataset_name].copy()
        sub["Variant"] = pd.Categorical(sub["Variant"], categories=VARIANT_ORDER, ordered=True)
        sub = sub.sort_values("Variant").reset_index(drop=True)

        baseline_row = sub[sub["Variant"] == "Baseline"]
        if baseline_row.empty:
            continue

        baseline = float(baseline_row["iLISI"].iloc[0])

        for _, row in sub.iterrows():
            delta_records.append({
                "Dataset": dataset_name,
                "Variant": str(row["Variant"]),
                "Delta_iLISI": float(row["iLISI"]) - baseline
            })

    delta_df = pd.DataFrame(delta_records)
    if delta_df.empty:
        print("⚠️ 没有可用的 ΔiLISI 数据，跳过绘图。")
        return

    delta_df["Dataset"] = pd.Categorical(delta_df["Dataset"], categories=complete_datasets, ordered=True)
    delta_df["Variant"] = pd.Categorical(delta_df["Variant"], categories=VARIANT_ORDER, ordered=True)
    delta_df = delta_df.sort_values(["Dataset", "Variant"]).reset_index(drop=True)

    fig, axes = plt.subplots(1, len(complete_datasets), figsize=(7 * len(complete_datasets), 6.5))
    fig.patch.set_facecolor("white")

    if len(complete_datasets) == 1:
        axes = [axes]

    for idx, (ax, dataset_name) in enumerate(zip(axes, complete_datasets), start=1):
        sub = delta_df[delta_df["Dataset"] == dataset_name].copy()

        methods = sub["Variant"].astype(str).tolist()
        scores = sub["Delta_iLISI"].tolist()
        colors = [METHOD_COLORS.get(m, "#999999") for m in methods]

        bars = ax.bar(
            range(len(methods)),
            scores,
            color=colors,
            edgecolor="black",
            linewidth=0.8
        )

        ax.axhline(0, color="black", linewidth=1.2)
        ax.set_title(f"({idx}) {dataset_name}", fontsize=15, fontweight="bold")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(
            methods,
            rotation=25,
            ha="right",
            fontsize=12,
            fontweight="bold"
        )
        ax.set_ylabel("ΔiLISI vs Baseline", fontsize=13, fontweight="bold")
        ax.set_facecolor("#F5F5F5")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        style_axis_ticks(
            ax,
            x_tick_fontsize=12,
            y_tick_fontsize=12,
            label_fontsize=13,
            title_fontsize=15,
            spine_width=1.2
        )

        ymin = min(scores) if len(scores) > 0 else -0.1
        ymax = max(scores) if len(scores) > 0 else 0.1
        ax.set_ylim(min(-0.25, ymin * 1.25), max(0.10, ymax * 1.35))

        for bar, score in zip(bars, scores):
            offset = 0.01 if score >= 0 else -0.015
            va = "bottom" if score >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{score:+.3f}",
                ha="center",
                va=va,
                fontsize=11,
                fontweight="bold"
            )

    fig.suptitle("Improvement over Baseline (ΔiLISI)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.savefig(save_pdf, bbox_inches="tight")
    plt.close()


def plot_ablation_line(summary_df, save_png, save_pdf):
    complete_datasets = get_complete_dataset_names(summary_df)
    if len(complete_datasets) == 0:
        print("⚠️ 没有完整数据集，跳过趋势折线图。")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F5F5F5")

    x = np.arange(len(VARIANT_ORDER))

    for dataset_name in complete_datasets:
        sub = summary_df[summary_df["Dataset"] == dataset_name].copy()
        sub["Variant"] = pd.Categorical(sub["Variant"], categories=VARIANT_ORDER, ordered=True)
        sub = sub.sort_values("Variant")

        y = sub["iLISI"].values
        if len(y) != len(VARIANT_ORDER):
            continue

        ax.plot(
            x, y,
            marker="o",
            linewidth=2.3,
            markersize=7,
            label=dataset_name,
            color=DATASET_COLORS[dataset_name]
        )

        for xi, yi in zip(x, y):
            ax.text(
                xi,
                yi + 0.01,
                f"{yi:.3f}",
                ha="center",
                va="bottom",
                fontsize=10.5,
                fontweight="bold"
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            VARIANT_ORDER,
            rotation=20,
            ha="right",
            fontsize=12,
            fontweight="bold"
        )
        ax.set_ylabel("iLISI", fontsize=13, fontweight="bold")
        ax.set_title(
            "Ablation Trend Across Three Datasets",
            fontsize=16,
            fontweight="bold"
        )
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        style_axis_ticks(
            ax,
            x_tick_fontsize=12,
            y_tick_fontsize=12,
            label_fontsize=13,
            title_fontsize=16,
            spine_width=1.2
        )

        ax.legend(
            frameon=True,
            prop={
                "size": 10.5,
                "weight": "bold"
            }
        )

    plt.tight_layout()
    plt.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.savefig(save_pdf, bbox_inches="tight")
    plt.close()


def plot_gain_summary(gain_df, save_png, save_pdf):
    if gain_df.empty:
        print("⚠️ 没有可用的增益数据，跳过增益总结图。")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    fig.patch.set_facecolor("white")

    metric_info = [
        ("Delta_iLISI", "Absolute Gain (ΔiLISI)"),
        ("Relative_Improvement_pct", "Relative Gain (%)"),
        ("Gap_Reduction_pct", "Gap Reduction to 2.0 (%)"),
    ]

    for ax, (col, title) in zip(axes, metric_info):
        sub = gain_df.copy()

        datasets = sub["Dataset"].tolist()
        values = sub[col].tolist()
        colors = [DATASET_COLORS[d] for d in datasets]

        bars = ax.bar(
            range(len(datasets)),
            values,
            color=colors,
            edgecolor="black",
            linewidth=0.8
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(
            datasets,
            rotation=20,
            ha="right",
            fontsize=11.5,
            fontweight="bold"
        )
        ax.set_facecolor("#F5F5F5")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        style_axis_ticks(
            ax,
            x_tick_fontsize=11.5,
            y_tick_fontsize=12,
            label_fontsize=13,
            title_fontsize=14,
            spine_width=1.2
        )

        ymin = min(values) if len(values) > 0 else 0.0
        ymax = max(values) if len(values) > 0 else 1.0
        if ymin >= 0:
            ax.set_ylim(0, ymax * 1.25 if ymax > 0 else 1.0)
        else:
            ax.set_ylim(ymin * 1.25, ymax * 1.25)

        for bar, val in zip(bars, values):
            suffix = "%" if "pct" in col.lower() else ""
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.01 * max(1, ymax)),
                f"{val:.2f}{suffix}",
                ha="center",
                va="bottom",
                fontsize=10.5,
                fontweight="bold"
            )

    fig.suptitle("Ada-GraphST Gain Summary Across Three Datasets", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.savefig(save_pdf, bbox_inches="tight")
    plt.close()

# =========================================================
# 9. 主程序
# =========================================================
if __name__ == "__main__":
    seed_everything(SEED)

    all_rows = []

    for dataset_name, dataset_cfg in DATASETS.items():
        for variant_name, variant_cfg in ABLATION_VARIANTS.items():
            try:
                row = run_one_ablation(dataset_name, dataset_cfg, variant_name, variant_cfg)
                all_rows.append(row)
            except Exception as e:
                print(f"❌ FAILED | Dataset={dataset_name} | Variant={variant_name}")
                print(e)
                all_rows.append({
                    "Dataset": dataset_name,
                    "Variant": variant_name,
                    "iLISI": np.nan,
                    "Input_H5AD": dataset_cfg["path"],
                    "Result_H5AD": "",
                    "UMAP_PNG": "",
                    "UMAP_PDF": "",
                    "Seed": SEED,
                    "Device": "",
                    "N_Obs": np.nan,
                    "N_Vars": np.nan,
                    "N_Clusters_For_Domain": np.nan,
                    "GT_Col_For_NClusters": "",
                    "Passed_GraphST_Kwargs": "",
                    "Ablation_Params": "",
                    "Error": str(e),
                })
                clear_memory()

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\n✅ 消融汇总表已保存: {SUMMARY_CSV}")

    ok_df = prepare_summary_df(summary_df)
    if not ok_df.empty:
        plot_ablation_summary(ok_df, SUMMARY_RAW_PNG, SUMMARY_RAW_PDF)
        print(f"✅ 原始 iLISI 柱状图已保存: {SUMMARY_RAW_PNG}")
        print(f"✅ 原始 iLISI 柱状图已保存: {SUMMARY_RAW_PDF}")

        plot_delta_summary(ok_df, SUMMARY_DELTA_PNG, SUMMARY_DELTA_PDF)
        print(f"✅ ΔiLISI 柱状图已保存: {SUMMARY_DELTA_PNG}")
        print(f"✅ ΔiLISI 柱状图已保存: {SUMMARY_DELTA_PDF}")

        plot_ablation_line(ok_df, SUMMARY_LINE_PNG, SUMMARY_LINE_PDF)
        print(f"✅ 趋势折线图已保存: {SUMMARY_LINE_PNG}")
        print(f"✅ 趋势折线图已保存: {SUMMARY_LINE_PDF}")

        gain_df = compute_gain_summary(ok_df)
        gain_df.to_csv(GAIN_CSV, index=False)
        print(f"✅ 增益汇总表已保存: {GAIN_CSV}")

        plot_gain_summary(gain_df, GAIN_PNG, GAIN_PDF)
        print(f"✅ Ada-GraphST 增益总结图已保存: {GAIN_PNG}")
        print(f"✅ Ada-GraphST 增益总结图已保存: {GAIN_PDF}")

    print("\n最终结果：")
    print(summary_df[["Dataset", "Variant", "iLISI"]].to_string(index=False))