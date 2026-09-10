import os
import sys
import gc
import random
import traceback

# =========================================================
# 0. 环境配置
# =========================================================
os.environ["R_HOME"] = "/data2/liangyefeng/miniconda3/envs/graphst/lib/R"

project_root = "/data2/liangyefeng/My_GraphST_Innovation"
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import numpy as np
import pandas as pd
import torch
import scanpy as sc
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from GraphST.GraphST import GraphST
import GraphST.utils as graphst_utils
from GraphST.utils import clustering


# =========================================================
# 0.1 全局画图格式配置
# =========================================================
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"


# =========================================================
# 1. mclust patch
# =========================================================
def patched_mclust_R(
    adata,
    num_cluster,
    modelNames="EEE",
    used_obsm="emb_pca",
    random_seed=2020
):
    x = np.asarray(adata.obsm[used_obsm], dtype=np.float64)

    if x.ndim != 2:
        raise ValueError(f"{used_obsm} must be 2D, but got shape {x.shape}")

    if not np.isfinite(x).all():
        raise ValueError(f"{used_obsm} contains NaN or Inf.")

    print(f"[patched_mclust_R] used_obsm={used_obsm}, shape={x.shape}, dtype={x.dtype}")

    labels = None

    try:
        import rpy2.robjects as robjects
        from rpy2.robjects import r
        from rpy2.robjects import numpy2ri

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
                cls <- NULL
            } else {
                cls <- res$classification
            }
            """
        )

        cls_obj = r["cls"]

        if cls_obj is None or "NULL" in str(type(cls_obj)):
            raise ValueError("mclust returned NULL classification.")

        labels = np.array(cls_obj).astype(int)
        print("✅ mclust 聚类成功")

    except Exception as e:
        print(f"⚠️ mclust 失败，准备回退到 sklearn。原因: {e}")

    if labels is None:
        try:
            gmm = GaussianMixture(
                n_components=int(num_cluster),
                covariance_type="full",
                reg_covar=1e-6,
                max_iter=500,
                n_init=5,
                random_state=int(random_seed),
            )
            labels = gmm.fit_predict(x).astype(int) + 1
            print("✅ 回退 GaussianMixture 聚类成功")
        except Exception as e:
            print(f"⚠️ GaussianMixture 失败，继续回退 KMeans。原因: {e}")

            km = KMeans(
                n_clusters=int(num_cluster),
                random_state=int(random_seed),
                n_init=20,
            )
            labels = km.fit_predict(x).astype(int) + 1
            print("✅ 回退 KMeans 聚类成功")

    adata.obs["mclust"] = labels
    adata.obs["mclust"] = adata.obs["mclust"].astype(int).astype("category")

    return adata


graphst_utils.mclust_R = patched_mclust_R


# =========================================================
# 2. 固定随机种子
# =========================================================
seed = 41


def seed_everything(seed=41):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)


# =========================================================
# 3. 路径配置
# =========================================================
data_root = "/data2/liangyefeng/My_GraphST_Innovation/data/1.DLPFC"

# 这里是你之前生成 baseline_xxx.h5ad 和 ours_best_xxx.h5ad 的目录
previous_h5ad_root = "/data2/liangyefeng/My_GraphST_Innovation/Result_H5AD"

# 新的消融实验输出目录
output_root = "/data2/liangyefeng/My_GraphST_Innovation/Result_Ablation_DLPFC_151508_151670_151675"
h5ad_out_root = os.path.join(output_root, "h5ad")
fig_out_root = os.path.join(output_root, "figures")

os.makedirs(output_root, exist_ok=True)
os.makedirs(h5ad_out_root, exist_ok=True)
os.makedirs(fig_out_root, exist_ok=True)


# =========================================================
# 4. 只跑这三个切片
# =========================================================
dataset_list = ["151508", "151670", "151675"]

cluster_map = {
    "151507": 7, "151508": 7, "151509": 7, "151510": 7,
    "151669": 5, "151670": 5, "151671": 5, "151672": 5,
    "151673": 7, "151674": 7, "151675": 7, "151676": 7,
}

radius = 50


# =========================================================
# 5. 公共参数：保持你原来的代码逻辑
# =========================================================
common_params = {
    "device": device,
    "random_seed": seed,
    "epochs": 600,
    "dim_output": 64,
    "datatype": "10X",
    "warmup_epochs": 200,
    "update_interval": 20,
    "graph_update_rate": 0.3,
    "gamma": 3.0,
    "graph_reg_weight": 0.0,
    "use_learnable_proj": False,
}


# =========================================================
# 6. 消融组
# =========================================================
ABLATION_VARIANTS = {
    "Baseline": {
        "w_smooth": 0.0,
        "w_sharpen": 0.0,
        "reuse_existing": True,
        "existing_type": "baseline",
    },
    "Smooth only": {
        "w_smooth": 0.01,
        "w_sharpen": 0.0,
        "reuse_existing": False,
        "existing_type": None,
    },
    "Sharpen only": {
        "w_smooth": 0.0,
        "w_sharpen": 0.05,
        "reuse_existing": False,
        "existing_type": None,
    },
    "Ada-GraphST": {
        "w_smooth": 0.01,
        "w_sharpen": 0.05,
        "reuse_existing": True,
        "existing_type": "ours",
    },
}

VARIANT_ORDER = ["Baseline", "Smooth only", "Sharpen only", "Ada-GraphST"]

# 经典红蓝绿橙风格
METHOD_COLORS = {
    "Baseline": "#d62728",      # 经典红，代表原始 GraphST baseline
    "Smooth only": "#2ca02c",   # 经典绿
    "Sharpen only": "#ff7f0e",  # 经典橙
    "Ada-GraphST": "#1f77b4",   # 经典蓝
}

# 折线图中三个切片的颜色
DATASET_LINE_COLORS = {
    "151508": "#1f77b4",  # 蓝
    "151670": "#ff7f0e",  # 橙
    "151675": "#2ca02c",  # 绿
}

# True 表示所有组都重新训练
# False 表示 Baseline 和 Ada-GraphST 优先读取已有 h5ad
FORCE_RERUN_ALL = False


# =========================================================
# 6.1 全局绘图字号和柱状图参数
# =========================================================
TITLE_FONTSIZE = 14
METHOD_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 12
AXIS_LABEL_FONTSIZE = 15
VALUE_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 12

BAR_WIDTH = 0.45
BAR_GAP_SCALE = 0.60

AXIS_LINEWIDTH = 1.0
TICK_LENGTH = 4
TICK_WIDTH = 1.0


# =========================================================
# 7. 工具函数
# =========================================================
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def attach_ground_truth(adata, file_fold):
    df_meta = pd.read_csv(os.path.join(file_fold, "metadata.tsv"), sep="\t")

    barcode_col = None
    for c in df_meta.columns:
        if "barcode" in c.lower():
            barcode_col = c
            break

    if barcode_col is not None:
        df_meta = df_meta.set_index(barcode_col)
        df_meta = df_meta.loc[adata.obs_names]
        adata.obs["ground_truth"] = df_meta["layer_guess"]
    else:
        if "layer_guess" not in df_meta.columns:
            raise KeyError("metadata.tsv does not contain column 'layer_guess'.")
        adata.obs["ground_truth"] = df_meta["layer_guess"].values

    return adata


def evaluate_ari_nmi(adata):
    if "ground_truth" not in adata.obs.columns:
        raise KeyError("adata.obs['ground_truth'] not found.")

    if "domain" not in adata.obs.columns:
        if "mclust" in adata.obs.columns:
            adata.obs["domain"] = adata.obs["mclust"].astype("category")
        else:
            raise KeyError("Neither adata.obs['domain'] nor adata.obs['mclust'] found.")

    adata_eval = adata[~pd.isnull(adata.obs["ground_truth"])].copy()

    ari = metrics.adjusted_rand_score(
        adata_eval.obs["domain"],
        adata_eval.obs["ground_truth"]
    )

    nmi = metrics.normalized_mutual_info_score(
        adata_eval.obs["domain"],
        adata_eval.obs["ground_truth"]
    )

    return float(ari), float(nmi)


def build_candidate(variant_name, variant_cfg):
    candidate = dict(common_params)
    candidate["w_smooth"] = float(variant_cfg["w_smooth"])
    candidate["w_sharpen"] = float(variant_cfg["w_sharpen"])
    candidate["name"] = variant_name.replace(" ", "_").replace("-", "_")
    return candidate


def get_existing_h5ad_path(dataset, existing_type):
    if existing_type == "baseline":
        return os.path.join(previous_h5ad_root, f"baseline_{dataset}.h5ad")
    elif existing_type == "ours":
        return os.path.join(previous_h5ad_root, f"ours_best_{dataset}.h5ad")
    return None


def save_h5ad_with_meta(
    adata,
    save_path,
    dataset,
    variant_name,
    ari,
    nmi,
    candidate,
    n_clusters
):
    adata.uns["run_info"] = {
        "dataset": dataset,
        "variant_name": variant_name,
        "ARI": float(ari) if pd.notna(ari) else np.nan,
        "NMI": float(nmi) if pd.notna(nmi) else np.nan,
        "n_clusters": int(n_clusters),
        "radius": int(radius),
        "seed": int(seed),
        "epochs": int(candidate["epochs"]),
        "dim_output": int(candidate["dim_output"]),
        "datatype": candidate["datatype"],
        "warmup_epochs": int(candidate["warmup_epochs"]),
        "update_interval": int(candidate["update_interval"]),
        "graph_update_rate": float(candidate["graph_update_rate"]),
        "gamma": float(candidate["gamma"]),
        "graph_reg_weight": float(candidate["graph_reg_weight"]),
        "use_learnable_proj": bool(candidate["use_learnable_proj"]),
        "w_smooth": float(candidate["w_smooth"]),
        "w_sharpen": float(candidate["w_sharpen"]),
    }

    adata.write(save_path)
    print(f"✅ saved h5ad: {save_path}")


def run_one_candidate_and_return_adata(
    adata_input,
    candidate,
    file_fold,
    n_clusters,
    radius,
    seed=41
):
    cand_name = candidate["name"]

    print("\n" + "-" * 100)
    print(f"Running candidate: {cand_name}")
    print(f"w_smooth={candidate['w_smooth']}, w_sharpen={candidate['w_sharpen']}")
    print("-" * 100)

    adata = adata_input.copy()

    graphst_params = {
        k: v for k, v in candidate.items()
        if k != "name"
    }

    model = GraphST(adata, **graphst_params)
    adata = model.train()

    if "emb" not in adata.obsm:
        raise KeyError("adata.obsm['emb'] not found after training.")

    emb = np.asarray(adata.obsm["emb"], dtype=np.float64)

    if not np.isfinite(emb).all():
        raise ValueError("adata.obsm['emb'] contains NaN or Inf.")

    n_pcs = min(20, emb.shape[1], max(2, emb.shape[0] - 1))

    adata.obsm["emb_pca"] = PCA(
        n_components=n_pcs,
        random_state=seed
    ).fit_transform(emb)

    clustering(
        adata,
        n_clusters,
        radius=radius,
        method="mclust",
        refinement=True
    )

    adata = attach_ground_truth(adata, file_fold)
    ari, nmi = evaluate_ari_nmi(adata)

    print(f"Candidate: {cand_name}")
    print(f"ARI: {ari:.6f}")
    print(f"NMI: {nmi:.6f}")

    del model
    clear_memory()

    return adata, ari, nmi


def run_or_load_variant(dataset, variant_name, variant_cfg):
    file_fold = os.path.join(data_root, dataset)
    n_clusters = cluster_map[dataset]

    candidate = build_candidate(variant_name, variant_cfg)

    variant_safe = variant_name.replace(" ", "_").replace("-", "_")
    save_dir = os.path.join(h5ad_out_root, dataset)
    os.makedirs(save_dir, exist_ok=True)

    save_h5ad_path = os.path.join(save_dir, f"{dataset}_{variant_safe}.h5ad")

    # =====================================================
    # 1. Baseline / Ada-GraphST 优先读取已有 h5ad
    # =====================================================
    if variant_cfg.get("reuse_existing", False) and not FORCE_RERUN_ALL:
        existing_path = get_existing_h5ad_path(dataset, variant_cfg["existing_type"])

        if existing_path is not None and os.path.exists(existing_path):
            print("\n" + "-" * 100)
            print(f"Loading existing h5ad for {dataset} | {variant_name}")
            print(existing_path)
            print("-" * 100)

            adata = sc.read_h5ad(existing_path)

            if "ground_truth" not in adata.obs.columns:
                adata = attach_ground_truth(adata, file_fold)

            if "domain" not in adata.obs.columns and "mclust" in adata.obs.columns:
                adata.obs["domain"] = adata.obs["mclust"].astype("category")

            # 如果已有 h5ad 缺少 domain，则重新聚类
            if "domain" not in adata.obs.columns:
                if "emb_pca" not in adata.obsm:
                    if "emb" not in adata.obsm:
                        raise KeyError("Existing h5ad has neither emb_pca nor emb.")

                    emb = np.asarray(adata.obsm["emb"], dtype=np.float64)
                    n_pcs = min(20, emb.shape[1], max(2, emb.shape[0] - 1))
                    adata.obsm["emb_pca"] = PCA(
                        n_components=n_pcs,
                        random_state=seed
                    ).fit_transform(emb)

                clustering(
                    adata,
                    n_clusters,
                    radius=radius,
                    method="mclust",
                    refinement=True
                )

            ari, nmi = evaluate_ari_nmi(adata)

            save_h5ad_with_meta(
                adata=adata,
                save_path=save_h5ad_path,
                dataset=dataset,
                variant_name=variant_name,
                ari=ari,
                nmi=nmi,
                candidate=candidate,
                n_clusters=n_clusters
            )

            clear_memory()

            return {
                "dataset": dataset,
                "variant": variant_name,
                "w_smooth": candidate["w_smooth"],
                "w_sharpen": candidate["w_sharpen"],
                "ARI": ari,
                "NMI": nmi,
                "h5ad_path": save_h5ad_path,
                "source": "loaded_existing_h5ad",
                "status": "success",
                "error": "",
            }

    # =====================================================
    # 2. Smooth only / Sharpen only，或者找不到已有 h5ad，则重新训练
    # =====================================================
    print("\n" + "=" * 120)
    print(f"Rerun GraphST for {dataset} | {variant_name}")
    print("=" * 120)

    adata_raw = sc.read_visium(
        file_fold,
        count_file="filtered_feature_bc_matrix.h5",
        load_images=True
    )
    adata_raw.var_names_make_unique()

    adata, ari, nmi = run_one_candidate_and_return_adata(
        adata_input=adata_raw,
        candidate=candidate,
        file_fold=file_fold,
        n_clusters=n_clusters,
        radius=radius,
        seed=seed
    )

    save_h5ad_with_meta(
        adata=adata,
        save_path=save_h5ad_path,
        dataset=dataset,
        variant_name=variant_name,
        ari=ari,
        nmi=nmi,
        candidate=candidate,
        n_clusters=n_clusters
    )

    del adata
    del adata_raw
    clear_memory()

    return {
        "dataset": dataset,
        "variant": variant_name,
        "w_smooth": candidate["w_smooth"],
        "w_sharpen": candidate["w_sharpen"],
        "ARI": ari,
        "NMI": nmi,
        "h5ad_path": save_h5ad_path,
        "source": "rerun_graphst",
        "status": "success",
        "error": "",
    }


# =========================================================
# 8. 结果整理
# =========================================================
def prepare_result_df(df):
    df = df.copy()
    df = df[df["status"] == "success"].copy()

    df["dataset"] = df["dataset"].astype(str)
    df["variant"] = df["variant"].astype(str)

    df["dataset"] = pd.Categorical(
        df["dataset"],
        categories=dataset_list,
        ordered=True
    )
    df["variant"] = pd.Categorical(
        df["variant"],
        categories=VARIANT_ORDER,
        ordered=True
    )

    df = df.sort_values(["dataset", "variant"]).reset_index(drop=True)
    return df


# =========================================================
# 8.1 画图辅助函数
# =========================================================
def style_axis_like_previous(ax):
    """
    统一成前面柱状图的论文风格：
    - 白底
    - 无网格
    - 只保留左边框和下边框
    - 刻度向外
    """
    ax.set_facecolor("white")
    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(AXIS_LINEWIDTH)
    ax.spines["bottom"].set_linewidth(AXIS_LINEWIDTH)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        length=TICK_LENGTH,
        width=TICK_WIDTH,
        colors="black"
    )


# =========================================================
# 9. 画图：每个切片一个子图，三个切片放在一个大图里
# =========================================================
def plot_per_dataset_barplots(df, metric_col, save_png, save_pdf):
    datasets = dataset_list

    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(5.2 * len(datasets), 4.2),
        dpi=300
    )
    fig.patch.set_facecolor("white")

    if len(datasets) == 1:
        axes = [axes]

    for idx, (ax, dataset) in enumerate(zip(axes, datasets), start=1):
        sub = df[df["dataset"].astype(str) == dataset].copy()

        sub["variant"] = pd.Categorical(
            sub["variant"],
            categories=VARIANT_ORDER,
            ordered=True
        )
        sub = sub.sort_values("variant").reset_index(drop=True)

        methods = sub["variant"].astype(str).tolist()
        scores = sub[metric_col].astype(float).tolist()
        colors = [METHOD_COLORS.get(m, "#999999") for m in methods]

        x = np.arange(len(methods)) * BAR_GAP_SCALE

        bars = ax.bar(
            x,
            scores,
            width=BAR_WIDTH,
            color=colors,
            edgecolor="none",
            linewidth=0
        )

        ax.set_title(
            f"({idx}) DLPFC {dataset}",
            fontsize=TITLE_FONTSIZE,
            fontweight="normal",
            pad=8
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            methods,
            rotation=35,
            ha="right",
            fontsize=METHOD_LABEL_FONTSIZE,
            fontweight="normal"
        )

        ax.set_ylabel(
            metric_col,
            fontsize=AXIS_LABEL_FONTSIZE,
            fontweight="normal"
        )

        ax.tick_params(axis="x", labelsize=METHOD_LABEL_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)

        ymax = max(scores) if len(scores) > 0 else 1.0
        ymin = min(scores) if len(scores) > 0 else 0.0

        lower = min(0.0, ymin - 0.05)
        upper = max(1.0, ymax * 1.18)
        ax.set_ylim(lower, upper)

        if len(x) > 0:
            ax.set_xlim(
                x[0] - BAR_WIDTH / 2 - 0.08,
                x[-1] + BAR_WIDTH / 2 + 0.08
            )

        style_axis_like_previous(ax)

        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=VALUE_LABEL_FONTSIZE,
                fontweight="normal",
                color="black",
                clip_on=False
            )

    fig.suptitle(
        f"{metric_col} of Ablation Variants Across 3 DLPFC Slices",
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
        y=1.02
    )

    plt.tight_layout(w_pad=2.0)
    plt.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.savefig(save_pdf, bbox_inches="tight")
    plt.close()

    print(f"✅ saved figure: {save_png}")
    print(f"✅ saved figure: {save_pdf}")


def plot_ablation_lineplot(df, metric_col, save_png, save_pdf):
    """
    折线图：
    x 轴是消融组；
    y 轴是 ARI 或 NMI；
    每条线代表一个切片。
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(VARIANT_ORDER))

    for dataset in dataset_list:
        sub = df[df["dataset"].astype(str) == dataset].copy()

        sub["variant"] = pd.Categorical(
            sub["variant"],
            categories=VARIANT_ORDER,
            ordered=True
        )
        sub = sub.sort_values("variant").reset_index(drop=True)

        if len(sub) != len(VARIANT_ORDER):
            print(f"⚠️ {dataset} 缺少某些消融组，跳过折线图中的该切片。")
            continue

        y = sub[metric_col].astype(float).values

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.0,
            markersize=6,
            label=f"DLPFC {dataset}",
            color=DATASET_LINE_COLORS.get(dataset, None)
        )

        # 每个点上标分数
        for xi, yi in zip(x, y):
            ax.text(
                xi,
                yi + 0.006,
                f"{yi:.3f}",
                ha="center",
                va="bottom",
                fontsize=VALUE_LABEL_FONTSIZE,
                fontweight="normal",
                color="black",
                clip_on=False
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        VARIANT_ORDER,
        rotation=35,
        ha="right",
        fontsize=METHOD_LABEL_FONTSIZE,
        fontweight="normal"
    )

    ax.set_ylabel(
        metric_col,
        fontsize=AXIS_LABEL_FONTSIZE,
        fontweight="normal"
    )

    ax.set_title(
        f"{metric_col} Trend Across 3 DLPFC Slices",
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
        pad=8
    )

    ax.tick_params(axis="x", labelsize=METHOD_LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)

    ax.legend(
        frameon=True,
        fontsize=LEGEND_FONTSIZE
    )

    # ARI / NMI 通常 0-1，自动留一点空间
    ymax = df[metric_col].max()
    ymin = df[metric_col].min()
    ax.set_ylim(
        min(0.0, ymin - 0.05),
        max(1.0, ymax * 1.15)
    )

    style_axis_like_previous(ax)

    plt.tight_layout()
    plt.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.savefig(save_pdf, bbox_inches="tight")
    plt.close()

    print(f"✅ saved lineplot: {save_png}")
    print(f"✅ saved lineplot: {save_pdf}")


def plot_delta_per_dataset_barplots(df, metric_col, save_png, save_pdf):
    records = []

    for dataset in dataset_list:
        sub = df[df["dataset"].astype(str) == dataset].copy()

        baseline_row = sub[sub["variant"].astype(str) == "Baseline"]
        if baseline_row.empty:
            print(f"⚠️ {dataset} has no Baseline, skip delta.")
            continue

        baseline_score = float(baseline_row[metric_col].iloc[0])

        for _, row in sub.iterrows():
            records.append({
                "dataset": dataset,
                "variant": str(row["variant"]),
                f"Delta_{metric_col}": float(row[metric_col]) - baseline_score
            })

    delta_df = pd.DataFrame(records)

    if delta_df.empty:
        print(f"⚠️ No delta data for {metric_col}")
        return

    delta_col = f"Delta_{metric_col}"

    fig, axes = plt.subplots(
        1,
        len(dataset_list),
        figsize=(5.2 * len(dataset_list), 4.2),
        dpi=300
    )
    fig.patch.set_facecolor("white")

    if len(dataset_list) == 1:
        axes = [axes]

    for idx, (ax, dataset) in enumerate(zip(axes, dataset_list), start=1):
        sub = delta_df[delta_df["dataset"].astype(str) == dataset].copy()

        sub["variant"] = pd.Categorical(
            sub["variant"],
            categories=VARIANT_ORDER,
            ordered=True
        )
        sub = sub.sort_values("variant").reset_index(drop=True)

        methods = sub["variant"].astype(str).tolist()
        scores = sub[delta_col].astype(float).tolist()
        colors = [METHOD_COLORS.get(m, "#999999") for m in methods]

        x = np.arange(len(methods)) * BAR_GAP_SCALE

        bars = ax.bar(
            x,
            scores,
            width=BAR_WIDTH,
            color=colors,
            edgecolor="none",
            linewidth=0
        )

        ax.axhline(0, color="black", linewidth=1.0)

        ax.set_title(
            f"({idx}) DLPFC {dataset}",
            fontsize=TITLE_FONTSIZE,
            fontweight="normal",
            pad=8
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            methods,
            rotation=35,
            ha="right",
            fontsize=METHOD_LABEL_FONTSIZE,
            fontweight="normal"
        )

        ax.set_ylabel(
            f"Δ{metric_col} vs Baseline",
            fontsize=AXIS_LABEL_FONTSIZE,
            fontweight="normal"
        )

        ax.tick_params(axis="x", labelsize=METHOD_LABEL_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)

        ymin = min(scores) if len(scores) > 0 else -0.1
        ymax = max(scores) if len(scores) > 0 else 0.1
        ax.set_ylim(min(-0.10, ymin * 1.25), max(0.10, ymax * 1.25))

        if len(x) > 0:
            ax.set_xlim(
                x[0] - BAR_WIDTH / 2 - 0.08,
                x[-1] + BAR_WIDTH / 2 + 0.08
            )

        style_axis_like_previous(ax)

        for bar, score in zip(bars, scores):
            offset = 0.006 if score >= 0 else -0.010
            va = "bottom" if score >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{score:+.3f}",
                ha="center",
                va=va,
                fontsize=VALUE_LABEL_FONTSIZE,
                fontweight="normal",
                color="black",
                clip_on=False
            )

    fig.suptitle(
        f"Δ{metric_col} vs Baseline Across 3 DLPFC Slices",
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
        y=1.02
    )

    plt.tight_layout(w_pad=2.0)
    plt.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.savefig(save_pdf, bbox_inches="tight")
    plt.close()

    delta_csv = save_png.replace(".png", ".csv")
    delta_df.to_csv(delta_csv, index=False)

    print(f"✅ saved delta figure: {save_png}")
    print(f"✅ saved delta figure: {save_pdf}")
    print(f"✅ saved delta csv: {delta_csv}")


# =========================================================
# 10. 主程序
# =========================================================
if __name__ == "__main__":
    seed_everything(seed)

    all_records = []
    failed_records = []

    for dataset in dataset_list:
        print("\n" + "=" * 120)
        print(f"START DATASET: {dataset}")
        print("=" * 120)

        for variant_name in VARIANT_ORDER:
            variant_cfg = ABLATION_VARIANTS[variant_name]

            try:
                row = run_or_load_variant(
                    dataset=dataset,
                    variant_name=variant_name,
                    variant_cfg=variant_cfg
                )
                all_records.append(row)

            except Exception as e:
                err = traceback.format_exc()

                print(f"❌ FAILED | dataset={dataset} | variant={variant_name}")
                print(err)

                failed_records.append({
                    "dataset": dataset,
                    "variant": variant_name,
                    "error": str(e),
                    "traceback": err,
                })

                all_records.append({
                    "dataset": dataset,
                    "variant": variant_name,
                    "w_smooth": variant_cfg["w_smooth"],
                    "w_sharpen": variant_cfg["w_sharpen"],
                    "ARI": np.nan,
                    "NMI": np.nan,
                    "h5ad_path": "",
                    "source": "",
                    "status": "failed",
                    "error": str(e),
                })

                clear_memory()

    # =====================================================
    # 11. 保存详细结果
    # =====================================================
    detail_df = pd.DataFrame(all_records)

    detail_csv = os.path.join(output_root, "ablation_3slices_detail_results.csv")
    detail_df.to_csv(detail_csv, index=False)
    print(f"\n✅ saved detail csv: {detail_csv}")

    failed_df = pd.DataFrame(failed_records)
    failed_csv = os.path.join(output_root, "ablation_3slices_failed_log.csv")
    failed_df.to_csv(failed_csv, index=False)
    print(f"✅ saved failed csv: {failed_csv}")

    ok_df = prepare_result_df(detail_df)

    if ok_df.empty:
        print("❌ No successful results. Skip plotting.")

    else:
        # =================================================
        # 12. 保存均值汇总
        # =================================================
        summary_df = (
            ok_df.groupby("variant", observed=False)
            .agg(
                mean_ARI=("ARI", "mean"),
                std_ARI=("ARI", "std"),
                mean_NMI=("NMI", "mean"),
                std_NMI=("NMI", "std"),
                n_slices=("dataset", "count"),
            )
            .reindex(VARIANT_ORDER)
            .reset_index()
        )

        summary_csv = os.path.join(output_root, "ablation_3slices_mean_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"✅ saved summary csv: {summary_csv}")

        # =================================================
        # 13. 画成统一风格：每个切片一个子图
        # =================================================

        # ARI：每个切片一个柱状图，三个柱状图放在一个大图
        plot_per_dataset_barplots(
            ok_df,
            metric_col="ARI",
            save_png=os.path.join(fig_out_root, "ablation_per_slice_ARI_subplots.png"),
            save_pdf=os.path.join(fig_out_root, "ablation_per_slice_ARI_subplots.pdf"),
        )

        # NMI：每个切片一个柱状图，三个柱状图放在一个大图
        plot_per_dataset_barplots(
            ok_df,
            metric_col="NMI",
            save_png=os.path.join(fig_out_root, "ablation_per_slice_NMI_subplots.png"),
            save_pdf=os.path.join(fig_out_root, "ablation_per_slice_NMI_subplots.pdf"),
        )

        # ΔARI：每个切片相对 Baseline 的提升
        plot_delta_per_dataset_barplots(
            ok_df,
            metric_col="ARI",
            save_png=os.path.join(fig_out_root, "ablation_per_slice_delta_ARI_subplots.png"),
            save_pdf=os.path.join(fig_out_root, "ablation_per_slice_delta_ARI_subplots.pdf"),
        )

        # ΔNMI：每个切片相对 Baseline 的提升
        plot_delta_per_dataset_barplots(
            ok_df,
            metric_col="NMI",
            save_png=os.path.join(fig_out_root, "ablation_per_slice_delta_NMI_subplots.png"),
            save_pdf=os.path.join(fig_out_root, "ablation_per_slice_delta_NMI_subplots.pdf"),
        )

        # ARI 折线图：每条线代表一个切片
        plot_ablation_lineplot(
            ok_df,
            metric_col="ARI",
            save_png=os.path.join(fig_out_root, "ablation_ARI_lineplot.png"),
            save_pdf=os.path.join(fig_out_root, "ablation_ARI_lineplot.pdf"),
        )

        # NMI 折线图：每条线代表一个切片
        plot_ablation_lineplot(
            ok_df,
            metric_col="NMI",
            save_png=os.path.join(fig_out_root, "ablation_NMI_lineplot.png"),
            save_pdf=os.path.join(fig_out_root, "ablation_NMI_lineplot.pdf"),
        )

        print("\n" + "=" * 120)
        print("FINAL DETAIL RESULTS")
        print("=" * 120)
        print(
            ok_df[
                ["dataset", "variant", "w_smooth", "w_sharpen", "ARI", "NMI", "source"]
            ].to_string(index=False)
        )

        print("\n" + "=" * 120)
        print("FINAL MEAN SUMMARY")
        print("=" * 120)
        print(summary_df.to_string(index=False))

    print("\n🎉 Ablation experiment finished.")
    print(f"Output root: {output_root}")