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
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA

try:
    import torch
except ImportError:
    torch = None

try:
    import harmonypy as hm
except ImportError:
    raise ImportError("请先安装 harmonypy: pip install harmonypy")

# =========================================================
# A. 数据集配置（Human Breast Cancer）
# =========================================================
PROJECT_ROOT = "/data2/liangyefeng/My_GraphST_Innovation"
DATASET_NAME = "Human Breast Cancer"
H5AD_PATH = "/data2/liangyefeng/My_GraphST_Innovation/data/Human-Breast-Cancer-Block-A/PASTE_then_GraphST_section1_section2/HBCA_section1_section2_PASTE_aligned_for_GraphST.h5ad"

CENTER_WS = 1.40
CENTER_WSH = 0.90
FIXED_GAMMA = 2.5
FIXED_WARMUP = 200
FIXED_INTERVAL = 20
FIXED_EMA = 0.3

OUT_ROOT = os.path.join(PROJECT_ROOT, "Result_ParameterSensitivity_OnlyWS_WSH", DATASET_NAME.replace(" ", "_"))
os.makedirs(OUT_ROOT, exist_ok=True)

SEED = 50
EPOCHS = 600
DIM_OUTPUT = 64
GRAPHST_DATATYPE = "10X"
USE_LEARNABLE_PROJ = False

STEP = 0.2

W_SMOOTH_LIST = sorted(set([
    round(max(0.0, CENTER_WS - STEP), 2),
    round(CENTER_WS, 2),
    round(CENTER_WS + STEP, 2),
]))

W_SHARPEN_LIST = sorted(set([
    round(max(0.0, CENTER_WSH - STEP), 2),
    round(CENTER_WSH, 2),
    round(CENTER_WSH + STEP, 2),
]))

SUMMARY_CSV = os.path.join(OUT_ROOT, "Human_Breast_Cancer_parameter_sensitivity.csv")
HEATMAP_PNG = os.path.join(OUT_ROOT, "Human_Breast_Cancer_heatmap.png")
HEATMAP_PDF = os.path.join(OUT_ROOT, "Human_Breast_Cancer_heatmap.pdf")
LINE_PNG = os.path.join(OUT_ROOT, "Human_Breast_Cancer_lineplot.png")
LINE_PDF = os.path.join(OUT_ROOT, "Human_Breast_Cancer_lineplot.pdf")

# =========================================================
# B. 环境
# =========================================================
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
os.environ["R_HOME"] = "/data2/liangyefeng/miniconda3/envs/graphst/lib/R"

from GraphST.GraphST import GraphST

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"

# =========================================================
# C. 工具函数
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
    adata.obsm["emb_pca"] = PCA(n_components=n_pca, random_state=seed).fit_transform(emb)
    lisi = hm.compute_lisi(adata.obsm["emb_pca"], adata.obs[["batch"]], ["batch"])
    return float(np.mean(lisi))


def build_graphst_kwargs(adata, device, w_smooth, w_sharpen):
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
        "w_smooth": float(w_smooth),
        "w_sharpen": float(w_sharpen),
        "gamma": float(FIXED_GAMMA),
        "warmup": int(FIXED_WARMUP),
        "warmup_epochs": int(FIXED_WARMUP),
        "interval": int(FIXED_INTERVAL),
        "update_interval": int(FIXED_INTERVAL),
        "ema": float(FIXED_EMA),
        "ema_decay": float(FIXED_EMA),
        "boundary_ema": float(FIXED_EMA),
    }

    final_kwargs = {}
    for k, v in kwargs.items():
        if k in supported:
            final_kwargs[k] = v
    return final_kwargs


def plot_parameter_sensitivity_heatmap(df, save_png, save_pdf):
    sub = df[df["iLISI"].notna()].copy()
    if sub.empty:
        print("⚠️ 没有可用参数敏感性结果，跳过 heatmap。")
        return

    pivot = pd.pivot_table(
        sub,
        index="w_sharpen",
        columns="w_smooth",
        values="iLISI",
        aggfunc="mean"
    ).sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    fig.patch.set_facecolor("white")
    mat = pivot.values.astype(float)
    im = ax.imshow(mat, aspect="auto")

    best_flat = np.nanargmax(mat)
    best_i, best_j = np.unravel_index(best_flat, mat.shape)

    ax.set_title(f"{DATASET_NAME}\nParameter Sensitivity Heatmap", fontsize=13, fontweight="bold")
    ax.set_xlabel("w_smooth", fontsize=11)
    ax.set_ylabel("w_sharpen", fontsize=11)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{y:.2f}" for y in pivot.index])

    mid_val = (np.nanmin(mat) + np.nanmax(mat)) / 2.0
    for i, row_val in enumerate(pivot.index):
        for j, col_val in enumerate(pivot.columns):
            val = float(pivot.loc[row_val, col_val])
            txt_color = "white" if val >= mid_val else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color=txt_color)

            if np.isclose(float(col_val), CENTER_WS) and np.isclose(float(row_val), CENTER_WSH):
                rect_center = Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2.6)
                ax.add_patch(rect_center)

    rect_best = Rectangle((best_j - 0.5, best_i - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2.4, linestyle="--")
    ax.add_patch(rect_best)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("iLISI")

    plt.tight_layout()
    plt.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.savefig(save_pdf, bbox_inches="tight")
    plt.close()


def plot_parameter_sensitivity_lines(sens_df, save_png, save_pdf):
    if sens_df is None or sens_df.empty:
        print("⚠️ 没有可用参数敏感性结果，跳过 line plot。")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    fig.patch.set_facecolor("white")
    ax1, ax2 = axes

    ax1.set_facecolor("#F5F5F5")
    ax2.set_facecolor("#F5F5F5")

    sub = sens_df[sens_df["iLISI"].notna()].copy()
    if sub.empty:
        print("⚠️ 没有可用参数敏感性结果，跳过 line plot。")
        plt.close()
        return

    dataset_name = str(sub["Dataset"].iloc[0])
    center_ws = float(sub["Center_w_smooth"].dropna().iloc[0])
    center_wsp = float(sub["Center_w_sharpen"].dropna().iloc[0])

    smooth_sub = sub[np.isclose(sub["w_sharpen"].astype(float), center_wsp)].copy()
    smooth_sub = smooth_sub.sort_values("w_smooth").reset_index(drop=True)

    if not smooth_sub.empty:
        ax1.plot(
            smooth_sub["w_smooth"].values,
            smooth_sub["iLISI"].values,
            marker="o",
            linewidth=2.5,
            markersize=7
        )

        center_row = smooth_sub[np.isclose(smooth_sub["w_smooth"].astype(float), center_ws)]
        if not center_row.empty:
            ax1.scatter(
                center_ws,
                float(center_row["iLISI"].iloc[0]),
                s=100,
                edgecolors="black",
                linewidths=1.2,
                zorder=6
            )

        ymin1 = float(smooth_sub["iLISI"].min())
        ymax1 = float(smooth_sub["iLISI"].max())
        offset1 = max(0.01, (ymax1 - ymin1) * 0.08 if ymax1 > ymin1 else 0.01)

        for x, y in zip(smooth_sub["w_smooth"].values, smooth_sub["iLISI"].values):
            ax1.text(x, y + offset1, f"{y:.3f}", ha="center", va="bottom", fontsize=9)

        ax1.set_ylim(
            ymin1 - max(0.02, (ymax1 - ymin1) * 0.15 if ymax1 > ymin1 else 0.02),
            ymax1 + max(0.04, (ymax1 - ymin1) * 0.25 if ymax1 > ymin1 else 0.04)
        )

    ax1.set_title(f"{dataset_name}\nSensitivity to w_smooth", fontsize=13, fontweight="bold")
    ax1.set_xlabel(f"w_smooth  (fixed w_sharpen={center_wsp:.2f})", fontsize=11)
    ax1.set_ylabel("iLISI", fontsize=11)
    ax1.set_xticks(sorted(smooth_sub["w_smooth"].astype(float).unique()))
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    sharpen_sub = sub[np.isclose(sub["w_smooth"].astype(float), center_ws)].copy()
    sharpen_sub = sharpen_sub.sort_values("w_sharpen").reset_index(drop=True)

    if not sharpen_sub.empty:
        ax2.plot(
            sharpen_sub["w_sharpen"].values,
            sharpen_sub["iLISI"].values,
            marker="o",
            linewidth=2.5,
            markersize=7
        )

        center_row = sharpen_sub[np.isclose(sharpen_sub["w_sharpen"].astype(float), center_wsp)]
        if not center_row.empty:
            ax2.scatter(
                center_wsp,
                float(center_row["iLISI"].iloc[0]),
                s=100,
                edgecolors="black",
                linewidths=1.2,
                zorder=6
            )

        ymin2 = float(sharpen_sub["iLISI"].min())
        ymax2 = float(sharpen_sub["iLISI"].max())
        offset2 = max(0.01, (ymax2 - ymin2) * 0.08 if ymax2 > ymin2 else 0.01)

        for x, y in zip(sharpen_sub["w_sharpen"].values, sharpen_sub["iLISI"].values):
            ax2.text(x, y + offset2, f"{y:.3f}", ha="center", va="bottom", fontsize=9)

        ax2.set_ylim(
            ymin2 - max(0.02, (ymax2 - ymin2) * 0.15 if ymax2 > ymin2 else 0.02),
            ymax2 + max(0.04, (ymax2 - ymin2) * 0.25 if ymax2 > ymin2 else 0.04)
        )

    ax2.set_title(f"{dataset_name}\nSensitivity to w_sharpen", fontsize=13, fontweight="bold")
    ax2.set_xlabel(f"w_sharpen  (fixed w_smooth={center_ws:.2f})", fontsize=11)
    ax2.set_ylabel("iLISI", fontsize=11)
    ax2.set_xticks(sorted(sharpen_sub["w_sharpen"].astype(float).unique()))
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.savefig(save_pdf, bbox_inches="tight")
    plt.close()

# =========================================================
# D. 主运行
# =========================================================
if __name__ == "__main__":
    seed_everything(SEED)

    adata_input = load_integrated_dataset(H5AD_PATH)
    device = "cuda:0" if (torch is not None and torch.cuda.is_available()) else "cpu"

    rows = []

    print(f"✅ DATASET_NAME = {DATASET_NAME}")
    print(f"✅ W_SMOOTH_LIST = {W_SMOOTH_LIST}")
    print(f"✅ W_SHARPEN_LIST = {W_SHARPEN_LIST}")

    for w_smooth in W_SMOOTH_LIST:
        for w_sharpen in W_SHARPEN_LIST:
            print("-" * 100)
            print(f"🚀 {DATASET_NAME} | w_smooth={w_smooth:.2f} | w_sharpen={w_sharpen:.2f}")

            try:
                adata_run = adata_input.copy()
                graphst_kwargs = build_graphst_kwargs(adata_run, device, w_smooth, w_sharpen)

                seed_everything(SEED)
                model = GraphST(**graphst_kwargs)
                out = model.train()
                if out is not None:
                    adata_run = out

                ilisi = compute_ilisi_from_emb(adata_run, emb_key="emb", seed=SEED)

                row = {
                    "Dataset": DATASET_NAME,
                    "w_smooth": float(w_smooth),
                    "w_sharpen": float(w_sharpen),
                    "iLISI": float(ilisi),
                    "Center_w_smooth": float(CENTER_WS),
                    "Center_w_sharpen": float(CENTER_WSH),
                    "gamma": float(FIXED_GAMMA),
                    "warmup": int(FIXED_WARMUP),
                    "interval": int(FIXED_INTERVAL),
                    "ema": float(FIXED_EMA),
                    "Input_H5AD": H5AD_PATH,
                    "Seed": SEED,
                    "Device": device,
                    "N_Obs": int(adata_run.n_obs),
                    "N_Vars": int(adata_run.n_vars),
                    "Passed_GraphST_Kwargs": json.dumps(
                        {k: (str(v) if k == "adata" else v) for k, v in graphst_kwargs.items()},
                        ensure_ascii=False
                    ),
                    "Error": "",
                }

                print(f"✅ DONE | iLISI={ilisi:.6f}")

                del model
                del adata_run
                clear_memory()

            except Exception as e:
                print(f"❌ FAILED | {DATASET_NAME} | w_smooth={w_smooth:.2f} | w_sharpen={w_sharpen:.2f}")
                print(e)

                row = {
                    "Dataset": DATASET_NAME,
                    "w_smooth": float(w_smooth),
                    "w_sharpen": float(w_sharpen),
                    "iLISI": np.nan,
                    "Center_w_smooth": float(CENTER_WS),
                    "Center_w_sharpen": float(CENTER_WSH),
                    "gamma": float(FIXED_GAMMA),
                    "warmup": int(FIXED_WARMUP),
                    "interval": int(FIXED_INTERVAL),
                    "ema": float(FIXED_EMA),
                    "Input_H5AD": H5AD_PATH,
                    "Seed": SEED,
                    "Device": device,
                    "N_Obs": np.nan,
                    "N_Vars": np.nan,
                    "Passed_GraphST_Kwargs": "",
                    "Error": str(e),
                }
                clear_memory()

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"\n✅ 敏感度结果已保存: {SUMMARY_CSV}")

    plot_parameter_sensitivity_heatmap(df, HEATMAP_PNG, HEATMAP_PDF)
    print(f"✅ heatmap 已保存: {HEATMAP_PNG}")
    print(f"✅ heatmap 已保存: {HEATMAP_PDF}")

    plot_parameter_sensitivity_lines(df, LINE_PNG, LINE_PDF)
    print(f"✅ line plot 已保存: {LINE_PNG}")
    print(f"✅ line plot 已保存: {LINE_PDF}")

    print("\n最终结果：")
    print(df[["Dataset", "w_smooth", "w_sharpen", "iLISI"]].to_string(index=False))
