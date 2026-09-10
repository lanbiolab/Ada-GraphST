# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


# =========================================================
# 0. 全局画图格式配置
# =========================================================
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"


# =========================================================
# 1. 路径配置
# =========================================================
PROJECT_ROOT = "/data2/liangyefeng/My_GraphST_Innovation"

OUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "Result_Ablation_AdaGraphST_3Datasets"
)

SUMMARY_CSV = os.path.join(
    OUT_ROOT,
    "ablation_summary.csv"
)

OUT_PNG = os.path.join(
    OUT_ROOT,
    "ablation_raw_ilisi_barplot_replot_like_DLPFC.png"
)

OUT_PDF = os.path.join(
    OUT_ROOT,
    "ablation_raw_ilisi_barplot_replot_like_DLPFC.pdf"
)


# =========================================================
# 2. 数据集和消融组顺序
# =========================================================
DATASET_ORDER = [
    "Mouse Brain",
    "Human Breast Cancer",
    "Mouse Breast Cancer"
]

VARIANT_ORDER = [
    "Baseline",
    "Smooth only",
    "Sharpen only",
    "Ada-GraphST"
]


# =========================================================
# 3. 颜色：经典红、绿、橙、蓝风格
# =========================================================
METHOD_COLORS = {
    "Baseline": "#d62728",      # 红
    "Smooth only": "#2ca02c",   # 绿
    "Sharpen only": "#ff7f0e",  # 橙
    "Ada-GraphST": "#1f77b4",   # 蓝
}


# =========================================================
# 4. 图形参数：按照前面 DLPFC ARI 图格式
# =========================================================
TITLE_FONTSIZE = 14
METHOD_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 12
AXIS_LABEL_FONTSIZE = 15
VALUE_LABEL_FONTSIZE = 12

# 柱体宽度：较窄
BAR_WIDTH = 0.35

# 柱子间距压缩系数：
# 越小柱子越紧密
BAR_GAP_SCALE = 0.48

AXIS_LINEWIDTH = 1.0
TICK_LENGTH = 4
TICK_WIDTH = 1.0

# iLISI 两个 batch 时理论上限为 2
YLIM_TOP = 2.0


# =========================================================
# 5. 工具函数
# =========================================================
def normalize_variant_name(x):
    x = str(x).strip().lower().replace("_", " ")

    mapping = {
        "baseline": "Baseline",
        "smooth only": "Smooth only",
        "sharpen only": "Sharpen only",
        "ada graphst": "Ada-GraphST",
        "ada-graphst": "Ada-GraphST",
    }

    return mapping.get(x, str(x).strip())


def load_summary(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"没有找到结果 CSV 文件:\n{csv_path}\n\n"
            f"请先确认前面的 iLISI 消融实验代码已经运行完成，"
            f"并生成 ablation_summary.csv。"
        )

    df = pd.read_csv(csv_path)

    required_cols = ["Dataset", "Variant", "iLISI"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV 缺少必要列: {missing_cols}\n"
            f"当前列名: {list(df.columns)}"
        )

    df = df.copy()
    df["Dataset"] = df["Dataset"].astype(str).str.strip()
    df["Variant"] = df["Variant"].apply(normalize_variant_name)
    df["iLISI"] = pd.to_numeric(df["iLISI"], errors="coerce")

    df = df.dropna(subset=["iLISI"]).copy()

    df = df[df["Dataset"].isin(DATASET_ORDER)].copy()
    df = df[df["Variant"].isin(VARIANT_ORDER)].copy()

    df["Dataset"] = pd.Categorical(
        df["Dataset"],
        categories=DATASET_ORDER,
        ordered=True
    )

    df["Variant"] = pd.Categorical(
        df["Variant"],
        categories=VARIANT_ORDER,
        ordered=True
    )

    df = df.sort_values(["Dataset", "Variant"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("CSV 中没有可用于绘图的有效 iLISI 数据。")

    return df


def get_complete_datasets(df):
    complete_datasets = []

    for dataset_name in DATASET_ORDER:
        sub = df[df["Dataset"].astype(str) == dataset_name].copy()
        variants = sub["Variant"].astype(str).tolist()

        missing = [v for v in VARIANT_ORDER if v not in variants]

        if missing:
            print(f"[Warning] {dataset_name} 缺少分组: {missing}，该数据集绘图时跳过。")
        else:
            complete_datasets.append(dataset_name)

    if len(complete_datasets) == 0:
        raise ValueError("没有任何一个数据集包含完整的 4 个消融组，无法绘图。")

    return complete_datasets


def style_axis(ax):
    ax.set_facecolor("white")
    ax.grid(False)

    # 只保留左边框和下边框
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
# 6. 画 iLISI 消融柱状图
# =========================================================
def plot_ilisi_ablation_barplot(summary_df, save_png, save_pdf):
    complete_datasets = get_complete_datasets(summary_df)

    fig, axes = plt.subplots(
        1,
        len(complete_datasets),
        figsize=(5.0 * len(complete_datasets), 5.2),
        dpi=300
    )

    fig.patch.set_facecolor("white")

    if len(complete_datasets) == 1:
        axes = [axes]

    for idx, (ax, dataset_name) in enumerate(zip(axes, complete_datasets), start=1):
        df_sub = summary_df[summary_df["Dataset"].astype(str) == dataset_name].copy()

        df_sub["Variant"] = pd.Categorical(
            df_sub["Variant"],
            categories=VARIANT_ORDER,
            ordered=True
        )

        df_sub = df_sub.sort_values("Variant").reset_index(drop=True)

        methods = df_sub["Variant"].astype(str).tolist()
        scores = df_sub["iLISI"].astype(float).tolist()
        colors = [METHOD_COLORS.get(m, "#999999") for m in methods]

        # 用数值型 x 坐标压缩柱子之间的距离
        x = np.arange(len(methods)) * BAR_GAP_SCALE

        bars = ax.bar(
            x,
            scores,
            width=BAR_WIDTH,
            color=colors,
            edgecolor="none",
            linewidth=0
        )

        # 子图标题
        ax.set_title(
            f"({idx}) {dataset_name}",
            fontsize=TITLE_FONTSIZE,
            fontweight="normal",
            pad=18
        )

        # x 轴标签
        ax.set_xticks(x)
        ax.set_xticklabels(
            methods,
            rotation=35,
            ha="right",
            fontsize=METHOD_LABEL_FONTSIZE,
            fontweight="normal"
        )

        # y 轴
        ax.set_ylabel(
            "iLISI",
            fontsize=AXIS_LABEL_FONTSIZE,
            fontweight="normal"
        )

        ax.tick_params(axis="x", labelsize=METHOD_LABEL_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)

        # iLISI y 轴固定为 0 到 2
        ax.set_ylim(0, YLIM_TOP)
        ax.set_yticks(np.arange(0, YLIM_TOP + 0.001, 0.5))

        # 收紧每个子图内部左右空白
        ax.set_xlim(
            x[0] - BAR_WIDTH / 2 - 0.06,
            x[-1] + BAR_WIDTH / 2 + 0.06
        )

        style_axis(ax)

        # 柱子顶部数值
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=VALUE_LABEL_FONTSIZE,
                fontweight="normal",
                color="black",
                clip_on=False
            )



    plt.tight_layout(w_pad=1.6, rect=[0, 0, 1, 0.96])

    plt.savefig(
        save_png,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.savefig(
        save_pdf,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    print(f"[Saved] {save_png}")
    print(f"[Saved] {save_pdf}")


# =========================================================
# 7. 主程序
# =========================================================
if __name__ == "__main__":
    summary_df = load_summary(SUMMARY_CSV)

    print("用于绘图的数据：")
    print(
        summary_df[
            ["Dataset", "Variant", "iLISI"]
        ].to_string(index=False)
    )

    plot_ilisi_ablation_barplot(
        summary_df=summary_df,
        save_png=OUT_PNG,
        save_pdf=OUT_PDF
    )