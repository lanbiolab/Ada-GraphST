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
project_root = "/data2/liangyefeng/My_GraphST_Innovation"

file_fold = os.path.join(
    project_root,
    "data",
    "Human-Breast-Cancer-Block-A",
    "section1"
)

result_root = os.path.join(
    file_fold,
    "HBCA_GraphST_Ablation"
)

detail_csv = os.path.join(
    result_root,
    "HBCA_section1_ablation_results.csv"
)

out_png = os.path.join(
    result_root,
    "HBCA_section1_Ablation_ARI_barplot_replot.png"
)

out_pdf = os.path.join(
    result_root,
    "HBCA_section1_Ablation_ARI_barplot_replot.pdf"
)


# =========================================================
# 2. 图形参数
# =========================================================
TITLE_FONTSIZE = 14
METHOD_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 12
AXIS_LABEL_FONTSIZE = 15
VALUE_LABEL_FONTSIZE = 12

# 柱体宽度：保持较窄
BAR_WIDTH = 0.25

# 柱子间距压缩系数：
# 越小柱子越紧密
BAR_GAP_SCALE = 0.35

AXIS_LINEWIDTH = 1.0
TICK_LENGTH = 4
TICK_WIDTH = 1.0


# =========================================================
# 3. 消融组顺序和颜色
# =========================================================
VARIANT_ORDER = [
    "Baseline",
    "Smooth only",
    "Sharpen only",
    "Ada-GraphST"
]

# 经典红蓝绿橙风格
PLOT_COLORS = {
    "Baseline": "#d62728",      # 红
    "Smooth only": "#2ca02c",   # 绿
    "Sharpen only": "#ff7f0e",  # 橙
    "Ada-GraphST": "#1f77b4",   # 蓝
}


# =========================================================
# 4. 工具函数
# =========================================================
def load_ablation_results(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"没有找到结果 CSV 文件:\n{csv_path}\n\n"
            f"请先运行前面的 HBCA section1 消融实验代码，"
            f"确保生成 HBCA_section1_ablation_results.csv。"
        )

    df = pd.read_csv(csv_path)

    required_cols = ["Variant", "ARI"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV 缺少必要列: {missing_cols}\n"
            f"当前列名: {list(df.columns)}"
        )

    df = df.copy()
    df["Variant"] = df["Variant"].astype(str)
    df["ARI"] = pd.to_numeric(df["ARI"], errors="coerce")

    df = df.dropna(subset=["ARI"]).copy()

    df["Variant"] = pd.Categorical(
        df["Variant"],
        categories=VARIANT_ORDER,
        ordered=True
    )

    df = df.dropna(subset=["Variant"]).copy()
    df = df.sort_values("Variant").reset_index(drop=True)

    if df.empty:
        raise ValueError("CSV 中没有可用于绘图的有效 ARI 数据。")

    return df


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
# 5. 只画 ARI 柱状图
# =========================================================
def plot_ari_barplot(results_df, save_png, save_pdf):
    methods = results_df["Variant"].astype(str).tolist()
    scores = results_df["ARI"].astype(float).tolist()
    colors = [PLOT_COLORS.get(m, "#999999") for m in methods]

    # 用数值型 x 坐标压缩柱子间距
    x = np.arange(len(methods)) * BAR_GAP_SCALE

    fig, ax = plt.subplots(
        figsize=(4.2, 5.2),
        dpi=300,
        facecolor="white"
    )

    bars = ax.bar(
        x,
        scores,
        width=BAR_WIDTH,
        color=colors,
        edgecolor="none",
        linewidth=0
    )

    # 标题
    ax.set_title(
        "HBCA section1 Ablation Study (ARI)",
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
        pad=8
    )

    # y 轴
    ax.set_ylabel(
        "ARI",
        fontsize=AXIS_LABEL_FONTSIZE,
        fontweight="normal"
    )

    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)

    # x 轴方法名
    ax.set_xticks(x)
    ax.set_xticklabels(
        methods,
        rotation=35,
        ha="right",
        fontsize=METHOD_LABEL_FONTSIZE,
        fontweight="normal"
    )

    ax.tick_params(axis="x", labelsize=METHOD_LABEL_FONTSIZE)

    # ARI 固定 0 到 1
    ax.set_ylim(0, 1.0)

    # 收紧左右空白
    ax.set_xlim(
        x[0] - BAR_WIDTH / 2 - 0.04,
        x[-1] + BAR_WIDTH / 2 + 0.04
    )

    style_axis(ax)

    # 柱子顶部数值
    for bar, val in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=VALUE_LABEL_FONTSIZE,
            fontweight="normal",
            color="black",
            clip_on=False
        )

    plt.tight_layout()

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
# 6. 主程序
# =========================================================
if __name__ == "__main__":
    results_df = load_ablation_results(detail_csv)

    print("用于绘图的数据：")
    print(results_df[["Variant", "ARI"]].to_string(index=False))

    plot_ari_barplot(
        results_df=results_df,
        save_png=out_png,
        save_pdf=out_pdf
    )