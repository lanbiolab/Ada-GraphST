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

output_root = os.path.join(
    project_root,
    "Result_Ablation_DLPFC_151508_151670_151675"
)

detail_csv = os.path.join(
    output_root,
    "ablation_3slices_detail_results.csv"
)

fig_out_root = os.path.join(
    output_root,
    "figures_replot_from_csv"
)

os.makedirs(fig_out_root, exist_ok=True)

save_pdf = os.path.join(
    fig_out_root,
    "ablation_per_slice_ARI_subplots.pdf"
)


# =========================================================
# 2. 基本配置
# =========================================================
dataset_list = ["151508", "151670", "151675"]

VARIANT_ORDER = [
    "Baseline",
    "Smooth only",
    "Sharpen only",
    "Ada-GraphST"
]

# 经典红蓝绿橙风格
METHOD_COLORS = {
    "Baseline": "#d62728",      # 红
    "Smooth only": "#2ca02c",   # 绿
    "Sharpen only": "#ff7f0e",  # 橙
    "Ada-GraphST": "#1f77b4",   # 蓝
}


# =========================================================
# 3. 字体和柱状图样式
# =========================================================
TITLE_FONTSIZE = 14
METHOD_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 12
AXIS_LABEL_FONTSIZE = 15
VALUE_LABEL_FONTSIZE = 12

# 柱体宽度：保持较窄
BAR_WIDTH = 0.25

# 柱子间距压缩系数：
# 越小柱子越紧密；0.48 会比 0.62 更紧密
BAR_GAP_SCALE = 0.35

AXIS_LINEWIDTH = 1.0
TICK_LENGTH = 4
TICK_WIDTH = 1.0


# =========================================================
# 4. 工具函数
# =========================================================
def prepare_result_df(df):
    df = df.copy()

    required_cols = ["dataset", "variant", "ARI"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV 缺少必要列: {missing_cols}\n"
            f"当前列名: {list(df.columns)}"
        )

    if "status" in df.columns:
        df = df[df["status"].astype(str) == "success"].copy()

    df["dataset"] = df["dataset"].astype(str)
    df["variant"] = df["variant"].astype(str)
    df["ARI"] = pd.to_numeric(df["ARI"], errors="coerce")

    df = df.dropna(subset=["ARI"]).copy()

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


def style_axis(ax):
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
# 5. 只画 ARI 消融柱状图
# =========================================================
def plot_ablation_ari_barplot(df, save_pdf):
    fig, axes = plt.subplots(
        1,
        len(dataset_list),
        figsize=(5.0 * len(dataset_list), 5.2),
        dpi=300
    )

    fig.patch.set_facecolor("white")

    if len(dataset_list) == 1:
        axes = [axes]

    for idx, (ax, dataset) in enumerate(zip(axes, dataset_list), start=1):
        sub = df[df["dataset"].astype(str) == dataset].copy()

        sub["variant"] = pd.Categorical(
            sub["variant"],
            categories=VARIANT_ORDER,
            ordered=True
        )
        sub = sub.sort_values("variant").reset_index(drop=True)

        if sub.empty:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                fontsize=VALUE_LABEL_FONTSIZE
            )
            ax.set_title(
                f"({idx}) DLPFC {dataset}",
                fontsize=TITLE_FONTSIZE,
                fontweight="normal",
                pad=8
            )
            ax.axis("off")
            continue

        methods = sub["variant"].astype(str).tolist()
        scores = sub["ARI"].astype(float).tolist()
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
            "ARI",
            fontsize=AXIS_LABEL_FONTSIZE,
            fontweight="normal"
        )

        ax.tick_params(axis="x", labelsize=METHOD_LABEL_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)

        # y 轴固定到 1
        ax.set_ylim(0, 1.0)

        # 收紧左右空白
        ax.set_xlim(
            x[0] - BAR_WIDTH / 2 - 0.06,
            x[-1] + BAR_WIDTH / 2 + 0.06
        )

        style_axis(ax)

        # 柱子顶部数值
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=VALUE_LABEL_FONTSIZE,
                fontweight="normal",
                color="black",
                clip_on=False
            )

    fig.suptitle(
        "ARI of Ablation Variants Across 3 DLPFC Slices",
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
        y=1.02
    )

    plt.tight_layout(w_pad=1.6)

    plt.savefig(
        save_pdf,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    print(f"[Saved] {save_pdf}")


# =========================================================
# 6. 主程序
# =========================================================
if __name__ == "__main__":
    if not os.path.exists(detail_csv):
        raise FileNotFoundError(
            f"没有找到 CSV 文件:\n{detail_csv}\n\n"
            f"请先确认前面的消融实验代码已经生成 ablation_3slices_detail_results.csv。"
        )

    detail_df = pd.read_csv(detail_csv)
    ok_df = prepare_result_df(detail_df)

    if ok_df.empty:
        raise ValueError("CSV 中没有可用于画图的成功 ARI 结果。")

    print("用于画图的数据：")
    print(
        ok_df[
            ["dataset", "variant", "ARI"]
        ].to_string(index=False)
    )

    plot_ablation_ari_barplot(
        df=ok_df,
        save_pdf=save_pdf
    )