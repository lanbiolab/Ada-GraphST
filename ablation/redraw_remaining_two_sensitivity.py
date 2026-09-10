import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle

# =========================================================
# 1. 基础配置
# =========================================================
PROJECT_ROOT = "/data2/liangyefeng/My_GraphST_Innovation"

BASE_DIR = os.path.join(
    PROJECT_ROOT,
    "Result_ParameterSensitivity_OnlyWS_WSH"
)

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"

# =========================================================
# 2. 需要重新画的两个数据集
# =========================================================
DATASET_CONFIGS = [
    {
        "dataset_name": "Human Breast Cancer",
        "folder_name": "Human_Breast_Cancer",
        "center_ws": 1.40,
        "center_wsh": 0.90,
        "csv_name": "Human_Breast_Cancer_parameter_sensitivity.csv",
        "heatmap_name": "Human_Breast_Cancer_heatmap",
        "lineplot_name": "Human_Breast_Cancer_lineplot",
    },
    {
        "dataset_name": "Mouse Breast Cancer",
        "folder_name": "Mouse_Breast_Cancer",
        "center_ws": 1.70,
        "center_wsh": 0.70,
        "csv_name": "Mouse_Breast_Cancer_parameter_sensitivity.csv",
        "heatmap_name": "Mouse_Breast_Cancer_heatmap",
        "lineplot_name": "Mouse_Breast_Cancer_lineplot",
    },
]


# =========================================================
# 3. Heatmap：数字统一黑色、加粗、变大
# =========================================================
def plot_parameter_sensitivity_heatmap(
    df,
    dataset_name,
    center_ws,
    center_wsh,
    save_png,
    save_pdf,
):
    sub = df[df["iLISI"].notna()].copy()
    if sub.empty:
        print(f"⚠️ {dataset_name}: 没有可用参数敏感性结果，跳过 heatmap。")
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

    ax.set_title(
        f"{dataset_name}\nParameter Sensitivity Heatmap",
        fontsize=13,
        fontweight="bold"
    )
    ax.set_xlabel("w_smooth", fontsize=11)
    ax.set_ylabel("w_sharpen", fontsize=11)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(
        [f"{x:.2f}" for x in pivot.columns],
        rotation=30,
        ha="right"
    )

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{y:.2f}" for y in pivot.index])

    for i, row_val in enumerate(pivot.index):
        for j, col_val in enumerate(pivot.columns):
            val = float(pivot.loc[row_val, col_val])

            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="black",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.65,
                    pad=1.8
                )
            )

            # 中心参数黑框
            if np.isclose(float(col_val), center_ws) and np.isclose(float(row_val), center_wsh):
                rect_center = Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="black",
                    linewidth=2.6
                )
                ax.add_patch(rect_center)

    # 最优参数红色虚线框
    rect_best = Rectangle(
        (best_j - 0.5, best_i - 0.5),
        1,
        1,
        fill=False,
        edgecolor="red",
        linewidth=2.4,
        linestyle="--"
    )
    ax.add_patch(rect_best)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("iLISI")

    plt.tight_layout()
    plt.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.savefig(save_pdf, bbox_inches="tight")
    plt.close()


# =========================================================
# 4. Lineplot：两个折线图数字统一黑色、加粗、变大
# =========================================================
def plot_parameter_sensitivity_lines(
    sens_df,
    dataset_name,
    center_ws,
    center_wsh,
    save_png,
    save_pdf,
):
    sub = sens_df[sens_df["iLISI"].notna()].copy()
    if sub.empty:
        print(f"⚠️ {dataset_name}: 没有可用参数敏感性结果，跳过 line plot。")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    fig.patch.set_facecolor("white")
    ax1, ax2 = axes

    ax1.set_facecolor("#F5F5F5")
    ax2.set_facecolor("#F5F5F5")

    # =====================================================
    # 左图：Sensitivity to w_smooth
    # =====================================================
    smooth_sub = sub[np.isclose(sub["w_sharpen"].astype(float), center_wsh)].copy()
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
            ax1.text(
                x,
                y + offset1,
                f"{y:.3f}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                color="black",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.65,
                    pad=1.5
                )
            )

        ax1.set_ylim(
            ymin1 - max(0.02, (ymax1 - ymin1) * 0.15 if ymax1 > ymin1 else 0.02),
            ymax1 + max(0.04, (ymax1 - ymin1) * 0.25 if ymax1 > ymin1 else 0.04)
        )

    ax1.set_title("Sensitivity to w_smooth", fontsize=13, fontweight="bold")
    ax1.set_xlabel(f"w_smooth  (fixed w_sharpen={center_wsh:.2f})", fontsize=11)
    ax1.set_ylabel("iLISI", fontsize=11)

    if not smooth_sub.empty:
        ax1.set_xticks(sorted(smooth_sub["w_smooth"].astype(float).unique()))

    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    # =====================================================
    # 右图：Sensitivity to w_sharpen
    # =====================================================
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

        center_row = sharpen_sub[np.isclose(sharpen_sub["w_sharpen"].astype(float), center_wsh)]
        if not center_row.empty:
            ax2.scatter(
                center_wsh,
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
            ax2.text(
                x,
                y + offset2,
                f"{y:.3f}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                color="black",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.65,
                    pad=1.5
                )
            )

        ax2.set_ylim(
            ymin2 - max(0.02, (ymax2 - ymin2) * 0.15 if ymax2 > ymin2 else 0.02),
            ymax2 + max(0.04, (ymax2 - ymin2) * 0.25 if ymax2 > ymin2 else 0.04)
        )

    ax2.set_title("Sensitivity to w_sharpen", fontsize=13, fontweight="bold")
    ax2.set_xlabel(f"w_sharpen  (fixed w_smooth={center_ws:.2f})", fontsize=11)
    ax2.set_ylabel("iLISI", fontsize=11)

    if not sharpen_sub.empty:
        ax2.set_xticks(sorted(sharpen_sub["w_sharpen"].astype(float).unique()))

    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle(dataset_name, fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.savefig(save_pdf, bbox_inches="tight")
    plt.close()


# =========================================================
# 5. 主程序：读取两个 CSV，重新生成图片
# =========================================================
if __name__ == "__main__":
    for cfg in DATASET_CONFIGS:
        dataset_name = cfg["dataset_name"]
        folder_name = cfg["folder_name"]
        center_ws = cfg["center_ws"]
        center_wsh = cfg["center_wsh"]

        out_root = os.path.join(BASE_DIR, folder_name)
        summary_csv = os.path.join(out_root, cfg["csv_name"])

        if not os.path.exists(summary_csv):
            raise FileNotFoundError(
                f"找不到 {dataset_name} 的结果 CSV，请确认之前已经跑过参数敏感度实验：\n{summary_csv}"
            )

        df = pd.read_csv(summary_csv)

        # 覆盖原图路径
        heatmap_png = os.path.join(out_root, cfg["heatmap_name"] + ".png")
        heatmap_pdf = os.path.join(out_root, cfg["heatmap_name"] + ".pdf")
        line_png = os.path.join(out_root, cfg["lineplot_name"] + ".png")
        line_pdf = os.path.join(out_root, cfg["lineplot_name"] + ".pdf")

        # 额外保存 black_text 备份
        heatmap_black_png = os.path.join(out_root, cfg["heatmap_name"] + "_black_text.png")
        heatmap_black_pdf = os.path.join(out_root, cfg["heatmap_name"] + "_black_text.pdf")
        line_black_png = os.path.join(out_root, cfg["lineplot_name"] + "_black_text.png")
        line_black_pdf = os.path.join(out_root, cfg["lineplot_name"] + "_black_text.pdf")

        print("=" * 100)
        print(f"正在重新绘制：{dataset_name}")
        print("CSV:", summary_csv)

        # 先覆盖原始图
        plot_parameter_sensitivity_heatmap(
            df,
            dataset_name,
            center_ws,
            center_wsh,
            heatmap_png,
            heatmap_pdf,
        )

        plot_parameter_sensitivity_lines(
            df,
            dataset_name,
            center_ws,
            center_wsh,
            line_png,
            line_pdf,
        )

        # 再保存 black_text 备份图
        plot_parameter_sensitivity_heatmap(
            df,
            dataset_name,
            center_ws,
            center_wsh,
            heatmap_black_png,
            heatmap_black_pdf,
        )

        plot_parameter_sensitivity_lines(
            df,
            dataset_name,
            center_ws,
            center_wsh,
            line_black_png,
            line_black_pdf,
        )

        print(f"✅ {dataset_name} 重新绘图完成")
        print("覆盖版 Heatmap PNG:", heatmap_png)
        print("覆盖版 Line PNG:", line_png)
        print("备份版 Heatmap PNG:", heatmap_black_png)
        print("备份版 Line PNG:", line_black_png)

    print("\n全部完成。")