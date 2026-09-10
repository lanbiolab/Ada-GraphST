import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle

# =========================================================
# A. 配置
# =========================================================
PROJECT_ROOT = "/data2/liangyefeng/My_GraphST_Innovation"

DATASET_NAME = "Mouse Brain"
CENTER_WS = 1.10
CENTER_WSH = 0.90

OUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "Result_ParameterSensitivity_OnlyWS_WSH",
    DATASET_NAME.replace(" ", "_")
)

SUMMARY_CSV = os.path.join(OUT_ROOT, "Mouse_Brain_parameter_sensitivity.csv")

HEATMAP_PNG = os.path.join(OUT_ROOT, "Mouse_Brain_heatmap_black_text.png")
HEATMAP_PDF = os.path.join(OUT_ROOT, "Mouse_Brain_heatmap_black_text.pdf")

LINE_PNG = os.path.join(OUT_ROOT, "Mouse_Brain_lineplot_black_text.png")
LINE_PDF = os.path.join(OUT_ROOT, "Mouse_Brain_lineplot_black_text.pdf")

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"


# =========================================================
# B. Heatmap
# =========================================================
def plot_parameter_sensitivity_heatmap(df, save_png, save_pdf):
    sub = df[df["iLISI"].notna()].copy()
    if sub.empty:
        print("没有可用参数敏感性结果，跳过 heatmap。")
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
        f"{DATASET_NAME}\nParameter Sensitivity Heatmap",
        fontsize=13,
        fontweight="bold"
    )
    ax.set_xlabel("w_smooth", fontsize=11)
    ax.set_ylabel("w_sharpen", fontsize=11)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns], rotation=30, ha="right")

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{y:.2f}" for y in pivot.index])

    for i, row_val in enumerate(pivot.index):
        for j, col_val in enumerate(pivot.columns):
            val = float(pivot.loc[row_val, col_val])

            # 数字统一改成黑色、加粗、变大，并加浅白底
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

            # 中心参数框
            if np.isclose(float(col_val), CENTER_WS) and np.isclose(float(row_val), CENTER_WSH):
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
# C. Line plot
# =========================================================
def plot_parameter_sensitivity_lines(sens_df, save_png, save_pdf):
    sub = sens_df[sens_df["iLISI"].notna()].copy()
    if sub.empty:
        print("没有可用参数敏感性结果，跳过 line plot。")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    fig.patch.set_facecolor("white")
    ax1, ax2 = axes

    ax1.set_facecolor("#F5F5F5")
    ax2.set_facecolor("#F5F5F5")

    center_ws = CENTER_WS
    center_wsp = CENTER_WSH

    # -------------------------
    # w_smooth 敏感度
    # -------------------------
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
    ax1.set_xlabel(f"w_smooth  (fixed w_sharpen={center_wsp:.2f})", fontsize=11)
    ax1.set_ylabel("iLISI", fontsize=11)
    ax1.set_xticks(sorted(smooth_sub["w_smooth"].astype(float).unique()))
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    # -------------------------
    # w_sharpen 敏感度
    # -------------------------
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
    ax2.set_xticks(sorted(sharpen_sub["w_sharpen"].astype(float).unique()))
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle(DATASET_NAME, fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.savefig(save_pdf, bbox_inches="tight")
    plt.close()


# =========================================================
# D. 主程序：只读 CSV，重新画图
# =========================================================
if __name__ == "__main__":
    if not os.path.exists(SUMMARY_CSV):
        raise FileNotFoundError(f"找不到结果 CSV，请确认之前已经跑过参数敏感度实验：\n{SUMMARY_CSV}")

    df = pd.read_csv(SUMMARY_CSV)

    plot_parameter_sensitivity_heatmap(df, HEATMAP_PNG, HEATMAP_PDF)
    plot_parameter_sensitivity_lines(df, LINE_PNG, LINE_PDF)

    print("重新绘图完成：")
    print("Heatmap PNG:", HEATMAP_PNG)
    print("Heatmap PDF:", HEATMAP_PDF)
    print("Line PNG:", LINE_PNG)
    print("Line PDF:", LINE_PDF)