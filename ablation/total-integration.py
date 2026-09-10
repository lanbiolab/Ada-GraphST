import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageChops

# =========================================================
# 1. 路径配置
# =========================================================
PROJECT_ROOT = "/data2/liangyefeng/My_GraphST_Innovation"

ABLATION_DIR = os.path.join(PROJECT_ROOT, "Result_Ablation_AdaGraphST_3Datasets")
SENS_DIR = os.path.join(PROJECT_ROOT, "Result_ParameterSensitivity_OnlyWS_WSH")

OUT_DIR = os.path.join(PROJECT_ROOT, "Result_Final_Composite_Figure")
os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# 2. 你的已有图路径
#    先确保这些 png 已经被前面的脚本生成出来
# =========================================================
# ---- Ablation 四张图（竖着排）----
ABLATION_RAW = os.path.join(ABLATION_DIR, "ablation_raw_ilisi_barplot.png")
ABLATION_DELTA = os.path.join(ABLATION_DIR, "ablation_delta_ilisi_barplot.png")
ABLATION_TREND = os.path.join(ABLATION_DIR, "ablation_trend_lineplot.png")
ABLATION_GAIN = os.path.join(ABLATION_DIR, "adagraphst_gain_summary.png")

# ---- 参数敏感性图：每个数据集 2 张图（line + heatmap）----
HBC_LINE = os.path.join(SENS_DIR, "Human_Breast_Cancer", "Human_Breast_Cancer_lineplot.png")
HBC_HEAT = os.path.join(SENS_DIR, "Human_Breast_Cancer", "Human_Breast_Cancer_heatmap.png")

MB_LINE = os.path.join(SENS_DIR, "Mouse_Brain", "Mouse_Brain_lineplot.png")
MB_HEAT = os.path.join(SENS_DIR, "Mouse_Brain", "Mouse_Brain_heatmap.png")

MBC_LINE = os.path.join(SENS_DIR, "Mouse_Breast_Cancer", "Mouse_Breast_Cancer_lineplot.png")
MBC_HEAT = os.path.join(SENS_DIR, "Mouse_Breast_Cancer", "Mouse_Breast_Cancer_heatmap.png")

# =========================================================
# 3. 检查文件是否存在
# =========================================================
all_paths = [
    ABLATION_RAW, ABLATION_DELTA, ABLATION_TREND, ABLATION_GAIN,
    HBC_LINE, HBC_HEAT,
    MB_LINE, MB_HEAT,
    MBC_LINE, MBC_HEAT
]

for p in all_paths:
    if not os.path.exists(p):
        raise FileNotFoundError(f"找不到图片，请先生成：\n{p}")

# =========================================================
# 4. 工具函数：裁白边
# =========================================================
def crop_white_border(pil_img, bg_color=(255, 255, 255), tol=8):
    """
    自动裁掉图片四周的白边，让总图更紧凑、更像论文排版
    """
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    bg = Image.new("RGB", pil_img.size, bg_color)
    diff = ImageChops.difference(pil_img, bg)
    diff = diff.point(lambda x: 255 if x > tol else 0)
    bbox = diff.getbbox()
    if bbox is None:
        return pil_img
    return pil_img.crop(bbox)

def load_image(path, crop=True):
    img = Image.open(path)
    if crop:
        img = crop_white_border(img)
    return np.asarray(img)

# =========================================================
# 5. 工具函数：画一张 panel
# =========================================================
def show_img(ax, img_path, letter=None, title=None, crop=True, letter_size=18, title_size=13):
    img = load_image(img_path, crop=crop)
    ax.imshow(img)
    ax.axis("off")

    if letter is not None:
        ax.text(
            -0.02, 1.02, letter,
            transform=ax.transAxes,
            fontsize=letter_size,
            fontweight="bold",
            va="bottom",
            ha="right"
        )

    if title is not None:
        ax.set_title(title, fontsize=title_size, pad=6, fontweight="bold")

# =========================================================
# 6. 创建总图（竖向论文排版）
# =========================================================
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

fig = plt.figure(figsize=(16, 28), facecolor="white")

# 外层：7行
# 1-4 行：ablation 四张图
# 5-7 行：三个数据集的参数敏感性
outer = GridSpec(
    nrows=7,
    ncols=1,
    figure=fig,
    height_ratios=[1.1, 1.1, 1.0, 0.9, 1.25, 1.25, 1.25],
    hspace=0.18
)

# =========================================================
# 7. Ablation 部分（竖着放 4 张）
# =========================================================
ax1 = fig.add_subplot(outer[0, 0])
show_img(ax1, ABLATION_RAW, crop=True)

ax2 = fig.add_subplot(outer[1, 0])
show_img(ax2, ABLATION_DELTA, crop=True)

ax3 = fig.add_subplot(outer[2, 0])
show_img(ax3, ABLATION_TREND, crop=True)

ax4 = fig.add_subplot(outer[3, 0])
show_img(ax4, ABLATION_GAIN, crop=True)

# =========================================================
# 8. 参数敏感性部分
#    每个数据集占一行：左边 lineplot，右边 heatmap
# =========================================================
def add_dataset_row(gs_slot, line_path, heat_path, row_letter):
    sub = gs_slot.subgridspec(
        nrows=1,
        ncols=2,
        width_ratios=[2.25, 1.0],   # 左边 lineplot 更宽，右边 heatmap 稍窄
        wspace=0.05
    )

    ax_left = fig.add_subplot(sub[0, 0])
    ax_right = fig.add_subplot(sub[0, 1])

    show_img(ax_left, line_path, letter=row_letter, crop=True)
    show_img(ax_right, heat_path, crop=True)

# 第5行：Human Breast Cancer
add_dataset_row(outer[4, 0], HBC_LINE, HBC_HEAT, "e")

# 第6行：Mouse Brain
add_dataset_row(outer[5, 0], MB_LINE, MB_HEAT, "f")

# 第7行：Mouse Breast Cancer
add_dataset_row(outer[6, 0], MBC_LINE, MBC_HEAT, "g")

# =========================================================
# 9. 总标题（可选）
# =========================================================
fig.suptitle(
    "Comprehensive Summary of Ada-GraphST Results",
    fontsize=20,
    fontweight="bold",
    y=0.995
)

# =========================================================
# 10. 保存
# =========================================================
save_png = os.path.join(OUT_DIR, "AdaGraphST_vertical_composite_figure.png")
save_pdf = os.path.join(OUT_DIR, "AdaGraphST_vertical_composite_figure.pdf")

plt.savefig(save_png, dpi=600, bbox_inches="tight", facecolor="white")
plt.savefig(save_pdf, bbox_inches="tight", facecolor="white")
plt.close()

print("✅ 竖向总图已保存：")
print("PNG:", save_png)
print("PDF:", save_pdf)