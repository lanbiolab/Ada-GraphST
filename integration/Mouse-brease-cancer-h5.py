import os
import sys
import random
import numpy as np
import torch
import scanpy as sc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib

# =========================================================
# 0. 环境与路径配置
# =========================================================
os.environ['R_HOME'] = '/data2/liangyefeng/miniconda3/envs/graphst/lib/R'
project_root = "/data2/liangyefeng/My_GraphST_Innovation"
sys.path.insert(0, os.path.join(project_root, "src"))

from GraphST.GraphST import GraphST
import GraphST.utils as graphst_utils
from GraphST.utils import clustering

# 设置保存路径
save_dir = '/data2/liangyefeng/My_GraphST_Innovation/Result'
os.makedirs(save_dir, exist_ok=True)

# 设置数据输入路径
data_paths = {
    "Section_1": "/data2/liangyefeng/My_GraphST_Innovation/data/9.Mouse_Brain_Merge_Anterior_Posterior_Section_1/filtered_feature_bc_matrix.h5ad",
    "Section_2": "/data2/liangyefeng/My_GraphST_Innovation/data/10.Mouse_Brain_Merge_Anterior_Posterior_Section_2/filtered_feature_bc_matrix.h5ad"
}

# 字体设置
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Arial"


# =========================================================
# 1. 工具函数 (mclust 补丁)
# =========================================================
def patched_mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    import rpy2.robjects as robjects
    from rpy2.robjects import r, numpy2ri
    x = np.asarray(adata.obsm[used_obsm], dtype=np.float64)
    numpy2ri.activate()
    robjects.globalenv["x_mat"] = x
    robjects.globalenv["n_cluster"] = int(num_cluster)
    robjects.globalenv["model_name"] = modelNames
    robjects.globalenv["seed"] = int(random_seed)
    r("""
        suppressMessages(library(mclust))
        set.seed(seed)
        x_mat <- as.matrix(x_mat)
        dimnames(x_mat) <- NULL
        res <- Mclust(x_mat, G=n_cluster, modelNames=model_name)
        cls <- res$classification
    """)
    # 正确的代码：
    mclust_res = np.array(r["cls"]).astype(int)
    adata.obs["mclust"] = mclust_res
    adata.obs["mclust"] = adata.obs["mclust"].astype("category")
    return adata


graphst_utils.mclust_R = patched_mclust_R


# =========================================================
# 2. 自定义柔和配色方案
# =========================================================
def get_soft_palette(n_colors):
    """
    生成柔和、自然的配色方案
    基于 matplotlib 的 tab20 但调整饱和度和亮度使其更柔和
    """
    if n_colors <= 20:
        # 使用 tab20 的柔和版本
        base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
        # 调整饱和度：降低饱和度使其更柔和
        # 转换到 HSV 空间，降低饱和度，保持明度
        for i in range(len(base_colors)):
            r, g, b, a = base_colors[i]
            # 简单调整：混合白色使颜色更柔和
            softness = 0.65  # 柔和度参数，越小颜色越柔和
            base_colors[i] = (r * softness + (1 - softness),
                              g * softness + (1 - softness),
                              b * softness + (1 - softness), a)
        return base_colors[:n_colors]
    else:
        # 超过20个颜色时，使用自定义的柔和色轮
        colors = []
        for i in range(n_colors):
            # 使用更柔和的色相分布
            hue = (i * 0.618033988749895) % 1.0  # 黄金比例
            # 固定饱和度和明度在柔和范围
            saturation = 0.55 + 0.15 * np.sin(i * 0.5)  # 饱和度在0.4-0.7之间
            value = 0.75 + 0.15 * np.cos(i * 0.3)  # 明度在0.6-0.9之间
            # 从HSV转换到RGB
            import colorsys
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb)
        return colors


# 也可以使用现成的柔和调色板
def get_alternative_palette(n_colors):
    """
    备选柔和配色方案：使用 Scientific colour maps 风格
    """
    # 使用 seaborn 的柔和调色板（如果可用）
    try:
        import seaborn as sns
        return sns.color_palette("Set2", n_colors)
    except:
        # 自定义柔和颜色列表
        soft_colors = [
            '#A6CEE3', '#B2DF8A', '#FB9A99', '#FDBF6F', '#CAB2D6',
            '#FFFF99', '#B15928', '#FCCDE5', '#8DD3C7', '#FFED6F',
            '#B3DE69', '#FCCDE5', '#D9D9D9', '#BC80BD', '#CCEBC5',
            '#FFED6F', '#E5C494', '#FDB462', '#80B1D3', '#B3DE69'
        ]
        # 如果需要更多颜色，重复循环并调整
        while len(soft_colors) < n_colors:
            soft_colors.extend([plt.cm.Set3(i % 12) for i in range(len(soft_colors), n_colors)])
        return soft_colors[:n_colors]


# =========================================================
# 3. 核心执行逻辑
# =========================================================
if __name__ == "__main__":
    # 官方教程在整合时使用的随机种子为 50
    SEED = 50
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_clusters = 26  # 官方教程中水平整合固定聚成 26 类

    print(f"✅ 输出目录已设定为: {save_dir}")

    # 生成柔和配色方案
    soft_palette = get_soft_palette(n_clusters)
    # 或者使用备选方案
    # soft_palette = get_alternative_palette(n_clusters)

    # 循环处理 Section 1 和 Section 2
    for section_name, data_path in data_paths.items():
        print("\n" + "=" * 80)
        print(f"🧠 开始水平整合实验 ({section_name})")
        print("=" * 80)

        print(f"📥 正在加载数据: {data_path}")
        adata_raw = sc.read_h5ad(data_path)
        adata_raw.var_names_make_unique()

        # ------------------------------------------
        # 任务 A: 运行 Baseline (w_sharpen=0)
        # ------------------------------------------
        print("\n⏳ [1/2] 正在运行 Baseline GraphST...")
        adata_base = adata_raw.copy()
        model_base = GraphST(
            adata_base, device=device, random_seed=SEED, epochs=600, dim_output=64, datatype="10X",
            w_smooth=0.0, w_sharpen=0.0, use_learnable_proj=False
        )
        adata_base = model_base.train()
        adata_base.obsm['emb_pca'] = PCA(n_components=min(20, adata_base.obsm['emb'].shape[1]),
                                         random_state=SEED).fit_transform(adata_base.obsm['emb'])

        print("   -> 正在聚类 Baseline...")
        clustering(adata_base, n_clusters, method='mclust', refinement=False)

        # ------------------------------------------
        # 任务 B: 运行 Boundary-Aware Ours (w_sharpen=0.20, gamma=2.5)
        # ------------------------------------------
        print("\n⏳ [2/2] 正在运行 Boundary-Aware Ours...")
        adata_ours = adata_raw.copy()
        model_ours = GraphST(
            adata_ours, device=device, random_seed=SEED, epochs=600, dim_output=64, datatype="10X",
            w_smooth=1.7, w_sharpen=0.7, gamma=2.5, use_learnable_proj=False
        )
        adata_ours = model_ours.train()
        adata_ours.obsm['emb_pca'] = PCA(n_components=min(20, adata_ours.obsm['emb'].shape[1]),
                                         random_state=SEED).fit_transform(adata_ours.obsm['emb'])

        print("   -> 正在聚类 Ours...")
        clustering(adata_ours, n_clusters, method='mclust', refinement=False)

        # ------------------------------------------
        # 任务 C: 可视化绘图
        # ------------------------------------------
        print(f"\n🎨 正在渲染 {section_name} 水平拼接对比图...")
        adata_base.obsm['spatial'][:, 1] = -1 * adata_base.obsm['spatial'][:, 1]
        adata_ours.obsm['spatial'][:, 1] = -1 * adata_ours.obsm['spatial'][:, 1]

        # 【修改】：使用柔和配色方案
        adata_base.uns['domain_colors'] = soft_palette
        adata_ours.uns['domain_colors'] = soft_palette

        fig, axs = plt.subplots(1, 2, figsize=(20, 8))

        # 调整点的大小和样式
        sc.pl.embedding(adata_base, basis="spatial", color="domain", s=150,
                        title=f"GraphST Baseline ({section_name})", show=False, ax=axs[0],
                        edgecolor='none', linewidth=0)  # 去掉边框

        sc.pl.embedding(adata_ours, basis="spatial", color="domain", s=150,
                        title=f"Boundary-Aware Ours ({section_name})", show=False, ax=axs[1],
                        edgecolor='none', linewidth=0)  # 去掉边框

        plt.tight_layout()

        # 保存图片
        save_path_png = os.path.join(save_dir, f"Fig_Horizontal_Integration_{section_name}.png")
        save_path_pdf = os.path.join(save_dir, f"Fig_Horizontal_Integration_{section_name}.pdf")
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight', facecolor='white')

        print(f"✅ {section_name} 绘图完毕！已保存至: {save_path_png}")

        # ------------------------------------------
        # 任务 D: 保存 h5ad 文件
        # ------------------------------------------
        print(f"\n💾 正在保存 {section_name} 的 h5ad 文件...")

        # 设置各自的保存路径
        h5ad_base_path = os.path.join(save_dir, f"{section_name}_Baseline.h5ad")
        h5ad_ours_path = os.path.join(save_dir, f"{section_name}_Ours.h5ad")

        # 写入文件
        adata_base.write_h5ad(h5ad_base_path)
        adata_ours.write_h5ad(h5ad_ours_path)

        print(f"✅ {section_name} 数据保存成功！")
        print(f"   - Baseline: {h5ad_base_path}")
        print(f"   - Ours: {h5ad_ours_path}")

    print("\n🎉 所有水平整合实验已全部完成！请前往 Result 文件夹查看图片和 h5ad 结果！")