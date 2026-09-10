import torch
from .preprocess import (
    preprocess_adj,
    preprocess_adj_sparse,
    preprocess,
    construct_interaction,
    construct_interaction_KNN,
    add_contrastive_label,
    get_feature,
    permutation,
    fix_seed
)
import numpy as np
from .model import Encoder, Encoder_sparse, Encoder_map, Encoder_sc, BoundaryAwareGraphRefiner
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import pandas as pd


class GraphST():
    def __init__(self,
                 adata,
                 device=torch.device('cpu'),
                 random_seed=2020,
                 epochs=600,
                 dim_input=3000,
                 dim_output=64,
                 datatype='10X',

                 # ===== 新增 / 保留的关键超参数 =====
                 w_smooth=0.0,              # 特征平滑强度
                 w_sharpen=0.0,             # 动态图融合强度
                 warmup_epochs=200,         # warm-up 轮数
                 update_interval=20,        # 图更新间隔
                 graph_update_rate=0.3,     # EMA式图更新率
                 gamma=3.0,                 # 锐化指数
                 graph_reg_weight=0.0,      # 图正则损失权重
                 use_learnable_proj=True    # graph refiner 是否使用可学习投影
                 ):

        self.adata = adata.copy()
        self.device = device
        self.epochs = epochs
        self.random_seed = random_seed
        self.datatype = datatype

        self.w_smooth = w_smooth
        self.w_sharpen = w_sharpen
        self.warmup_epochs = warmup_epochs
        self.update_interval = update_interval
        self.graph_update_rate = graph_update_rate
        self.gamma = gamma
        self.graph_reg_weight = graph_reg_weight
        self.use_learnable_proj = use_learnable_proj

        # 原始 GraphST 超参数
        self.learning_rate = 0.001
        self.weight_decay = 0.00
        self.alpha = 10
        self.beta = 1
        self.lamda1 = 10
        self.lamda2 = 1

        fix_seed(self.random_seed)

        # =========================
        # 1. 数据预处理
        # =========================
        if 'highly_variable' not in adata.var.keys():
            preprocess(self.adata)

        if 'adj' not in adata.obsm.keys():
            construct_interaction(self.adata)

        if 'label_CSL' not in adata.obsm.keys():
            add_contrastive_label(self.adata)

        if 'feat' not in adata.obsm.keys():
            get_feature(self.adata)

        # 特征
        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
        self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device)

        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output

        # =========================
        # 2. 图结构拆分：mask图 + 传播图
        # =========================
        raw_adj = self.adata.obsm['adj'].copy()   # 原始物理邻接图

        # (1) 二值物理 mask：只负责限制“允许哪些边被动态调整”
        adj_mask = (raw_adj > 0).astype(np.float32)

        # (2) 传播图：归一化后给 GNN 使用
        adj_phy = preprocess_adj(raw_adj)

        self.adj_mask = torch.FloatTensor(adj_mask).to(self.device)
        self.adj_phy = torch.FloatTensor(adj_phy).to(self.device)

        # 当前训练中使用的动态图，初始为物理图
        self.adj_current = self.adj_phy.clone()

        # graph_neigh 保持原始 GraphST 逻辑
        self.graph_neigh = torch.FloatTensor(
            self.adata.obsm['graph_neigh'].copy() + np.eye(raw_adj.shape[0])
        ).to(self.device)

    def train(self):
        # =========================
        # 3. 初始化主模型
        # =========================
        self.model = Encoder(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)

        # 只有开启动态图时才初始化 refiner
        if self.w_sharpen > 0:
            self.graph_refiner = BoundaryAwareGraphRefiner(
                adj_mask=self.adj_mask,
                dim_hidden=self.dim_output,
                gamma=self.gamma,
                use_proj=self.use_learnable_proj
            ).to(self.device)

        # =========================
        # 4. 损失与优化器
        # =========================
        self.loss_CSL = nn.BCEWithLogitsLoss()

        params = list(self.model.parameters())
        if self.w_sharpen > 0:
            params += list(self.graph_refiner.parameters())

        self.optimizer = torch.optim.Adam(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        print("=" * 70)
        print("🔥 GraphST Running with Dynamic Boundary-Aware Graph Refinement")
        print(f"Smooth={self.w_smooth} | Sharpen={self.w_sharpen} | "
              f"Warmup={self.warmup_epochs} | Interval={self.update_interval} | "
              f"Gamma={self.gamma} | EMA={self.graph_update_rate}")
        print("=" * 70)

        self.model.train()

        # =========================
        # 5. 特征平滑（只基于物理图）
        # =========================
        if self.w_smooth > 0:
            print("   -> Applying Feature Smoothing on Physical Graph...")
            self.features = (
                (1 - self.w_smooth) * self.features
                + self.w_smooth * torch.matmul(self.adj_phy, self.features)
            )

        # =========================
        # 6. 训练循环
        # =========================
        for epoch in tqdm(range(self.epochs)):
            self.model.train()

            # ---- warm-up 阶段：只使用物理图 ----
            if epoch < self.warmup_epochs:
                adj_for_encoder = self.adj_phy
            else:
                adj_for_encoder = self.adj_current

            # 数据增强
            self.features_a = permutation(self.features)

            # 前向传播
            self.hiden_feat, self.emb, ret, ret_a = self.model(
                self.features,
                self.features_a,
                adj_for_encoder
            )

            # 原始 GraphST 损失
            self.loss_sl_1 = self.loss_CSL(ret, self.label_CSL)
            self.loss_sl_2 = self.loss_CSL(ret_a, self.label_CSL)
            self.loss_feat = F.mse_loss(self.features, self.emb)

            loss = self.alpha * self.loss_feat + self.beta * (self.loss_sl_1 + self.loss_sl_2)

            # =========================
            # 7. 动态图更新（warm-up后，按间隔更新）
            # =========================
            if (
                self.w_sharpen > 0
                and epoch >= self.warmup_epochs
                and epoch % self.update_interval == 0
            ):
                with torch.no_grad():
                    # 基于当前 hidden embedding 生成动态图
                    adj_dyn = self.graph_refiner(self.hiden_feat)

                    # 动态图与物理图融合
                    adj_fused = (
                        (1 - self.w_sharpen) * self.adj_phy
                        + self.w_sharpen * adj_dyn
                    )

                    # EMA 式平滑更新，避免图突然跳变
                    self.adj_current = (
                        (1 - self.graph_update_rate) * self.adj_current
                        + self.graph_update_rate * adj_fused
                    )

                    # 再归一化一次，增强数值稳定性
                    row_sum = torch.sum(self.adj_current, dim=1, keepdim=True) + 1e-12
                    self.adj_current = self.adj_current / row_sum

            # =========================
            # 8. 可选图正则
            # =========================
            if (
                self.w_sharpen > 0
                and self.graph_reg_weight > 0
                and epoch >= self.warmup_epochs
            ):
                with torch.no_grad():
                    adj_dyn_for_reg = self.graph_refiner(self.hiden_feat)
                graph_reg = F.mse_loss(adj_dyn_for_reg, self.adj_phy)
                loss = loss + self.graph_reg_weight * graph_reg

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # =========================
        # =========================
        # 9. 输出 embedding，并保存机制分析所需图结构
        # =========================
        with torch.no_grad():
            self.model.eval()

            hidden_final, emb_rec, _, _ = self.model(
                self.features,
                self.features_a,
                self.adj_current
            )

            # -------------------------------------------------
            # 1. GraphST 原本用于后续 clustering 的 embedding
            # 注意：emb_rec 是 decoder 重构空间，不是动态图 refiner 的输入
            # -------------------------------------------------
            self.emb_rec = emb_rec.detach().cpu().numpy()
            self.adata.obsm["emb"] = self.emb_rec

            # -------------------------------------------------
            # 2. 保存真正用于动态图锐化的 hidden feature
            # 训练中 BoundaryAwareGraphRefiner 使用的是 self.hiden_feat
            # 这里用最终模型重新计算 hidden_final
            # -------------------------------------------------
            self.adata.obsm["hidden_refine"] = hidden_final.detach().cpu().numpy()

            # -------------------------------------------------
            # 3. 保存物理 mask 和物理传播图
            # A_mask: 二值物理邻接，只表示哪些边允许被动态调整
            # A_phy: 归一化后的物理传播图，用于 GNN message passing
            # -------------------------------------------------
            A_mask = self.adj_mask.detach().cpu()
            A_phy = self.adj_phy.detach().cpu()

            # -------------------------------------------------
            # 4. 保存动态图相关矩阵
            # A_gate_raw: ReLU(cos)^gamma，未行归一化，最适合机制图
            # A_dyn: row-normalized dynamic graph
            # A_fused: (1-w_sharpen)A_phy + w_sharpen A_dyn
            # A_current: 训练结束时实际用于 encoder 的 EMA 图
            # -------------------------------------------------
            if self.w_sharpen > 0:
                z = hidden_final

                if self.graph_refiner.use_proj:
                    z = self.graph_refiner.proj(z)

                z = F.normalize(z, p=2, dim=1)
                sim = torch.mm(z, z.t())

                # raw semantic gate: only physical-neighbor edges are valid
                A_gate_raw = torch.pow(F.relu(sim * self.adj_mask), self.gamma)

                row_sum = torch.sum(A_gate_raw, dim=1, keepdim=True) + 1e-12
                A_dyn = A_gate_raw / row_sum

                A_fused = (
                    (1 - self.w_sharpen) * self.adj_phy
                    + self.w_sharpen * A_dyn
                )
                A_fused = A_fused / (torch.sum(A_fused, dim=1, keepdim=True) + 1e-12)

            else:
                # Baseline / Smooth only 没有真实动态图
                # 为保持 h5ad key 一致，这里用物理图占位
                A_gate_raw = self.adj_mask.clone()
                A_dyn = self.adj_phy.clone()
                A_fused = self.adj_phy.clone()

            A_current = self.adj_current.detach().cpu()

            # -------------------------------------------------
            # 5. 保存到 obsp
            # DLPFC spot 数量约 3k-4k，保存这些矩阵没有问题。
            # 大规模数据如 10 万 spot 时，不建议保存 dense graph。
            # -------------------------------------------------
            self.adata.obsp["A_mask"] = csr_matrix(A_mask.numpy())
            self.adata.obsp["A_phy"] = csr_matrix(A_phy.numpy())
            self.adata.obsp["A_gate_raw"] = csr_matrix(A_gate_raw.detach().cpu().numpy())
            self.adata.obsp["A_dyn"] = csr_matrix(A_dyn.detach().cpu().numpy())
            self.adata.obsp["A_fused"] = csr_matrix(A_fused.detach().cpu().numpy())
            self.adata.obsp["A_current"] = csr_matrix(A_current.numpy())

            self.adata.uns["graph_refinement"] = {
                "w_smooth": float(self.w_smooth),
                "w_sharpen": float(self.w_sharpen),
                "warmup_epochs": int(self.warmup_epochs),
                "update_interval": int(self.update_interval),
                "graph_update_rate": float(self.graph_update_rate),
                "gamma": float(self.gamma),
                "graph_reg_weight": float(self.graph_reg_weight),
                "use_learnable_proj": bool(self.use_learnable_proj),
                "has_dynamic_graph": bool(self.w_sharpen > 0),
                "A_mask": "binary physical adjacency mask",
                "A_phy": "row-normalized physical graph used by GraphST",
                "A_gate_raw": "ReLU(cos(hidden_i, hidden_j) * physical_mask)^gamma, before row normalization",
                "A_dyn": "row-normalized dynamic semantic graph",
                "A_fused": "(1-w_sharpen) * A_phy + w_sharpen * A_dyn",
                "A_current": "final EMA-updated graph used by encoder",
                "mechanism_note": (
                    "For boundary-aware mechanism figures, use A_gate_raw or A_dyn. "
                    "Do not use A_fused/A_current as the main visualization object, "
                    "because w_sharpen is small and most physical edges are retained."
                ),
            }

            return self.adata
