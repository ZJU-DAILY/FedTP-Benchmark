import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSAGELayer(nn.Module):
    """
    纯正的 GraphSAGE 聚合层 (对应论文 Eq. 2)
    将自身特征与邻居聚合特征进行 Concat，再做非线性映射
    """
    def __init__(self, in_dim, out_dim):
        super(GraphSAGELayer, self).__init__()
        self.proj = nn.Linear(in_dim * 2, out_dim)
        
    def forward(self, x, adj):
        # x: [B, N, C]
        # adj: [N, N]
        # 度归一化用于邻居均值聚合 (Mean Aggregation)
        deg = adj.sum(dim=-1, keepdim=True) + 1e-6
        adj_norm = adj / deg
        
        # 聚合邻居特征
        neigh_x = torch.matmul(adj_norm, x) # [B, N, C]
        
        # 拼接自身与邻居 (Concat)
        cat_x = torch.cat([x, neigh_x], dim=-1) # [B, N, 2C]
        out = F.relu(self.proj(cat_x))
        return out

class FGNNEH_Client(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim, hidden_dim, backbone_extractor, epsilon=1e-5):
        super(FGNNEH_Client, self).__init__()
        self.num_nodes = num_nodes
        self.backbone_extractor = backbone_extractor
        self.epsilon = epsilon
        
        # 1. 动态图节点嵌入
        self.embedding = nn.Embedding(num_nodes, hidden_dim)
        
        # 2. 本地特征提取 (GraphSAGE 替换原本的单一 Linear)
        self.lin_in = nn.Linear(in_dim, hidden_dim) # 先把输入投影到 hidden_dim
        self.sage_layer = GraphSAGELayer(hidden_dim, hidden_dim)
        
        # 3. 骨干 GCN 聚合权重 (对应文献 Eq. 8)
        self.gcn_weight = nn.Linear(hidden_dim, hidden_dim, bias=False) 
        
        # 4. 超节点融合 MLP (对应文献 Eq. 12-13)
        self.hyper_mlp = nn.Sequential(
            nn.Linear(backbone_extractor.n_components, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim) # 输出 U_i
        )
        
        # 5. 最终预测层
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward_local(self, x, adj):
        B, N, T, C = x.shape
        x_flat = x.reshape(B, N, T * C)
        
        # --- 第一阶段：GraphSAGE 本地编码 ---
        h_in = F.relu(self.lin_in(x_flat))
        h_sage = self.sage_layer(h_in, adj) # [B, N, H]
        
        # --- 第二阶段：骨干 GCN 更新 ---
        E_emb = self.embedding.weight 
        A_tilde = F.relu(torch.matmul(E_emb, E_emb.T)) + self.epsilon * torch.eye(N, device=x.device)
        D_inv = torch.diag(1.0 / (A_tilde.sum(dim=1) + 1e-8))
        A_dynamic = torch.matmul(D_inv, A_tilde) 
        
        indices = self.backbone_extractor.extract_backbone_indices()
        h_backbone = h_sage[:, indices, :] 
        
        A_Q_mask = (adj[indices][:, indices] > 0).float()
        A_Q = A_dynamic[indices][:, indices] * A_Q_mask 
        
        gcn_out = F.relu(self.gcn_weight(torch.matmul(A_Q, h_backbone)))
        h_backbone_updated = h_backbone + gcn_out
        
        h_updated = h_sage.clone()
        h_updated[:, indices, :] = h_backbone_updated
        
        # 返回更新后的全局特征 和 骨干特征 (供超节点使用)
        return h_updated, h_backbone_updated

    def generate_hypernode(self, h_backbone):
        """修复点：接收动态交通流特征，不使用 detach()，激活梯度回传"""
        # h_backbone: [B, N_backbone, H]
        # 在 Batch 维度取平均，获得当前子图宏观的动态状态表示
        backbone_embs = h_backbone.mean(dim=0) # [N_backbone, H]
        
        pca_vecs = self.backbone_extractor.kernel_pca_projection(backbone_embs)
        pca_feat = pca_vecs.mean(dim=1) 
        
        U_i = self.hyper_mlp(pca_feat)
        return U_i

    def forward_predict(self, h, context_vector):
        B, N, H = h.shape
        ctx = context_vector.view(1, 1, H).expand(B, N, H)
        combined = torch.cat([h, ctx], dim=-1)
        out = self.regressor(combined)
        return out

class FGNNEH_Server(nn.Module):
    def __init__(self, num_clients, hidden_dim):
        super(FGNNEH_Server, self).__init__()
        self.num_clients = num_clients
        self.gcn = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, hypernode_embs, adj_matrix):
        support = torch.mm(adj_matrix, hypernode_embs)
        out = F.relu(self.gcn(support))
        return out