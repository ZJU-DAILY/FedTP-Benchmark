import torch
import torch.nn.functional as F

class FGNNEH_Backbone:
    """
    论文 III.A.2 & III.A.3: 拓扑骨干提取与核矩阵分析
    """
    def __init__(self, adj_matrix, P, gamma, n_components, device):
        self.adj = adj_matrix
        self.P = P
        self.gamma = gamma
        self.n_components = n_components
        self.device = device
        self.num_nodes = adj_matrix.shape[0]

    def extract_backbone_indices(self):
        """提取骨干节点的索引 (基于度中心性)"""
        # 1. 计算度中心性 (Eq. 5)
        # 加上 1e-6 防止除0
        degrees = torch.sum(self.adj > 0, dim=1).float()
        centrality = degrees / (self.num_nodes - 1 + 1e-6)
        
        # 2. 排序与累计贡献率 (Eq. 6-7)
        sorted_vals, sorted_indices = torch.sort(centrality, descending=True)
        total_sum = torch.sum(centrality)
        cumulative_sum = torch.cumsum(sorted_vals, dim=0)
        
        # 截断
        # 注意：如果P设得太小或图太稀疏，可能选出的节点很少
        cutoff = torch.searchsorted(cumulative_sum, self.P * total_sum).item()
        # 强制至少选1个，且 +1 包含边界节点
        cutoff = max(1, cutoff + 1)
        return sorted_indices[:cutoff]
        
        return sorted_indices[:cutoff]

    def kernel_pca_projection(self, backbone_embeddings):
        """
        对应论文 III.A.3: Matrix Analysis -> High-Dimensional Spaces
        输入: [num_backbone, hidden_dim]
        输出: [n_components, num_backbone] (作为特征向量，自动Padding)
        """
        N_backbone = backbone_embeddings.shape[0]

        # === [Fix 1] 处理只有1个节点导致的 NaN 问题 ===
        if N_backbone > 1:
            mean = backbone_embeddings.mean(dim=0)
            std = backbone_embeddings.std(dim=0) + 1e-6
            z_norm = (backbone_embeddings - mean) / std
        else:
            # 只有一个节点时，不进行标准化或简单置0，避免 std 为 NaN
            z_norm = backbone_embeddings

        # 2. RBF Kernel (Eq. 9)
        dist_sq = torch.cdist(z_norm, z_norm) ** 2
        K = torch.exp(-self.gamma * dist_sq)
        
        # 3. Center Kernel (Eq. 10)
        N = K.shape[0]
        one_n = torch.ones((N, N), device=self.device) / N
        K_tilde = K - torch.mm(one_n, K) - torch.mm(K, one_n) + torch.mm(torch.mm(one_n, K), one_n)
        
        # 4. Eigendecomposition (Eq. 11)
        try:
            # 使用 eigh 处理对称矩阵
            eig_vals, eig_vecs = torch.linalg.eigh(K_tilde)
        except:
            # 数值稳定性回退
            eig_vals, eig_vecs = torch.linalg.eigh(K_tilde + 1e-4 * torch.eye(N, device=self.device))
            
        # === [Fix 2] 自动 Padding 逻辑，修复 Dimension Mismatch 报错 ===
        # 能够提取的最大特征数受限于节点数 N
        num_available = eig_vals.shape[0]
        # 我们实际能取的数量
        k = min(self.n_components, num_available)
        
        # 取最大的前 k 个
        idx = torch.argsort(eig_vals, descending=True)[:k]
        
        # === [Fix] 补充缺失的特征值缩放 (Eq. 11) ===
        top_vals = eig_vals[idx].clamp(min=1e-8) # 防止负数开根号报错
        top_vecs = eig_vecs[:, idx] # [N_backbone, k]
        
        # 转置并乘上 sqrt(lambda) -> [k, N_backbone]
        result = (top_vecs * torch.sqrt(top_vals).unsqueeze(0)).t()
        
        # 如果提取出的特征数 k 小于模型要求的 n_components (8)，则补零
        if k < self.n_components:
            pad_size = self.n_components - k
            # F.pad 参数顺序: (最后一维左, 最后一维右, 倒数第二维上, 倒数第二维下)
            # 我们要在 维度0 (Feature维度) 下方补 pad_size 行
            result = F.pad(result, (0, 0, 0, pad_size))
            
        # 确保形状是 [n_components, N_backbone]
        return result