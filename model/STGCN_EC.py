import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils

class SpatialTemporalAttention(nn.Module):
    """
    严格对齐原文献的 Spatial-Temporal Attention 机制
    """
    def __init__(self, num_nodes, time_steps):
        super().__init__()
        # 空间注意力参数
        self.W_s = nn.Parameter(torch.empty(time_steps, time_steps))
        self.b_s = nn.Parameter(torch.empty(1, num_nodes, num_nodes))
        self.U_s = nn.Parameter(torch.empty(1, num_nodes, num_nodes))
        
        # 时间注意力参数
        self.W_e = nn.Parameter(torch.empty(num_nodes, num_nodes))
        self.b_e = nn.Parameter(torch.empty(1, time_steps, time_steps))
        self.U_e = nn.Parameter(torch.empty(1, time_steps, time_steps))
        
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_s)
        nn.init.xavier_uniform_(self.b_s)
        nn.init.xavier_uniform_(self.U_s)
        nn.init.xavier_uniform_(self.W_e)
        nn.init.xavier_uniform_(self.b_e)
        nn.init.xavier_uniform_(self.U_e)

    def forward(self, X):
        """
        X shape: (Batch, Num_nodes, Time_steps)
        """
        # 1. 空间注意力 S_i (Eq. 8 & 9)
        # X @ W_s @ X^T -> (B, N, T) @ (T, T) @ (B, T, N) -> (B, N, N)
        S_raw = torch.matmul(torch.matmul(X, self.W_s), X.transpose(1, 2)) + self.b_s
        S_score = self.U_s * torch.sigmoid(S_raw)
        S = F.softmax(S_score, dim=-1) # (B, N, N)

        # 2. 时间注意力 E_i (Eq. 10 & 11)
        # X^T @ W_e @ X -> (B, T, N) @ (N, N) @ (B, N, T) -> (B, T, T)
        E_raw = torch.matmul(torch.matmul(X.transpose(1, 2), self.W_e), X) + self.b_e
        E_score = self.U_e * torch.sigmoid(E_raw)
        E = F.softmax(E_score, dim=-1) # (B, T, T)

        # 3. 融合注意力 (Algorithm 1, Step 6)
        # \hat{X} = S * X * E
        X_hat = torch.matmul(torch.matmul(S, X), E) # (B, N, T)
        
        return X_hat

class ChebConvDense(nn.Module):
    """
    密集的切比雪夫多项式图卷积 (Algorithm 1, Step 7)
    """
    def __init__(self, K, in_features, out_features):
        super().__init__()
        self.K = K
        self.weight = nn.Parameter(torch.empty(K, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, cheb_polynomials):
        """
        x: (Batch * Time, Num_nodes, in_features)
        cheb_polynomials: List of (Num_nodes, Num_nodes) tensors length K
        """
        B_T, N, F_in = x.shape
        out = torch.zeros(B_T, N, self.weight.shape[2], device=x.device)
        
        for k in range(self.K):
            T_k = cheb_polynomials[k] # (N, N)
            # T_k @ x -> (N, N) @ (B_T, N, F_in) => 适配 einsum
            # 'nm, bmf -> bnf'
            x_k = torch.einsum('nm, bmf -> bnf', T_k, x) 
            out += torch.matmul(x_k, self.weight[k])
            
        return out + self.bias

class SpatioTemporalModel(nn.Module):
    def __init__(self, feat_dim, hidden_dim, time_steps, output_dim=1, K=3, max_nodes=None):
        super().__init__()
        # 为了兼容不同的客户端节点数，优先使用 max_nodes 初始化 Attention
        self.num_nodes = max_nodes if max_nodes is not None else 307 
        self.time_steps = time_steps
        self.output_dim = output_dim 
        self.hidden_dim = hidden_dim
        self.K = K # Chebyshev polynomial order (论文中设为2或3)

        # 1. 空间-时间注意力模块
        self.st_attention = SpatialTemporalAttention(self.num_nodes, time_steps)

        # 2. 切比雪夫图卷积层
        self.cheb_conv = ChebConvDense(K=self.K, in_features=feat_dim, out_features=hidden_dim)

        # 3. GRU层
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # 4. 输出层
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # 注册图结构相关的缓冲区
        self.register_buffer('edge_index', None)
        self.cheb_polynomials = None 

    def compute_cheb_polynomials(self, edge_index, N, device):
        """计算缩放拉普拉斯矩阵的切比雪夫多项式 (Eq. 13)"""
        adj = pyg_utils.to_dense_adj(edge_index, max_num_nodes=N)[0].to(device)
        
        # 计算归一化拉普拉斯 L = I - D^{-1/2} A D^{-1/2}
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        L_norm = torch.eye(N, device=device) - torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)
        
        # 计算缩放拉普拉斯 \tilde{L} = (2 / \lambda_{max}) * L - I
        # 简化处理：近似假定归一化拉普拉斯的最大特征值为 2
        L_tilde = L_norm - torch.eye(N, device=device)
        
        # 计算切比雪夫多项式 T_k(\tilde{L})
        cheb_polys = [torch.eye(N, device=device), L_tilde]
        for k in range(2, self.K):
            T_k = 2 * torch.mm(L_tilde, cheb_polys[k-1]) - cheb_polys[k-2]
            cheb_polys.append(T_k)
            
        return cheb_polys

    def forward(self, x, edge_index=None):
        """
        x: (B, N, T, Feat)
        Output: (B, N, T_out, 1)
        """
        # --- 0. 数据清洗与维度对齐 ---
        if x.is_sparse: x = x.to_dense()
        if x.dim() == 5:
            if x.shape[-1] == 1: x = x.squeeze(-1)
            elif x.shape[1] == 1: x = x.squeeze(1)
        if x.dim() == 3: x = x.unsqueeze(0)
        
        B, N_real, T, Feat = x.size()
        
        # ==========================================
        # 🌟 关键修复：自动 Padding 机制对齐最大节点数
        # ==========================================
        if N_real < self.num_nodes:
            pad_size = self.num_nodes - N_real
            # F.pad 从最后一个维度开始往前填充：(Feat, T, N, B)
            # 我们要在倒数第 2 维 (Node维) 的右侧补 pad_size 个 0
            x_padded = F.pad(x, (0, 0, 0, 0, 0, pad_size)) 
        elif N_real > self.num_nodes:
            raise ValueError(f"Input nodes {N_real} exceeds max_nodes {self.num_nodes}!")
        else:
            x_padded = x

        # 此刻统一使用模型初始化的全局最大节点数进行运算
        N = self.num_nodes 

        # --- 1. 准备切比雪夫多项式 ---
        if self.cheb_polynomials is None or self.cheb_polynomials[0].shape[0] != N:
            if edge_index is None: edge_index = self.edge_index
            if edge_index is None: raise ValueError("Edge Index missing")
            # 传入 N 会自动让 PyG 生成带有 Padding 孤立节点的全局等大邻接矩阵
            self.cheb_polynomials = self.compute_cheb_polynomials(edge_index, N, x.device)

        # ==========================================
        # Step 1: 空间-时间注意力机制 (Spatial-Temporal Attention)
        # ==========================================
        # 如果 Feat=1，把最后一个维度去掉以便运算 (B, N, T)
        x_squeeze = x_padded.squeeze(-1) if Feat == 1 else x_padded.mean(dim=-1)
        
        # 严格执行 \hat{X} = S * X * E
        x_hat = self.st_attention(x_squeeze) # (B, N, T)
        x_hat = x_hat.unsqueeze(-1) if Feat == 1 else x_hat.unsqueeze(-1).repeat(1, 1, 1, Feat)

        # ==========================================
        # Step 2: 切比雪夫图卷积网络 (GCN)
        # ==========================================
        # 变换形状以共享时间步的图卷积: (B, N, T, Feat) -> (B*T, N, Feat)
        x_gcn_in = x_hat.transpose(1, 2).reshape(B * T, N, Feat)
        
        # 获取 Graph Embedding (G_x)
        g_x = self.cheb_conv(x_gcn_in, self.cheb_polynomials) # (B*T, N, Hidden)
        g_x = F.relu(g_x)
        
        # 恢复形状: (B, N, T, Hidden)
        g_x = g_x.view(B, T, N, self.hidden_dim).transpose(1, 2)

        # ==========================================
        # Step 3: Gated Recurrent Unit (GRU)
        # ==========================================
        # GRU 序列建模: (B, N, T, Hidden) -> (B*N, T, Hidden)
        x_gru_in = g_x.reshape(B * N, T, self.hidden_dim)
        
        _, h_n = self.gru(x_gru_in) # h_n: (1, B*N, Hidden)
        h_n = h_n.squeeze(0) # 取最后时刻的记忆状态 (B*N, Hidden)
        
        # ==========================================
        # Step 4: 输出层 (Prediction) 与 逆向裁剪
        # ==========================================
        out = self.fc(h_n) # (B*N, output_dim)
        out = out.view(B, N, self.output_dim)
        
        # 🌟 关键收尾：把 Padding 进来的假节点切掉，还原为真实节点的形状
        out = out[:, :N_real, :] # (B, N_real, output_dim)
        
        return out.unsqueeze(-1)