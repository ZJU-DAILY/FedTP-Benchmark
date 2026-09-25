import torch
import torch.nn as nn
import torch.nn.functional as F

class TPBank(nn.Module):
    """
    Temporal Pattern Bank (TP-Bank) 模块
    对应论文 3.1 节，用于存储和更新长期演化模式。
    """
    def __init__(self, K, d, input_dim, seq_len, feature_dim):
        super(TPBank, self).__init__()
        self.K = K
        self.d = d
        
        # 模式库 B: [K, d]
        self.B = nn.Parameter(torch.randn(K, d))
        
        # Query 映射: W_q, b_q
        self.W_q = nn.Linear(input_dim, d)
        
        # 用于计算正则化矩阵 \Omega 的可学习参数 U1, U2, U3 (对应 Eq. 5)
        # 假设 X 展平后经过 U1
        self.U1 = nn.Linear(seq_len * feature_dim, d)
        self.U2 = nn.Linear(d, d)
        self.U3 = nn.Linear(d, d)

    def forward(self, h_tau, X_tau, B_prev=None):
        """
        h_tau: 短期时序嵌入 [Batch, Nodes, hidden_dim]
        X_tau: 原始输入数据 [Batch, Nodes, seq_len, feature_dim]
        B_prev: 上一个时间间隔的模式库 (用于计算 L_k)，如果为 None 则不计算 L_k
        """
        B, N, H = h_tau.shape
        
        # 1. 计算 Query 向量 q (Eq. 2)
        q = self.W_q(h_tau)  # [B, N, d]
        
        # 2. 注意力机制匹配长序列模式 (Eq. 3)
        # q: [B, N, d], B: [K, d] -> scores: [B, N, K]
        scores = torch.matmul(q, self.B.transpose(0, 1))
        p = F.softmax(scores, dim=-1)
        
        # 3. 提取长序列特征 z (Eq. 3)
        # p: [B, N, K], B: [K, d] -> z: [B, N, d]
        z_tau = torch.matmul(p, self.B)
        
        # 4. 计算自适应更新的正则化损失 L_k (Eq. 5)
        L_k = torch.tensor(0.0, device=h_tau.device)
        if B_prev is not None:
            # 展平 X 以计算 Omega
            X_flat = X_tau.reshape(B, N, -1) # [B, N, seq_len * feature_dim]
            # (X * U1)^T
            part1 = self.U1(X_flat).transpose(1, 2) # [B, d, N]
            # U3 * B_prev
            part3 = self.U3(B_prev) # [K, d]
            
            # 由于维度匹配问题，这里我们做适当的平均或降维操作以符合批处理
            # 简化版 Omega 计算：利用特征均值代表整体动态
            part1_mean = part1.mean(dim=0) # [d, N]
            Omega = torch.sigmoid(torch.matmul(part1_mean.transpose(0, 1), self.U2(part3).transpose(0, 1))) # [N, K]
            # 将 Omega 适配成 [K, d] 的更新约束矩阵
            Omega_reduced = Omega.mean(dim=0).unsqueeze(1) # [K, 1]
            
            # L_k = || B_tau - B_{tau-1} * \Omega ||^2
            L_k = torch.norm(self.B - B_prev * Omega_reduced, p=2) ** 2
            
        return z_tau, L_k

class FedSTG_Client(nn.Module):
    """
    FedSTG 客户端本地模型
    """
    def __init__(self, in_dim, out_dim, hidden_dim, K, d, seq_len, num_nodes):
        super(FedSTG_Client, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # 1. 短期特征提取 GRU
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden_dim, batch_first=True)
        
        # 2. 长期特征提取 TP-Bank
        self.tp_bank = TPBank(K=K, d=d, input_dim=hidden_dim, seq_len=seq_len, feature_dim=in_dim)
        
        # 3. 预测器 Predictor
        # 结合短期 h_tau, 长期 z_tau, 以及服务端返回的全局结构 h_G
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + d + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
        # 缓存上一轮的 TP-Bank 参数用于正则化
        self.B_prev = None

    def forward_encode(self, X):
        """
        第一阶段：编码阶段
        X: [B, N, T, F]
        """
        B, N, T, F = X.shape
        X_flat = X.reshape(B * N, T, F)
        
        # GRU 提取短期时序模式
        gru_out, h_n = self.gru(X_flat)
        h_tau = h_n[-1].reshape(B, N, self.hidden_dim) # [B, N, hidden_dim]
        
        # TP-Bank 提取长期模式，并计算 L_k
        z_tau, L_k = self.tp_bank(h_tau, X, self.B_prev) # z_tau: [B, N, d]
        
        return h_tau, z_tau, L_k

    def forward_predict(self, h_tau, z_tau, h_G):
        """
        第二阶段：预测阶段
        h_G: 从 Server 获取的全局结构特征 [B, N, hidden_dim]
        """
        # 拼接特征: [B, N, hidden_dim + d + hidden_dim]
        fused_features = torch.cat([h_tau, z_tau, h_G], dim=-1)
        
        # 预测未来步长
        pred = self.predictor(fused_features) # [B, N, out_dim]
        # 调整维度以对齐系统接口要求 [B, N, out_dim, 1]
        pred = pred.unsqueeze(-1)
        
        return pred
        
    def update_B_prev(self):
        """每轮训练结束/开始时调用，缓存上一轮的 B 参数"""
        self.B_prev = self.tp_bank.B.detach().clone()


class FedSTG_Server(nn.Module):
    """
    FedSTG 服务端模型：图结构学习与融合
    对应论文 3.2 节
    """
    def __init__(self, hidden_dim, beta=0.5):
        super(FedSTG_Server, self).__init__()
        self.hidden_dim = hidden_dim
        self.beta = beta
        
        # 静态图 GCN 参数 (Eq. 8)
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        
        # 动态/演化图 GCN 参数 (Eq. 8)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 门控融合参数 (Eq. 9)
        self.Wg1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wg2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bg = nn.Parameter(torch.zeros(hidden_dim))

    def build_evolutionary_graph(self, z_c):
        """
        基于长期模式 z_c 构建演化图 A^E (Eq. 7)
        z_c: [Num_Clients, d] -> 代表每个 Client 的长期模式期望
        """
        Num_Clients = z_c.shape[0]
        A_E = torch.zeros((Num_Clients, Num_Clients), device=z_c.device)
        
        # 计算余弦相似度
        z_norm = F.normalize(z_c, p=2, dim=1)
        sim_matrix = torch.matmul(z_norm, z_norm.transpose(0, 1))
        
        # 应用阈值 Beta 过滤
        mask = sim_matrix >= self.beta
        A_E[mask] = sim_matrix[mask]
        
        return A_E

    def forward(self, h_c, z_c_mean, A_S):
        """
        h_c: 所有客户端短期特征拼合 [Batch, Total_Nodes, hidden_dim]
        z_c_mean: 各节点长期模式期望 [Total_Nodes, d]
        A_S: 静态图邻接矩阵 [Total_Nodes, Total_Nodes]
        """
        B, N, H = h_c.shape
        
        # 1. 动态构建演化图 A_E
        A_E = self.build_evolutionary_graph(z_c_mean)
        
        # 加入自环 (Eq. 8 中 A + I)
        I_S = torch.eye(N, device=h_c.device)
        A_S_tilde = A_S + I_S
        A_E_tilde = A_E + I_S
        
        # 2. 双流 GCN 计算 (Eq. 8)
        # h_G^S = \sigma((A^S + I) h_c W1)
        h_c_w1 = self.W1(h_c) # [B, N, H]
        h_G_S = F.relu(torch.matmul(A_S_tilde, h_c_w1))
        
        # h_G^E = \sigma((A^E + I) h_c W2)
        h_c_w2 = self.W2(h_c) # [B, N, H]
        h_G_E = F.relu(torch.matmul(A_E_tilde, h_c_w2))
        
        # 3. 门控融合 Gated Fusion (Eq. 9)
        # g = \sigma(h_G^S W_{g,1} + h_G^E W_{g,2} + b_g)
        g = torch.sigmoid(self.Wg1(h_G_S) + self.Wg2(h_G_E) + self.bg)
        
        # h_G = g \odot h_G^S + (1-g) \odot h_G^E
        h_G = g * h_G_S + (1 - g) * h_G_E
        
        return h_G