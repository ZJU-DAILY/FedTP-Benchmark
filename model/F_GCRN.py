import torch
import torch.nn as nn
import torch.nn.functional as F


class F_GCRN(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, embed_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # 动态图生成
        self.embed = nn.Parameter(torch.randn(num_nodes, embed_dim))

        # GRU参数
        self.Wz = nn.Linear(input_dim, hidden_dim)
        self.Uz = nn.Linear(hidden_dim, hidden_dim)
        self.Wr = nn.Linear(input_dim, hidden_dim)
        self.Ur = nn.Linear(hidden_dim, hidden_dim)
        self.Wh = nn.Linear(input_dim, hidden_dim)
        self.Uh = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, h_prev):
        # 动态邻接矩阵
        A = torch.softmax(F.relu(torch.mm(self.embed, self.embed.T)), dim=1)

        # 图卷积操作
        Ax = torch.mm(A, x)  # [N, D]
        Ah_prev = torch.mm(A, h_prev)  # [N, H]

        # GRU门控
        z = torch.sigmoid(self.Wz(Ax) + self.Uz(Ah_prev))
        r = torch.sigmoid(self.Wr(Ax) + self.Ur(Ah_prev))
        h_tilde = torch.tanh(self.Wh(Ax) + self.Uh(r * Ah_prev))
        h = z * h_prev + (1 - z) * h_tilde

        return h