import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FCGCN_Layer(nn.Module):
    """
    单层图卷积:
        H = A_hat X W + b

    输入:
        x:
            - (B, N, Fin) 或
            - (N, Fin)
        adj:
            - (N, N)，且默认已经在外部完成归一化
    输出:
        - (B, N, Fout) 或
        - (N, Fout)
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.linear.weight.size(1))
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, adj):
        # 先做特征线性变换: XW
        support = self.linear(x)

        # 再做图传播: A_hat @ (XW)
        # support: (B, N, F) or (N, F)
        # adj:     (N, N)
        if support.dim() == 3:
            # 正确的 batch 图卷积
            output = torch.einsum("ij,bjf->bif", adj, support)
        elif support.dim() == 2:
            output = torch.matmul(adj, support)
        else:
            raise ValueError(
                f"FCGCN_Layer 期望输入维度为 2 或 3，实际收到 {support.dim()} 维张量，shape={support.shape}"
            )

        if self.bias is not None:
            output = output + self.bias

        return output


class FCGCN(nn.Module):
    """
    适配当前工程的 FCGCN:
    - 输入默认来自交通预测窗口:
        x shape = (B, N, T_in, 1)
      也兼容:
        x shape = (B, N, T_in)
    - 输出:
        (B, N, T_out, 1)

    注意:
    1. adj_matrix 应当已经在外部完成加自环与归一化。
    2. 默认最后一层不加激活，更适合回归任务。
       若要更贴近论文公式，可设置 final_activation=True。
    """
    def __init__(
        self,
        num_nodes,
        in_dim,
        out_dim,
        hidden_dim,
        adj_matrix=None,
        dropout=0.5,
        final_activation=False,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.final_activation = final_activation

        self.gc1 = FCGCN_Layer(in_dim, hidden_dim)
        self.gc2 = FCGCN_Layer(hidden_dim, out_dim)

        if adj_matrix is not None:
            if not isinstance(adj_matrix, torch.Tensor):
                adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
            else:
                adj_matrix = adj_matrix.float()
            self.register_buffer("adj", adj_matrix)
        else:
            self.register_buffer("adj", torch.eye(num_nodes, dtype=torch.float32))

    def _reshape_input(self, x):
        """
        统一输入形状到 (B, N, Fin)
        当前工程中 FCGCN 的 in_dim = args.t_in，
        因此默认假设输入是 (B, N, T_in, 1) 或 (B, N, T_in)。
        """
        if x.dim() == 4:
            # 常见情况: (B, N, T, 1)
            if x.size(-1) != 1:
                raise ValueError(
                    f"当前 FCGCN 假设最后一个维度为 1，但收到 x.shape={x.shape}。"
                    f"若你要支持多特征输入，需要同时把 fate_main.py 中的 in_dim 改为 T_in * input_dim。"
                )
            x = x.squeeze(-1)  # -> (B, N, T)

        elif x.dim() == 3:
            # 已经是 (B, N, Fin)
            pass

        elif x.dim() == 2:
            # 极少数情况下允许 (N, Fin)
            pass

        else:
            raise ValueError(
                f"FCGCN 只支持 2/3/4 维输入，实际收到 {x.dim()} 维，shape={x.shape}"
            )

        # 最后一维必须和 in_dim 一致
        if x.shape[-1] != self.in_dim:
            raise ValueError(
                f"输入特征维不匹配: x.shape[-1]={x.shape[-1]}, 但模型 in_dim={self.in_dim}"
            )

        return x

    def forward(self, x):
        x = self._reshape_input(x)

        # 第一层: ReLU(A_hat X W0 + b)
        h = self.gc1(x, self.adj)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # 第二层: A_hat H W1 + b
        out = self.gc2(h, self.adj)

        # 若你要机械贴论文公式，可打开这个开关
        if self.final_activation:
            out = F.relu(out)

        # 与你当前工程的 loss / trainer 输出格式保持一致
        if out.dim() == 3:
            out = out.unsqueeze(-1)   # (B, N, T_out, 1)
        elif out.dim() == 2:
            out = out.unsqueeze(0).unsqueeze(-1)  # (1, N, T_out, 1)

        return out