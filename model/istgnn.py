import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class ISTGNN(nn.Module):
    """
    T-ISTGNN / ISTGNN backbone
    输入:
        x: [B, N, T, F]
    输出:
        y: [B, P, N, 1]

    结构:
        Feature Extractor = 2-layer GraphSAGE
        Predictor         = GRU + FC1 + FC2
    """

    def __init__(
        self,
        edge_index,
        in_channels,
        hidden_channels,
        gru_hidden_size,
        num_nodes,
        pre_len
    ):
        super().__init__()

        # ===== 基本属性 =====
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_index = edge_index.long().contiguous()

        # register_buffer: 跟随 model.to(device) 自动搬运，但不会被优化器更新
        self.register_buffer("edge_index", edge_index)

        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.gru_hidden_size = int(gru_hidden_size)
        self.num_nodes = int(num_nodes)
        self.pre_len = int(pre_len)

        # ===== Feature Extractor: 2-layer GraphSAGE =====
        self.sage1 = SAGEConv(self.in_channels, self.hidden_channels)
        self.sage2 = SAGEConv(self.hidden_channels, self.hidden_channels)

        # ===== Predictor: GRU + FC1 + FC2 =====
        self.temporal = nn.GRU(
            input_size=self.hidden_channels,
            hidden_size=self.gru_hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.fc1 = nn.Linear(self.gru_hidden_size, self.gru_hidden_size)
        self.fc2 = nn.Linear(self.gru_hidden_size, self.pre_len)

        self._reset_parameters()

    def _reset_parameters(self):
        """
        更稳妥的初始化。
        """
        for m in self.modules():
            if isinstance(m, (nn.Linear, SAGEConv)):
                # SAGEConv 内部有多个 Linear，PyG 自己会初始化；
                # 这里显式处理 Linear 层即可。
                pass

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        for name, param in self.temporal.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def _build_batched_edge_index(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        把单图 edge_index 扩展成 batch 图 edge_index。
        当前客户端本地子图节点数固定为 self.num_nodes。
        若 batch_size=B，则总节点数视为 B 个互不相连的同构子图拼接。

        原 edge_index: [2, E]
        返回 batched_edge_index: [2, B*E]
        """
        edge_index = self.edge_index.to(device)

        if batch_size == 1:
            return edge_index

        e = edge_index.size(1)

        # 复制 B 份
        batched = edge_index.repeat(1, batch_size)  # [2, B*E]

        # 对第 b 份图的节点编号整体加上 b * num_nodes
        offsets = torch.arange(batch_size, device=device).repeat_interleave(e) * self.num_nodes
        batched = batched + offsets.unsqueeze(0)

        return batched

    def forward_feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        空间特征提取
        输入:
            x: [B, N, T, F]
        输出:
            z: [B, N, T, H]
        """
        if x.dim() != 4:
            raise ValueError(f"ISTGNN expects x with 4 dims [B,N,T,F], but got shape={tuple(x.shape)}")

        B, N, T, Fin = x.shape
        if N != self.num_nodes:
            raise ValueError(
                f"Node count mismatch: model.num_nodes={self.num_nodes}, but input has N={N}"
            )
        if Fin != self.in_channels:
            raise ValueError(
                f"Input feature dim mismatch: model.in_channels={self.in_channels}, but input has F={Fin}"
            )

        batched_edge_index = self._build_batched_edge_index(B, x.device)

        spatial_seq = []
        for t in range(T):
            # [B, N, F] -> [B*N, F]
            xt = x[:, :, t, :].reshape(B * N, Fin)

            h = self.sage1(xt, batched_edge_index)
            h = F.relu(h)

            h = self.sage2(h, batched_edge_index)
            h = F.relu(h)

            # [B*N, H] -> [B, N, H]
            h = h.view(B, N, self.hidden_channels)
            spatial_seq.append(h)

        # list(T * [B,N,H]) -> [B,N,T,H]
        z = torch.stack(spatial_seq, dim=2)
        return z

    def forward_predict(self, z: torch.Tensor) -> torch.Tensor:
        """
        时序预测头
        输入:
            z: [B, N, T, H]
        输出:
            y: [B, P, N, 1]
        """
        if z.dim() != 4:
            raise ValueError(f"forward_predict expects z with 4 dims [B,N,T,H], got shape={tuple(z.shape)}")

        B, N, T, H = z.shape
        if H != self.hidden_channels:
            raise ValueError(
                f"Hidden dim mismatch: model.hidden_channels={self.hidden_channels}, but z has H={H}"
            )

        # 每个节点一条时序:
        # [B, N, T, H] -> [B*N, T, H]
        z_seq = z.reshape(B * N, T, H)

        # 只取最后时刻隐藏状态
        _, h_n = self.temporal(z_seq)   # h_n: [1, B*N, Hgru]
        h_last = h_n[-1]                # [B*N, Hgru]

        out = self.fc1(h_last)
        out = F.relu(out)
        out = self.fc2(out)             # [B*N, P]

        # [B*N, P] -> [B, N, P]
        out = out.view(B, N, self.pre_len)

        # -> [B, P, N, 1]
        out = out.permute(0, 2, 1).unsqueeze(-1)
        return out

    def forward(self, x: torch.Tensor, return_feature: bool = False):
        """
        默认返回预测；
        return_feature=True 时返回 (pred, feature)
        """
        z = self.forward_feature(x)
        y = self.forward_predict(z)

        if return_feature:
            return y, z
        return y

    # ========= 下面这些接口是给 T-ISTGNN(i-c) 迁移阶段准备的 =========

    def freeze_predictor(self):
        """
        冻结 Predictor = GRU + FC1 + FC2
        """
        for module in [self.temporal, self.fc1, self.fc2]:
            for p in module.parameters():
                p.requires_grad = False

    def unfreeze_predictor(self):
        """
        解冻 Predictor
        """
        for module in [self.temporal, self.fc1, self.fc2]:
            for p in module.parameters():
                p.requires_grad = True

    def freeze_extractor(self):
        """
        冻结 Feature Extractor = GraphSAGE x 2
        """
        for module in [self.sage1, self.sage2]:
            for p in module.parameters():
                p.requires_grad = False

    def unfreeze_extractor(self):
        """
        解冻 Feature Extractor
        """
        for module in [self.sage1, self.sage2]:
            for p in module.parameters():
                p.requires_grad = True

    def copy_extractor_from(self, other: "ISTGNN"):
        """
        只复制 extractor 参数
        """
        self.sage1.load_state_dict(other.sage1.state_dict())
        self.sage2.load_state_dict(other.sage2.state_dict())

    def copy_predictor_from(self, other: "ISTGNN"):
        """
        只复制 predictor 参数
        """
        self.temporal.load_state_dict(other.temporal.state_dict())
        self.fc1.load_state_dict(other.fc1.state_dict())
        self.fc2.load_state_dict(other.fc2.state_dict())

    def extractor_parameters(self):
        """
        返回 extractor 参数列表，便于 target adaptation 只训 extractor
        """
        return list(self.sage1.parameters()) + list(self.sage2.parameters())

    def predictor_parameters(self):
        """
        返回 predictor 参数列表
        """
        return list(self.temporal.parameters()) + list(self.fc1.parameters()) + list(self.fc2.parameters())