import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils


def _normalize_dense_adj(adj: torch.Tensor) -> torch.Tensor:
    degree = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return adj / degree


class TemporalPooling(nn.Module):
    def __init__(self, ratio: int):
        super().__init__()
        self.ratio = int(ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ratio <= 1:
            return x
        bsz, steps, nodes, channels = x.shape
        if steps % self.ratio != 0:
            trim = steps - (steps % self.ratio)
            x = x[:, :trim]
            steps = x.size(1)
        x = x.reshape(bsz, steps // self.ratio, self.ratio, nodes, channels)
        return x.mean(dim=2)


class DynamicSTInteraction(nn.Module):
    def __init__(self, hidden_dim: int, num_nodes: int, window_size: int = 3):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_nodes = int(num_nodes)
        self.window_size = int(window_size)
        self.padding = self.window_size - 1
        self.proj_1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.proj_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.register_buffer("expanded_adj", torch.empty(0))

    def set_static_adj(self, adj: torch.Tensor) -> None:
        expanded = torch.zeros(
            self.num_nodes,
            self.num_nodes * self.window_size,
            dtype=adj.dtype,
            device=adj.device,
        )
        expanded[:, -self.num_nodes:] = adj
        expanded[:, :] += torch.eye(self.num_nodes, dtype=adj.dtype, device=adj.device).repeat(1, self.window_size)
        self.expanded_adj = _normalize_dense_adj(expanded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, steps, nodes, channels = x.shape
        if self.expanded_adj.numel() == 0:
            eye = torch.eye(nodes, dtype=x.dtype, device=x.device)
            self.set_static_adj(eye)
        pad = torch.zeros(bsz, self.padding, nodes, channels, dtype=x.dtype, device=x.device)
        feat = torch.cat([pad, x], dim=1)
        outputs = []
        adj = self.expanded_adj.to(dtype=x.dtype, device=x.device)
        for idx in range(steps):
            window_feat = feat[:, idx : idx + self.window_size]
            large_graph_feat = window_feat.reshape(bsz, -1, channels)
            lhs = adj @ self.proj_1(large_graph_feat)
            rhs = adj @ self.proj_2(large_graph_feat)
            interactive = self.activation(lhs * rhs)
            outputs.append(interactive + lhs)
        stacked = torch.stack(outputs, dim=1)
        return self.norm(stacked + x)


class HypergraphLearning(nn.Module):
    def __init__(self, hidden_dim: int, num_edges: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_edges = int(num_edges)
        self.edge_clf = nn.Parameter(torch.randn(self.hidden_dim, self.num_edges) / max(self.num_edges, 1) ** 0.5)
        self.edge_map = nn.Parameter(torch.randn(self.num_edges, self.num_edges) / max(self.num_edges, 1) ** 0.5)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, steps, nodes, channels = x.shape
        feat = x.reshape(bsz, steps * nodes, channels)
        assignment = torch.softmax(feat @ self.edge_clf, dim=-1)
        hyper_feat = assignment.transpose(1, 2) @ feat
        hyper_feat = self.activation(self.edge_map @ hyper_feat)
        out = self.activation(assignment @ hyper_feat)
        out = out.reshape(bsz, steps, nodes, channels)
        return self.norm(out + x)


class LearnedAdjGNNLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_nodes: int, use_learned_adj: bool, padding: int = 0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_nodes = int(num_nodes)
        self.use_learned_adj = bool(use_learned_adj)
        self.padding = int(padding)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        if self.use_learned_adj:
            self.weights = nn.Parameter(torch.rand(self.hidden_dim))
        self.register_buffer("predefined_adj", torch.empty(0))

    def set_predefined_adj(self, adj: torch.Tensor) -> None:
        stacked = torch.zeros(3 * self.num_nodes, 3 * self.num_nodes, dtype=adj.dtype, device=adj.device)
        stacked[: self.num_nodes, : self.num_nodes] = adj
        stacked[self.num_nodes : 2 * self.num_nodes, self.num_nodes : 2 * self.num_nodes] = adj
        stacked[-self.num_nodes :, -self.num_nodes :] = adj
        stacked = stacked + torch.eye(self.num_nodes, dtype=adj.dtype, device=adj.device).repeat(3, 3)
        self.predefined_adj = _normalize_dense_adj(stacked).unsqueeze(0)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        bsz, steps, nodes, channels = feat.shape
        if self.padding > 0:
            pad = torch.zeros(bsz, self.padding, nodes, channels, dtype=feat.dtype, device=feat.device)
            feat = torch.cat([feat, pad], dim=1)
        if self.predefined_adj.numel() == 0:
            self.set_predefined_adj(torch.eye(nodes, dtype=feat.dtype, device=feat.device))

        weighted_feat = None
        if self.use_learned_adj:
            weighted_feat = F.normalize(feat * torch.sigmoid(self.weights), p=2, dim=-1)

        outputs = []
        predefined_adj = self.predefined_adj.to(dtype=feat.dtype, device=feat.device)
        for idx in range(2, feat.size(1)):
            feature = feat[:, idx - 2 : idx + 1].reshape(bsz, -1, channels)
            feature_sum = feat[:, idx]
            if self.use_learned_adj:
                local_weighted = weighted_feat[:, idx - 2 : idx + 1].reshape(bsz, -1, channels)
                learned_adj = local_weighted @ local_weighted.transpose(1, 2)
                learned_adj = _normalize_dense_adj(learned_adj)
                message = learned_adj @ feature
            else:
                message = predefined_adj @ feature
            message = self.ffn(message[:, -self.num_nodes :, :])
            outputs.append(self.layer_norm(feature_sum + message))
        return torch.stack(outputs, dim=1)


class STBackbone(nn.Module):
    def __init__(self, hidden_dim: int, num_nodes: int, num_layers: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_nodes = int(num_nodes)
        self.num_layers = int(num_layers)
        self.static_layers = nn.ModuleList(
            [LearnedAdjGNNLayer(self.hidden_dim, self.num_nodes, use_learned_adj=False, padding=2) for _ in range(self.num_layers)]
        )
        self.dynamic_layers = nn.ModuleList(
            [LearnedAdjGNNLayer(self.hidden_dim, self.num_nodes, use_learned_adj=True, padding=2) for _ in range(self.num_layers)]
        )

    def set_predefined_adj(self, adj: torch.Tensor) -> None:
        for layer in list(self.static_layers) + list(self.dynamic_layers):
            layer.set_predefined_adj(adj)

    def _run_path(self, x: torch.Tensor, layers: nn.ModuleList) -> torch.Tensor:
        for layer in layers:
            x = layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        static_feat = self._run_path(x, self.static_layers)
        dynamic_feat = self._run_path(x, self.dynamic_layers)
        return torch.stack([static_feat, dynamic_feat], dim=3).max(dim=3)[0]


class STGCNWithHypergraphLearning(nn.Module):
    def __init__(self, hidden_dim: int, num_nodes: int, num_edges: int, depth: int, winsize: int, dropout: float):
        super().__init__()
        self.st_layers = nn.ModuleList(
            [DynamicSTInteraction(hidden_dim, num_nodes, window_size=winsize) for _ in range(int(depth))]
        )
        self.hyper = HypergraphLearning(hidden_dim, num_edges)
        self.dropout = nn.Dropout(float(dropout))

    def set_static_adj(self, adj: torch.Tensor) -> None:
        for layer in self.st_layers:
            layer.set_static_adj(adj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.st_layers):
            local = layer(x)
            hyper = self.hyper(x)
            x = 0.5 * (local + hyper)
            if idx != len(self.st_layers) - 1:
                x = self.dropout(x)
        return x


class DyHSL(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        t_in: int,
        t_out: int,
        input_dim: int = 1,
        output_dim: int = 1,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        num_backbone_layers: int = 2,
        num_head_layers: int = 2,
        num_hyper_edge: int = 32,
        winsize: int = 3,
        scales=(1, 3, 6, 12),
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.t_in = int(t_in)
        self.t_out = int(t_out)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.scales = tuple(int(scale) for scale in scales if int(scale) > 0 and self.t_in % int(scale) == 0)
        if not self.scales:
            self.scales = (1,)

        self.input_embedding = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU())
        self.time_embedding = nn.Embedding(max(self.t_in, 32), self.hidden_dim)
        self.node_embedding = nn.Embedding(self.num_nodes, self.hidden_dim)

        self.backbone = STBackbone(self.hidden_dim, self.num_nodes, num_backbone_layers)
        self.multi_scale_heads = nn.ModuleList()
        for scale in self.scales:
            pooling = TemporalPooling(scale)
            head = STGCNWithHypergraphLearning(
                hidden_dim=self.hidden_dim,
                num_nodes=self.num_nodes,
                num_edges=num_hyper_edge,
                depth=num_head_layers,
                winsize=winsize,
                dropout=self.dropout,
            )
            self.multi_scale_heads.append(nn.ModuleDict({"pool": pooling, "head": head}))

        fused_dim = self.hidden_dim * len(self.scales)
        self.local_fusion = nn.Sequential(nn.Linear(fused_dim, self.hidden_dim), nn.ReLU())
        self.global_fusion = nn.Sequential(nn.Linear(fused_dim, self.hidden_dim), nn.ReLU())
        self.pred_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.t_out * self.output_dim),
        )
        self.register_buffer("adj", torch.empty(0))

    def set_edge_index(self, edge_index: torch.Tensor) -> None:
        if edge_index is None:
            dense_adj = torch.eye(self.num_nodes, dtype=torch.float32)
        else:
            dense_adj = pyg_utils.to_dense_adj(edge_index.long(), max_num_nodes=self.num_nodes)[0].float()
            dense_adj.fill_diagonal_(1.0)
        dense_adj = _normalize_dense_adj(dense_adj)
        self.adj = dense_adj
        self.backbone.set_predefined_adj(dense_adj)
        for item in self.multi_scale_heads:
            item["head"].set_static_adj(dense_adj)

    def _format_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        if x.dim() != 4:
            raise ValueError(f"DyHSL expects 4D input, got shape={tuple(x.shape)}")
        if x.shape[1] == self.num_nodes and x.shape[2] == self.t_in:
            x = x.transpose(1, 2).contiguous()
        if x.shape[1] != self.t_in:
            raise ValueError(f"DyHSL expected temporal length {self.t_in}, got {x.shape[1]}")
        if x.shape[2] != self.num_nodes:
            raise ValueError(f"DyHSL expected {self.num_nodes} nodes, got {x.shape[2]}")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._format_input(x)
        bsz, steps, nodes, _ = x.shape

        if self.adj.numel() == 0:
            self.set_edge_index(None)

        feat = self.input_embedding(x)
        time_idx = torch.arange(steps, device=x.device)
        feat = feat + self.time_embedding(time_idx).view(1, steps, 1, self.hidden_dim)
        node_idx = torch.arange(nodes, device=x.device)
        feat = feat + self.node_embedding(node_idx).view(1, 1, nodes, self.hidden_dim)

        feat = self.backbone(feat)

        local_features = []
        global_features = []
        for item in self.multi_scale_heads:
            pooled = item["pool"](feat)
            y = item["head"](pooled)
            local_features.append(y[:, -1, :, :])
            global_features.append(y.mean(dim=1))

        local_feature = self.local_fusion(torch.cat(local_features, dim=-1))
        global_feature = self.global_fusion(torch.cat(global_features, dim=-1))
        fused = torch.cat([local_feature, global_feature], dim=-1)
        pred = self.pred_head(fused)
        pred = pred.view(bsz, nodes, self.t_out, self.output_dim)
        if self.output_dim == 1:
            pred = pred.squeeze(-1)
        return pred
