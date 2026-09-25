import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils


class DiffusionGraphConv(nn.Module):
    def __init__(self, channels, dropout=0.0, support_len=1, order=2):
        super().__init__()
        self.channels = int(channels)
        self.support_len = int(support_len)
        self.order = int(order)
        input_channels = (self.support_len * self.order + 1) * self.channels
        self.proj = nn.Conv2d(input_channels, self.channels, kernel_size=(1, 1))
        self.dropout = float(dropout)

    def forward(self, x, supports):
        out = [x]
        for support in supports:
            x1 = torch.einsum("nm,bcmt->bcnt", support, x)
            out.append(x1)
            xk = x1
            for _ in range(2, self.order + 1):
                xk = torch.einsum("nm,bcmt->bcnt", support, xk)
                out.append(xk)
        h = torch.cat(out, dim=1)
        h = self.proj(h)
        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class LightGraphWaveNet(nn.Module):
    """
    Lightweight GraphWaveNet-style backbone for UFCL.
    The channel defaults are chosen so the total trainable parameters stay
    close to the historical UFCL/FedGRU scale (~38k) rather than the much
    larger original GraphWaveNet configurations.
    """

    def __init__(
        self,
        num_nodes,
        input_dim,
        output_dim,
        horizon,
        residual_channels=29,
        skip_channels=58,
        end_channels=58,
        blocks=2,
        layers=2,
        kernel_size=2,
        dropout=0.1,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.horizon = int(horizon)
        self.residual_channels = int(residual_channels)
        self.skip_channels = int(skip_channels)
        self.end_channels = int(end_channels)
        self.blocks = int(blocks)
        self.layers = int(layers)
        self.kernel_size = int(kernel_size)

        self.start_conv = nn.Conv2d(self.input_dim, self.residual_channels, kernel_size=(1, 1))

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.graph_convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        receptive_field = 1
        dilation = 1
        for _ in range(self.blocks):
            for _ in range(self.layers):
                self.filter_convs.append(
                    nn.Conv2d(
                        self.residual_channels,
                        self.residual_channels,
                        kernel_size=(1, self.kernel_size),
                        dilation=(1, dilation),
                    )
                )
                self.gate_convs.append(
                    nn.Conv2d(
                        self.residual_channels,
                        self.residual_channels,
                        kernel_size=(1, self.kernel_size),
                        dilation=(1, dilation),
                    )
                )
                self.residual_convs.append(
                    nn.Conv2d(self.residual_channels, self.residual_channels, kernel_size=(1, 1))
                )
                self.skip_convs.append(
                    nn.Conv2d(self.residual_channels, self.skip_channels, kernel_size=(1, 1))
                )
                self.graph_convs.append(
                    DiffusionGraphConv(self.residual_channels, dropout=dropout, support_len=1, order=2)
                )
                self.batch_norms.append(nn.BatchNorm2d(self.residual_channels))
                receptive_field += (self.kernel_size - 1) * dilation
                dilation *= 2

        self.receptive_field = receptive_field
        self.end_conv_1 = nn.Conv2d(self.skip_channels, self.end_channels, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(self.end_channels, self.horizon * self.output_dim, kernel_size=(1, 1))

        self.register_buffer("supports", torch.empty(0))

    def set_edge_index(self, edge_index):
        if edge_index is None:
            self.supports = torch.eye(self.num_nodes, dtype=torch.float32).unsqueeze(0)
            return
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.LongTensor(edge_index)
        edge_index = edge_index.long()
        dense_adj = pyg_utils.to_dense_adj(edge_index, max_num_nodes=self.num_nodes)[0].float()
        dense_adj.fill_diagonal_(1.0)
        row_sum = dense_adj.sum(dim=1, keepdim=True).clamp_min(1.0)
        support = dense_adj / row_sum
        self.supports = support.unsqueeze(0)

    def _format_input(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        if x.dim() != 4:
            raise ValueError(f"LightGraphWaveNet expects a 4D tensor, got shape={tuple(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"LightGraphWaveNet input_dim mismatch: expected last dim {self.input_dim}, got {x.shape[-1]}"
            )

        # Input data can be laid out as either [B, N, T, F] or [B, T, N, F].
        # Do not infer the layout by comparing N and T: with 32 clients on
        # PeMS04, local N is 9/10 while t_in is 12, which incorrectly treated
        # the time axis as the node axis.  The model's local node count is the
        # unambiguous discriminator.
        if x.shape[1] == self.num_nodes:
            pass
        elif x.shape[2] == self.num_nodes:
            x = x.transpose(1, 2).contiguous()
        else:
            raise ValueError(
                "LightGraphWaveNet cannot identify the node dimension: "
                f"expected {self.num_nodes} nodes, got input shape={tuple(x.shape)}"
            )
        return x

    def forward(self, x):
        x = self._format_input(x)
        batch_size, num_nodes, time_steps, _ = x.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"LightGraphWaveNet expected {self.num_nodes} nodes, got {num_nodes}")

        x = x.permute(0, 3, 1, 2).contiguous()
        if time_steps < self.receptive_field:
            x = F.pad(x, (self.receptive_field - time_steps, 0, 0, 0))

        if self.supports.numel() == 0:
            device_support = torch.eye(self.num_nodes, dtype=x.dtype, device=x.device)
            supports = [device_support]
        else:
            supports = [self.supports[0].to(device=x.device, dtype=x.dtype)]

        x = self.start_conv(x)
        skip = None
        for layer_idx in range(len(self.filter_convs)):
            residual = x
            filter_out = torch.tanh(self.filter_convs[layer_idx](residual))
            gate_out = torch.sigmoid(self.gate_convs[layer_idx](residual))
            x = filter_out * gate_out

            skip_part = self.skip_convs[layer_idx](x)
            skip = skip_part if skip is None else skip[..., -skip_part.size(-1):] + skip_part

            x = self.graph_convs[layer_idx](x, supports)
            x = self.residual_convs[layer_idx](x)
            x = x + residual[..., -x.size(-1):]
            x = self.batch_norms[layer_idx](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        x = x[..., -1]
        x = x.view(batch_size, self.horizon, self.output_dim, self.num_nodes)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
