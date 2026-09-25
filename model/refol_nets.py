import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter
from torch_geometric.utils import degree

# =============================================================================
# Part 1: Graph Convolution Network (原 AggregationGCN.py 的内容)
# =============================================================================

class GCNConv(MessagePassing):
    def __init__(self):
        super(GCNConv, self).__init__(aggr='add')

    def forward(self, x, edge_index):
        row, col = edge_index
        num_objects = x.size(0)
        deg = degree(col, num_objects, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        weights = scatter(norm, col, dim=0, reduce="sum")
        for i, node in enumerate(col):
            norm[i] /= weights[node]

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=x, norm=norm)
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class AttGCN(torch.nn.Module):
    def __init__(self):
        super(AttGCN, self).__init__()
        self.conv1 = GCNConv()
        self.conv2 = GCNConv()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


# =============================================================================
# Part 2: GRU Prediction Model (原 fl_model.py 的内容)
# =============================================================================

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout, gru_num_layers):
        super().__init__()
        self.decoder = nn.GRU(
            input_size, hidden_size, num_layers=gru_num_layers, dropout=dropout
        )
        self.out_net = nn.Linear(hidden_size, output_size)

    def _empty_attr(self, ref, batch_num, node_num, steps, attr_dim=0):
        if ref is not None:
            return ref.new_zeros((batch_num, steps, node_num, attr_dim))
        return None

    def _flatten_decoder_input(self, target_step, attr_step):
        if attr_step is not None and attr_step.shape[-1] > 0:
            target_step = torch.cat((target_step, attr_step), dim=-1)
        return target_step.permute(1, 0, 2, 3).flatten(1, 2)

    def _autoregressive_decode(self, x_input, y, y_attr, batch_num, node_num):
        horizon = y.shape[1]
        hidden = None
        decoder_input = x_input[-1:]
        outputs = []

        future_attr = y_attr
        if future_attr is None:
            future_attr = self._empty_attr(None, batch_num, node_num, horizon)

        for step in range(horizon):
            out_hidden, hidden = self.decoder(decoder_input, hidden)
            pred_step = self.out_net(out_hidden)
            outputs.append(pred_step)

            if step + 1 < horizon:
                pred_step_view = pred_step.view(1, batch_num, node_num, pred_step.shape[-1]).permute(1, 0, 2, 3)
                attr_step = None
                if future_attr is not None and future_attr.shape[-1] > 0:
                    attr_step = future_attr[:, step:step + 1, :, :]
                decoder_input = self._flatten_decoder_input(pred_step_view, attr_step)

        return torch.cat(outputs, dim=0)

    def forward(self, data):
        # B x T x N x F
        x, x_attr, y, y_attr = data['x'], data['x_attr'], data['y'], data['y_attr']
        batch_num, node_num = x.shape[0], x.shape[2]
        if x_attr is None:
            x_attr = self._empty_attr(x, batch_num, node_num, x.shape[1], attr_dim=0)
        if y_attr is None:
            y_attr = self._empty_attr(y, batch_num, node_num, y.shape[1], attr_dim=0)
        x_input = torch.cat((x, x_attr), dim=-1).permute(1, 0, 2, 3).flatten(1, 2) # T x (B x N) x F

        teacher_forcing = data.get('teacher_forcing', self.training)
        if teacher_forcing:
            y_input = torch.cat((y, y_attr), dim=-1).permute(1, 0, 2, 3).flatten(1, 2)
            y_input = torch.cat((x_input[-1:], y_input[:-1]), dim=0)
            out_hidden, _ = self.decoder(y_input)
        else:
            out_hidden = self._autoregressive_decode(x_input, y, y_attr, batch_num, node_num)

        out = self.out_net(out_hidden) if teacher_forcing else out_hidden
        out = out.view(out.shape[0], batch_num, node_num, out.shape[-1]).permute(1, 0, 2, 3)
        return out
