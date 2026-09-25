import torch
import torch.nn as nn


class TrafficLSTM(nn.Module):
    def __init__(self, num_nodes, t_in, input_size=1, hidden_size=64, output_size=3, output_dim=1):
        super().__init__()
        self.num_nodes = num_nodes
        self.t_in = t_in
        self.output_size = output_size
        self.output_dim = output_dim
        # Shared parameters do not depend on node count, so they work for
        # unequal client partitions as well as the centralized full grid.
        self.spatial_context = nn.Linear(input_size, input_size, bias=False)
        self.spatial_gate_logit = nn.Parameter(torch.tensor(-2.0))
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * output_dim)

    def forward(self, x):
        # Align to [Batch, N, T_in, C].
        if x.dim() == 4:
            if x.shape[1] == self.t_in and x.shape[2] == self.num_nodes:
                x = x.transpose(1, 2).contiguous()
        elif x.dim() == 3:
            if x.shape[1] == self.t_in and x.shape[2] == self.num_nodes:
                x = x.transpose(1, 2).contiguous()
            x = x.unsqueeze(-1)

        B, N, T, C = x.shape
        # A compact traffic context is available over the nodes held by the
        # current job.  Global training therefore sees the full network while
        # local/federated clients remain restricted to their own partitions.
        network_context = x.mean(dim=1, keepdim=True)
        context_delta = network_context - x
        spatial_gain = torch.sigmoid(self.spatial_gate_logit)
        x = x + spatial_gain * self.spatial_context(context_delta)
        x = x.reshape(B * N, T, C)

        out, _ = self.lstm(x)
        out = out[:, -1, :]

        pred = self.fc(out)
        pred = pred.reshape(B, N, self.output_size, self.output_dim)
        if self.output_dim == 1:
            pred = pred.squeeze(-1)

        return pred
