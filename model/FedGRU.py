import torch
import torch.nn as nn


class FedGRU_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, pre_len, num_layers=2):
        super(FedGRU_Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pre_len = pre_len
        self.out_dim = out_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, pre_len * out_dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        if x.dim() != 4:
            raise ValueError(f"FedGRU_Model expects a 4D tensor, got shape={tuple(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"FedGRU_Model input_dim mismatch: expected last dim {self.input_dim}, got {x.shape[-1]}"
            )

        # Support both [B, T, N, F] and [B, N, T, F] layouts.
        if x.shape[1] < x.shape[2]:
            x = x.transpose(1, 2).contiguous()

        batch_size, num_nodes, time_steps, feature_dim = x.shape
        x = x.view(batch_size * num_nodes, time_steps, feature_dim)

        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        out = out.view(batch_size, num_nodes, self.pre_len, self.out_dim)
        return out
