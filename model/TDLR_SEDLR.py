import torch
import torch.nn as nn


class StreamingTrafficLSTM(nn.Module):
    def __init__(
        self,
        num_nodes,
        t_in,
        input_size,
        hidden_size,
        output_size,
        output_dim=1,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.t_in = t_in
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_dim = output_dim

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, output_size * output_dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        if x.dim() != 4:
            raise ValueError(
                f"StreamingTrafficLSTM expects [B, T, N, C] or [B, N, T, C], got {tuple(x.shape)}"
            )

        if x.shape[1] == self.num_nodes and x.shape[2] == self.t_in:
            x = x.transpose(1, 2).contiguous()
        elif x.shape[1] != self.t_in and x.shape[2] == self.t_in:
            x = x.transpose(1, 2).contiguous()

        batch_size, time_steps, num_nodes, feature_dim = x.shape
        if time_steps != self.t_in:
            raise ValueError(f"Expected t_in={self.t_in}, got {time_steps}")
        if feature_dim != self.input_size:
            raise ValueError(f"Expected input_size={self.input_size}, got {feature_dim}")

        seq = x.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, time_steps, feature_dim)
        _, (hidden, _) = self.lstm(seq)
        last_hidden = hidden[-1]
        out = self.head(last_hidden)
        out = out.view(batch_size, num_nodes, self.output_size, self.output_dim)
        out = out.permute(0, 2, 1, 3).contiguous()

        if self.output_dim == 1:
            out = out.squeeze(-1)
        return out
