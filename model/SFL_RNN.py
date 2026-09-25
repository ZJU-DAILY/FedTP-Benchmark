"""Plain recurrent base model used by the SFL baseline.

This module intentionally contains no adjacency, graph convolution, attention,
or client identifier embedding.  SFL's server-side structure learning remains
in the protocol; the client predictor itself is a single-layer vanilla RNN.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SFLPureRNN(nn.Module):
    """Node-wise vanilla RNN traffic forecaster.

    Inputs may be ``[B, T, N, F]`` or ``[B, N, T, F]``.  Each node is treated
    as one independent sequence and the final RNN state predicts all horizons.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, t_in: int, t_out: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.t_in = int(t_in)
        self.t_out = int(t_out)
        self.rnn = nn.RNN(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            nonlinearity="tanh",
            batch_first=True,
        )
        self.readout = nn.Linear(self.hidden_dim, self.t_out * self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        if x.dim() != 4:
            raise ValueError(f"SFLPureRNN expects [B,T,N,F] or [B,N,T,F], got {tuple(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"SFLPureRNN input_dim mismatch: expected {self.input_dim}, got {x.shape[-1]}"
            )

        # Dataset loaders normally supply [B, N, T, F]; accept [B, T, N, F]
        # too so the model remains usable by the common privacy adapters.
        if x.shape[1] == self.t_in and x.shape[2] != self.t_in:
            x = x.transpose(1, 2).contiguous()
        batch_size, num_nodes, steps, feature_dim = x.shape
        if steps != self.t_in:
            raise ValueError(f"SFLPureRNN expected t_in={self.t_in}, got {steps}")

        sequences = x.reshape(batch_size * num_nodes, steps, feature_dim)
        outputs, _ = self.rnn(sequences)
        prediction = self.readout(outputs[:, -1, :])
        return prediction.reshape(batch_size, num_nodes, self.t_out, self.output_dim)
