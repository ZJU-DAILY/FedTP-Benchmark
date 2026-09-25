import torch
import torch.nn as nn


class PageMatrixTransform(nn.Module):
    """
    Lightweight holder for Fed-mSSA page-matrix hyper-parameters.
    The actual page-matrix construction/recovery lives in lib/fedmssa_utils.py.
    """

    def __init__(self, page_length, stride=None):
        super().__init__()
        self.page_length = int(page_length)
        self.stride = int(stride) if stride is not None else int(page_length)

    def forward(self, x):
        return x


class FedmSSAPredictor(nn.Module):
    """
    GRU prediction head for denoised spatio-temporal sequences.
    Input:  [B, N, T_in, F]
    Output: [B, N, T_out, F]
    """

    def __init__(self, input_dim, hidden_dim, out_dim, pre_len, num_layers=2, dropout=0.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.pre_len = int(pre_len)
        self.num_layers = int(num_layers)
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(self.hidden_dim, self.pre_len * self.out_dim)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"FedmSSAPredictor expects [B, N, T, F], got shape={tuple(x.shape)}")
        batch_size, num_nodes, time_steps, feature_dim = x.shape
        if feature_dim != self.input_dim:
            raise ValueError(
                f"FedmSSAPredictor input_dim mismatch: expected {self.input_dim}, got {feature_dim}"
            )
        x = x.reshape(batch_size * num_nodes, time_steps, feature_dim)
        output, _ = self.gru(x)
        output = output[:, -1, :]
        output = self.head(output)
        output = output.view(batch_size, num_nodes, self.pre_len, self.out_dim)
        return output


class FedmSSA_Model(nn.Module):
    """
    Wrapper used by the unified benchmark.
    The low-rank denoising stage is orchestrated by lib/fedmssa_trainer.py.
    """

    def __init__(
        self,
        num_nodes,
        input_dim,
        hidden_dim,
        out_dim,
        pre_len,
        page_length,
        selected_nodes=None,
        num_layers=2,
        dropout=0.0,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.input_dim = int(input_dim)
        self.out_dim = int(out_dim)
        self.pre_len = int(pre_len)
        self.selected_nodes = list(selected_nodes) if selected_nodes is not None else None
        self.page_transform = PageMatrixTransform(page_length=page_length)
        self.predictor = FedmSSAPredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            pre_len=pre_len,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, x):
        return self.predictor(x)
