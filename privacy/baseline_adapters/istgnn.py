from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from model.istgnn import ISTGNN
from privacy.baseline_adapters.common import load_state_dict_if_available
from privacy.baseline_adapters.graph_common import edge_index_from_batch, prediction_gradient


class ISTGNNPrivacyAdapter:
    name = "ISTGNN"
    default_attack = "gradient"
    requires_dummy_y = True

    def build_model(self, args: Any, batch: Any) -> nn.Module:
        num_nodes = int(batch.real_x.shape[1])
        edge_index = edge_index_from_batch(batch, num_nodes, args.device)
        model = ISTGNN(
            edge_index=edge_index,
            in_channels=args.input_dim,
            hidden_channels=args.hidden_dim,
            gru_hidden_size=args.hidden_dim,
            num_nodes=num_nodes,
            pre_len=args.t_out,
        ).to(args.device)
        load_state_dict_if_available(model, args)
        return model

    def compute_leak(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        batch: Any,
        create_graph: bool,
        leak_type: str,
    ):
        if leak_type != "gradient":
            raise ValueError("ISTGNN currently supports gradient reconstruction only.")
        was_training = model.training
        model.train()
        try:
            with torch.backends.cudnn.flags(enabled=False):
                pred = model(x)
        finally:
            model.train(was_training)
        return prediction_gradient(model=model, pred=pred, y=y, batch=batch, create_graph=create_graph)
