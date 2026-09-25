from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from model.ST_Net import STNET_pFedCTP
from privacy.baseline_adapters.common import load_state_dict_if_available
from privacy.baseline_adapters.graph_common import edge_index_from_batch, prediction_gradient


class PFedCTPPrivacyAdapter:
    name = "pFedCTP"
    default_attack = "gradient"
    requires_dummy_y = True

    def build_model(self, args: Any, batch: Any) -> nn.Module:
        num_nodes = int(batch.real_x.shape[1])
        edge_index = edge_index_from_batch(batch, num_nodes, args.device)
        model = STNET_pFedCTP(
            num_nodes=num_nodes,
            hidden_dim=args.hidden_dim,
            his_num=args.t_in,
            pred_num=args.t_out,
            message_dim=args.input_dim,
            gcn_layers=1,
            edge_index=edge_index,
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
            raise ValueError("pFedCTP currently supports gradient reconstruction only.")
        was_training = model.training
        model.train()
        try:
            with torch.backends.cudnn.flags(enabled=False):
                pred = model(x)
        finally:
            model.train(was_training)
        return prediction_gradient(model=model, pred=pred, y=y, batch=batch, create_graph=create_graph)
