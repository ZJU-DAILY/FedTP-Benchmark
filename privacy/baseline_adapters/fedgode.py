from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from model.FedGODE import ODEGCN
from privacy.baseline_adapters.common import args_from_batch, dense_adj_from_batch, fixed_quantized_prediction, load_state_dict_if_available
from privacy.baseline_adapters.graph_common import prediction_gradient


class FedGODEPrivacyAdapter:
    name = "FedGODE"
    default_attack = "gradient"
    requires_dummy_y = True

    def build_model(self, args: Any, batch: Any) -> nn.Module:
        num_nodes = int(batch.real_x.shape[1])
        adj = dense_adj_from_batch(batch, num_nodes, args.device)
        model = ODEGCN(
            num_nodes=num_nodes,
            num_features=args.input_dim,
            num_timesteps_input=args.t_in,
            num_timesteps_output=args.t_out,
            A_sp_hat=adj,
            A_se_hat=adj,
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
        if leak_type == "activation" and bool(getattr(args_from_batch(batch), "quantized_prediction_sidechannel", False)):
            return fixed_quantized_prediction(model(x), args_from_batch(batch))
        if leak_type != "gradient":
            raise ValueError("FedGODE currently supports gradient reconstruction only.")
        pred = model(x)
        return prediction_gradient(model=model, pred=pred, y=y, batch=batch, create_graph=create_graph)
