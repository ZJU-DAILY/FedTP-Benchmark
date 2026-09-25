from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from model.FedSTG import FedSTG_Client
from privacy.baseline_adapters.common import args_from_batch, fixed_quantized_prediction, load_state_dict_if_available
from privacy.baseline_adapters.graph_common import prediction_gradient, zero_global_hidden


class FedSTGPrivacyAdapter:
    name = "FedSTG"
    default_attack = "activation"
    requires_dummy_y = True
    # These names deliberately match the payload types calibrated by the
    # federated FedSTG trainer.  h_tau and z_tau have separate DP clipping
    # bounds, so they must not be concatenated into one artificial upload.
    dp_payload_group_keys = ("htau", "ztau")

    def build_model(self, args: Any, batch: Any) -> nn.Module:
        num_nodes = int(batch.real_x.shape[1])
        model = FedSTG_Client(
            in_dim=args.input_dim,
            out_dim=args.t_out,
            hidden_dim=args.hidden_dim,
            K=getattr(args, "fedstg_K", 10),
            d=getattr(args, "fedstg_d", getattr(args, "node_emb_dim", 32)),
            seq_len=args.t_in,
            num_nodes=num_nodes,
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
        was_training = model.training
        model.train()
        try:
            with torch.backends.cudnn.flags(enabled=False):
                h_tau, z_tau, _ = model.forward_encode(x)
        finally:
            model.train(was_training)
        if leak_type == "activation":
            if bool(getattr(args_from_batch(batch), "quantized_prediction_sidechannel", False)):
                h_g = zero_global_hidden(x, model.num_nodes, model.hidden_dim)
                return fixed_quantized_prediction(model.forward_predict(h_tau, z_tau, h_g), args_from_batch(batch))
            # fate_main.py uploads h_tau directly but uploads the temporal
            # pattern embedding after averaging its batch axis.  Reproduce
            # that exact server-visible surface here; returning raw z_tau
            # would attack information that the DP server never receives.
            return {
                "htau": h_tau,
                "ztau": z_tau.mean(dim=0),
            }

        if leak_type != "gradient":
            raise ValueError("FedSTG currently supports activation and gradient reconstruction only.")

        h_g = zero_global_hidden(x, model.num_nodes, model.hidden_dim)
        pred = model.forward_predict(h_tau, z_tau, h_g)
        return prediction_gradient(model=model, pred=pred, y=y, batch=batch, create_graph=create_graph)
