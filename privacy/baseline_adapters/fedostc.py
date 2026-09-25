from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from model.FedOSTC import FedOSTC
from privacy.baseline_adapters.common import (
    align_prediction_and_target,
    args_from_batch,
    fixed_quantized_prediction,
    load_state_dict_if_available,
    loss_fn,
)
from privacy.baseline_adapters.graph_common import edge_index_from_batch


class FedOSTCPrivacyAdapter:
    name = "FedOSTC"
    default_attack = "activation"
    requires_dummy_y = True
    attack_surface = "h_time+h_spatio+g_spatio"
    activation_requires_grad = True
    # FedOSTC calibrates these two client-to-Arbiter messages independently.
    # `h_spatio` is an Arbiter-internal tensor, so it is deliberately excluded
    # from the DP reconstruction observation.
    dp_payload_group_keys = ("h_time", "g_spatio")

    def build_model(self, args: Any, batch: Any) -> nn.Module:
        num_nodes = int(batch.real_x.shape[1])
        edge_index = edge_index_from_batch(batch, num_nodes, args.device)
        model = FedOSTC(
            enc_dim=args.hidden_dim,
            gat_dim=args.hidden_dim,
            pred_steps=args.t_out,
        ).to(args.device)
        model.edge_index = edge_index
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
                # These are the three tensors visible to the FedOSTC server in
                # one split-learning step: client h_time, server h_spatio, and
                # client g_spatio returned after local decoding.
                h_time = model.forward_encoder(x)
                h_spatio = model.forward_server_gat(h_time, model.edge_index)
                pred = model.forward_decoder(h_spatio)
                pred, target = align_prediction_and_target(pred, y)
                split_loss = loss_fn(args_from_batch(batch))(pred, target)
                g_spatio = torch.autograd.grad(
                    split_loss,
                    h_spatio,
                    create_graph=create_graph,
                    retain_graph=create_graph,
                    allow_unused=False,
                )[0]
        finally:
            model.train(was_training)
        if leak_type == "activation":
            if bool(getattr(args_from_batch(batch), "quantized_prediction_sidechannel", False)):
                return fixed_quantized_prediction(pred, args_from_batch(batch))
            if str(getattr(args_from_batch(batch), "protection", "plain")).lower() == "dp":
                return {
                    "h_time": h_time,
                    "g_spatio": g_spatio,
                }
            return {
                "h_time": h_time,
                "h_spatio": h_spatio,
                "g_spatio": g_spatio,
            }

        if leak_type != "gradient":
            raise ValueError("FedOSTC currently supports activation and gradient reconstruction only.")

        raise ValueError(
            "FedOSTC privacy reconstruction is defined on the server-visible "
            "activation triplet h_time+h_spatio+g_spatio."
        )
