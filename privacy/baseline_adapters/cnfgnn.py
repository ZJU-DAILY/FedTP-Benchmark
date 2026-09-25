from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from model.CNFGNN import CNFGNN
from privacy.baseline_adapters.common import (
    align_prediction_and_target,
    args_from_batch,
    fixed_quantized_prediction,
    gradient_tuple,
    load_state_dict_if_available,
    loss_fn,
    prediction_sidechannel_tensor,
)


class CNFGNNPrivacyAdapter:
    name = "CNFGNN"
    default_attack = "activation"
    requires_dummy_y = True
    attack_surface = "h_encode"
    # The activation reconstruction surface corresponds to the explicit
    # `encoding` upload calibrated in lib/cnfgnn_trainer.py.
    dp_payload_group = "encoding"

    def _local_edge_index_from_batch(
        self,
        batch: Any,
        num_nodes: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        edge_index = getattr(batch, "edge_index", None)
        if edge_index is None:
            idx = torch.arange(num_nodes, dtype=torch.long, device=device)
            return torch.stack([idx, idx], dim=0)

        if not torch.is_tensor(edge_index):
            edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        edge_index = edge_index.long().to(device)
        if edge_index.dim() != 2 or edge_index.shape[0] != 2 or edge_index.numel() == 0:
            idx = torch.arange(num_nodes, dtype=torch.long, device=device)
            return torch.stack([idx, idx], dim=0)

        row, col = edge_index[0], edge_index[1]
        valid = (row >= 0) & (row < num_nodes) & (col >= 0) & (col < num_nodes)
        row, col = row[valid], col[valid]

        loop = torch.arange(num_nodes, dtype=torch.long, device=device)
        row = torch.cat([row, loop])
        col = torch.cat([col, loop])
        return torch.stack([row, col], dim=0).contiguous()

    def build_model(self, args: Any, batch: Any) -> nn.Module:
        num_nodes = int(batch.real_x.shape[1])
        edge_index = self._local_edge_index_from_batch(batch, num_nodes, args.device)
        edge_weight = torch.ones(edge_index.shape[1], 1, dtype=torch.float32, device=args.device)
        model = CNFGNN(
            num_nodes=num_nodes,
            in_dim=args.t_in,
            out_dim=args.t_out,
            hidden_dim=args.hidden_dim,
            edge_index=edge_index,
            edge_weight=edge_weight,
            dropout=0.0,
        ).to(args.device)
        load_state_dict_if_available(model, args)
        return model

    def _server_visible_encoding(self, model: CNFGNN, x: torch.Tensor) -> torch.Tensor:
        was_training = model.training
        model.train()
        try:
            with torch.backends.cudnn.flags(enabled=False):
                h_encode, _ = model.forward_client_encoder(x)
            return h_encode.squeeze(0).contiguous()
        finally:
            model.train(was_training)

    def _local_prediction(self, model: CNFGNN, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with torch.backends.cudnn.flags(enabled=False):
            h_encode, x_reshaped = model.forward_client_encoder(x)
            batch_size = int(x.shape[0])
            h_spatial = model.forward_server_gnn(
                h_encode.squeeze(0),
                batch_size=batch_size,
                total_nodes=model.num_nodes,
            )
            return model.forward_client_decoder(x_reshaped, y, h_encode, h_spatial)

    def compute_leak(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        batch: Any,
        create_graph: bool,
        leak_type: str,
    ):
        if leak_type == "activation":
            if bool(getattr(args_from_batch(batch), "quantized_prediction_sidechannel", False)):
                # Decoder teacher forcing is disabled for the serving-side
                # prediction message; the zero tensor is a public start token.
                args = args_from_batch(batch)
                prediction = self._local_prediction(model, x, torch.zeros_like(y))
                return fixed_quantized_prediction(
                    prediction_sidechannel_tensor(prediction, args), args
                )
            return self._server_visible_encoding(model, x)

        if leak_type != "gradient":
            raise ValueError("CNFGNN currently supports activation and gradient reconstruction only.")

        args = args_from_batch(batch)
        pred = self._local_prediction(model, x, y)
        pred, target = align_prediction_and_target(pred, y)
        loss = loss_fn(args)(pred, target)
        return gradient_tuple(model=model.client_model, loss=loss, create_graph=create_graph)
