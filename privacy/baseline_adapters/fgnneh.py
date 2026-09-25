from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from lib.fgnneh_algo import FGNNEH_Backbone
from model.FGNNEH import FGNNEH_Client


def _dense_adj_from_edge_index(edge_index: Any, num_nodes: int, device: str) -> torch.Tensor:
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    if edge_index is None:
        return adj

    if not torch.is_tensor(edge_index):
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.to(device=device, dtype=torch.long)
    if edge_index.numel() == 0:
        return adj

    row, col = edge_index[0], edge_index[1]
    mask = (row >= 0) & (row < num_nodes) & (col >= 0) & (col < num_nodes)
    if mask.any():
        adj[row[mask], col[mask]] = 1.0
        adj[col[mask], row[mask]] = 1.0
    return adj


class FGNNEHPrivacyAdapter:
    name = "FGNNEH"
    default_attack = "activation"
    requires_dummy_y = False

    @staticmethod
    def _observed_hypernode_view(hypernode: torch.Tensor, args: Any) -> torch.Tensor:
        """Return only the declared, deterministic prefix disclosure."""
        ratio = float(getattr(args, "fgnneh_hypernode_leak_ratio", 1.0))
        if not 0.0 < ratio <= 1.0:
            raise ValueError("--fgnneh_hypernode_leak_ratio must be in (0, 1].")
        count = max(1, min(hypernode.shape[-1], int(round(hypernode.shape[-1] * ratio))))
        return hypernode[..., :count]

    @staticmethod
    def _quantized_prediction_view(prediction: torch.Tensor, args: Any) -> torch.Tensor:
        bits = int(getattr(args, "fgnneh_quant_prediction_bits", 4))
        clip = float(getattr(args, "fgnneh_quant_prediction_clip", 3.0))
        if bits < 2 or clip <= 0:
            raise ValueError("FGNNEH quantized-prediction settings require bits>=2 and clip>0.")
        levels = float((1 << (bits - 1)) - 1)
        quantized = (prediction.clamp(-clip, clip) * (levels / clip)).round().mul(clip / levels)
        # Straight-through derivative: server sees the rounded forward value,
        # while inversion can optimize through the declared quantizer.
        return prediction + (quantized - prediction).detach() if prediction.requires_grad else quantized

    def build_model(self, args: Any, batch: Any) -> nn.Module:
        num_nodes = batch.real_x.shape[1]
        adj_dense = _dense_adj_from_edge_index(batch.edge_index, num_nodes, args.device)
        backbone_extractor = FGNNEH_Backbone(
            adj_matrix=adj_dense,
            P=args.fgnneh_P,
            gamma=args.fgnneh_gamma,
            n_components=args.fgnneh_n_components,
            device=args.device,
        )
        model = FGNNEH_Client(
            num_nodes=num_nodes,
            in_dim=args.t_in * args.input_dim,
            out_dim=args.t_out,
            hidden_dim=args.hidden_dim,
            backbone_extractor=backbone_extractor,
        ).to(args.device)

        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=args.device)
            if isinstance(checkpoint, dict):
                state_dict = (
                    checkpoint.get("state_dict")
                    or checkpoint.get("model_state_dict")
                    or checkpoint.get("model")
                    or checkpoint
                )
            else:
                state_dict = checkpoint
            if isinstance(state_dict, dict):
                state_dict = {
                    key.replace("module.", "", 1): value
                    for key, value in state_dict.items()
                }
            model.load_state_dict(state_dict, strict=False)

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
        if leak_type != "activation":
            raise ValueError("FGNNEH currently supports activation reconstruction only.")

        h, h_backbone = model.forward_local(x, model.backbone_extractor.adj)
        scope = getattr(args_from_batch(batch), "fgnneh_activation_scope", "hypernode")
        if scope == "backbone":
            return h_backbone
        if scope == "quantized_prediction":
            context = getattr(batch, "fgnneh_server_context", None)
            if not torch.is_tensor(context):
                raise ValueError("FGNNEH quantized-prediction attack requires server-visible context from its trace.")
            prediction = model.forward_predict(h, context.to(x.device, dtype=x.dtype))
            if prediction.dim() == 4 and prediction.shape[-1] == 1:
                prediction = prediction.squeeze(-1)
            return (self._quantized_prediction_view(prediction, args_from_batch(batch)), context.to(x.device, dtype=x.dtype))
        if scope != "hypernode":
            raise ValueError(f"Unsupported FGNNEH activation scope: {scope}")
        return self._observed_hypernode_view(
            model.generate_hypernode(h_backbone), args_from_batch(batch)
        )


def args_from_batch(batch: Any) -> Any:
    return getattr(batch, "args", None)
