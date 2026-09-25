from __future__ import annotations

from typing import Any, Callable, List

import torch
import torch.nn as nn


def args_from_batch(batch: Any) -> Any:
    args = getattr(batch, "args", None)
    if args is not None:
        return args

    class _DefaultArgs:
        loss_func = "mse"
        input_dim = 1
        output_dim = 1
        t_out = 3

    return _DefaultArgs()


def loss_fn(args: Any) -> nn.Module:
    if getattr(args, "loss_func", "mse") in ("mae", "l1"):
        return nn.L1Loss()
    return nn.MSELoss()


def load_state_dict_if_available(model: nn.Module, args: Any) -> None:
    checkpoint_path = getattr(args, "checkpoint", "")
    if not checkpoint_path:
        return

    checkpoint = torch.load(checkpoint_path, map_location=getattr(args, "device", "cpu"))
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
    # Reconstruction can use a batch-size-one dummy model while a traced
    # training state contains batch-shaped auxiliary parameters (e.g. FUELS
    # W_n).  Load every compatible protocol parameter and safely skip only
    # shape-mismatched auxiliary state.
    model_state = model.state_dict()
    compatible = {
        key: value for key, value in state_dict.items()
        if key in model_state and getattr(value, "shape", None) == model_state[key].shape
    }
    skipped = sorted(set(state_dict) - set(compatible))
    model.load_state_dict(compatible, strict=False)
    if skipped:
        print(
            f"[Privacy] checkpoint loaded compatible tensors={len(compatible)}; "
            f"skipped incompatible/auxiliary tensors={len(skipped)}",
            flush=True,
        )


def gradient_tuple(
    *,
    model: nn.Module,
    loss: torch.Tensor,
    create_graph: bool,
) -> tuple[torch.Tensor, ...]:
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=create_graph,
        retain_graph=create_graph,
        allow_unused=True,
    )

    leak: List[torch.Tensor] = []
    for param, grad in zip(params, grads):
        leak.append(torch.zeros_like(param) if grad is None else grad)
    return tuple(leak)


def align_prediction_and_target(pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pred.dim() == 4 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)
    if y.dim() == 4 and y.shape[-1] == 1:
        y = y.squeeze(-1)

    if pred.shape == y.shape:
        return pred, y

    if pred.dim() == 3 and y.dim() == 3:
        if pred.shape[1] == y.shape[2] and pred.shape[2] == y.shape[1]:
            return pred.transpose(1, 2).contiguous(), y

    if pred.dim() == 4 and y.dim() == 4:
        if pred.shape[1] == y.shape[2] and pred.shape[2] == y.shape[1]:
            return pred.permute(0, 2, 1, 3).contiguous(), y

    return pred.reshape_as(y), y


def btnf_from_bntf(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"Expected 4D tensor [B,N,T,F], got shape={tuple(x.shape)}")
    return x.permute(0, 2, 1, 3).contiguous()


def empty_attr_like(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty(*x.shape[:-1], 0)


def dense_adj_from_batch(batch: Any, num_nodes: int, device: torch.device | str) -> torch.Tensor:
    adj = torch.eye(num_nodes, dtype=torch.float32, device=device)
    edge_index = getattr(batch, "edge_index", None)
    if edge_index is None:
        return adj

    if not torch.is_tensor(edge_index):
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.long().to(device)

    if edge_index.dim() != 2 or edge_index.shape[0] != 2 or edge_index.numel() == 0:
        return adj

    row, col = edge_index[0], edge_index[1]
    valid = (row >= 0) & (row < num_nodes) & (col >= 0) & (col < num_nodes)
    row, col = row[valid], col[valid]
    if row.numel() > 0:
        adj[row, col] = 1.0
        adj[col, row] = 1.0

    row_sum = adj.sum(dim=1, keepdim=True).clamp_min(1.0)
    return adj / row_sum


def deterministic_forward(model: nn.Module, fn: Callable[[], torch.Tensor]) -> torch.Tensor:
    was_training = model.training
    model.eval()
    try:
        return fn()
    finally:
        model.train(was_training)


def fixed_quantized_prediction(prediction: torch.Tensor, args: Any) -> torch.Tensor:
    """Fixed public quantizer for the explicitly revised prediction side channel."""
    bits = max(1, int(getattr(args, "quant_prediction_bits", 4)))
    clip = max(1e-6, float(getattr(args, "quant_prediction_clip", 3.0)))
    levels = float((1 << bits) - 1)
    clipped = prediction.clamp(-clip, clip)
    quantized = torch.round((clipped + clip) * levels / (2.0 * clip)) * (2.0 * clip / levels) - clip
    return prediction + (quantized - prediction).detach() if prediction.requires_grad else quantized


def prediction_sidechannel_tensor(prediction: torch.Tensor, args: Any) -> torch.Tensor:
    """Return the explicitly configured public prediction side-channel value."""
    mode = str(getattr(args, "quant_prediction_summary", "full")).lower()
    if mode == "full":
        return prediction
    if mode == "mean_std":
        if prediction.dim() < 2:
            raise ValueError("mean_std prediction side channel requires a batched prediction tensor.")
        reduce_dims = tuple(range(1, prediction.dim()))
        return torch.stack(
            (
                prediction.mean(dim=reduce_dims),
                prediction.std(dim=reduce_dims, unbiased=False),
            ),
            dim=-1,
        )
    raise ValueError(f"Unsupported quant_prediction_summary: {mode!r}")
