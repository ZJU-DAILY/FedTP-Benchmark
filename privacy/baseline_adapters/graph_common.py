from __future__ import annotations

from typing import Any

import torch

from privacy.baseline_adapters.common import (
    align_prediction_and_target,
    args_from_batch,
    gradient_tuple,
    loss_fn,
)


def edge_index_from_batch(
    batch: Any,
    num_nodes: int,
    device: torch.device | str,
    *,
    self_loops: bool = True,
) -> torch.Tensor:
    edge_index = getattr(batch, "edge_index", None)
    if edge_index is None:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    elif not torch.is_tensor(edge_index):
        edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=device)
    else:
        edge_index = edge_index.to(device=device, dtype=torch.long)

    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    if edge_index.numel() > 0:
        row, col = edge_index[0], edge_index[1]
        valid = (row >= 0) & (row < num_nodes) & (col >= 0) & (col < num_nodes)
        edge_index = torch.stack([row[valid], col[valid]], dim=0)

    if self_loops:
        loop = torch.arange(num_nodes, dtype=torch.long, device=device)
        loops = torch.stack([loop, loop], dim=0)
        edge_index = torch.cat([edge_index, loops], dim=1)

    return edge_index.contiguous()


def prediction_gradient(
    *,
    model: torch.nn.Module,
    pred: torch.Tensor,
    y: torch.Tensor,
    batch: Any,
    create_graph: bool,
) -> tuple[torch.Tensor, ...]:
    args = args_from_batch(batch)
    pred, target = align_prediction_and_target(pred, y)
    loss = loss_fn(args)(pred, target)
    return gradient_tuple(model=model, loss=loss, create_graph=create_graph)


def zero_global_hidden(
    x: torch.Tensor,
    num_nodes: int,
    hidden_dim: int,
) -> torch.Tensor:
    return x.new_zeros(x.shape[0], num_nodes, hidden_dim)
