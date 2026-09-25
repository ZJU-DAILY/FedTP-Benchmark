from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from lib.fca import get_equiconcept_matrix
from model.FCFedGCN import FC_FedGCN_Traffic
from privacy.baseline_adapters.common import (
    align_prediction_and_target,
    args_from_batch,
    deterministic_forward,
    gradient_tuple,
    load_state_dict_if_available,
    loss_fn,
)

try:
    from torch.nn.utils.stateless import functional_call as _functional_call
except ImportError:  # pragma: no cover - depends on the installed torch version.
    from torch.func import functional_call as _functional_call


class FCFedGCNPrivacyAdapter:
    name = "FCFedGCN"
    default_attack = "model_update"
    requires_dummy_y = True

    def build_model(self, args: Any, batch: Any) -> nn.Module:
        num_nodes = int(batch.real_x.shape[1])
        fca_dim = int(getattr(args, "fca_dim", 100))
        edge_index = _local_edge_index_from_batch(batch, num_nodes, args.device)
        adj = _fcfedgcn_adj_from_batch(args, batch, num_nodes, edge_index)
        fca_features = _fcfedgcn_fca_features(edge_index, num_nodes, fca_dim, args.device)
        model = FC_FedGCN_Traffic(
            num_nodes=num_nodes,
            in_dim=args.t_in,
            out_dim=args.t_out,
            hidden_dim=args.hidden_dim,
            fca_dim=fca_dim,
            adj_matrix=adj,
        ).to(args.device)

        model.set_fca_features(fca_features)
        model.dropout = 0.0
        load_state_dict_if_available(model, args)
        return model

    def _loss_with_state(
        self,
        model: nn.Module,
        params: OrderedDict[str, torch.Tensor],
        buffers: OrderedDict[str, torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
        batch: Any,
    ) -> torch.Tensor:
        state = OrderedDict()
        state.update(params)
        state.update(buffers)
        pred = _functional_call(model, state, (x,))
        pred, target = align_prediction_and_target(pred, y)
        return loss_fn(args_from_batch(batch))(pred, target)

    def _differentiable_local_update(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        batch: Any,
        create_graph: bool,
    ) -> tuple[torch.Tensor, ...]:
        args = args_from_batch(batch)
        steps = max(1, int(getattr(args, "local_update_steps", 1)))
        lr = float(getattr(args, "local_update_lr", 1e-3))
        wd_arg = getattr(args, "local_update_wd", None)
        weight_decay = float(getattr(args, "wd", 0.0) if wd_arg is None else wd_arg)

        initial_params = OrderedDict(
            (name, param.detach().clone().requires_grad_(True))
            for name, param in model.named_parameters()
            if param.requires_grad
        )
        params = OrderedDict((name, param) for name, param in initial_params.items())
        buffers = OrderedDict(
            (name, buffer.detach().clone())
            for name, buffer in model.named_buffers()
        )

        for _ in range(steps):
            loss = self._loss_with_state(model, params, buffers, x, y, batch)
            grads = torch.autograd.grad(
                loss,
                tuple(params.values()),
                create_graph=create_graph,
                retain_graph=create_graph,
                allow_unused=True,
            )

            next_params = OrderedDict()
            for (name, param), grad in zip(params.items(), grads):
                if grad is None:
                    grad = torch.zeros_like(param)
                if weight_decay != 0.0:
                    grad = grad + weight_decay * param
                next_params[name] = param - lr * grad
            params = next_params

        return tuple(params[name] - initial_params[name] for name in initial_params.keys())

    def compute_leak(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        batch: Any,
        create_graph: bool,
        leak_type: str,
    ):
        if leak_type == "model_update":
            return self._differentiable_local_update(
                model=model,
                x=x,
                y=y,
                batch=batch,
                create_graph=create_graph,
            )

        if leak_type != "gradient":
            raise ValueError("FCFedGCN currently supports gradient and model_update reconstruction only.")

        pred = deterministic_forward(model, lambda: model(x))
        pred, target = align_prediction_and_target(pred, y)
        loss = loss_fn(args_from_batch(batch))(pred, target)
        return gradient_tuple(model=model, loss=loss, create_graph=create_graph)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_dataset_name(dataset_name: str) -> str:
    return "PeMSD7" if dataset_name == "PeMS07" else dataset_name


def _local_edge_index_from_batch(
    batch: Any,
    num_nodes: int,
    device: torch.device | str,
) -> torch.Tensor | None:
    edge_index = getattr(batch, "edge_index", None)
    if edge_index is None:
        return None
    if not torch.is_tensor(edge_index):
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.long().to(device)
    if edge_index.dim() != 2 or edge_index.shape[0] != 2 or edge_index.numel() == 0:
        return None

    row, col = edge_index[0], edge_index[1]
    valid = (row >= 0) & (row < num_nodes) & (col >= 0) & (col < num_nodes)
    if not torch.any(valid):
        return None
    return torch.stack([row[valid], col[valid]], dim=0)


def _symmetric_normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    adj = adj.clone()
    adj.diagonal().add_(1.0)
    row_sum = adj.sum(dim=1)
    d_inv_sqrt = torch.pow(row_sum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    return d_inv_sqrt.view(-1, 1) * adj * d_inv_sqrt.view(1, -1)


def _edge_index_fallback_adj(
    edge_index: torch.Tensor | None,
    num_nodes: int,
    device: torch.device | str,
) -> torch.Tensor:
    local_adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    if edge_index is not None:
        row, col = edge_index
        local_adj[row, col] = 1.0
        local_adj[col, row] = 1.0
    return _symmetric_normalize_adj(local_adj)


def _read_distance_rows(distance_path: Path) -> list[dict[str, float]]:
    with distance_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"[FCFedGCN] distance.csv is empty: {distance_path}")
        required = {"from", "to", "cost"}
        if not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"[FCFedGCN] distance.csv must contain from/to/cost columns, got {reader.fieldnames}"
            )

        rows = []
        for row in reader:
            rows.append(
                {
                    "from": int(float(row["from"])),
                    "to": int(float(row["to"])),
                    "cost": float(row["cost"]),
                }
            )
        return rows


def _fcfedgcn_adj_from_batch(
    args: Any,
    batch: Any,
    num_nodes: int,
    edge_index: torch.Tensor | None,
) -> torch.Tensor:
    device = args.device
    dataset_name = _normalize_dataset_name(args.dataset_name)
    distance_path = _repo_root() / "data" / dataset_name / "distance.csv"

    if not distance_path.exists():
        return _edge_index_fallback_adj(edge_index, num_nodes, device)

    selected_nodes = [int(node) for node in batch.selected_nodes]
    selected_set = set(selected_nodes)
    node_to_idx = {node: idx for idx, node in enumerate(selected_nodes)}
    rows = _read_distance_rows(distance_path)
    local_rows = [
        row for row in rows
        if row["from"] in selected_set and row["to"] in selected_set and row["from"] != row["to"]
    ]
    if not local_rows:
        return _edge_index_fallback_adj(edge_index, num_nodes, device)

    is_already_weight = max(row["cost"] for row in rows) <= 1.0
    if is_already_weight:
        top_k = 8
        by_src: dict[int, list[dict[str, float]]] = {}
        for row in local_rows:
            by_src.setdefault(int(row["from"]), []).append(row)

        topk_rows: list[dict[str, float]] = []
        for src_rows in by_src.values():
            topk_rows.extend(
                sorted(src_rows, key=lambda row: row["cost"], reverse=True)[:top_k]
            )

        with_reverse = topk_rows + [
            {"from": row["to"], "to": row["from"], "cost": row["cost"]}
            for row in topk_rows
        ]
        deduped: dict[tuple[int, int], dict[str, float]] = {}
        for row in with_reverse:
            deduped[(int(row["from"]), int(row["to"]))] = row
        local_rows = list(deduped.values())

    local_adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    for row in local_rows:
        u = int(row["from"])
        v = int(row["to"])
        cost = float(row["cost"])
        if cost <= 0.0 or u not in node_to_idx or v not in node_to_idx:
            continue
        idx_u = node_to_idx[u]
        idx_v = node_to_idx[v]
        weight = cost if is_already_weight else 1.0 / cost
        local_adj[idx_u, idx_v] = weight
        local_adj[idx_v, idx_u] = weight

    if torch.count_nonzero(local_adj).item() == 0:
        return _edge_index_fallback_adj(edge_index, num_nodes, device)
    return _symmetric_normalize_adj(local_adj)


def _fcfedgcn_fca_features(
    edge_index: torch.Tensor | None,
    num_nodes: int,
    fca_dim: int,
    device: torch.device | str,
) -> torch.Tensor:
    if edge_index is None:
        return torch.zeros(num_nodes, fca_dim, device=device)
    return get_equiconcept_matrix(
        edge_index=edge_index,
        num_nodes=num_nodes,
        device=device,
        max_fca_dim=fca_dim,
    )
