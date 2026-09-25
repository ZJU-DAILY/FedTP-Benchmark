from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn

from lib.custom_loss import FedAGATLoss
from model.AGAT import ASTGAT
from privacy.baseline_adapters.common import (
    align_prediction_and_target,
    args_from_batch,
    gradient_tuple,
    fixed_quantized_prediction,
    load_state_dict_if_available,
)

try:
    from torch.func import functional_call as _functional_call
except ImportError:  # pragma: no cover - depends on the installed torch version.
    from torch.nn.utils.stateless import functional_call as _functional_call


class FedAGATPrivacyAdapter:
    name = "FedAGAT"
    default_attack = "model_update"
    requires_dummy_y = True

    def build_model(self, args: Any, batch: Any) -> nn.Module:
        num_nodes = int(batch.real_x.shape[1])
        selected_nodes = _selected_node_ids(batch, num_nodes, args.device)
        # The federated trainer allocates ASTGAT's ID-indexed parameter matrix
        # over every global node slot, not merely this client's subgraph.
        # Reconstructing with a client-local maximum made
        # generator.param_matrix incompatible with the captured HE trace.
        all_client_nodes = getattr(batch, "nodes_per", None) or []
        global_slots = [max(nodes) + 1 for nodes in all_client_nodes if nodes]
        max_nodes = max(num_nodes, int(selected_nodes.max().item()) + 1, *(global_slots or [0]))
        adj = _fedagat_adj_from_batch(batch, num_nodes, args.device)

        model = ASTGAT(
            num_nodes=num_nodes,
            in_dim=args.t_in,
            pred_len=args.t_out,
            adj=adj,
            emb_dim=args.hidden_dim,
            max_nodes=max_nodes,
            node_ids=selected_nodes,
        ).to(args.device)
        load_state_dict_if_available(model, args)
        return model

    def _base_loss_fn(self, batch: Any) -> nn.Module:
        args = args_from_batch(batch)
        loss_name = getattr(args, "loss_func", "mse")
        if loss_name in ("mae", "l1"):
            return nn.L1Loss()
        return nn.MSELoss()

    def _make_loss(self, batch: Any, step_idx: int) -> FedAGATLoss:
        args = args_from_batch(batch)
        loss = FedAGATLoss(
            base_loss_func=self._base_loss_fn(batch),
            max_epochs=max(1, int(getattr(args, "fedagat_max_epochs", 5))),
        ).to(getattr(args, "device", "cpu"))
        batches_per_epoch = int(getattr(args, "fedagat_batches_per_epoch", 0))
        loss.batches_per_epoch = max(1, batches_per_epoch)
        start_step = int(getattr(args, "fedagat_loss_start_step", 0))
        loss.batch_count = start_step + step_idx
        return loss

    def _forward_with_state(
        self,
        model: nn.Module,
        params: OrderedDict[str, torch.Tensor],
        buffers: OrderedDict[str, torch.Tensor],
        x: torch.Tensor,
        rng_seed: int,
    ):
        state = OrderedDict()
        state.update(params)
        state.update(buffers)

        was_training = model.training
        model.train(True)
        try:
            return _fedagat_seeded_forward(
                lambda: _functional_call(model, state, (x,)),
                x.device,
                rng_seed,
            )
        finally:
            model.train(was_training)

    def _task_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        batch: Any,
        *,
        step_idx: int,
    ) -> torch.Tensor:
        output = _fedagat_seeded_forward(
            lambda: self._forward_training(model, x),
            x.device,
            _fedagat_step_seed(args_from_batch(batch), step_idx),
        )
        return self._loss_from_output(output, y, batch, step_idx=step_idx)

    def _forward_training(self, model: nn.Module, x: torch.Tensor):
        was_training = model.training
        model.train(True)
        try:
            return model(x)
        finally:
            model.train(was_training)

    def _loss_from_output(
        self,
        output,
        y: torch.Tensor,
        batch: Any,
        *,
        step_idx: int,
    ) -> torch.Tensor:
        pred = output[0] if isinstance(output, tuple) else output
        pred, target = align_prediction_and_target(pred, y)
        loss_input = (pred,) + tuple(output[1:]) if isinstance(output, tuple) else pred
        return self._make_loss(batch, step_idx)(loss_input, target)

    def _loss_with_state(
        self,
        model: nn.Module,
        params: OrderedDict[str, torch.Tensor],
        buffers: OrderedDict[str, torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
        batch: Any,
        *,
        step_idx: int,
    ) -> torch.Tensor:
        output = self._forward_with_state(
            model,
            params,
            buffers,
            x,
            rng_seed=_fedagat_step_seed(args_from_batch(batch), step_idx),
        )
        return self._loss_from_output(output, y, batch, step_idx=step_idx)

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
        update_optimizer = getattr(args, "fedagat_update_optimizer", "adam").lower()

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
        exp_avg = OrderedDict((name, torch.zeros_like(param)) for name, param in params.items())
        exp_avg_sq = OrderedDict((name, torch.zeros_like(param)) for name, param in params.items())

        for step_idx in range(steps):
            loss = self._loss_with_state(
                model,
                params,
                buffers,
                x,
                y,
                batch,
                step_idx=step_idx,
            )
            grads = torch.autograd.grad(
                loss,
                tuple(params.values()),
                create_graph=create_graph,
                retain_graph=create_graph,
                allow_unused=True,
            )

            if update_optimizer == "adam":
                params, exp_avg, exp_avg_sq = self._adam_step(
                    params=params,
                    grads=grads,
                    exp_avg=exp_avg,
                    exp_avg_sq=exp_avg_sq,
                    step_no=step_idx + 1,
                    lr=lr,
                    weight_decay=weight_decay,
                    args=args,
                )
            elif update_optimizer == "sgd":
                params = self._sgd_step(
                    params=params,
                    grads=grads,
                    lr=lr,
                    weight_decay=weight_decay,
                )
            else:
                raise ValueError(f"Unsupported fedagat_update_optimizer: {update_optimizer}")

        return tuple(params[name] - initial_params[name] for name in initial_params.keys())

    def _sgd_step(
        self,
        params: OrderedDict[str, torch.Tensor],
        grads: tuple[torch.Tensor | None, ...],
        *,
        lr: float,
        weight_decay: float,
    ) -> OrderedDict[str, torch.Tensor]:
        next_params = OrderedDict()
        for (name, param), grad in zip(params.items(), grads):
            if grad is None:
                grad = torch.zeros_like(param)
            if weight_decay != 0.0:
                grad = grad + weight_decay * param
            next_params[name] = param - lr * grad
        return next_params

    def _adam_step(
        self,
        params: OrderedDict[str, torch.Tensor],
        grads: tuple[torch.Tensor | None, ...],
        exp_avg: OrderedDict[str, torch.Tensor],
        exp_avg_sq: OrderedDict[str, torch.Tensor],
        *,
        step_no: int,
        lr: float,
        weight_decay: float,
        args: Any,
    ) -> tuple[OrderedDict[str, torch.Tensor], OrderedDict[str, torch.Tensor], OrderedDict[str, torch.Tensor]]:
        beta1 = float(getattr(args, "fedagat_adam_beta1", 0.9))
        beta2 = float(getattr(args, "fedagat_adam_beta2", 0.999))
        eps = float(getattr(args, "fedagat_adam_eps", 1e-8))
        bias_correction1 = 1.0 - beta1 ** step_no
        bias_correction2 = 1.0 - beta2 ** step_no

        next_params = OrderedDict()
        next_exp_avg = OrderedDict()
        next_exp_avg_sq = OrderedDict()
        for (name, param), grad in zip(params.items(), grads):
            if grad is None:
                grad = torch.zeros_like(param)
            if weight_decay != 0.0:
                grad = grad + weight_decay * param

            m = beta1 * exp_avg[name] + (1.0 - beta1) * grad
            v = beta2 * exp_avg_sq[name] + (1.0 - beta2) * grad.pow(2)
            step_size = lr / bias_correction1
            denom = v.sqrt() / math.sqrt(bias_correction2) + eps

            next_params[name] = param - step_size * m / denom
            next_exp_avg[name] = m
            next_exp_avg_sq[name] = v
        return next_params, next_exp_avg, next_exp_avg_sq

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
            output = _fedagat_seeded_forward(
                lambda: self._forward_training(model, x), x.device, _fedagat_step_seed(args_from_batch(batch), 0)
            )
            prediction = output[0] if isinstance(output, tuple) else output
            return fixed_quantized_prediction(prediction, args_from_batch(batch))
        if leak_type == "model_update":
            return self._differentiable_local_update(
                model=model,
                x=x,
                y=y,
                batch=batch,
                create_graph=create_graph,
            )

        if leak_type != "gradient":
            raise ValueError("FedAGAT currently supports gradient and model_update reconstruction only.")

        loss = self._task_loss(model, x, y, batch, step_idx=0)
        return gradient_tuple(model=model, loss=loss, create_graph=create_graph)


def _selected_node_ids(batch: Any, num_nodes: int, device: torch.device | str) -> torch.Tensor:
    selected_nodes = getattr(batch, "selected_nodes", None)
    if selected_nodes is None:
        return torch.arange(num_nodes, dtype=torch.long, device=device)
    node_ids = torch.as_tensor(selected_nodes, dtype=torch.long, device=device)
    if node_ids.numel() != num_nodes:
        return torch.arange(num_nodes, dtype=torch.long, device=device)
    return node_ids


def _fedagat_adj_from_batch(batch: Any, num_nodes: int, device: torch.device | str) -> torch.Tensor:
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
    return adj


def _fedagat_step_seed(args: Any, step_idx: int) -> int:
    return int(getattr(args, "seed", 0)) + 1009 * int(step_idx)


def _fedagat_seeded_forward(fn, device: torch.device, seed: int):
    device_list: list[int] = []
    if device.type == "cuda" and device.index is not None:
        device_list = [device.index]
    with torch.random.fork_rng(devices=device_list):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        return fn()
