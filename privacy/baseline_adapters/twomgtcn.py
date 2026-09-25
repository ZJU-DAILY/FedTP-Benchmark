from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from privacy.baseline_adapters.common import (
    args_from_batch,
    btnf_from_bntf,
    dense_adj_from_batch,
    deterministic_forward,
    gradient_tuple,
    load_state_dict_if_available,
    loss_fn,
)

try:
    from torch.func import functional_call as _functional_call
except ImportError:  # pragma: no cover - depends on the installed torch version.
    from torch.nn.utils.stateless import functional_call as _functional_call


def _feature_factorization_loss(features: torch.Tensor) -> torch.Tensor:
    batch_nodes = features.shape[0] * features.shape[1]
    hidden_dim = features.shape[2]
    reshaped = features.reshape(batch_nodes, hidden_dim)
    centered = reshaped - reshaped.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, keepdim=True, unbiased=False) + 1e-6)
    normalized = centered / std
    corr = normalized.T.matmul(normalized) / max(batch_nodes - 1, 1)
    eye = torch.eye(hidden_dim, dtype=features.dtype, device=features.device)
    return 0.5 * torch.norm(corr - eye, p="fro") ** 2


class TwoMGTCNPrivacyAdapter:
    name = "TwoMGTCN"
    default_attack = "model_update"
    requires_dummy_y = True

    def _adj_from_batch(self, batch: Any, num_nodes: int, device: torch.device | str) -> torch.Tensor:
        dataset = getattr(batch, "dataset", None)
        adj = getattr(dataset, "adj", None)
        if adj is not None:
            adj = torch.as_tensor(adj, dtype=torch.float32, device=device)
            if adj.dim() == 2 and adj.shape[0] == num_nodes and adj.shape[1] == num_nodes:
                row_sum = adj.sum(dim=1, keepdim=True).clamp_min(1.0)
                return adj / row_sum
        return dense_adj_from_batch(batch, num_nodes, device)

    def build_model(self, args: Any, batch: Any) -> nn.Module:
        from model.TwoMGTCN import TwoMGTCN

        num_nodes = int(batch.real_x.shape[1])
        adj = self._adj_from_batch(batch, num_nodes, args.device)
        use_ext = bool(int(getattr(args, "twomgtcn_use_ext", 0)))
        if use_ext and getattr(batch, "x_ext", None) is not None:
            ext_dim = int(batch.x_ext.shape[-1])
        else:
            ext_dim = 0
        model = TwoMGTCN(
            A=adj,
            T_in=args.t_in,
            T_out=args.t_out,
            hidden_size=args.hidden_dim,
            num_layers=3,
            nb_flow=args.input_dim,
            ext_dim=ext_dim,
        ).to(args.device)
        load_state_dict_if_available(model, args)
        return model

    def _x_ext_for_model(self, model: nn.Module, x: torch.Tensor, batch: Any, args: Any) -> torch.Tensor:
        ext_dim = int(getattr(model, "ext_dim", getattr(args, "ext_dim", 0)))
        if ext_dim <= 0:
            return x.new_zeros(x.shape[0], args.t_out, 0)

        use_ext = bool(int(getattr(args, "twomgtcn_use_ext", 0)))
        x_ext = getattr(batch, "x_ext", None) if use_ext else None
        if x_ext is None:
            return x.new_zeros(x.shape[0], args.t_out, ext_dim)
        return x_ext.to(device=x.device, dtype=x.dtype)

    def _task_loss_with_state(
        self,
        model: nn.Module,
        params: OrderedDict[str, torch.Tensor],
        buffers: OrderedDict[str, torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
        batch: Any,
    ) -> torch.Tensor:
        args = args_from_batch(batch)
        x_flow = btnf_from_bntf(x)
        y_target = btnf_from_bntf(y)
        x_ext = self._x_ext_for_model(model, x, batch, args)

        state = OrderedDict()
        state.update(params)
        state.update(buffers)

        was_training = model.training
        model.eval()
        try:
            with torch.backends.cudnn.flags(enabled=False):
                pred, fused_feat = _functional_call(model, state, (x_flow, x_ext))
        finally:
            model.train(was_training)

        loss = loss_fn(args)(pred, y_target)
        lfac_alpha = float(getattr(args, "twomgtcn_lfac_alpha", 0.01))
        if lfac_alpha != 0.0:
            loss = loss + lfac_alpha * _feature_factorization_loss(fused_feat)
        return loss

    def _weighted(self, tensor: torch.Tensor, weight: float) -> torch.Tensor:
        return tensor * math.sqrt(max(0.0, float(weight)))

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
        beta1 = float(getattr(args, "twomgtcn_adam_beta1", 0.9))
        beta2 = float(getattr(args, "twomgtcn_adam_beta2", 0.999))
        eps = float(getattr(args, "twomgtcn_adam_eps", 1e-8))
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

    def _fpass_weight_vector(self, params: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
        flat = torch.cat([param.reshape(-1) for param in params.values()])
        return F.normalize(flat, p=2, dim=0)

    def _differentiable_local_update(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        batch: Any,
        create_graph: bool,
    ):
        args = args_from_batch(batch)
        steps = max(1, int(getattr(args, "local_update_steps", 1)))
        lr = float(getattr(args, "local_update_lr", getattr(args, "lr", 1e-3)))
        wd_arg = getattr(args, "local_update_wd", None)
        weight_decay = float(getattr(args, "wd", 0.0) if wd_arg is None else wd_arg)
        update_optimizer = getattr(args, "twomgtcn_update_optimizer", "adam").lower()
        if update_optimizer not in ("adam", "sgd"):
            raise ValueError(f"Unsupported twomgtcn_update_optimizer: {update_optimizer}")

        initial_params = OrderedDict(
            (name, param.detach().clone().requires_grad_(True))
            for name, param in model.named_parameters()
            if param.requires_grad
        )
        params = OrderedDict((name, param) for name, param in initial_params.items())
        buffers = OrderedDict((name, buffer.detach().clone()) for name, buffer in model.named_buffers())
        exp_avg = OrderedDict((name, torch.zeros_like(param)) for name, param in params.items())
        exp_avg_sq = OrderedDict((name, torch.zeros_like(param)) for name, param in params.items())

        for step_idx in range(steps):
            loss = self._task_loss_with_state(model, params, buffers, x, y, batch)
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
            else:
                params = self._sgd_step(
                    params=params,
                    grads=grads,
                    lr=lr,
                    weight_decay=weight_decay,
                )

        update_map = OrderedDict(
            (name, params[name] - initial_params[name])
            for name in initial_params.keys()
        )
        updates = tuple(update_map.values())
        surface = getattr(args, "twomgtcn_attack_surface", "update_fpass")
        if surface == "feature_update":
            return OrderedDict(
                (name, update)
                for name, update in update_map.items()
                if name.startswith("feature_fusion.")
            )
        if surface == "feature_weight_update":
            return OrderedDict(
                (name, update)
                for name, update in update_map.items()
                if name == "feature_fusion.weight"
            )
        if surface == "update":
            return updates
        if surface != "update_fpass":
            raise ValueError(f"Unsupported twomgtcn_attack_surface: {surface}")

        return {
            "model_update": updates,
            "fpass_weight_vector": self._weighted(
                self._fpass_weight_vector(params),
                float(getattr(args, "twomgtcn_fpass_weight", 1.0)),
            ),
        }

    def compute_leak(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        batch: Any,
        create_graph: bool,
        leak_type: str,
    ):
        args = args_from_batch(batch)
        x_flow = btnf_from_bntf(x)
        y_target = btnf_from_bntf(y)
        x_ext = self._x_ext_for_model(model, x, batch, args)

        if leak_type == "model_update":
            return self._differentiable_local_update(
                model=model,
                x=x,
                y=y,
                batch=batch,
                create_graph=create_graph,
            )

        pred, fused_feat = deterministic_forward(model, lambda: model(x_flow, x_ext))
        if leak_type == "activation":
            return fused_feat

        loss = loss_fn(args)(pred, y_target)
        lfac_alpha = float(getattr(args, "twomgtcn_lfac_alpha", 0.01))
        if lfac_alpha != 0.0:
            loss = loss + lfac_alpha * _feature_factorization_loss(fused_feat)
        grads = gradient_tuple(model=model, loss=loss, create_graph=create_graph)

        if leak_type == "gradient":
            return grads

        if leak_type != "model_update":
            raise ValueError("TwoMGTCN supports activation, gradient, and model_update reconstruction.")
