from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn

from privacy.baseline_adapters.common import (
    align_prediction_and_target,
    args_from_batch,
    btnf_from_bntf,
    deterministic_forward,
    fixed_quantized_prediction,
    gradient_tuple,
    load_state_dict_if_available,
    loss_fn,
)

try:
    from torch.func import functional_call as _functional_call
except ImportError:  # pragma: no cover - depends on the installed torch version.
    from torch.nn.utils.stateless import functional_call as _functional_call


class Fed4TPPrivacyAdapter:
    name = "Fed4TP"
    default_attack = "model_update"
    requires_dummy_y = True

    def build_model(self, args: Any, batch: Any) -> nn.Module:
        # Fed4TP now uses a DyHSL-style local prediction backbone inside the
        # federated protocol branch.  The privacy entrypoint should mirror that
        # trainable local model rather than a placeholder surrogate.
        from model.DyHSL import DyHSL

        num_nodes = int(batch.real_x.shape[1])
        scale_values = tuple(
            int(item.strip())
            for item in str(getattr(args, "dyhsl_scales", "1,3,6,12")).split(",")
            if item.strip()
        )
        model = DyHSL(
            num_nodes=num_nodes,
            t_in=args.t_in,
            t_out=args.t_out,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            hidden_dim=getattr(args, "hidden_dim", 64),
            dropout=getattr(args, "dyhsl_dropout", 0.1),
            num_backbone_layers=getattr(args, "dyhsl_num_backbone_layers", 2),
            num_head_layers=getattr(args, "dyhsl_num_head_layers", 2),
            num_hyper_edge=getattr(args, "dyhsl_num_hyper_edge", 32),
            winsize=getattr(args, "dyhsl_winsize", 3),
            scales=scale_values,
        ).to(args.device)
        load_state_dict_if_available(model, args)
        return model

    def _y_cov(self, x: torch.Tensor, args: Any) -> torch.Tensor:
        return torch.zeros(
            x.shape[0],
            args.t_out,
            x.shape[1],
            args.output_dim,
            device=x.device,
            dtype=x.dtype,
        )

    def _task_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        batch: Any,
    ) -> torch.Tensor:
        args = args_from_batch(batch)
        with torch.backends.cudnn.flags(enabled=False):
            pred = deterministic_forward(model, lambda: model(x))
        pred, target = align_prediction_and_target(pred, y)
        return loss_fn(args)(pred, target)

    def _loss_with_state(
        self,
        model: nn.Module,
        params: OrderedDict[str, torch.Tensor],
        buffers: OrderedDict[str, torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
        batch: Any,
    ) -> torch.Tensor:
        args = args_from_batch(batch)
        y_cov = self._y_cov(x, args)
        labels = btnf_from_bntf(y)
        state = OrderedDict()
        state.update(params)
        state.update(buffers)

        was_training = model.training
        model.eval()
        try:
            with torch.backends.cudnn.flags(enabled=False):
                pred = _functional_call(model, state, (x,))
        finally:
            model.train(was_training)

        pred, target = align_prediction_and_target(pred, y)
        return loss_fn(args)(pred, target)

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
        lr = float(getattr(args, "local_update_lr", 1e-3))
        wd_arg = getattr(args, "local_update_wd", None)
        weight_decay = float(getattr(args, "wd", 0.0) if wd_arg is None else wd_arg)
        attack_surface = getattr(args, "fed4tp_attack_surface", "weights_mask")
        update_optimizer = getattr(args, "fed4tp_update_optimizer", "adam").lower()
        if update_optimizer not in ("sgd", "adam"):
            raise ValueError(f"Unsupported fed4tp_update_optimizer: {update_optimizer}")
        include_mask = attack_surface == "weights_mask"

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
        grad_sums = OrderedDict((name, torch.zeros_like(param)) for name, param in params.items())
        exp_avg = OrderedDict((name, torch.zeros_like(param)) for name, param in params.items())
        exp_avg_sq = OrderedDict((name, torch.zeros_like(param)) for name, param in params.items())

        for step_idx in range(steps):
            loss = self._loss_with_state(model, params, buffers, x, y, batch)
            grads = torch.autograd.grad(
                loss,
                tuple(params.values()),
                create_graph=create_graph,
                retain_graph=create_graph,
                allow_unused=True,
            )

            for (name, param), grad in zip(params.items(), grads):
                if grad is None:
                    grad = torch.zeros_like(param)
                grad_sums[name] = grad_sums[name] + grad.abs()

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

        payload = {
            "model_update": tuple(params[name] - initial_params[name] for name in initial_params.keys())
        }
        if include_mask:
            avg_abs_grads = tuple(grad_sums[name] / float(steps) for name in initial_params.keys())
            payload["top_k_mask"] = _fed4tp_topk_mask_leak(
                grads=avg_abs_grads,
                create_graph=create_graph,
                args=args,
                already_abs=True,
            )
        return payload

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
        beta1 = float(getattr(args, "fed4tp_adam_beta1", 0.9))
        beta2 = float(getattr(args, "fed4tp_adam_beta2", 0.999))
        eps = float(getattr(args, "fed4tp_adam_eps", 1e-8))
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
            with torch.backends.cudnn.flags(enabled=False):
                prediction = deterministic_forward(model, lambda: model(x))
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
            raise ValueError("Fed4TP currently supports gradient and model_update reconstruction only.")

        loss = self._task_loss(model, x, y, batch)
        return gradient_tuple(model=model, loss=loss, create_graph=create_graph)


def _fed4tp_topk_mask_leak(
    *,
    grads: tuple[torch.Tensor, ...],
    create_graph: bool,
    args: Any,
    already_abs: bool = False,
) -> tuple[torch.Tensor, ...]:
    if not grads:
        return tuple()

    scores = tuple(grad if already_abs else grad.abs() for grad in grads)
    flat_scores = torch.cat([score.reshape(-1) for score in scores])
    k = max(1, int(flat_scores.numel() * 0.2))
    threshold = torch.topk(flat_scores.detach(), k).values[-1]

    mask_weight = max(0.0, float(getattr(args, "fed4tp_mask_weight", 1.0)))
    scale = math.sqrt(mask_weight)
    if scale == 0.0:
        return tuple(score.new_zeros(score.shape) for score in scores)

    if not create_graph:
        return tuple(((score >= threshold).to(dtype=score.dtype) * scale) for score in scores)

    temperature = max(1e-6, float(getattr(args, "fed4tp_mask_temperature", 0.1)))
    score_scale = flat_scores.detach().std().clamp_min(1e-6)
    denom = temperature * score_scale
    return tuple(torch.sigmoid((score - threshold) / denom) * scale for score in scores)
