from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class FedTPSPrivacyAdapter:
    name = "FedTPS"
    default_attack = "model_update"
    requires_dummy_y = True

    def build_model(self, args: Any, batch: Any) -> nn.Module:
        # Import lazily so pytorch_wavelets is only loaded when FedTPS is used.
        from model.FedTPS import DCRNN_TP

        num_nodes = int(batch.real_x.shape[1])
        wave_mapping = {
            "PeMS03": "haar",
            "PeMS04": "coif1",
            "PeMSD7": "bior1.3",
            "PeMS07": "bior1.3",
            "PeMS08": "haar",
        }
        model = DCRNN_TP(
            num_nodes=num_nodes,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            horizon=args.t_out,
            rnn_units=64,
            num_layers=2,
            cheb_k=2,
            ycov_dim=args.output_dim,
            wave=wave_mapping.get(args.dataset_name, "coif1"),
        ).to(args.device)
        model.use_curriculum_learning = False
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
        y_cov = self._y_cov(x, args)
        labels = btnf_from_bntf(y)
        with torch.backends.cudnn.flags(enabled=False):
            pred = deterministic_forward(model, lambda: model(x, y_cov=y_cov, labels=labels))
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
                pred = _functional_call(model, state, (x,), {"y_cov": y_cov, "labels": labels})
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
    ) -> tuple[torch.Tensor]:
        args = args_from_batch(batch)
        steps = max(1, int(getattr(args, "local_update_steps", 1)))
        lr = float(getattr(args, "local_update_lr", 1e-3))
        wd_arg = getattr(args, "local_update_wd", None)
        weight_decay = float(getattr(args, "wd", 0.0) if wd_arg is None else wd_arg)
        update_scope = getattr(args, "fedtps_update_scope", "patterns")
        update_optimizer = getattr(args, "fedtps_update_optimizer", "adam").lower()

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
            loss = self._loss_with_state(model, params, buffers, x, y, batch)
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
                raise ValueError(f"Unsupported fedtps_update_optimizer: {update_optimizer}")

        if update_scope == "patterns":
            if "Patterns" not in params:
                raise KeyError("FedTPS model does not expose a Patterns parameter.")
            patterns_update = params["Patterns"] - initial_params["Patterns"]
            if not bool(int(getattr(args, "fedtps_include_agg_patterns", 1))):
                return (patterns_update,)
            return {
                "patterns_update": patterns_update,
                "agg_patterns_update": self._weighted(
                    self._server_agg_patterns_proxy(params["Patterns"], args)
                    - self._server_agg_patterns_proxy(initial_params["Patterns"], args),
                    float(getattr(args, "fedtps_agg_patterns_weight", 0.05)),
                ),
            }
        if update_scope == "full":
            return tuple(params[name] - initial_params[name] for name in initial_params.keys())
        raise ValueError(f"Unsupported FedTPS update scope: {update_scope}")

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
        beta1 = float(getattr(args, "fedtps_adam_beta1", 0.9))
        beta2 = float(getattr(args, "fedtps_adam_beta2", 0.999))
        eps = float(getattr(args, "fedtps_adam_eps", 1e-8))
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

    def _weighted(self, tensor: torch.Tensor, weight: float) -> torch.Tensor:
        return tensor * math.sqrt(max(0.0, float(weight)))

    def _server_agg_patterns_proxy(self, patterns: torch.Tensor, args: Any) -> torch.Tensor:
        # The real server aggregates across all clients by top-k cosine pattern
        # similarity. The privacy runner attacks one client sample, so this
        # single-client proxy preserves the same top-k operation without
        # inventing unavailable host-client patterns.
        top_k = max(1, int(getattr(args, "fedtps_k", 2)))
        pattern_num = patterns.shape[0]
        actual_k = min(top_k, pattern_num)
        sim = F.cosine_similarity(patterns.unsqueeze(1), patterns.unsqueeze(0), dim=-1)
        _, topk_indices = torch.topk(sim, actual_k, dim=1)
        gathered = patterns[topk_indices]
        return gathered.mean(dim=1)

    def compute_leak(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        batch: Any,
        create_graph: bool,
        leak_type: str,
    ):
        if leak_type == "activation" and bool(
            getattr(args_from_batch(batch), "quantized_prediction_sidechannel", False)
        ):
            args = args_from_batch(batch)
            y_cov = self._y_cov(x, args)
            # Use the serving path (no teacher-forced labels), matching the
            # client prediction returned by the revised monitoring protocol.
            with torch.backends.cudnn.flags(enabled=False):
                prediction = deterministic_forward(model, lambda: model(x, y_cov=y_cov))
            return fixed_quantized_prediction(prediction, args)
        if leak_type == "model_update":
            return self._differentiable_local_update(
                model=model,
                x=x,
                y=y,
                batch=batch,
                create_graph=create_graph,
            )

        if leak_type != "gradient":
            raise ValueError("FedTPS currently supports gradient and model_update reconstruction only.")

        loss = self._task_loss(model, x, y, batch)
        return gradient_tuple(model=model, loss=loss, create_graph=create_graph)
