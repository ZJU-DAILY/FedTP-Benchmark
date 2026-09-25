from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn

from model.FedTSE import TrafficLSTM
from privacy.baseline_adapters.common import (
    align_prediction_and_target,
    args_from_batch,
    gradient_tuple,
    load_state_dict_if_available,
    loss_fn,
    fixed_quantized_prediction,
)

try:
    from torch.func import functional_call as _functional_call
except ImportError:  # pragma: no cover
    from torch.nn.utils.stateless import functional_call as _functional_call


class FedTSEPrivacyAdapter:
    name = "FedTSE"
    default_attack = "model_update"
    requires_dummy_y = True
    # A local loss reduces over the mini-batch before the client update is
    # uploaded, so one model update is a batch-aggregated observation.
    batch_aggregate_leak = True

    def build_model(self, args: Any, batch: Any) -> nn.Module:
        num_nodes = int(batch.real_x.shape[1])
        model = TrafficLSTM(
            num_nodes=num_nodes,
            t_in=args.t_in,
            input_size=args.input_dim,
            hidden_size=args.hidden_dim,
            output_size=args.t_out,
            output_dim=args.output_dim,
        ).to(args.device)
        load_state_dict_if_available(model, args)
        return model

    def _task_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        batch: Any,
    ) -> torch.Tensor:
        with torch.backends.cudnn.flags(enabled=False):
            pred = model(x)
        pred, y = align_prediction_and_target(pred, y)
        return loss_fn(args_from_batch(batch))(pred, y)

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

        was_training = model.training
        model.eval()
        try:
            with torch.backends.cudnn.flags(enabled=False):
                pred = _functional_call(model, state, (x,))
        finally:
            model.train(was_training)

        pred, y = align_prediction_and_target(pred, y)
        return loss_fn(args_from_batch(batch))(pred, y)

    def _simulate_local_update(
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
        update_optimizer = getattr(args, "fedtse_update_optimizer", "sgd").lower()

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
            loss = self._loss_with_state(model, params, buffers, x, y, batch)
            grads = torch.autograd.grad(
                loss,
                tuple(params.values()),
                create_graph=create_graph,
                retain_graph=create_graph,
                allow_unused=True,
            )

            if update_optimizer == "adam":
                beta1 = float(getattr(args, "fedtse_adam_beta1", 0.9))
                beta2 = float(getattr(args, "fedtse_adam_beta2", 0.999))
                eps = float(getattr(args, "fedtse_adam_eps", 1e-8))
                bias_correction1 = 1.0 - beta1 ** (step_idx + 1)
                bias_correction2 = 1.0 - beta2 ** (step_idx + 1)
                next_params = OrderedDict()
                next_exp_avg = OrderedDict()
                next_exp_avg_sq = OrderedDict()
                for (name, param), grad in zip(params.items(), grads):
                    grad = torch.zeros_like(param) if grad is None else grad
                    if weight_decay != 0.0:
                        grad = grad + weight_decay * param
                    m = beta1 * exp_avg[name] + (1.0 - beta1) * grad
                    v = beta2 * exp_avg_sq[name] + (1.0 - beta2) * grad.pow(2)
                    step_size = lr / bias_correction1
                    denom = v.sqrt() / math.sqrt(bias_correction2) + eps
                    next_params[name] = param - step_size * m / denom
                    next_exp_avg[name] = m
                    next_exp_avg_sq[name] = v
                params = next_params
                exp_avg = next_exp_avg
                exp_avg_sq = next_exp_avg_sq
            else:
                next_params = OrderedDict()
                for (name, param), grad in zip(params.items(), grads):
                    grad = torch.zeros_like(param) if grad is None else grad
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
        if leak_type == "activation" and bool(getattr(args_from_batch(batch), "quantized_prediction_sidechannel", False)):
            with torch.backends.cudnn.flags(enabled=False):
                prediction = model(x)
            return fixed_quantized_prediction(prediction, args_from_batch(batch))
        if leak_type == "gradient":
            loss = self._task_loss(model, x, y, batch)
            return gradient_tuple(model=model, loss=loss, create_graph=create_graph)

        if leak_type != "model_update":
            raise ValueError("FedTSE currently supports gradient and model_update reconstruction only.")

        return self._simulate_local_update(
            model=model,
            x=x,
            y=y,
            batch=batch,
            create_graph=create_graph,
        )
