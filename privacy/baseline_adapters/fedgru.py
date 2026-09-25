from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, List

import torch
import torch.nn as nn
from privacy.baseline_adapters.common import fixed_quantized_prediction

from model.FedGRU import FedGRU_Model
from privacy.baseline_adapters.common import align_prediction_and_target

try:
    from torch.func import functional_call as _functional_call
except ImportError:  # pragma: no cover
    from torch.nn.utils.stateless import functional_call as _functional_call


class FedGRUPrivacyAdapter:
    name = "FedGRU"
    default_attack = "model_update"
    requires_dummy_y = True

    def build_model(self, args: Any, batch: Any) -> nn.Module:
        model = FedGRU_Model(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            out_dim=args.output_dim,
            pre_len=args.t_out,
            num_layers=2,
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

    def _loss_fn(self, args: Any):
        if args.loss_func in ("mae", "l1"):
            return nn.L1Loss()
        return nn.MSELoss()

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
        return self._loss_fn(batch_args(batch))(pred, y)

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
        return self._loss_fn(batch_args(batch))(pred, y)

    def _simulate_local_update(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        batch: Any,
        create_graph: bool,
    ) -> tuple[torch.Tensor, ...]:
        args = batch_args(batch)
        steps = max(1, int(getattr(args, "local_update_steps", 1)))
        lr = float(getattr(args, "local_update_lr", getattr(args, "lr", 1e-3)))
        wd_arg = getattr(args, "local_update_wd", None)
        weight_decay = float(getattr(args, "wd", 0.0) if wd_arg is None else wd_arg)
        update_optimizer = getattr(args, "fedgru_update_optimizer", "adam").lower()

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
                beta1 = float(getattr(args, "fedgru_adam_beta1", 0.9))
                beta2 = float(getattr(args, "fedgru_adam_beta2", 0.999))
                eps = float(getattr(args, "fedgru_adam_eps", 1e-8))
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
            elif update_optimizer == "sgd":
                next_params = OrderedDict()
                for (name, param), grad in zip(params.items(), grads):
                    grad = torch.zeros_like(param) if grad is None else grad
                    if weight_decay != 0.0:
                        grad = grad + weight_decay * param
                    next_params[name] = param - lr * grad
                params = next_params
            else:
                raise ValueError(f"Unsupported fedgru_update_optimizer: {update_optimizer}")

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
        if leak_type == "activation" and bool(getattr(batch_args(batch), "quantized_prediction_sidechannel", False)):
            with torch.backends.cudnn.flags(enabled=False):
                return fixed_quantized_prediction(model(x), batch_args(batch))
        if leak_type == "model_update":
            return self._simulate_local_update(
                model=model,
                x=x,
                y=y,
                batch=batch,
                create_graph=create_graph,
            )

        if leak_type != "gradient":
            raise ValueError("FedGRU currently supports gradient and model_update reconstruction only.")

        # Gradient reconstruction differentiates through parameter gradients,
        # which requires second-order derivatives w.r.t. dummy_x. cuDNN RNNs do
        # not support that path, so run GRU forwards through PyTorch's backend.
        with torch.backends.cudnn.flags(enabled=False):
            pred = model(x)
        pred, y = align_prediction_and_target(pred, y)
        loss = self._loss_fn(batch_args(batch))(pred, y)

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


def batch_args(batch: Any) -> Any:
    args = getattr(batch, "args", None)
    if args is not None:
        return args

    class _DefaultArgs:
        loss_func = "mse"

    return _DefaultArgs()
