"""Privacy adapter for SFL's vanilla-RNN client predictor."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn

from model.SFL_RNN import SFLPureRNN
from privacy.baseline_adapters.common import align_prediction_and_target, fixed_quantized_prediction


class SFLRNNPrivacyAdapter:
    name = "SFL_RNN"
    default_attack = "model_update"
    requires_dummy_y = True

    def build_model(self, args: Any, batch: Any) -> nn.Module:
        model = SFLPureRNN(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            t_in=args.t_in,
            t_out=args.t_out,
        ).to(args.device)
        if args.checkpoint:
            state = torch.load(args.checkpoint, map_location=args.device)
            if isinstance(state, dict):
                state = state.get("state_dict") or state.get("model_state_dict") or state.get("model") or state
            if isinstance(state, dict):
                model.load_state_dict({key.replace("module.", "", 1): value for key, value in state.items()}, strict=False)
        return model

    @staticmethod
    def _loss(args: Any, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction, target = align_prediction_and_target(prediction, target)
        return nn.L1Loss()(prediction, target) if args.loss_func in ("mae", "l1") else nn.MSELoss()(prediction, target)

    def compute_leak(self, model, x, y, batch, create_graph: bool, leak_type: str):
        if leak_type == "activation" and bool(getattr(batch.args, "quantized_prediction_sidechannel", False)):
            with torch.backends.cudnn.flags(enabled=False):
                return fixed_quantized_prediction(model(x), batch.args)
        if leak_type not in ("gradient", "model_update"):
            raise ValueError("SFL_RNN supports gradient and model_update reconstruction only.")
        args = batch.args
        with torch.backends.cudnn.flags(enabled=False):
            loss = self._loss(args, model(x), y)
        params = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
        grads = torch.autograd.grad(loss, params, create_graph=create_graph, retain_graph=create_graph, allow_unused=True)
        if leak_type == "gradient":
            return tuple(torch.zeros_like(parameter) if grad is None else grad for parameter, grad in zip(params, grads))
        lr = float(getattr(args, "local_update_lr", getattr(args, "lr", 1e-3)))
        wd_value = getattr(args, "local_update_wd", None)
        wd = float(getattr(args, "wd", 0.0) if wd_value is None else wd_value)
        return tuple(
            -lr * ((torch.zeros_like(parameter) if grad is None else grad) + wd * parameter)
            for parameter, grad in zip(params, grads)
        )
