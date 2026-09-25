from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from privacy.baseline_adapters.common import (
    args_from_batch,
    btnf_from_bntf,
    deterministic_forward,
    fixed_quantized_prediction,
    empty_attr_like,
    gradient_tuple,
    load_state_dict_if_available,
    loss_fn,
)


class REFOLPrivacyAdapter:
    name = "REFOL"
    default_attack = "model_update"
    requires_dummy_y = True

    def build_model(self, args: Any, batch: Any) -> nn.Module:
        from model.refol_nets import GRU

        model = GRU(
            input_size=args.input_dim,
            hidden_size=args.hidden_dim,
            output_size=args.output_dim,
            dropout=getattr(args, "dropout", 0.0),
            gru_num_layers=1,
        ).to(args.device)
        load_state_dict_if_available(model, args)
        return model

    def compute_leak(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        batch: Any,
        create_graph: bool,
        leak_type: str,
    ):
        x_refol = btnf_from_bntf(x)
        y_refol = btnf_from_bntf(y)
        data = {
            "x": x_refol,
            "x_attr": empty_attr_like(x_refol),
            "y": y_refol,
            "y_attr": empty_attr_like(y_refol),
        }

        with torch.backends.cudnn.flags(enabled=False):
            pred = deterministic_forward(model, lambda: model(data))
        if leak_type == "activation" and bool(getattr(args_from_batch(batch), "quantized_prediction_sidechannel", False)):
            prediction_data = dict(data)
            prediction_data["y"] = torch.zeros_like(y_refol)
            with torch.backends.cudnn.flags(enabled=False):
                prediction = deterministic_forward(model, lambda: model(prediction_data))
            # Privacy batches use [B, N, T, F], while REFOL's decoder uses
            # [B, T, N, F].  The trace is saved in the privacy-batch layout.
            prediction = prediction.transpose(1, 2).contiguous()
            return fixed_quantized_prediction(prediction, args_from_batch(batch))
        loss = loss_fn(args_from_batch(batch))(pred, y_refol)
        grads = gradient_tuple(model=model, loss=loss, create_graph=create_graph)

        if leak_type == "gradient":
            return grads

        if leak_type != "model_update":
            raise ValueError("REFOL supports gradient diagnostics and model_update reconstruction.")

        args = args_from_batch(batch)
        lr = float(getattr(args, "local_update_lr", getattr(args, "lr", 1e-3)))
        wd_arg = getattr(args, "local_update_wd", None)
        weight_decay = float(getattr(args, "wd", 0.0) if wd_arg is None else wd_arg)

        updates = []
        params = [param for param in model.parameters() if param.requires_grad]
        for param, grad in zip(params, grads):
            if weight_decay != 0.0:
                grad = grad + weight_decay * param
            updates.append(-lr * grad)
        return tuple(updates)
