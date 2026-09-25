from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.FedMetro import FedMetro_Client_Model
from privacy.baseline_adapters.common import (
    align_prediction_and_target,
    args_from_batch,
    load_state_dict_if_available,
    loss_fn,
    fixed_quantized_prediction,
)
from privacy.baseline_adapters.graph_common import prediction_gradient


class FedMetroPrivacyAdapter:
    name = "FedMetro"
    default_attack = "model_update"
    requires_dummy_y = True

    def build_model(self, args: Any, batch: Any) -> nn.Module:
        num_nodes = int(batch.real_x.shape[1])
        model = FedMetro_Client_Model(
            num_nodes=num_nodes,
            t_in=args.t_in,
            t_out=args.t_out,
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            d_E=getattr(args, "node_emb_dim", 4),
            K=getattr(args, "poly_k", 4),
        ).to(args.device)
        load_state_dict_if_available(model, args)
        return model

    def _phase1(self, model: nn.Module, x: torch.Tensor):
        return model.forward_phase1(x)

    def _deterministic_phase1(self, model: nn.Module, x: torch.Tensor):
        was_training = model.training
        model.eval()
        try:
            return self._phase1(model, x)
        finally:
            model.train(was_training)

    def _split_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        agg_global: torch.Tensor,
        f_e_seq: torch.Tensor,
        e_seq: torch.Tensor,
        args: Any,
    ) -> torch.Tensor:
        pred = model.forward_phase2(x, f_e_seq, e_seq, agg_global)
        pred, target = align_prediction_and_target(pred, y)
        main_loss = loss_fn(args)(pred, target)

        last_m = getattr(model.dyn_emb_mask.mask_generator, "last_m", None)
        if last_m is None:
            return main_loss
        reg_weight = float(getattr(args, "lambda_reg", getattr(args, "fedmetro_lambda_reg", 0.001)))
        return main_loss + reg_weight * torch.mean(last_m)

    def _weighted(self, tensor: torch.Tensor, weight: float) -> torch.Tensor:
        return tensor * math.sqrt(max(0.0, float(weight)))

    def _uploaded_weight_update_vector(
        self,
        model: nn.Module,
        loss: torch.Tensor,
        *,
        create_graph: bool,
        args: Any,
    ) -> torch.Tensor:
        named_params = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad and "static_E" not in name and "node_embeddings" not in name
        ]
        if not named_params:
            return loss.new_zeros(1)

        params = tuple(param for _, param in named_params)
        grads = torch.autograd.grad(
            loss,
            params,
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=True,
        )
        lr = float(getattr(args, "local_update_lr", 1e-3))
        wd_arg = getattr(args, "local_update_wd", None)
        weight_decay = float(getattr(args, "wd", 0.0) if wd_arg is None else wd_arg)

        updates = []
        for (_, param), grad in zip(named_params, grads):
            if grad is None:
                grad = torch.zeros_like(param)
            if weight_decay != 0.0:
                grad = grad + weight_decay * param
            updates.append((-lr * grad).reshape(-1))

        flat = torch.cat(updates)
        return F.normalize(flat, p=2, dim=0)

    def _server_visible_joint_leak(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        batch: Any,
        create_graph: bool,
    ):
        args = args_from_batch(batch)
        surface = getattr(args, "fedmetro_attack_surface", "agg_gagg_update")

        was_training = model.training
        model.eval()
        try:
            agg_seq, f_e_seq, e_seq = self._phase1(model, x)
            leak = {
                "agg_seq": self._weighted(
                    agg_seq,
                    float(getattr(args, "fedmetro_agg_weight", 1.0)),
                )
            }
            if surface == "agg":
                return leak

            # The server sends an aggregated tensor back as a detached value in
            # real training. For a single-client privacy sample, use this
            # client's AGG as the local proxy while keeping the split-gradient
            # path faithful to the detached server return.
            agg_global = agg_seq.detach().clone().requires_grad_(True)
            split_loss = self._split_loss(
                model=model,
                x=x,
                y=y,
                agg_global=agg_global,
                f_e_seq=f_e_seq,
                e_seq=e_seq,
                args=args,
            )
            g_agg = torch.autograd.grad(
                split_loss,
                agg_global,
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=False,
            )[0]
            leak["g_agg"] = self._weighted(
                g_agg,
                float(getattr(args, "fedmetro_gagg_weight", 0.05)),
            )

            if surface == "agg_gagg":
                return leak
            if surface != "agg_gagg_update":
                raise ValueError(f"Unsupported fedmetro_attack_surface: {surface}")

            update_vec = self._uploaded_weight_update_vector(
                model,
                split_loss,
                create_graph=create_graph,
                args=args,
            )
            leak["weight_update"] = self._weighted(
                update_vec,
                float(getattr(args, "fedmetro_update_weight", 0.005)),
            )
            return leak
        finally:
            model.train(was_training)

    def compute_leak(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        batch: Any,
        create_graph: bool,
        leak_type: str,
    ):
        if leak_type == "activation":
            if bool(getattr(args_from_batch(batch), "quantized_prediction_sidechannel", False)):
                agg_seq, f_e_seq, e_seq = self._deterministic_phase1(model, x)
                prediction = model.forward_phase2(x, f_e_seq, e_seq, agg_seq)
                return fixed_quantized_prediction(prediction, args_from_batch(batch))
            agg_seq, f_e_seq, _ = self._deterministic_phase1(model, x)
            args = getattr(batch, "args", None)
            scope = getattr(args, "fedmetro_activation_scope", "server")
            if scope == "server":
                return agg_seq
            if scope == "full":
                return {
                    "agg_seq": agg_seq,
                    "f_e_seq": f_e_seq,
                }
            raise ValueError(f"Unsupported fedmetro_activation_scope: {scope}")

        if leak_type == "model_update":
            return self._server_visible_joint_leak(
                model=model,
                x=x,
                y=y,
                batch=batch,
                create_graph=create_graph,
            )

        if leak_type != "gradient":
            raise ValueError("FedMetro currently supports activation, gradient, and model_update reconstruction.")

        agg_seq, f_e_seq, e_seq = self._deterministic_phase1(model, x)
        pred = model.forward_phase2(x, f_e_seq, e_seq, agg_seq)
        return prediction_gradient(model=model, pred=pred, y=y, batch=batch, create_graph=create_graph)
