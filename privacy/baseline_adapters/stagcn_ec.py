from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.STGCN_EC import SpatioTemporalModel
from privacy.baseline_adapters.common import (
    align_prediction_and_target,
    args_from_batch,
    gradient_tuple,
    load_state_dict_if_available,
    loss_fn,
)

try:
    from torch.func import functional_call as _functional_call
except ImportError:  # pragma: no cover - depends on the installed torch version.
    from torch.nn.utils.stateless import functional_call as _functional_call


class STAGCNECPrivacyAdapter:
    name = "STAGCN-EC"
    default_attack = "server_realistic"
    requires_dummy_y = True
    server_visible_surface = (
        "STAGCN-EC server receives the METIS client adjacency, clients' initial "
        "random model weights for neighbor routing, and early-stopping validation "
        "scalars. It does not receive per-batch activations, gradients, or trained "
        "client updates in the implemented trainer."
    )

    def _edge_index_from_batch(
        self,
        batch: Any,
        num_nodes: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        edge_index = getattr(batch, "edge_index", None)
        if edge_index is None:
            idx = torch.arange(num_nodes, dtype=torch.long, device=device)
            return torch.stack([idx, idx], dim=0)

        if not torch.is_tensor(edge_index):
            edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        edge_index = edge_index.long().to(device)
        if edge_index.dim() != 2 or edge_index.shape[0] != 2 or edge_index.numel() == 0:
            idx = torch.arange(num_nodes, dtype=torch.long, device=device)
            return torch.stack([idx, idx], dim=0)

        row, col = edge_index[0], edge_index[1]
        valid = (row >= 0) & (row < num_nodes) & (col >= 0) & (col < num_nodes)
        row, col = row[valid], col[valid]
        loop = torch.arange(num_nodes, dtype=torch.long, device=device)
        row = torch.cat([row, loop])
        col = torch.cat([col, loop])
        return torch.stack([row, col], dim=0).contiguous()

    def build_model(self, args: Any, batch: Any) -> nn.Module:
        num_nodes = int(batch.real_x.shape[1])
        max_nodes = max(len(nodes) for nodes in getattr(batch, "nodes_per", [[0] * num_nodes]))
        edge_index = self._edge_index_from_batch(batch, num_nodes, args.device)
        model = SpatioTemporalModel(
            feat_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            time_steps=args.t_in,
            output_dim=args.t_out,
            K=3,
            max_nodes=max_nodes,
        ).to(args.device)
        model.edge_index = edge_index
        load_state_dict_if_available(model, args)
        return model

    def _forward_loss_with_state(
        self,
        model: nn.Module,
        params: OrderedDict[str, torch.Tensor],
        buffers: OrderedDict[str, torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
        args: Any,
    ) -> torch.Tensor:
        state = OrderedDict()
        state.update(params)
        state.update(buffers)

        was_training = model.training
        model.train()
        try:
            with torch.backends.cudnn.flags(enabled=False):
                pred = _functional_call(model, state, (x,))
        finally:
            model.train(was_training)

        pred, target = align_prediction_and_target(pred, y)
        return loss_fn(args)(pred, target)

    def _sgd_step(
        self,
        params: OrderedDict[str, torch.Tensor],
        grads: tuple[torch.Tensor | None, ...],
        weight_decay: float,
        lr: float,
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
        step_no: int,
        weight_decay: float,
        lr: float,
        args: Any,
    ) -> tuple[OrderedDict[str, torch.Tensor], OrderedDict[str, torch.Tensor], OrderedDict[str, torch.Tensor]]:
        beta1 = float(getattr(args, "stagcn_ec_adam_beta1", 0.9))
        beta2 = float(getattr(args, "stagcn_ec_adam_beta2", 0.999))
        eps = float(getattr(args, "stagcn_ec_adam_eps", 1e-8))
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
            # The literal Adam update uses sqrt(v)+eps.  During inversion we
            # differentiate through that update: parameters whose gradient is
            # exactly zero have v=0, where the second derivative of sqrt is
            # singular and produces NaNs in the dummy-input gradient.  The
            # floor is far below normal FP32 Adam moments, preserves the
            # forward update for nonzero moments, and gives zero-gradient
            # coordinates a finite differentiable surrogate.
            safe_v = v.clamp_min(eps * eps)
            denom = safe_v.sqrt() / math.sqrt(bias_correction2) + eps

            next_params[name] = param - step_size * m / denom
            next_exp_avg[name] = m
            next_exp_avg_sq[name] = v

        return next_params, next_exp_avg, next_exp_avg_sq

    def _activation_leak(self, model: nn.Module, x: torch.Tensor):
        was_training = model.training
        model.train()
        try:
            with torch.backends.cudnn.flags(enabled=False):
                if x.is_sparse:
                    x = x.to_dense()
                if x.dim() == 5:
                    if x.shape[-1] == 1:
                        x = x.squeeze(-1)
                    elif x.shape[1] == 1:
                        x = x.squeeze(1)
                if x.dim() == 3:
                    x = x.unsqueeze(0)

                batch_size, num_real_nodes, time_steps, feat_dim = x.size()
                if num_real_nodes < model.num_nodes:
                    x_padded = F.pad(x, (0, 0, 0, 0, 0, model.num_nodes - num_real_nodes))
                elif num_real_nodes > model.num_nodes:
                    raise ValueError(
                        f"Input nodes {num_real_nodes} exceeds max_nodes {model.num_nodes}!"
                    )
                else:
                    x_padded = x

                edge_index = getattr(model, "edge_index", None)
                if model.cheb_polynomials is None or model.cheb_polynomials[0].shape[0] != model.num_nodes:
                    if edge_index is None:
                        raise ValueError("Edge Index missing")
                    model.cheb_polynomials = model.compute_cheb_polynomials(
                        edge_index,
                        model.num_nodes,
                        x.device,
                    )

                x_squeeze = x_padded.squeeze(-1) if feat_dim == 1 else x_padded.mean(dim=-1)
                x_att = model.st_attention(x_squeeze)
                x_att_feat = x_att.unsqueeze(-1) if feat_dim == 1 else x_att.unsqueeze(-1).repeat(1, 1, 1, feat_dim)

                gcn_in = x_att_feat.transpose(1, 2).reshape(
                    batch_size * time_steps,
                    model.num_nodes,
                    feat_dim,
                )
                gcn_embedding = F.relu(model.cheb_conv(gcn_in, model.cheb_polynomials))
                gcn_embedding = gcn_embedding.view(
                    batch_size,
                    time_steps,
                    model.num_nodes,
                    model.hidden_dim,
                ).transpose(1, 2)

                gru_in = gcn_embedding.reshape(batch_size * model.num_nodes, time_steps, model.hidden_dim)
                _, h_n = model.gru(gru_in)
                gru_state = h_n.squeeze(0).view(batch_size, model.num_nodes, model.hidden_dim)

                return {
                    "attention_x": x_att[:, :num_real_nodes, :],
                    "gcn_embedding": gcn_embedding[:, :num_real_nodes, :, :] * 0.1,
                    "gru_state": gru_state[:, :num_real_nodes, :] * 0.1,
                }
        finally:
            model.train(was_training)

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
        lr = float(getattr(args, "local_update_lr", getattr(args, "lr", 1e-3)))
        wd_arg = getattr(args, "local_update_wd", None)
        weight_decay = float(getattr(args, "wd", 0.0) if wd_arg is None else wd_arg)
        update_optimizer = getattr(args, "stagcn_ec_update_optimizer", "sgd").lower()
        if update_optimizer not in ("sgd", "adam"):
            raise ValueError(f"Unsupported stagcn_ec_update_optimizer: {update_optimizer}")

        initial_params = OrderedDict(
            (name, param.detach().clone().requires_grad_(True))
            for name, param in model.named_parameters()
            if param.requires_grad
        )
        params = OrderedDict((name, param) for name, param in initial_params.items())
        buffers = OrderedDict(
            (name, buffer.detach().clone())
            for name, buffer in model.named_buffers()
            if buffer is not None
        )
        exp_avg = OrderedDict((name, torch.zeros_like(param)) for name, param in params.items())
        exp_avg_sq = OrderedDict((name, torch.zeros_like(param)) for name, param in params.items())

        for step_idx in range(steps):
            loss = self._forward_loss_with_state(model, params, buffers, x, y, args)
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
                    weight_decay=weight_decay,
                    lr=lr,
                    args=args,
                )
            else:
                params = self._sgd_step(params, grads, weight_decay, lr)

        return tuple(params[name] - initial_params[name] for name in initial_params)

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
        if leak_type == "activation":
            return self._activation_leak(model, x)

        if leak_type == "model_update":
            return self._differentiable_local_update(
                model=model,
                x=x,
                y=y,
                batch=batch,
                create_graph=create_graph,
            )

        if leak_type != "gradient":
            raise ValueError(
                "STAGCN-EC supports server_realistic no-signal, activation/gradient "
                "malicious-edge diagnostics, and model_update upper-bound reconstruction."
            )

        was_training = model.training
        model.train()
        try:
            with torch.backends.cudnn.flags(enabled=False):
                pred = model(x)
        finally:
            model.train(was_training)

        pred, target = align_prediction_and_target(pred, y)
        loss = loss_fn(args)(pred, target)
        return gradient_tuple(
            model=model,
            loss=loss,
            create_graph=create_graph,
        )
