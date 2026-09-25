from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.UFCL_GWN import LightGraphWaveNet
from privacy.baseline_adapters.common import align_prediction_and_target, load_state_dict_if_available

try:
    from torch.func import functional_call as _functional_call
except ImportError:  # pragma: no cover - depends on the installed torch version.
    from torch.nn.utils.stateless import functional_call as _functional_call


class UFCLPrivacyAdapter:
    name = "UFCL"
    default_attack = "model_update"
    requires_dummy_y = True
    # One uploaded UFCL update combines the ordered local trajectory: the
    # first batch seeds replay and the following batch consumes it through
    # mixup/KD. Evaluate record 0 as the fixed victim while the later records
    # remain nuisance inputs needed to reproduce that one update.
    batch_aggregate_leak = True

    # Match the actual UFCL entrypoint: trainer_mode=ufcl uses the lightweight
    # GraphWaveNet backbone (``model=UFCL_GWN``).  It does not use STGCN-EC.
    # The original mismatch meant a captured UFCL_GWN update could never be
    # replayed against this adapter's parameter tree.
    _NOISE_STD = 0.05
    _KD_WEIGHT = 1.0
    _MIXUP_ALPHA = 0.2
    # This adapter differentiates through two local Adam steps. The training
    # optimizer's default 1e-8 epsilon is forward-safe but makes the resulting
    # second-order input derivative numerically singular near inactive paths.
    # This value is used only by the privacy reconstruction surrogate, never by
    # the original UFCL trainer.
    attack_local_adam_epsilon = 1e-4

    def build_model(self, args: Any, batch: Any) -> nn.Module:
        num_nodes = int(batch.real_x.shape[1])
        model = LightGraphWaveNet(
            num_nodes=num_nodes,
            input_dim=int(args.input_dim),
            output_dim=int(args.output_dim),
            horizon=int(args.t_out),
            residual_channels=int(getattr(args, "ufcl_gwn_channels", 29)),
            skip_channels=int(getattr(args, "ufcl_gwn_skip_channels", 58)),
            end_channels=int(getattr(args, "ufcl_gwn_end_channels", 58)),
            blocks=int(getattr(args, "ufcl_gwn_blocks", 2)),
            layers=int(getattr(args, "ufcl_gwn_layers", 2)),
            dropout=float(getattr(args, "ufcl_gwn_dropout", 0.1)),
        ).to(args.device)
        edge_index = self._edge_index_from_batch(batch, num_nodes, args.device)
        model.set_edge_index(edge_index)
        load_state_dict_if_available(model, args)
        return model

    def _edge_index_from_batch(
        self,
        batch: Any,
        num_nodes: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        edge_index = getattr(batch, "edge_index", None)
        if edge_index is None:
            nodes = torch.arange(num_nodes, dtype=torch.long, device=device)
            return torch.stack([nodes, nodes], dim=0)
        edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=device)
        row, col = edge_index[0], edge_index[1]
        valid = (row >= 0) & (row < num_nodes) & (col >= 0) & (col < num_nodes)
        loops = torch.arange(num_nodes, dtype=torch.long, device=device)
        return torch.stack([torch.cat([row[valid], loops]), torch.cat([col[valid], loops])], dim=0)

    def _state(self, model: nn.Module, *, requires_grad: bool) -> tuple[OrderedDict, OrderedDict]:
        params = OrderedDict(
            (name, param.detach().clone().requires_grad_(requires_grad))
            for name, param in model.named_parameters()
            if param.requires_grad
        )
        buffers = OrderedDict((name, buffer.detach().clone()) for name, buffer in model.named_buffers())
        return params, buffers

    def _forward(
        self,
        model: nn.Module,
        params: OrderedDict,
        buffers: OrderedDict,
        x: torch.Tensor,
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
        return pred[0] if isinstance(pred, (tuple, list)) else pred

    def _pad_local_nodes(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        real_nodes = int(x.shape[1])
        if real_nodes > model.num_nodes:
            raise ValueError(f"UFCL local node count {real_nodes} exceeds model max_nodes {model.num_nodes}.")
        if real_nodes == model.num_nodes:
            return x, y, real_nodes
        pad_nodes = model.num_nodes - real_nodes
        return F.pad(x, (0, 0, 0, 0, 0, pad_nodes)), F.pad(y, (0, 0, 0, 0, 0, pad_nodes)), real_nodes

    def _fixed_noise(self, reference: torch.Tensor, seed: int) -> torch.Tensor:
        devices = [reference.device.index] if reference.device.type == "cuda" and reference.device.index is not None else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if reference.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            return torch.randn_like(reference) * self._NOISE_STD

    def _mix_lambda(self, seed: int) -> float:
        return float(np.random.RandomState(seed).beta(self._MIXUP_ALPHA, self._MIXUP_ALPHA))

    def _adam_step(
        self,
        params: OrderedDict,
        grads: tuple[torch.Tensor | None, ...],
        exp_avg: OrderedDict,
        exp_avg_sq: OrderedDict,
        step_no: int,
        args: Any,
    ) -> tuple[OrderedDict, OrderedDict, OrderedDict]:
        lr = float(getattr(args, "local_update_lr", 1e-3))
        weight_decay = float(getattr(args, "wd", 0.0) if getattr(args, "local_update_wd", None) is None else args.local_update_wd)
        beta1, beta2 = 0.9, 0.999
        eps = self.attack_local_adam_epsilon
        bias1, bias2 = 1.0 - beta1 ** step_no, 1.0 - beta2 ** step_no
        next_params, next_avg, next_avg_sq = OrderedDict(), OrderedDict(), OrderedDict()
        for (name, param), grad in zip(params.items(), grads):
            grad = torch.zeros_like(param) if grad is None else grad
            if weight_decay:
                grad = grad + weight_decay * param
            mean = beta1 * exp_avg[name] + (1.0 - beta1) * grad
            mean_sq = beta2 * exp_avg_sq[name] + (1.0 - beta2) * grad.square()
            # The literal Adam denominator ``sqrt(v) + eps`` is forward-safe,
            # but its second derivative is singular at v=0. UFCL has inactive
            # attention/GRU paths with exactly-zero entries, and reconstruction
            # differentiates through this optimizer step. Add eps inside the
            # square root in the algebraically matched form: at v=0 it is still
            # eps, while the higher-order derivative remains finite.
            sqrt_bias2 = math.sqrt(bias2)
            denom = torch.sqrt(mean_sq + (eps * sqrt_bias2) ** 2) / sqrt_bias2
            next_params[name] = param - (lr / bias1) * mean / denom
            next_avg[name], next_avg_sq[name] = mean, mean_sq
        return next_params, next_avg, next_avg_sq

    def _ufcl_update(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        batch: Any,
        create_graph: bool,
    ) -> tuple[torch.Tensor, ...]:
        args = getattr(batch, "args")
        steps = int(getattr(batch, "ufcl_sequence_steps", 1))
        train_batch_size = int(getattr(batch, "ufcl_train_batch_size", 0))
        if train_batch_size <= 0 or x.shape[0] != steps * train_batch_size:
            raise ValueError("UFCL attack requires a trajectory shaped as local_update_steps x batch_size.")

        params, buffers = self._state(model, requires_grad=True)
        initial_params = OrderedDict((name, param) for name, param in params.items())
        teacher_params = OrderedDict((name, param.detach().clone()) for name, param in params.items())
        teacher_buffers = OrderedDict((name, buffer.detach().clone()) for name, buffer in buffers.items())
        exp_avg = OrderedDict((name, torch.zeros_like(param)) for name, param in params.items())
        exp_avg_sq = OrderedDict((name, torch.zeros_like(param)) for name, param in params.items())
        replay_x = replay_y = None

        for step_idx in range(steps):
            begin, end = step_idx * train_batch_size, (step_idx + 1) * train_batch_size
            real_x, real_y, real_nodes = self._pad_local_nodes(model, x[begin:end], y[begin:end])
            if replay_x is None:
                train_x, train_y = real_x, real_y
            else:
                lam = self._mix_lambda(int(args.seed) + 13007 + step_idx)
                train_x = lam * real_x + (1.0 - lam) * replay_x
                train_y = lam * real_y + (1.0 - lam) * replay_y

            pred = self._forward(model, params, buffers, train_x)
            pred, target = align_prediction_and_target(pred, train_y)
            mask = torch.zeros_like(pred)
            mask[:, :real_nodes] = 1.0
            task_loss = ((pred - target).square() * mask).sum() / (mask.sum() + 1e-9)
            with torch.no_grad():
                teacher_pred = self._forward(model, teacher_params, teacher_buffers, real_x)
                teacher_pred, _ = align_prediction_and_target(teacher_pred, real_y)
                syn_x = real_x + self._fixed_noise(real_x, int(args.seed) + 7919 + step_idx)
                syn_y = self._forward(model, teacher_params, teacher_buffers, syn_x)
            kd_loss = ((pred - teacher_pred).square() * mask).sum() / (mask.sum() + 1e-9)
            loss = task_loss + self._KD_WEIGHT * kd_loss
            grads = torch.autograd.grad(loss, tuple(params.values()), create_graph=create_graph, retain_graph=create_graph, allow_unused=True)
            params, exp_avg, exp_avg_sq = self._adam_step(params, grads, exp_avg, exp_avg_sq, step_idx + 1, args)
            # ReplayBuffer.push() detaches generated samples before the next
            # step; preserve that exact boundary in the differentiable model.
            replay_x, replay_y = syn_x.detach(), syn_y.detach()

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
        if leak_type == "model_update":
            return self._ufcl_update(model, x, y, batch, create_graph)
        if leak_type == "gradient":
            # Diagnostic upper bound only: expose the gradient of one ordinary
            # UFCL prediction batch before replay, mixup, KD, and local Adam
            # compress it into a model update.  This is not an UFCL message.
            params, buffers = self._state(model, requires_grad=True)
            x, y, real_nodes = self._pad_local_nodes(model, x, y)
            pred = self._forward(model, params, buffers, x)
            pred, target = align_prediction_and_target(pred, y)
            mask = torch.zeros_like(pred)
            mask[:, :real_nodes] = 1.0
            loss = ((pred - target).square() * mask).sum() / (mask.sum() + 1e-9)
            return torch.autograd.grad(
                loss,
                tuple(params.values()),
                create_graph=create_graph,
                retain_graph=create_graph,
                allow_unused=True,
            )
        raise ValueError(f"UFCL does not support privacy leak type: {leak_type}")
