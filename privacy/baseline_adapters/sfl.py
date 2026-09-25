from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from privacy.baseline_adapters.common import align_prediction_and_target, loss_fn

try:
    from torch.func import functional_call as _functional_call
except ImportError:  # pragma: no cover - depends on the installed torch version.
    from torch.nn.utils.stateless import functional_call as _functional_call


class SFLPrivacyAdapter:
    name = "SFL"
    default_attack = "model_update"
    requires_dummy_y = True

    def __init__(self) -> None:
        self.base_adapter: Any | None = None
        self.base_model_name: str | None = None

    def _resolve_base_adapter(self, args: Any):
        base_model = getattr(args, "sfl_base_model", "FedGRU")
        if base_model == self.name:
            raise ValueError("SFL requires a concrete --sfl_base_model, not SFL itself.")

        if self.base_adapter is not None and self.base_model_name == base_model:
            return self.base_adapter

        from privacy.privacy_registry import get_adapter

        self.base_adapter = get_adapter(base_model)
        self.base_model_name = base_model
        return self.base_adapter

    def build_model(self, args: Any, batch: Any) -> nn.Module:
        base_adapter = self._resolve_base_adapter(args)
        model = base_adapter.build_model(args, batch)
        setattr(model, "_sfl_base_model", self.base_model_name)
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
                outputs = _functional_call(model, state, (x,))
        finally:
            model.train(was_training)

        pred = outputs[0] if isinstance(outputs, tuple) else outputs
        pred, target = align_prediction_and_target(pred, y)
        return loss_fn(args)(pred, target)

    def _regularizer_loss(
        self,
        params: OrderedDict[str, torch.Tensor],
        global_anchor: OrderedDict[str, torch.Tensor] | None,
        personal_anchor: OrderedDict[str, torch.Tensor] | None,
        args: Any,
    ) -> torch.Tensor:
        if global_anchor is None and personal_anchor is None:
            return next(iter(params.values())).new_zeros(())

        dist_w = next(iter(params.values())).new_zeros(())
        dist_u = next(iter(params.values())).new_zeros(())
        for name, param in params.items():
            if global_anchor is not None and name in global_anchor and param.shape == global_anchor[name].shape:
                dist_w = dist_w + F.mse_loss(param, global_anchor[name].detach())
            if personal_anchor is not None and name in personal_anchor and param.shape == personal_anchor[name].shape:
                dist_u = dist_u + F.mse_loss(param, personal_anchor[name].detach())
        return float(getattr(args, "sfl_lambda", 0.1)) * (dist_w + dist_u)

    def _anchor_state(
        self,
        initial_params: OrderedDict[str, torch.Tensor],
        args: Any,
    ) -> tuple[OrderedDict[str, torch.Tensor] | None, OrderedDict[str, torch.Tensor] | None]:
        anchor_mode = getattr(args, "sfl_reg_anchor", "initial")
        if anchor_mode in ("none", "server"):
            return None, None
        anchor = OrderedDict((name, param.detach().clone()) for name, param in initial_params.items())
        return anchor, OrderedDict((name, param.detach().clone()) for name, param in initial_params.items())

    def _clone_state(
        self,
        params: OrderedDict[str, torch.Tensor],
        *,
        detach: bool,
    ) -> OrderedDict[str, torch.Tensor]:
        copied = OrderedDict()
        for name, param in params.items():
            tensor = param.detach().clone() if detach else param.clone()
            copied[name] = tensor
        return copied

    def _state_to_tuple(self, params: OrderedDict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        return tuple(params[name] for name in params.keys())

    def _shared_state_keys(self, states: list[OrderedDict[str, torch.Tensor]]) -> list[str]:
        if not states:
            return []
        keys = list(states[0].keys())
        valid: list[str] = []
        for key in keys:
            shape0 = states[0][key].shape
            if all((key in state) and (state[key].shape == shape0) for state in states[1:]):
                valid.append(key)
        return valid

    def _flatten_state(
        self,
        state: OrderedDict[str, torch.Tensor],
        keys: list[str],
        device: torch.device,
    ) -> torch.Tensor:
        return torch.cat([state[key].reshape(-1).to(device) for key in keys], dim=0)

    def _surrogate_client_states(
        self,
        initial_params: OrderedDict[str, torch.Tensor],
        args: Any,
    ) -> list[OrderedDict[str, torch.Tensor]]:
        requested = int(getattr(args, "sfl_num_surrogate_clients", 0) or 0)
        if requested <= 0:
            requested = max(int(getattr(args, "num_clients", 4) or 4) - 1, 3)
        return [self._clone_state(initial_params, detach=True) for _ in range(requested)]

    def _server_structure_aggregate(
        self,
        attacked_state: OrderedDict[str, torch.Tensor],
        peer_states: list[OrderedDict[str, torch.Tensor]],
        args: Any,
    ) -> tuple[OrderedDict[str, torch.Tensor], OrderedDict[str, torch.Tensor], torch.Tensor]:
        states = [attacked_state] + peer_states
        keys = self._shared_state_keys(states)
        if not keys:
            raise ValueError("SFL server aggregation found no shared parameters.")

        device = next(iter(attacked_state.values())).device
        stacked = torch.stack([self._flatten_state(state, keys, device) for state in states], dim=0)

        if stacked.shape[0] == 1:
            adjacency = torch.ones((1, 1), device=device, dtype=stacked.dtype)
            personalized = stacked
        else:
            normalized = F.normalize(stacked, p=2, dim=1)
            sim = torch.matmul(normalized, normalized.T)
            sim = 0.5 * (sim + sim.T)
            eye = torch.eye(sim.shape[0], device=device, dtype=torch.bool)
            sim = torch.where(eye, torch.ones_like(sim), sim)

            gamma = max(float(getattr(args, "sfl_gamma", 0.01) or 0.01), 1e-6)
            topk = int(getattr(args, "sfl_topk", 0) or 0)
            if topk <= 0:
                topk = min(max(2, int(math.ceil(math.sqrt(sim.shape[0])))), sim.shape[0])

            logits = sim / gamma
            if topk < sim.shape[0]:
                _, indices = torch.topk(sim, k=topk, dim=1)
                mask = torch.zeros_like(sim, dtype=torch.bool)
                mask.scatter_(1, indices, True)
                mask = torch.logical_or(mask, mask.T)
                logits = logits.masked_fill(~mask, float("-inf"))
            adjacency = torch.softmax(logits, dim=1)

            personalized = stacked
            for _ in range(max(int(getattr(args, "sfl_m_steps", 1) or 1), 1)):
                personalized = torch.matmul(adjacency, personalized)

        readout = torch.ones((personalized.shape[0],), device=device, dtype=personalized.dtype)
        readout = readout / max(int(readout.numel()), 1)
        global_vec = torch.matmul(readout.unsqueeze(0), personalized).squeeze(0)

        shapes = {name: states[0][name].shape for name in keys}
        numels = {name: states[0][name].numel() for name in keys}

        personalized_attacked = OrderedDict()
        ptr = 0
        for name in keys:
            numel = numels[name]
            personalized_attacked[name] = personalized[0, ptr:ptr + numel].view(shapes[name])
            ptr += numel

        global_anchor = OrderedDict()
        ptr = 0
        for name in keys:
            numel = numels[name]
            global_anchor[name] = global_vec[ptr:ptr + numel].view(shapes[name])
            ptr += numel

        return global_anchor, personalized_attacked, adjacency

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
        beta1 = float(getattr(args, "sfl_adam_beta1", 0.9))
        beta2 = float(getattr(args, "sfl_adam_beta2", 0.999))
        eps = float(getattr(args, "sfl_adam_eps", 1e-8))
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

    def _server_graph_vector(
        self,
        tensors: tuple[torch.Tensor, ...],
        args: Any,
    ) -> torch.Tensor:
        flat = torch.cat([tensor.reshape(-1) for tensor in tensors])
        graph_weight = max(0.0, float(getattr(args, "sfl_graph_weight", 1.0)))
        return F.normalize(flat, p=2, dim=0) * math.sqrt(graph_weight)

    def _run_local_epoch(
        self,
        model: nn.Module,
        start_params: OrderedDict[str, torch.Tensor],
        buffers: OrderedDict[str, torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
        args: Any,
        global_anchor: OrderedDict[str, torch.Tensor] | None,
        personal_anchor: OrderedDict[str, torch.Tensor] | None,
        create_graph: bool,
    ) -> OrderedDict[str, torch.Tensor]:
        steps = max(1, int(getattr(args, "local_update_steps", 1)))
        lr = float(getattr(args, "local_update_lr", 1e-3))
        wd_arg = getattr(args, "local_update_wd", None)
        weight_decay = float(getattr(args, "wd", 0.0) if wd_arg is None else wd_arg)
        update_optimizer = getattr(args, "sfl_update_optimizer", "adam").lower()
        if update_optimizer not in ("sgd", "adam"):
            raise ValueError(f"Unsupported SFL update optimizer: {update_optimizer}")
        params = OrderedDict((name, param.requires_grad_(True)) for name, param in start_params.items())
        exp_avg = OrderedDict((name, torch.zeros_like(param)) for name, param in params.items())
        exp_avg_sq = OrderedDict((name, torch.zeros_like(param)) for name, param in params.items())

        for step_idx in range(steps):
            task_loss = self._forward_loss_with_state(model, params, buffers, x, y, args)
            reg_loss = self._regularizer_loss(params, global_anchor, personal_anchor, args)
            loss = task_loss + reg_loss
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
                params = self._sgd_step(params=params, grads=grads, weight_decay=weight_decay, lr=lr)
        return params

    def _legacy_differentiable_local_update(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        batch: Any,
        create_graph: bool,
    ):
        args = getattr(batch, "args", None)
        if args is None:
            raise ValueError("SFL adapter requires batch.args to compute local update settings.")
        initial_params = OrderedDict(
            (name, param.detach().clone().requires_grad_(True))
            for name, param in model.named_parameters()
            if param.requires_grad
        )
        buffers = OrderedDict((name, buffer.detach().clone()) for name, buffer in model.named_buffers())
        global_anchor, personal_anchor = self._anchor_state(initial_params, args)
        params = self._run_local_epoch(
            model=model,
            start_params=initial_params,
            buffers=buffers,
            x=x,
            y=y,
            args=args,
            global_anchor=global_anchor,
            personal_anchor=personal_anchor,
            create_graph=create_graph,
        )

        updates = tuple(params[name] - initial_params[name] for name in initial_params.keys())
        if getattr(args, "sfl_attack_surface", "v_graph") == "v":
            return updates

        uploaded_v = tuple(params[name] for name in initial_params.keys())
        return {
            "v_update": updates,
            "server_graph_vector": self._server_graph_vector(uploaded_v, args),
        }

    def _realistic_server_update(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        batch: Any,
        create_graph: bool,
    ):
        args = getattr(batch, "args", None)
        if args is None:
            raise ValueError("SFL adapter requires batch.args to compute local update settings.")

        initial_params = OrderedDict(
            (name, param.detach().clone().requires_grad_(True))
            for name, param in model.named_parameters()
            if param.requires_grad
        )
        buffers = OrderedDict((name, buffer.detach().clone()) for name, buffer in model.named_buffers())
        peer_states = self._surrogate_client_states(initial_params, args)
        params = initial_params
        global_anchor = None
        personal_anchor = None
        final_adjacency = None

        unroll_epochs = max(1, int(getattr(args, "sfl_server_unroll_epochs", 2) or 2))
        for _ in range(unroll_epochs):
            params = self._run_local_epoch(
                model=model,
                start_params=params,
                buffers=buffers,
                x=x,
                y=y,
                args=args,
                global_anchor=global_anchor,
                personal_anchor=personal_anchor,
                create_graph=create_graph,
            )
            global_anchor, personal_anchor, final_adjacency = self._server_structure_aggregate(params, peer_states, args)

        uploaded_v = self._state_to_tuple(params)
        if getattr(args, "sfl_attack_surface", "v_graph") == "v":
            return uploaded_v

        graph_vector = self._server_graph_vector(uploaded_v, args)
        leak = {
            "uploaded_v": uploaded_v,
            "server_graph_vector": graph_vector,
        }
        if final_adjacency is not None:
            leak["server_graph_matrix"] = final_adjacency
        return leak

    def compute_leak(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        batch: Any,
        create_graph: bool,
        leak_type: str,
    ):
        args = getattr(batch, "args", None)
        if args is None:
            raise ValueError("SFL adapter requires batch.args.")

        base_adapter = self._resolve_base_adapter(args)
        if leak_type == "activation" and bool(getattr(args, "quantized_prediction_sidechannel", False)):
            return base_adapter.compute_leak(
                model=model, x=x, y=y, batch=batch, create_graph=create_graph, leak_type="activation"
            )
        if leak_type == "gradient":
            return base_adapter.compute_leak(
                model=model,
                x=x,
                y=y,
                batch=batch,
                create_graph=create_graph,
                leak_type="gradient",
            )

        if leak_type != "model_update":
            raise ValueError("SFL currently supports gradient and model_update reconstruction only.")

        if getattr(args, "sfl_reg_anchor", "none") == "server":
            return self._realistic_server_update(model, x, y, batch, create_graph)

        return self._legacy_differentiable_local_update(model, x, y, batch, create_graph)
