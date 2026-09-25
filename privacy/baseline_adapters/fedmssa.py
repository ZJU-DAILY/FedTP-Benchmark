from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn

from lib.fedmssa_utils import (
    build_initial_basis_from_observation,
    build_page_observation,
    denoise_raw_series_with_observation,
    orthonormalize_basis,
)
from model.FedmSSA import FedmSSA_Model
from privacy.baseline_adapters.common import (
    align_prediction_and_target,
    args_from_batch,
    fixed_quantized_prediction,
    load_state_dict_if_available,
    loss_fn,
)

try:
    from torch.func import functional_call as _functional_call
except ImportError:  # pragma: no cover - depends on the installed torch version.
    from torch.nn.utils.stateless import functional_call as _functional_call


class FedmSSAPrivacyAdapter:
    name = "FedmSSA"
    default_attack = "model_update"
    requires_dummy_y = False
    # The federated trainer clips these client-to-Arbiter payload categories
    # independently.  In particular, the consensus basis is a server return
    # and must never be treated as an attacker-observed client upload.
    dp_payload_group_keys = (
        "phase1_init_basis",
        "phase1_local_deltas",
        "phase2_predictor_update",
    )

    def dp_payload_group_keys_for_leak(self, leak: Any, args: Any) -> tuple[str, ...]:
        """Return the DP-protected observation matching the chosen attack surface.

        ``model_update`` reproduces the three semantic uploads of the FedmSSA
        training protocol.  The legacy Plain reconstruction experiment instead
        uses the predictor gradient.  For a like-for-like DP reconstruction
        *upper bound*, protect that visible gradient as one complete payload;
        do not attempt to look up the model-update keys which are absent from
        the gradient leak tree.
        """
        if isinstance(leak, dict) and "phase2_predictor_gradient" in leak:
            return ("phase2_predictor_gradient",)
        return self.dp_payload_group_keys

    def build_model(self, args: Any, batch: Any) -> nn.Module:
        num_nodes = int(batch.real_x.shape[1])
        selected_nodes = getattr(batch, "selected_nodes", None)
        model = FedmSSA_Model(
            num_nodes=num_nodes,
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            out_dim=args.output_dim,
            pre_len=args.t_out,
            page_length=int(getattr(args, "fedmssa_page_length", max(4, args.t_in))),
            selected_nodes=selected_nodes,
            num_layers=int(getattr(args, "fedmssa_gru_layers", 2)),
            dropout=float(getattr(args, "fedmssa_dropout", 0.0)),
        ).to(args.device)
        load_state_dict_if_available(model, args)
        return model

    def _phase1_seed(self, args: Any, batch: Any) -> int:
        return int(getattr(args, "seed", 0)) + int(getattr(args, "client_rank", 0)) * 1000

    def _raw_series_from_dataset(self, batch: Any) -> torch.Tensor:
        dataset = getattr(batch, "dataset", None)
        if dataset is None:
            raise ValueError("FedmSSA requires batch.dataset to reconstruct the split-level raw series.")

        if hasattr(dataset, "flow_split"):
            raw = dataset.flow_split
            if torch.is_tensor(raw):
                return raw.detach().cpu().float().clone()
            return torch.as_tensor(raw, dtype=torch.float32).clone()

        tensors = getattr(dataset, "tensors", None)
        if not tensors or len(tensors) < 1 or not torch.is_tensor(tensors[0]):
            raise ValueError("FedmSSA requires a dataset exposing flow_split or TensorDataset tensors.")

        x_all = tensors[0].detach().cpu().float()
        if x_all.dim() != 4 or x_all.size(0) == 0:
            raise ValueError(f"Unexpected FedmSSA dataset tensor shape: {tuple(x_all.shape)}")
        first_window = x_all[0].permute(1, 0, 2).contiguous()
        tail = x_all[1:, :, -1, :].contiguous()
        if tail.numel() == 0:
            return first_window.clone()
        return torch.cat([first_window, tail], dim=0).clone()

    def _batch_input_series(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = int(x.shape[0])
        first_window = x[0].permute(1, 0, 2).contiguous()
        if batch_size == 1:
            return first_window
        tail = x[1:, :, -1, :].permute(0, 1, 2).contiguous()
        return torch.cat([first_window, tail], dim=0)

    def _inject_batch_into_raw_series(self, x: torch.Tensor, batch: Any) -> torch.Tensor:
        raw_series = self._raw_series_from_dataset(batch).to(device=x.device, dtype=x.dtype)
        segment = self._batch_input_series(x)
        start = int(getattr(batch, "sample_index", 0))
        end = min(start + segment.size(0), raw_series.size(0))
        usable = end - start
        if usable <= 0:
            raise ValueError("FedmSSA sample_index points outside the reconstructed raw series.")
        raw_series = raw_series.clone()
        raw_series[start:end] = segment[:usable]
        return raw_series

    def _build_observation_pack(self, x: torch.Tensor, batch: Any, args: Any) -> dict[str, torch.Tensor]:
        raw_series = self._inject_batch_into_raw_series(x, batch)
        base_seed = self._phase1_seed(args, batch)
        ratio = float(getattr(args, "fedmssa_missing_ratio", 0.0))
        page_length = int(getattr(args, "fedmssa_page_length", max(4, args.t_in)))
        scope = str(getattr(args, "fedmssa_series_scope", "full")).lower()
        raw_start = 0
        raw_end = int(raw_series.size(0))

        if scope == "local":
            sample_start = int(getattr(batch, "sample_index", 0))
            batch_size = int(x.shape[0])
            context_pages = max(0, int(getattr(args, "fedmssa_context_pages", 2)))
            context = context_pages * page_length
            target_end = sample_start + int(args.t_in) + int(args.t_out) + batch_size - 1
            raw_start = max(0, sample_start - context)
            raw_end = min(int(raw_series.size(0)), target_end + context)
            min_required = int(args.t_in) + int(args.t_out) + batch_size
            if (raw_end - raw_start) < min_required:
                raw_end = min(int(raw_series.size(0)), raw_start + min_required)
            raw_series = raw_series[raw_start:raw_end].contiguous()

        return {
            "train": build_page_observation(
                raw_series.to(device="cpu", dtype=torch.float32),
                page_length=page_length,
                missing_ratio=ratio,
                seed=base_seed + 11,
            ),
            "raw_start": raw_start,
            "raw_end": raw_end,
        }

    def _local_phase1_objective(
        self,
        observation: dict[str, torch.Tensor],
        basis: torch.Tensor,
        consensus_basis: torch.Tensor,
        args: Any,
    ) -> torch.Tensor:
        page_obs = observation["page_matrix_obs"].to(basis.device, dtype=basis.dtype)
        obs_mask = observation["obs_mask"].to(basis.device, dtype=basis.dtype)
        reconstruction = basis @ (basis.transpose(0, 1) @ page_obs)
        diff = obs_mask * (page_obs - reconstruction)
        recon = diff.pow(2).sum() / obs_mask.sum().clamp_min(1.0)

        consensus = (basis - consensus_basis).pow(2).mean()

        rank = int(basis.size(1))
        identity = torch.eye(rank, device=basis.device, dtype=basis.dtype)
        ortho = (basis.transpose(0, 1) @ basis - identity).pow(2).mean()

        weighted = page_obs * obs_mask
        covariance = weighted @ weighted.transpose(0, 1)
        projected = basis.transpose(0, 1) @ covariance @ basis
        off_diag = projected - torch.diag(torch.diagonal(projected))
        diag = off_diag.pow(2).mean()

        return (
            recon
            + float(getattr(args, "fedmssa_consensus_weight", 1.0)) * consensus
            + float(getattr(args, "fedmssa_ortho_weight", 1.0)) * ortho
            + float(getattr(args, "fedmssa_diag_weight", 0.2)) * diag
        )

    def _simulate_phase1(
        self,
        x: torch.Tensor,
        batch: Any,
        args: Any,
        create_graph: bool,
        keep_history: bool,
    ):
        with torch.enable_grad():
            observation_pack = self._build_observation_pack(x, batch, args)
            observation = observation_pack["train"]
            phase1_device = torch.device(getattr(args, "fedmssa_phase1_device", "cpu"))
            phase1_dtype = torch.float32
            page_obs = observation["page_matrix_obs"].to(device=phase1_device, dtype=phase1_dtype)

            init_basis = build_initial_basis_from_observation(
                page_obs,
                num_columns=int(observation["num_columns"]),
                explicit_rank=getattr(args, "fedmssa_rank", 0),
                sv_scale=getattr(args, "fedmssa_sv_scale", 2.0),
            ).to(device=phase1_device, dtype=phase1_dtype)
            consensus_basis = init_basis

            rounds = max(1, int(getattr(args, "fedmssa_impute_rounds", 20)))
            local_steps = max(1, int(getattr(args, "fedmssa_impute_local_steps", 10)))
            lr = float(getattr(args, "fedmssa_impute_lr", 5e-2))
            server_momentum = float(getattr(args, "fedmssa_server_momentum", 0.0))

            local_basis_sequence = [] if keep_history else None
            consensus_sequence = [] if keep_history else None
            server_velocity = torch.zeros_like(consensus_basis)
            last_local_basis = consensus_basis

            for _ in range(rounds):
                basis = orthonormalize_basis(consensus_basis.detach().clone(), rank=consensus_basis.size(1))
                basis = basis.to(device=phase1_device, dtype=phase1_dtype)

                for _ in range(local_steps):
                    basis = basis.detach().clone().requires_grad_(True)
                    objective = self._local_phase1_objective(observation, basis, consensus_basis, args)
                    grad = torch.autograd.grad(
                        objective,
                        basis,
                        create_graph=create_graph,
                        retain_graph=create_graph,
                        allow_unused=False,
                    )[0]
                    basis = basis - lr * grad
                    basis = orthonormalize_basis(basis, rank=consensus_basis.size(1))
                    if not create_graph:
                        basis = basis.detach()

                last_local_basis = basis
                if keep_history:
                    local_basis_sequence.append(basis)
                if server_momentum > 0.0:
                    server_velocity = server_momentum * server_velocity + (1.0 - server_momentum) * (basis - consensus_basis)
                    consensus_basis = orthonormalize_basis(consensus_basis + server_velocity, rank=consensus_basis.size(1))
                else:
                    consensus_basis = orthonormalize_basis(basis, rank=basis.size(1))
                if keep_history:
                    consensus_sequence.append(consensus_basis)

            return {
                "observation": observation,
                "raw_start": int(observation_pack.get("raw_start", 0)),
                "raw_end": int(observation_pack.get("raw_end", observation["raw_series"].size(0))),
                "init_basis": init_basis,
                "local_basis_sequence": tuple(local_basis_sequence) if keep_history else tuple(),
                "consensus_basis_sequence": tuple(consensus_sequence) if keep_history else tuple(),
                "last_local_basis": last_local_basis,
                "final_basis": consensus_basis,
            }

    def _phase1_gradient_summary(self, phase1_state: dict[str, Any]) -> dict[str, torch.Tensor]:
        init_basis = phase1_state["init_basis"]
        last_local_basis = phase1_state["last_local_basis"]
        final_basis = phase1_state["final_basis"]
        return {
            "phase1_init_basis": init_basis,
            "phase1_final_local_basis": last_local_basis,
            "phase1_final_consensus_basis": final_basis,
            "phase1_basis_shift": final_basis - init_basis,
        }

    def _phase1_activation_summary(self, phase1_state: dict[str, Any], args: Any) -> dict[str, Any]:
        surface = str(getattr(args, "fedmssa_activation_surface", "phase1_trace")).lower()
        summary: dict[str, Any] = {
            "phase1_init_basis": phase1_state["init_basis"],
            "phase1_final_local_basis": phase1_state["last_local_basis"],
            "phase1_final_consensus_basis": phase1_state["final_basis"],
            "phase1_basis_shift": phase1_state["final_basis"] - phase1_state["init_basis"],
        }
        if surface in ("phase1_local", "phase1_local_consensus", "phase1_trace"):
            summary["phase1_local_basis_sequence"] = phase1_state["local_basis_sequence"]
        if surface in ("phase1_local_consensus", "phase1_trace"):
            summary["phase1_consensus_basis_sequence"] = phase1_state["consensus_basis_sequence"]
        return summary

    def _extract_window_batch_from_series(
        self,
        raw_series: torch.Tensor,
        *,
        start: int,
        batch_size: int,
        t_in: int,
        t_out: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xs = []
        ys = []
        time_steps = int(raw_series.size(0))
        max_start = time_steps - int(t_in) - int(t_out)
        if start < 0 or start + batch_size - 1 > max_start:
            raise IndexError(
                f"FedmSSA requested window range [{start}, {start + batch_size}) "
                f"outside valid 0..{max_start}"
            )
        for idx in range(start, start + batch_size):
            xs.append(raw_series[idx:idx + t_in])
            ys.append(raw_series[idx + t_in:idx + t_in + t_out])
        x_tensor = torch.stack(xs, dim=0).permute(0, 2, 1, 3).contiguous()
        y_tensor = torch.stack(ys, dim=0).permute(0, 2, 1, 3).contiguous()
        return x_tensor.float(), y_tensor.float()

    def _phase2_batch_from_basis(self, x: torch.Tensor, batch: Any, args: Any, phase1_state: dict[str, Any]):
        observation = phase1_state["observation"]
        basis = phase1_state["final_basis"]
        raw_series = denoise_raw_series_with_observation(
            observation,
            basis.to(device="cpu", dtype=torch.float32),
            page_length=int(getattr(args, "fedmssa_page_length", max(4, args.t_in))),
        )
        global_start = int(getattr(batch, "sample_index", 0))
        local_offset = int(phase1_state.get("raw_start", 0))
        start = global_start - local_offset
        batch_size = int(x.shape[0])
        denoised_x, denoised_y = self._extract_window_batch_from_series(
            raw_series,
            start=start,
            batch_size=batch_size,
            t_in=args.t_in,
            t_out=args.t_out,
        )
        denoised_x = denoised_x.to(device=x.device, dtype=x.dtype)
        denoised_y = denoised_y.to(device=x.device, dtype=x.dtype)
        return denoised_x, denoised_y

    def _task_loss(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch: Any) -> torch.Tensor:
        with torch.backends.cudnn.flags(enabled=False):
            pred = model(x)
        pred, y = align_prediction_and_target(pred, y)
        return loss_fn(args_from_batch(batch))(pred, y)

    def _selected_gradient_leak(
        self,
        model: nn.Module,
        loss: torch.Tensor,
        batch: Any,
        create_graph: bool,
    ) -> dict[str, torch.Tensor]:
        args = args_from_batch(batch)
        scope = str(getattr(args, "fedmssa_gradient_scope", "head")).lower()

        if scope == "full":
            named_params = [
                (name, param)
                for name, param in model.named_parameters()
                if param.requires_grad
            ]
        elif scope == "predictor":
            named_params = [
                (name, param)
                for name, param in model.named_parameters()
                if param.requires_grad and name.startswith("predictor.")
            ]
        else:
            named_params = [
                (name, param)
                for name, param in model.named_parameters()
                if param.requires_grad and name.startswith("predictor.head.")
            ]

        params = [param for _, param in named_params]
        grads = torch.autograd.grad(
            loss,
            params,
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=True,
        )
        return {
            name: (torch.zeros_like(param) if grad is None else grad)
            for (name, param), grad in zip(named_params, grads)
        }

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

    def _sgd_or_adam_update(
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
        update_optimizer = getattr(args, "fedmssa_update_optimizer", "sgd").lower()

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
                beta1 = float(getattr(args, "fedmssa_adam_beta1", 0.9))
                beta2 = float(getattr(args, "fedmssa_adam_beta2", 0.999))
                eps = float(getattr(args, "fedmssa_adam_eps", 1e-8))
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
        args = args_from_batch(batch)
        if leak_type == "activation" and bool(getattr(args, "quantized_prediction_sidechannel", False)):
            with torch.backends.cudnn.flags(enabled=False):
                return fixed_quantized_prediction(model(x), args)
        if leak_type == "gradient":
            phase1_state = self._simulate_phase1(
                x,
                batch,
                args,
                create_graph=False,
                keep_history=False,
            )
        else:
            phase1_state = self._simulate_phase1(
                x,
                batch,
                args,
                create_graph=False,
                keep_history=True,
            )

        if leak_type == "activation":
            return self._phase1_activation_summary(phase1_state, args)

        denoised_x, denoised_y = self._phase2_batch_from_basis(x, batch, args, phase1_state)

        if leak_type == "gradient":
            loss = self._task_loss(model, denoised_x, denoised_y, batch)
            # For FedmSSA, including phase-1 basis tensors in the differentiable
            # gradient leak makes the reconstruction graph extremely large on
            # TaxiBJ-size grid data and stalls on higher-order backward.
            # The realistic gradient-bearing signal for optimization is the
            # phase-2 predictor gradient; use that as the attack target.
            return {
                "phase2_predictor_gradient": self._selected_gradient_leak(
                    model=model,
                    loss=loss,
                    batch=batch,
                    create_graph=create_graph,
                )
            }

        if leak_type != "model_update":
            raise ValueError("FedmSSA currently supports gradient, activation, and model_update reconstruction only.")

        # The real phase-1 protocol uploads the initial local basis once, then
        # uploads a local *delta* in each consensus round.  The corresponding
        # consensus bases are downlink messages and are intentionally omitted.
        previous_bases = (phase1_state["init_basis"],) + phase1_state["consensus_basis_sequence"][:-1]
        local_deltas = tuple(
            local_basis - previous_basis
            for local_basis, previous_basis in zip(
                phase1_state["local_basis_sequence"], previous_bases
            )
        )
        return {
            "phase1_init_basis": phase1_state["init_basis"],
            "phase1_local_deltas": local_deltas,
            "phase2_predictor_update": self._sgd_or_adam_update(
                model=model,
                x=denoised_x,
                y=denoised_y,
                batch=batch,
                create_graph=create_graph,
            ),
        }
