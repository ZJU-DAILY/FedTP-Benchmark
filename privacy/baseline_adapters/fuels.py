from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.FUELS import FUELS_Model
from privacy.baseline_adapters.common import (
    args_from_batch,
    fixed_quantized_prediction,
    load_state_dict_if_available,
)
from privacy.baseline_adapters.graph_common import prediction_gradient


class FUELSPrivacyAdapter:
    name = "FUELS"
    default_attack = "activation"
    requires_dummy_y = True

    @staticmethod
    def _protocol_batch_size(args: Any) -> int:
        """Batch size used by the real client when constructing ``R_n``.

        ``args.batch_size`` belongs to the attacked dummy batch and is often
        one.  Reusing it here silently changes FUELS' client-wide prototype
        protocol, so a separate option is required for a faithful attack.
        """
        configured = int(getattr(args, "fuels_protocol_batch_size", 0) or 0)
        return max(1, configured if configured > 0 else int(getattr(args, "batch_size", 1)))

    def build_model(self, args: Any, batch: Any) -> nn.Module:
        num_nodes = int(batch.real_x.shape[1])
        # TaxiBJ flow contains inflow/outflow channels.  The reconstruction
        # model must follow the actual batch schema rather than an inherited
        # graph-data default of input_dim=1.
        input_dim = int(batch.real_x.shape[-1])
        output_dim = int(batch.real_y.shape[-1])
        model = FUELS_Model(
            num_nodes=num_nodes,
            in_dim=input_dim,
            out_dim=output_dim,
            hidden_dim=args.hidden_dim,
            dr=getattr(args, "fuels_dr", 64),
            batch_size=max(1, int(getattr(args, "batch_size", 1))),
            seq_len=args.t_in,
            pred_len=args.t_out,
            device=args.device,
            fuels_c=getattr(args, "fuels_c", 3),
            fuels_q=getattr(args, "fuels_q", 3),
            aug_noise_std=getattr(args, "fuels_aug_noise_std", 0.01),
            aug_mask_ratio=getattr(args, "fuels_aug_mask_ratio", 0.10),
            aug_shift_prob=getattr(args, "fuels_aug_shift_prob", 0.50),
            aug_shift_pad_mode=getattr(args, "fuels_aug_shift_pad_mode", "edge"),
        ).to(args.device)
        load_state_dict_if_available(model, args)
        return model

    def _sample_x(self, dataset: Any, idx: int, batch: Any) -> torch.Tensor:
        sample = dataset[idx]
        if not isinstance(sample, (tuple, list)) or len(sample) < 1:
            raise TypeError("FUELS expects dataset samples to expose x as the first tuple element.")
        x = sample[0]
        if not torch.is_tensor(x) or x.dim() != 3:
            raise ValueError(f"FUELS expects 3D sample x, got {type(x)!r} shape={getattr(x, 'shape', None)}")
        num_nodes = int(batch.real_x.shape[1])
        if x.shape[0] == num_nodes:
            return x.contiguous().float()
        if x.shape[1] == num_nodes:
            return x.permute(1, 0, 2).contiguous().float()
        raise ValueError(f"FUELS cannot align sample shape {tuple(x.shape)} with num_nodes={num_nodes}")

    def _stack_x_batch(self, xs: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(xs, dim=0)

    def _build_proto_batch(
        self,
        dataset: Any,
        batch: Any,
        proto_start: int,
        proto_bs: int,
        dummy_x: torch.Tensor,
    ) -> torch.Tensor:
        attack_start = int(getattr(batch, "sample_index", 0))
        attack_end = attack_start + int(dummy_x.shape[0])
        xs = []
        for idx in range(proto_start, proto_start + proto_bs):
            if attack_start <= idx < attack_end:
                xs.append(dummy_x[idx - attack_start])
            else:
                xs.append(self._sample_x(dataset, idx, batch).to(device=dummy_x.device, dtype=dummy_x.dtype))
        return self._stack_x_batch(xs)

    def _cached_fixed_proto(
        self,
        model: nn.Module,
        batch: Any,
        args: Any,
    ) -> tuple[torch.Tensor, int, int]:
        cached = getattr(batch, "_fuels_fixed_proto_cache", None)
        attack_start = int(getattr(batch, "sample_index", 0))
        proto_bs = self._protocol_batch_size(args)
        target_batch_start = (attack_start // proto_bs) * proto_bs
        if cached is not None and cached.get("target_batch_start") == target_batch_start:
            return cached["fixed_sum"], cached["proto_count"], cached["target_batch_start"]

        dataset = getattr(batch, "dataset", None)
        if dataset is None:
            raise ValueError("FUELS prototype leak requires batch.dataset.")
        dataset_len = len(dataset)
        full_steps = dataset_len // proto_bs
        if full_steps <= 0:
            raise RuntimeError("FUELS prototype construction needs at least one full proto batch.")

        fixed_sum = None
        proto_count = 0
        model.eval()
        with torch.no_grad():
            for step_idx in range(full_steps):
                proto_start = step_idx * proto_bs
                if proto_start == target_batch_start:
                    continue
                xs = [
                    self._sample_x(dataset, idx, batch).to(device=args.device, dtype=batch.real_x.dtype)
                    for idx in range(proto_start, proto_start + proto_bs)
                ]
                x_proto = self._stack_x_batch(xs)
                r_proto = model.encode(x_proto).detach()
                if fixed_sum is None:
                    fixed_sum = torch.zeros_like(r_proto)
                fixed_sum = fixed_sum + r_proto
                proto_count += 1

        if fixed_sum is None:
            fixed_sum = torch.zeros(
                proto_bs,
                int(getattr(args, "fuels_dr", 64)),
                device=args.device,
                dtype=batch.real_x.dtype,
            )

        cache = {
            "target_batch_start": target_batch_start,
            "fixed_sum": fixed_sum.detach(),
            "proto_count": proto_count,
        }
        setattr(batch, "_fuels_fixed_proto_cache", cache)
        return fixed_sum, proto_count, target_batch_start

    def _cached_proto_bank(
        self,
        model: nn.Module,
        batch: Any,
        args: Any,
    ) -> tuple[list[torch.Tensor], int]:
        cached = getattr(batch, "_fuels_proto_bank_cache", None)
        attack_start = int(getattr(batch, "sample_index", 0))
        proto_bs = self._protocol_batch_size(args)
        target_batch_start = (attack_start // proto_bs) * proto_bs
        if cached is not None and cached.get("target_batch_start") == target_batch_start:
            return cached["proto_bank"], cached["target_batch_start"]

        dataset = getattr(batch, "dataset", None)
        if dataset is None:
            raise ValueError("FUELS prototype bank construction requires batch.dataset.")

        dataset_len = len(dataset)
        full_steps = dataset_len // proto_bs
        proto_bank: list[torch.Tensor] = []
        model.eval()
        with torch.no_grad():
            for step_idx in range(full_steps):
                proto_start = step_idx * proto_bs
                if proto_start == target_batch_start:
                    continue
                xs = [
                    self._sample_x(dataset, idx, batch).to(device=args.device, dtype=batch.real_x.dtype)
                    for idx in range(proto_start, proto_start + proto_bs)
                ]
                x_proto = self._stack_x_batch(xs)
                proto_bank.append(model.encode(x_proto).detach())

        cache = {
            "target_batch_start": target_batch_start,
            "proto_bank": proto_bank,
        }
        setattr(batch, "_fuels_proto_bank_cache", cache)
        return proto_bank, target_batch_start

    def _select_surrogate_prototypes(
        self,
        proto_bank: list[torch.Tensor],
        args: Any,
    ) -> list[torch.Tensor]:
        if not proto_bank:
            return []
        requested = int(getattr(args, "fuels_num_surrogate_clients", 0) or 0)
        if requested <= 0:
            requested = max(int(getattr(args, "num_clients", 4) or 4) - 1, 1)
        requested = min(requested, len(proto_bank))
        if requested >= len(proto_bank):
            return proto_bank
        indices = torch.linspace(0, len(proto_bank) - 1, steps=requested).round().long().tolist()
        return [proto_bank[idx] for idx in indices]

    def _compute_jsd_distance(self, p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        p = p.clamp_min(eps)
        q = q.clamp_min(eps)
        m = 0.5 * (p + q)
        kl_pm = (p * (p.log() - m.log())).sum(dim=-1).mean()
        kl_qm = (q * (q.log() - m.log())).sum(dim=-1).mean()
        return 0.5 * (kl_pm + kl_qm)

    def _server_feedback_prototypes(
        self,
        local_proto: torch.Tensor,
        model: nn.Module,
        batch: Any,
        args: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        proto_bank, _ = self._cached_proto_bank(model, batch, args)
        peers = self._select_surrogate_prototypes(proto_bank, args)
        if not peers:
            zeros = torch.zeros_like(local_proto)
            return local_proto, zeros

        peer_probs = [F.softmax(peer.to(device=local_proto.device, dtype=local_proto.dtype), dim=-1) for peer in peers]
        local_prob = F.softmax(local_proto, dim=-1)
        local_jsds = torch.stack([self._compute_jsd_distance(local_prob, peer_prob) for peer_prob in peer_probs], dim=0)

        peer_pair_jsds: list[torch.Tensor] = []
        for i in range(len(peer_probs)):
            for j in range(i + 1, len(peer_probs)):
                peer_pair_jsds.append(self._compute_jsd_distance(peer_probs[i], peer_probs[j]).detach())
        if peer_pair_jsds:
            active_jsds = torch.stack(peer_pair_jsds, dim=0)
            beta = torch.quantile(active_jsds, float(getattr(args, "fuels_beta_percentile", 50.0)) / 100.0)
        else:
            beta = local_jsds.detach().median()

        temp = max(float(getattr(args, "fuels_feedback_temperature", 0.05) or 0.05), 1e-4)
        pos_weights = torch.sigmoid((beta - local_jsds) / temp)
        neg_weights = 1.0 - pos_weights

        peer_stack = torch.stack([peer.to(device=local_proto.device, dtype=local_proto.dtype) for peer in peers], dim=0)
        pos_den = pos_weights.sum().clamp_min(1e-6)
        neg_den = neg_weights.sum().clamp_min(1e-6)

        positive_proto = (local_proto + (pos_weights.view(-1, 1, 1) * peer_stack).sum(dim=0)) / (1.0 + pos_den)
        if torch.all(neg_weights <= 1e-6):
            negative_proto = torch.zeros_like(local_proto)
        else:
            negative_proto = (neg_weights.view(-1, 1, 1) * peer_stack).sum(dim=0) / neg_den
        return positive_proto, negative_proto

    def _local_prototype(self, model: nn.Module, x: torch.Tensor, batch: Any, args: Any) -> torch.Tensor:
        proto_bs = self._protocol_batch_size(args)
        fixed_sum, proto_count, target_batch_start = self._cached_fixed_proto(model, batch, args)
        dataset = getattr(batch, "dataset", None)
        proto_x = self._build_proto_batch(dataset, batch, target_batch_start, proto_bs, x)
        r_proto = model.encode(proto_x)
        total_sum = fixed_sum.to(device=x.device, dtype=x.dtype) + r_proto
        total_count = proto_count + 1
        R_n = total_sum / float(max(total_count, 1))

        dp_noise = float(getattr(args, "dp_noise", 0.0))
        if dp_noise > 0.0:
            # Privacy runner evaluates deterministic leakage. Keep the expected
            # uploaded prototype rather than sampling fresh noise each step.
            R_n = R_n + torch.zeros_like(R_n)
        return R_n

    def _activation_leak(self, model: nn.Module, x: torch.Tensor, batch: Any, args: Any) -> dict[str, torch.Tensor]:
        scope = getattr(args, "fuels_activation_scope", "prototype_batch")
        self.attack_surface = f"fuels_{scope}"
        self.single_sample_representation_upper_bound = scope == "batch_repr_upper"

        # Diagnostic upper bound: the server is assumed to see the attacked
        # batch's individual encoder output rather than FUELS' uploaded
        # client-wide prototype. Do not add prototype or augmentation terms.
        if scope == "batch_repr_upper":
            return {"batch_repr": model.encode(x)}

        leak = {"prototype": self._local_prototype(model, x, batch, args)}

        if scope in ("prototype_batch", "prototype_batch_aug", "prototype_batch_aug_prnr"):
            batch_repr = model.encode(x)
            leak["batch_repr"] = batch_repr

        if scope in ("prototype_batch_aug", "prototype_batch_aug_prnr"):
            x_prime = model.make_augmented_view(x)
            leak["batch_aug_repr"] = model.encode(x_prime)

        if scope == "prototype_batch_aug_prnr":
            positive_proto, negative_proto = self._server_feedback_prototypes(leak["prototype"], model, batch, args)
            leak["positive_proto"] = positive_proto
            leak["negative_proto"] = negative_proto

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
        if leak_type not in ("gradient", "activation"):
            raise ValueError("FUELS currently supports gradient and activation/prototype reconstruction only.")

        args = args_from_batch(batch)
        was_training = model.training
        model.train()
        try:
            with torch.backends.cudnn.flags(enabled=False):
                if leak_type == "activation" and bool(
                    getattr(args, "quantized_prediction_sidechannel", False)
                ):
                    # This explicit revised-protocol surface replaces the
                    # normal FUELS prototype leakage for this ablation.
                    return fixed_quantized_prediction(model(x), args)
                activation_leak = self._activation_leak(model, x, batch, args)
                if leak_type == "activation":
                    return activation_leak

                pred = model(x)
                grad_leak = prediction_gradient(model=model, pred=pred, y=y, batch=batch, create_graph=create_graph)
                return {
                    **activation_leak,
                    "predictor_gradient": grad_leak,
                }
        finally:
            model.train(was_training)
