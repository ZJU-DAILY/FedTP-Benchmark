from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn

from model.FedSTN import FedSTN
from privacy.baseline_adapters.common import fixed_quantized_prediction, load_state_dict_if_available
from privacy.baseline_adapters.graph_common import edge_index_from_batch, prediction_gradient


class FedSTNPrivacyAdapter:
    name = "FedSTN"
    default_attack = "activation"
    requires_dummy_y = True
    attack_surface = "fedstn_hs_context_agg"
    batch_aggregate_only = False

    def dp_payload_group_keys_for_leak(self, real_leak: Any, args: Any) -> tuple[str, ...]:
        """Return only client-to-arbiter payloads protected in real FedSTN.

        The trainer applies DP to ``h_s_i`` before the arbiter derives the
        attention context and aggregate.  The latter two are not independent
        uploads and must never receive separate noise in reconstruction.
        """
        if isinstance(real_leak, Mapping) and "h_s_i" in real_leak:
            return ("h_s_i",)
        return ()

    def postprocess_dp_observed_leak(self, protected: Any, args: Any) -> Any:
        """Rebuild arbiter-derived observations from the noisy hidden state."""
        if not isinstance(protected, Mapping) or "h_s_i" not in protected:
            return protected
        if str(getattr(args, "fedstn_activation_surface", "hs_context_agg")) != "hs_context_agg":
            return protected

        hs_weight = float(getattr(args, "fedstn_hs_weight", 1.0))
        hs_scale = math.sqrt(max(0.0, hs_weight))
        # ``h_s_i`` is stored after matching-loss weighting.  Undo that
        # weighting before running the server proxy, then restore the weights
        # used by the attacker objective below.
        noisy_hs = protected["h_s_i"] if hs_scale == 0.0 else protected["h_s_i"] / hs_scale
        server_attention_context, agg_hs = self._server_attention_proxy(noisy_hs)
        rebuilt = dict(protected)
        rebuilt["server_attention_context"] = self._weighted(
            server_attention_context,
            float(getattr(args, "fedstn_context_weight", 0.1)),
        )
        rebuilt["agg_hs"] = self._weighted(
            agg_hs,
            float(getattr(args, "fedstn_agg_weight", 0.1)),
        )
        return rebuilt

    def build_model(self, args: Any, batch: Any) -> nn.Module:
        num_nodes = int(batch.real_x.shape[1])
        edge_index = edge_index_from_batch(batch, num_nodes, args.device)
        model = FedSTN(
            num_nodes=num_nodes,
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            out_dim=args.t_out,
            edge_index=edge_index,
            output_features=args.output_dim,
            ext_dim=getattr(args, "ext_dim", 21),
        ).to(args.device)
        load_state_dict_if_available(model, args)
        return model

    def _phase1(self, model: nn.Module, x: torch.Tensor, args: Any):
        x_ext = getattr(args, "_privacy_x_ext", None)
        if x_ext is None:
            x_ext = torch.zeros(
                x.shape[0],
                args.t_out,
                getattr(args, "ext_dim", 21),
                device=x.device,
                dtype=x.dtype,
            )
        else:
            x_ext = x_ext.to(device=x.device, dtype=x.dtype)

        was_training = model.training
        model.train()
        try:
            with torch.backends.cudnn.flags(enabled=False):
                return model.forward_phase1(x, x_ext=x_ext)
        finally:
            model.train(was_training)

    def _weighted(self, tensor: torch.Tensor, weight: float) -> torch.Tensor:
        return tensor * math.sqrt(max(0.0, float(weight)))

    def _server_attention_proxy(self, h_s_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # In the real server path, the target client's context is a weighted
        # combination of pooled client states. The privacy runner has a single
        # attacked client sample, so the faithful local proxy reduces to that
        # client's pooled state expanded back over local nodes.
        pooled_hs = h_s_i.mean(dim=1)
        server_attention_context = pooled_hs.unsqueeze(1).repeat(1, h_s_i.shape[1], 1)
        agg_hs = h_s_i + server_attention_context
        return server_attention_context, agg_hs

    def _fedgat_batch_context(self, h_s_i: torch.Tensor) -> torch.Tensor:
        """Return a batch-aggregated, node-preserving hidden representation.

        This restricted observation hides which batch record produced each
        activation, while retaining node identity.  Consequently B=1 exposes
        the full per-node hidden state, whereas B>1 only exposes its batch
        average and makes a particular victim record underdetermined.
        """
        return h_s_i.mean(dim=0, keepdim=True)

    def _activation_leak(
        self,
        h_s_i: torch.Tensor,
        rlcn_out: torch.Tensor,
        scn_out: torch.Tensor,
        args: Any,
    ):
        surface = getattr(args, "fedstn_activation_surface", "hs_context_agg")
        self.attack_surface = f"fedstn_{surface}"
        self.batch_aggregate_only = surface == "fedgat_batch_context"
        if surface == "hs":
            return self._weighted(h_s_i, float(getattr(args, "fedstn_hs_weight", 1.0)))
        if surface == "fedgat_batch_context":
            return self._weighted(
                self._fedgat_batch_context(h_s_i),
                float(getattr(args, "fedstn_context_weight", 1.0)),
            )
        if surface == "hs_local_upper":
            return {
                "h_s_i": self._weighted(
                    h_s_i,
                    float(getattr(args, "fedstn_hs_weight", 1.0)),
                ),
                "rlcn_out": self._weighted(
                    rlcn_out,
                    float(getattr(args, "fedstn_rlcn_weight", 0.1)),
                ),
                "scn_out": self._weighted(
                    scn_out,
                    float(getattr(args, "fedstn_scn_weight", 0.1)),
                ),
            }
        if surface != "hs_context_agg":
            raise ValueError(f"Unsupported fedstn_activation_surface: {surface}")

        server_attention_context, agg_hs = self._server_attention_proxy(h_s_i)
        return {
            "h_s_i": self._weighted(
                h_s_i,
                float(getattr(args, "fedstn_hs_weight", 1.0)),
            ),
            "server_attention_context": self._weighted(
                server_attention_context,
                float(getattr(args, "fedstn_context_weight", 0.1)),
            ),
            "agg_hs": self._weighted(
                agg_hs,
                float(getattr(args, "fedstn_agg_weight", 0.1)),
            ),
        }

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
        old_x_ext = getattr(args, "_privacy_x_ext", None)
        setattr(args, "_privacy_x_ext", getattr(batch, "x_ext", None))
        try:
            h_s_i, rlcn_out, scn_out = self._phase1(model, x, args)
        finally:
            if old_x_ext is None:
                try:
                    delattr(args, "_privacy_x_ext")
                except AttributeError:
                    pass
            else:
                setattr(args, "_privacy_x_ext", old_x_ext)

        if leak_type == "activation":
            if bool(getattr(args, "quantized_prediction_sidechannel", False)):
                return fixed_quantized_prediction(model.forward_phase2(h_s_i, rlcn_out, scn_out), args)
            return self._activation_leak(h_s_i, rlcn_out, scn_out, args)

        if leak_type != "gradient":
            raise ValueError("FedSTN currently supports activation and gradient reconstruction only.")

        pred = model.forward_phase2(h_s_i, rlcn_out, scn_out)
        return prediction_gradient(model=model, pred=pred, y=y, batch=batch, create_graph=create_graph)
