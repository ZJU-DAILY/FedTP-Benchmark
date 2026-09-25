from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from model.FedGTP import FedGTP_Model
from privacy.baseline_adapters.common import (
    align_prediction_and_target,
    args_from_batch,
    gradient_tuple,
    fixed_quantized_prediction,
    load_state_dict_if_available,
    loss_fn,
)


class FedGTPPrivacyAdapter:
    name = "FedGTP"
    default_attack = "activation"
    requires_dummy_y = True
    # Matches `fedgtp_dp_clips["eh"]` in the federated trainer.
    dp_payload_group = "eh"

    def build_model(self, args: Any, batch: Any) -> nn.Module:
        num_nodes = int(batch.real_x.shape[1])
        max_nodes = max(len(nodes) for nodes in getattr(batch, "nodes_per", [[0] * num_nodes]))
        model = FedGTP_Model(
            num_nodes=num_nodes,
            max_nodes=max_nodes,
            in_dim=args.t_in,
            out_dim=args.t_out,
            feature_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            emb_dim=getattr(args, "node_emb_dim", 4),
            poly_k=getattr(args, "poly_k", 2),
        ).to(args.device)
        load_state_dict_if_available(model, args)
        return model

    def _single_client_comm_hook(self, tensors, _tag):
        return tensors

    @staticmethod
    def _uploaded_eh_view(eh: torch.Tensor) -> torch.Tensor:
        """Reproduce FedGTP's observable EH serialization numerically.

        The federated trainer uploads ``EH.detach().to(float16).cpu()``.  A
        reconstruction attacker must therefore observe the FP16-rounded
        values, not the internal FP32 activations.  During dummy optimization
        we use a straight-through view: its forward value equals the uploaded
        FP16 payload after it is read back on the model device, while its
        backward derivative is that of the original EH tensor.
        """
        serialized = eh.detach().to(dtype=torch.float16, device="cpu").contiguous()
        received = serialized.to(device=eh.device, dtype=eh.dtype)
        if eh.requires_grad:
            return eh + (received - eh).detach()
        return received

    def _capture_eh_leak(self, model: nn.Module, x: torch.Tensor):
        old_capture = getattr(model, "privacy_capture_eh", False)
        old_records = getattr(model, "privacy_eh_records", None)
        model.privacy_capture_eh = True
        model.privacy_eh_records = []
        try:
            model(x, comm_hook=self._single_client_comm_hook, batch_tag="privacy")
            records = tuple(
                tuple(self._uploaded_eh_view(item) for item in record)
                for record in model.privacy_eh_records
            )
        finally:
            model.privacy_capture_eh = old_capture
            model.privacy_eh_records = old_records

        if not records:
            raise RuntimeError("FedGTP did not expose any EH communication tensors for privacy attack.")
        return records

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
                prediction = model(x, comm_hook=self._single_client_comm_hook, batch_tag="privacy_prediction")
                return fixed_quantized_prediction(prediction, args_from_batch(batch))
            return self._capture_eh_leak(model, x)

        if leak_type != "gradient":
            raise ValueError("FedGTP supports activation(EH communication) and gradient diagnostics.")

        args = args_from_batch(batch)
        pred = model(x, comm_hook=self._single_client_comm_hook, batch_tag="privacy")
        pred, target = align_prediction_and_target(pred, y)
        loss = loss_fn(args)(pred, target)
        return gradient_tuple(model=model, loss=loss, create_graph=create_graph)
