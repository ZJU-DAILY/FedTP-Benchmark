"""Minimal, role-aware HE communication traces for reconstruction experiments.

The external trace deliberately contains only ciphertext-visible metadata.  A
decrypted client payload is never written unless the caller explicitly enables
the separately labelled insider-upper-bound option.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from privacy.baseline_adapters.common import prediction_sidechannel_tensor


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "message"


def _tree_schema(value: Any) -> Any:
    if torch.is_tensor(value):
        return {
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "numel": int(value.numel()),
            "requires_grad": bool(value.requires_grad),
        }
    if isinstance(value, Mapping):
        return {str(key): _tree_schema(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_tree_schema(item) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_tree_schema(item) for item in value]}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {"kind": type(value).__name__, "value": value}
    return {"kind": type(value).__name__}


def _cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    return value


def capture_revised_quantized_prediction(
    ctx: Any,
    args: Any,
    tag: str,
    *,
    prediction: torch.Tensor,
    model_state_dict: Mapping[str, Any],
) -> None:
    """Capture the explicit, non-ciphertext prediction side channel once.

    This is intentionally separate from HE payload traces: it describes a
    revised protocol in which the client sends a fixed-point prediction for
    monitoring, while the native client upload remains HE protected.
    """
    if not (
        bool(getattr(args, "privacy_trace_quantized_prediction", False))
        or bool(os.environ.get("PRIVACY_TRACE_QUANTIZED_PREDICTION", ""))
    ):
        return
    # Centralized simulators (FedmSSA) have no FATE context.  Their first
    # package is the designated target client for a trace-only experiment.
    rank = int(getattr(ctx, "rank", 0))
    if rank != int(getattr(args, "privacy_trace_client", 0)):
        return
    if int(getattr(args, "_privacy_quant_prediction_records", 0)) >= 1:
        return
    root = _trace_root(args)
    if root is None:
        return
    bits = max(1, int(getattr(args, "quant_prediction_bits", 4)))
    clip = max(1e-6, float(getattr(args, "quant_prediction_clip", 3.0)))
    levels = float((1 << bits) - 1)
    summary_mode = str(getattr(args, "quant_prediction_summary", "full")).lower()
    sidechannel_value = prediction_sidechannel_tensor(prediction.detach(), args)
    value = sidechannel_value.clamp(-clip, clip)
    quantized = torch.round((value + clip) * levels / (2.0 * clip)) * (2.0 * clip / levels) - clip
    root.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(args, backend=str(getattr(args, "he_backend", "he_ttp")), observer="revised_protocol_public_model_quantized_prediction", rank=rank, tag=tag, value=quantized)
    metadata.update({
        "threat_model": "Revised protocol: public model plus client-returned fixed-point prediction side channel",
        "attack": "activation",
        "observed_signal": f"4-bit_quantized_prediction_{summary_mode}",
        "original_he_ciphertext_only": False,
        "quant_prediction_bits": bits,
        "quant_prediction_clip": clip,
        "quant_prediction_summary": summary_mode,
    })
    path = root / f"{_safe_name(getattr(args, 'model', 'model'))}_{_safe_name(getattr(args, 'dataset_name', 'dataset'))}_client{rank}_revised_quantized_prediction_{_safe_name(tag)}.pt"
    torch.save({"metadata": metadata, "observed_leak": quantized.cpu(), "model_state_dict": _cpu_tree(model_state_dict)}, path)
    print(f"[PrivacyTrace] revised quantized-prediction trace saved: {path}", flush=True)
    args._privacy_quant_prediction_records = 1


def _payload_type(tag: str) -> str:
    normalized = str(tag).lower()
    if "delta" in normalized or "weight" in normalized or "payload" in normalized:
        return "model_update"
    if "grad" in normalized or "g_" in normalized:
        return "gradient"
    if any(item in normalized for item in ("activation", "agg", "h_", "hs")):
        return "activation"
    if any(item in normalized for item in ("prototype", "pattern", "basis")):
        return "prototype"
    return "hidden_state"


def _phase(tag: str) -> str:
    normalized = str(tag).lower()
    if normalized.startswith("init") or "public_init" in normalized:
        return "initialization"
    if "test" in normalized or normalized.startswith("te_"):
        return "test"
    if "val" in normalized or normalized.startswith("va_"):
        return "validation"
    return "train"


def _trace_root(args: Any) -> Path | None:
    raw = str(getattr(args, "privacy_trace_dir", "") or "").strip()
    return Path(raw).expanduser() if raw else None


def _should_capture(args: Any, *, rank: int, tag: str, require_target_rank: bool) -> bool:
    if _trace_root(args) is None:
        return False
    if require_target_rank and rank != int(getattr(args, "privacy_trace_client", 0)):
        return False
    wanted = str(getattr(args, "privacy_trace_tag", "") or "")
    if wanted and wanted not in str(tag):
        return False
    # A common model initialisation is public protocol state, not an observed
    # client training upload.  Unless a caller explicitly filters for it,
    # never let it consume the single-record reconstruction trace budget.
    normalized_tag = str(tag).lower()
    if not wanted and any(item in normalized_tag for item in ("public_init", "initial_state", "ckks_context")):
        return False
    seen = int(getattr(args, "_privacy_trace_records", 0))
    return seen < max(1, int(getattr(args, "privacy_trace_max_records", 1)))


def _metadata(args: Any, *, backend: str, observer: str, rank: int, tag: str, value: Any) -> dict[str, Any]:
    model_label = "Fed4TP" if str(getattr(args, "trainer_mode", "")).lower() == "fed4tp" else str(getattr(args, "model", ""))
    trainer_mode = str(getattr(args, "trainer_mode", "")).lower()
    if trainer_mode == "ufcl":
        model_label = "UFCL"
    elif trainer_mode == "sfl":
        model_label = "SFL"
    return {
        "schema_version": 1,
        "created_unix": time.time(),
        "model": model_label,
        "dataset_name": str(getattr(args, "dataset_name", "")),
        "feature_type": str(getattr(args, "feature_type", "")),
        "t_in": int(getattr(args, "t_in", 12)),
        "t_out": int(getattr(args, "t_out", 3)),
        "num_clients": int(getattr(args, "num_clients", 4)),
        "seed": int(getattr(args, "seed", 42)),
        "he_backend": backend,
        "observer": observer,
        "client_rank": rank,
        "tag": str(tag),
        "phase": _phase(tag),
        "payload_type": _payload_type(tag),
        "payload_schema": _tree_schema(value),
    }


def capture_he_ttp_upload(ctx: Any, args: Any, tag: str, value: Any) -> None:
    """Export only external-visible metadata for one target client upload.

    When ``privacy_trace_save_insider`` is set, an additional local tensor file
    is produced.  It is explicitly marked ``insider_upper_bound`` and must
    never be used for the external-server HE result.
    """
    rank = int(getattr(ctx, "rank", -1))
    if not _should_capture(args, rank=rank, tag=tag, require_target_rank=True):
        return
    root = _trace_root(args)
    assert root is not None
    root.mkdir(parents=True, exist_ok=True)
    record_index = int(getattr(args, "_privacy_trace_records", 0))
    stem = (
        f"{_safe_name(getattr(args, 'model', 'model'))}_"
        f"{_safe_name(getattr(args, 'dataset_name', 'dataset'))}_"
        f"client{rank}_he_ttp_external_{record_index:02d}_{_safe_name(tag)}"
    )
    external = _metadata(args, backend="he_ttp", observer="external_server", rank=rank, tag=tag, value=value)
    external_path = root / f"{stem}.json"
    external_path.write_text(json.dumps(external, indent=2), encoding="utf-8")
    print(f"[PrivacyTrace] external HE-TTP metadata saved: {external_path}", flush=True)

    if bool(getattr(args, "privacy_trace_save_insider", False)):
        insider = dict(external)
        insider["observer"] = "trusted_arbiter_insider_upper_bound"
        insider_path = root / f"{stem}_insider.pt"
        torch.save({"metadata": insider, "payload": _cpu_tree(value)}, insider_path)
        print(f"[PrivacyTrace] HE-TTP insider upper-bound payload saved: {insider_path}", flush=True)
    args._privacy_trace_records = record_index + 1


def capture_he_ttp_insider_upper_bound(ctx: Any, args: Any, tag: str, *, observed_leak: Any, model_state_dict: Mapping[str, Any] | None, leak_type: str, observer: str = "trusted_arbiter_insider_upper_bound", threat_model: str = "HE-TTP decrypting Arbiter insider upper bound") -> None:
    """Save a replay-ready target payload for an explicitly labelled upper bound."""
    if not bool(getattr(args, "privacy_trace_save_insider", False)):
        return
    rank = int(getattr(ctx, "rank", -1))
    if rank != int(getattr(args, "privacy_trace_client", 0)) or int(getattr(args, "_privacy_he_ttp_insider_upper_records", 0)) >= 1:
        return
    root = _trace_root(args)
    if root is None:
        return
    root.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(args, backend="he_ttp", observer=observer, rank=rank, tag=tag, value=observed_leak)
    metadata.update({"threat_model": threat_model, "attack": leak_type, "target_client": rank, "observed_signal": "decrypted_target_client_payload"})
    path = root / f"{_safe_name(getattr(args, 'model', 'model'))}_{_safe_name(getattr(args, 'dataset_name', 'dataset'))}_client{rank}_he_ttp_insider_upper_{_safe_name(tag)}.pt"
    torch.save({"metadata": metadata, "observed_leak": _cpu_tree(observed_leak), "model_state_dict": None if model_state_dict is None else _cpu_tree(model_state_dict)}, path)
    print(f"[PrivacyTrace] HE-TTP Arbiter-insider upper-bound trace saved: {path}", flush=True)
    args._privacy_he_ttp_insider_upper_records = 1


def capture_he_sa_aggregate(ctx: Any, args: Any, tag: str, aggregate: Any) -> None:
    """Export the only plaintext quantity available to an HE-SA arbiter."""
    rank = int(getattr(ctx, "rank", -1))
    if not _should_capture(args, rank=rank, tag=tag, require_target_rank=False):
        return
    root = _trace_root(args)
    assert root is not None
    root.mkdir(parents=True, exist_ok=True)
    record_index = int(getattr(args, "_privacy_trace_records", 0))
    stem = (
        f"{_safe_name(getattr(args, 'model', 'model'))}_"
        f"{_safe_name(getattr(args, 'dataset_name', 'dataset'))}_"
        f"aggregate_he_sa_{record_index:02d}_{_safe_name(tag)}"
    )
    metadata = _metadata(args, backend="he_sa", observer="external_server_aggregate_only", rank=rank, tag=tag, value=aggregate)
    path = root / f"{stem}.pt"
    torch.save({"metadata": metadata, "aggregate": _cpu_tree(aggregate)}, path)
    print(f"[PrivacyTrace] HE-SA aggregate trace saved: {path}", flush=True)
    args._privacy_trace_records = record_index + 1


def capture_he_sa_arbiter_insider(
    ctx: Any,
    args: Any,
    tag: str,
    *,
    payload: Any,
    model_state_dict: Mapping[str, Any],
    leak_type: str = "model_update",
) -> None:
    """Save one explicitly privileged HE-SA replay trace.

    This represents a compromised decrypting Arbiter that can decrypt the
    target ciphertext *and* knows the public round-start model.  It is an
    upper-bound threat model, never an external-server HE result.
    """
    if not bool(getattr(args, "privacy_trace_save_insider", False)):
        return
    rank = int(getattr(ctx, "rank", -1))
    if rank != int(getattr(args, "privacy_trace_client", 0)):
        return
    if int(getattr(args, "_privacy_he_sa_insider_records", 0)) >= 1:
        return
    root = _trace_root(args)
    if root is None:
        return
    root.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(
        args,
        backend="he_sa",
        observer="decrypting_arbiter__client_state_assisted_upper_bound",
        rank=rank,
        tag=tag,
        value=payload,
    )
    metadata.update({
        "threat_model": "HE-SA decrypting Arbiter insider upper bound",
        "attack": str(leak_type),
        "observed_signal": "target_client_pre_aggregation_update",
    })
    stem = (
        f"{_safe_name(getattr(args, 'model', 'model'))}_"
        f"{_safe_name(getattr(args, 'dataset_name', 'dataset'))}_"
        f"client{rank}_he_sa_insider_{_safe_name(tag)}"
    )
    path = root / f"{stem}.pt"
    torch.save(
        {
            "metadata": metadata,
            "observed_leak": _cpu_tree(payload),
            "model_state_dict": _cpu_tree(model_state_dict),
        },
        path,
    )
    print(f"[PrivacyTrace] HE-SA Arbiter-insider upper-bound trace saved: {path}", flush=True)
    args._privacy_he_sa_insider_records = 1


def capture_he_sa_collusion_residual(
    ctx: Any,
    args: Any,
    tag: str,
    *,
    payload: Any,
    model_state_dict: Mapping[str, Any],
) -> None:
    """Save one HE-SA K-1 collusion replay trace.

    This trace represents a server that receives the decrypted weighted
    aggregate and colludes with every non-target client.  Those clients reveal
    their own uploads, so the server can derive the target update by
    subtraction.  The stored value is that derived target update; it must be
    reported as a collusion attack, never as an external-server-only result or
    as an Arbiter-insider upper bound.
    """
    if not bool(getattr(args, "privacy_trace_save_collusion", False)):
        return
    rank = int(getattr(ctx, "rank", -1))
    if rank != int(getattr(args, "privacy_trace_client", 0)):
        return
    if int(getattr(args, "_privacy_he_sa_collusion_records", 0)) >= 1:
        return
    root = _trace_root(args)
    if root is None:
        return
    root.mkdir(parents=True, exist_ok=True)
    total_clients = max(1, int(getattr(args, "num_clients", 1)))
    colluding = [index for index in range(total_clients) if index != rank]
    metadata = _metadata(
        args,
        backend="he_sa",
        observer="server_k_minus_1_client_collusion",
        rank=rank,
        tag=tag,
        value=payload,
    )
    metadata.update({
        "threat_model": "HE-SA server plus K-1 colluding clients residual attack",
        "attack": "model_update",
        "observed_signal": "target_update_derived_from_decrypted_aggregate_minus_colluding_client_updates",
        "target_client": rank,
        "colluding_clients": colluding,
    })
    stem = (
        f"{_safe_name(getattr(args, 'model', 'model'))}_"
        f"{_safe_name(getattr(args, 'dataset_name', 'dataset'))}_"
        f"client{rank}_he_sa_collusion_{_safe_name(tag)}"
    )
    path = root / f"{stem}.pt"
    torch.save(
        {
            "metadata": metadata,
            "observed_leak": _cpu_tree(payload),
            "model_state_dict": _cpu_tree(model_state_dict),
        },
        path,
    )
    print(f"[PrivacyTrace] HE-SA K-1 collusion residual trace saved: {path}", flush=True)
    args._privacy_he_sa_collusion_records = 1


def capture_he_sa_kminus2_hidden_term(
    ctx: Any,
    args: Any,
    tag: str,
    *,
    payload: Any,
    model_state_dict: Mapping[str, Any],
    aggregation_weight: float,
) -> None:
    """Save private instrumentation for one of two non-colluding HE-SA clients.

    The generated files are not attacker-visible traces.  The standalone
    trace builder combines the two weighted terms into exactly the residual
    that a server obtains after subtracting two colluding-client uploads from
    a decrypted four-client aggregate.
    """
    if not bool(getattr(args, "privacy_trace_save_kminus2", False)):
        return
    rank = int(getattr(ctx, "rank", -1))
    target = int(getattr(args, "privacy_trace_client", 0))
    peer = (target + 1) % max(1, int(getattr(args, "num_clients", 1)))
    if rank not in (target, peer):
        return
    marker = f"_privacy_he_sa_kminus2_client{rank}_records"
    if int(getattr(args, marker, 0)) >= 1:
        return
    root = _trace_root(args)
    if root is None:
        return
    root = root / "_protocol_simulation"
    root.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(
        args,
        backend="he_sa",
        observer="protocol_instrumentation_not_attacker_visible",
        rank=rank,
        tag=tag,
        value=payload,
    )
    metadata.update({
        "target_client": target,
        "peer_hidden_client": peer,
        "aggregation_weight": float(aggregation_weight),
        "purpose": "construct K-2 collusion residual trace",
    })
    stem = (
        f"{_safe_name(getattr(args, 'model', 'model'))}_"
        f"{_safe_name(getattr(args, 'dataset_name', 'dataset'))}_"
        f"kminus2_hidden_client{rank}_{_safe_name(tag)}"
    )
    path = root / f"{stem}.pt"
    torch.save(
        {
            "metadata": metadata,
            "payload": _cpu_tree(payload),
            "model_state_dict": _cpu_tree(model_state_dict),
        },
        path,
    )
    print(f"[PrivacyTrace] HE-SA K-2 hidden protocol term saved: {path}", flush=True)
    setattr(args, marker, 1)


def capture_he_sa_kminus3_hidden_term(
    ctx: Any,
    args: Any,
    tag: str,
    *,
    payload: Any,
    model_state_dict: Mapping[str, Any],
    aggregation_weight: float,
) -> None:
    """Save instrumentation for the three hidden terms in one-client collusion."""
    if not bool(getattr(args, "privacy_trace_save_kminus3", False)):
        return
    rank = int(getattr(ctx, "rank", -1))
    target = int(getattr(args, "privacy_trace_client", 0))
    total_clients = max(1, int(getattr(args, "num_clients", 1)))
    hidden = [target, (target + 1) % total_clients, (target + 2) % total_clients]
    if rank not in hidden:
        return
    marker = f"_privacy_he_sa_kminus3_client{rank}_records"
    if int(getattr(args, marker, 0)) >= 1:
        return
    root = _trace_root(args)
    if root is None:
        return
    root = root / "_protocol_simulation"
    root.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(
        args,
        backend="he_sa",
        observer="protocol_instrumentation_not_attacker_visible",
        rank=rank,
        tag=tag,
        value=payload,
    )
    metadata.update({
        "target_client": target,
        "hidden_clients": hidden,
        "aggregation_weight": float(aggregation_weight),
        "purpose": "construct one-client-collusion residual trace",
    })
    stem = (
        f"{_safe_name(getattr(args, 'model', 'model'))}_"
        f"{_safe_name(getattr(args, 'dataset_name', 'dataset'))}_"
        f"kminus3_hidden_client{rank}_{_safe_name(tag)}"
    )
    path = root / f"{stem}.pt"
    torch.save(
        {
            "metadata": metadata,
            "payload": _cpu_tree(payload),
            "model_state_dict": _cpu_tree(model_state_dict),
        },
        path,
    )
    print(f"[PrivacyTrace] HE-SA K-3 hidden protocol term saved: {path}", flush=True)
    setattr(args, marker, 1)


def capture_he_sa_server_aggregate_hidden_term(
    ctx: Any,
    args: Any,
    tag: str,
    *,
    payload: Any,
    model_state_dict: Mapping[str, Any],
    aggregation_weight: float,
    leak_type: str = "model_update",
) -> None:
    """Save four local terms used offline to construct a server-only aggregate.

    These files are protocol instrumentation, not attacker-visible artifacts.
    The generated replay trace contains only their weighted sum, which is the
    decrypted aggregate visible to the server after HE-SA aggregation.
    """
    if not bool(getattr(args, "privacy_trace_save_server_aggregate", False)):
        return
    rank = int(getattr(ctx, "rank", -1))
    total_clients = max(1, int(getattr(args, "num_clients", 1)))
    if rank < 0 or rank >= total_clients:
        return
    marker = f"_privacy_he_sa_server_aggregate_client{rank}_records"
    if int(getattr(args, marker, 0)) >= 1:
        return
    root = _trace_root(args)
    if root is None:
        return
    root = root / "_protocol_simulation"
    root.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(
        args, backend="he_sa", observer="protocol_instrumentation_not_attacker_visible",
        rank=rank, tag=tag, value=payload,
    )
    metadata.update({
        "target_client": int(getattr(args, "privacy_trace_client", 0)),
        "hidden_clients": list(range(total_clients)),
        "aggregation_weight": float(aggregation_weight),
        "aggregate_leak_type": str(leak_type),
        "purpose": "construct server-only decrypted aggregate trace",
    })
    stem = (
        f"{_safe_name(getattr(args, 'model', 'model'))}_"
        f"{_safe_name(getattr(args, 'dataset_name', 'dataset'))}_"
        f"server_aggregate_hidden_client{rank}_{_safe_name(tag)}"
    )
    path = root / f"{stem}.pt"
    torch.save({"metadata": metadata, "payload": _cpu_tree(payload), "model_state_dict": _cpu_tree(model_state_dict)}, path)
    print(f"[PrivacyTrace] HE-SA server-aggregate protocol term saved: {path}", flush=True)
    setattr(args, marker, 1)


def capture_dp_reconstruction_trace(
    ctx: Any,
    args: Any,
    tag: str,
    *,
    model_state: Mapping[str, Any],
    observed_leak: Any,
    dp_info: Mapping[str, Any] | None = None,
) -> None:
    """Save one explicit, labelled DP replay trace for local reconstruction.

    The stored payload is exactly the noisy object received by the arbiter.
    The local model state is intentionally included to create a
    *client-state-assisted replay upper bound*; it must not be described as an
    external server-only attack when FUELS keeps client models private.
    """
    raw = str(getattr(args, "dp_reconstruction_trace_dir", "") or "").strip()
    if not raw or int(getattr(ctx, "rank", -1)) != int(getattr(args, "dp_reconstruction_trace_client", 0)):
        return
    wanted = str(getattr(args, "dp_reconstruction_trace_tag", "") or "")
    if wanted and wanted not in str(tag):
        return
    count = int(getattr(args, "_dp_reconstruction_trace_records", 0))
    if count >= 1:
        return
    root = Path(raw).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    rank = int(getattr(ctx, "rank", -1))
    stem = (
        f"{_safe_name(getattr(args, 'model', 'model'))}_"
        f"{_safe_name(getattr(args, 'dataset_name', 'dataset'))}_"
        f"client{rank}_dp_replay_{count:02d}_{_safe_name(tag)}"
    )
    payload = {
        "metadata": {
            "schema_version": 1,
            "observer": "arbiter_noisy_payload__client_state_assisted_replay_upper_bound",
            "model": str(getattr(args, "model", "")),
            "dataset_name": str(getattr(args, "dataset_name", "")),
            "feature_type": str(getattr(args, "feature_type", "")),
            "client_rank": rank,
            "tag": str(tag),
            "dp_sigma": float(getattr(args, "dp_sigma", 0.0)),
            "dp_info": dict(dp_info or {}),
        },
        "model_state_dict": _cpu_tree(model_state),
        "observed_leak": _cpu_tree(observed_leak),
    }
    path = root / f"{stem}.pt"
    torch.save(payload, path)
    print(f"[DPReplayTrace] saved client-state-assisted noisy-payload trace: {path}", flush=True)
    args._dp_reconstruction_trace_records = count + 1
