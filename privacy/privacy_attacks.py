from __future__ import annotations

import contextlib
from dataclasses import dataclass
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch
import torch.nn.functional as F

from privacy.privacy_metrics import reconstruction_metrics
from privacy.protection import gaussian_dp_upload, l2_norm


TensorTree = Any


def _tree_to_device(value: TensorTree, device: torch.device | str) -> TensorTree:
    if torch.is_tensor(value):
        return value.detach().to(device)
    if isinstance(value, Mapping):
        return {key: _tree_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_tree_to_device(item, device) for item in value]
    return value


def _load_he_replay_trace(
    path_text: str,
    model: torch.nn.Module,
    expected: TensorTree,
    *,
    expected_observer: str,
    argument_name: str,
) -> tuple[TensorTree, Dict[str, Any]]:
    path = Path(path_text).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"HE insider trace is missing: {path}")
    trace = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(trace, Mapping) or trace.get("observed_leak") is None:
        raise ValueError(f"HE replay trace {path} has no observed_leak payload.")
    metadata = dict(trace.get("metadata", {}))
    observer_label = str(metadata.get("observer", "")).lower()
    if observer_label != expected_observer:
        raise ValueError(
            f"{argument_name} requires observer={expected_observer!r}; "
            f"trace records {observer_label!r}."
        )
    observed = _tree_to_device(trace["observed_leak"], next(model.parameters()).device)
    # Model-update adapters expose a tuple ordered by trainable parameters,
    # whereas federated transport traces use a state-dict mapping.  Convert
    # only when the names and tensor shapes match exactly.
    if isinstance(expected, tuple) and isinstance(observed, Mapping):
        converted = []
        expected_names = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            expected_names.append(name)
            if name not in observed:
                continue
            value = observed[name]
            if not torch.is_tensor(value) or tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(f"HE insider payload tensor mismatch for parameter {name!r}.")
            converted.append(value)
        if not converted:
            observed_names = [str(name) for name in observed.keys()]
            raise ValueError(
                "HE replay trace and reconstruction model have no matching trainable parameter names. "
                f"Trace examples={observed_names[:3]}; model examples={expected_names[:3]}. "
                "Rebuild the adapter with the architecture used for HE trace capture."
            )
        if len(converted) != len(expected_names):
            missing = [name for name in expected_names if name not in observed]
            extra = [str(name) for name in observed.keys() if name not in set(expected_names)]
            raise ValueError(
                "HE replay trace and reconstruction model expose different trainable parameter sets: "
                f"matched={len(converted)}, model={len(expected_names)}, trace={len(observed)}; "
                f"missing_model_keys={missing[:3]}, extra_trace_keys={extra[:3]}."
            )
        observed = tuple(converted)
    if type(observed) is not type(expected):
        raise TypeError(
            f"HE insider payload type {type(observed).__name__} does not match "
            f"adapter leak type {type(expected).__name__}."
        )
    return observed, metadata


def _load_he_insider_trace(path_text: str, model: torch.nn.Module, expected: TensorTree) -> tuple[TensorTree, Dict[str, Any]]:
    path = Path(path_text).expanduser()
    trace = torch.load(path, map_location="cpu", weights_only=False)
    observer = str(dict(trace.get("metadata", {})).get("observer", "")) if isinstance(trace, Mapping) else ""
    if observer not in (
        "decrypting_arbiter__client_state_assisted_upper_bound",
        "trusted_arbiter_insider_upper_bound",
        "malicious_client_state_oracle_upper_bound",
        "revised_protocol_public_model_quantized_prediction",
    ):
        raise ValueError(f"--he_insider_trace has unsupported observer={observer!r}.")
    return _load_he_replay_trace(path_text, model, expected, expected_observer=observer, argument_name="--he_insider_trace")


def _load_he_collusion_trace(path_text: str, model: torch.nn.Module, expected: TensorTree) -> tuple[TensorTree, Dict[str, Any]]:
    return _load_he_replay_trace(
        path_text, model, expected,
        expected_observer="server_k_minus_1_client_collusion",
        argument_name="--he_collusion_trace",
    )


def _load_he_kminus2_trace(path_text: str, model: torch.nn.Module, expected: TensorTree) -> tuple[TensorTree, Dict[str, Any]]:
    observed, metadata = _load_he_replay_trace(
        path_text, model, expected,
        expected_observer="server_k_minus_2_client_collusion",
        argument_name="--he_kminus2_trace",
    )
    if int(metadata.get("target_client", -1)) == int(metadata.get("peer_hidden_client", -1)):
        raise ValueError("K-2 trace must contain two distinct hidden clients.")
    if float(metadata.get("target_weight", 0.0)) <= 0 or float(metadata.get("peer_weight", 0.0)) <= 0:
        raise ValueError("K-2 trace has invalid aggregation weights.")
    return observed, metadata


def _load_he_kminus3_trace(path_text: str, model: torch.nn.Module, expected: TensorTree) -> tuple[TensorTree, Dict[str, Any]]:
    observed, metadata = _load_he_replay_trace(
        path_text, model, expected,
        expected_observer="server_plus_one_client_collusion",
        argument_name="--he_kminus3_trace",
    )
    hidden = [int(value) for value in metadata.get("hidden_clients", [])]
    weights = [float(value) for value in metadata.get("hidden_weights", [])]
    if len(hidden) != 3 or len(set(hidden)) != 3 or len(weights) != 3 or any(value <= 0 for value in weights):
        raise ValueError("K-3 trace must contain three distinct hidden clients with positive weights.")
    return observed, metadata


def _lock_fedmssa_attack_rank(
    adapter: Any,
    observed_leak: TensorTree,
    args: Any,
) -> None:
    """Fix FedmSSA's data-derived phase-1 rank from the observed upload.

    In the actual protocol the rank is visible from the uploaded basis shape.
    If the attack recomputes an automatic SVD rank from every dummy iterate,
    the dummy and observed tensor trees can have different dimensions and are
    not comparable.  Locking it after observing the real initial basis is both
    protocol-faithful and necessary for stable reconstruction optimization.
    """
    if getattr(adapter, "name", "") != "FedmSSA":
        return
    if int(getattr(args, "fedmssa_rank", 0) or 0) > 0:
        return
    if not isinstance(observed_leak, Mapping):
        return
    basis = observed_leak.get("phase1_init_basis")
    if not torch.is_tensor(basis) or basis.dim() < 2:
        return
    rank = int(basis.shape[-1])
    if rank <= 0:
        return
    args.fedmssa_rank = rank
    print(
        f"[Privacy][FedmSSA] locked phase-1 rank={rank} from observed initial-basis shape.",
        flush=True,
    )


def _observed_dp_clip_norms(args: Any) -> Dict[str, float]:
    """Load the clip radii actually selected by the DP prediction run.

    A number of split/interactive baselines calibrate a distinct C for each
    semantic upload.  Reusing those logged values is essential: a single
    arbitrary attack-side C would describe a different DP mechanism.
    """
    cached = getattr(args, "_privacy_dp_clip_norms", None)
    if cached is not None:
        return dict(cached)

    clip_norms: Dict[str, float] = {}
    raw = str(getattr(args, "dp_clip_norms", "") or "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("--dp_clip_norms must be a JSON object mapping payload type to C.") from exc
        if not isinstance(parsed, Mapping):
            raise ValueError("--dp_clip_norms must be a JSON object mapping payload type to C.")
        for name, value in parsed.items():
            radius = float(value)
            if radius <= 0:
                raise ValueError(f"DP clip norm for {name!r} must be positive, got {radius}.")
            clip_norms[str(name)] = radius

    calibration_log = str(getattr(args, "dp_calibration_log", "") or "").strip()
    if calibration_log:
        path = Path(calibration_log).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"DP calibration log is missing: {path}")
        # Examples emitted by all explicit DP trainers:
        # [DPCalibration] FedOSTC rank=0 type=h_time tag=... clip_norm=1.234
        pattern = re.compile(
            r"\[DPCalibration\].*?\btype=(?P<kind>[A-Za-z0-9_.-]+).*?"
            r"\bclip_norm=(?P<radius>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        )
        collected: Dict[str, list[float]] = {}
        for match in pattern.finditer(path.read_text(encoding="utf-8", errors="replace")):
            collected.setdefault(match.group("kind"), []).append(float(match.group("radius")))
        # FedAvg-style trainers have one complete upload and historically log
        # only ``clip_norm=...`` (without a semantic ``type=...`` field).
        # It is still an unambiguous replay target: expose that one calibrated
        # radius as ``default``.  Do *not* use this fallback when typed records
        # exist, because split/interactive methods can have one C per payload.
        if not collected:
            generic_pattern = re.compile(
                r"\[DPCalibration\].*?\bclip_norm="
                r"(?P<radius>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
            )
            generic_radii = [float(match.group("radius")) for match in generic_pattern.finditer(
                path.read_text(encoding="utf-8", errors="replace")
            )]
            if generic_radii:
                collected["default"] = generic_radii
            else:
                raise ValueError(
                    f"No [DPCalibration] clip_norm=... record was found in {path}. "
                    "Run the matching one-round DP calibration first."
                )
        for name, radii in collected.items():
            # Every rank should receive the same arbiter C.  Median protects
            # against duplicated rank lines and harmless print truncation.
            clip_norms.setdefault(name, float(sorted(radii)[len(radii) // 2]))

    fallback = float(getattr(args, "dp_clip_norm", 0.0))
    if fallback > 0:
        clip_norms.setdefault("default", fallback)
    args._privacy_dp_clip_norms = dict(clip_norms)
    return clip_norms


def _clip_for_dp_group(group: str, clip_norms: Mapping[str, float]) -> float:
    if group in clip_norms:
        return float(clip_norms[group])
    if "default" in clip_norms:
        return float(clip_norms["default"])
    # A single semantic C is unambiguous even if its trainer used a
    # model-specific name such as `delta` or `params`.
    non_default = [float(value) for key, value in clip_norms.items() if key != "default"]
    if len(non_default) == 1:
        return non_default[0]
    raise ValueError(
        f"DP reconstruction needs a clip C for payload group {group!r}; "
        f"available groups are {sorted(clip_norms)}. Pass --dp_clip_norms or the matching --dp_calibration_log."
    )


def _resolved_attack_clip(
    group: str,
    value: TensorTree,
    clip_norms: Mapping[str, float],
    args: Any,
) -> tuple[float, str]:
    """Resolve a replay C, or use the explicit batch-size-one fallback."""
    try:
        return _clip_for_dp_group(group, clip_norms), "prediction_calibration"
    except ValueError:
        if not bool(getattr(args, "dp_attack_auto_clip", False)):
            raise
        radius = float(l2_norm(value).item())
        if radius <= 0:
            raise ValueError(f"Cannot auto-calibrate DP attack C for zero-norm group {group!r}.")
        return radius, "attack_local_l2"


def protect_observed_leak(
    real_leak: TensorTree,
    args: Any,
    adapter: Any | None = None,
) -> tuple[TensorTree, Dict[str, Any]]:
    """Return the one fixed server-observed upload used throughout an attack.

    In the DP condition, this replays clipping and Gaussian noise once on the
    *observed* payload.  The dummy reconstruction is never noised; it is
    optimized against this one fixed noisy observation, which is the correct
    attacker view.
    """
    protection = str(getattr(args, "protection", "plain")).lower()
    if protection == "plain":
        return real_leak, {"protection": "plain"}
    if protection == "he":
        raise RuntimeError(
            "HE reconstruction requires an aggregate-decrypted upload trace from a real multi-client "
            "training round. A single-client privacy batch is not a valid HE observation; pass an "
            "exported aggregate trace through the HE attack runner instead."
        )
    if protection != "dp":
        raise ValueError(f"Unsupported protection condition: {protection}")
    trace_path = str(getattr(args, "dp_observed_trace", "") or "").strip()
    if trace_path:
        path = Path(trace_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"DP observed trace is missing: {path}")
        try:
            trace = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch < 2.0
            trace = torch.load(path, map_location="cpu")
        observed = trace.get("observed_leak") if isinstance(trace, Mapping) else None
        if observed is None:
            raise ValueError(f"DP trace {path} has no observed_leak payload.")
        if not isinstance(observed, type(real_leak)):
            raise TypeError(
                f"DP trace payload type {type(observed).__name__} does not match "
                f"adapter leak type {type(real_leak).__name__}."
            )
        print(f"[Privacy][DP] replaying noisy observed payload from trace={path}", flush=True)
        return observed, {
            "protection": "dp",
            "dp_observation_source": "explicit_training_trace",
            "dp_observed_trace": str(path),
            "dp_trace_metadata": dict(trace.get("metadata", {})) if isinstance(trace, Mapping) else {},
        }
    clip_norms = _observed_dp_clip_norms(args)
    if not clip_norms and not bool(getattr(args, "dp_attack_auto_clip", False)):
        raise ValueError(
            "DP reconstruction requires the C used by the matching DP prediction run. "
            "Pass --dp_calibration_log or --dp_clip_norm/--dp_clip_norms."
        )
    noise_seed = int(getattr(args, "dp_noise_seed", -1))
    if noise_seed < 0:
        noise_seed = int(getattr(args, "seed", 0)) + 104729
    # Most adapters expose a fixed list of independently protected uploads.
    # A few protocols (notably FedSTN) expose different leak shapes for
    # different attack surfaces, so let them select groups from the concrete
    # observed object instead of forcing every surface to be a mapping.
    group_provider = getattr(adapter, "dp_payload_group_keys_for_leak", None)
    if callable(group_provider):
        payload_groups = tuple(group_provider(real_leak, args) or ())
    else:
        payload_groups = tuple(getattr(adapter, "dp_payload_group_keys", ()) or ())
    if payload_groups:
        if not isinstance(real_leak, Mapping):
            raise TypeError(
                f"Adapter {getattr(adapter, 'name', '<unknown>')} declares DP groups "
                "but did not return a mapping leak."
            )
        protected = dict(real_leak)
        metadata: Dict[str, Any] = {"dp_groups": {}, "dp_clip_sources": {}}
        for index, group in enumerate(payload_groups):
            if group not in real_leak:
                raise KeyError(f"DP leak group {group!r} is absent from the observed leak.")
            generator = torch.Generator(device="cpu")
            generator.manual_seed(noise_seed + 1_000_003 * index)
            radius, clip_source = _resolved_attack_clip(group, real_leak[group], clip_norms, args)
            protected_group, group_metadata = gaussian_dp_upload(
                real_leak[group],
                clip_norm=radius,
                sigma=float(getattr(args, "dp_sigma", 0.5)),
                experimental_noise_clip_ratio=float(getattr(args, "experimental_noise_clip_ratio", 0.0)),
                generator=generator,
            )
            protected[group] = protected_group
            metadata["dp_groups"][group] = group_metadata
            metadata["dp_clip_sources"][group] = clip_source
    else:
        group = str(getattr(adapter, "dp_payload_group", "upload"))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(noise_seed)
        radius, clip_source = _resolved_attack_clip(group, real_leak, clip_norms, args)
        protected, group_metadata = gaussian_dp_upload(
            real_leak,
            clip_norm=radius,
            sigma=float(getattr(args, "dp_sigma", 0.5)),
            experimental_noise_clip_ratio=float(getattr(args, "experimental_noise_clip_ratio", 0.0)),
            generator=generator,
        )
        metadata = {
            "dp_groups": {group: group_metadata},
            "dp_clip_sources": {group: clip_source},
        }
    # Some protocol observations contain server-side values derived from a
    # protected client upload.  They must be recomputed from the *same noisy
    # upload*, rather than independently noised as if they were client
    # messages.  The adapter hook keeps the attack view protocol-faithful.
    postprocess = getattr(adapter, "postprocess_dp_observed_leak", None)
    if callable(postprocess):
        protected = postprocess(protected, args)

    metadata.update(
        {
            "protection": "dp",
            "dp_noise_seed": noise_seed,
            "dp_sigma": float(getattr(args, "dp_sigma", 0.5)),
            "dp_clip_norms_used": {
                group: metadata["dp_groups"][group]["dp_clip_norm"]
                for group in (payload_groups or (str(getattr(adapter, "dp_payload_group", "upload")),))
            },
            "dp_calibration_log": str(getattr(args, "dp_calibration_log", "") or ""),
        }
    )
    # Make the actual replayed DP scale visible in the reconstruction log.
    # This is especially important for the explicitly labelled batch-size-one
    # diagnostic protocol, whose C is intentionally not borrowed from a
    # batch-size-64 federated prediction run.
    for group, group_metadata in metadata["dp_groups"].items():
        print(
            f"[Privacy][DP] group={group} source={metadata['dp_clip_sources'][group]} "
            f"upload_l2={group_metadata['upload_l2_norm']:.8f} "
            f"C={group_metadata['dp_clip_norm']:.8f} "
            f"sigma={group_metadata['dp_sigma']:.8f} "
            f"noise_std={group_metadata['noise_std']:.8f} "
            f"noise_l2={group_metadata['noise_l2_norm']:.8f} "
            f"noise_clip_coeff={group_metadata['experimental_noise_clip_coefficient']:.8f}",
            flush=True,
        )
    return protected, metadata


@dataclass
class AttackResult:
    reconstructed_x: torch.Tensor
    reconstructed_y: torch.Tensor | None
    final_loss: float
    history: List[float]
    attack_name: str
    metadata: Dict[str, Any]


def detach_tree(value: TensorTree) -> TensorTree:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {k: detach_tree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(detach_tree(v) for v in value)
    if isinstance(value, list):
        return [detach_tree(v) for v in value]
    return value


def flatten_tensors(value: TensorTree) -> List[torch.Tensor]:
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, Mapping):
        tensors: List[torch.Tensor] = []
        for key in sorted(value.keys()):
            tensors.extend(flatten_tensors(value[key]))
        return tensors
    if isinstance(value, (tuple, list)):
        tensors = []
        for item in value:
            tensors.extend(flatten_tensors(item))
        return tensors
    return []


def leak_distance(pred_leak: TensorTree, real_leak: TensorTree, mode: str = "mse") -> torch.Tensor:
    pred_tensors = flatten_tensors(pred_leak)
    real_tensors = flatten_tensors(real_leak)
    if len(pred_tensors) != len(real_tensors):
        raise ValueError(
            f"Leak structures do not match: {len(pred_tensors)} tensors vs {len(real_tensors)} tensors"
        )

    # Update-level DP clips and perturbs the complete upload as one vector.
    # Matching each parameter tensor after independent normalization overweights
    # tiny tensors whose directions are noise-dominated.  The global variants
    # preserve the same full-upload geometry as the DP mechanism.
    if mode in ("global_mse", "global_normalized", "global_mixed"):
        pred_flat = torch.cat([tensor.reshape(-1) for tensor in pred_tensors])
        real_flat = torch.cat([
            tensor.to(pred.device, dtype=pred.dtype).reshape(-1)
            for pred, tensor in zip(pred_tensors, real_tensors)
        ])
        if mode == "global_mse":
            # Gaussian DP adds i.i.d. noise with the same variance to every
            # coordinate of the globally clipped upload.  Up to a constant,
            # this is its negative log-likelihood and is therefore the right
            # attacker-visible restart-selection objective.
            return F.mse_loss(pred_flat, real_flat)
        pred_norm = pred_flat.norm(p=2)
        real_norm = real_flat.norm(p=2)
        if pred_norm <= 1e-8 or real_norm <= 1e-8:
            objective = F.mse_loss(pred_flat, real_flat)
        else:
            objective = F.mse_loss(pred_flat / pred_norm, real_flat / real_norm)
        if mode == "global_mixed":
            objective = objective + 0.1 * F.mse_loss(pred_flat, real_flat)
        return objective

    loss = None
    for pred, real in zip(pred_tensors, real_tensors):
        real = real.to(pred.device, dtype=pred.dtype)
        if mode in ("normalized", "mixed"):
            pred_flat = pred.reshape(-1)
            real_flat = real.reshape(-1)
            pred_norm = pred_flat.norm(p=2)
            real_norm = real_flat.norm(p=2)
            # Normalizing an all-zero (or near-zero) update creates an unstable
            # second derivative. In that case its magnitude is meaningful too,
            # so fall back to ordinary MSE for this tensor.
            if pred_norm <= 1e-8 or real_norm <= 1e-8:
                curr = F.mse_loss(pred, real)
            else:
                curr = F.mse_loss(pred_flat / pred_norm, real_flat / real_norm)
            if mode == "mixed":
                curr = curr + 0.1 * F.mse_loss(pred, real)
        elif mode == "mse":
            curr = F.mse_loss(pred, real)
        else:
            raise ValueError(f"Unsupported leak distance mode: {mode}")
        loss = curr if loss is None else loss + curr

    if loss is None:
        raise ValueError("Leak object contains no tensors.")
    return loss


def _init_like(reference: torch.Tensor, init: str) -> torch.Tensor:
    if init == "zeros":
        return torch.zeros_like(reference)
    if init == "randn":
        return torch.randn_like(reference)
    raise ValueError(f"Unsupported init: {init}")


def _build_attack_optimizer(params: Sequence[torch.Tensor], args: Any) -> torch.optim.Optimizer:
    optimizer_name = getattr(args, "attack_optimizer", "adam").lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=args.attack_lr)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=args.attack_lr)
    raise ValueError(f"Unsupported attack optimizer: {optimizer_name}")


def _stabilize_ufcl_dummy_gradients(params: Sequence[torch.Tensor], args: Any) -> Dict[str, Any] | None:
    """Keep UFCL's nested second-order reconstruction step numerically finite."""
    if str(getattr(args, "model", "")).upper() != "UFCL":
        return None

    finite_limit = 1e4
    total_elements = 0
    nonfinite_elements = 0
    for param in params:
        if param.grad is not None:
            total_elements += param.grad.numel()
            nonfinite_elements += int((~torch.isfinite(param.grad)).sum().item())
            param.grad.nan_to_num_(nan=0.0, posinf=finite_limit, neginf=-finite_limit)
    torch.nn.utils.clip_grad_norm_(params, max_norm=100.0)
    return {
        "total_elements": total_elements,
        "nonfinite_elements": nonfinite_elements,
        "all_nonfinite": total_elements > 0 and nonfinite_elements == total_elements,
    }


def _as_like_tensor(value: Any, reference: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=reference.device, dtype=reference.dtype)
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def _training_x_bounds(batch: Any, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    cached = getattr(batch, "_privacy_training_x_bounds", None)
    if cached is not None:
        lower, upper = cached
        return (
            lower.to(device=reference.device, dtype=reference.dtype),
            upper.to(device=reference.device, dtype=reference.dtype),
        )

    dataset = getattr(batch, "dataset", None)
    tensors = getattr(dataset, "tensors", None)
    if tensors and torch.is_tensor(tensors[0]):
        train_x = tensors[0].detach()
    elif dataset is not None and hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        # Grid datasets (TaxiBJ/TaxiNYC/BikeNYC) expose samples through
        # __getitem__ rather than TensorDataset.tensors.  Previously this
        # returned None, silently disabling the advertised range prior for
        # exactly the data used by the TDLR K-1 experiment.
        max_samples = max(1, int(getattr(getattr(batch, "args", None), "traffic_prior_samples", 256)))
        n_samples = min(len(dataset), max_samples)
        samples = [_sample_x_from_dataset(dataset, idx) for idx in range(n_samples)]
        samples = [sample for sample in samples if sample is not None]
        if not samples:
            return None
        train_x = torch.stack(samples, dim=0)
    else:
        return None

    train_x = train_x.to(device=reference.device, dtype=reference.dtype)
    lower = train_x.amin()
    upper = train_x.amax()
    if not torch.isfinite(lower) or not torch.isfinite(upper) or upper <= lower:
        return None
    setattr(batch, "_privacy_training_x_bounds", (lower.detach().cpu(), upper.detach().cpu()))
    return lower, upper


def _sample_x_from_dataset(dataset: Any, index: int) -> torch.Tensor | None:
    sample = dataset[index]
    if not isinstance(sample, (tuple, list)) or len(sample) < 1:
        return None
    x = sample[0]
    if not torch.is_tensor(x):
        return None
    if x.dim() == 3:
        # Graph/grid view samples are usually [T,N,F]. Convert to [N,T,F].
        return x.permute(1, 0, 2).contiguous()
    return x


def _training_x_std(batch: Any, reference: torch.Tensor, args: Any) -> torch.Tensor | None:
    cached = getattr(batch, "_privacy_training_x_std", None)
    if cached is not None:
        return cached.to(device=reference.device, dtype=reference.dtype)

    dataset = getattr(batch, "dataset", None)
    tensors = getattr(dataset, "tensors", None)
    if tensors and torch.is_tensor(tensors[0]):
        train_x = tensors[0].detach().float()
    elif dataset is not None and hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        max_samples = max(1, int(getattr(args, "traffic_prior_samples", 256)))
        n_samples = min(len(dataset), max_samples)
        samples = []
        for idx in range(n_samples):
            x_i = _sample_x_from_dataset(dataset, idx)
            if x_i is not None:
                samples.append(x_i.float())
        if not samples:
            return None
        train_x = torch.stack(samples, dim=0)
    else:
        return None

    std = train_x.reshape(-1).std(unbiased=False)
    if not torch.isfinite(std) or std <= 1e-12:
        return None
    std = std.to(device=reference.device, dtype=reference.dtype)
    setattr(batch, "_privacy_training_x_std", std.detach().cpu())
    return std


def _physical_values(x: torch.Tensor, batch: Any) -> torch.Tensor:
    scaler = getattr(batch, "scaler", None)
    if scaler is None or not hasattr(scaler, "inverse_transform"):
        return x
    return scaler.inverse_transform(x)


def _edge_index_for_x(batch: Any, x: torch.Tensor) -> torch.Tensor | None:
    edge_index = getattr(batch, "edge_index", None)
    if edge_index is None or x.dim() < 4:
        return None
    if not torch.is_tensor(edge_index):
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.long().to(x.device)
    if edge_index.dim() != 2 or edge_index.shape[0] != 2 or edge_index.numel() == 0:
        return None
    num_nodes = x.shape[1]
    row, col = edge_index[0], edge_index[1]
    valid = (row >= 0) & (row < num_nodes) & (col >= 0) & (col < num_nodes)
    if not torch.any(valid):
        return None
    return torch.stack([row[valid], col[valid]], dim=0)


def traffic_prior_loss(dummy_x: torch.Tensor, batch: Any, args: Any) -> torch.Tensor:
    if not bool(int(getattr(args, "traffic_prior", 0))):
        return dummy_x.new_zeros(())

    total = dummy_x.new_zeros(())

    range_weight = float(getattr(args, "traffic_range_weight", 0.0))
    bounds = _training_x_bounds(batch, dummy_x)
    if range_weight > 0 and bounds is not None:
        lower, upper = bounds
        margin = 0.05 * (upper - lower)
        range_loss = (
            F.relu((lower - margin) - dummy_x).pow(2).mean()
            + F.relu(dummy_x - (upper + margin)).pow(2).mean()
        )
        total = total + range_weight * range_loss

    nonnegative_weight = float(getattr(args, "traffic_nonnegative_weight", 0.0))
    if nonnegative_weight > 0:
        physical_x = _physical_values(dummy_x, batch)
        scaler = getattr(batch, "scaler", None)
        scale_value = getattr(scaler, "metrics_coef", 1.0)
        scale = _as_like_tensor(scale_value, dummy_x).abs().mean().clamp_min(1.0)
        nonnegative_loss = F.relu(-physical_x).div(scale).pow(2).mean()
        total = total + nonnegative_weight * nonnegative_loss

    temporal_weight = float(getattr(args, "traffic_temporal_weight", 0.0))
    if temporal_weight > 0 and dummy_x.dim() == 4 and dummy_x.shape[2] > 1:
        temporal_loss = (dummy_x[:, :, 1:, :] - dummy_x[:, :, :-1, :]).pow(2).mean()
        total = total + temporal_weight * temporal_loss

    spatial_weight = float(getattr(args, "traffic_spatial_weight", 0.0))
    edge_index = _edge_index_for_x(batch, dummy_x)
    if spatial_weight > 0 and edge_index is not None:
        row, col = edge_index
        spatial_loss = (dummy_x[:, row, :, :] - dummy_x[:, col, :, :]).pow(2).mean()
        total = total + spatial_weight * spatial_loss

    std_weight = float(getattr(args, "traffic_std_weight", 0.0))
    if std_weight > 0:
        train_std = _training_x_std(batch, dummy_x, args)
        if train_std is not None:
            std_ratio = float(getattr(args, "traffic_std_ratio", 1.0))
            dummy_std = dummy_x.reshape(-1).std(unbiased=False)
            target_std = train_std * std_ratio
            std_loss = F.relu(target_std - dummy_std).pow(2)
            total = total + std_weight * std_loss

    return total


def fuels_periodicity_prior_loss(
    model: torch.nn.Module,
    dummy_x: torch.Tensor,
    batch: Any,
    args: Any,
) -> torch.Tensor:
    weight = float(getattr(args, "fuels_periodicity_weight", 0.0))
    if weight <= 0.0:
        return dummy_x.new_zeros(())

    model_name = str(getattr(args, "model", "") or "")
    if model_name.lower() != "fuels":
        return dummy_x.new_zeros(())
    if not hasattr(model, "encode"):
        return dummy_x.new_zeros(())
    if dummy_x.dim() != 4 or int(dummy_x.shape[0]) <= 1:
        return dummy_x.new_zeros(())

    # FUELS paper uses a periodicity-aware prototype obtained by averaging
    # batch representations. Encourage the reconstructed batch encodings to
    # lie in a compact, average-able semantic cluster.
    with torch.backends.cudnn.flags(enabled=False):
        reprs = model.encode(dummy_x)
    if not torch.is_tensor(reprs) or reprs.dim() != 2 or reprs.shape[0] <= 1:
        return dummy_x.new_zeros(())

    repr_mean = reprs.mean(dim=0, keepdim=True)
    mode = str(getattr(args, "fuels_periodicity_mode", "mixed")).lower()

    total = dummy_x.new_zeros(())
    if mode in ("mean_l2", "mixed"):
        total = total + (reprs - repr_mean).pow(2).mean()
    if mode in ("mean_cos", "mixed"):
        repr_norm = F.normalize(reprs, p=2, dim=-1)
        mean_norm = F.normalize(repr_mean, p=2, dim=-1)
        cos_loss = 1.0 - (repr_norm * mean_norm).sum(dim=-1).mean()
        total = total + cos_loss

    return weight * total


def model_specific_prior_loss(
    model: torch.nn.Module,
    dummy_x: torch.Tensor,
    batch: Any,
    args: Any,
) -> torch.Tensor:
    total = traffic_prior_loss(dummy_x, batch, args)
    total = total + fuels_periodicity_prior_loss(model, dummy_x, batch, args)
    return total


def _is_improved(current: float, best: float, min_delta: float) -> bool:
    return current < best - min_delta


def _should_log_round(round_idx: int, total_rounds: int, interval: int) -> bool:
    if interval <= 0:
        return False
    return (round_idx + 1) % interval == 0 or (round_idx + 1) == total_rounds


def _should_checkpoint_round(round_idx: int, interval: int) -> bool:
    if interval <= 0:
        return False
    return (round_idx + 1) % interval == 0


def _format_metric_log(
    *,
    attack_name: str,
    round_idx: int,
    current_loss: float,
    best_loss: float,
    stale_rounds: int,
    dummy_x: torch.Tensor,
    real_x: torch.Tensor,
    batch: Any,
    args: Any,
) -> str:
    metrics = reconstruction_metrics(
        reconstructed_x=dummy_x.detach(),
        real_x=real_x,
        scaler=getattr(batch, "scaler", None),
        mape_eps=getattr(args, "mape_eps", 10.0),
    )
    metric_text = " ".join(
        f"{key}={metrics[key]:.6f}"
        for key in ("PCC", "MAE", "MSE", "RMSE", "MAPE", "WMAPE", "MAPE_VALID_RATIO")
    )
    return (
        f"[Privacy][{attack_name}] round={round_idx + 1} "
        f"attack_loss={current_loss:.6e} best={best_loss:.6e} "
        f"stale={stale_rounds} {metric_text}"
    )


def _checkpoint_stem(args: Any, attack_name: str) -> str:
    return (
        f"{args.model}_{args.dataset_name}_{args.feature_type}_"
        f"client{args.client_rank}_{args.split}{args.sample_index}_{attack_name}"
    )


def _save_attack_checkpoint(
    *,
    attack_name: str,
    round_idx: int,
    current_loss: float,
    best_loss: float,
    stale_rounds: int,
    dummy_x: torch.Tensor,
    dummy_y: torch.Tensor | None,
    optimizer: torch.optim.Optimizer | None,
    real_x: torch.Tensor,
    batch: Any,
    args: Any,
) -> Path:
    metrics = reconstruction_metrics(
        reconstructed_x=dummy_x.detach(),
        real_x=real_x,
        scaler=getattr(batch, "scaler", None),
        mape_eps=getattr(args, "mape_eps", 10.0),
    )
    checkpoint_dir = Path(args.result_dir) / args.model / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    round_no = round_idx + 1
    checkpoint_path = checkpoint_dir / f"{_checkpoint_stem(args, attack_name)}_round{round_no}.pt"
    torch.save(
        {
            "round": round_no,
            "attack": attack_name,
            "model": args.model,
            "dataset_name": args.dataset_name,
            "feature_type": args.feature_type,
            "client_rank": args.client_rank,
            "split": args.split,
            "sample_index": args.sample_index,
            "attack_node_index": args.attack_node_index,
            "attack_loss": current_loss,
            "best_attack_loss": best_loss,
            "stale_rounds": stale_rounds,
            "metrics": metrics,
            "dummy_x": dummy_x.detach().cpu(),
            "dummy_y": None if dummy_y is None else dummy_y.detach().cpu(),
            "optimizer_state": None if optimizer is None else optimizer.state_dict(),
            "real_x": real_x.detach().cpu(),
        },
        checkpoint_path,
    )
    return checkpoint_path


def _load_attack_checkpoint(path: str, device: torch.device | str) -> Dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Attack checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device)


def _maybe_resume_dummy_x(
    *,
    real_x: torch.Tensor,
    init: str,
    args: Any,
) -> tuple[torch.Tensor, Dict[str, Any] | None]:
    resume_path = getattr(args, "resume_attack_checkpoint", "")
    if not resume_path:
        return _init_like(real_x, init).detach().requires_grad_(True), None

    checkpoint = _load_attack_checkpoint(resume_path, real_x.device)
    if "dummy_x" not in checkpoint:
        raise KeyError(f"Attack checkpoint {resume_path} does not contain dummy_x.")
    dummy_x = checkpoint["dummy_x"].to(device=real_x.device, dtype=real_x.dtype)
    if tuple(dummy_x.shape) != tuple(real_x.shape):
        raise ValueError(
            f"Checkpoint dummy_x shape {tuple(dummy_x.shape)} does not match real_x shape {tuple(real_x.shape)}."
        )
    print(
        f"[Privacy] Resuming attack from {resume_path} at round={checkpoint.get('round', 0)}",
        flush=True,
    )
    return dummy_x.detach().requires_grad_(True), checkpoint


def _maybe_resume_dummy_y(
    *,
    real_y: torch.Tensor,
    init: str,
    use_dummy_y: bool,
    checkpoint: Dict[str, Any] | None,
) -> torch.Tensor | None:
    if not use_dummy_y:
        return None
    if checkpoint is not None and checkpoint.get("dummy_y") is not None:
        dummy_y = checkpoint["dummy_y"].to(device=real_y.device, dtype=real_y.dtype)
        if tuple(dummy_y.shape) != tuple(real_y.shape):
            raise ValueError(
                f"Checkpoint dummy_y shape {tuple(dummy_y.shape)} does not match real_y shape {tuple(real_y.shape)}."
            )
        return dummy_y.detach().requires_grad_(True)
    return _init_like(real_y, init).detach().requires_grad_(True)


class GradientMatchingAttack:
    name = "gradient"

    def run(self, adapter: Any, model: torch.nn.Module, batch: Any, args: Any) -> AttackResult:
        model.train()
        for param in model.parameters():
            param.requires_grad_(True)

        real_x = batch.real_x.detach()
        real_y = batch.real_y.detach()

        model.zero_grad(set_to_none=True)
        print(
            f"[Privacy][{self.name}] computing real leak...",
            flush=True,
        )
        real_leak = adapter.compute_leak(
            model=model,
            x=real_x,
            y=real_y,
            batch=batch,
            create_graph=False,
            leak_type=self.name,
        )
        _lock_fedmssa_attack_rank(adapter, real_leak, args)
        real_leak = detach_tree(real_leak)
        he_insider_trace = str(getattr(args, "he_insider_trace", "") or getattr(args, "he_ttp_insider_trace", "") or "").strip()
        he_collusion_trace = str(getattr(args, "he_collusion_trace", "") or "").strip()
        if he_insider_trace and he_collusion_trace:
            raise ValueError("Pass only one of --he_insider_trace or --he_collusion_trace.")
        if he_insider_trace:
            real_leak, trace_metadata = _load_he_insider_trace(he_insider_trace, model, real_leak)
            protection_metadata = {
                "protection": "he",
                "he_observation_source": "explicit_insider_training_trace",
                "he_insider_trace": he_insider_trace,
                "he_trace_metadata": trace_metadata,
                "threat_model": trace_metadata.get("threat_model", "HE insider upper bound"),
            }
            print(f"[Privacy][HE] replaying privileged observed payload from trace={he_insider_trace}", flush=True)
        elif he_collusion_trace:
            real_leak, trace_metadata = _load_he_collusion_trace(he_collusion_trace, model, real_leak)
            protection_metadata = {
                "protection": "he",
                "he_observation_source": "aggregate_minus_k_minus_1_colluding_client_updates",
                "he_collusion_trace": he_collusion_trace,
                "he_trace_metadata": trace_metadata,
                "threat_model": trace_metadata.get(
                    "threat_model", "HE-SA server plus K-1 colluding clients residual attack"
                ),
            }
            print(f"[Privacy][HE] replaying K-1 collusion residual from trace={he_collusion_trace}", flush=True)
        else:
            real_leak, protection_metadata = protect_observed_leak(real_leak, args, adapter)
        print(
            f"[Privacy][{self.name}] real leak ready.",
            flush=True,
        )

        dummy_y_mode = getattr(args, "dummy_y_mode", "optimize")
        use_dummy_y = getattr(adapter, "requires_dummy_y", False) and dummy_y_mode == "optimize"
        if getattr(args, "optimize_dummy_y", False):
            dummy_y_mode = "optimize"
            use_dummy_y = True
        if getattr(args, "no_optimize_dummy_y", False):
            dummy_y_mode = "real"
            use_dummy_y = False

        dummy_x, resume_checkpoint = _maybe_resume_dummy_x(real_x=real_x, init=args.init, args=args)

        dummy_y = None
        opt_params: List[torch.Tensor] = [dummy_x]
        if use_dummy_y:
            dummy_y = _maybe_resume_dummy_y(
                real_y=real_y,
                init=args.init,
                use_dummy_y=use_dummy_y,
                checkpoint=resume_checkpoint,
            )
            opt_params.append(dummy_y)
        elif dummy_y_mode == "zeros":
            dummy_y = torch.zeros_like(real_y).detach()

        optimizer = _build_attack_optimizer(opt_params, args)
        resume_optimizer_state = bool(int(getattr(args, "resume_optimizer_state", 0)))
        if (
            resume_optimizer_state
            and resume_checkpoint is not None
            and resume_checkpoint.get("optimizer_state") is not None
        ):
            try:
                optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
            except ValueError as exc:
                print(f"[Privacy] Skip optimizer resume: {exc}", flush=True)
        elif resume_checkpoint is not None:
            print(
                f"[Privacy] Using checkpoint dummy tensors with fresh Adam optimizer lr={args.attack_lr}",
                flush=True,
            )
        history: List[float] = []
        start_round = int(resume_checkpoint.get("round", 0)) if resume_checkpoint is not None else 0
        best_loss = (
            float(resume_checkpoint.get("best_attack_loss", float("inf")))
            if resume_checkpoint is not None
            else float("inf")
        )
        final_dummy_x = dummy_x.detach().clone()
        final_dummy_y = None if dummy_y is None else dummy_y.detach().clone()
        # The attacker cannot select by PCC/MAE because real_x is unknown.
        # It can, however, retain the iterate that best matches the observed
        # server-side leak.  Keep that pre-step iterate and return it below.
        best_dummy_x = final_dummy_x.detach().clone()
        best_dummy_y = None if final_dummy_y is None else final_dummy_y.detach().clone()
        best_round = start_round
        stale_rounds = int(resume_checkpoint.get("stale_rounds", 0)) if resume_checkpoint is not None else 0
        patience = max(0, int(getattr(args, "early_stop_patience", 0)))
        min_delta = float(getattr(args, "early_stop_min_delta", 0.0))
        nonfinite_stopped = False
        ufcl_nonfinite_grad_events = 0
        ufcl_max_nonfinite_grad_ratio = 0.0

        match_mode = getattr(args, "grad_match", "mse")
        log_interval = int(getattr(args, "log_interval", 0))
        checkpoint_interval = int(getattr(args, "attack_checkpoint_interval", 0))

        if start_round >= args.attack_iters:
            print(
                f"[Privacy] Resume round {start_round} is already >= attack_iters {args.attack_iters}; "
                "no extra optimization will run.",
                flush=True,
            )

        verbose_round_cutoff = int(getattr(args, "attack_verbose_rounds", 3))

        for round_idx in range(start_round, args.attack_iters):
            round_begin = time.time()
            optimizer.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)

            leak_y = dummy_y if dummy_y is not None else real_y
            verbose_round = (round_idx - start_round) < verbose_round_cutoff
            if verbose_round:
                print(
                    f"[Privacy][{self.name}] computing dummy leak for round={round_idx + 1}...",
                    flush=True,
                )
            dummy_leak = adapter.compute_leak(
                model=model,
                x=dummy_x,
                y=leak_y,
                batch=batch,
                create_graph=True,
                leak_type=self.name,
            )
            if verbose_round:
                print(
                    f"[Privacy][{self.name}] dummy leak ready for round={round_idx + 1}.",
                    flush=True,
                )
                print(
                    f"[Privacy][{self.name}] computing leak distance for round={round_idx + 1}...",
                    flush=True,
                )
            leak_loss = leak_distance(dummy_leak, real_leak, mode=match_mode)
            if verbose_round:
                print(
                    f"[Privacy][{self.name}] leak distance ready for round={round_idx + 1}.",
                    flush=True,
                )
            prior_loss = model_specific_prior_loss(model, dummy_x, batch, args)
            attack_loss = leak_loss + prior_loss
            if not torch.isfinite(attack_loss):
                nonfinite_stopped = True
                print(
                    f"[Privacy][{self.name}] stopping at round={round_idx + 1}: non-finite attack loss.",
                    flush=True,
                )
                break
            prev_dummy_x = dummy_x.detach().clone()
            prev_dummy_y = None if dummy_y is None else dummy_y.detach().clone()
            if verbose_round:
                print(
                    f"[Privacy][{self.name}] backward() for round={round_idx + 1}...",
                    flush=True,
                )
            attack_loss.backward()
            grad_stats = _stabilize_ufcl_dummy_gradients(opt_params, args)
            if grad_stats is not None and grad_stats["nonfinite_elements"]:
                ufcl_nonfinite_grad_events += 1
                total_elements = max(1, grad_stats["total_elements"])
                nonfinite_ratio = grad_stats["nonfinite_elements"] / total_elements
                ufcl_max_nonfinite_grad_ratio = max(ufcl_max_nonfinite_grad_ratio, nonfinite_ratio)
                print(
                    f"[Privacy][UFCL] round={round_idx + 1}: replaced "
                    f"{grad_stats['nonfinite_elements']}/{total_elements} non-finite dummy-gradient elements "
                    "before clipping.",
                    flush=True,
                )
                if grad_stats["all_nonfinite"]:
                    nonfinite_stopped = True
                    final_dummy_x = prev_dummy_x
                    final_dummy_y = prev_dummy_y
                    print(
                        f"[Privacy][UFCL] stopping at round={round_idx + 1}: all dummy gradients were "
                        "non-finite; no zero-gradient update will be treated as a valid attack step.",
                        flush=True,
                    )
                    break
            if verbose_round:
                print(
                    f"[Privacy][{self.name}] optimizer.step() for round={round_idx + 1}...",
                    flush=True,
                )
            optimizer.step()
            if verbose_round:
                print(
                    f"[Privacy][{self.name}] round={round_idx + 1} step finished "
                    f"(elapsed={time.time() - round_begin:.2f}s).",
                    flush=True,
                )
            dummy_y_finite = dummy_y is None or torch.isfinite(dummy_y).all()
            if not torch.isfinite(dummy_x).all() or not dummy_y_finite:
                nonfinite_stopped = True
                final_dummy_x = prev_dummy_x
                final_dummy_y = prev_dummy_y
                print(
                    f"[Privacy][{self.name}] stopping at round={round_idx + 1}: dummy tensor became non-finite.",
                    flush=True,
                )
                break
            current_loss = float(attack_loss.detach().cpu())
            history.append(current_loss)
            final_dummy_x = dummy_x.detach().clone()
            final_dummy_y = None if dummy_y is None else dummy_y.detach().clone()

            if _is_improved(current_loss, best_loss, min_delta):
                best_loss = current_loss
                stale_rounds = 0
                # attack_loss was evaluated before optimizer.step(), so the
                # matching reconstruction is prev_dummy_x rather than the
                # just-updated tensor.
                best_dummy_x = prev_dummy_x
                best_dummy_y = prev_dummy_y
                best_round = round_idx + 1
            else:
                stale_rounds += 1

            if _should_log_round(round_idx, args.attack_iters, log_interval):
                print(
                    _format_metric_log(
                        attack_name=self.name,
                        round_idx=round_idx,
                        current_loss=current_loss,
                        best_loss=best_loss,
                        stale_rounds=stale_rounds,
                        dummy_x=dummy_x,
                        real_x=real_x,
                        batch=batch,
                        args=args,
                    ),
                    flush=True,
                )

            if _should_checkpoint_round(round_idx, checkpoint_interval):
                checkpoint_path = _save_attack_checkpoint(
                    attack_name=self.name,
                    round_idx=round_idx,
                    current_loss=current_loss,
                    best_loss=best_loss,
                    stale_rounds=stale_rounds,
                    dummy_x=dummy_x,
                    dummy_y=dummy_y,
                    optimizer=optimizer,
                    real_x=real_x,
                    batch=batch,
                    args=args,
                )
                print(f"[Privacy][{self.name}] saved checkpoint to {checkpoint_path}", flush=True)

            if patience > 0 and stale_rounds >= patience:
                print(
                    f"[Privacy][{self.name}] early stopping at round={round_idx + 1}: "
                    f"attack_loss did not improve for {stale_rounds} rounds "
                    f"(best={best_loss:.6e}).",
                    flush=True,
                )
                break

        return AttackResult(
            reconstructed_x=best_dummy_x,
            reconstructed_y=best_dummy_y,
            final_loss=best_loss if best_loss < float("inf") else float("nan"),
            history=history,
            attack_name=self.name,
            metadata={
                "optimized_dummy_y": dummy_y is not None,
                "dummy_y_mode": dummy_y_mode,
                "grad_match": match_mode,
                "attack_optimizer": getattr(args, "attack_optimizer", "adam"),
                "local_update_steps": getattr(args, "local_update_steps", None),
                "local_update_lr": getattr(args, "local_update_lr", None),
                "local_update_wd": getattr(args, "local_update_wd", None),
                "ufcl_sequence_steps": getattr(batch, "ufcl_sequence_steps", None),
                "ufcl_train_batch_size": getattr(batch, "ufcl_train_batch_size", None),
                "ufcl_synthetic_replay": getattr(args, "model", "").upper() == "UFCL",
                "ufcl_attack_local_adam_epsilon": getattr(adapter, "attack_local_adam_epsilon", None),
                "ufcl_attack_grad_stabilized": getattr(args, "model", "").upper() == "UFCL",
                "ufcl_attack_grad_clip_norm": 100.0 if getattr(args, "model", "").upper() == "UFCL" else None,
                "ufcl_nonfinite_gradient_events": ufcl_nonfinite_grad_events,
                "ufcl_max_nonfinite_gradient_ratio": ufcl_max_nonfinite_grad_ratio,
                "batch_aggregate_leak": bool(getattr(adapter, "batch_aggregate_leak", False)),
                "attack_surface": (
                    "ufcl_per_batch_gradient_upper"
                    if str(getattr(adapter, "name", "")).upper() == "UFCL" and self.name == "gradient"
                    else self.name
                ),
                "sfl_attack_surface": getattr(args, "sfl_attack_surface", None),
                "sfl_update_optimizer": getattr(args, "sfl_update_optimizer", None),
                "sfl_graph_weight": getattr(args, "sfl_graph_weight", None),
                "sfl_reg_anchor": getattr(args, "sfl_reg_anchor", None),
                "sfl_lambda": getattr(args, "sfl_lambda", None),
                "sfl_m_steps": getattr(args, "sfl_m_steps", None),
                "twomgtcn_attack_surface": getattr(args, "twomgtcn_attack_surface", None),
                "twomgtcn_update_optimizer": getattr(args, "twomgtcn_update_optimizer", None),
                "twomgtcn_lfac_alpha": getattr(args, "twomgtcn_lfac_alpha", None),
                "twomgtcn_fpass_weight": getattr(args, "twomgtcn_fpass_weight", None),
                "twomgtcn_use_ext": getattr(args, "twomgtcn_use_ext", None),
                "fedtps_update_scope": getattr(args, "fedtps_update_scope", None),
                "fed4tp_attack_surface": getattr(args, "fed4tp_attack_surface", None),
                "fed4tp_update_optimizer": getattr(args, "fed4tp_update_optimizer", None),
                "fed4tp_mask_weight": getattr(args, "fed4tp_mask_weight", None),
                "fed4tp_mask_temperature": getattr(args, "fed4tp_mask_temperature", None),
                "fedtps_update_optimizer": getattr(args, "fedtps_update_optimizer", None),
                "fedtps_include_agg_patterns": getattr(args, "fedtps_include_agg_patterns", None),
                "fedtps_agg_patterns_weight": getattr(args, "fedtps_agg_patterns_weight", None),
                "fedtps_k": getattr(args, "fedtps_k", None),
                "stfam_update_optimizer": (
                    "adam_one_step_surrogate" if getattr(args, "model", "").upper() == "STFAM" else None
                ),
                "stfam_attack_local_adam_epsilon": getattr(adapter, "attack_local_adam_epsilon", None),
                "fedagat_update_optimizer": getattr(args, "fedagat_update_optimizer", None),
                "fedagat_loss_start_step": getattr(args, "fedagat_loss_start_step", None),
                "fedagat_batches_per_epoch": getattr(args, "fedagat_batches_per_epoch", None),
                "fedagat_train_batch_size": getattr(args, "fedagat_train_batch_size", None),
                "fedagat_max_epochs": getattr(args, "fedagat_max_epochs", None),
                "stagcn_ec_update_optimizer": getattr(args, "stagcn_ec_update_optimizer", None),
                "stagcn_ec_threat_model": (
                    "malicious_rsu_update" if getattr(args, "model", "").upper() in ("STAGCN-EC", "STAGCN_EC", "STGCN")
                    and self.name == "model_update" else None
                ),
                "fedmetro_attack_surface": getattr(args, "fedmetro_attack_surface", None),
                "fedmetro_agg_weight": getattr(args, "fedmetro_agg_weight", None),
                "fedmetro_gagg_weight": getattr(args, "fedmetro_gagg_weight", None),
                "fedmetro_update_weight": getattr(args, "fedmetro_update_weight", None),
                "fedmetro_lambda_reg": getattr(args, "fedmetro_lambda_reg", None),
                "fedmssa_page_length": getattr(args, "fedmssa_page_length", None),
                "fedmssa_rank": getattr(args, "fedmssa_rank", None),
                "fedmssa_missing_ratio": getattr(args, "fedmssa_missing_ratio", None),
                "fedmssa_impute_rounds": getattr(args, "fedmssa_impute_rounds", None),
                "fedmssa_impute_local_steps": getattr(args, "fedmssa_impute_local_steps", None),
                "fedmssa_impute_lr": getattr(args, "fedmssa_impute_lr", None),
                "fedmssa_phase1_device": getattr(args, "fedmssa_phase1_device", None),
                "fedmssa_update_optimizer": getattr(args, "fedmssa_update_optimizer", None),
                "fedtse_update_optimizer": getattr(args, "fedtse_update_optimizer", None),
                "fedgru_update_optimizer": getattr(args, "fedgru_update_optimizer", None),
                "tdlr_update_optimizer": getattr(args, "tdlr_update_optimizer", None),
                "fuels_activation_scope": getattr(args, "fuels_activation_scope", None),
                "fuels_dr": getattr(args, "fuels_dr", None),
                "traffic_prior": bool(int(getattr(args, "traffic_prior", 0))),
                "traffic_range_weight": getattr(args, "traffic_range_weight", 0.0),
                "traffic_nonnegative_weight": getattr(args, "traffic_nonnegative_weight", 0.0),
                "traffic_temporal_weight": getattr(args, "traffic_temporal_weight", 0.0),
                "traffic_spatial_weight": getattr(args, "traffic_spatial_weight", 0.0),
                "traffic_std_weight": getattr(args, "traffic_std_weight", 0.0),
                "traffic_std_ratio": getattr(args, "traffic_std_ratio", 1.0),
                "resume_optimizer_state": resume_optimizer_state,
                "best_attack_loss": best_loss,
                "selected_attack_round": best_round,
                "last_attack_loss": history[-1] if history else float("nan"),
                "resume_start_round": start_round,
                "new_fit_rounds": len(history),
                "actual_fit_rounds": start_round + len(history),
                "early_stopped": patience > 0 and stale_rounds >= patience,
                "nonfinite_stopped": nonfinite_stopped,
                **protection_metadata,
            },
        )


class ModelUpdateMatchingAttack(GradientMatchingAttack):
    name = "model_update"


class HEKMinus2CollusionModelUpdateAttack:
    """Jointly match the two-client residual visible under HE-SA K-2 collusion.

    The attacker knows neither hidden client's x or y.  It optimizes two dummy
    records whose weighted local updates match the residual.  Only client0's
    dummy is evaluated; the peer dummy is a nuisance variable, not a metric
    target.  Any adapter that exposes a differentiable ``model_update`` leak
    can use this residual objective.
    """

    name = "he_kminus2_model_update"

    @staticmethod
    def _combine(left: tuple[torch.Tensor, ...], right: tuple[torch.Tensor, ...], lw: float, rw: float) -> tuple[torch.Tensor, ...]:
        if len(left) != len(right):
            raise ValueError("K-2 dummy updates have different parameter counts.")
        return tuple(lw * a + rw * b for a, b in zip(left, right))

    def run(self, adapter: Any, model: torch.nn.Module, batch: Any, args: Any) -> AttackResult:
        trace_path = str(getattr(args, "he_kminus2_trace", "") or "").strip()
        peer_model = getattr(args, "_he_kminus2_peer_model", None)
        peer_batch = getattr(args, "_he_kminus2_peer_batch", None)
        if not trace_path or peer_model is None or peer_batch is None:
            raise ValueError("K-2 attack requires --he_kminus2_trace plus prepared peer model and batch.")
        if str(getattr(args, "dummy_y_mode", "optimize")) != "optimize":
            raise ValueError("K-2 collusion attack requires --dummy_y_mode optimize; real labels are not attacker-visible.")

        model.train()
        peer_model.train()
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        for parameter in peer_model.parameters():
            parameter.requires_grad_(True)

        # This zero tuple supplies only the public parameter layout.  No
        # target x/y-derived leak is computed before loading the observation.
        expected = tuple(torch.zeros_like(parameter) for parameter in model.parameters() if parameter.requires_grad)
        observed, trace_metadata = _load_he_kminus2_trace(trace_path, model, expected)
        target_weight = float(trace_metadata["target_weight"])
        peer_weight = float(trace_metadata["peer_weight"])
        print(
            f"[Privacy][HE] K-2 residual: target={trace_metadata['target_client']} "
            f"peer={trace_metadata['peer_hidden_client']} weights=({target_weight:.1f},{peer_weight:.1f})",
            flush=True,
        )
        return _run_he_kminus2_attack(
            self, adapter, model, batch, args, peer_model, peer_batch, observed,
            trace_path, trace_metadata, target_weight, peer_weight,
        )


def _run_he_kminus2_attack(
    attack: HEKMinus2CollusionModelUpdateAttack,
    adapter: Any,
    model: torch.nn.Module,
    batch: Any,
    args: Any,
    peer_model: torch.nn.Module,
    peer_batch: Any,
    observed: TensorTree,
    trace_path: str,
    trace_metadata: Dict[str, Any],
    target_weight: float,
    peer_weight: float,
) -> AttackResult:
    """Fit the two unknown local updates in a K-2 HE-SA residual."""
    real_x = batch.real_x.detach()  # evaluator-only; never used in matching.
    dummy_x = _init_like(real_x, args.init).detach().requires_grad_(True)
    peer_dummy_x = _init_like(peer_batch.real_x.detach(), args.init).detach().requires_grad_(True)
    dummy_y = _init_like(batch.real_y.detach(), args.init).detach().requires_grad_(True)
    peer_dummy_y = _init_like(peer_batch.real_y.detach(), args.init).detach().requires_grad_(True)
    opt_params = [dummy_x, peer_dummy_x, dummy_y, peer_dummy_y]
    optimizer = _build_attack_optimizer(opt_params, args)
    best_loss, best_x, best_y, best_round, stale_rounds = float("inf"), dummy_x.detach().clone(), dummy_y.detach().clone(), 0, 0
    history: List[float] = []
    patience = max(0, int(getattr(args, "early_stop_patience", 0)))
    min_delta = float(getattr(args, "early_stop_min_delta", 0.0))
    match_mode, log_interval = getattr(args, "grad_match", "mixed"), int(getattr(args, "log_interval", 0))

    for round_idx in range(int(args.attack_iters)):
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        peer_model.zero_grad(set_to_none=True)
        target_update = adapter.compute_leak(model=model, x=dummy_x, y=dummy_y, batch=batch, create_graph=True, leak_type="model_update")
        peer_update = adapter.compute_leak(model=peer_model, x=peer_dummy_x, y=peer_dummy_y, batch=peer_batch, create_graph=True, leak_type="model_update")
        residual = attack._combine(target_update, peer_update, target_weight, peer_weight)
        attack_loss = leak_distance(residual, observed, mode=match_mode) + model_specific_prior_loss(model, dummy_x, batch, args) + model_specific_prior_loss(peer_model, peer_dummy_x, peer_batch, args)
        if not torch.isfinite(attack_loss):
            print(f"[Privacy][HE K-2] non-finite loss at round={round_idx + 1}", flush=True)
            break
        previous_x, previous_y = dummy_x.detach().clone(), dummy_y.detach().clone()
        attack_loss.backward()
        optimizer.step()
        if not all(torch.isfinite(value).all() for value in opt_params):
            print(f"[Privacy][HE K-2] non-finite dummy at round={round_idx + 1}", flush=True)
            break
        current_loss = float(attack_loss.detach().cpu())
        history.append(current_loss)
        if _is_improved(current_loss, best_loss, min_delta):
            best_loss, best_x, best_y, best_round, stale_rounds = current_loss, previous_x, previous_y, round_idx + 1, 0
        else:
            stale_rounds += 1
        if _should_log_round(round_idx, int(args.attack_iters), log_interval):
            print(_format_metric_log(attack_name=attack.name, round_idx=round_idx, current_loss=current_loss, best_loss=best_loss, stale_rounds=stale_rounds, dummy_x=previous_x, real_x=real_x, batch=batch, args=args), flush=True)
        if patience > 0 and stale_rounds >= patience:
            print(f"[Privacy][HE K-2] early stopping at round={round_idx + 1}", flush=True)
            break

    return AttackResult(
        reconstructed_x=best_x, reconstructed_y=best_y, final_loss=best_loss, history=history, attack_name=attack.name,
        metadata={
            "protection": "he", "attack_surface": "two_hidden_client_weighted_model_update_residual",
            "threat_model": trace_metadata["threat_model"], "he_kminus2_trace": trace_path,
            "he_trace_metadata": trace_metadata, "target_client": trace_metadata["target_client"],
            "peer_hidden_client": trace_metadata["peer_hidden_client"], "colluding_clients": trace_metadata["colluding_clients"],
            "target_weight": target_weight, "peer_weight": peer_weight, "labels_known": False,
            "optimized_dummy_y": True, "grad_match": match_mode, "best_attack_loss": best_loss,
            "selected_attack_round": best_round, "reporting_iterate": "minimum_matching_loss",
            "actual_fit_rounds": len(history), "early_stopped": patience > 0 and stale_rounds >= patience,
        },
    )


class HEOneClientCollusionModelUpdateAttack:
    """Joint attack against the three-client residual visible with one colluder."""

    name = "he_kminus3_model_update"

    def run(self, adapter: Any, model: torch.nn.Module, batch: Any, args: Any) -> AttackResult:
        trace_path = str(getattr(args, "he_kminus3_trace", "") or "").strip()
        peer_models = list(getattr(args, "_he_kminus3_peer_models", []) or [])
        peer_batches = list(getattr(args, "_he_kminus3_peer_batches", []) or [])
        if not trace_path or len(peer_models) != 2 or len(peer_batches) != 2:
            raise ValueError("K-3 attack requires --he_kminus3_trace plus two prepared peer models/batches.")
        if str(getattr(args, "dummy_y_mode", "optimize")) != "optimize":
            raise ValueError("One-client collusion attack requires --dummy_y_mode optimize; real labels are not attacker-visible.")
        models = [model, *peer_models]
        batches = [batch, *peer_batches]
        for current_model in models:
            current_model.train()
            for parameter in current_model.parameters():
                parameter.requires_grad_(True)
        expected = tuple(torch.zeros_like(parameter) for parameter in model.parameters() if parameter.requires_grad)
        observed, metadata = _load_he_kminus3_trace(trace_path, model, expected)
        weights = [float(value) for value in metadata["hidden_weights"]]
        print(
            f"[Privacy][HE] one-client collusion residual: hidden={metadata['hidden_clients']} "
            f"colluding={metadata['colluding_clients']} weights={weights}",
            flush=True,
        )

        # real_x is evaluator-only.  All three x/y variables below are unknown
        # to the attacker and jointly optimized from the residual.
        dummy_xs = [_init_like(current_batch.real_x.detach(), args.init).detach().requires_grad_(True) for current_batch in batches]
        dummy_ys = [_init_like(current_batch.real_y.detach(), args.init).detach().requires_grad_(True) for current_batch in batches]
        optimizer = _build_attack_optimizer([*dummy_xs, *dummy_ys], args)
        best_loss, best_x, best_y, best_round, stale = float("inf"), dummy_xs[0].detach().clone(), dummy_ys[0].detach().clone(), 0, 0
        terminal_x, terminal_y, terminal_loss = best_x.detach().clone(), best_y.detach().clone(), float("inf")
        history: List[float] = []
        patience = max(0, int(getattr(args, "early_stop_patience", 0)))
        min_delta = float(getattr(args, "early_stop_min_delta", 0.0))
        match_mode, log_interval = getattr(args, "grad_match", "mixed"), int(getattr(args, "log_interval", 0))

        for round_idx in range(int(args.attack_iters)):
            optimizer.zero_grad(set_to_none=True)
            for current_model in models:
                current_model.zero_grad(set_to_none=True)
            updates = [
                adapter.compute_leak(model=current_model, x=dummy_x, y=dummy_y, batch=current_batch, create_graph=True, leak_type="model_update")
                for current_model, current_batch, dummy_x, dummy_y in zip(models, batches, dummy_xs, dummy_ys)
            ]
            if not all(len(update) == len(updates[0]) for update in updates):
                raise ValueError("K-3 hidden clients have incompatible model-update layouts.")
            residual = tuple(sum(weight * update[index] for weight, update in zip(weights, updates)) for index in range(len(updates[0])))
            prior = sum(model_specific_prior_loss(current_model, dummy_x, current_batch, args) for current_model, dummy_x, current_batch in zip(models, dummy_xs, batches))
            attack_loss = leak_distance(residual, observed, mode=match_mode) + prior
            if not torch.isfinite(attack_loss):
                print(f"[Privacy][HE K-3] non-finite loss at round={round_idx + 1}", flush=True)
                break
            previous_x, previous_y = dummy_xs[0].detach().clone(), dummy_ys[0].detach().clone()
            attack_loss.backward()
            optimizer.step()
            if not all(torch.isfinite(value).all() for value in [*dummy_xs, *dummy_ys]):
                print(f"[Privacy][HE K-3] non-finite dummy at round={round_idx + 1}", flush=True)
                break
            current_loss = float(attack_loss.detach().cpu())
            history.append(current_loss)
            terminal_x = dummy_xs[0].detach().clone()
            terminal_y = dummy_ys[0].detach().clone()
            terminal_loss = current_loss
            if _is_improved(current_loss, best_loss, min_delta):
                best_loss, best_x, best_y, best_round, stale = current_loss, previous_x, previous_y, round_idx + 1, 0
            else:
                stale += 1
            if _should_log_round(round_idx, int(args.attack_iters), log_interval):
                print(_format_metric_log(
                    attack_name=self.name, round_idx=round_idx, current_loss=current_loss, best_loss=best_loss,
                    # attack_loss was evaluated on previous_x; log the same
                    # attacker-visible candidate that may be retained below.
                    stale_rounds=stale, dummy_x=previous_x, real_x=batch.real_x, batch=batch, args=args,
                ), flush=True)
            if patience > 0 and stale >= patience:
                print(f"[Privacy][HE K-3] early stopping at round={round_idx + 1}", flush=True)
                break
        terminal_mode = bool(getattr(args, "attack_return_terminal_iterate", False))
        reported_x, reported_y = (terminal_x, terminal_y) if terminal_mode else (best_x, best_y)
        reported_loss = terminal_loss if terminal_mode else best_loss
        return AttackResult(
            reconstructed_x=reported_x, reconstructed_y=reported_y, final_loss=reported_loss, history=history, attack_name=self.name,
            metadata={
                "protection": "he", "attack_surface": "three_hidden_client_weighted_model_update_residual",
                "threat_model": metadata["threat_model"], "he_kminus3_trace": trace_path,
                "he_trace_metadata": metadata, "hidden_clients": metadata["hidden_clients"],
                "colluding_clients": metadata["colluding_clients"], "hidden_weights": weights,
                "labels_known": False, "optimized_dummy_y": True, "grad_match": match_mode,
                "best_attack_loss": best_loss, "selected_attack_round": best_round,
                "reporting_iterate": "terminal_early_stop" if terminal_mode else "minimum_matching_loss",
                "terminal_attack_loss": terminal_loss,
                "actual_fit_rounds": len(history), "early_stopped": patience > 0 and stale >= patience,
            },
        )

        real_x = batch.real_x.detach()  # evaluator-only; never used in matching.
        dummy_x = _init_like(real_x, args.init).detach().requires_grad_(True)
        peer_dummy_x = _init_like(peer_batch.real_x.detach(), args.init).detach().requires_grad_(True)
        dummy_y = _init_like(batch.real_y.detach(), args.init).detach().requires_grad_(True)
        peer_dummy_y = _init_like(peer_batch.real_y.detach(), args.init).detach().requires_grad_(True)
        opt_params = [dummy_x, peer_dummy_x, dummy_y, peer_dummy_y]
        optimizer = _build_attack_optimizer(opt_params, args)

        best_loss = float("inf")
        best_x = dummy_x.detach().clone()
        best_y = dummy_y.detach().clone()
        history: List[float] = []
        best_round = 0
        stale_rounds = 0
        patience = max(0, int(getattr(args, "early_stop_patience", 0)))
        min_delta = float(getattr(args, "early_stop_min_delta", 0.0))
        match_mode = getattr(args, "grad_match", "mixed")
        log_interval = int(getattr(args, "log_interval", 0))

        for round_idx in range(int(args.attack_iters)):
            optimizer.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            peer_model.zero_grad(set_to_none=True)
            target_update = adapter.compute_leak(
                model=model, x=dummy_x, y=dummy_y, batch=batch, create_graph=True, leak_type="model_update"
            )
            peer_update = adapter.compute_leak(
                model=peer_model, x=peer_dummy_x, y=peer_dummy_y, batch=peer_batch,
                create_graph=True, leak_type="model_update"
            )
            residual = self._combine(target_update, peer_update, target_weight, peer_weight)
            leak_loss = leak_distance(residual, observed, mode=match_mode)
            prior_loss = (
                model_specific_prior_loss(model, dummy_x, batch, args)
                + model_specific_prior_loss(peer_model, peer_dummy_x, peer_batch, args)
            )
            attack_loss = leak_loss + prior_loss
            if not torch.isfinite(attack_loss):
                print(f"[Privacy][HE K-2] non-finite loss at round={round_idx + 1}", flush=True)
                break
            previous_x = dummy_x.detach().clone()
            previous_y = dummy_y.detach().clone()
            attack_loss.backward()
            optimizer.step()
            if not all(torch.isfinite(value).all() for value in opt_params):
                print(f"[Privacy][HE K-2] non-finite dummy at round={round_idx + 1}", flush=True)
                break
            current_loss = float(attack_loss.detach().cpu())
            history.append(current_loss)
            if _is_improved(current_loss, best_loss, min_delta):
                best_loss = current_loss
                best_x = previous_x
                best_y = previous_y
                best_round = round_idx + 1
                stale_rounds = 0
            else:
                stale_rounds += 1
            if _should_log_round(round_idx, int(args.attack_iters), log_interval):
                print(
                    _format_metric_log(
                        attack_name=self.name, round_idx=round_idx, current_loss=current_loss,
                        best_loss=best_loss, stale_rounds=stale_rounds, dummy_x=dummy_x,
                        real_x=real_x, batch=batch, args=args,
                    ),
                    flush=True,
                )
            if patience > 0 and stale_rounds >= patience:
                print(f"[Privacy][HE K-2] early stopping at round={round_idx + 1}", flush=True)
                break

        return AttackResult(
            reconstructed_x=best_x,
            reconstructed_y=best_y,
            final_loss=best_loss,
            history=history,
            attack_name=self.name,
            metadata={
                "protection": "he",
                "attack_surface": "two_hidden_client_weighted_model_update_residual",
                "threat_model": trace_metadata["threat_model"],
                "he_kminus2_trace": trace_path,
                "he_trace_metadata": trace_metadata,
                "target_client": trace_metadata["target_client"],
                "peer_hidden_client": trace_metadata["peer_hidden_client"],
                "colluding_clients": trace_metadata["colluding_clients"],
                "target_weight": target_weight,
                "peer_weight": peer_weight,
                "labels_known": False,
                "optimized_dummy_y": True,
                "grad_match": match_mode,
                "best_attack_loss": best_loss,
                "selected_attack_round": best_round,
                "actual_fit_rounds": len(history),
                "early_stopped": patience > 0 and stale_rounds >= patience,
            },
        )


class HEServerOnlyAggregateModelUpdateAttack:
    """Jointly reconstruct from the one aggregate visible to a non-colluding server.

    No individual client upload is exposed.  All four client records are
    nuisance variables in the same aggregate-matching objective; only client0
    is evaluated after fitting.
    """

    name = "he_server_aggregate_model_update"

    def run(self, adapter: Any, model: torch.nn.Module, batch: Any, args: Any) -> AttackResult:
        trace_path = str(getattr(args, "he_server_aggregate_trace", "") or "").strip()
        peer_models = list(getattr(args, "_he_server_aggregate_peer_models", []) or [])
        peer_batches = list(getattr(args, "_he_server_aggregate_peer_batches", []) or [])
        if not trace_path or len(peer_models) != 3 or len(peer_batches) != 3:
            raise ValueError("Server-only aggregate attack requires a four-client trace and three prepared peers.")
        if str(getattr(args, "dummy_y_mode", "optimize")) != "optimize":
            raise ValueError("Server-only aggregate attack requires --dummy_y_mode optimize; labels are not attacker-visible.")

        models, batches = [model, *peer_models], [batch, *peer_batches]
        for current_model in models:
            current_model.train()
            for parameter in current_model.parameters():
                parameter.requires_grad_(True)
        expected = tuple(torch.zeros_like(parameter) for parameter in model.parameters() if parameter.requires_grad)
        observed, metadata = _load_he_replay_trace(
            trace_path, model, expected,
            expected_observer="server_only_decrypted_aggregate",
            argument_name="--he_server_aggregate_trace",
        )
        weights = [float(value) for value in metadata.get("hidden_weights", [])]
        hidden = [int(value) for value in metadata.get("hidden_clients", [])]
        if len(weights) != 4 or len(hidden) != 4 or len(set(hidden)) != 4 or any(weight <= 0 for weight in weights):
            raise ValueError("Server-only aggregate trace must contain four distinct clients with positive weights.")
        leak_type = str(metadata.get("aggregate_leak_type", "model_update"))
        print(
            f"[Privacy][HE] server-only aggregate: all clients hidden={hidden}; weights={weights}; no colluder.",
            flush=True,
        )

        dummy_xs = [_init_like(current_batch.real_x.detach(), args.init).detach().requires_grad_(True) for current_batch in batches]
        dummy_ys = [_init_like(current_batch.real_y.detach(), args.init).detach().requires_grad_(True) for current_batch in batches]
        optimizer = _build_attack_optimizer([*dummy_xs, *dummy_ys], args)
        best_loss, best_x, best_y, best_round, stale = float("inf"), dummy_xs[0].detach().clone(), dummy_ys[0].detach().clone(), 0, 0
        terminal_x, terminal_y, terminal_loss = best_x.detach().clone(), best_y.detach().clone(), float("inf")
        history: List[float] = []
        patience = max(0, int(getattr(args, "early_stop_patience", 0)))
        min_delta = float(getattr(args, "early_stop_min_delta", 0.0))
        match_mode, log_interval = getattr(args, "grad_match", "mixed"), int(getattr(args, "log_interval", 0))

        for round_idx in range(int(args.attack_iters)):
            optimizer.zero_grad(set_to_none=True)
            for current_model in models:
                current_model.zero_grad(set_to_none=True)
            updates = [
                adapter.compute_leak(model=current_model, x=dummy_x, y=dummy_y, batch=current_batch,
                                     create_graph=True, leak_type=leak_type)
                for current_model, current_batch, dummy_x, dummy_y in zip(models, batches, dummy_xs, dummy_ys)
            ]
            if not all(len(update) == len(updates[0]) for update in updates):
                raise ValueError("Server-only hidden clients have incompatible model-update layouts.")
            aggregate = tuple(sum(weight * update[index] for weight, update in zip(weights, updates)) for index in range(len(updates[0])))
            prior = sum(model_specific_prior_loss(current_model, dummy_x, current_batch, args)
                        for current_model, dummy_x, current_batch in zip(models, dummy_xs, batches))
            attack_loss = leak_distance(aggregate, observed, mode=match_mode) + prior
            if not torch.isfinite(attack_loss):
                print(f"[Privacy][HE aggregate] non-finite loss at round={round_idx + 1}", flush=True)
                break
            previous_x, previous_y = dummy_xs[0].detach().clone(), dummy_ys[0].detach().clone()
            attack_loss.backward()
            optimizer.step()
            if not all(torch.isfinite(value).all() for value in [*dummy_xs, *dummy_ys]):
                print(f"[Privacy][HE aggregate] non-finite dummy at round={round_idx + 1}", flush=True)
                break
            current_loss = float(attack_loss.detach().cpu())
            history.append(current_loss)
            terminal_x, terminal_y, terminal_loss = dummy_xs[0].detach().clone(), dummy_ys[0].detach().clone(), current_loss
            if _is_improved(current_loss, best_loss, min_delta):
                best_loss, best_x, best_y, best_round, stale = current_loss, previous_x, previous_y, round_idx + 1, 0
            else:
                stale += 1
            if _should_log_round(round_idx, int(args.attack_iters), log_interval):
                print(_format_metric_log(
                    attack_name=self.name, round_idx=round_idx, current_loss=current_loss, best_loss=best_loss,
                    stale_rounds=stale, dummy_x=previous_x, real_x=batch.real_x, batch=batch, args=args,
                ), flush=True)
            if patience > 0 and stale >= patience:
                print(f"[Privacy][HE aggregate] early stopping at round={round_idx + 1}", flush=True)
                break

        terminal_mode = bool(getattr(args, "attack_return_terminal_iterate", False))
        reported_x, reported_y = (terminal_x, terminal_y) if terminal_mode else (best_x, best_y)
        reported_loss = terminal_loss if terminal_mode else best_loss
        return AttackResult(
            reconstructed_x=reported_x, reconstructed_y=reported_y, final_loss=reported_loss, history=history,
            attack_name=self.name,
            metadata={
                "protection": "he", "attack_surface": "four_hidden_client_weighted_decrypted_aggregate",
                "threat_model": metadata.get("threat_model", "HE-SA server-only decrypted aggregate attack; no colluding client"),
                "he_server_aggregate_trace": trace_path, "he_trace_metadata": metadata,
                "hidden_clients": hidden, "colluding_clients": [], "hidden_weights": weights,
                "labels_known": False, "optimized_dummy_y": True, "aggregate_leak_type": leak_type, "grad_match": match_mode,
                "best_attack_loss": best_loss, "selected_attack_round": best_round,
                "reporting_iterate": "terminal_early_stop" if terminal_mode else "minimum_matching_loss",
                "terminal_attack_loss": terminal_loss, "actual_fit_rounds": len(history),
                "early_stopped": patience > 0 and stale >= patience,
            },
        )


class ServerRealisticNoSignalAttack:
    name = "server_realistic"

    def run(self, adapter: Any, model: torch.nn.Module, batch: Any, args: Any) -> AttackResult:
        real_x = batch.real_x.detach()
        # A realistic STAGCN-EC server has no differentiable, sample-dependent
        # tensor to match. Return the normalized training-mean prior instead of
        # pretending there is an optimizable reconstruction signal.
        dummy_x = torch.zeros_like(real_x)

        metadata = {
            "optimized_dummy_y": False,
            "attack_optimizer": "none",
            "traffic_prior": False,
            "actual_fit_rounds": 0,
            "early_stopped": False,
            "best_attack_loss": 0.0,
            "server_realistic_no_signal": True,
            "server_visible_surface": getattr(
                adapter,
                "server_visible_surface",
                "No per-sample differentiable tensor is visible to the server.",
            ),
        }
        return AttackResult(
            reconstructed_x=dummy_x.detach(),
            reconstructed_y=None,
            final_loss=0.0,
            history=[],
            attack_name=self.name,
            metadata=metadata,
        )


class ActivationMatchingAttack:
    name = "activation"

    def run(self, adapter: Any, model: torch.nn.Module, batch: Any, args: Any) -> AttackResult:
        model.eval()
        real_x = batch.real_x.detach()
        real_y = batch.real_y.detach()

        # Most activation surfaces are pure forwards. FedOSTC additionally
        # exposes the client-to-server split gradient g_spatio, so its observed
        # activation trace needs a first-order autograd pass even for real_x.
        observed_context = (
            contextlib.nullcontext()
            if getattr(adapter, "activation_requires_grad", False)
            else torch.no_grad()
        )
        with observed_context:
            real_leak = adapter.compute_leak(
                model=model,
                x=real_x,
                y=real_y,
                batch=batch,
                create_graph=False,
                leak_type=self.name,
            )
        real_leak = detach_tree(real_leak)
        he_insider_trace = str(
            getattr(args, "he_insider_trace", "")
            or getattr(args, "he_ttp_insider_trace", "")
            or ""
        ).strip()
        if he_insider_trace:
            real_leak, trace_metadata = _load_he_insider_trace(he_insider_trace, model, real_leak)
            protection_metadata = {
                "protection": "he",
                "he_observation_source": "explicit_ttp_arbiter_insider_trace",
                "he_insider_trace": he_insider_trace,
                "he_trace_metadata": trace_metadata,
                "threat_model": trace_metadata.get("threat_model", "HE-TTP decrypting Arbiter insider upper bound"),
            }
            print(f"[Privacy][HE] replaying privileged activation payload from trace={he_insider_trace}", flush=True)
        elif bool(getattr(args, "quantized_prediction_sidechannel", False)):
            protection_metadata = {
                "protection": "he",
                "he_observation_source": "revised_protocol_quantized_prediction_sidechannel",
                "threat_model": (
                    "Revised-protocol ablation: the server holds the public model and receives a "
                    "fixed-point quantized client prediction in addition to the HE-protected native upload."
                ),
                "quant_prediction_bits": int(getattr(args, "quant_prediction_bits", 4)),
                "quant_prediction_clip": float(getattr(args, "quant_prediction_clip", 3.0)),
                "original_he_ciphertext_only": False,
            }
            print("[Privacy][HE] using revised-protocol quantized-prediction side channel (not ciphertext-only).", flush=True)
        else:
            real_leak, protection_metadata = protect_observed_leak(real_leak, args, adapter)

        dummy_x, resume_checkpoint = _maybe_resume_dummy_x(real_x=real_x, init=args.init, args=args)
        optimizer = _build_attack_optimizer([dummy_x], args)
        resume_optimizer_state = bool(int(getattr(args, "resume_optimizer_state", 0)))
        if (
            resume_optimizer_state
            and resume_checkpoint is not None
            and resume_checkpoint.get("optimizer_state") is not None
        ):
            try:
                optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
            except ValueError as exc:
                print(f"[Privacy] Skip optimizer resume: {exc}", flush=True)
        elif resume_checkpoint is not None:
            print(
                f"[Privacy] Using checkpoint dummy tensors with fresh Adam optimizer lr={args.attack_lr}",
                flush=True,
            )
        history: List[float] = []
        start_round = int(resume_checkpoint.get("round", 0)) if resume_checkpoint is not None else 0
        best_loss = (
            float(resume_checkpoint.get("best_attack_loss", float("inf")))
            if resume_checkpoint is not None
            else float("inf")
        )
        final_dummy_x = dummy_x.detach().clone()
        # Select by the leak-matching objective available to the attacker,
        # never by inaccessible reconstruction metrics against real_x.
        best_dummy_x = final_dummy_x.detach().clone()
        best_round = start_round
        stale_rounds = int(resume_checkpoint.get("stale_rounds", 0)) if resume_checkpoint is not None else 0
        patience = max(0, int(getattr(args, "early_stop_patience", 0)))
        min_delta = float(getattr(args, "early_stop_min_delta", 0.0))
        nonfinite_stopped = False

        log_interval = int(getattr(args, "log_interval", 0))
        checkpoint_interval = int(getattr(args, "attack_checkpoint_interval", 0))

        if start_round >= args.attack_iters:
            print(
                f"[Privacy] Resume round {start_round} is already >= attack_iters {args.attack_iters}; "
                "no extra optimization will run.",
                flush=True,
            )

        for round_idx in range(start_round, args.attack_iters):
            optimizer.zero_grad(set_to_none=True)
            dummy_leak = adapter.compute_leak(
                model=model,
                x=dummy_x,
                y=real_y,
                batch=batch,
                create_graph=True,
                leak_type=self.name,
            )
            match_mode = getattr(args, "grad_match", "mse")
            leak_loss = leak_distance(dummy_leak, real_leak, mode=match_mode)
            prior_loss = model_specific_prior_loss(model, dummy_x, batch, args)
            attack_loss = leak_loss + prior_loss
            if not torch.isfinite(attack_loss):
                nonfinite_stopped = True
                print(
                    f"[Privacy][{self.name}] stopping at round={round_idx + 1}: non-finite attack loss.",
                    flush=True,
                )
                break
            prev_dummy_x = dummy_x.detach().clone()
            attack_loss.backward()
            optimizer.step()
            if not torch.isfinite(dummy_x).all():
                nonfinite_stopped = True
                final_dummy_x = prev_dummy_x
                print(
                    f"[Privacy][{self.name}] stopping at round={round_idx + 1}: dummy tensor became non-finite.",
                    flush=True,
                )
                break
            current_loss = float(attack_loss.detach().cpu())
            history.append(current_loss)
            final_dummy_x = dummy_x.detach().clone()

            if _is_improved(current_loss, best_loss, min_delta):
                best_loss = current_loss
                stale_rounds = 0
                # attack_loss was evaluated on the pre-step dummy tensor.
                best_dummy_x = prev_dummy_x
                best_round = round_idx + 1
            else:
                stale_rounds += 1

            if _should_log_round(round_idx, args.attack_iters, log_interval):
                print(
                    _format_metric_log(
                        attack_name=self.name,
                        round_idx=round_idx,
                        current_loss=current_loss,
                        best_loss=best_loss,
                        stale_rounds=stale_rounds,
                        dummy_x=dummy_x,
                        real_x=real_x,
                        batch=batch,
                        args=args,
                    ),
                    flush=True,
                )

            if _should_checkpoint_round(round_idx, checkpoint_interval):
                checkpoint_path = _save_attack_checkpoint(
                    attack_name=self.name,
                    round_idx=round_idx,
                    current_loss=current_loss,
                    best_loss=best_loss,
                    stale_rounds=stale_rounds,
                    dummy_x=dummy_x,
                    dummy_y=None,
                    optimizer=optimizer,
                    real_x=real_x,
                    batch=batch,
                    args=args,
                )
                print(f"[Privacy][{self.name}] saved checkpoint to {checkpoint_path}", flush=True)

            if patience > 0 and stale_rounds >= patience:
                break

        return AttackResult(
            reconstructed_x=best_dummy_x,
            reconstructed_y=None,
            final_loss=best_loss if best_loss < float("inf") else float("nan"),
            history=history,
            attack_name=self.name,
            metadata={
                "optimized_dummy_y": False,
                "attack_optimizer": getattr(args, "attack_optimizer", "adam"),
                "attack_surface": getattr(adapter, "attack_surface", self.name),
                "model_state_source": "checkpoint" if getattr(args, "checkpoint", "") else "random_init",
                "leak_match_mode": getattr(args, "grad_match", "mse"),
                "fedstn_activation_surface": getattr(args, "fedstn_activation_surface", None),
                "fedstn_batch_aggregate_only": bool(getattr(adapter, "batch_aggregate_only", False)),
                "fedstn_hs_weight": getattr(args, "fedstn_hs_weight", None),
                "fedstn_context_weight": getattr(args, "fedstn_context_weight", None),
                "fedstn_agg_weight": getattr(args, "fedstn_agg_weight", None),
                "fedstn_rlcn_weight": getattr(args, "fedstn_rlcn_weight", None),
                "fedstn_scn_weight": getattr(args, "fedstn_scn_weight", None),
                "fedmetro_activation_scope": getattr(args, "fedmetro_activation_scope", None),
                "fedmetro_attack_surface": getattr(args, "fedmetro_attack_surface", None),
                "fedmetro_agg_weight": getattr(args, "fedmetro_agg_weight", None),
                "fedmetro_gagg_weight": getattr(args, "fedmetro_gagg_weight", None),
                "fedmetro_update_weight": getattr(args, "fedmetro_update_weight", None),
                "fuels_activation_scope": getattr(args, "fuels_activation_scope", None),
                "fuels_dr": getattr(args, "fuels_dr", None),
                "fuels_threat_model": (
                    "single_sample_representation_upper_bound"
                    if bool(getattr(adapter, "single_sample_representation_upper_bound", False))
                    else "protocol_surface"
                ),
                "traffic_prior": bool(int(getattr(args, "traffic_prior", 0))),
                "traffic_range_weight": getattr(args, "traffic_range_weight", 0.0),
                "traffic_nonnegative_weight": getattr(args, "traffic_nonnegative_weight", 0.0),
                "traffic_temporal_weight": getattr(args, "traffic_temporal_weight", 0.0),
                "traffic_spatial_weight": getattr(args, "traffic_spatial_weight", 0.0),
                "traffic_std_weight": getattr(args, "traffic_std_weight", 0.0),
                "traffic_std_ratio": getattr(args, "traffic_std_ratio", 1.0),
                "resume_optimizer_state": resume_optimizer_state,
                "best_attack_loss": best_loss,
                "selected_attack_round": best_round,
                "last_attack_loss": history[-1] if history else float("nan"),
                "resume_start_round": start_round,
                "new_fit_rounds": len(history),
                "actual_fit_rounds": start_round + len(history),
                "early_stopped": patience > 0 and stale_rounds >= patience,
                "nonfinite_stopped": nonfinite_stopped,
                **protection_metadata,
            },
        )


ATTACKS = {
    "gradient": GradientMatchingAttack,
    "model_update": ModelUpdateMatchingAttack,
    "he_kminus2_model_update": HEKMinus2CollusionModelUpdateAttack,
    "he_kminus3_model_update": HEOneClientCollusionModelUpdateAttack,
    "he_server_aggregate_model_update": HEServerOnlyAggregateModelUpdateAttack,
    "server_realistic": ServerRealisticNoSignalAttack,
    "activation": ActivationMatchingAttack,
}


def get_attack(name: str):
    if name not in ATTACKS:
        raise KeyError(f"Unknown attack {name!r}. Available: {sorted(ATTACKS)}")
    return ATTACKS[name]()
