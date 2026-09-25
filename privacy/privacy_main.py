from __future__ import annotations

import csv
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from privacy.privacy_attacks import get_attack
from privacy.privacy_config import build_arg_parser
from privacy.privacy_data import load_privacy_batch
from privacy.privacy_metrics import reconstruction_metrics
from privacy.privacy_registry import get_adapter


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Reconstruction comparisons must not change merely because CuDNN chose
        # a different convolution implementation on the same fixed trace.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[Privacy] Requested {device}, but CUDA is unavailable. Falling back to cpu.")
        return "cpu"
    return device


def _load_compatible_state(model: torch.nn.Module, state: Mapping[str, Any], device: str) -> int:
    compatible = {
        key: value.to(device) for key, value in state.items()
        if key in model.state_dict() and torch.is_tensor(value) and tuple(value.shape) == tuple(model.state_dict()[key].shape)
    }
    model.load_state_dict(compatible, strict=False)
    return len(compatible)


def _jsonable_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, np.integer)):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def _add_slice_metrics(
    metrics: Dict[str, float],
    *,
    prefix: str,
    reconstructed_x: torch.Tensor,
    real_x: torch.Tensor,
    scaler: Any,
    mape_eps: float,
) -> None:
    slice_metrics = reconstruction_metrics(
        reconstructed_x=reconstructed_x,
        real_x=real_x,
        scaler=scaler,
        mape_eps=mape_eps,
    )
    metrics.update({f"{prefix}_{key}": value for key, value in slice_metrics.items()})


def save_outputs(args: Any, attack_result: Any, metrics: Dict[str, float]) -> Path:
    result_dir = Path(args.result_dir)
    if not bool(getattr(args, "result_dir_flat", False)):
        result_dir = result_dir / args.model
    result_dir.mkdir(parents=True, exist_ok=True)

    stem = (
        f"{args.model}_{args.dataset_name}_{args.feature_type}_"
        f"client{args.client_rank}_{args.split}{args.sample_index}_"
        f"bs{getattr(args, 'batch_size', 1)}_{attack_result.attack_name}"
    )
    result_tag = str(getattr(args, "result_tag", "")).strip()
    if result_tag:
        stem = f"{stem}_{result_tag}"
    json_path = result_dir / f"{stem}.json"
    csv_path = result_dir / "privacy_results.csv"

    payload = {
        "model": args.model,
        "dataset_name": args.dataset_name,
        "feature_type": args.feature_type,
        "client_rank": args.client_rank,
        "split": args.split,
        "sample_index": args.sample_index,
        "batch_size": getattr(args, "batch_size", 1),
        "attack_node_index": args.attack_node_index,
        "attack": attack_result.attack_name,
        "attack_surface": attack_result.metadata.get("attack_surface", attack_result.attack_name),
        "model_state_source": attack_result.metadata.get("model_state_source", "unknown"),
        "attack_iters": args.attack_iters,
        "attack_lr": args.attack_lr,
        "protection": getattr(args, "protection", "plain"),
        "he_backend": getattr(args, "he_backend", None),
        "he_attack_scenario": getattr(args, "he_attack_scenario", None),
        "dp_sigma": getattr(args, "dp_sigma", None),
        "dp_clip_norm": getattr(args, "dp_clip_norm", None),
        "result_tag": result_tag,
        "attack_restarts": getattr(args, "attack_restarts", 1),
        "final_attack_loss": attack_result.final_loss,
        "metadata": attack_result.metadata,
        "metrics": _jsonable_metrics(metrics),
        "history": attack_result.history,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    row = {
        "Model": args.model,
        "Dataset": args.dataset_name,
        "Feature": args.feature_type,
        "Client": args.client_rank,
        "Split": args.split,
        "SampleIndex": args.sample_index,
        "BatchSize": getattr(args, "batch_size", 1),
        "AttackNodeIndex": args.attack_node_index,
        "Attack": attack_result.attack_name,
        "Protection": getattr(args, "protection", "plain"),
        "HEBackend": getattr(args, "he_backend", None),
        "HEAttackScenario": getattr(args, "he_attack_scenario", None),
        "DPSigma": getattr(args, "dp_sigma", None),
        "DPClipNorm": getattr(args, "dp_clip_norm", None),
        "ResultTag": result_tag,
        "AttackRestarts": getattr(args, "attack_restarts", 1),
        "SelectedRestart": attack_result.metadata.get("selected_restart", 0),
        "FinalAttackLoss": attack_result.final_loss,
        **_jsonable_metrics(metrics),
    }
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    if args.save_reconstruction:
        torch.save(
            {
                "reconstructed_x": attack_result.reconstructed_x.cpu(),
                "reconstructed_y": None
                if attack_result.reconstructed_y is None
                else attack_result.reconstructed_y.cpu(),
            },
            result_dir / f"{stem}.pt",
        )

    return json_path


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    set_seed(args.seed)

    adapter = get_adapter(args.model)
    attack_name = adapter.default_attack if args.attack == "auto" else args.attack
    print(
        f"[Privacy Run] resolved_attack={attack_name} "
        f"(requested={args.attack}, adapter_default={adapter.default_attack})",
        flush=True,
    )

    print(
        f"[Privacy Run] loading privacy batch for model={args.model}, "
        f"dataset={args.dataset_name}, split={args.split}, client={args.client_rank}",
        flush=True,
    )
    batch = load_privacy_batch(args)
    print(
        f"[Privacy Run] privacy batch ready | "
        f"real_x_shape={tuple(batch.real_x.shape)} real_y_shape={tuple(batch.real_y.shape)}",
        flush=True,
    )
    setattr(batch, "args", args)

    print(f"[Privacy Run] building model adapter={adapter.name}", flush=True)
    model = adapter.build_model(args, batch)
    he_insider_trace = str(getattr(args, "he_insider_trace", "") or getattr(args, "he_ttp_insider_trace", "") or "").strip()
    he_collusion_trace = str(getattr(args, "he_collusion_trace", "") or "").strip()
    he_kminus2_trace = str(getattr(args, "he_kminus2_trace", "") or "").strip()
    he_kminus3_trace = str(getattr(args, "he_kminus3_trace", "") or "").strip()
    he_server_aggregate_trace = str(getattr(args, "he_server_aggregate_trace", "") or "").strip()
    if sum(bool(item) for item in (he_insider_trace, he_collusion_trace, he_kminus2_trace, he_kminus3_trace, he_server_aggregate_trace)) > 1:
        raise ValueError("Pass at most one HE replay trace option.")
    he_replay_trace = he_insider_trace or he_collusion_trace
    if he_replay_trace:
        trace = torch.load(Path(he_replay_trace).expanduser(), map_location=args.device, weights_only=False)
        state_dict = trace.get("model_state_dict") if isinstance(trace, dict) else None
        if not isinstance(state_dict, dict):
            raise ValueError("HE replay trace has no model_state_dict.")
        compatible_count = _load_compatible_state(model, state_dict, args.device)
        if (
            str(args.model).upper() == "FGNNEH"
            and str(getattr(args, "fgnneh_activation_scope", "")) == "quantized_prediction"
        ):
            observed = trace.get("observed_leak") if isinstance(trace, Mapping) else None
            if (
                not isinstance(observed, tuple)
                or len(observed) != 2
                or not torch.is_tensor(observed[0])
                or not torch.is_tensor(observed[1])
            ):
                raise ValueError("FGNNEH quantized-prediction trace must contain (quantized_prediction, server_context).")
            expected_nodes = int(batch.real_x.shape[1])
            traced_nodes = int(observed[0].shape[1]) if observed[0].dim() >= 2 else -1
            if traced_nodes != expected_nodes:
                raise ValueError(
                    "FGNNEH replay partition mismatch: the trace contains "
                    f"{traced_nodes} client nodes but the privacy batch contains {expected_nodes}. "
                    "Use the same partition/<strategy>/PeMS04_flow_4clients_seed42.json "
                    "artifact for FATE training and privacy replay."
                )
            setattr(batch, "fgnneh_server_context", observed[1].to(args.device))
        replay_label = "insider" if he_insider_trace else "K-1 collusion"
        print(f"[Privacy][HE] loaded {compatible_count} round-start tensors from {replay_label} trace.", flush=True)
    if he_kminus2_trace:
        trace = torch.load(Path(he_kminus2_trace).expanduser(), map_location=args.device, weights_only=False)
        if not isinstance(trace, Mapping) or not isinstance(trace.get("metadata"), Mapping):
            raise ValueError("HE K-2 trace has an invalid schema.")
        metadata = dict(trace["metadata"])
        if str(metadata.get("observer", "")) != "server_k_minus_2_client_collusion":
            raise ValueError("--he_kminus2_trace must be labelled as a server K-2 collusion trace.")
        target_rank = int(metadata.get("target_client", -1))
        peer_rank = int(metadata.get("peer_hidden_client", -1))
        if target_rank != int(args.client_rank) or peer_rank < 0 or peer_rank == target_rank:
            raise ValueError("HE K-2 trace target/peer ranks do not match the requested target client.")
        target_state = trace.get("target_model_state_dict")
        peer_state = trace.get("peer_model_state_dict")
        if not isinstance(target_state, Mapping) or not isinstance(peer_state, Mapping):
            raise ValueError("HE K-2 trace has no public round-start states for both hidden clients.")
        target_count = _load_compatible_state(model, target_state, args.device)
        peer_args = copy.deepcopy(args)
        peer_args.client_rank = peer_rank
        peer_batch = load_privacy_batch(peer_args)
        setattr(peer_batch, "args", peer_args)
        peer_model = adapter.build_model(peer_args, peer_batch)
        peer_count = _load_compatible_state(peer_model, peer_state, args.device)
        args._he_kminus2_peer_batch = peer_batch
        args._he_kminus2_peer_model = peer_model
        print(
            f"[Privacy][HE] K-2 trace loaded: target={target_rank} ({target_count} state tensors), "
            f"peer={peer_rank} ({peer_count} state tensors); labels remain unknown.",
            flush=True,
        )
    if he_kminus3_trace:
        trace = torch.load(Path(he_kminus3_trace).expanduser(), map_location=args.device, weights_only=False)
        if not isinstance(trace, Mapping) or not isinstance(trace.get("metadata"), Mapping):
            raise ValueError("HE K-3 trace has an invalid schema.")
        metadata = dict(trace["metadata"])
        hidden_clients = [int(value) for value in metadata.get("hidden_clients", [])]
        states = trace.get("hidden_model_state_dicts")
        if str(metadata.get("observer", "")) != "server_plus_one_client_collusion":
            raise ValueError("--he_kminus3_trace must be labelled as a one-client-collusion trace.")
        if len(hidden_clients) != 3 or hidden_clients[0] != int(args.client_rank) or len(set(hidden_clients)) != 3:
            raise ValueError("HE K-3 trace hidden-client ordering must start with the requested target client.")
        if not isinstance(states, list) or len(states) != 3 or not all(isinstance(state, Mapping) for state in states):
            raise ValueError("HE K-3 trace has no three-client public round-start states.")
        target_count = _load_compatible_state(model, states[0], args.device)
        peer_models, peer_batches = [], []
        peer_counts = []
        for peer_rank, state in zip(hidden_clients[1:], states[1:]):
            peer_args = copy.deepcopy(args)
            peer_args.client_rank = peer_rank
            peer_batch = load_privacy_batch(peer_args)
            setattr(peer_batch, "args", peer_args)
            peer_model = adapter.build_model(peer_args, peer_batch)
            peer_counts.append(_load_compatible_state(peer_model, state, args.device))
            peer_models.append(peer_model)
            peer_batches.append(peer_batch)
        args._he_kminus3_peer_models = peer_models
        args._he_kminus3_peer_batches = peer_batches
        print(
            f"[Privacy][HE] K-3 trace loaded: target={hidden_clients[0]} ({target_count} state tensors), "
            f"peers={hidden_clients[1:]} ({peer_counts}); labels remain unknown.",
            flush=True,
        )
    if he_server_aggregate_trace:
        trace = torch.load(Path(he_server_aggregate_trace).expanduser(), map_location=args.device, weights_only=False)
        if not isinstance(trace, Mapping) or not isinstance(trace.get("metadata"), Mapping):
            raise ValueError("HE server-only aggregate trace has an invalid schema.")
        metadata = dict(trace["metadata"])
        hidden_clients = [int(value) for value in metadata.get("hidden_clients", [])]
        public_state = trace.get("public_model_state_dict")
        if str(metadata.get("observer", "")) != "server_only_decrypted_aggregate":
            raise ValueError("--he_server_aggregate_trace must be labelled as a server-only aggregate trace.")
        if len(hidden_clients) != int(args.num_clients) or hidden_clients[0] != int(args.client_rank) or len(set(hidden_clients)) != len(hidden_clients):
            raise ValueError("Server-only aggregate trace must contain every client and begin with the requested target client.")
        if not isinstance(public_state, Mapping):
            raise ValueError(
                "HE server-only aggregate trace has no single public round-start state. "
                "Rebuild the trace with the current builder; legacy traces containing per-client states are not valid server-only inputs."
            )
        target_count = _load_compatible_state(model, public_state, args.device)
        peer_models, peer_batches, peer_counts = [], [], []
        for peer_rank in hidden_clients[1:]:
            peer_args = copy.deepcopy(args)
            peer_args.client_rank = peer_rank
            peer_batch = load_privacy_batch(peer_args)
            setattr(peer_batch, "args", peer_args)
            peer_model = adapter.build_model(peer_args, peer_batch)
            peer_counts.append(_load_compatible_state(peer_model, public_state, args.device))
            peer_models.append(peer_model)
            peer_batches.append(peer_batch)
        args._he_server_aggregate_peer_models = peer_models
        args._he_server_aggregate_peer_batches = peer_batches
        print(
            f"[Privacy][HE] server-only aggregate trace loaded: one common public state "
            f"({target_count} tensors) for target={hidden_clients[0]} and hidden peers={hidden_clients[1:]} "
            f"({peer_counts}); labels, individual uploads, and client-private states remain unknown.",
            flush=True,
        )
    print(f"[Privacy Run] model built successfully", flush=True)
    attack = get_attack(attack_name)
    restart_count = max(1, int(getattr(args, "attack_restarts", 1)))
    restart_base_seed = int(getattr(args, "attack_restart_seed", 10000))
    print(
        f"[Privacy Run] starting attack.run(...) | restarts={restart_count} "
        "| selection=min attacker-visible matching loss",
        flush=True,
    )
    restart_results = []
    for restart_index in range(restart_count):
        # The model and the DP observation remain fixed.  Only dummy_x's
        # random initialization changes, so this is a valid attacker restart
        # rather than a search over private reconstruction metrics.
        single_attack_seed = int(getattr(args, "attack_initialization_seed", -1))
        if restart_count > 1:
            active_attack_seed = restart_base_seed + restart_index
            set_seed(active_attack_seed)
        elif single_attack_seed >= 0:
            active_attack_seed = single_attack_seed
            set_seed(active_attack_seed)
        else:
            active_attack_seed = None
        candidate = attack.run(adapter, model, batch, args)
        candidate_loss = float(candidate.final_loss)
        restart_results.append((candidate_loss, restart_index, candidate))
        print(
            f"[Privacy] restart={restart_index + 1}/{restart_count} "
            f"seed={active_attack_seed if active_attack_seed is not None else '<unchanged>'} "
            f"attacker_loss={candidate_loss:.6f}",
            flush=True,
        )
    finite_results = [item for item in restart_results if np.isfinite(item[0])]
    if not finite_results:
        raise RuntimeError("All reconstruction attack restarts produced non-finite losses.")
    _, selected_restart, attack_result = min(finite_results, key=lambda item: item[0])
    attack_result.metadata.update(
        {
            "attack_restarts": restart_count,
            "attack_restart_base_seed": restart_base_seed if restart_count > 1 else None,
            "selected_restart": selected_restart,
            "restart_attacker_losses": [float(item[0]) for item in restart_results],
            "restart_selection_rule": "minimum_attacker_visible_matching_loss",
        }
    )
    print(
        f"[Privacy] selected restart={selected_restart + 1}/{restart_count} by minimum matching loss; "
        "PCC was not used for selection.",
        flush=True,
    )

    metric_reconstructed_x = attack_result.reconstructed_x
    metric_real_x = batch.real_x
    is_batch_aggregate_observation = bool(
        attack_result.metadata.get("fedstn_batch_aggregate_only", False)
        or attack_result.metadata.get("batch_aggregate_leak", False)
    )
    if is_batch_aggregate_observation and metric_real_x.shape[0] > 1:
        # One server observation mixes several records. Assess a fixed victim
        # while retaining the remaining dummy records as unknown nuisance
        # inputs in the attack.
        metric_reconstructed_x = metric_reconstructed_x[:1]
        metric_real_x = metric_real_x[:1]
        attack_result.metadata["metric_scope"] = "target_record_0"
        attack_result.metadata["target_record_index"] = 0
    else:
        attack_result.metadata["metric_scope"] = "all_reconstructed_records"

    metrics = reconstruction_metrics(
        reconstructed_x=metric_reconstructed_x,
        real_x=metric_real_x,
        scaler=batch.scaler,
        mape_eps=args.mape_eps,
    )
    print(
        f"[Privacy] Metric scope: {attack_result.metadata['metric_scope']} "
        f"| evaluated_shape={tuple(metric_real_x.shape)}",
        flush=True,
    )
    if args.model.upper() == "REFOL" and batch.real_x.dim() >= 3:
        _add_slice_metrics(
            metrics,
            prefix="REFOL_LAST_STEP",
            reconstructed_x=attack_result.reconstructed_x[:, :, -1:, :],
            real_x=batch.real_x[:, :, -1:, :],
            scaler=batch.scaler,
            mape_eps=args.mape_eps,
        )
    if args.model.upper() == "FEDSTN" and batch.real_x.dim() >= 4:
        _add_slice_metrics(
            metrics,
            prefix="FEDSTN_LAST_STEP",
            reconstructed_x=metric_reconstructed_x[:, :, -1:, :],
            real_x=metric_real_x[:, :, -1:, :],
            scaler=batch.scaler,
            mape_eps=args.mape_eps,
        )

    output_path = save_outputs(args, attack_result, metrics)
    print("[Privacy] Reconstruction metrics:")
    for key in ("PCC", "MAE", "MSE", "RMSE", "MAPE", "WMAPE", "MAPE_VALID_RATIO", "MAPE_EPS"):
        print(f"  {key}: {metrics[key]:.6f}")
    if args.model.upper() == "REFOL":
        print("[Privacy] REFOL visible last-step metrics:")
        for key in (
            "REFOL_LAST_STEP_PCC",
            "REFOL_LAST_STEP_MAE",
            "REFOL_LAST_STEP_MSE",
            "REFOL_LAST_STEP_RMSE",
            "REFOL_LAST_STEP_MAPE",
            "REFOL_LAST_STEP_WMAPE",
        ):
            if key in metrics:
                print(f"  {key}: {metrics[key]:.6f}")
    if args.model.upper() == "FEDSTN":
        print("[Privacy] FedSTN last-input-step metrics:")
        for key in (
            "FEDSTN_LAST_STEP_PCC",
            "FEDSTN_LAST_STEP_MAE",
            "FEDSTN_LAST_STEP_MSE",
            "FEDSTN_LAST_STEP_RMSE",
            "FEDSTN_LAST_STEP_MAPE",
            "FEDSTN_LAST_STEP_WMAPE",
        ):
            if key in metrics:
                print(f"  {key}: {metrics[key]:.6f}")
    print(
        "[Privacy] Fit summary: "
        f"actual_rounds={attack_result.metadata.get('actual_fit_rounds')}, "
        f"best_attack_loss={attack_result.metadata.get('best_attack_loss'):.6f}, "
        f"early_stopped={attack_result.metadata.get('early_stopped')}"
    )
    print(f"[Privacy] Saved details to {output_path}")


if __name__ == "__main__":
    main()
