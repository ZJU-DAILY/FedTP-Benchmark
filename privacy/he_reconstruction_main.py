"""External-server HE reconstruction evaluation.

This runner intentionally never decrypts a client HE-TTP upload and never
substitutes a Plain attack result.  With no colluding clients, an HE-SA
aggregate is not identifiable as one target client's payload; with HE-TTP,
the ordinary server has ciphertext only.  The resulting normalized-zero prior
is a reproducible no-semantic-signal baseline, evaluated against one held-out
client sample solely by the experiment evaluator.

The separate ``*_insider.pt`` trace created only with
``--privacy_trace_save_insider`` is reserved for a future, explicitly labelled
trusted-Arbiter insider upper-bound runner.  It is deliberately rejected here.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from privacy.privacy_attacks import AttackResult
from privacy.privacy_config import build_arg_parser
from privacy.privacy_data import load_privacy_batch
from privacy.privacy_main import save_outputs
from privacy.privacy_metrics import reconstruction_metrics


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_trace(path_text: str) -> dict[str, Any]:
    path = Path(path_text).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"HE trace is missing: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError(f"HE JSON trace must contain an object: {path}")
        return data
    if path.suffix.lower() == ".pt":
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch < 2.6
            data = torch.load(path, map_location="cpu")
        if not isinstance(data, dict) or not isinstance(data.get("metadata"), dict):
            raise TypeError(f"HE aggregate trace has an invalid schema: {path}")
        return dict(data["metadata"])
    raise ValueError(f"Unsupported HE trace extension {path.suffix!r}; expected .json or .pt")


def _validate_trace(args: Any, metadata: dict[str, Any]) -> None:
    expected_backend = str(args.he_backend).lower()
    observed_backend = str(metadata.get("he_backend", "")).lower()
    if observed_backend != expected_backend:
        raise ValueError(
            f"HE trace backend mismatch: command requests {expected_backend!r}, trace records {observed_backend!r}."
        )
    for key in ("model", "dataset_name", "feature_type"):
        expected = str(getattr(args, key))
        observed = str(metadata.get(key, ""))
        if observed != expected:
            raise ValueError(f"HE trace {key} mismatch: command={expected!r}, trace={observed!r}.")
    if expected_backend == "he_ttp" and metadata.get("observer") != "external_server":
        raise ValueError("The external HE-TTP runner accepts only ciphertext-visible metadata traces.")
    if expected_backend == "he_sa" and metadata.get("observer") != "external_server_aggregate_only":
        raise ValueError("The HE-SA runner requires an aggregate-only trace exported by the Arbiter.")


def main() -> None:
    args = build_arg_parser().parse_args()
    if str(args.protection).lower() != "he":
        raise ValueError("he_reconstruction_main.py requires --protection he.")
    if int(args.batch_size) != 1:
        raise ValueError("HE reconstruction is fixed to --batch_size 1.")
    if not str(args.he_trace_path).strip():
        raise ValueError("Pass --he_trace_path exported by the corresponding one-round HE trace run.")
    expected_scenario = (
        "aggregate_only" if str(args.he_backend).lower() == "he_sa" else "ciphertext_only_prior"
    )
    if str(args.he_attack_scenario).lower() != expected_scenario:
        raise ValueError(
            f"HE backend {args.he_backend!r} requires --he_attack_scenario {expected_scenario!r} "
            "for this external-server runner."
        )
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    _set_seed(int(args.seed))

    metadata = _load_trace(args.he_trace_path)
    _validate_trace(args, metadata)
    batch = load_privacy_batch(args)
    # The evaluator has real_x only to compute the final privacy metric.  The
    # external attacker does not receive it, nor any plaintext tensor from the
    # trace.  Zero is the fixed prior in the normalized representation.
    reconstructed_x = torch.zeros_like(batch.real_x)
    attack_name = (
        "aggregate_only_unidentifiable_prior"
        if str(args.he_backend).lower() == "he_sa"
        else "ciphertext_only_prior"
    )
    result = AttackResult(
        reconstructed_x=reconstructed_x,
        reconstructed_y=None,
        final_loss=0.0,
        history=[],
        attack_name=attack_name,
        metadata={
            "attack_surface": metadata.get("payload_type", "unknown"),
            "model_state_source": "real_he_training_trace",
            "actual_fit_rounds": 0,
            "best_attack_loss": 0.0,
            "early_stopped": False,
            "external_observer": True,
            "observed_trace": str(Path(args.he_trace_path).expanduser()),
            "trace_tag": metadata.get("tag"),
            "trace_phase": metadata.get("phase"),
            "observed_signal_used_for_target_reconstruction": False,
            "prior": "normalized_zero",
            "security_interpretation": (
                "HE-SA aggregate is not identifiable as one client without collusion"
                if str(args.he_backend).lower() == "he_sa"
                else "semantic-security ciphertext has no plaintext reconstruction signal"
            ),
        },
    )
    metrics = reconstruction_metrics(
        reconstructed_x=reconstructed_x,
        real_x=batch.real_x,
        scaler=batch.scaler,
        mape_eps=args.mape_eps,
    )
    # A ciphertext-only observer has no sample-dependent plaintext signal.
    # The constant normalized-zero prior has a computational PCC of zero, but
    # reporting that number as an attack outcome would incorrectly imply that
    # an inversion was attempted or succeeded.  Preserve it for diagnostics
    # while marking the privacy metric as not applicable.
    metrics["ZERO_PRIOR_PCC"] = metrics["PCC"]
    metrics["PCC"] = None
    metrics["PCC_STATUS"] = "not_applicable_no_sample_dependent_observation"
    output_path = save_outputs(args, result, metrics)
    print(
        f"[HE Reconstruction] scenario={args.he_attack_scenario} model={args.model} "
        f"client={args.client_rank} sample={args.sample_index} batch_size=1",
        flush=True,
    )
    print("  PCC: N/A (ciphertext/prior-only; no sample-dependent observation)", flush=True)
    for key in ("MAE", "RMSE", "MAPE"):
        print(f"  {key}: {metrics[key]:.6f}", flush=True)
    print(f"[HE Reconstruction] saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()
