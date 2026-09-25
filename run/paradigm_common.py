import argparse
import copy
import importlib
import math
import os
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from torch.utils.data import DataLoader
from lib.grid_partition import grid_rectangular_split
from torch.utils.data._utils.collate import default_collate
from config_args import args
from lib.fedmssa_trainer import train_fedmssa_centralized_sim, train_fedmssa_standalone
from lib.utils import init_seed
import paradigm_results as _paradigm_results


_FALLBACK_RUN_ID = os.environ.get("BENCHMARK_RUN_ID") or os.environ.get("FCGCN_RUN_ID") or time.strftime("%Y%m%d_%H%M%S")
_PARADIGM_RESULTS_COMPAT = all(
    callable(getattr(_paradigm_results, name, None))
    for name in (
        "current_run_id",
        "display_dataset_name",
        "row_sample_metrics",
        "write_raw_result",
        "write_summary_results",
    )
)


def current_run_id():
    fn = getattr(_paradigm_results, "current_run_id", None)
    return fn() if callable(fn) else _FALLBACK_RUN_ID


def display_dataset_name(dataset_name):
    fn = getattr(_paradigm_results, "display_dataset_name", None)
    if callable(fn):
        return fn(dataset_name)
    return "PeMSD7" if dataset_name in ("PeMS07", "PeMSD7") else dataset_name


def row_sample_metrics(row):
    fn = getattr(_paradigm_results, "row_sample_metrics", None)
    if callable(fn):
        return fn(row)

    total_elements = _to_int_result(row.get("elements"))
    total_abs = _to_float_result(row.get("abs_error_sum"))
    total_sq = _to_float_result(row.get("sq_error_sum"))
    total_mape = _to_float_result(row.get("mape_error_sum"))
    total_mape_elements = _to_int_result(row.get("mape_elements"))
    if total_elements <= 0:
        return {
            "sample_mae": row.get("mae", ""),
            "sample_mse": row.get("mse", ""),
            "sample_rmse": row.get("rmse", ""),
            "sample_mape": row.get("mape", ""),
            "sample_elements": "",
        }
    sample_mse = total_sq / max(total_elements, 1)
    return {
        "sample_mae": round(total_abs / max(total_elements, 1), 4),
        "sample_mse": round(sample_mse, 4),
        "sample_rmse": round(sample_mse ** 0.5, 4),
        "sample_mape": round(total_mape / total_mape_elements, 4) if total_mape_elements else 0.0,
        "sample_elements": total_elements,
    }


def write_raw_result(paradigm, row):
    fn = getattr(_paradigm_results, "write_raw_result", None)
    if _PARADIGM_RESULTS_COMPAT and callable(fn):
        return fn(paradigm, row)

    row = dict(row)
    row.setdefault("run_id", current_run_id())
    row["dataset"] = display_dataset_name(row["dataset"])
    raw_path, _ = _fallback_output_paths(paradigm, row.get("method"))
    _fallback_write_rows(raw_path, [row])


def write_summary_results(paradigm, raw_rows):
    fn = getattr(_paradigm_results, "write_summary_results", None)
    if _PARADIGM_RESULTS_COMPAT and callable(fn):
        return fn(paradigm, raw_rows)

    grouped = {}
    for row in raw_rows:
        key = (
            row.get("run_id", current_run_id()),
            row["section"],
            row["method"],
            row["model_arg"],
            display_dataset_name(row["dataset"]),
            row["feature"],
            row["seed"],
        )
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (run_id, section, method, model_arg, dataset, feature, seed), rows in grouped.items():
        sample_metrics = _fallback_sample_level_metrics(rows)
        summary_rows.append({
            "run_id": run_id,
            "section": section,
            "method": method,
            "model_arg": model_arg,
            "dataset": dataset,
            "feature": feature,
            "client": "mean" if paradigm == "local" else "global",
            "client_mae": round(sum(float(r["mae"]) for r in rows) / max(len(rows), 1), 4),
            "client_mse": round(sum(float(r["mse"]) for r in rows) / max(len(rows), 1), 4),
            "client_rmse": round(sum(float(r["rmse"]) for r in rows) / max(len(rows), 1), 4),
            "client_mape": round(sum(float(r["mape"]) for r in rows) / max(len(rows), 1), 4),
            "mae": round(sum(float(r["mae"]) for r in rows) / max(len(rows), 1), 4),
            "mse": round(sum(float(r["mse"]) for r in rows) / max(len(rows), 1), 4),
            "rmse": round(sum(float(r["rmse"]) for r in rows) / max(len(rows), 1), 4),
            "mape": round(sum(float(r["mape"]) for r in rows) / max(len(rows), 1), 4),
            "sample_mae": sample_metrics["sample_mae"],
            "sample_mse": sample_metrics["sample_mse"],
            "sample_rmse": sample_metrics["sample_rmse"],
            "sample_mape": sample_metrics["sample_mape"],
            "sample_elements": sample_metrics["sample_elements"],
            "seed": seed,
        })

    for method in sorted({row.get("method") for row in summary_rows}):
        rows = [row for row in summary_rows if row.get("method") == method]
        _, summary_path = _fallback_output_paths(paradigm, method)
        _fallback_write_rows(summary_path, rows)
    return summary_rows


def _to_float_result(value, default=0.0):
    if value in ("", None):
        return default
    return float(value)


def _to_int_result(value, default=0):
    if value in ("", None):
        return default
    return int(float(value))


_FALLBACK_RESULT_FIELDNAMES = [
    "run_id",
    "section",
    "method",
    "model_arg",
    "dataset",
    "feature",
    "client",
    "client_mae",
    "client_mse",
    "client_rmse",
    "client_mape",
    "mae",
    "mse",
    "rmse",
    "mape",
    "elements",
    "abs_error_sum",
    "sq_error_sum",
    "mape_error_sum",
    "mape_elements",
    "sample_mae",
    "sample_mse",
    "sample_rmse",
    "sample_mape",
    "sample_elements",
    "seed",
]


def _fallback_baseline_dirname(method):
    import re

    name = str(method or "unknown").strip()
    name = re.sub(r'[<>:"/\\|?*\s]+', "_", name)
    return name or "unknown"


def _fallback_output_paths(paradigm, method):
    log_root = os.path.join(PROJECT_ROOT, f"{paradigm}_logs", _fallback_baseline_dirname(method))
    return (
        os.path.join(log_root, f"training_paradigm_{paradigm}_raw.csv"),
        os.path.join(log_root, f"training_paradigm_{paradigm}_summary.csv"),
    )


def _fallback_write_rows(path, rows):
    import csv

    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FALLBACK_RESULT_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in _FALLBACK_RESULT_FIELDNAMES})


def _fallback_sample_level_metrics(rows):
    total_elements = sum(_to_int_result(r.get("elements")) for r in rows)
    total_abs = sum(_to_float_result(r.get("abs_error_sum")) for r in rows)
    total_sq = sum(_to_float_result(r.get("sq_error_sum")) for r in rows)
    total_mape = sum(_to_float_result(r.get("mape_error_sum")) for r in rows)
    total_mape_elements = sum(_to_int_result(r.get("mape_elements")) for r in rows)

    if total_elements == 0:
        total_elements = len(rows)
        total_abs = sum(float(r["mae"]) for r in rows)
        total_sq = sum(float(r["mse"]) for r in rows)
        total_mape = sum(float(r["mape"]) for r in rows)
        total_mape_elements = len(rows)

    sample_mse = total_sq / max(total_elements, 1)
    return {
        "sample_mae": round(total_abs / max(total_elements, 1), 4),
        "sample_mse": round(sample_mse, 4),
        "sample_rmse": round(sample_mse ** 0.5, 4),
        "sample_mape": round(total_mape / total_mape_elements, 4) if total_mape_elements else 0.0,
        "sample_elements": total_elements,
    }


TOTAL_NODES_MAP = {
    "PeMS03": 358,
    "PeMS04": 307,
    "PeMSD7": 228,
    "PeMS08": 170,
    "TaxiBJ": 1024,
    "TaxiNYC": 75,
    "BikeNYC": 128,
}

GRAPH_DATASETS = ["PeMS03", "PeMS04", "PeMSD7", "PeMS08"]
GRID_DATASETS = ["TaxiBJ", "TaxiNYC", "BikeNYC"]
GRAPH_FEATURES = ["flow", "speed", "occ"]
GRID_FEATURES = ["flow"]
DEFAULT_LOCAL_CLIENTS = 4
GRID_HW_MAP = {"TaxiBJ": (32, 32), "TaxiNYC": (15, 5), "BikeNYC": (16, 8)}


@dataclass(frozen=True)
class MethodSpec:
    section: str
    method: str
    model_arg: str
    trainer_mode: str = "fedavg"


METHODS = [
    MethodSpec("Cross-Silo / Graph", "UFCL", "UFCL_GWN", "ufcl"),
    MethodSpec("Cross-Silo / Graph", "FedAGAT", "ASTGAT"),
    MethodSpec("Cross-Silo / Graph", "FedGTP", "FedGTP"),
    MethodSpec("Cross-Silo / Graph", "STAGCN-EC", "STGCN"),
    MethodSpec("Cross-Silo / Graph", "FCGCN", "FCGCN"),
    MethodSpec("Cross-Silo / Graph", "FC-FedGCN", "FCFedGCN"),
    MethodSpec("Cross-Silo / Graph", "FedTPS", "FedTPS"),
    MethodSpec("Cross-Silo / Graph", "Fed4TP", "FedTSE", "fed4tp"),
    MethodSpec("Cross-Silo / Graph", "FGNNEH", "FGNNEH"),
    MethodSpec("Cross-Silo / Graph", "FedMetro", "FedMetro"),
    MethodSpec("Cross-Silo / Graph", "pFedCTP", "pFedCTP"),
    MethodSpec("Cross-Silo / Graph", "T-ISTGNN", "ISTGNN", "tistgnn_ic"),
    MethodSpec("Cross-device / Graph", "FedOSTC", "FedOSTC"),
    MethodSpec("Cross-device / Graph", "REFOL", "REFOL"),
    MethodSpec("Cross-device / Graph", "CNFGNN", "CNFGNN"),
    MethodSpec("Cross-device / Graph", "FedSTG", "FedSTG"),
    MethodSpec("Cross-device / Graph", "FedGODE", "FedGODE"),
    MethodSpec("Cross-device / Graph", "SFT/SFL", "STGCN", "sfl"),
    MethodSpec("Cross-Silo / Grid", "FedmSSA", "FedmSSA", "fedmssa"),
    MethodSpec("Cross-Silo / Grid", "FedGRU", "FedGRU"),
    MethodSpec("Cross-Silo / Grid", "TDLR", "TDLR"),
    MethodSpec("Cross-Silo / Grid", "SEDLR", "SEDLR"),
    MethodSpec("Cross-Silo / Grid", "TDLR_SEDLR", "TDLR_SEDLR"),
    MethodSpec("Cross-device / Grid", "FUELS", "FUELS"),
    MethodSpec("Cross-device / Grid", "FedTSE", "FedTSE"),
    MethodSpec("Cross-silo / Grid", "FedSTN", "FedSTN"),
    MethodSpec("Cross-silo / Grid", "STFAM", "STFAM"),
    MethodSpec("Cross-silo / Grid", "2MGTCN", "TwoMGTCN"),
]

BASELINE_TRAINING_POLICY = {
    "UFCL": (
        "keep UFCLTrainer replay/mixup/teacher loss on the local training set",
        "remove FedAvg server aggregation and cross-client model exchange",
    ),
    "FedAGAT": (
        "keep ASTGAT client model, optimizer, validation, and physical-scale test metrics",
        "remove FedAvg parameter aggregation",
    ),
    "FedGTP": (
        "keep four FedGTP client experts, server-side EH summation, and weighted PartialFedAvg in one centralized process",
        "remove network transport only; FedGTP communication hooks are simulated locally",
    ),
    "STAGCN-EC": (
        "keep STGCN/STAGCN-EC model and client-side supervised training",
        "remove federated weight averaging",
    ),
    "FCGCN": (
        "keep FCGCN graph model and dataset-specific get_setting setup",
        "remove federated weight averaging",
    ),
    "FC-FedGCN": (
        "keep FCFedGCN model path and local supervised objective",
        "remove federated graph/model aggregation",
    ),
    "FedTPS": (
        "keep local Traffic Pattern Repository learning inside FedTPS",
        "remove cross-client Pattern upload, similarity aggregation, and Pattern download",
    ),
    "Fed4TP": (
        "keep TWT window slicing, empty-window borrowing, local gradient training, and validation",
        "remove MPL top-k intersection, GLD server flags, grouped aggregation, and weight broadcast",
    ),
    "FedmSSA": (
        "keep Page-matrix low-rank denoising and GRU prediction on denoised sequences",
        "replace the paper's ADMM-Grassmann solver with an engineering-first shared-subspace approximation for the unified benchmark",
    ),
    "FGNNEH": (
        "keep GraphSAGE/backbone extraction, hypernode generation, and context-conditioned prediction",
        "remove server hypergraph context aggregation, hypernode gradient return, and topology evolution",
    ),
    "FedMetro": (
        "keep phase1/phase2 dynamic embedding path and local regularization",
        "remove batch-level AGG summation, AGG gradient exchange, and epoch weight aggregation",
    ),
    "pFedCTP": (
        "keep pFedCTP model construction and local supervised training through common loop",
        "remove Stage-1 shared-model aggregation and target-client coordination",
    ),
    "T-ISTGNN": (
        "keep ISTGNN model construction and local supervised training through common loop",
        "remove target-client selection, source global training, and cross-client transfer aggregation",
    ),
    "FedGRU": (
        "keep FedGRU model and secure-aggregation-compatible get_setting defaults",
        "remove secure FedAvg aggregation and encrypted parameter exchange",
    ),
    "FedOSTC": (
        "keep four client encoder/decoder experts plus centralized server hidden-state concat/GAT/split",
        "remove network transport only; client/server FedOSTC modules run in one process",
    ),
    "REFOL": (
        "keep REFOL GRU predictor and REFOL sequence formatting",
        "remove concept-drift-triggered client selection and graph-based server aggregation",
    ),
    "CNFGNN": (
        "keep client GRU encoder/decoder and GraphNet spatial module on the available standalone graph",
        "remove server-mediated hidden-state exchange, gradient return, and FedAvg aggregation",
    ),
    "FedSTG": (
        "keep TP-Bank temporal pattern learning and prediction head",
        "remove global static/evolutionary graph fusion from the server and cross-client pattern exchange",
    ),
    "FedGODE": (
        "keep ODEGCN spatial/semantic graph construction and local supervised objective",
        "remove federated parameter exchange and client selection",
    ),
    "SFT/SFL": (
        "keep STGCN client backbones, SFL graph-based aggregation, and global/personalized regularizers",
        "remove only the real federated transport; SFL is simulated in one process for local/global paradigms",
    ),
    "FUELS": (
        "keep FUELS dual-view encoder, temporal augmentation, and local intra-client contrastive loss",
        "remove prototype upload, PR/NR prototype broadcast, and inter-client contrastive loss",
    ),
    "FedTSE": (
        "keep TrafficLSTM/FedTSE local sequence model and supervised objective",
        "remove asynchronous weight aggregation and server weight broadcast",
    ),
    "FedSTN": (
        "keep four quadrant FedSTN expert modules plus FedGAT hidden-state concat/attention/split in one centralized process",
        "remove network transport only; all data and FedGAT aggregation run on the same machine",
    ),
    "STFAM": (
        "keep four quadrant STFAM experts, global autoencoder pretraining, semantic embeddings, and extractor aggregation in one centralized process",
        "remove federated transport only; global pretraining and aggregation are simulated locally",
    ),
    "2MGTCN": (
        "keep TwoMGTCN multimodal flow/external-feature fusion, GCN, and TCN prediction",
        "remove FPASS spatial-similarity aggregation, personalized weight broadcast, and domain adaptation exchange",
    ),
}


class StandaloneCtx:
    def __init__(self, rank):
        self.rank = rank
        self.is_on_arbiter = False
        self.is_on_guest = rank == 0
        self.is_on_host = rank > 0


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def safe_name(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def job_log_path(paradigm, spec, dataset_name, feature):
    log_dir = os.path.join(PROJECT_ROOT, f"{paradigm}_logs", safe_name(effective_method_name(spec)))
    os.makedirs(log_dir, exist_ok=True)
    display_dataset = display_dataset_name(dataset_name)
    filename = f"{safe_name(display_dataset)}_{safe_name(feature)}.log"
    return os.path.join(log_dir, filename)


def parse_paradigm_cli():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--methods", type=str, default="all")
    parser.add_argument("--datasets", type=str, default="all")
    parser.add_argument("--features", type=str, default="all")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument(
        "--scaler_fit_scope",
        type=str,
        default=getattr(args, "scaler_fit_scope", "selected"),
        choices=["selected", "full"],
        help="Fit graph scaler on each selected subgraph or the full graph training split.",
    )
    parser.add_argument(
        "--fcgcn_adj_residual_alpha",
        type=float,
        default=getattr(args, "fcgcn_adj_residual_alpha", 0.9),
        help="FCGCN residual adjacency mix: alpha*I + (1-alpha)*A_hat. PeMSD7 remains identity.",
    )
    parser.add_argument(
        "--local_norm_scope",
        type=str,
        default=getattr(args, "local_norm_scope", "global"),
        choices=["global", "node", "column"],
        help="Scaler fitting scope used by local paradigm jobs.",
    )
    parser.add_argument(
        "--global_norm_scope",
        type=str,
        default=getattr(args, "global_norm_scope", "global"),
        choices=["global", "node", "column"],
        help="Scaler fitting scope used by global paradigm jobs.",
    )
    parser.add_argument(
        "--ufcl_ablation",
        type=str,
        default=getattr(args, "ufcl_ablation", "full"),
        choices=["full", "no_replay", "no_kd", "task_only"],
    )
    parser.add_argument(
        "--stfam_ablation",
        type=str,
        default=getattr(args, "stfam_ablation", "full"),
        choices=["full", "task_only", "low_recon", "custom"],
    )
    parser.add_argument(
        "--fedgtp_global_mode",
        type=str,
        default=getattr(args, "fedgtp_global_mode", "centralized_sim"),
        choices=["upper", "centralized_sim"],
        help=(
            "FedGTP global mode: centralized_sim (default) keeps the original "
            "four-client EH aggregation and PartialFedAvg protocol in one process; "
            "upper is the full-graph standalone ablation."
        ),
    )
    parser.add_argument(
        "--stfam_global_mode",
        type=str,
        default=getattr(args, "stfam_global_mode", "upper"),
        choices=["upper", "centralized_sim"],
        help=(
            "STFAM global mode: upper trains one full-grid centralized compact STFAM; "
            "centralized_sim keeps the four-quadrant federated STFAM protocol in one process."
        ),
    )
    parser.add_argument(
        "--fuels_global_mode",
        type=str,
        default=getattr(args, "fuels_global_mode", "upper"),
        choices=["upper", "centralized_sim", "partitioned_upper"],
        help=(
            "FUELS global mode: upper trains one full-graph FUELS model; "
            "centralized_sim keeps the four-client FUELS prototype protocol in one process; "
            "partitioned_upper trains all FUELS subgraph models centrally without communication."
        ),
    )
    parser.add_argument(
        "--stagcn_global_mode",
        type=str,
        default=getattr(args, "stagcn_global_mode", "full_graph"),
        choices=["full_graph", "partitioned_upper", "transfer_upper"],
        help=(
            "STAGCN-EC global mode: full_graph trains one centralized full-graph model; "
            "partitioned_upper trains all subgraph models centrally without communication; "
            "transfer_upper adds centralized neighbor-parameter transfer without communication."
        ),
    )
    parser.add_argument("--stagcn_transfer_warmup_steps", type=int, default=10)
    parser.add_argument(
        "--local_stagcn_eval_mode",
        type=str,
        default=getattr(args, "local_stagcn_eval_mode", "best"),
        choices=["best", "final"],
        help=(
            "STAGCN-EC local evaluation mode. best restores each client's best "
            "validation checkpoint; final evaluates the final checkpoint reached "
            "after local early stopping."
        ),
    )
    parser.add_argument(
        "--local_stagcn_graph_mode",
        type=str,
        default=getattr(args, "local_stagcn_graph_mode", "original"),
        choices=["original", "no_graph"],
        help=(
            "STAGCN-EC local graph ablation. original uses the local subgraph; "
            "no_graph replaces edge_index with self-loops before training."
        ),
    )
    parser.add_argument(
        "--fedmetro_standalone_ablation",
        type=str,
        default=getattr(args, "fedmetro_standalone_ablation", "full"),
        choices=["full", "no_agg", "no_mask_reg", "final_ckpt", "bias_only"],
        help=(
            "FedMetro standalone/local ablation. no_agg removes the local AGG "
            "context, no_mask_reg removes mask regularization, final_ckpt "
            "evaluates the final checkpoint instead of restoring best validation, "
            "bias_only disables dynamic/spatial recurrence and keeps only the "
            "output-layer bias as a weak local predictor."
        ),
    )
    parser.add_argument(
        "--fedgru_local_stop_scope",
        type=str,
        default=getattr(args, "fedgru_local_stop_scope", "client_mean"),
        choices=["client_mean", "per_client"],
        help=(
            "FedGRU local early-stop scope. client_mean trains all clients epoch by "
            "epoch and stops on the mean validation MAE; per_client keeps the legacy "
            "independent early stop for each client."
        ),
    )
    parser.add_argument(
        "--fedagat_local_eval_mode",
        type=str,
        default=getattr(args, "fedagat_local_eval_mode", "best"),
        choices=["best", "final"],
        help=(
            "FedAGAT local avg-stop evaluation mode. best restores the global "
            "mean-validation best checkpoint; final evaluates the final checkpoint "
            "after early stopping, which is a stricter ordinary-local protocol."
        ),
    )
    opts, _ = parser.parse_known_args()
    args.ufcl_ablation = opts.ufcl_ablation
    args.stfam_ablation = opts.stfam_ablation
    args.fedgtp_global_mode = opts.fedgtp_global_mode
    args.stfam_global_mode = opts.stfam_global_mode
    args.fuels_global_mode = opts.fuels_global_mode
    args.stagcn_global_mode = opts.stagcn_global_mode
    args.stagcn_transfer_warmup_steps = opts.stagcn_transfer_warmup_steps
    args.local_stagcn_eval_mode = opts.local_stagcn_eval_mode
    args.local_stagcn_graph_mode = opts.local_stagcn_graph_mode
    args.fedmetro_standalone_ablation = opts.fedmetro_standalone_ablation
    args.scaler_fit_scope = opts.scaler_fit_scope
    args.fcgcn_adj_residual_alpha = opts.fcgcn_adj_residual_alpha
    args.local_norm_scope = opts.local_norm_scope
    args.global_norm_scope = opts.global_norm_scope
    args.fedgru_local_stop_scope = opts.fedgru_local_stop_scope
    args.fedagat_local_eval_mode = opts.fedagat_local_eval_mode
    return opts


def ufcl_ablation_mode():
    return getattr(args, "ufcl_ablation", "full") or "full"


def stfam_ablation_mode():
    return getattr(args, "stfam_ablation", "full") or "full"


def effective_method_name(spec):
    mode = ufcl_ablation_mode()
    if spec.method == "UFCL" and mode != "full":
        return f"UFCL_{mode}"
    stfam_mode = stfam_ablation_mode()
    if spec.method == "STFAM" and stfam_mode != "full":
        return f"STFAM_{stfam_mode}"
    return spec.method


def _split_csv(value, default):
    if value == "all":
        return list(default)
    return [x.strip() for x in value.split(",") if x.strip()]


def selected_methods(method_names):
    if method_names == "all":
        return list(METHODS)
    wanted = set(_split_csv(method_names, []))
    if wanted & {"SFT", "SFL", "SFT_SFL"}:
        wanted.add("SFT/SFL")
    exact_method_matches = [m for m in METHODS if m.method in wanted]
    if exact_method_matches:
        return exact_method_matches
    return [m for m in METHODS if m.model_arg in wanted]


def total_nodes_for(dataset_name):
    normalized = "PeMSD7" if dataset_name == "PeMS07" else dataset_name
    for key, value in TOTAL_NODES_MAP.items():
        if key in normalized:
            return value
    raise ValueError(f"Unknown total node count for dataset {dataset_name}")


def configure_method(spec: MethodSpec):
    args.model = spec.model_arg
    args.trainer_mode = spec.trainer_mode


def configure_dataset(dataset_name, feature):
    args.dataset_name = "PeMSD7" if dataset_name == "PeMS07" else dataset_name
    args.feature_type = feature
    if args.dataset_name in GRID_DATASETS:
        args.input_dim = 2
        args.output_dim = 2
    else:
        args.input_dim = 1
        args.output_dim = 1


def configure_clients_for_local(dataset_name):
    if not any(arg == "--num_clients" or arg.startswith("--num_clients=") for arg in sys.argv):
        args.num_clients = DEFAULT_LOCAL_CLIENTS


def configure_clients_for_global(dataset_name):
    args.num_clients = 1
    args.nodes_per = [list(range(total_nodes_for(dataset_name)))]


def equal_split(total_nodes, num_clients):
    quotient, remainder = divmod(total_nodes, num_clients)
    nodes_per = []
    start = 0
    for idx in range(num_clients):
        count = quotient + (1 if idx < remainder else 0)
        nodes_per.append(list(range(start, start + count)))
        start += count
    return nodes_per


def graph_nodes_per_split(dataset_name, feature, num_clients):
    from data import dividing

    normalized = "PeMSD7" if dataset_name == "PeMS07" else dataset_name
    feature_key = {"flow": "FLOW", "speed": "SPEED", "occ": "OCCUPANCY"}.get(feature, "FLOW")
    alias_map = {
        "PeMS04": ["PeMS04", "PeMSD4"],
        "PeMSD4": ["PeMSD4", "PeMS04"],
        "PeMS08": ["PeMS08", "PeMSD8"],
        "PeMSD8": ["PeMSD8", "PeMS08"],
        "PeMS07": ["PeMSD7", "PeMS07"],
        "PeMSD7": ["PeMSD7", "PeMS07"],
    }
    base_names = []
    for name in alias_map.get(normalized, [normalized]) + alias_map.get(dataset_name, [dataset_name]):
        if name not in base_names:
            base_names.append(name)

    candidates = []
    for name in base_names:
        candidates.extend([
            f"{name}{feature_key}_{num_clients}p_metis",
            f"{name}FLOW_{num_clients}p_metis",
            f"{name}_{num_clients}p_metis",
        ])
    for name in candidates:
        if hasattr(dividing, name):
            return copy.deepcopy(getattr(dividing, name)), name
    return equal_split(total_nodes_for(normalized), num_clients), "equal_split"


def configure_clients_for_fedostc_centralized_sim(dataset_name, feature):
    args.num_clients = 4
    args.nodes_per, split_name = graph_nodes_per_split(dataset_name, feature, args.num_clients)
    return split_name


def partition_nodes_per_split(dataset_name, feature, num_clients):
    if dataset_name in GRID_DATASETS:
        return grid_rectangular_split(dataset_name, num_clients), "grid_rectangular_split"
    return graph_nodes_per_split(dataset_name, feature, num_clients)


def grid_quadrant_split(dataset_name):
    requested_clients = int(getattr(args, "num_clients", DEFAULT_LOCAL_CLIENTS) or DEFAULT_LOCAL_CLIENTS)
    return grid_rectangular_split(dataset_name, requested_clients)


def configure_clients_for_fedstn_centralized_sim(dataset_name):
    args.nodes_per = grid_quadrant_split(dataset_name)


def configure_clients_for_grid_quadrant_sim(dataset_name):
    args.nodes_per = grid_quadrant_split(dataset_name)


def get_setting(ctx):
    fate_main = importlib.import_module("fate_main")
    return fate_main.get_setting(ctx)


def _move_batch(batch, device):
    if isinstance(batch, (list, tuple)):
        return tuple(x.to(device) if torch.is_tensor(x) else x for x in batch)
    return batch.to(device) if torch.is_tensor(batch) else batch


def _unpack_xy(batch):
    if len(batch) == 5:
        x_c, _, _, x_ext, y = batch
        return x_c, y, x_ext
    x, y = batch[0], batch[1]
    return x, y, None


def _first_tensor(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def _as_pred(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def _refol_batch(x, y):
    if x.dim() == 4 and x.shape[1] != args.t_in and x.shape[2] == args.t_in:
        x_seq = x.transpose(1, 2).contiguous()
    else:
        x_seq = x
    if y.dim() == 4 and y.shape[1] != args.t_out and y.shape[2] == args.t_out:
        y_seq = y.transpose(1, 2).contiguous()
    else:
        y_seq = y
    if x_seq.dim() != 4 or y_seq.dim() != 4:
        raise RuntimeError(f"REFOL expects 4D tensors after formatting, got x={tuple(x_seq.shape)} y={tuple(y_seq.shape)}")
    if x_seq.shape[1] != args.t_in or y_seq.shape[1] != args.t_out:
        raise RuntimeError(
            f"REFOL sequence formatting failed: expected x [B,{args.t_in},N,F] "
            f"and y [B,{args.t_out},N,F], got x={tuple(x_seq.shape)} y={tuple(y_seq.shape)}"
        )
    if x_seq.shape[0] != y_seq.shape[0] or x_seq.shape[2] != y_seq.shape[2]:
        raise RuntimeError(
            f"REFOL batch/node dimensions mismatch after formatting: "
            f"x={tuple(x_seq.shape)} y={tuple(y_seq.shape)}"
        )
    x_attr = torch.empty(*x_seq.shape[:-1], 0, device=x_seq.device, dtype=x_seq.dtype)
    y_attr = torch.empty(*y_seq.shape[:-1], 0, device=y_seq.device, dtype=y_seq.dtype)
    return {
        "x": x_seq,
        "x_attr": x_attr,
        "y": y_seq,
        "y_attr": y_attr,
    }


def _cnfgnn_predict(model, x, y=None):
    h_encode, x_reshaped = model.forward_client_encoder(x)
    h_payload = h_encode.squeeze(0)
    batch_size = x.shape[0]
    h_spatial = model.forward_server_gnn(h_payload, batch_size, model.num_nodes)
    return model.forward_client_decoder(x_reshaped, y, h_encode, h_spatial)


def _forward_model_output(model, batch, loss_func=None):
    x, y, x_ext = _unpack_xy(batch)
    model_name = args.model
    aux_loss = None

    if model_name == "TwoMGTCN":
        output = model(x, x_ext)
    elif model_name == "FedSTN":
        h_s, r_out, s_out = model.forward_phase1(x, x_ext)
        local_context = h_s.mean(dim=1, keepdim=True).repeat(1, h_s.shape[1], 1)
        output = model.forward_phase2(h_s + local_context, r_out, s_out)
    elif model_name == "FedMetro":
        agg_seq, f_e_seq, e_seq = model.forward_phase1(x)
        output = model.forward_phase2(x, f_e_seq, e_seq, agg_seq)
    elif model_name == "FedSTG":
        h_tau, z_tau, l_k = model.forward_encode(x)
        h_g = torch.zeros_like(h_tau)
        output = model.forward_predict(h_tau, z_tau, h_g)
        aux_loss = l_k
    elif model_name == "FedOSTC":
        h_time = model.forward_encoder(x)
        edge_index = getattr(model, "edge_index", None)
        if edge_index is not None:
            h_time = model.forward_server_gat(h_time, edge_index.to(x.device))
        output = model.forward_decoder(h_time)
    elif model_name == "FGNNEH":
        adj = getattr(model.backbone_extractor, "adj", None)
        if adj is None:
            adj = model.backbone_extractor.adj_matrix
        adj = adj.to(x.device)
        h, h_backbone = model.forward_local(x, adj)
        context = model.generate_hypernode(h_backbone)
        output = model.forward_predict(h, context)
    elif model_name == "REFOL":
        output = model(_refol_batch(x, y))
    elif model_name == "CNFGNN":
        output = _cnfgnn_predict(model, x, y)
    elif model_name == "FedTPS":
        output = model(x, y_cov=None, labels=y if model.training else None)
    else:
        output = model(x)

    return output, y, aux_loss


def _loss_input_from_output(output, aligned_pred, loss_func=None):
    loss_name = loss_func.__class__.__name__ if loss_func is not None else ""
    if isinstance(output, tuple) and loss_name == "FedAGATLoss":
        return (aligned_pred,) + output[1:]
    return aligned_pred


def forward_model(model, batch, loss_func=None):
    output, y, aux_loss = _forward_model_output(model, batch, loss_func)
    pred = _as_pred(output)
    return pred, y, aux_loss


def forward_model_with_loss_input(model, batch, loss_func=None):
    output, y, aux_loss = _forward_model_output(model, batch, loss_func)
    pred = _as_pred(output)
    return pred, y, aux_loss, output


def align_pred_and_target(pred, y):
    if pred.dim() == 4 and pred.shape[-1] == 1 and y.dim() == 3:
        pred = pred.squeeze(-1)
    if y.dim() == 4 and y.shape[-1] == 1 and pred.dim() == 3:
        y = y.squeeze(-1)
    if pred.shape != y.shape:
        if pred.dim() == 3 and y.dim() == 3 and pred.shape[1] == y.shape[2] and pred.shape[2] == y.shape[1]:
            pred = pred.transpose(1, 2).contiguous()
        elif pred.dim() == 4 and y.dim() == 4 and pred.shape[1] == y.shape[2] and pred.shape[2] == y.shape[1]:
            pred = pred.transpose(1, 2).contiguous()
        else:
            pred = pred.reshape_as(y)
    return pred, y


def normalized_mae(model, loader, device):
    model.eval()
    total_abs = 0.0
    total_elements = 0
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            pred, y, _ = forward_model(model, batch)
            pred, y = align_pred_and_target(pred, y)
            total_abs += torch.abs(pred - y).sum().item()
            total_elements += y.numel()
    return total_abs / max(total_elements, 1)


def test_metrics(model, loader, scaler, device):
    model.eval()
    total_abs = 0.0
    total_sq = 0.0
    total_elements = 0
    total_mape = 0.0
    total_mape_elements = 0

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            pred, y, _ = forward_model(model, batch)
            pred, y = align_pred_and_target(pred, y)
            y_real = scaler.inverse_transform(y).detach().cpu().numpy()
            pred_real = scaler.inverse_transform(pred).detach().cpu().numpy()
            diff = pred_real - y_real
            total_abs += np.abs(diff).sum()
            total_sq += np.square(diff).sum()
            total_elements += y_real.size
            mask = y_real > 0.5
            if np.any(mask):
                total_mape += (np.abs(diff[mask]) / y_real[mask]).sum()
                total_mape_elements += int(mask.sum())

    mae = total_abs / max(total_elements, 1)
    mse = total_sq / max(total_elements, 1)
    rmse = math.sqrt(mse)
    mape = total_mape / total_mape_elements if total_mape_elements > 0 else 0.0
    return {
        "mae": round(float(mae), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(rmse), 4),
        "mape": round(float(mape), 4),
        "elements": int(total_elements),
        "abs_error_sum": float(total_abs),
        "sq_error_sum": float(total_sq),
        "mape_error_sum": float(total_mape),
        "mape_elements": int(total_mape_elements),
    }


def _metrics_from_loader(model, loader, scaler, device, forward_fn):
    model.eval()
    total_abs = 0.0
    total_sq = 0.0
    total_elements = 0
    total_mape = 0.0
    total_mape_elements = 0

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            pred, y = forward_fn(model, batch, training=False)
            pred, y = align_pred_and_target(_as_pred(pred), y)
            y_real = scaler.inverse_transform(y).detach().cpu().numpy()
            pred_real = scaler.inverse_transform(pred).detach().cpu().numpy()
            diff = pred_real - y_real
            total_abs += np.abs(diff).sum()
            total_sq += np.square(diff).sum()
            total_elements += y_real.size
            mask = y_real > 0.5
            if np.any(mask):
                total_mape += (np.abs(diff[mask]) / y_real[mask]).sum()
                total_mape_elements += int(mask.sum())

    mse = total_sq / max(total_elements, 1)
    return {
        "mae": round(float(total_abs / max(total_elements, 1)), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(math.sqrt(mse)), 4),
        "mape": round(float(total_mape / total_mape_elements), 4) if total_mape_elements else 0.0,
        "elements": int(total_elements),
        "abs_error_sum": float(total_abs),
        "sq_error_sum": float(total_sq),
        "mape_error_sum": float(total_mape),
        "mape_elements": int(total_mape_elements),
    }


def _normalized_mae_with_forward(model, loader, device, forward_fn):
    model.eval()
    total_abs = 0.0
    total_elements = 0
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            pred, y = forward_fn(model, batch, training=False)
            pred, y = align_pred_and_target(_as_pred(pred), y)
            if not torch.isfinite(pred).all() or not torch.isfinite(y).all():
                return float("nan")
            total_abs += torch.abs(pred - y).sum().item()
            total_elements += y.numel()
    return total_abs / max(total_elements, 1)


def _real_mae_with_forward(model, loader, scaler, device, forward_fn):
    model.eval()
    total_abs = 0.0
    total_elements = 0
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            pred, y = forward_fn(model, batch, training=False)
            pred, y = align_pred_and_target(_as_pred(pred), y)
            if not torch.isfinite(pred).all() or not torch.isfinite(y).all():
                return float("nan")
            y_real = scaler.inverse_transform(y).detach().cpu().numpy()
            pred_real = scaler.inverse_transform(pred).detach().cpu().numpy()
            diff = pred_real - y_real
            total_abs += np.abs(diff).sum()
            total_elements += y_real.size
    return float(total_abs / max(total_elements, 1))


def _standard_train_loop(
    model,
    optimizer,
    loss_func,
    train_loader,
    val_loader,
    test_loader,
    scaler,
    opts,
    forward_fn,
    trainer_name,
    extra_loss_fn=None,
    grad_clip_norm=None,
    stop_on_nonfinite=False,
    restore_best=True,
):
    device = args.device
    model.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if loss_func is None:
        loss_func = torch.nn.MSELoss().to(device)
    elif hasattr(loss_func, "to"):
        loss_func = loss_func.to(device)

    best_state = None
    last_finite_state = None
    best_val = float("inf")
    patience_count = 0
    stopped_for_nonfinite = False

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch_idx, batch in enumerate(train_loader):
            if opts.max_batches and batch_idx >= opts.max_batches:
                break
            batch = _move_batch(batch, device)
            optimizer.zero_grad()
            output, y = forward_fn(model, batch, training=True)
            pred, y = align_pred_and_target(_as_pred(output), y)
            loss = loss_func(_loss_input_from_output(output, pred, loss_func), y)
            if extra_loss_fn is not None:
                loss = loss + extra_loss_fn(model)
            if not torch.isfinite(loss):
                print(
                    f"[{trainer_name}] Epoch {epoch} batch {batch_idx} | "
                    f"non_finite_loss={loss.detach().item()} | stopping and rolling back to best",
                    flush=True,
                )
                stopped_for_nonfinite = True
                break
            loss.backward()
            if grad_clip_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                if not torch.isfinite(grad_norm):
                    print(
                        f"[{trainer_name}] Epoch {epoch} batch {batch_idx} | "
                        f"non_finite_grad_norm={grad_norm} | stopping and rolling back to best",
                        flush=True,
                    )
                    stopped_for_nonfinite = True
                    break
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        if stopped_for_nonfinite:
            break

        val_mae = _normalized_mae_with_forward(model, val_loader, device, forward_fn)
        print(
            f"[{trainer_name}] Epoch {epoch} | Train Loss(Norm): "
            f"{total_loss / max(steps, 1):.4f} | Val MAE(Norm): {val_mae:.4f}"
        )
        if not math.isfinite(val_mae):
            print(
                f"[{trainer_name}] Epoch {epoch} | non_finite_val_mae={val_mae} "
                f"| stopping and rolling back to best",
                flush=True,
            )
            if stop_on_nonfinite:
                stopped_for_nonfinite = True
                break
        elif not restore_best:
            last_finite_state = copy.deepcopy(model.state_dict())
        if val_mae < best_val - opts.min_delta:
            best_val = val_mae
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= opts.patience:
                print(f"[{trainer_name}] Early stop at epoch {epoch}, best_val={best_val:.4f}")
                break

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    elif stopped_for_nonfinite and last_finite_state is not None:
        print(
            f"[{trainer_name}] non-finite stop; restoring last finite checkpoint "
            "for final-checkpoint evaluation",
            flush=True,
        )
        model.load_state_dict(last_finite_state)
    elif stopped_for_nonfinite:
        raise RuntimeError(f"{trainer_name} hit non-finite values before any valid checkpoint was saved.")
    return _metrics_from_loader(model, test_loader, scaler, device, forward_fn)


def _basic_forward(model, batch, training=False):
    pred, y, aux_loss = forward_model(model, batch)
    if aux_loss is not None and training:
        # The common loop handles auxiliary losses only for models that still
        # expose them through forward_model. Add it by wrapping pred as-is.
        pass
    return pred, y


def _stfam_loaders(train_set, val_set, test_set):
    from lib.stfam_loader import unified_dataset_to_stfam_xy

    max_local_nodes = max(len(nodes) for nodes in args.nodes_per)
    if hasattr(train_set, "adj") and train_set.adj is not None:
        edge_index = torch.tensor(train_set.adj, dtype=torch.float32).nonzero(as_tuple=False).t().contiguous()
        x_train, y_train = unified_dataset_to_stfam_xy(train_set, args.t_in, args.t_out)
        x_val, y_val = unified_dataset_to_stfam_xy(val_set, args.t_in, args.t_out)
        x_test, y_test = unified_dataset_to_stfam_xy(test_set, args.t_in, args.t_out)
    else:
        raise NotImplementedError("STFAM standalone baseline currently expects grid datasets with train_set.adj")

    train_data = CompactSTFAMDataset(x_train, y_train, edge_index, max_local_nodes)
    val_data = CompactSTFAMDataset(x_val, y_val, edge_index, max_local_nodes)
    test_data = CompactSTFAMDataset(x_test, y_test, edge_index, max_local_nodes)
    return (
        DataLoader(train_data, batch_size=args.batch_size, shuffle=True),
        DataLoader(val_data, batch_size=args.batch_size, shuffle=False),
        DataLoader(test_data, batch_size=args.batch_size, shuffle=False),
        max_local_nodes,
    )


class CompactSTFAMDataset(torch.utils.data.Dataset):
    def __init__(self, x_data, y_data, edge_index, max_nodes):
        import torch.nn.functional as F

        self.x = x_data.float() if torch.is_tensor(x_data) else torch.as_tensor(x_data, dtype=torch.float32)
        self.y = y_data.float() if torch.is_tensor(y_data) else torch.as_tensor(y_data, dtype=torch.float32)
        self.max_nodes = max_nodes
        node_count = self.x.shape[1]
        pad_len = max_nodes - node_count
        if pad_len > 0:
            self.x = F.pad(self.x, (0, 0, 0, 0, 0, pad_len))
            self.y = F.pad(self.y, (0, 0, 0, 0, 0, pad_len))

        adj = torch.zeros(max_nodes, max_nodes, dtype=torch.float32)
        if edge_index.numel() > 0:
            edge_index = edge_index.long()
            mask = (edge_index[0] < max_nodes) & (edge_index[1] < max_nodes)
            adj[edge_index[0, mask], edge_index[1, mask]] = 1.0
        adj.fill_diagonal_(1.0)
        row_sum = adj.sum(dim=1, keepdim=True).clamp_min(1.0)
        self.a_prob = adj / row_sum

        total_flow = self.x.mean(dim=(0, 2))
        total_agg = torch.matmul(self.a_prob, total_flow)
        self.u = torch.cat([total_flow, total_agg], dim=-1).reshape(1, -1)
        print(
            f"[STFAM compact standalone] samples={len(self.x)} nodes={max_nodes} "
            f"Tr_dim={max_nodes * self.x.shape[-1] * 2} "
            f"instead_of_dense_Tr_dim={max_nodes * max_nodes * self.x.shape[-1]}",
            flush=True,
        )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_i = self.x[idx]
        flow = x_i.permute(1, 0, 2).contiguous()
        agg = torch.einsum("ij,tjc->tic", self.a_prob, flow)
        tr = torch.cat([flow, agg], dim=-1).reshape(flow.shape[0], -1)
        return tr, self.u, self.y[idx]


class CompactSTFAMStandaloneModel(torch.nn.Module):
    def __init__(self, t_in, num_nodes, embed_dim, in_channels, pred_steps, out_channels):
        super().__init__()
        from model.STFAM import (
            Global_1D_CNN_Autoencoder,
            Local_LSTM_Autoencoder,
            STFAM_Data_Fusion,
            STFAM_Predictor,
        )

        self.num_nodes = num_nodes
        self.pred_steps = pred_steps
        self.out_channels = out_channels
        self.tr_dim = num_nodes * in_channels * 2
        self.local_lstm = Local_LSTM_Autoencoder(input_dim=self.tr_dim, hidden_dim=embed_dim, embed_dim=embed_dim)
        self.local_cnn = Global_1D_CNN_Autoencoder(input_dim=self.tr_dim, embed_dim=embed_dim)
        self.fusion = STFAM_Data_Fusion(embed_dim)
        self.predictor = STFAM_Predictor(embed_dim, num_nodes * pred_steps * out_channels)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, tr, u, v_d, p_embed):
        r_tr, tr_recon = self.local_lstm(tr)
        v_u, u_recon = self.local_cnn(u)
        batch_size = r_tr.size(0)
        v_d_exp = v_d.expand(batch_size, -1) if v_d.size(0) == 1 and batch_size > 1 else v_d
        p_embed_exp = p_embed.expand(batch_size, -1) if p_embed.size(0) == 1 and batch_size > 1 else p_embed
        r, v = self.fusion(r_tr, v_u, v_d_exp, p_embed_exp)
        out = self.predictor(r, v)
        out = out.view(batch_size, self.num_nodes, self.pred_steps, self.out_channels)
        return out, tr_recon, u_recon

    def compute_loss(self, tr, u, y, v_d, p_embed, alpha=0.1, beta=0.1):
        pred, tr_recon, u_recon = self.forward(tr, u, v_d, p_embed)
        pred, y = align_pred_and_target(pred, y)
        return self.loss_fn(pred, y) + alpha * self.loss_fn(tr_recon, tr) + beta * self.loss_fn(u_recon, u)

    def loss_components(self, tr, u, y, v_d, p_embed):
        pred, tr_recon, u_recon = self.forward(tr, u, v_d, p_embed)
        pred, y = align_pred_and_target(pred, y)
        return (
            self.loss_fn(pred, y),
            self.loss_fn(tr_recon, tr),
            self.loss_fn(u_recon, u),
        )


def _stfam_metrics(model, loader, scaler, device, v_d, p_embed):
    model.eval()
    total_abs = 0.0
    total_sq = 0.0
    total_elements = 0
    total_mape = 0.0
    total_mape_elements = 0
    y_sum = 0.0
    pred_sum = 0.0
    y_sq_sum = 0.0
    pred_sq_sum = 0.0

    with torch.no_grad():
        for tr, u, y in loader:
            tr, u, y = tr.to(device), u.to(device), y.to(device)
            pred, _, _ = model(tr, u, v_d, p_embed)
            pred, y = align_pred_and_target(pred, y)
            y_real = scaler.inverse_transform(y).detach().cpu().numpy()
            pred_real = scaler.inverse_transform(pred).detach().cpu().numpy()
            diff = pred_real - y_real
            total_abs += np.abs(diff).sum()
            total_sq += np.square(diff).sum()
            total_elements += y_real.size
            y_sum += y_real.sum()
            pred_sum += pred_real.sum()
            y_sq_sum += np.square(y_real).sum()
            pred_sq_sum += np.square(pred_real).sum()
            mask = y_real > 0.5
            if np.any(mask):
                total_mape += (np.abs(diff[mask]) / y_real[mask]).sum()
                total_mape_elements += int(mask.sum())

    mae = total_abs / max(total_elements, 1)
    mse = total_sq / max(total_elements, 1)
    y_mean = y_sum / max(total_elements, 1)
    pred_mean = pred_sum / max(total_elements, 1)
    y_var = max(y_sq_sum / max(total_elements, 1) - y_mean * y_mean, 0.0)
    pred_var = max(pred_sq_sum / max(total_elements, 1) - pred_mean * pred_mean, 0.0)
    print(
        f"[STFAM compact standalone][test] MAE={mae:.4f} RMSE={math.sqrt(mse):.4f} "
        f"y_mean={y_mean:.4f} pred_mean={pred_mean:.4f} "
        f"y_std={math.sqrt(y_var):.4f} pred_std={math.sqrt(pred_var):.4f}",
        flush=True,
    )
    return {
        "mae": round(float(mae), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(math.sqrt(mse)), 4),
        "mape": round(float(total_mape / total_mape_elements), 4) if total_mape_elements else 0.0,
    }


def _stfam_dataset_nodes(dataset_name):
    for key, val in TOTAL_NODES_MAP.items():
        if key in dataset_name:
            return val
    raise ValueError(f"Unknown STFAM dataset node count for {dataset_name}")


def _stfam_recon_weights():
    mode = stfam_ablation_mode()
    if mode == "task_only":
        return 0.0, 0.0
    if mode == "low_recon":
        return 0.001, 0.001
    return float(getattr(args, "stfam_recon_alpha", 0.1)), float(getattr(args, "stfam_recon_beta", 0.1))


def _pretrain_stfam_global_context(num_nodes, device):
    from lib.stfam_loader import build_global_data
    from model.STFAM import Global_1D_CNN_Autoencoder, Global_2D_CNN_Autoencoder

    dataset_nodes = _stfam_dataset_nodes(args.dataset_name)
    D_tensor, P_tensor = build_global_data(args.dataset_name, dataset_nodes, PROJECT_ROOT)
    if dataset_nodes != num_nodes:
        D_tensor = D_tensor[:, :num_nodes, :num_nodes]
        P_tensor = P_tensor[:num_nodes]

    global_2d_cnn = Global_2D_CNN_Autoencoder(1, args.stfam_embed_dim).to(device)
    global_1d_cnn = Global_1D_CNN_Autoencoder(1, args.stfam_embed_dim).to(device)
    optimizer = torch.optim.Adam(
        list(global_2d_cnn.parameters()) + list(global_1d_cnn.parameters()),
        lr=args.lr,
    )
    mse_loss = torch.nn.MSELoss()
    D_input = D_tensor.to(device).unsqueeze(0)
    P_input = P_tensor.to(device).view(1, 1, -1)
    pretrain_epochs = int(getattr(args, "stfam_global_pretrain_epochs", 200))

    print(
        f"[STFAM centralized global] Pretraining global autoencoders "
        f"nodes={num_nodes} epochs={pretrain_epochs}",
        flush=True,
    )
    global_2d_cnn.train()
    global_1d_cnn.train()
    for _ in range(pretrain_epochs):
        optimizer.zero_grad()
        _, D_recon = global_2d_cnn(D_input)
        _, P_recon = global_1d_cnn(P_input)
        loss = mse_loss(D_recon, D_input) + mse_loss(P_recon, P_input)
        loss.backward()
        optimizer.step()

    global_2d_cnn.eval()
    global_1d_cnn.eval()
    with torch.no_grad():
        v_d, _ = global_2d_cnn(D_input)
        p_embed, _ = global_1d_cnn(P_input)
    print(
        f"[STFAM centralized global] Semantic embeddings ready "
        f"V_D={tuple(v_d.shape)} P_embed={tuple(p_embed.shape)}",
        flush=True,
    )
    return v_d.detach(), p_embed.detach(), global_1d_cnn.state_dict()


def train_stfam_standalone(train_set, val_set, test_set, loss_func, scaler, opts):
    train_loader, val_loader, test_loader, max_local_nodes = _stfam_loaders(train_set, val_set, test_set)
    device = args.device
    in_channels = getattr(args, "input_dim", 2)
    out_channels = getattr(args, "output_dim", 2)
    model = CompactSTFAMStandaloneModel(
        t_in=args.t_in,
        num_nodes=max_local_nodes,
        embed_dim=args.stfam_embed_dim,
        in_channels=in_channels,
        pred_steps=args.t_out,
        out_channels=out_channels,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    v_d, p_embed, global_1d_state = _pretrain_stfam_global_context(max_local_nodes, device)
    model.local_cnn.load_state_dict(global_1d_state, strict=True)
    recon_alpha, recon_beta = _stfam_recon_weights()

    best_state = None
    best_val = float("inf")
    patience_count = 0
    print(
        f"[STFAM compact standalone] train_batches={len(train_loader)} "
        f"val_batches={len(val_loader)} test_batches={len(test_loader)} "
        f"nodes={max_local_nodes} ablation={stfam_ablation_mode()} "
        f"recon_alpha={recon_alpha} recon_beta={recon_beta}",
        flush=True,
    )

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        pred_loss_sum = 0.0
        tr_loss_sum = 0.0
        u_loss_sum = 0.0
        steps = 0
        for batch_idx, (tr, u, y) in enumerate(train_loader):
            if opts.max_batches and batch_idx >= opts.max_batches:
                break
            tr, u, y = tr.to(device), u.to(device), y.to(device)
            optimizer.zero_grad()
            pred_loss, tr_loss, u_loss = model.loss_components(tr, u, y, v_d, p_embed)
            loss = pred_loss + recon_alpha * tr_loss + recon_beta * u_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()
            pred_loss_sum += pred_loss.item()
            tr_loss_sum += tr_loss.item()
            u_loss_sum += u_loss.item()
            steps += 1

        model.eval()
        val_abs = 0.0
        val_elements = 0
        with torch.no_grad():
            for tr, u, y in val_loader:
                tr, u, y = tr.to(device), u.to(device), y.to(device)
                pred, _, _ = model(tr, u, v_d, p_embed)
                pred, y = align_pred_and_target(pred, y)
                val_abs += torch.abs(pred - y).sum().item()
                val_elements += y.numel()
        val_mae = val_abs / max(val_elements, 1)
        print(
            f"[STFAM compact standalone] Epoch {epoch} | Train Loss(Norm): "
            f"{train_loss / max(steps, 1):.4f} | Val MAE(Norm): {val_mae:.4f} "
            f"| pred={pred_loss_sum / max(steps, 1):.6f} "
            f"tr_recon={tr_loss_sum / max(steps, 1):.6f} "
            f"u_recon={u_loss_sum / max(steps, 1):.6f} "
            f"| best={best_val:.4f} | patience={patience_count}/{opts.patience}",
            flush=True,
        )
        if val_mae < best_val - opts.min_delta:
            best_val = val_mae
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= opts.patience:
                print(f"[STFAM compact standalone] Early stop at epoch {epoch}, best_val={best_val:.4f}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return _stfam_metrics(model, test_loader, scaler, device, v_d, p_embed)


def _pretrain_stfam_global_bundle(device):
    from lib.stfam_loader import build_global_data
    from model.STFAM import Global_1D_CNN_Autoencoder, Global_2D_CNN_Autoencoder

    num_global_nodes = _stfam_dataset_nodes(args.dataset_name)
    D_tensor, P_tensor = build_global_data(args.dataset_name, num_global_nodes, PROJECT_ROOT)
    global_2d_cnn = Global_2D_CNN_Autoencoder(1, args.stfam_embed_dim).to(device)
    global_1d_cnn = Global_1D_CNN_Autoencoder(1, args.stfam_embed_dim).to(device)
    optimizer = torch.optim.Adam(
        list(global_2d_cnn.parameters()) + list(global_1d_cnn.parameters()),
        lr=args.lr,
    )
    mse_loss = torch.nn.MSELoss()
    D_input = D_tensor.to(device).unsqueeze(0)
    P_input = P_tensor.to(device).view(1, 1, -1)
    pretrain_epochs = int(getattr(args, "stfam_global_pretrain_epochs", 200))

    print(
        f"[STFAM centralized-upper] Pretraining global autoencoders "
        f"nodes={num_global_nodes} epochs={pretrain_epochs}",
        flush=True,
    )
    global_2d_cnn.train()
    global_1d_cnn.train()
    for _ in range(pretrain_epochs):
        optimizer.zero_grad()
        _, D_recon = global_2d_cnn(D_input)
        _, P_recon = global_1d_cnn(P_input)
        loss = mse_loss(D_recon, D_input) + mse_loss(P_recon, P_input)
        loss.backward()
        optimizer.step()

    return (
        D_tensor,
        P_tensor,
        {k: v.detach().cpu().clone() for k, v in global_2d_cnn.state_dict().items()},
        {k: v.detach().cpu().clone() for k, v in global_1d_cnn.state_dict().items()},
    )


def _stfam_semantic_context_for_nodes(selected_nodes, full_D, full_P, global_2d_state, global_1d_state, device):
    from model.STFAM import Global_1D_CNN_Autoencoder, Global_2D_CNN_Autoencoder

    local_idx = torch.tensor(selected_nodes, dtype=torch.long)
    sub_D = full_D[0][local_idx][:, local_idx].unsqueeze(0).unsqueeze(0).to(device)
    sub_P = full_P[local_idx, :].to(device).view(1, 1, -1)

    local_g2d = Global_2D_CNN_Autoencoder(1, args.stfam_embed_dim).to(device)
    local_g1d = Global_1D_CNN_Autoencoder(1, args.stfam_embed_dim).to(device)
    local_g2d.load_state_dict(global_2d_state)
    local_g1d.load_state_dict(global_1d_state)
    local_g2d.eval()
    local_g1d.eval()
    with torch.no_grad():
        v_d, _ = local_g2d(sub_D)
        p_embed, _ = local_g1d(sub_P)
    return v_d.detach(), p_embed.detach()


def _stfam_upper_load_parts(device):
    from lib.load_dataset import load_grid_dataset_for_fedstn
    from lib.stfam_loader import load_stfam_client_dataset, unified_dataset_to_stfam_xy
    from model.STFAM import STFAM_Client_Model

    full_D, full_P, global_2d_state, global_1d_state = _pretrain_stfam_global_bundle(device)
    max_local_nodes = max(len(nodes) for nodes in args.nodes_per)
    train_loaders, val_loaders, test_loaders = [], [], []
    scalers, models, optimizers, semantic_contexts = [], [], [], []
    in_channels = getattr(args, "input_dim", 2)
    out_channels = getattr(args, "output_dim", 2)

    for rank, selected_nodes in enumerate(args.nodes_per):
        train_set_raw, val_set_raw, test_set_raw, edge_index, scaler = load_grid_dataset_for_fedstn(
            dataset_name=args.dataset_name,
            t_in=args.t_in,
            t_out=args.t_out,
            device=device,
            selected_nodes=selected_nodes,
            model_name="STFAM",
        )
        x_train, y_train = unified_dataset_to_stfam_xy(train_set_raw, args.t_in, args.t_out)
        x_val, y_val = unified_dataset_to_stfam_xy(val_set_raw, args.t_in, args.t_out)
        x_test, y_test = unified_dataset_to_stfam_xy(test_set_raw, args.t_in, args.t_out)

        train_data = load_stfam_client_dataset(x_train, y_train, edge_index, max_local_nodes, device)
        val_data = load_stfam_client_dataset(x_val, y_val, edge_index, max_local_nodes, device)
        test_data = load_stfam_client_dataset(x_test, y_test, edge_index, max_local_nodes, device)
        train_loaders.append(DataLoader(train_data, batch_size=args.batch_size, shuffle=True))
        val_loaders.append(DataLoader(val_data, batch_size=args.batch_size, shuffle=False))
        test_loaders.append(DataLoader(test_data, batch_size=args.batch_size, shuffle=False))

        init_seed(args.seed)
        model = STFAM_Client_Model(
            t_in=args.t_in,
            num_local_nodes=max_local_nodes,
            embed_dim=args.stfam_embed_dim,
            in_channels=in_channels,
            pred_steps=args.t_out,
            out_channels=out_channels,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        v_d, p_embed = _stfam_semantic_context_for_nodes(
            selected_nodes, full_D, full_P, global_2d_state, global_1d_state, device
        )

        scalers.append(scaler)
        models.append(model)
        optimizers.append(optimizer)
        semantic_contexts.append((v_d, p_embed))
        print(
            f"[STFAM centralized-upper] client{rank} quadrant_nodes={len(selected_nodes)} "
            f"padded_nodes={max_local_nodes} train_batches={len(train_loaders[-1])} "
            f"val_batches={len(val_loaders[-1])} test_batches={len(test_loaders[-1])}",
            flush=True,
        )

    return train_loaders, val_loaders, test_loaders, scalers, models, optimizers, semantic_contexts


def _stfam_upper_tsvd_vector(train_loader):
    from scipy.sparse.linalg import svds

    all_tr = []
    sample_count = 0
    for tr_batch, _, _ in train_loader:
        all_tr.append(tr_batch.cpu().numpy())
        sample_count += tr_batch.shape[0]
        if sample_count >= 256:
            break
    all_tr = np.concatenate(all_tr, axis=0)
    tr_matrix = all_tr.reshape(-1, all_tr.shape[-1]).T
    k_svd = min(args.tsvd_dim, min(tr_matrix.shape) - 1)
    principal = np.zeros(args.tsvd_dim, dtype=np.float32)
    if k_svd > 0:
        _, sigma, _ = svds(tr_matrix, k=k_svd)
        sigma = sigma[::-1]
    else:
        _, sigma, _ = np.linalg.svd(tr_matrix, full_matrices=False)
    fill_len = min(args.tsvd_dim, len(sigma))
    principal[:fill_len] = sigma[:fill_len]
    return principal


def _stfam_upper_cluster_clients(train_loaders):
    from sklearn.cluster import KMeans

    vectors = np.stack([_stfam_upper_tsvd_vector(loader) for loader in train_loaders], axis=0)
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    n_clusters = min(args.num_clusters, len(train_loaders))
    clusters = KMeans(n_clusters=n_clusters, random_state=args.seed).fit(vectors).labels_
    print(f"[STFAM centralized-upper] TSVD/KMeans clusters={clusters.tolist()}", flush=True)
    return clusters


def _stfam_upper_local_state(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if "local_" in key
    }


def _stfam_upper_delta_norm(model, old_params):
    delta = 0.0
    for key, value in model.named_parameters():
        if "local_" in key:
            delta += torch.norm(value.detach() - old_params[key], p=2).item() ** 2
    return delta


def _stfam_upper_average_states(states):
    return {
        key: torch.stack([state[key] for state in states], dim=0).mean(dim=0)
        for key in states[0].keys()
    }


def _stfam_upper_aggregate_local_extractors(models, delta_norms, clusters, previous_state):
    threshold = float(np.mean(delta_norms))
    uploaded = []
    skipped = []
    all_payloads = []
    for idx, model in enumerate(models):
        if delta_norms[idx] >= threshold:
            payload = _stfam_upper_local_state(model)
            uploaded.append((idx, payload))
            all_payloads.append(payload)
        else:
            skipped.append(idx)
            all_payloads.append("SKIP")

    final_states = [None] * len(models)
    for idx, payload in uploaded:
        final_states[idx] = payload

    for idx in skipped:
        peer_states = [
            all_payloads[j]
            for j in range(len(models))
            if clusters[j] == clusters[idx] and not isinstance(all_payloads[j], str)
        ]
        if peer_states:
            final_states[idx] = _stfam_upper_average_states(peer_states)
        elif previous_state is not None:
            final_states[idx] = previous_state
        elif uploaded:
            final_states[idx] = uploaded[0][1]
        else:
            final_states[idx] = _stfam_upper_local_state(models[idx])

    global_state = _stfam_upper_average_states(final_states)
    for model in models:
        model.load_state_dict(global_state, strict=False)
    return global_state, threshold, len(uploaded), len(skipped)


def _stfam_upper_normalized_mae(models, val_loaders, semantic_contexts, device):
    for model in models:
        model.eval()
    total_abs = 0.0
    total_elements = 0
    with torch.no_grad():
        for model, loader, (v_d, p_embed) in zip(models, val_loaders, semantic_contexts):
            for tr, u, y in loader:
                tr, u, y = tr.to(device), u.to(device), y.to(device)
                pred, _, _ = model(tr, u, v_d, p_embed)
                pred, y = align_pred_and_target(pred, y)
                total_abs += torch.abs(pred - y).sum().item()
                total_elements += y.numel()
    return total_abs / max(total_elements, 1)


def _stfam_upper_test_metrics(models, test_loaders, scalers, semantic_contexts, device):
    for model in models:
        model.eval()
    total_abs = 0.0
    total_sq = 0.0
    total_elements = 0
    total_mape = 0.0
    total_mape_elements = 0
    client_summaries = []
    with torch.no_grad():
        for client_idx, (model, loader, scaler, (v_d, p_embed)) in enumerate(
            zip(models, test_loaders, scalers, semantic_contexts)
        ):
            client_abs = 0.0
            client_sq = 0.0
            client_elements = 0
            y_sum = 0.0
            pred_sum = 0.0
            y_sq_sum = 0.0
            pred_sq_sum = 0.0
            for tr, u, y in loader:
                tr, u, y = tr.to(device), u.to(device), y.to(device)
                pred, _, _ = model(tr, u, v_d, p_embed)
                pred, y = align_pred_and_target(pred, y)
                y_real = scaler.inverse_transform(y).detach().cpu().numpy()
                pred_real = scaler.inverse_transform(pred).detach().cpu().numpy()
                diff = pred_real - y_real
                abs_sum = np.abs(diff).sum()
                sq_sum = np.square(diff).sum()
                total_abs += abs_sum
                total_sq += sq_sum
                total_elements += y_real.size
                client_abs += abs_sum
                client_sq += sq_sum
                client_elements += y_real.size
                y_sum += y_real.sum()
                pred_sum += pred_real.sum()
                y_sq_sum += np.square(y_real).sum()
                pred_sq_sum += np.square(pred_real).sum()
                mask = y_real > 0.5
                if np.any(mask):
                    total_mape += (np.abs(diff[mask]) / y_real[mask]).sum()
                    total_mape_elements += int(mask.sum())
            if client_elements:
                y_mean = y_sum / client_elements
                pred_mean = pred_sum / client_elements
                y_var = max(y_sq_sum / client_elements - y_mean * y_mean, 0.0)
                pred_var = max(pred_sq_sum / client_elements - pred_mean * pred_mean, 0.0)
                client_summaries.append(
                    {
                        "client": client_idx,
                        "mae": client_abs / client_elements,
                        "rmse": math.sqrt(client_sq / client_elements),
                        "y_mean": y_mean,
                        "pred_mean": pred_mean,
                        "y_std": math.sqrt(y_var),
                        "pred_std": math.sqrt(pred_var),
                    }
                )

    mse = total_sq / max(total_elements, 1)
    metrics = {
        "mae": round(float(total_abs / max(total_elements, 1)), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(math.sqrt(mse)), 4),
        "mape": round(float(total_mape / total_mape_elements), 4) if total_mape_elements else 0.0,
    }
    for summary in client_summaries:
        print(
            "[STFAM centralized-upper][test client{client}] "
            "MAE={mae:.4f} RMSE={rmse:.4f} "
            "y_mean={y_mean:.4f} pred_mean={pred_mean:.4f} "
            "y_std={y_std:.4f} pred_std={pred_std:.4f}".format(**summary),
            flush=True,
        )
    print(
        f"[STFAM centralized-upper][test overall] "
        f"MAE={metrics['mae']:.4f} MSE={metrics['mse']:.4f} "
        f"RMSE={metrics['rmse']:.4f} MAPE={metrics['mape']:.4f}",
        flush=True,
    )
    return metrics


def train_stfam_centralized_upper(setting, opts):
    print(
        "[STFAM centralized-upper] Running four-quadrant STFAM with global AE "
        "pretraining, semantic embeddings, TSVD clustering, and local extractor aggregation in one process.",
        flush=True,
    )
    device = args.device
    train_loaders, val_loaders, test_loaders, scalers, models, optimizers, semantic_contexts = _stfam_upper_load_parts(device)
    clusters = _stfam_upper_cluster_clients(train_loaders)
    recon_alpha, recon_beta = _stfam_recon_weights()
    print(
        f"[STFAM centralized-upper] ablation={stfam_ablation_mode()} "
        f"recon_alpha={recon_alpha} recon_beta={recon_beta}",
        flush=True,
    )
    best_states = None
    best_val = float("inf")
    patience_count = 0
    previous_global_state = None

    for epoch in range(args.epochs):
        train_losses = []
        delta_norms = []
        for model, optimizer, loader, (v_d, p_embed) in zip(models, optimizers, train_loaders, semantic_contexts):
            model.train()
            old_params = {
                key: value.detach().clone()
                for key, value in model.named_parameters()
                if "local_" in key
            }
            total_loss = 0.0
            steps = 0
            for batch_idx, (tr, u, y) in enumerate(loader):
                if opts.max_batches and batch_idx >= opts.max_batches:
                    break
                tr, u, y = tr.to(device), u.to(device), y.to(device)
                optimizer.zero_grad()
                loss = model.compute_loss(tr, u, y, v_d, p_embed, alpha=recon_alpha, beta=recon_beta)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += loss.item()
                steps += 1
            train_losses.append(total_loss / max(steps, 1))
            delta_norms.append(_stfam_upper_delta_norm(model, old_params))

        previous_global_state, threshold, uploaded_count, skipped_count = _stfam_upper_aggregate_local_extractors(
            models, delta_norms, clusters, previous_global_state
        )
        val_mae = _stfam_upper_normalized_mae(models, val_loaders, semantic_contexts, device)
        if val_mae < best_val - opts.min_delta:
            best_val = val_mae
            best_states = [copy.deepcopy(model.state_dict()) for model in models]
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= opts.patience:
                print(
                    f"[STFAM centralized-upper] Epoch {epoch} | Train Loss(Norm): "
                    f"{float(np.mean(train_losses)):.4f} | Val MAE(Norm): {val_mae:.4f} "
                    f"| threshold={threshold:.4f} uploaded={uploaded_count} skipped={skipped_count} "
                    f"| best={best_val:.4f} | patience={patience_count}/{opts.patience} | early_stop=True",
                    flush=True,
                )
                break
        print(
            f"[STFAM centralized-upper] Epoch {epoch} | Train Loss(Norm): "
            f"{float(np.mean(train_losses)):.4f} | Val MAE(Norm): {val_mae:.4f} "
            f"| threshold={threshold:.4f} uploaded={uploaded_count} skipped={skipped_count} "
            f"| best={best_val:.4f} | patience={patience_count}/{opts.patience}",
            flush=True,
        )

    if best_states is not None:
        for model, state in zip(models, best_states):
            model.load_state_dict(state)
    return _stfam_upper_test_metrics(models, test_loaders, scalers, semantic_contexts, device)


def train_ufcl_standalone(setting, opts):
    from lib.ufcl_trainer import UFCLTrainer

    train_set, val_set, test_set, model, optimizer, loss_func, _, train_args, _, scaler, _ = setting
    device = args.device
    model.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if train_args is None:
        raise RuntimeError("UFCL requires TrainingArguments from fate_main.get_setting().")

    def compute_metrics(_):
        return {}

    trainer = UFCLTrainer(
        model=model,
        args=train_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
        ufcl_max_nodes=getattr(train_args, "ufcl_max_nodes", None),
        use_replay_mixup=ufcl_ablation_mode() not in ("no_replay", "task_only"),
        use_synthetic_replay=ufcl_ablation_mode() not in ("no_replay", "task_only"),
        use_teacher_kd=ufcl_ablation_mode() not in ("no_kd", "task_only"),
        noise_std=getattr(args, "ufcl_noise_std", 0.05),
        kd_weight=getattr(args, "ufcl_kd_weight", 1.0),
    )
    trainer.model.to(device)
    model = trainer.model
    print(
        f"[UFCLTrainer standalone] forced_model_device={next(model.parameters()).device} "
        f"ablation={ufcl_ablation_mode()} "
        f"replay_mixup={trainer.use_replay_mixup} "
        f"synthetic_replay={trainer.use_synthetic_replay} "
        f"teacher_kd={trainer.use_teacher_kd} "
        f"kd_weight={trainer.kd_weight} "
        f"noise_std={trainer.noise_std}",
        flush=True,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    best_state = None
    best_val = float("inf")
    patience_count = 0

    for epoch in range(args.epochs):
        if trainer.needs_teacher:
            trainer.teacher_model = copy.deepcopy(model).to(device)
            trainer.teacher_model.eval()
            for p in trainer.teacher_model.parameters():
                p.requires_grad_(False)
        else:
            trainer.teacher_model = None

        model.train()
        total_loss = 0.0
        steps = 0
        last_stats = {}
        for batch_idx, batch in enumerate(train_loader):
            if opts.max_batches and batch_idx >= opts.max_batches:
                break
            batch = _move_batch(batch, device)
            optimizer.zero_grad()
            loss = trainer.compute_loss(model, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
            last_stats = getattr(trainer, "last_ufcl_stats", {})

        val_mae = normalized_mae(model, val_loader, device)
        stats_msg = ""
        if last_stats:
            mix_lam = last_stats.get("mix_lambda")
            mix_lam = "None" if mix_lam is None else f"{mix_lam:.4f}"
            stats_msg = (
                f" | replay={last_stats.get('used_replay')} "
                f"buffer={last_stats.get('buffer_before')}->{last_stats.get('buffer_after')} "
                f"lambda={mix_lam} task={last_stats.get('task_loss', 0.0):.4f} "
                f"kd={last_stats.get('kd_loss', 0.0):.4f} "
                f"kd_w={last_stats.get('kd_weight', 0.0):.4f}"
            )
        print(
            f"[UFCLTrainer standalone] Epoch {epoch} | Train Loss(Norm): "
            f"{total_loss / max(steps, 1):.4f} | Val MAE(Norm): {val_mae:.4f}"
            f"{stats_msg}"
        )
        if val_mae < best_val - opts.min_delta:
            best_val = val_mae
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= opts.patience:
                print(f"[UFCLTrainer standalone] Early stop at epoch {epoch}, best_val={best_val:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return test_metrics(model, test_loader, scaler, device)


def _fedtps_forward(model, batch, training=False):
    x, y, _ = _unpack_xy(batch)
    y_cov = batch[2] if isinstance(batch, (tuple, list)) and len(batch) > 2 else None
    pred = model(x, y_cov=y_cov, labels=y if training else None)
    return pred, y


def train_fedtps_standalone(setting, opts):
    train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = setting
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    return _standard_train_loop(
        model,
        optimizer,
        loss_func,
        train_loader,
        val_loader,
        test_loader,
        scaler,
        opts,
        _fedtps_forward,
        "FedTPS no-communication",
    )


def train_fed4tp_standalone(setting, opts):
    from lib.fed4tp_utils import slice_data_for_twt

    train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = setting
    device = args.device
    model.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if loss_func is None:
        loss_func = torch.nn.MSELoss().to(device)
    elif hasattr(loss_func, "to"):
        loss_func = loss_func.to(device)

    twt_data_list = slice_data_for_twt(train_set, args.time_window_num)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    best_state = None
    best_val = float("inf")
    patience_count = 0
    global_epoch = 0
    stop = False

    for window_idx in range(args.time_window_num):
        current_dataset = twt_data_list[window_idx]
        if len(current_dataset) == 0:
            for offset in range(1, len(twt_data_list)):
                right, left = window_idx + offset, window_idx - offset
                if right < len(twt_data_list) and len(twt_data_list[right]) > 0:
                    current_dataset = twt_data_list[right]
                    break
                if left >= 0 and len(twt_data_list[left]) > 0:
                    current_dataset = twt_data_list[left]
                    break
        train_loader = DataLoader(current_dataset, batch_size=args.batch_size, shuffle=True)

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            steps = 0
            for batch_idx, batch in enumerate(train_loader):
                if opts.max_batches and batch_idx >= opts.max_batches:
                    break
                batch = _move_batch(batch, device)
                x, y, _ = _unpack_xy(batch)
                optimizer.zero_grad()
                pred = _as_pred(model(x))
                pred, y = align_pred_and_target(pred, y)
                loss = loss_func(pred, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1

            val_mae = normalized_mae(model, val_loader, device)
            print(
                f"[Fed4TP TWT no-communication] Window {window_idx} Epoch {epoch} "
                f"(global {global_epoch}) | Train Loss(Norm): "
                f"{total_loss / max(steps, 1):.4f} | Val MAE(Norm): {val_mae:.4f}"
            )
            if val_mae < best_val - opts.min_delta:
                best_val = val_mae
                best_state = copy.deepcopy(model.state_dict())
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= opts.patience:
                    print(f"[Fed4TP TWT no-communication] Early stop at global epoch {global_epoch}")
                    stop = True
                    break
            global_epoch += 1
        if stop:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return test_metrics(model, test_loader, scaler, device)


def _fgnneh_forward(model, batch, training=False):
    x, y, _ = _unpack_xy(batch)
    adj = getattr(model.backbone_extractor, "adj", None)
    if adj is None:
        adj = model.backbone_extractor.adj_matrix
    adj = adj.to(x.device)
    h, h_backbone = model.forward_local(x, adj)
    hyper_node = model.generate_hypernode(h_backbone)
    pred = model.forward_predict(h, hyper_node)
    return pred, y


def train_fgnneh_standalone(setting, opts):
    train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = setting
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    return _standard_train_loop(
        model,
        optimizer,
        loss_func,
        train_loader,
        val_loader,
        test_loader,
        scaler,
        opts,
        _fgnneh_forward,
        "FGNNEH local-hypernode",
    )


def _fedmetro_forward(model, batch, training=False):
    x, y, _ = _unpack_xy(batch)
    if getattr(args, "fedmetro_standalone_ablation", "full") == "bias_only":
        x_seq = model._normalize_input_shape(x)
        B, _, N, _ = x_seq.shape
        h = torch.zeros(B, N, model.hidden_dim, device=x.device, dtype=x.dtype)
        return model.out_layer(h), y
    agg_seq, f_e_seq, e_seq = model.forward_phase1(x)
    if getattr(args, "fedmetro_standalone_ablation", "full") == "no_agg":
        agg_seq = torch.zeros_like(agg_seq)
    pred = model.forward_phase2(x, f_e_seq, e_seq, agg_seq)
    return pred, y


def train_fedmetro_standalone(setting, opts):
    train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = setting
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    ablation = getattr(args, "fedmetro_standalone_ablation", "full")
    def metro_reg_loss(model):
        if ablation == "no_mask_reg":
            return torch.tensor(0.0, device=args.device)
        mask = getattr(getattr(model, "dyn_emb_mask", None), "mask_generator", None)
        last_m = getattr(mask, "last_m", None)
        if last_m is None:
            return torch.tensor(0.0, device=args.device)
        return getattr(args, "lambda_reg", 0.001) * torch.mean(last_m)

    metrics = _standard_train_loop(
        model,
        optimizer,
        loss_func,
        train_loader,
        val_loader,
        test_loader,
        scaler,
        opts,
        _fedmetro_forward,
        "FedMetro local-AGG",
        metro_reg_loss,
        grad_clip_norm=5.0,
        stop_on_nonfinite=True,
        restore_best=(ablation != "final_ckpt"),
    )
    return metrics


def train_pfedctp_standalone(setting, opts):
    print("[pFedCTP no-communication] Stage-1 shared-model aggregation is disabled; running local supervised pFedCTP.")
    return train_common_standalone(setting, opts)


def train_tistgnn_standalone(setting, opts):
    print("[T-ISTGNN no-communication] Cross-client source/target transfer is disabled; running local ISTGNN.")
    return train_common_standalone(setting, opts)


def _patch_fcfedgcn_adj_for_paradigm(setting, rank):
    import pandas as pd

    _, _, _, model, _, _, _, _, _, _, _ = setting
    if model is None or not hasattr(model, "adj"):
        raise RuntimeError("FC-FedGCN standalone expected a model with an adj buffer.")

    selected_nodes = args.nodes_per[rank]
    num_local_nodes = len(selected_nodes)
    dist_file = os.path.join(PROJECT_ROOT, "data", args.dataset_name, "distance.csv")
    if not os.path.exists(dist_file):
        print(
            f"[FC-FedGCN standalone adj patch] distance.csv not found at {dist_file}; "
            "keeping get_setting() fallback adjacency.",
            flush=True,
        )
        return

    dist_df = pd.read_csv(dist_file)
    is_already_weight = dist_df["cost"].max() <= 1.0
    if is_already_weight:
        k = 8
        topk_dist = dist_df.sort_values(["from", "cost"], ascending=[True, False]).groupby("from").head(k)
        rev_dist = topk_dist.copy()
        rev_dist["from"], rev_dist["to"] = topk_dist["to"], topk_dist["from"]
        dist_df = pd.concat([topk_dist, rev_dist], ignore_index=True).drop_duplicates(subset=["from", "to"])

    selected_nodes_set = set(selected_nodes)
    relevant_dists = dist_df[
        dist_df["from"].isin(selected_nodes_set) & dist_df["to"].isin(selected_nodes_set)
    ]

    device = model.adj.device
    node_to_idx = {global_id: i for i, global_id in enumerate(selected_nodes)}
    local_adj = torch.zeros((num_local_nodes, num_local_nodes), device=device)
    for _, row in relevant_dists.iterrows():
        u, v, cost = int(row["from"]), int(row["to"]), float(row["cost"])
        if u == v:
            continue
        if cost > 0 and u in node_to_idx and v in node_to_idx:
            idx_u, idx_v = node_to_idx[u], node_to_idx[v]
            weight = cost if is_already_weight else 1.0 / cost
            local_adj[idx_u, idx_v] = weight
            local_adj[idx_v, idx_u] = weight

    deg_before_loop = local_adj.sum(dim=1)
    zero_deg = int((deg_before_loop == 0).sum().item())
    edge_entries = int((local_adj > 0).sum().item())
    local_adj.diagonal().add_(1.0)
    row_sum = local_adj.sum(dim=1)
    d_inv_sqrt = torch.pow(row_sum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    a_hat = d_inv_sqrt.view(-1, 1) * local_adj * d_inv_sqrt.view(1, -1)

    model.adj.data.copy_(a_hat.to(model.adj.device, dtype=model.adj.dtype))
    print(
        f"[FC-FedGCN standalone adj patch] rank={rank} nodes={num_local_nodes} "
        f"relevant_edges={len(relevant_dists)} nonzero_adj_entries={edge_entries} "
        f"zero_degree_before_self_loop={zero_deg} "
        f"deg_min={float(deg_before_loop.min().item()):.4f} "
        f"deg_max={float(deg_before_loop.max().item()):.4f} "
        f"deg_mean={float(deg_before_loop.mean().item()):.4f}",
        flush=True,
    )


def train_fcfedgcn_standalone(setting, opts):
    rank = getattr(opts, "current_rank", 0)
    _patch_fcfedgcn_adj_for_paradigm(setting, rank)
    print("[FC-FedGCN standalone] Patched adjacency for training-paradigm run only; fate_main.py is unchanged.")
    return train_common_standalone(setting, opts)


def train_fedostc_standalone(setting, opts):
    print("[FedOSTC no-communication] Running encoder + local/global graph GAT + decoder without server hidden-state exchange.")
    return train_common_standalone(setting, opts)


def _fedostc_sim_load_parts(device):
    from lib.load_dataset import load_dataset
    from model.FedOSTC import FedOSTC

    train_sets, val_sets, test_sets, scalers, client_models = [], [], [], [], []
    normalized = "PeMSD7" if args.dataset_name == "PeMS07" else args.dataset_name
    for rank, selected_nodes in enumerate(args.nodes_per):
        train_set, val_set, test_set, _, scaler = load_dataset(
            dataset_name=normalized,
            feature_type=args.feature_type,
            normalizer=args.normalizer,
            T_in=args.t_in,
            T_out=args.t_out,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            return_edge_index=True,
            device=device,
            selected_nodes=selected_nodes,
        )
        init_seed(args.seed)
        model = FedOSTC(
            enc_dim=args.hidden_dim,
            gat_dim=args.hidden_dim,
            pred_steps=args.t_out,
        ).to(device)
        train_sets.append(train_set)
        val_sets.append(val_set)
        test_sets.append(test_set)
        scalers.append(scaler)
        client_models.append(model)
        print(
            f"[FedOSTC centralized-sim] client{rank} nodes={len(selected_nodes)} "
            f"train={len(train_set)} val={len(val_set)} test={len(test_set)}",
            flush=True,
        )

    all_nodes = list(range(total_nodes_for(normalized)))
    _, _, _, global_edge_index, _ = load_dataset(
        dataset_name=normalized,
        feature_type=args.feature_type,
        normalizer=args.normalizer,
        T_in=args.t_in,
        T_out=args.t_out,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        return_edge_index=True,
        device=device,
        selected_nodes=all_nodes,
    )
    server_model = FedOSTC(
        enc_dim=args.hidden_dim,
        gat_dim=args.hidden_dim,
        pred_steps=args.t_out,
    ).to(device)
    global_edge_index = global_edge_index.to(device)
    print(
        f"[FedOSTC centralized-sim] server_global_nodes={len(all_nodes)} "
        f"global_edges={global_edge_index.shape[1] if global_edge_index is not None else 0}",
        flush=True,
    )
    return train_sets, val_sets, test_sets, scalers, client_models, server_model, global_edge_index


def _fedostc_sim_batches(datasets, batch_size, shuffle, seed):
    length = min(len(dataset) for dataset in datasets)
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        indices = torch.randperm(length, generator=generator).tolist()
    else:
        indices = list(range(length))
    for start in range(0, length, batch_size):
        batch_indices = indices[start:start + batch_size]
        yield [
            default_collate([dataset[idx] for idx in batch_indices])
            for dataset in datasets
        ]


def _fedostc_sim_forward(client_models, server_model, global_edge_index, batch_parts, device):
    h_parts, y_parts = [], []
    for model, batch in zip(client_models, batch_parts):
        batch = _move_batch(batch, device)
        x, y, _ = _unpack_xy(batch)
        h_parts.append(model.forward_encoder(x))
        y_parts.append(y)

    split_sizes = [h_time.shape[1] for h_time in h_parts]
    h_global = torch.cat(h_parts, dim=1)
    h_spatio_global = server_model.forward_server_gat(h_global, global_edge_index)
    h_spatio_parts = list(torch.split(h_spatio_global, split_sizes, dim=1))

    pred_parts, aligned_y_parts = [], []
    for model, h_spatio, y in zip(client_models, h_spatio_parts, y_parts):
        pred = model.forward_decoder(h_spatio)
        pred, y = align_pred_and_target(pred, y)
        pred_parts.append(pred)
        aligned_y_parts.append(y)
    return pred_parts, aligned_y_parts


def _fedostc_sim_normalized_mae(client_models, server_model, global_edge_index, datasets, device):
    for model in client_models:
        model.eval()
    server_model.eval()
    total_abs = 0.0
    total_elements = 0
    with torch.no_grad():
        for batch_parts in _fedostc_sim_batches(datasets, args.batch_size, shuffle=False, seed=args.seed):
            pred_parts, y_parts = _fedostc_sim_forward(client_models, server_model, global_edge_index, batch_parts, device)
            for pred, y in zip(pred_parts, y_parts):
                total_abs += torch.abs(pred - y).sum().item()
                total_elements += y.numel()
    return total_abs / max(total_elements, 1)


def _fedostc_sim_test_metrics(client_models, server_model, global_edge_index, datasets, scalers, device):
    for model in client_models:
        model.eval()
    server_model.eval()
    total_abs = 0.0
    total_sq = 0.0
    total_elements = 0
    total_mape = 0.0
    total_mape_elements = 0

    with torch.no_grad():
        for batch_parts in _fedostc_sim_batches(datasets, args.batch_size, shuffle=False, seed=args.seed):
            pred_parts, y_parts = _fedostc_sim_forward(client_models, server_model, global_edge_index, batch_parts, device)
            for pred, y, scaler in zip(pred_parts, y_parts, scalers):
                y_real = scaler.inverse_transform(y).detach().cpu().numpy()
                pred_real = scaler.inverse_transform(pred).detach().cpu().numpy()
                diff = pred_real - y_real
                total_abs += np.abs(diff).sum()
                total_sq += np.square(diff).sum()
                total_elements += y_real.size
                mask = y_real > 0.5
                if np.any(mask):
                    total_mape += (np.abs(diff[mask]) / y_real[mask]).sum()
                    total_mape_elements += int(mask.sum())

    mse = total_sq / max(total_elements, 1)
    return {
        "mae": round(float(total_abs / max(total_elements, 1)), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(math.sqrt(mse)), 4),
        "mape": round(float(total_mape / total_mape_elements), 4) if total_mape_elements else 0.0,
    }


def train_fedostc_centralized_sim(setting, opts):
    print(
        "[FedOSTC centralized-sim] Running four client encoder/decoder experts "
        "with single-process server hidden-state concat/GAT/split; network communication is disabled.",
        flush=True,
    )
    device = args.device
    train_sets, val_sets, test_sets, scalers, client_models, server_model, global_edge_index = _fedostc_sim_load_parts(device)
    loss_func = torch.nn.MSELoss().to(device)
    all_params = [param for model in client_models for param in model.parameters()]
    all_params += list(server_model.gat.parameters())
    optimizer = torch.optim.Adam(all_params, lr=args.lr, weight_decay=args.wd)

    train_steps = math.ceil(min(len(dataset) for dataset in train_sets) / args.batch_size)
    val_steps = math.ceil(min(len(dataset) for dataset in val_sets) / args.batch_size)
    test_steps = math.ceil(min(len(dataset) for dataset in test_sets) / args.batch_size)
    print(
        f"[FedOSTC centralized-sim] train_batches={train_steps} "
        f"val_batches={val_steps} test_batches={test_steps} batch_size={args.batch_size}",
        flush=True,
    )

    best_client_states = None
    best_server_state = None
    best_val = float("inf")
    patience_count = 0

    for epoch in range(args.epochs):
        for model in client_models:
            model.train()
        server_model.train()
        total_loss = 0.0
        steps = 0

        for batch_idx, batch_parts in enumerate(
            _fedostc_sim_batches(train_sets, args.batch_size, shuffle=True, seed=args.seed + epoch)
        ):
            if opts.max_batches and batch_idx >= opts.max_batches:
                break
            optimizer.zero_grad()
            pred_parts, y_parts = _fedostc_sim_forward(client_models, server_model, global_edge_index, batch_parts, device)
            loss = sum(loss_func(pred, y) for pred, y in zip(pred_parts, y_parts)) / len(pred_parts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        val_mae = _fedostc_sim_normalized_mae(client_models, server_model, global_edge_index, val_sets, device)
        if val_mae < best_val - opts.min_delta:
            best_val = val_mae
            best_client_states = [copy.deepcopy(model.state_dict()) for model in client_models]
            best_server_state = copy.deepcopy(server_model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= opts.patience:
                print(
                    f"[FedOSTC centralized-sim] Epoch {epoch} | Train Loss(Norm): "
                    f"{total_loss / max(steps, 1):.4f} | Val MAE(Norm): {val_mae:.4f} "
                    f"| best={best_val:.4f} | patience={patience_count}/{opts.patience} | early_stop=True",
                    flush=True,
                )
                break
        print(
            f"[FedOSTC centralized-sim] Epoch {epoch} | Train Loss(Norm): "
            f"{total_loss / max(steps, 1):.4f} | Val MAE(Norm): {val_mae:.4f} "
            f"| best={best_val:.4f} | patience={patience_count}/{opts.patience}",
            flush=True,
        )

    if best_client_states is not None:
        for model, state in zip(client_models, best_client_states):
            model.load_state_dict(state)
    if best_server_state is not None:
        server_model.load_state_dict(best_server_state)
    return _fedostc_sim_test_metrics(client_models, server_model, global_edge_index, test_sets, scalers, device)


def _fedgtp_sim_load_parts(device):
    from lib.load_dataset import load_dataset
    from model.FedGTP import FedGTP_Model

    normalized = "PeMSD7" if args.dataset_name == "PeMS07" else args.dataset_name
    max_nodes = max(len(nodes) for nodes in args.nodes_per)
    train_sets, val_sets, test_sets, scalers, models, optimizers = [], [], [], [], [], []
    for rank, selected_nodes in enumerate(args.nodes_per):
        train_set, val_set, test_set, _, scaler = load_dataset(
            dataset_name=normalized,
            feature_type=args.feature_type,
            normalizer=args.normalizer,
            T_in=args.t_in,
            T_out=args.t_out,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            return_edge_index=True,
            device=device,
            selected_nodes=selected_nodes,
            # Each expert's EH is aggregated with the others, so global mode
            # must honour a shared full-graph scaler when requested.
            norm_scope=getattr(args, "norm_scope", "global"),
            scaler_fit_scope=getattr(args, "scaler_fit_scope", "selected"),
        )
        init_seed(args.seed)
        model = FedGTP_Model(
            num_nodes=len(selected_nodes),
            max_nodes=max_nodes,
            in_dim=args.t_in,
            out_dim=args.t_out,
            feature_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            emb_dim=args.node_emb_dim,
            poly_k=args.poly_k,
        ).to(device)
        if hasattr(model, "set_debug"):
            model.set_debug(enabled=False, full=False, flush=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        train_sets.append(train_set)
        val_sets.append(val_set)
        test_sets.append(test_set)
        scalers.append(scaler)
        models.append(model)
        optimizers.append(optimizer)
        print(
            f"[FedGTP centralized-sim] client{rank} nodes={len(selected_nodes)} "
            f"max_nodes={max_nodes} train={len(train_set)} val={len(val_set)} test={len(test_set)}",
            flush=True,
        )
    return train_sets, val_sets, test_sets, scalers, models, optimizers


def _fedgtp_sim_batches(datasets, batch_size, shuffle, seed):
    length = min(len(dataset) for dataset in datasets)
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        indices = torch.randperm(length, generator=generator).tolist()
    else:
        indices = list(range(length))
    for start in range(0, length, batch_size):
        batch_indices = indices[start:start + batch_size]
        yield [
            default_collate([dataset[idx] for idx in batch_indices])
            for dataset in datasets
        ]


def _fedgtp_prepare_source(model, x):
    if x.shape[1] == model.num_nodes:
        return x.permute(0, 2, 1, 3).contiguous()
    return x.contiguous()


def _fedgtp_transformed_embeddings(model):
    real_embeddings = model.node_embeddings[:model.num_nodes, :]
    return [
        model.encoder.dcrnn_cells[0].gate.transform(k, real_embeddings)
        for k in range(model.poly_k + 1)
    ]


def _fedgtp_avwgcn_comm(
    avw_parts,
    h_parts,
    embeddings,
    transformed_parts,
    coeffs,
    poly_k,
    client_node_counts,
):
    """Reproduce FedGTP's server-side node-count-weighted EH aggregation.

    FedGTP communicates ``EH.detach().to(float16).cpu()``.  In particular, the
    server reply is a *message*, not a differentiable all-client operation.  Do
    not keep ``EH`` in this process's autograd graph merely because the four
    experts happen to be evaluated in one Python process.
    """
    eh_parts = [
        [
            # Match AVWGCN.forward: detach before transport and use float16
            # payloads.  The reply is cast back to H.dtype below.
            torch.einsum("dn,bnc->bdc", transformed.transpose(0, 1), h)
            .detach()
            .to(torch.float16)
            .contiguous()
            for transformed in transformed_list
        ]
        for h, transformed_list in zip(h_parts, transformed_parts)
    ]
    total_nodes = sum(client_node_counts)
    if total_nodes <= 0:
        raise ValueError("FedGTP requires non-empty client node partitions.")

    weighted_mean_eh = []
    for k in range(poly_k + 1):
        # fate_main.py aggregates float16 client payloads in float32 then
        # broadcasts a contiguous float16 payload to every client.
        aggregate = sum(
            client_eh[k].float() * (node_count / total_nodes)
            for client_eh, node_count in zip(eh_parts, client_node_counts)
        )
        weighted_mean_eh.append(aggregate.to(torch.float16).contiguous())

    outputs = []
    for avw, h, embedding, transformed_list, coeff in zip(avw_parts, h_parts, embeddings, transformed_parts, coeffs):
        z_list = [
            torch.einsum(
                "nd,bdc->bnc",
                transformed_list[k],
                weighted_mean_eh[k].to(h.device, dtype=h.dtype),
            )
            for k in range(poly_k + 1)
        ]
        z_stack = torch.stack(z_list)
        z_poly = torch.einsum("ak,kbnc->abnc", coeff, z_stack)[0]
        z = h + z_poly

        weights = torch.einsum("nd,dio->nio", embedding, avw.weights_pool)
        bias = torch.matmul(embedding, avw.bias_pool)
        outputs.append(torch.einsum("bni,nio->bno", z, weights) + bias)
    return outputs


def _fedgtp_sim_forward(models, batch_parts, device):
    x_parts, y_parts = [], []
    for model, batch in zip(models, batch_parts):
        batch = _move_batch(batch, device)
        x, y, _ = _unpack_xy(batch)
        x_parts.append(_fedgtp_prepare_source(model, x))
        y_parts.append(y)

    batch_size = x_parts[0].shape[0]
    states = [
        [state.to(device) for state in model.encoder.init_hidden(batch_size)]
        for model in models
    ]
    embeddings = [model.node_embeddings[:model.num_nodes, :] for model in models]
    transformed_parts = [_fedgtp_transformed_embeddings(model) for model in models]
    coeffs = [model.poly_coefficients for model in models]
    client_node_counts = [model.num_nodes for model in models]

    current_inputs = x_parts
    num_layers = models[0].encoder.num_layers
    poly_k = models[0].poly_k
    for layer_idx in range(num_layers):
        cells = [model.encoder.dcrnn_cells[layer_idx] for model in models]
        state_parts = [client_states[layer_idx] for client_states in states]
        inner_states = [[] for _ in models]
        seq_length = current_inputs[0].shape[1]
        for t in range(seq_length):
            inputs_t = [current[:, t, :, :] for current in current_inputs]
            input_state = [
                torch.cat((x_t, state), dim=-1)
                for x_t, state in zip(inputs_t, state_parts)
            ]
            zr_parts = _fedgtp_avwgcn_comm(
                [cell.gate for cell in cells],
                input_state,
                embeddings,
                transformed_parts,
                coeffs,
                poly_k,
                client_node_counts,
            )
            z_parts, r_parts = zip(*[
                torch.split(zr, cells[idx].hidden_dim, dim=-1)
                for idx, zr in enumerate(zr_parts)
            ])
            z_parts = [torch.sigmoid(z) for z in z_parts]
            r_parts = [torch.sigmoid(r) for r in r_parts]
            candidates = [
                torch.cat((x_t, z * state), dim=-1)
                for x_t, z, state in zip(inputs_t, z_parts, state_parts)
            ]
            hc_parts = _fedgtp_avwgcn_comm(
                [cell.update for cell in cells],
                candidates,
                embeddings,
                transformed_parts,
                coeffs,
                poly_k,
                client_node_counts,
            )
            hc_parts = [torch.tanh(hc) for hc in hc_parts]
            state_parts = [
                r * state + (1 - r) * hc
                for r, state, hc in zip(r_parts, state_parts, hc_parts)
            ]
            for idx, state in enumerate(state_parts):
                inner_states[idx].append(state)
        current_inputs = [torch.stack(states_i, dim=1) for states_i in inner_states]

    pred_parts, aligned_y_parts = [], []
    for model, output, y in zip(models, current_inputs, y_parts):
        output = output[:, -1:, :, :]
        pred = model.end_conv(output)
        pred = pred.squeeze(-1).reshape(-1, model.horizon, model.output_dim, model.num_nodes)
        pred = pred.permute(0, 1, 3, 2)
        pred, y = align_pred_and_target(pred, y)
        pred_parts.append(pred)
        aligned_y_parts.append(y)
    return pred_parts, aligned_y_parts


def _fedgtp_shared_state(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if "node_embeddings" not in key
    }


def _fedgtp_weighted_partial_fedavg(models, apply_to_models=True):
    payloads = [
        {"num_nodes": model.num_nodes, "weights": _fedgtp_shared_state(model)}
        for model in models
    ]
    total_nodes = sum(payload["num_nodes"] for payload in payloads)
    first_weights = payloads[0]["weights"]
    global_weights = {}
    for key, first_tensor in first_weights.items():
        if torch.is_tensor(first_tensor) and torch.is_floating_point(first_tensor):
            agg = None
            for payload in payloads:
                coef = payload["num_nodes"] / total_nodes
                contrib = payload["weights"][key].float() * coef
                agg = contrib if agg is None else agg + contrib
            global_weights[key] = agg.to(first_tensor.dtype).contiguous()
        else:
            global_weights[key] = first_tensor

    if apply_to_models:
        for model in models:
            if hasattr(model, "load_shared_params"):
                global_on_device = {
                    key: value.to(next(model.parameters()).device) if torch.is_tensor(value) else value
                    for key, value in global_weights.items()
                }
                model.load_shared_params(global_on_device)
            else:
                model.load_state_dict(global_weights, strict=False)
    return global_weights


def _fedgtp_load_shared_weights(models, global_weights):
    if global_weights is None:
        return
    for model in models:
        global_on_device = {
            key: value.to(next(model.parameters()).device) if torch.is_tensor(value) else value
            for key, value in global_weights.items()
        }
        if hasattr(model, "load_shared_params"):
            model.load_shared_params(global_on_device)
        else:
            model.load_state_dict(global_on_device, strict=False)


def _fedgtp_sim_normalized_mae(models, datasets, device):
    for model in models:
        model.eval()
    total_abs = 0.0
    total_elements = 0
    with torch.no_grad():
        for batch_parts in _fedgtp_sim_batches(datasets, args.batch_size, shuffle=False, seed=args.seed):
            pred_parts, y_parts = _fedgtp_sim_forward(models, batch_parts, device)
            for pred, y in zip(pred_parts, y_parts):
                total_abs += torch.abs(pred - y).sum().item()
                total_elements += y.numel()
    return total_abs / max(total_elements, 1)


def _fedgtp_sim_test_metrics(models, datasets, scalers, device):
    for model in models:
        model.eval()
    total_abs = 0.0
    total_sq = 0.0
    total_elements = 0
    total_mape = 0.0
    total_mape_elements = 0
    with torch.no_grad():
        for batch_parts in _fedgtp_sim_batches(datasets, args.batch_size, shuffle=False, seed=args.seed):
            pred_parts, y_parts = _fedgtp_sim_forward(models, batch_parts, device)
            for pred, y, scaler in zip(pred_parts, y_parts, scalers):
                y_real = scaler.inverse_transform(y).detach().cpu().numpy()
                pred_real = scaler.inverse_transform(pred).detach().cpu().numpy()
                diff = pred_real - y_real
                total_abs += np.abs(diff).sum()
                total_sq += np.square(diff).sum()
                total_elements += y_real.size
                mask = y_real > 0.5
                if np.any(mask):
                    total_mape += (np.abs(diff[mask]) / y_real[mask]).sum()
                    total_mape_elements += int(mask.sum())
    mse = total_sq / max(total_elements, 1)
    return {
        "mae": round(float(total_abs / max(total_elements, 1)), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(math.sqrt(mse)), 4),
        "mape": round(float(total_mape / total_mape_elements), 4) if total_mape_elements else 0.0,
    }


def train_fedgtp_centralized_sim(setting, opts):
    print(
        "[FedGTP centralized-sim] Running four FedGTP client experts with "
        "lockstep server EH summation and weighted PartialFedAvg in one process.",
        flush=True,
    )
    device = args.device
    train_sets, val_sets, test_sets, scalers, models, optimizers = _fedgtp_sim_load_parts(device)
    loss_func = torch.nn.MSELoss().to(device)
    train_steps = math.ceil(min(len(dataset) for dataset in train_sets) / args.batch_size)
    val_steps = math.ceil(min(len(dataset) for dataset in val_sets) / args.batch_size)
    test_steps = math.ceil(min(len(dataset) for dataset in test_sets) / args.batch_size)
    val_every = int(os.environ.get("FEDGTP_SIM_VAL_EVERY", "5"))
    client_node_counts = [model.num_nodes for model in models]
    print(
        f"[FedGTP centralized-sim] train_batches={train_steps} val_batches={val_steps} "
        f"test_batches={test_steps} batch_size={args.batch_size} val_every={val_every} "
        "EH_aggregation=node_count_weighted_mean "
        "EH_transport=float16_detached client_nodes="
        f"{client_node_counts}",
        flush=True,
    )

    best_states = None
    best_val = float("inf")
    patience_count = 0
    pending_global_weights = None

    for epoch in range(args.epochs):
        if epoch > 0:
            _fedgtp_load_shared_weights(models, pending_global_weights)
        for model in models:
            model.train()
        total_loss = 0.0
        steps = 0
        for batch_idx, batch_parts in enumerate(
            _fedgtp_sim_batches(train_sets, args.batch_size, shuffle=True, seed=args.seed + epoch)
        ):
            if opts.max_batches and batch_idx >= opts.max_batches:
                break
            for optimizer in optimizers:
                optimizer.zero_grad()
            pred_parts, y_parts = _fedgtp_sim_forward(models, batch_parts, device)
            loss = sum(loss_func(pred, y) for pred, y in zip(pred_parts, y_parts)) / len(pred_parts)
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            total_loss += loss.item()
            steps += 1

        pending_global_weights = _fedgtp_weighted_partial_fedavg(models, apply_to_models=False)
        do_val = ((epoch + 1) % val_every == 0) or (epoch == args.epochs - 1)
        if not do_val:
            print(
                f"[FedGTP centralized-sim] Epoch {epoch} | Train Loss(Norm): "
                f"{total_loss / max(steps, 1):.4f} | Val MAE(Norm): skipped",
                flush=True,
            )
            continue

        val_mae = _fedgtp_sim_normalized_mae(models, val_sets, device)
        if val_mae < best_val - opts.min_delta:
            best_val = val_mae
            best_states = [copy.deepcopy(model.state_dict()) for model in models]
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= opts.patience:
                print(
                    f"[FedGTP centralized-sim] Epoch {epoch} | Train Loss(Norm): "
                    f"{total_loss / max(steps, 1):.4f} | Val MAE(Norm): {val_mae:.4f} "
                    f"| best={best_val:.4f} | patience={patience_count}/{opts.patience} | early_stop=True",
                    flush=True,
                )
                break
        print(
            f"[FedGTP centralized-sim] Epoch {epoch} | Train Loss(Norm): "
            f"{total_loss / max(steps, 1):.4f} | Val MAE(Norm): {val_mae:.4f} "
            f"| best={best_val:.4f} | patience={patience_count}/{opts.patience}",
            flush=True,
        )

    if best_states is not None:
        for model, state in zip(models, best_states):
            model.load_state_dict(state, strict=False)
    return _fedgtp_sim_test_metrics(models, test_sets, scalers, device)


def train_refol_standalone(setting, opts):
    print("[REFOL no-communication] Running REFOL GRU predictor without concept-drift client selection or server graph aggregation.")
    return train_common_standalone(setting, opts)


def train_cnfgnn_standalone(setting, opts):
    print("[CNFGNN no-communication] Running client encoder, standalone GraphNet spatial pass, and decoder locally.")
    return train_common_standalone(setting, opts)


def train_fedstg_standalone(setting, opts):
    print("[FedSTG no-communication] Running TP-Bank client path with zero local h_G; server graph fusion is disabled.")
    return train_common_standalone(setting, opts)


def train_fedgode_standalone(setting, opts):
    print("[FedGODE no-communication] Running ODEGCN local supervised loop without federated parameter exchange.")
    return train_common_standalone(setting, opts)


def _sfl_client_weights(device):
    node_counts = [len(nodes) for nodes in getattr(args, "nodes_per", []) if len(nodes) > 0]
    if not node_counts:
        node_counts = [1] * max(int(getattr(args, "num_clients", 1)), 1)
    weights = torch.tensor(node_counts, dtype=torch.float32, device=device)
    weights = weights / weights.sum().clamp_min(1.0)
    return weights


def _sfl_shared_param_keys(state_dicts):
    if not state_dicts:
        return []
    shared = []
    first = state_dicts[0]
    for name, tensor in first.items():
        if all(name in state and state[name].shape == tensor.shape for state in state_dicts[1:]):
            shared.append(name)
    return shared


def _sfl_flatten_state(state_dict, keys, device):
    return torch.cat([state_dict[name].detach().to(device).reshape(-1) for name in keys], dim=0)


def _sfl_parameter_mse(model, reference_state):
    total_sq = None
    total_numel = 0
    for name, param in model.named_parameters():
        ref = reference_state.get(name)
        if ref is None or ref.shape != param.shape:
            continue
        diff = param - ref.detach().to(param.device)
        sq = diff.pow(2).sum()
        total_sq = sq if total_sq is None else total_sq + sq
        total_numel += diff.numel()
    if total_sq is None or total_numel <= 0:
        return torch.zeros((), device=next(model.parameters()).device)
    return total_sq / float(total_numel)


def _sfl_structure_aggregate(local_states, device):
    num_clients = len(local_states)
    if num_clients == 0:
        raise ValueError("SFL aggregation requires at least one client state.")

    param_keys = _sfl_shared_param_keys(local_states)
    if not param_keys:
        raise ValueError("SFL aggregation found no shared trainable parameters across clients.")

    flat_states = torch.stack(
        [_sfl_flatten_state(state, param_keys, device) for state in local_states],
        dim=0,
    )

    if num_clients == 1:
        adjacency = torch.ones((1, 1), device=device)
        personalized = flat_states.clone()
    else:
        sim = torch.matmul(F.normalize(flat_states, p=2, dim=1), F.normalize(flat_states, p=2, dim=1).T)
        sim = 0.5 * (sim + sim.T)
        sim.fill_diagonal_(1.0)

        gamma = max(float(getattr(args, "sfl_gamma", 0.01) or 0.01), 1e-6)
        topk = int(getattr(args, "sfl_topk", 0) or 0)
        if topk <= 0:
            topk = min(max(2, int(math.ceil(math.sqrt(num_clients)))), num_clients)

        logits = sim / gamma
        if topk < num_clients:
            _, indices = torch.topk(sim, k=topk, dim=1)
            mask = torch.zeros_like(sim, dtype=torch.bool)
            mask.scatter_(1, indices, True)
            mask = torch.logical_or(mask, mask.T)
            logits = logits.masked_fill(~mask, float("-inf"))
        adjacency = torch.softmax(logits, dim=1)

        personalized = flat_states.clone()
        for _ in range(max(int(getattr(args, "sfl_m_steps", 1) or 1), 1)):
            personalized = torch.matmul(adjacency, personalized)

    readout_weights = _sfl_client_weights(device)
    if readout_weights.numel() != personalized.shape[0]:
        readout_weights = torch.ones((personalized.shape[0],), device=device) / max(personalized.shape[0], 1)
    global_vec = torch.matmul(readout_weights.unsqueeze(0), personalized).squeeze(0)

    shapes = {name: local_states[0][name].shape for name in param_keys}
    numels = {name: local_states[0][name].numel() for name in param_keys}

    personalized_states = []
    for client_idx in range(num_clients):
        ptr = 0
        state = {}
        for name in param_keys:
            numel = numels[name]
            state[name] = personalized[client_idx, ptr:ptr + numel].view(shapes[name]).detach().cpu()
            ptr += numel
        personalized_states.append(state)

    ptr = 0
    global_state = {}
    for name in param_keys:
        numel = numels[name]
        global_state[name] = global_vec[ptr:ptr + numel].view(shapes[name]).detach().cpu()
        ptr += numel

    return global_state, personalized_states, adjacency.detach().cpu()


def train_sfl_centralized_sim(settings, opts, trainer_name="SFL centralized-sim", return_package=False):
    device = args.device
    clients = []

    for rank, setting in enumerate(settings):
        train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = setting
        if model is None:
            raise NotImplementedError("SFL centralized simulation requires a concrete client model.")

        model.to(device)
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        if loss_func is None:
            loss_func = torch.nn.MSELoss().to(device)
        elif hasattr(loss_func, "to"):
            loss_func = loss_func.to(device)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        clients.append({
            "rank": rank,
            "model": model,
            "optimizer": optimizer,
            "loss_func": loss_func,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "scaler": scaler,
        })
        print(
            f"[{trainer_name}] client{rank} train_batches={len(train_loader)} "
            f"val_batches={len(val_loader)} test_batches={len(test_loader)}",
            flush=True,
        )

    global_state = None
    personalized_states = None
    adjacency = None

    best_avg_val = float("inf")
    best_client_states = None
    best_global_state = None
    best_personalized_states = None
    best_adjacency = None
    patience_count = 0

    for epoch in range(args.epochs):
        lambda_t = float(getattr(args, "sfl_lambda", 0.1) or 0.1) if epoch > 0 else 0.0
        train_losses = []
        reg_losses = []

        for client in clients:
            model = client["model"]
            optimizer = client["optimizer"]
            loss_func = client["loss_func"]
            model.train()
            total_loss = 0.0
            total_reg = 0.0
            steps = 0

            ref_global = global_state
            ref_personal = None if personalized_states is None else personalized_states[client["rank"]]
            for batch_idx, batch in enumerate(client["train_loader"]):
                if opts.max_batches and batch_idx >= opts.max_batches:
                    break
                batch = _move_batch(batch, device)
                optimizer.zero_grad()
                pred, y, aux_loss, output = forward_model_with_loss_input(model, batch, loss_func)
                pred, y = align_pred_and_target(pred, y)
                base_loss = loss_func(_loss_input_from_output(output, pred, loss_func), y)
                if aux_loss is not None:
                    base_loss = base_loss + aux_loss

                reg_loss = torch.zeros((), device=device)
                if lambda_t > 0.0 and ref_global is not None and ref_personal is not None:
                    reg_loss = lambda_t * (
                        _sfl_parameter_mse(model, ref_global) +
                        _sfl_parameter_mse(model, ref_personal)
                    )

                loss = base_loss + reg_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += loss.item()
                total_reg += reg_loss.item()
                steps += 1

            train_losses.append(total_loss / max(steps, 1))
            reg_losses.append(total_reg / max(steps, 1))

        local_states = [_state_dict_cpu(client["model"]) for client in clients]
        global_state, personalized_states, adjacency = _sfl_structure_aggregate(local_states, device)

        val_maes = [
            normalized_mae(client["model"], client["val_loader"], device)
            for client in clients
        ]
        avg_val = sum(val_maes) / max(len(val_maes), 1)
        avg_train = sum(train_losses) / max(len(train_losses), 1)
        avg_reg = sum(reg_losses) / max(len(reg_losses), 1)

        if avg_val < best_avg_val - opts.min_delta:
            best_avg_val = avg_val
            best_client_states = [copy.deepcopy(client["model"].state_dict()) for client in clients]
            best_global_state = copy.deepcopy(global_state)
            best_personalized_states = copy.deepcopy(personalized_states)
            best_adjacency = adjacency.clone()
            patience_count = 0
        else:
            patience_count += 1

        val_detail = ", ".join(f"c{idx}={val_mae:.4f}" for idx, val_mae in enumerate(val_maes))
        print(
            f"[{trainer_name}] Epoch {epoch} | Avg Train Loss(Norm): {avg_train:.4f} | "
            f"Avg Reg: {avg_reg:.4f} | Avg Val MAE(Norm): {avg_val:.4f} | "
            f"lambda={lambda_t:.4f} | {val_detail} | best_avg={best_avg_val:.4f} | "
            f"patience={patience_count}/{opts.patience}",
            flush=True,
        )
        if best_adjacency is not None and epoch % 10 == 0:
            print(f"[{trainer_name}] adjacency(epoch={epoch})=\n{best_adjacency.numpy().round(3)}", flush=True)

        if patience_count >= opts.patience:
            print(
                f"[{trainer_name}] Early stop at epoch {epoch}, best_avg_val={best_avg_val:.4f}",
                flush=True,
            )
            break

    if best_client_states is not None:
        for client, state in zip(clients, best_client_states):
            client["model"].load_state_dict(state)

    metric_list = [
        test_metrics(client["model"], client["test_loader"], client["scaler"], device)
        for client in clients
    ]
    if return_package:
        return {
            "client_metrics": metric_list,
            "aggregate_metrics": _aggregate_metric_dicts(metric_list),
            "global_state": best_global_state,
            "personalized_states": best_personalized_states,
            "adjacency": best_adjacency,
        }
    return metric_list


def train_sfl_standalone(setting, opts):
    print("[SFT/SFL centralized-sim] Delegated to the multi-client SFL simulator.")
    return train_common_standalone(setting, opts)


def _fuels_forward(model, batch, training=False):
    x, y, _ = _unpack_xy(batch)
    return model(x), y


def train_fuels_standalone(setting, opts):
    train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = setting
    device = args.device
    model.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if loss_func is None:
        loss_func = torch.nn.MSELoss().to(device)
    elif hasattr(loss_func, "to"):
        loss_func = loss_func.to(device)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    print(
        "[FUELS no-communication] Running prediction + local intra-client contrastive loss; "
        "inter-client PR/NR prototype exchange is disabled.",
        flush=True,
    )
    intra_weight = float(getattr(args, "fuels_intra_weight", 1.0))
    val_metric = getattr(args, "fuels_val_metric", "normalized_mae")
    print(f"[FUELS no-communication] intra_weight={intra_weight}", flush=True)
    print(f"[FUELS no-communication] val_metric={val_metric}", flush=True)
    print(
        f"[FUELS no-communication] train_batches={len(train_loader)} "
        f"val_batches={len(val_loader)} test_batches={len(test_loader)}",
        flush=True,
    )

    best_state = None
    best_val = float("inf")
    patience_count = 0

    for epoch in range(args.epochs):
        model.train()
        total_pred = 0.0
        total_intra = 0.0
        total_loss = 0.0
        steps = 0
        for batch_idx, batch in enumerate(train_loader):
            if opts.max_batches and batch_idx >= opts.max_batches:
                break
            batch = _move_batch(batch, device)
            x, y, _ = _unpack_xy(batch)
            optimizer.zero_grad()

            x_prime = model.make_augmented_view(x)
            r_n = model.encode(x)
            r_n_prime = model.encode(x_prime)
            pred = model.decode(r_n)
            pred, y = align_pred_and_target(pred, y)

            loss_pred = loss_func(pred, y)
            loss_intra = model.compute_intra_loss(r_n, r_n_prime, tau=args.fuels_tau)
            loss = loss_pred + (intra_weight * loss_intra)
            if not torch.isfinite(loss):
                print(f"[FUELS no-communication] Epoch {epoch} batch {batch_idx} non_finite_loss={loss.item()}", flush=True)
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_pred += loss_pred.item()
            total_intra += loss_intra.item()
            total_loss += loss.item()
            steps += 1

        if val_metric == "real_mae":
            val_mae = _real_mae_with_forward(model, val_loader, scaler, device, _fuels_forward)
            val_label = "Val MAE(Real)"
        else:
            val_mae = _normalized_mae_with_forward(model, val_loader, device, _fuels_forward)
            val_label = "Val MAE(Norm)"
        print(
            f"[FUELS no-communication] Epoch {epoch} | "
            f"Pred Loss(Norm): {total_pred / max(steps, 1):.4f} | "
            f"Intra: {total_intra / max(steps, 1):.4f} | "
            f"Train Loss(Norm): {total_loss / max(steps, 1):.4f} | "
            f"{val_label}: {val_mae:.4f} | best={best_val:.4f} | "
            f"patience={patience_count}/{opts.patience}",
            flush=True,
        )
        if val_mae < best_val - opts.min_delta:
            best_val = val_mae
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= opts.patience:
                print(f"[FUELS no-communication] Early stop at epoch {epoch}, best_val={best_val:.4f}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return _metrics_from_loader(model, test_loader, scaler, device, _fuels_forward)


def _fuels_pad_to_full_batch(x, target_bs):
    cur_bs = x.shape[0]
    if cur_bs == target_bs:
        return x
    if cur_bs > target_bs:
        return x[:target_bs]
    repeat_shape = [target_bs - cur_bs] + [1] * (x.dim() - 1)
    return torch.cat([x, x[-1:].repeat(*repeat_shape)], dim=0)


def _fuels_build_local_prototype(model, proto_loader, device):
    model.eval()
    target_bs = args.batch_size
    proto_sum = None
    proto_count = 0

    with torch.no_grad():
        for batch in proto_loader:
            x, _, _ = _unpack_xy(batch)
            if x.shape[0] != target_bs:
                continue
            x = x.to(device)
            r_proto = model.encode(x).detach()
            if proto_sum is None:
                proto_sum = torch.zeros_like(r_proto)
            proto_sum += r_proto
            proto_count += 1

        if proto_count == 0:
            fallback = next(iter(proto_loader))
            x, _, _ = _unpack_xy(fallback)
            x = _fuels_pad_to_full_batch(x, target_bs).to(device)
            proto_sum = model.encode(x).detach()
            proto_count = 1

    return proto_sum / proto_count


def _fuels_jsd_distance(p, q, eps=1e-8):
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1).mean()
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1).mean()
    return 0.5 * (kl_pm + kl_qm)


def _fuels_make_pr_nr(prototypes):
    probs = [torch.softmax(r, dim=-1) for r in prototypes]
    num_clients = len(prototypes)
    jsd_matrix = torch.zeros((num_clients, num_clients), device=probs[0].device)
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            jsd_val = _fuels_jsd_distance(probs[i], probs[j])
            jsd_matrix[i, j] = jsd_val
            jsd_matrix[j, i] = jsd_val

    active_jsds = jsd_matrix[jsd_matrix > 0]
    beta = (
        torch.quantile(active_jsds, args.fuels_beta_percentile / 100.0)
        if active_jsds.numel() > 0
        else torch.tensor(0.0, device=jsd_matrix.device)
    )

    pr_dict, nr_dict = {}, {}
    for i in range(num_clients):
        pos_protos, neg_protos = [], []
        for j in range(num_clients):
            if i == j or jsd_matrix[i, j] <= beta:
                pos_protos.append(prototypes[j])
            else:
                neg_protos.append(prototypes[j])
        pr_dict[i] = torch.stack(pos_protos, dim=0).mean(dim=0)
        nr_dict[i] = torch.stack(neg_protos, dim=0).mean(dim=0) if neg_protos else torch.zeros_like(prototypes[i])
    return pr_dict, nr_dict, beta


def _fuels_sim_normalized_mae(models, val_loaders, device):
    total_abs = 0.0
    total_elements = 0
    for model, loader in zip(models, val_loaders):
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = _move_batch(batch, device)
                pred, y = _fuels_forward(model, batch, training=False)
                pred, y = align_pred_and_target(pred, y)
                total_abs += torch.abs(pred - y).sum().item()
                total_elements += y.numel()
    return total_abs / max(total_elements, 1)


def _aggregate_metric_dicts(metric_dicts):
    def _as_float(value, default=0.0):
        if value in ("", None):
            return default
        return float(value)

    def _as_int(value, default=0):
        if value in ("", None):
            return default
        return int(float(value))

    total_elements = sum(_as_int(m.get("elements")) for m in metric_dicts)
    total_abs = sum(_as_float(m.get("abs_error_sum")) for m in metric_dicts)
    total_sq = sum(_as_float(m.get("sq_error_sum")) for m in metric_dicts)
    total_mape = sum(_as_float(m.get("mape_error_sum")) for m in metric_dicts)
    total_mape_elements = sum(_as_int(m.get("mape_elements")) for m in metric_dicts)

    mse = total_sq / max(total_elements, 1)
    return {
        "mae": round(float(total_abs / max(total_elements, 1)), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(math.sqrt(mse)), 4),
        "mape": round(float(total_mape / total_mape_elements), 4) if total_mape_elements else 0.0,
        "elements": int(total_elements),
        "abs_error_sum": float(total_abs),
        "sq_error_sum": float(total_sq),
        "mape_error_sum": float(total_mape),
        "mape_elements": int(total_mape_elements),
    }


def train_fuels_centralized_sim(setting, opts):
    print(
        "[FUELS centralized-sim] Running four local FUELS clients with in-process PR/NR prototype exchange.",
        flush=True,
    )
    device = args.device
    settings = []
    for rank in range(args.num_clients):
        ctx = StandaloneCtx(rank)
        settings.append(get_setting(ctx))

    models, optimizers, loss_funcs, scalers = [], [], [], []
    train_loaders, proto_loaders, val_loaders, test_loaders = [], [], [], []
    for rank, local_setting in enumerate(settings):
        train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = local_setting
        model.to(device)
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        if loss_func is None:
            loss_func = torch.nn.MSELoss().to(device)
        elif hasattr(loss_func, "to"):
            loss_func = loss_func.to(device)

        models.append(model)
        optimizers.append(optimizer)
        loss_funcs.append(loss_func)
        scalers.append(scaler)
        train_loaders.append(DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True))
        proto_loaders.append(DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=False))
        val_loaders.append(DataLoader(val_set, batch_size=args.batch_size, shuffle=False))
        test_loaders.append(DataLoader(test_set, batch_size=args.batch_size, shuffle=False))
        print(
            f"[FUELS centralized-sim] client{rank} train_batches={len(train_loaders[-1])} "
            f"val_batches={len(val_loaders[-1])} test_batches={len(test_loaders[-1])}",
            flush=True,
        )

    steps = min(len(loader) for loader in train_loaders)
    best_states = None
    best_val = float("inf")
    patience_count = 0
    pr_dict, nr_dict = None, None

    for epoch in range(args.epochs):
        total_pred = 0.0
        total_intra = 0.0
        total_inter = 0.0
        total_loss = 0.0
        total_steps = 0

        for rank, (model, optimizer, loss_func, train_loader) in enumerate(
            zip(models, optimizers, loss_funcs, train_loaders)
        ):
            model.train()
            iterator = iter(train_loader)
            pr_n = pr_dict[rank].to(device) if pr_dict is not None else None
            nr_n = nr_dict[rank].to(device) if nr_dict is not None else None

            for batch_idx in range(steps):
                if opts.max_batches and batch_idx >= opts.max_batches:
                    break
                batch = _move_batch(next(iterator), device)
                x, y, _ = _unpack_xy(batch)
                optimizer.zero_grad()

                x_prime = model.make_augmented_view(x)
                r_n = model.encode(x)
                r_n_prime = model.encode(x_prime)
                pred = model.decode(r_n)
                pred, y = align_pred_and_target(pred, y)

                loss_pred = loss_func(pred, y)
                loss_intra = model.compute_intra_loss(r_n, r_n_prime, tau=args.fuels_tau)
                loss_inter = torch.tensor(0.0, device=device)
                if pr_n is not None and nr_n is not None:
                    loss_inter = model.compute_inter_loss(r_n, pr_n, nr_n, tau=args.fuels_tau)
                loss = loss_pred + loss_intra + (args.fuels_rho * loss_inter)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"FUELS centralized-sim hit non-finite loss at epoch={epoch} client={rank}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                total_pred += loss_pred.item()
                total_intra += loss_intra.item()
                total_inter += loss_inter.item()
                total_loss += loss.item()
                total_steps += 1

        prototypes = [
            _fuels_build_local_prototype(model, proto_loader, device)
            for model, proto_loader in zip(models, proto_loaders)
        ]
        pr_dict, nr_dict, beta = _fuels_make_pr_nr(prototypes)

        val_mae = _fuels_sim_normalized_mae(models, val_loaders, device)
        print(
            f"[FUELS centralized-sim] Epoch {epoch} | "
            f"Pred Loss(Norm): {total_pred / max(total_steps, 1):.4f} | "
            f"Intra: {total_intra / max(total_steps, 1):.4f} | "
            f"Inter: {total_inter / max(total_steps, 1):.4f} | "
            f"Train Loss(Norm): {total_loss / max(total_steps, 1):.4f} | "
            f"Val MAE(Norm): {val_mae:.4f} | beta={beta.item():.6f} | "
            f"best={best_val:.4f} | patience={patience_count}/{opts.patience}",
            flush=True,
        )

        if val_mae < best_val - opts.min_delta:
            best_val = val_mae
            best_states = [copy.deepcopy(model.state_dict()) for model in models]
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= opts.patience:
                print(f"[FUELS centralized-sim] Early stop at epoch {epoch}, best_val={best_val:.4f}", flush=True)
                break

    if best_states is not None:
        for model, state in zip(models, best_states):
            model.load_state_dict(state)

    metric_dicts = [
        _metrics_from_loader(model, test_loader, scaler, device, _fuels_forward)
        for model, test_loader, scaler in zip(models, test_loaders, scalers)
    ]
    return _aggregate_metric_dicts(metric_dicts)


def train_fedtse_standalone(setting, opts):
    print("[FedTSE no-communication] Running local TrafficLSTM without asynchronous server weight aggregation.")
    return train_common_standalone(setting, opts)


def train_fedstn_standalone(setting, opts):
    print("[FedSTN no-communication] Running RLCN/AMFN/SCN locally; server FedGAT attention aggregation is disabled.")
    return train_common_standalone(setting, opts)


def _fedstn_sim_load_parts(device):
    from lib.load_dataset import load_grid_dataset_for_fedstn
    from model.FedSTN import FedSTN

    train_sets, val_sets, test_sets, scalers, models = [], [], [], [], []
    for rank, selected_nodes in enumerate(args.nodes_per):
        train_set, val_set, test_set, edge_index, scaler = load_grid_dataset_for_fedstn(
            dataset_name=args.dataset_name,
            t_in=args.t_in,
            t_out=args.t_out,
            device=device,
            selected_nodes=selected_nodes,
            model_name="FedSTN",
        )
        ext_dim = len(getattr(train_set, "ext_names", [])) or getattr(args, "ext_dim", 21)
        init_seed(args.seed)
        model = FedSTN(
            num_nodes=len(selected_nodes),
            input_dim=2,
            hidden_dim=args.hidden_dim,
            out_dim=args.t_out,
            edge_index=edge_index.to(device),
            output_features=2,
            ext_dim=ext_dim,
        ).to(device)
        train_sets.append(train_set)
        val_sets.append(val_set)
        test_sets.append(test_set)
        scalers.append(scaler)
        models.append(model)
        print(
            f"[FedSTN centralized-sim] client{rank} quadrant_nodes={len(selected_nodes)} "
            f"train={len(train_set)} val={len(val_set)} test={len(test_set)}",
            flush=True,
        )
    return train_sets, val_sets, test_sets, scalers, models


def _fedstn_sim_batches(datasets, batch_size, shuffle, seed):
    length = min(len(dataset) for dataset in datasets)
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        indices = torch.randperm(length, generator=generator).tolist()
    else:
        indices = list(range(length))
    for start in range(0, length, batch_size):
        batch_indices = indices[start:start + batch_size]
        yield [
            default_collate([dataset[idx] for idx in batch_indices])
            for dataset in datasets
        ]


def _fedstn_sim_aggregate(h_s_parts):
    split_sizes = [h_s.shape[1] for h_s in h_s_parts]
    h_s_global = torch.cat(h_s_parts, dim=1)
    h_s_parts = list(torch.split(h_s_global, split_sizes, dim=1))

    pooled_parts = [h_s.mean(dim=1) for h_s in h_s_parts]
    aggregated = []
    scale = math.sqrt(max(args.hidden_dim, 1))
    for i, h_s in enumerate(h_s_parts):
        scores = [
            (pooled_parts[i] * pooled_parts[j]).sum(dim=-1, keepdim=True) / scale
            for j in range(len(h_s_parts))
        ]
        attention = torch.nn.functional.softmax(torch.cat(scores, dim=1), dim=1)
        context = sum(
            attention[:, j:j + 1] * pooled_parts[j]
            for j in range(len(h_s_parts))
        )
        aggregated.append(h_s + context.unsqueeze(1).expand(-1, h_s.shape[1], -1))
    return aggregated


def _fedstn_sim_forward(models, batch_parts, device):
    h_s_parts, rlcn_parts, scn_parts, y_parts = [], [], [], []
    for model, batch in zip(models, batch_parts):
        batch = _move_batch(batch, device)
        x, y, x_ext = _unpack_xy(batch)
        h_s, r_out, s_out = model.forward_phase1(x, x_ext)
        h_s_parts.append(h_s)
        rlcn_parts.append(r_out)
        scn_parts.append(s_out)
        y_parts.append(y)

    aggregated_parts = _fedstn_sim_aggregate(h_s_parts)
    pred_parts = []
    aligned_y_parts = []
    for model, aggregated_h_s, r_out, s_out, y in zip(models, aggregated_parts, rlcn_parts, scn_parts, y_parts):
        pred = model.forward_phase2(aggregated_h_s, r_out, s_out)
        pred, y = align_pred_and_target(pred, y)
        pred_parts.append(pred)
        aligned_y_parts.append(y)
    return pred_parts, aligned_y_parts


def _fedstn_sim_normalized_mae(models, datasets, device):
    for model in models:
        model.eval()
    total_abs = 0.0
    total_elements = 0
    with torch.no_grad():
        for batch_parts in _fedstn_sim_batches(datasets, args.batch_size, shuffle=False, seed=args.seed):
            pred_parts, y_parts = _fedstn_sim_forward(models, batch_parts, device)
            for pred, y in zip(pred_parts, y_parts):
                total_abs += torch.abs(pred - y).sum().item()
                total_elements += y.numel()
    return total_abs / max(total_elements, 1)


def _fedstn_sim_test_metrics(models, datasets, scalers, device):
    for model in models:
        model.eval()
    total_abs = 0.0
    total_sq = 0.0
    total_elements = 0
    total_mape = 0.0
    total_mape_elements = 0

    with torch.no_grad():
        for batch_parts in _fedstn_sim_batches(datasets, args.batch_size, shuffle=False, seed=args.seed):
            pred_parts, y_parts = _fedstn_sim_forward(models, batch_parts, device)
            for pred, y, scaler in zip(pred_parts, y_parts, scalers):
                y_real = scaler.inverse_transform(y).detach().cpu().numpy()
                pred_real = scaler.inverse_transform(pred).detach().cpu().numpy()
                diff = pred_real - y_real
                total_abs += np.abs(diff).sum()
                total_sq += np.square(diff).sum()
                total_elements += y_real.size
                mask = y_real > 0.5
                if np.any(mask):
                    total_mape += (np.abs(diff[mask]) / y_real[mask]).sum()
                    total_mape_elements += int(mask.sum())

    mse = total_sq / max(total_elements, 1)
    return {
        "mae": round(float(total_abs / max(total_elements, 1)), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(math.sqrt(mse)), 4),
        "mape": round(float(total_mape / total_mape_elements), 4) if total_mape_elements else 0.0,
    }


def train_fedstn_centralized_sim(setting, opts):
    print(
        "[FedSTN centralized-sim] Running centralized four-quadrant FedSTN with "
        "single-process FedGAT hidden-state concat/attention/split; network communication is disabled.",
        flush=True,
    )
    device = args.device
    train_sets, val_sets, test_sets, scalers, models = _fedstn_sim_load_parts(device)
    loss_func = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(
        [param for model in models for param in model.parameters()],
        lr=args.lr,
        weight_decay=args.wd,
    )

    train_steps = math.ceil(min(len(dataset) for dataset in train_sets) / args.batch_size)
    val_steps = math.ceil(min(len(dataset) for dataset in val_sets) / args.batch_size)
    test_steps = math.ceil(min(len(dataset) for dataset in test_sets) / args.batch_size)
    print(
        f"[FedSTN centralized-sim] train_batches={train_steps} "
        f"val_batches={val_steps} test_batches={test_steps} batch_size={args.batch_size}",
        flush=True,
    )

    best_states = None
    best_val = float("inf")
    patience_count = 0

    for epoch in range(args.epochs):
        for model in models:
            model.train()
        total_loss = 0.0
        steps = 0

        for batch_idx, batch_parts in enumerate(
            _fedstn_sim_batches(train_sets, args.batch_size, shuffle=True, seed=args.seed + epoch)
        ):
            if opts.max_batches and batch_idx >= opts.max_batches:
                break
            optimizer.zero_grad()
            pred_parts, y_parts = _fedstn_sim_forward(models, batch_parts, device)
            loss = sum(loss_func(pred, y) for pred, y in zip(pred_parts, y_parts)) / len(pred_parts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [param for model in models for param in model.parameters()],
                max_norm=5.0,
            )
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        val_mae = _fedstn_sim_normalized_mae(models, val_sets, device)
        if val_mae < best_val - opts.min_delta:
            best_val = val_mae
            best_states = [copy.deepcopy(model.state_dict()) for model in models]
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= opts.patience:
                print(
                    f"[FedSTN centralized-sim] Epoch {epoch} | Train Loss(Norm): "
                    f"{total_loss / max(steps, 1):.4f} | Val MAE(Norm): {val_mae:.4f} "
                    f"| best={best_val:.4f} | patience={patience_count}/{opts.patience} | early_stop=True",
                    flush=True,
                )
                break
        print(
            f"[FedSTN centralized-sim] Epoch {epoch} | Train Loss(Norm): "
            f"{total_loss / max(steps, 1):.4f} | Val MAE(Norm): {val_mae:.4f} "
            f"| best={best_val:.4f} | patience={patience_count}/{opts.patience}",
            flush=True,
        )

    if best_states is not None:
        for model, state in zip(models, best_states):
            model.load_state_dict(state)
    return _fedstn_sim_test_metrics(models, test_sets, scalers, device)


def train_stfam_grid_standalone(setting, opts):
    print("[STFAM centralized global] Running compact full-grid STFAM with local global-autoencoder pretraining; federated broadcast is disabled.")
    return train_common_standalone(setting, opts)


def train_twomgtcn_standalone(setting, opts):
    print("[2MGTCN no-communication] Running local multimodal GCN/TCN; FPASS aggregation and domain adaptation exchange are disabled.")
    train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = setting
    if model is None:
        raise NotImplementedError("2MGTCN standalone requires a TwoMGTCN model from get_setting().")

    device = args.device
    model.to(device)
    topk = int(os.environ.get("TWOMGTCN_TOPK", "32"))
    if hasattr(getattr(model, "gcn", None), "set_sparse_topk"):
        model.gcn.set_sparse_topk(topk)
        nnz = model.gcn.A_sparse._nnz() if model.gcn.A_sparse is not None else model.gcn.A.numel()
        print(
            f"[2MGTCN no-communication] sparse_topk={topk} "
            f"adj_shape={tuple(model.gcn.A.shape)} nnz={nnz}",
            flush=True,
        )
    if os.environ.get("TWOMGTCN_FAST_TEMPORAL", "1") != "0" and hasattr(model, "enable_fast_temporal"):
        model.enable_fast_temporal()
        print(
            "[2MGTCN no-communication] fast_temporal=enabled "
            "(node-wise MLP temporal branch replaces the B*N Conv1d TCN bottleneck)",
            flush=True,
        )

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if loss_func is None:
        loss_func = torch.nn.MSELoss().to(device)
    elif hasattr(loss_func, "to"):
        loss_func = loss_func.to(device)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    print(
        f"[2MGTCN no-communication] train_batches={len(train_loader)} "
        f"val_batches={len(val_loader)} test_batches={len(test_loader)} "
        f"batch_size={args.batch_size}",
        flush=True,
    )

    best_state = None
    best_val = float("inf")
    patience_count = 0
    progress_every = int(os.environ.get("TWOMGTCN_PROGRESS_EVERY", "10"))

    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()
        batch_window_start = time.time()
        total_loss = 0.0
        steps = 0
        for batch_idx, batch in enumerate(train_loader):
            if opts.max_batches and batch_idx >= opts.max_batches:
                break
            batch = _move_batch(batch, device)
            optimizer.zero_grad()
            pred, y, aux_loss, output = forward_model_with_loss_input(model, batch, loss_func)
            pred, y = align_pred_and_target(pred, y)
            loss = loss_func(_loss_input_from_output(output, pred, loss_func), y)
            if aux_loss is not None:
                loss = loss + aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1

            if progress_every > 0 and (batch_idx + 1) % progress_every == 0:
                elapsed = time.time() - batch_window_start
                print(
                    f"[2MGTCN no-communication] Epoch {epoch} batch "
                    f"{batch_idx + 1}/{len(train_loader)} | "
                    f"avg_loss={total_loss / max(steps, 1):.4f} | "
                    f"{elapsed / progress_every:.3f}s/batch",
                    flush=True,
                )
                batch_window_start = time.time()

        val_start = time.time()
        val_mae = normalized_mae(model, val_loader, device)
        val_seconds = time.time() - val_start
        epoch_seconds = time.time() - epoch_start
        if val_mae < best_val - opts.min_delta:
            best_val = val_mae
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= opts.patience:
                print(
                    f"[2MGTCN no-communication] Epoch {epoch} | Train Loss(Norm): "
                    f"{total_loss / max(steps, 1):.4f} | Val MAE(Norm): {val_mae:.4f} "
                    f"| best={best_val:.4f} | patience={patience_count}/{opts.patience} "
                    f"| epoch_sec={epoch_seconds:.2f} | val_sec={val_seconds:.2f} | early_stop=True",
                    flush=True,
                )
                break
        print(
            f"[2MGTCN no-communication] Epoch {epoch} | Train Loss(Norm): "
            f"{total_loss / max(steps, 1):.4f} | Val MAE(Norm): {val_mae:.4f} "
            f"| best={best_val:.4f} | patience={patience_count}/{opts.patience} "
            f"| epoch_sec={epoch_seconds:.2f} | val_sec={val_seconds:.2f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return test_metrics(model, test_loader, scaler, device)


def _apply_stagcn_local_no_graph(model, train_set, device):
    if not hasattr(model, "edge_index"):
        return

    num_nodes = int(getattr(model, "num_nodes", 0) or 0)
    if num_nodes <= 0 and hasattr(train_set, "tensors") and len(train_set.tensors) > 0:
        num_nodes = int(train_set.tensors[0].shape[1])
    if num_nodes <= 0:
        raise ValueError("Cannot infer STAGCN-EC node count for local_no_graph ablation.")

    node_ids = torch.arange(num_nodes, dtype=torch.long, device=device)
    self_loop_edge_index = torch.stack([node_ids, node_ids], dim=0)
    model.register_buffer("edge_index", self_loop_edge_index)
    if hasattr(model, "cheb_polynomials"):
        model.cheb_polynomials = None
    print(
        f"[common standalone] STAGCN-EC local_no_graph: "
        f"edge_index replaced by {num_nodes} self-loops.",
        flush=True,
    )


def train_common_standalone(setting, opts):
    train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = setting
    if model is None:
        if args.model == "STFAM":
            return train_stfam_standalone(train_set, val_set, test_set, loss_func, scaler, opts)
        raise NotImplementedError(f"Standalone baseline is not available for model {args.model}")

    device = args.device
    model.to(device)
    if (
        getattr(opts, "paradigm", None) == "local"
        and args.model == "STGCN"
        and getattr(opts, "local_stagcn_graph_mode", "original") == "no_graph"
    ):
        _apply_stagcn_local_no_graph(model, train_set, device)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if loss_func is None:
        loss_func = torch.nn.MSELoss().to(device)
    elif hasattr(loss_func, "to"):
        loss_func = loss_func.to(device)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    print(
        f"[common standalone] train_batches={len(train_loader)} "
        f"val_batches={len(val_loader)} test_batches={len(test_loader)}",
        flush=True,
    )

    best_state = None
    best_val = float("inf")
    patience_count = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch_idx, batch in enumerate(train_loader):
            if opts.max_batches and batch_idx >= opts.max_batches:
                break
            batch = _move_batch(batch, device)
            optimizer.zero_grad()
            pred, y, aux_loss, output = forward_model_with_loss_input(model, batch, loss_func)
            pred, y = align_pred_and_target(pred, y)
            loss = loss_func(_loss_input_from_output(output, pred, loss_func), y)
            if aux_loss is not None:
                loss = loss + aux_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        val_mae = normalized_mae(model, val_loader, device)
        if val_mae < best_val - opts.min_delta:
            best_val = val_mae
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= opts.patience:
                print(
                    f"[common standalone] Epoch {epoch} | Train Loss(Norm): "
                    f"{total_loss / max(steps, 1):.4f} | Val MAE(Norm): {val_mae:.4f} "
                    f"| best={best_val:.4f} | patience={patience_count}/{opts.patience} | early_stop=True",
                    flush=True,
                )
                break
        print(
            f"[common standalone] Epoch {epoch} | Train Loss(Norm): "
            f"{total_loss / max(steps, 1):.4f} | Val MAE(Norm): {val_mae:.4f} "
            f"| best={best_val:.4f} | patience={patience_count}/{opts.patience}",
            flush=True,
        )

    restore_best = True
    if (
        getattr(opts, "paradigm", None) == "local"
        and args.model == "STGCN"
        and getattr(opts, "local_stagcn_eval_mode", "best") == "final"
    ):
        restore_best = False
        print(
            "[common standalone] STAGCN-EC strict local: evaluating final "
            "checkpoint without per-client best-val rollback.",
            flush=True,
        )

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    return test_metrics(model, test_loader, scaler, device)


def train_local_avg_early_stop(settings, opts, trainer_name="local avg-stop"):
    device = args.device
    clients = []

    for rank, setting in enumerate(settings):
        train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = setting
        if model is None:
            raise NotImplementedError("FedGRU local average early stop requires a concrete model.")

        model.to(device)
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        if loss_func is None:
            loss_func = torch.nn.MSELoss().to(device)
        elif hasattr(loss_func, "to"):
            loss_func = loss_func.to(device)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        loss_name = loss_func.__class__.__name__
        loss_input_mode = "full_tuple" if loss_name == "FedAGATLoss" else "pred_only"
        clients.append({
            "rank": rank,
            "model": model,
            "optimizer": optimizer,
            "loss_func": loss_func,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "scaler": scaler,
        })
        print(
            f"[{trainer_name}] client{rank} train_batches={len(train_loader)} "
            f"val_batches={len(val_loader)} test_batches={len(test_loader)} "
            f"loss={loss_name} loss_input={loss_input_mode}",
            flush=True,
        )

    best_states = None
    best_avg_val = float("inf")
    patience_count = 0

    for epoch in range(args.epochs):
        train_losses = []
        for client in clients:
            model = client["model"]
            optimizer = client["optimizer"]
            loss_func = client["loss_func"]
            model.train()
            total_loss = 0.0
            steps = 0

            for batch_idx, batch in enumerate(client["train_loader"]):
                if opts.max_batches and batch_idx >= opts.max_batches:
                    break
                batch = _move_batch(batch, device)
                optimizer.zero_grad()
                pred, y, aux_loss, output = forward_model_with_loss_input(model, batch, loss_func)
                pred, y = align_pred_and_target(pred, y)
                loss = loss_func(_loss_input_from_output(output, pred, loss_func), y)
                if aux_loss is not None:
                    loss = loss + aux_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1

            train_losses.append(total_loss / max(steps, 1))

        val_maes = [
            normalized_mae(client["model"], client["val_loader"], device)
            for client in clients
        ]
        avg_val = sum(val_maes) / max(len(val_maes), 1)
        avg_train = sum(train_losses) / max(len(train_losses), 1)

        if avg_val < best_avg_val - opts.min_delta:
            best_avg_val = avg_val
            best_states = [copy.deepcopy(client["model"].state_dict()) for client in clients]
            patience_count = 0
        else:
            patience_count += 1

        val_detail = ", ".join(
            f"c{idx}={val_mae:.4f}" for idx, val_mae in enumerate(val_maes)
        )
        print(
            f"[{trainer_name}] Epoch {epoch} | "
            f"Avg Train Loss(Norm): {avg_train:.4f} | "
            f"Avg Val MAE(Norm): {avg_val:.4f} | {val_detail} | "
            f"best_avg={best_avg_val:.4f} | patience={patience_count}/{opts.patience}",
            flush=True,
        )

        if patience_count >= opts.patience:
            print(
                f"[{trainer_name}] Early stop at epoch {epoch}, "
                f"best_avg_val={best_avg_val:.4f}",
                flush=True,
            )
            break

    restore_best = not (
        trainer_name.startswith("FedAGAT")
        and getattr(opts, "fedagat_local_eval_mode", "best") == "final"
    )
    if best_states is not None and restore_best:
        for client, state in zip(clients, best_states):
            client["model"].load_state_dict(state)
    elif best_states is not None and not restore_best:
        print(
            f"[{trainer_name}] eval_mode=final: evaluating final checkpoint "
            "instead of restoring best average-validation checkpoint.",
            flush=True,
        )

    return [
        test_metrics(client["model"], client["test_loader"], client["scaler"], device)
        for client in clients
    ]


def train_fedgru_local_avg_early_stop(settings, opts):
    return train_local_avg_early_stop(settings, opts, trainer_name="FedGRU local avg-stop")


def _fedstg_centralized_forward(client_model, server_model, static_adj, batch):
    x, y, _ = _unpack_xy(batch)
    h_tau, z_tau, l_k = client_model.forward_encode(x)
    z_c = z_tau.mean(dim=0)
    h_g = server_model(h_tau, z_c, static_adj)
    pred = client_model.forward_predict(h_tau, z_tau, h_g)
    pred, y = align_pred_and_target(pred, y)
    return pred, y, l_k


def _fedstg_centralized_normalized_mae(client_model, server_model, static_adj, loader, device):
    client_model.eval()
    server_model.eval()
    total_abs = 0.0
    total_elements = 0
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            pred, y, _ = _fedstg_centralized_forward(client_model, server_model, static_adj, batch)
            total_abs += torch.abs(pred - y).sum().item()
            total_elements += y.numel()
    return total_abs / max(total_elements, 1)


def _fedstg_centralized_test_metrics(client_model, server_model, static_adj, loader, scaler, device):
    client_model.eval()
    server_model.eval()
    total_abs = 0.0
    total_sq = 0.0
    total_elements = 0
    total_mape = 0.0
    total_mape_elements = 0
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            pred, y, _ = _fedstg_centralized_forward(client_model, server_model, static_adj, batch)
            y_real = scaler.inverse_transform(y).detach().cpu().numpy()
            pred_real = scaler.inverse_transform(pred).detach().cpu().numpy()
            diff = pred_real - y_real
            total_abs += np.abs(diff).sum()
            total_sq += np.square(diff).sum()
            total_elements += y_real.size
            mask = y_real > 0.5
            if np.any(mask):
                total_mape += (np.abs(diff[mask]) / y_real[mask]).sum()
                total_mape_elements += int(mask.sum())

    mse = total_sq / max(total_elements, 1)
    return {
        "mae": round(float(total_abs / max(total_elements, 1)), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(math.sqrt(mse)), 4),
        "mape": round(float(total_mape / total_mape_elements), 4) if total_mape_elements else 0.0,
        "elements": int(total_elements),
        "abs_error_sum": float(total_abs),
        "sq_error_sum": float(total_sq),
        "mape_error_sum": float(total_mape),
        "mape_elements": int(total_mape_elements),
    }


def train_fedstg_centralized_upper(setting, opts):
    print("[FedSTG centralized-upper] Running full-graph FedSTG client + server graph fusion.")
    from lib.load_dataset import build_fedstg_static_adj
    from model.FedSTG import FedSTG_Server

    train_set, val_set, test_set, client_model, optimizer, loss_func, _, _, _, scaler, _ = setting
    device = args.device
    client_model.to(device)
    server_model = FedSTG_Server(
        hidden_dim=args.hidden_dim,
        beta=getattr(args, "fedstg_beta", 0.5),
    ).to(device)
    static_adj = build_fedstg_static_adj(
        dataset_name=args.dataset_name,
        num_nodes=total_nodes_for(args.dataset_name),
        sigma=getattr(args, "fedstg_sigma", 0.0),
        kappa=getattr(args, "fedstg_kappa", 0.0),
        project_root=PROJECT_ROOT,
    ).to(device)

    if optimizer is None:
        optimizer = torch.optim.Adam(client_model.parameters(), lr=args.lr, weight_decay=args.wd)
    server_optimizer = torch.optim.Adam(server_model.parameters(), lr=args.lr, weight_decay=args.wd)
    if loss_func is None:
        loss_func = torch.nn.MSELoss().to(device)
    elif hasattr(loss_func, "to"):
        loss_func = loss_func.to(device)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    best_val = float("inf")
    best_client_state = None
    best_server_state = None
    patience_count = 0
    alpha = float(getattr(args, "fedstg_alpha", 0.01))

    for epoch in range(args.epochs):
        client_model.update_B_prev()
        client_model.train()
        server_model.train()
        total_loss = 0.0
        steps = 0
        for batch_idx, batch in enumerate(train_loader):
            if opts.max_batches and batch_idx >= opts.max_batches:
                break
            batch = _move_batch(batch, device)
            optimizer.zero_grad()
            server_optimizer.zero_grad()
            pred, y, l_k = _fedstg_centralized_forward(client_model, server_model, static_adj, batch)
            loss = loss_func(pred, y) + alpha * l_k
            loss.backward()
            torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(server_model.parameters(), max_norm=5.0)
            optimizer.step()
            server_optimizer.step()
            total_loss += loss.item()
            steps += 1

        val_mae = _fedstg_centralized_normalized_mae(client_model, server_model, static_adj, val_loader, device)
        if val_mae < best_val - opts.min_delta:
            best_val = val_mae
            best_client_state = copy.deepcopy(client_model.state_dict())
            best_server_state = copy.deepcopy(server_model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= opts.patience:
                print(
                    f"[FedSTG centralized-upper] Epoch {epoch} | "
                    f"Train Loss(Norm): {total_loss / max(steps, 1):.4f} | "
                    f"Val MAE(Norm): {val_mae:.4f} | best={best_val:.4f} | "
                    f"patience={patience_count}/{opts.patience} | early_stop=True",
                    flush=True,
                )
                break
        print(
            f"[FedSTG centralized-upper] Epoch {epoch} | "
            f"Train Loss(Norm): {total_loss / max(steps, 1):.4f} | "
            f"Val MAE(Norm): {val_mae:.4f} | best={best_val:.4f} | "
            f"patience={patience_count}/{opts.patience}",
            flush=True,
        )

    if best_client_state is not None:
        client_model.load_state_dict(best_client_state)
    if best_server_state is not None:
        server_model.load_state_dict(best_server_state)
    return _fedstg_centralized_test_metrics(client_model, server_model, static_adj, test_loader, scaler, device)


def train_baseline(setting, spec, opts):
    if (
        spec.method == "FedGTP"
        and getattr(opts, "paradigm", None) == "global"
        and getattr(opts, "fedgtp_global_mode", "centralized_sim") == "centralized_sim"
    ):
        trainer = train_fedgtp_centralized_sim
        print(f"[dispatcher] method={spec.method} trainer={trainer.__name__}")
        return trainer(setting, opts)
    if spec.method == "FedOSTC" and getattr(opts, "paradigm", None) == "global":
        trainer = train_fedostc_centralized_sim
        print(f"[dispatcher] method={spec.method} trainer={trainer.__name__}")
        return trainer(setting, opts)
    if spec.method == "FedSTN" and getattr(opts, "paradigm", None) == "global":
        trainer = train_fedstn_centralized_sim
        print(f"[dispatcher] method={spec.method} trainer={trainer.__name__}")
        return trainer(setting, opts)
    if (
        spec.method == "STFAM"
        and getattr(opts, "paradigm", None) == "global"
        and getattr(opts, "stfam_global_mode", "upper") == "centralized_sim"
    ):
        trainer = train_stfam_centralized_upper
        print(f"[dispatcher] method={spec.method} trainer={trainer.__name__}")
        return trainer(setting, opts)
    if (
        spec.method == "FUELS"
        and getattr(opts, "paradigm", None) == "global"
        and getattr(opts, "fuels_global_mode", "upper") == "centralized_sim"
    ):
        trainer = train_fuels_centralized_sim
        print(f"[dispatcher] method={spec.method} trainer={trainer.__name__}")
        return trainer(setting, opts)
    if spec.method == "FedSTG" and getattr(opts, "paradigm", None) == "global":
        trainer = train_fedstg_centralized_upper
        print(f"[dispatcher] method={spec.method} trainer={trainer.__name__}")
        return trainer(setting, opts)
    if spec.method == "FedmSSA" and getattr(opts, "paradigm", None) == "global":
        trainer = train_fedmssa_centralized_sim
        print(f"[dispatcher] method={spec.method} trainer={trainer.__name__}")
        return trainer(setting, opts)

    dispatch = {
        "UFCL": train_ufcl_standalone,
        "FedTPS": train_fedtps_standalone,
        "Fed4TP": train_fed4tp_standalone,
        "FedmSSA": train_fedmssa_standalone,
        "FedGRU": train_common_standalone,
        "FC-FedGCN": train_fcfedgcn_standalone,
        "FGNNEH": train_fgnneh_standalone,
        "FedMetro": train_fedmetro_standalone,
        "pFedCTP": train_pfedctp_standalone,
        "T-ISTGNN": train_tistgnn_standalone,
        "FedOSTC": train_fedostc_standalone,
        "REFOL": train_refol_standalone,
        "CNFGNN": train_cnfgnn_standalone,
        "FedSTG": train_fedstg_standalone,
        "FedGODE": train_fedgode_standalone,
        "SFT/SFL": train_sfl_standalone,
        "FUELS": train_fuels_standalone,
        "FedTSE": train_fedtse_standalone,
        "TDLR": train_common_standalone,
        "SEDLR": train_common_standalone,
        "TDLR_SEDLR": train_common_standalone,
        "FedSTN": train_fedstn_standalone,
        "STFAM": train_stfam_grid_standalone,
        "2MGTCN": train_twomgtcn_standalone,
    }
    trainer = dispatch.get(spec.method, train_common_standalone)
    print(f"[dispatcher] method={spec.method} trainer={trainer.__name__}")
    return trainer(setting, opts)


def result_row(spec, dataset_name, feature, client, metrics):
    row = {
        "run_id": current_run_id(),
        "section": spec.section,
        "method": effective_method_name(spec),
        "model_arg": spec.model_arg,
        "dataset": display_dataset_name(dataset_name),
        "feature": feature,
        "client": client,
        "client_mae": metrics["mae"],
        "client_mse": metrics["mse"],
        "client_rmse": metrics["rmse"],
        "client_mape": metrics["mape"],
        "mae": metrics["mae"],
        "mse": metrics["mse"],
        "rmse": metrics["rmse"],
        "mape": metrics["mape"],
        "elements": metrics.get("elements", ""),
        "abs_error_sum": metrics.get("abs_error_sum", ""),
        "sq_error_sum": metrics.get("sq_error_sum", ""),
        "mape_error_sum": metrics.get("mape_error_sum", ""),
        "mape_elements": metrics.get("mape_elements", ""),
        "seed": args.seed,
    }
    row.update(row_sample_metrics(row))
    return row


def stagcn_local_variant_name(opts):
    parts = []
    if getattr(opts, "local_stagcn_graph_mode", "original") == "no_graph":
        parts.append("no_graph")
    if getattr(opts, "local_stagcn_eval_mode", "best") == "final":
        parts.append("final")
    if not parts:
        return "STAGCN-EC"
    return "STAGCN-EC_local_" + "_".join(parts)


def _stagcn_flow_split(dataset_name, num_clients):
    from data import dividing

    normalized = "PeMSD7" if dataset_name == "PeMS07" else dataset_name
    candidates = [
        f"{normalized}FLOW_{num_clients}p_metis",
        f"{dataset_name}FLOW_{num_clients}p_metis",
    ]
    for name in candidates:
        if hasattr(dividing, name):
            return copy.deepcopy(getattr(dividing, name)), name
    return equal_split(total_nodes_for(normalized), num_clients), "equal_split"


def _stagcn_partition_neighbors(dataset_name, feature, nodes_per):
    from lib.load_dataset import load_dataset

    total_nodes = total_nodes_for(dataset_name)
    _, _, _, full_edge_index, _ = load_dataset(
        dataset_name=dataset_name,
        feature_type=feature,
        normalizer=args.normalizer,
        T_in=args.t_in,
        T_out=args.t_out,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        return_edge_index=True,
        device="cpu",
        selected_nodes=list(range(total_nodes)),
    )
    node_to_client = {}
    for client_idx, nodes in enumerate(nodes_per):
        for node in nodes:
            node_to_client[node] = client_idx

    neighbors = [set() for _ in nodes_per]
    edge_array = full_edge_index.cpu().numpy()
    for edge_idx in range(edge_array.shape[1]):
        src, dst = int(edge_array[0, edge_idx]), int(edge_array[1, edge_idx])
        if src in node_to_client and dst in node_to_client:
            c_src, c_dst = node_to_client[src], node_to_client[dst]
            if c_src != c_dst:
                neighbors[c_src].add(c_dst)
                neighbors[c_dst].add(c_src)
    return [sorted(items) for items in neighbors]


def _state_dict_cpu(model):
    param_names = {name for name, _ in model.named_parameters()}
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if name in param_names
    }


def _load_matching_state(model, state_dict):
    current = model.state_dict()
    param_names = {name for name, _ in model.named_parameters()}
    filtered = {
        name: tensor.to(next(model.parameters()).device)
        for name, tensor in state_dict.items()
        if name in param_names and name in current and current[name].shape == tensor.shape
    }
    model.load_state_dict(filtered, strict=False)


def _stagcn_warmup_state(model, state_dict, train_loader, loss_func, device, steps, lr):
    temp_model = copy.deepcopy(model)
    _load_matching_state(temp_model, state_dict)
    temp_model.to(device)
    temp_model.train()
    temp_optimizer = torch.optim.Adam(temp_model.parameters(), lr=lr, weight_decay=args.wd)
    total_loss = 0.0
    actual_steps = 0
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= steps:
            break
        batch = _move_batch(batch, device)
        temp_optimizer.zero_grad()
        pred, y, aux_loss, output = forward_model_with_loss_input(temp_model, batch, loss_func)
        pred, y = align_pred_and_target(pred, y)
        loss = loss_func(_loss_input_from_output(output, pred, loss_func), y)
        if aux_loss is not None:
            loss = loss + aux_loss
        loss.backward()
        temp_optimizer.step()
        total_loss += loss.item()
        actual_steps += 1
    return total_loss / max(actual_steps, 1), _state_dict_cpu(temp_model)


def train_stagcn_transfer_upper(setting, opts, candidate_states, rank):
    train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = setting
    if model is None:
        raise NotImplementedError("STAGCN-EC transfer upper requires a concrete STGCN model.")

    device = args.device
    model.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if loss_func is None:
        loss_func = torch.nn.MSELoss().to(device)
    elif hasattr(loss_func, "to"):
        loss_func = loss_func.to(device)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    warmup_steps = int(getattr(opts, "stagcn_transfer_warmup_steps", 10))
    local_init = _state_dict_cpu(model)
    selected_label = "local_init"
    if candidate_states:
        choices = []
        local_loss, local_state = _stagcn_warmup_state(
            model, local_init, train_loader, loss_func, device, warmup_steps, args.lr
        )
        choices.append(("local_init", local_loss, local_state))
        for source_rank, candidate_state in candidate_states:
            candidate_loss, candidate_warm_state = _stagcn_warmup_state(
                model, candidate_state, train_loader, loss_func, device, warmup_steps, args.lr
            )
            choices.append((f"neighbor{source_rank}", candidate_loss, candidate_warm_state))
        selected_label, selected_loss, selected_state = min(choices, key=lambda item: item[1])
        _load_matching_state(model, selected_state)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        print(
            f"[STAGCN-EC transfer_upper] partition{rank} selected={selected_label} "
            f"warmup_loss={selected_loss:.6f} candidates="
            + ",".join(f"{label}:{loss:.6f}" for label, loss, _ in choices),
            flush=True,
        )
    else:
        print(
            f"[STAGCN-EC transfer_upper] partition{rank} selected={selected_label} "
            "reason=no_trained_neighbor",
            flush=True,
        )

    best_state = None
    best_val = float("inf")
    patience_count = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch_idx, batch in enumerate(train_loader):
            if opts.max_batches and batch_idx >= opts.max_batches:
                break
            batch = _move_batch(batch, device)
            optimizer.zero_grad()
            pred, y, aux_loss, output = forward_model_with_loss_input(model, batch, loss_func)
            pred, y = align_pred_and_target(pred, y)
            loss = loss_func(_loss_input_from_output(output, pred, loss_func), y)
            if aux_loss is not None:
                loss = loss + aux_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        val_mae = normalized_mae(model, val_loader, device)
        if val_mae < best_val - opts.min_delta:
            best_val = val_mae
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= opts.patience:
                print(
                    f"[STAGCN-EC transfer_upper] partition{rank} Epoch {epoch} | "
                    f"Train Loss(Norm): {total_loss / max(steps, 1):.4f} | "
                    f"Val MAE(Norm): {val_mae:.4f} | best={best_val:.4f} | "
                    f"patience={patience_count}/{opts.patience} | early_stop=True",
                    flush=True,
                )
                break
        print(
            f"[STAGCN-EC transfer_upper] partition{rank} Epoch {epoch} | "
            f"Train Loss(Norm): {total_loss / max(steps, 1):.4f} | "
            f"Val MAE(Norm): {val_mae:.4f} | best={best_val:.4f} | "
            f"patience={patience_count}/{opts.patience}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = test_metrics(model, test_loader, scaler, device)
    return metrics, _state_dict_cpu(model)


def run_one_local(spec, dataset_name, feature, opts):
    configure_method(spec)
    configure_dataset(dataset_name, feature)
    configure_clients_for_local(args.dataset_name)
    args.norm_scope = getattr(opts, "local_norm_scope", "global")
    opts.paradigm = "local"
    print(
        f"[local] num_clients={args.num_clients} norm_scope={args.norm_scope} "
        f"scaler_fit_scope={getattr(args, 'scaler_fit_scope', 'selected')} "
        f"stagcn_eval_mode={getattr(opts, 'local_stagcn_eval_mode', 'best')} "
        f"stagcn_graph_mode={getattr(opts, 'local_stagcn_graph_mode', 'original')} "
        f"fedgru_stop_scope={getattr(opts, 'fedgru_local_stop_scope', 'client_mean')} "
        f"fedagat_eval_mode={getattr(opts, 'fedagat_local_eval_mode', 'best')}"
    )

    rows = []
    use_client_mean_stop = (
        spec.method == "FedAGAT"
        or (
            spec.method == "FedGRU"
            and getattr(opts, "fedgru_local_stop_scope", "client_mean") == "client_mean"
        )
    )
    if spec.method == "SFT/SFL":
        settings = []
        for rank in range(args.num_clients):
            init_seed(args.seed)
            ctx = StandaloneCtx(rank)
            opts.current_rank = rank
            settings.append(get_setting(ctx))

        metric_list = train_sfl_centralized_sim(settings, opts, trainer_name="SFL local centralized-sim")
        for rank, metrics in enumerate(metric_list):
            row = result_row(spec, args.dataset_name, feature, f"client{rank}", metrics)
            write_raw_result("local", row)
            rows.append(row)
        write_summary_results("local", rows)
        return rows

    if use_client_mean_stop:
        settings = []
        for rank in range(args.num_clients):
            init_seed(args.seed)
            ctx = StandaloneCtx(rank)
            opts.current_rank = rank
            settings.append(get_setting(ctx))

        metric_list = train_local_avg_early_stop(settings, opts, trainer_name=f"{spec.method} local avg-stop")
        for rank, metrics in enumerate(metric_list):
            row = result_row(spec, args.dataset_name, feature, f"client{rank}", metrics)
            write_raw_result("local", row)
            rows.append(row)
        write_summary_results("local", rows)
        return rows

    for rank in range(args.num_clients):
        init_seed(args.seed)
        ctx = StandaloneCtx(rank)
        setting = get_setting(ctx)
        opts.current_rank = rank
        metrics = train_baseline(setting, spec, opts)
        row = result_row(spec, args.dataset_name, feature, f"client{rank}", metrics)
        if (
            spec.method == "FedMetro"
            and getattr(opts, "fedmetro_standalone_ablation", "full") != "full"
        ):
            row["method"] = f"FedMetro_local_{opts.fedmetro_standalone_ablation}"
        if spec.method == "STAGCN-EC":
            row["method"] = stagcn_local_variant_name(opts)
        write_raw_result("local", row)
        rows.append(row)
    write_summary_results("local", rows)
    return rows


def run_one_global(spec, dataset_name, feature, opts):
    configure_method(spec)
    configure_dataset(dataset_name, feature)
    args.norm_scope = getattr(opts, "global_norm_scope", "global")
    opts.paradigm = "global"
    print(f"[global] norm_scope={args.norm_scope}")
    if spec.method == "SFT/SFL":
        configure_clients_for_local(args.dataset_name)
        args.nodes_per, split_name = graph_nodes_per_split(args.dataset_name, feature, args.num_clients)
        print(
            f"[global] {spec.method} centralized_sim split={split_name} "
            f"num_clients={args.num_clients} sizes={[len(nodes) for nodes in args.nodes_per]}",
            flush=True,
        )

        sfl_settings = []
        for rank in range(args.num_clients):
            init_seed(args.seed)
            ctx = StandaloneCtx(rank)
            opts.current_rank = rank
            sfl_settings.append(get_setting(ctx))

        package = train_sfl_centralized_sim(
            sfl_settings,
            opts,
            trainer_name="SFL global centralized-sim",
            return_package=True,
        )

        configure_clients_for_global(args.dataset_name)
        init_seed(args.seed)
        full_setting = get_setting(StandaloneCtx(0))
        _, _, test_set, global_model, _, _, _, _, _, scaler, _ = full_setting
        if global_model is None:
            raise NotImplementedError("SFL global upper-bound requires a full-graph model instance.")
        global_model.to(args.device)
        if package.get("global_state"):
            _load_matching_state(global_model, package["global_state"])
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        metrics = test_metrics(global_model, test_loader, scaler, args.device)
        row = result_row(spec, args.dataset_name, feature, "global", metrics)
        write_raw_result("global", row)
        write_summary_results("global", [row])
        return row

    if spec.method == "STAGCN-EC" and getattr(opts, "stagcn_global_mode", "full_graph") == "transfer_upper":
        configure_clients_for_local(args.dataset_name)
        args.nodes_per, split_name = _stagcn_flow_split(args.dataset_name, args.num_clients)
        neighbors = _stagcn_partition_neighbors(args.dataset_name, feature, args.nodes_per)
        print(
            f"[global] {spec.method} transfer_upper split={split_name} "
            f"num_clients={args.num_clients} sizes={[len(nodes) for nodes in args.nodes_per]} "
            f"neighbors={neighbors} communication=disabled",
            flush=True,
        )

        rows = []
        trained_states = {}
        for rank in range(args.num_clients):
            init_seed(args.seed)
            ctx = StandaloneCtx(rank)
            setting = get_setting(ctx)
            opts.current_rank = rank
            candidate_states = [
                (neighbor_rank, trained_states[neighbor_rank])
                for neighbor_rank in neighbors[rank]
                if neighbor_rank in trained_states
            ]
            metrics, trained_state = train_stagcn_transfer_upper(setting, opts, candidate_states, rank)
            trained_states[rank] = trained_state
            row = result_row(spec, args.dataset_name, feature, f"partition{rank}", metrics)
            row["method"] = "STAGCN-EC_transfer_upper"
            write_raw_result("global", row)
            rows.append(row)
        write_summary_results("global", rows)
        return rows

    if spec.method == "STAGCN-EC" and getattr(opts, "stagcn_global_mode", "full_graph") == "partitioned_upper":
        configure_clients_for_local(args.dataset_name)
        print(
            f"[global] {spec.method} partitioned_upper num_clients={args.num_clients} "
            "communication=disabled aggregation=sample_weighted_summary"
        )

        rows = []
        for rank in range(args.num_clients):
            init_seed(args.seed)
            ctx = StandaloneCtx(rank)
            setting = get_setting(ctx)
            opts.current_rank = rank
            metrics = train_baseline(setting, spec, opts)
            row = result_row(spec, args.dataset_name, feature, f"partition{rank}", metrics)
            row["method"] = "STAGCN-EC_partitioned_upper"
            write_raw_result("global", row)
            rows.append(row)
        write_summary_results("global", rows)
        return rows

    fedgtp_centralized_sim = (
        spec.method == "FedGTP"
        and getattr(opts, "fedgtp_global_mode", "centralized_sim") == "centralized_sim"
    )
    if spec.method == "FedmSSA":
        requested_clients = int(getattr(args, "num_clients", DEFAULT_LOCAL_CLIENTS) or DEFAULT_LOCAL_CLIENTS)
        args.num_clients = requested_clients
        args.nodes_per, split_name = partition_nodes_per_split(args.dataset_name, feature, args.num_clients)
        print(
            f"[global] {spec.method} centralized_sim split={split_name} "
            f"num_clients={args.num_clients} sizes={[len(nodes) for nodes in args.nodes_per]}",
            flush=True,
        )
        settings = []
        for rank in range(args.num_clients):
            init_seed(args.seed)
            settings.append(get_setting(StandaloneCtx(rank)))
        opts.current_rank = 0
        metrics = train_baseline(settings, spec, opts)
        row = result_row(spec, args.dataset_name, feature, "global", metrics)
        write_raw_result("global", row)
        write_summary_results("global", [row])
        return row
    if spec.method == "FedGTP" and not fedgtp_centralized_sim:
        configure_clients_for_global(args.dataset_name)
        print(
            f"[global] FedGTP centralized_upper num_clients={args.num_clients} "
            f"sizes={[len(nodes) for nodes in args.nodes_per]}"
        )
    elif spec.method in {"FedGTP", "FedOSTC"}:
        split_name = configure_clients_for_fedostc_centralized_sim(args.dataset_name, feature)
        print(
            f"[global] {spec.method} centralized_sim split={split_name} "
            f"num_clients={args.num_clients} sizes={[len(nodes) for nodes in args.nodes_per]}"
        )
    elif spec.method == "STFAM":
        if getattr(opts, "stfam_global_mode", "upper") == "centralized_sim":
            configure_clients_for_grid_quadrant_sim(args.dataset_name)
            print(
                f"[global] {spec.method} centralized_sim num_clients={args.num_clients} "
                f"sizes={[len(nodes) for nodes in args.nodes_per]}"
            )
        else:
            configure_clients_for_global(args.dataset_name)
            print(
                f"[global] {spec.method} centralized_upper num_clients={args.num_clients} "
                f"sizes={[len(nodes) for nodes in args.nodes_per]}"
            )
    elif spec.method == "FedSTN":
        configure_clients_for_grid_quadrant_sim(args.dataset_name)
        print(f"[global] {spec.method} centralized_sim num_clients={args.num_clients} sizes={[len(nodes) for nodes in args.nodes_per]}")
    elif spec.method == "FUELS" and getattr(opts, "fuels_global_mode", "upper") == "centralized_sim":
        requested_clients = int(getattr(args, "num_clients", DEFAULT_LOCAL_CLIENTS) or DEFAULT_LOCAL_CLIENTS)
        args.num_clients = requested_clients
        args.nodes_per, split_name = partition_nodes_per_split(args.dataset_name, feature, args.num_clients)
        print(
            f"[global] {spec.method} centralized_sim split={split_name} "
            f"num_clients={args.num_clients} sizes={[len(nodes) for nodes in args.nodes_per]}"
        )
    elif spec.method == "FUELS" and getattr(opts, "fuels_global_mode", "upper") == "partitioned_upper":
        requested_clients = int(getattr(args, "num_clients", DEFAULT_LOCAL_CLIENTS) or DEFAULT_LOCAL_CLIENTS)
        args.num_clients = requested_clients
        args.nodes_per, split_name = partition_nodes_per_split(args.dataset_name, feature, args.num_clients)
        print(
            f"[global] {spec.method} partitioned_upper split={split_name} "
            f"num_clients={args.num_clients} sizes={[len(nodes) for nodes in args.nodes_per]} "
            "communication=disabled prototype_exchange=disabled aggregation=client_mean_summary",
            flush=True,
        )

        rows = []
        for rank in range(args.num_clients):
            init_seed(args.seed)
            ctx = StandaloneCtx(rank)
            setting = get_setting(ctx)
            opts.current_rank = rank
            metrics = train_baseline(setting, spec, opts)
            row = result_row(spec, args.dataset_name, feature, f"partition{rank}", metrics)
            write_raw_result("global", row)
            rows.append(row)
        write_summary_results("global", rows)
        return rows
    else:
        configure_clients_for_global(args.dataset_name)

    init_seed(args.seed)
    ctx = StandaloneCtx(0)
    setting = (
        None
        if (
            spec.method in {"FedOSTC", "FedSTN"}
            or (spec.method == "STFAM" and getattr(opts, "stfam_global_mode", "upper") == "centralized_sim")
            or (spec.method == "FUELS" and getattr(opts, "fuels_global_mode", "upper") == "centralized_sim")
            or fedgtp_centralized_sim
        )
        else get_setting(ctx)
    )
    opts.current_rank = 0
    metrics = train_baseline(setting, spec, opts)
    row = result_row(spec, args.dataset_name, feature, "global", metrics)
    write_raw_result("global", row)
    write_summary_results("global", [row])
    return row


def iter_jobs(methods: Iterable[MethodSpec], datasets: List[str], features: Optional[List[str]] = None):
    for spec in methods:
        if "Grid" in spec.section:
            candidate_datasets = [d for d in datasets if d in GRID_DATASETS]
            candidate_features = ["flow"] if features is None else [f for f in features if f == "flow"]
        else:
            candidate_datasets = [d for d in datasets if d in GRAPH_DATASETS or d == "PeMS07"]
            candidate_features = GRAPH_FEATURES if features is None else features

        for dataset_name in candidate_datasets:
            for feature in candidate_features:
                yield spec, dataset_name, feature


def run_paradigm(paradigm):
    opts = parse_paradigm_cli()
    methods = selected_methods(opts.methods)
    if not methods:
        known = ", ".join(m.method for m in METHODS)
        print(f"[{paradigm}] No methods matched --methods {opts.methods!r}. Known methods: {known}")
        return

    default_datasets = GRAPH_DATASETS + GRID_DATASETS
    datasets = _split_csv(opts.datasets, default_datasets)
    features = None if opts.features == "all" else _split_csv(opts.features, GRAPH_FEATURES)

    if paradigm == "local":
        runner = run_one_local
    elif paradigm == "global":
        runner = run_one_global
    else:
        raise ValueError(f"Unknown paradigm: {paradigm}")

    os.makedirs(os.path.join(PROJECT_ROOT, f"{paradigm}_logs"), exist_ok=True)

    jobs = list(iter_jobs(methods, datasets, features))
    if not jobs:
        print(
            f"[{paradigm}] No jobs matched. "
            f"methods={opts.methods!r}, datasets={opts.datasets!r}, features={opts.features!r}"
        )
        return

    for spec, dataset_name, feature in jobs:
        log_path = job_log_path(paradigm, spec, dataset_name, feature)
        if (
            paradigm == "local"
            and spec.method == "STAGCN-EC"
            and stagcn_local_variant_name(opts) != "STAGCN-EC"
        ):
            log_dir = os.path.join(PROJECT_ROOT, "local_logs", stagcn_local_variant_name(opts))
            os.makedirs(log_dir, exist_ok=True)
            display_dataset = display_dataset_name(dataset_name)
            log_path = os.path.join(
                log_dir,
                f"{safe_name(display_dataset)}_{safe_name(feature)}.log",
            )
        with open(log_path, "a", encoding="utf-8") as log_file:
            tee_out = Tee(sys.stdout, log_file)
            tee_err = Tee(sys.stderr, log_file)
            with redirect_stdout(tee_out), redirect_stderr(tee_err):
                print(
                    f"[{paradigm}] section={spec.section} method={spec.method} "
                    f"model={spec.model_arg} dataset={dataset_name} feature={feature}"
                )
                print(
                    f"[{paradigm}] training_control max_epochs={args.epochs} "
                    f"patience={opts.patience} min_delta={opts.min_delta}"
                )
                if spec.method in BASELINE_TRAINING_POLICY:
                    keep, remove = BASELINE_TRAINING_POLICY[spec.method]
                    print(f"[{paradigm}] policy_keep={keep}")
                    print(f"[{paradigm}] policy_remove={remove}")
                print(f"[{paradigm}] log_file={log_path}")
                try:
                    runner(spec, dataset_name, feature, opts)
                except Exception as exc:
                    print(
                        f"[{paradigm}] SKIP/FAILED method={spec.method} "
                        f"dataset={dataset_name} feature={feature}: {exc}"
                    )
                    traceback.print_exc()
                    raise
                print(
                    f"[{paradigm}] DONE method={spec.method} "
                    f"dataset={display_dataset_name(dataset_name)} feature={feature}"
                )
