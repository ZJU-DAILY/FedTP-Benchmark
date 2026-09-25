"""Global training entry point.

TDLR-family baselines use a streaming training protocol in ``fate_main.py``.
The regular paradigm dispatcher is intentionally retained for every other
baseline, while this entry point supplies the equivalent no-aggregation
streaming upper bound for TDLR, SEDLR, and TDLR_SEDLR.
"""

import copy
import math
from collections import deque

import torch
from torch.utils.data import DataLoader

from paradigm_common import (
    DEFAULT_LOCAL_CLIENTS,
    GRAPH_DATASETS,
    GRID_DATASETS,
    GRAPH_FEATURES,
    StandaloneCtx,
    _split_csv,
    configure_clients_for_global,
    configure_clients_for_local,
    configure_dataset,
    configure_method,
    current_run_id,
    display_dataset_name,
    init_seed,
    iter_jobs,
    parse_paradigm_cli,
    partition_nodes_per_split,
    result_row,
    run_paradigm,
    selected_methods,
    write_raw_result,
    write_summary_results,
)
from config_args import args
from lib.tdlr_sedlr_trainer import (
    _clone_batch_to_cpu,
    _eval_loader_normalized,
    _eval_loader_real,
    _optimize_lr_with_surrogate,
    _parse_tdlr_schedule,
    _recent_target_mean,
    _scheduled_lr,
    _sedlr_trigger,
    _set_optimizer_lr,
    _should_run_sedlr_bayes,
    _stream_round_batch,
    _train_batches,
)


TDLR_METHODS = {"TDLR", "SEDLR", "TDLR_SEDLR"}


def _tdlr_only_request(opts):
    """Use the specialized runner only when every requested method is TDLR-family."""
    raw = str(opts.methods).strip()
    if not raw or raw.lower() == "all":
        return False
    requested = {name.strip() for name in raw.split(",") if name.strip()}
    return bool(requested) and requested.issubset(TDLR_METHODS)


def _fedmssa_only_request(opts):
    return str(opts.methods).strip().lower() == "fedmssa"


def _fedmssa_uneven_phase1(observation_packs):
    """Federated phase-1 basis learning that supports unequal client widths.

    A page matrix has ``num_nodes * num_features * num_pages`` columns, so
    adding raw page matrices is invalid when grid partitions have different
    node counts (for example TaxiNYC: 20/20/20/15).  Bases are all L-by-rank,
    however, and can safely be aligned to a common rank then averaged.
    """
    from lib.fedmssa_trainer import _optimize_local_basis
    from lib.fedmssa_utils import build_initial_basis_from_observation, orthonormalize_basis

    initial_bases = [
        build_initial_basis_from_observation(
            pack["train"]["page_matrix_obs"],
            num_columns=pack["train"]["num_columns"],
            explicit_rank=getattr(args, "fedmssa_rank", 0),
            sv_scale=getattr(args, "fedmssa_sv_scale", 2.0),
        ).to(args.device)
        for pack in observation_packs
    ]
    common_rank = min(basis.size(1) for basis in initial_bases)
    aligned = [basis[:, :common_rank] for basis in initial_bases]
    consensus_basis = orthonormalize_basis(torch.stack(aligned, dim=0).mean(dim=0), rank=common_rank)
    server_velocity = torch.zeros_like(consensus_basis)
    round_logs = []

    for round_idx in range(max(int(getattr(args, "fedmssa_impute_rounds", 20)), 1)):
        local_bases = []
        metric_sums = {"recon": 0.0, "consensus": 0.0, "ortho": 0.0, "diag": 0.0, "total": 0.0}
        for pack in observation_packs:
            local_basis, metrics = _optimize_local_basis(pack["train"], consensus_basis)
            local_bases.append(local_basis)
            for key in metric_sums:
                metric_sums[key] += float(metrics.get(key, 0.0))

        average_basis = torch.stack(local_bases, dim=0).mean(dim=0)
        momentum = float(getattr(args, "fedmssa_server_momentum", 0.0))
        if momentum > 0.0:
            server_velocity = momentum * server_velocity + (1.0 - momentum) * (average_basis - consensus_basis)
            consensus_basis = consensus_basis + server_velocity
        else:
            consensus_basis = average_basis
        consensus_basis = orthonormalize_basis(consensus_basis, rank=common_rank)

        round_logs.append({
            "round": round_idx,
            **{key: value / max(len(local_bases), 1) for key, value in metric_sums.items()},
        })
    return consensus_basis.detach().cpu(), round_logs


def train_fedmssa_uneven_centralized_sim(settings, opts):
    """Existing FedmSSA centralized simulation with unequal grid partitions."""
    from lib.fedmssa_trainer import (
        _aggregate_state_dicts,
        _build_split_observations,
        _datasets_from_observations,
        _fresh_optimizer,
        _loader_metrics,
        _train_one_epoch,
    )

    device = args.device
    observation_packs = [
        _build_split_observations(setting, rank_offset=rank)
        for rank, setting in enumerate(settings)
    ]
    shared_basis, impute_logs = _fedmssa_uneven_phase1(observation_packs)
    clients = []
    for rank, (setting, observation_pack) in enumerate(zip(settings, observation_packs)):
        model = setting[3].to(device)
        optimizer = setting[4] or _fresh_optimizer(model)
        loss_func = setting[5] or torch.nn.MSELoss().to(device)
        if hasattr(loss_func, "to"):
            loss_func = loss_func.to(device)
        train_ds, val_ds, test_ds = _datasets_from_observations(observation_pack, shared_basis)
        clients.append({
            "rank": rank,
            "model": model,
            "optimizer": optimizer,
            "loss_func": loss_func,
            "scaler": setting[9],
            "train_loader": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True),
            "val_loader": DataLoader(val_ds, batch_size=args.batch_size, shuffle=False),
            "test_loader": DataLoader(test_ds, batch_size=args.batch_size, shuffle=False),
            "train_size": len(train_ds),
        })

    print(
        f"[FedmSSA global uneven-sim] clients={len(clients)} "
        f"sizes={[client['model'].num_nodes for client in clients]} "
        f"phase1_rank={shared_basis.size(1)} phase1_rounds={len(impute_logs)}",
        flush=True,
    )
    global_state = copy.deepcopy(clients[0]["model"].state_dict())
    best_global_state = copy.deepcopy(global_state)
    best_val = float("inf")
    patience_count = 0

    for round_idx in range(args.epochs):
        train_losses, local_states, sample_weights = [], [], []
        for client in clients:
            client["model"].load_state_dict(global_state)
            for _ in range(max(int(args.local_epochs), 1)):
                train_loss, _ = _train_one_epoch(
                    client["model"], client["optimizer"], client["loss_func"],
                    client["train_loader"], device, max_batches=getattr(opts, "max_batches", 0),
                )
            train_losses.append(train_loss)
            local_states.append(copy.deepcopy(client["model"].state_dict()))
            sample_weights.append(client["train_size"])

        global_state = _aggregate_state_dicts(local_states, sample_weights)
        val_abs, val_elements = 0.0, 0
        for client in clients:
            client["model"].load_state_dict(global_state)
            metrics = _loader_metrics(client["model"], client["val_loader"], client["scaler"], device)
            val_abs += float(metrics["abs_error_sum"])
            val_elements += int(metrics["elements"])
        val_mae = val_abs / max(val_elements, 1)

        if val_mae < best_val - float(opts.min_delta):
            best_val = val_mae
            best_global_state = copy.deepcopy(global_state)
            patience_count = 0
        else:
            patience_count += 1
        print(
            f"[FedmSSA global uneven-sim] Round {round_idx} | "
            f"Train Loss(Norm): {sum(train_losses) / max(len(train_losses), 1):.4f} | "
            f"Val MAE: {val_mae:.4f} | best={best_val:.4f} | "
            f"patience={patience_count}/{opts.patience}",
            flush=True,
        )
        if patience_count >= opts.patience:
            break

    totals = {"abs_error_sum": 0.0, "sq_error_sum": 0.0, "mape_error_sum": 0.0, "elements": 0, "mape_elements": 0}
    for client in clients:
        client["model"].load_state_dict(best_global_state)
        metrics = _loader_metrics(client["model"], client["test_loader"], client["scaler"], device)
        for key in totals:
            totals[key] += metrics[key]
    mse = totals["sq_error_sum"] / max(totals["elements"], 1)
    return {
        "mae": totals["abs_error_sum"] / max(totals["elements"], 1),
        "mse": mse,
        "rmse": mse ** 0.5,
        "mape": totals["mape_error_sum"] / max(totals["mape_elements"], 1) if totals["mape_elements"] else 0.0,
        **totals,
    }


def run_fedmssa_uneven_global(opts):
    methods = selected_methods("FedmSSA")
    datasets = _split_csv(opts.datasets, GRAPH_DATASETS + GRID_DATASETS)
    features = None if opts.features == "all" else _split_csv(opts.features, GRAPH_FEATURES)
    from fate_main import get_setting

    for spec, dataset_name, feature in iter_jobs(methods, datasets, features):
        configure_method(spec)
        configure_dataset(dataset_name, feature)
        args.norm_scope = getattr(opts, "global_norm_scope", "global")
        opts.paradigm = "global"
        requested_clients = int(getattr(args, "num_clients", DEFAULT_LOCAL_CLIENTS) or DEFAULT_LOCAL_CLIENTS)
        args.num_clients = requested_clients
        args.nodes_per, split_name = partition_nodes_per_split(args.dataset_name, feature, args.num_clients)
        print(
            f"[global] FedmSSA uneven-sim split={split_name} num_clients={args.num_clients} "
            f"sizes={[len(nodes) for nodes in args.nodes_per]}",
            flush=True,
        )
        settings = []
        for rank in range(args.num_clients):
            init_seed(args.seed)
            settings.append(get_setting(StandaloneCtx(rank)))
        metrics = train_fedmssa_uneven_centralized_sim(settings, opts)
        row = result_row(spec, args.dataset_name, feature, "global", metrics)
        write_raw_result("global", row)
        write_summary_results("global", [row])


def run_fedmssa_strict_global(opts):
    """Strict global FedmSSA: one full-grid model and no FedAvg."""
    from lib.fedmssa_trainer import train_fedmssa_standalone
    from fate_main import get_setting

    methods = selected_methods("FedmSSA")
    datasets = _split_csv(opts.datasets, GRAPH_DATASETS + GRID_DATASETS)
    features = None if opts.features == "all" else _split_csv(opts.features, GRAPH_FEATURES)

    for spec, dataset_name, feature in iter_jobs(methods, datasets, features):
        configure_method(spec)
        configure_dataset(dataset_name, feature)
        configure_clients_for_global(args.dataset_name)
        args.norm_scope = getattr(opts, "global_norm_scope", "global")
        opts.paradigm = "global"
        init_seed(args.seed)

        print(
            f"[global] FedmSSA strict-centralized num_clients={args.num_clients} "
            f"sizes={[len(nodes) for nodes in args.nodes_per]}",
            flush=True,
        )
        setting = get_setting(StandaloneCtx(0))
        metrics = train_fedmssa_standalone(setting, opts)
        row = result_row(spec, args.dataset_name, feature, "global", metrics)
        write_raw_result("global", row)
        write_summary_results("global", [row])


def train_tdlr_streaming_standalone(setting, opts, label):
    """Run the FATE TDLR/SED-LR client protocol without model aggregation."""
    (
        train_set,
        val_set,
        test_set,
        model,
        optimizer,
        loss_func,
        _,
        _,
        _,
        scaler,
        _,
    ) = setting

    model_name = str(args.model)
    use_tdlr = model_name in {"TDLR", "TDLR_SEDLR"}
    use_sedlr = model_name in {"SEDLR", "TDLR_SEDLR"}
    total_rounds = max(int(getattr(args, "tdlr_stream_rounds", 0) or args.epochs), 1)
    patience = int(getattr(args, "tdlr_patience", opts.patience))
    min_delta = float(getattr(args, "tdlr_min_delta", opts.min_delta))
    schedule = _parse_tdlr_schedule(getattr(args, "tdlr_schedule", "0:1.0"))
    pin_memory = isinstance(args.device, str) and args.device.startswith("cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)

    recent_batches = deque(maxlen=max(int(getattr(args, "tdlr_recent_buffer_size", 8)), 2))
    recent_target_stats = deque(maxlen=max(int(getattr(args, "tdlr_recent_buffer_size", 8)), 2))
    best_state = None
    best_val_mae = float("inf")
    best_round = -1
    patience_counter = 0
    last_trigger_round = -1
    last_sedlr_bayes_round = -1
    actual_rounds = 0

    print(
        f"[{label}] streaming standalone | method={model_name} rounds={total_rounds} "
        f"train={len(train_set)} val={len(val_set)} test={len(test_set)} "
        f"local_epochs={args.local_epochs}",
        flush=True,
    )

    for round_idx in range(total_rounds):
        current_batch = _stream_round_batch(train_set, round_idx, args)
        recent_batches.append(_clone_batch_to_cpu(current_batch))
        recent_target_stats.append(_recent_target_mean(current_batch))

        base_lr = float(args.lr)
        round_lr = _scheduled_lr(base_lr, round_idx, schedule) if use_tdlr else base_lr
        lr_policy = "tdlr_schedule" if use_tdlr else "base"
        trigger_reason = "disabled"
        sedlr_triggered = False

        if use_sedlr:
            sedlr_triggered, trigger_reason = _sedlr_trigger(
                round_idx, current_batch, recent_target_stats, last_trigger_round, args
            )
            if sedlr_triggered:
                last_trigger_round = round_idx
                if getattr(args, "sedlr_bayes_enable", False):
                    bayes_ready, _ = _should_run_sedlr_bayes(
                        round_idx, last_sedlr_bayes_round, args
                    )
                    if bayes_ready:
                        round_lr = _optimize_lr_with_surrogate(
                            model, optimizer, loss_func, list(recent_batches), args, round_lr
                        )
                        last_sedlr_bayes_round = round_idx
                        lr_policy = "sedlr_trigger_bayes"
                    else:
                        round_lr *= float(getattr(args, "sedlr_aggressive_mult", 2.0))
                        lr_policy = "sedlr_trigger_multiplier_fallback"
                else:
                    round_lr *= float(getattr(args, "sedlr_aggressive_mult", 2.0))
                    lr_policy = "sedlr_trigger_multiplier"
            else:
                round_lr *= float(getattr(args, "sedlr_calm_mult", 1.0))
                lr_policy = "sedlr_calm_multiplier"

        if use_tdlr and getattr(args, "tdlr_bayes_enable", False):
            every = max(int(getattr(args, "tdlr_bayes_every", 50)), 1)
            if round_idx > 0 and round_idx % every == 0:
                round_lr = _optimize_lr_with_surrogate(
                    model, optimizer, loss_func, list(recent_batches), args, round_lr
                )
                lr_policy = "tdlr_periodic_bayes"

        _set_optimizer_lr(optimizer, round_lr)
        round_loss = 0.0
        for _ in range(max(int(args.local_epochs), 1)):
            round_loss = _train_batches(model, optimizer, loss_func, [current_batch], args)

        val_mae, _, _, _, _, _ = _eval_loader_normalized(model, val_loader, args)
        actual_rounds = round_idx + 1
        if val_mae < best_val_mae - min_delta:
            best_val_mae = val_mae
            best_round = round_idx
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"[{label}] round={round_idx} loss={round_loss:.4f} val_mae={val_mae:.4f} "
            f"best_round={best_round} lr={round_lr:.6f} policy={lr_policy} "
            f"sedlr_triggered={int(sedlr_triggered)} reason={trigger_reason} "
            f"patience={patience_counter}/{patience}",
            flush=True,
        )
        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = _eval_loader_real(model, test_loader, scaler, args)
    metrics["best_round"] = best_round
    metrics["actual_rounds"] = actual_rounds
    return metrics


def run_tdlr_streaming_paradigm(paradigm, opts):
    """Run TDLR-family global/local paradigms with comparable streaming semantics."""
    methods = [spec for spec in selected_methods(opts.methods) if spec.method in TDLR_METHODS]
    datasets = _split_csv(opts.datasets, GRAPH_DATASETS + GRID_DATASETS)
    features = None if opts.features == "all" else _split_csv(opts.features, GRAPH_FEATURES)

    for spec, dataset_name, feature in iter_jobs(methods, datasets, features):
        configure_method(spec)
        configure_dataset(dataset_name, feature)
        args.norm_scope = (
            getattr(opts, "global_norm_scope", "global")
            if paradigm == "global"
            else getattr(opts, "local_norm_scope", "global")
        )
        opts.paradigm = paradigm

        if paradigm == "global":
            configure_clients_for_global(args.dataset_name)
            init_seed(args.seed)
            from fate_main import get_setting

            metrics = train_tdlr_streaming_standalone(
                get_setting(StandaloneCtx(0)), opts, "TDLR global"
            )
            row = result_row(spec, args.dataset_name, feature, "global", metrics)
            write_raw_result("global", row)
            write_summary_results("global", [row])
            continue

        configure_clients_for_local(args.dataset_name)
        from fate_main import get_setting

        rows = []
        for rank in range(args.num_clients):
            init_seed(args.seed)
            opts.current_rank = rank
            metrics = train_tdlr_streaming_standalone(
                get_setting(StandaloneCtx(rank)), opts, f"TDLR local client{rank}"
            )
            row = result_row(spec, args.dataset_name, feature, f"client{rank}", metrics)
            write_raw_result("local", row)
            rows.append(row)
        write_summary_results("local", rows)


if __name__ == "__main__":
    options = parse_paradigm_cli()
    if _tdlr_only_request(options):
        run_tdlr_streaming_paradigm("global", options)
    elif _fedmssa_only_request(options):
        run_fedmssa_strict_global(options)
    else:
        run_paradigm("global")
