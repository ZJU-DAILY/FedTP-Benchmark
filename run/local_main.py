"""Local training entry point with synchronized FedmSSA early stopping."""

import copy

import torch
from torch.utils.data import DataLoader

from global_main import _tdlr_only_request, run_tdlr_streaming_paradigm
from paradigm_common import (
    GRAPH_DATASETS,
    GRID_DATASETS,
    GRAPH_FEATURES,
    StandaloneCtx,
    _split_csv,
    configure_clients_for_local,
    configure_dataset,
    configure_method,
    init_seed,
    iter_jobs,
    parse_paradigm_cli,
    result_row,
    run_paradigm,
    selected_methods,
    write_raw_result,
    write_summary_results,
)
from config_args import args


def _fedmssa_only_request(opts):
    raw = str(opts.methods).strip()
    return raw.lower() == "fedmssa"


def _aggregate_fedmssa_metrics(metrics_list):
    total_abs = sum(float(metrics["abs_error_sum"]) for metrics in metrics_list)
    total_sq = sum(float(metrics["sq_error_sum"]) for metrics in metrics_list)
    total_mape = sum(float(metrics["mape_error_sum"]) for metrics in metrics_list)
    total_elements = sum(int(metrics["elements"]) for metrics in metrics_list)
    total_mape_elements = sum(int(metrics["mape_elements"]) for metrics in metrics_list)
    mse = total_sq / max(total_elements, 1)
    return {
        "mae": total_abs / max(total_elements, 1),
        "mse": mse,
        "rmse": mse ** 0.5,
        "mape": total_mape / max(total_mape_elements, 1) if total_mape_elements else 0.0,
        "elements": total_elements,
        "abs_error_sum": total_abs,
        "sq_error_sum": total_sq,
        "mape_error_sum": total_mape,
        "mape_elements": total_mape_elements,
    }


def train_fedmssa_synchronized_local(settings, opts):
    """Independent FedmSSA clients with one sample-weighted validation stop rule.

    Phase 1 is strictly local for every client.  In phase 2, models never
    exchange parameters; only their validation error sums determine the shared
    best epoch and the shared early-stop decision.
    """
    from lib.fedmssa_trainer import (
        _build_split_observations,
        _datasets_from_observations,
        _fresh_optimizer,
        _loader_metrics,
        _run_federated_imputation,
        _train_one_epoch,
    )

    device = args.device
    clients = []
    for rank, setting in enumerate(settings):
        observation_pack = _build_split_observations(setting)
        local_basis, phase1_logs = _run_federated_imputation([observation_pack])
        train_ds, val_ds, test_ds = _datasets_from_observations(observation_pack, local_basis)
        model = setting[3].to(device)
        optimizer = setting[4] or _fresh_optimizer(model)
        loss_func = setting[5] or torch.nn.MSELoss().to(device)
        if hasattr(loss_func, "to"):
            loss_func = loss_func.to(device)
        clients.append({
            "rank": rank,
            "model": model,
            "optimizer": optimizer,
            "loss_func": loss_func,
            "scaler": setting[9],
            "train_loader": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True),
            "val_loader": DataLoader(val_ds, batch_size=args.batch_size, shuffle=False),
            "test_loader": DataLoader(test_ds, batch_size=args.batch_size, shuffle=False),
        })
        print(
            f"[FedmSSA synchronized-local] client{rank} phase1_rounds={len(phase1_logs)} "
            f"rank={local_basis.size(1)} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}",
            flush=True,
        )

    best_states = None
    best_val = float("inf")
    patience_count = 0
    for epoch in range(args.epochs):
        train_losses = []
        val_metrics_list = []
        for client in clients:
            train_loss, _ = _train_one_epoch(
                client["model"], client["optimizer"], client["loss_func"],
                client["train_loader"], device, max_batches=getattr(opts, "max_batches", 0),
            )
            train_losses.append(train_loss)
            val_metrics_list.append(
                _loader_metrics(client["model"], client["val_loader"], client["scaler"], device)
            )

        aggregate_val = _aggregate_fedmssa_metrics(val_metrics_list)
        val_mae = aggregate_val["mae"]
        if val_mae < best_val - float(opts.min_delta):
            best_val = val_mae
            best_states = [copy.deepcopy(client["model"].state_dict()) for client in clients]
            patience_count = 0
        else:
            patience_count += 1

        detail = ", ".join(
            f"c{client['rank']}={metrics['mae']:.4f}"
            for client, metrics in zip(clients, val_metrics_list)
        )
        print(
            f"[FedmSSA synchronized-local] Epoch {epoch} | "
            f"Avg Train Loss(Norm): {sum(train_losses) / max(len(train_losses), 1):.4f} | "
            f"Global Val MAE: {val_mae:.4f} | best={best_val:.4f} | "
            f"patience={patience_count}/{opts.patience} | {detail}",
            flush=True,
        )
        if patience_count >= opts.patience:
            print(f"[FedmSSA synchronized-local] Early stop at epoch {epoch}.", flush=True)
            break

    if best_states is not None:
        for client, state in zip(clients, best_states):
            client["model"].load_state_dict(state)
    return [
        _loader_metrics(client["model"], client["test_loader"], client["scaler"], device)
        for client in clients
    ]


def run_fedmssa_synchronized_local(opts):
    methods = selected_methods("FedmSSA")
    datasets = _split_csv(opts.datasets, GRAPH_DATASETS + GRID_DATASETS)
    features = None if opts.features == "all" else _split_csv(opts.features, GRAPH_FEATURES)

    from fate_main import get_setting

    for spec, dataset_name, feature in iter_jobs(methods, datasets, features):
        configure_method(spec)
        configure_dataset(dataset_name, feature)
        configure_clients_for_local(args.dataset_name)
        args.norm_scope = getattr(opts, "local_norm_scope", "global")
        opts.paradigm = "local"

        settings = []
        for rank in range(args.num_clients):
            init_seed(args.seed)
            settings.append(get_setting(StandaloneCtx(rank)))

        metrics_list = train_fedmssa_synchronized_local(settings, opts)
        rows = []
        for rank, metrics in enumerate(metrics_list):
            row = result_row(spec, args.dataset_name, feature, f"client{rank}", metrics)
            write_raw_result("local", row)
            rows.append(row)
        write_summary_results("local", rows)


if __name__ == "__main__":
    options = parse_paradigm_cli()
    if _tdlr_only_request(options):
        run_tdlr_streaming_paradigm("local", options)
    elif _fedmssa_only_request(options):
        run_fedmssa_synchronized_local(options)
    else:
        run_paradigm("local")
