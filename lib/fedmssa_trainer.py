import copy
import time

import torch
from torch.utils.data import DataLoader

from config_args import args
from lib.fedmssa_utils import (
    build_initial_basis_from_observation,
    build_page_observation,
    decorrelation_penalty,
    denoise_raw_series_with_observation,
    masked_reconstruction_loss,
    orthogonality_penalty,
    orthonormalize_basis,
    reconstruct_series_from_windows,
    tensor_dataset_from_series,
)
from lib.utils import align_prediction_and_target, extract_ctx_data
from privacy.protection import l2_norm
from privacy.runtime_protection import protected_arbiter_put
from privacy.attack_trace import capture_revised_quantized_prediction


def _clone_state_dict(state_dict):
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def _state_delta(current, reference):
    return {key: current[key].detach().cpu() - reference[key].detach().cpu() for key in current}


def _state_add(reference, delta):
    return {key: reference[key].detach().cpu() + delta[key].detach().cpu() for key in reference}


def _client_dp_clip(ctx, tag, value):
    """Calibrate a separate DP clipping bound for one FedmSSA payload type."""
    if args.protection != "dp":
        return None
    configured = float(getattr(args, "dp_clip_norm", 0.0))
    if configured > 0:
        return configured
    ctx.arbiter.put(f"{tag}_norm", float(l2_norm(value).item()))
    clip = extract_ctx_data(ctx, ctx.arbiter.get(f"{tag}_clip"))
    if isinstance(clip, (list, tuple)):
        clip = clip[0]
    clip = float(clip)
    print(f"[DPCalibration] FedmSSA rank={ctx.rank} tag={tag} clip_norm={clip:.8f}", flush=True)
    return clip


def _series_splits_from_setting(setting):
    train_set, val_set, test_set = setting[0], setting[1], setting[2]
    return (
        reconstruct_series_from_windows(train_set),
        reconstruct_series_from_windows(val_set),
        reconstruct_series_from_windows(test_set),
    )


def _build_split_observations(setting, rank_offset=0):
    train_raw, val_raw, test_raw = _series_splits_from_setting(setting)
    base_seed = int(args.seed) + int(rank_offset) * 1000
    ratio = float(getattr(args, "fedmssa_missing_ratio", 0.0))
    return {
        "train": build_page_observation(
            train_raw,
            page_length=args.fedmssa_page_length,
            missing_ratio=ratio,
            seed=base_seed + 11,
        ),
        "val": build_page_observation(
            val_raw,
            page_length=args.fedmssa_page_length,
            missing_ratio=ratio,
            seed=base_seed + 23,
        ),
        "test": build_page_observation(
            test_raw,
            page_length=args.fedmssa_page_length,
            missing_ratio=ratio,
            seed=base_seed + 37,
        ),
    }


def _initialize_consensus_basis(observations):
    page_sum = None
    total_columns = 0
    for pack in observations:
        page_obs = pack["train"]["page_matrix_obs"]
        page_sum = page_obs if page_sum is None else page_sum + page_obs
        total_columns += int(pack["train"]["num_columns"])
    if page_sum is None:
        raise RuntimeError("FedmSSA could not initialize the global consensus basis.")
    return build_initial_basis_from_observation(
        page_sum / max(len(observations), 1),
        num_columns=total_columns,
        explicit_rank=getattr(args, "fedmssa_rank", 0),
        sv_scale=getattr(args, "fedmssa_sv_scale", 2.0),
    )


def _local_phase1_objective(observation, basis, consensus_basis):
    page_obs = observation["page_matrix_obs"].to(basis.device)
    obs_mask = observation["obs_mask"].to(basis.device)
    recon = masked_reconstruction_loss(page_obs, obs_mask, basis)
    consensus = (basis - consensus_basis).pow(2).mean()
    ortho = orthogonality_penalty(basis)
    diag = decorrelation_penalty(page_obs, obs_mask, basis)
    total = (
        recon
        + float(getattr(args, "fedmssa_consensus_weight", 1.0)) * consensus
        + float(getattr(args, "fedmssa_ortho_weight", 1.0)) * ortho
        + float(getattr(args, "fedmssa_diag_weight", 0.2)) * diag
    )
    metrics = {
        "recon": float(recon.detach().cpu()),
        "consensus": float(consensus.detach().cpu()),
        "ortho": float(ortho.detach().cpu()),
        "diag": float(diag.detach().cpu()),
        "total": float(total.detach().cpu()),
    }
    return total, metrics


def _optimize_local_basis(observation, consensus_basis):
    basis = consensus_basis.detach().clone()
    basis = orthonormalize_basis(basis, rank=basis.size(1))
    basis = basis.requires_grad_(True)
    optimizer = torch.optim.SGD([basis], lr=float(getattr(args, "fedmssa_impute_lr", 5e-2)))
    last_metrics = None

    for _ in range(max(int(getattr(args, "fedmssa_impute_local_steps", 10)), 1)):
        optimizer.zero_grad()
        objective, metrics = _local_phase1_objective(observation, basis, consensus_basis)
        objective.backward()
        optimizer.step()
        with torch.no_grad():
            basis.copy_(orthonormalize_basis(basis.detach(), rank=consensus_basis.size(1)))
        last_metrics = metrics

    return basis.detach().clone(), last_metrics or {}


def _run_federated_imputation(observation_packs):
    consensus_basis = _initialize_consensus_basis(observation_packs)
    server_velocity = torch.zeros_like(consensus_basis)
    round_logs = []

    for round_idx in range(max(int(getattr(args, "fedmssa_impute_rounds", 20)), 1)):
        local_bases = []
        total_metrics = {"recon": 0.0, "consensus": 0.0, "ortho": 0.0, "diag": 0.0, "total": 0.0}

        for pack in observation_packs:
            local_basis, metrics = _optimize_local_basis(pack["train"], consensus_basis)
            local_bases.append(local_basis)
            for key in total_metrics:
                total_metrics[key] += float(metrics.get(key, 0.0))

        avg_basis = torch.stack(local_bases, dim=0).mean(dim=0)
        momentum = float(getattr(args, "fedmssa_server_momentum", 0.0))
        if momentum > 0.0:
            server_velocity = momentum * server_velocity + (1.0 - momentum) * (avg_basis - consensus_basis)
            consensus_basis = consensus_basis + server_velocity
        else:
            consensus_basis = avg_basis
        consensus_basis = orthonormalize_basis(consensus_basis, rank=consensus_basis.size(1))

        client_count = max(len(local_bases), 1)
        round_metrics = {key: value / client_count for key, value in total_metrics.items()}
        round_metrics["round"] = round_idx
        round_logs.append(round_metrics)

    return consensus_basis, round_logs


def _datasets_from_observations(observation_pack, basis):
    train_series = denoise_raw_series_with_observation(
        observation_pack["train"],
        basis,
        page_length=args.fedmssa_page_length,
    )
    val_series = denoise_raw_series_with_observation(
        observation_pack["val"],
        basis,
        page_length=args.fedmssa_page_length,
    )
    test_series = denoise_raw_series_with_observation(
        observation_pack["test"],
        basis,
        page_length=args.fedmssa_page_length,
    )
    return (
        tensor_dataset_from_series(train_series, args.t_in, args.t_out, args.device),
        tensor_dataset_from_series(val_series, args.t_in, args.t_out, args.device),
        tensor_dataset_from_series(test_series, args.t_in, args.t_out, args.device),
    )


def _loader_metrics(model, loader, scaler, device):
    model.eval()
    total_abs, total_sq = 0.0, 0.0
    total_mape, total_elements, total_mape_elements = 0.0, 0, 0

    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            pred = model(x)
            pred, y = align_prediction_and_target(pred, y)

            pred_real = scaler.inverse_transform(pred).detach().cpu().numpy()
            y_real = scaler.inverse_transform(y).detach().cpu().numpy()
            diff = pred_real - y_real

            total_abs += float(abs(diff).sum())
            total_sq += float((diff ** 2).sum())
            total_elements += int(y_real.size)

            mask = y_real > 0.5
            valid = int(mask.sum())
            if valid > 0:
                total_mape += float((abs(diff[mask]) / y_real[mask]).sum())
                total_mape_elements += valid

    mae = total_abs / max(total_elements, 1)
    mse = total_sq / max(total_elements, 1)
    rmse = mse ** 0.5
    mape = total_mape / max(total_mape_elements, 1) if total_mape_elements > 0 else 0.0
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "elements": total_elements,
        "abs_error_sum": total_abs,
        "sq_error_sum": total_sq,
        "mape_error_sum": total_mape,
        "mape_elements": total_mape_elements,
    }


def _train_one_epoch(model, optimizer, loss_func, loader, device, max_batches=None):
    model.train()
    total_loss = 0.0
    steps = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        x, y = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        pred = model(x)
        pred, y = align_prediction_and_target(pred, y)
        loss = loss_func(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        steps += 1
    return total_loss / max(steps, 1), steps


def _fresh_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)


def train_fedmssa_standalone(setting, opts):
    train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = setting
    device = args.device
    model.to(device)
    if optimizer is None:
        optimizer = _fresh_optimizer(model)
    if loss_func is None:
        loss_func = torch.nn.MSELoss().to(device)
    elif hasattr(loss_func, "to"):
        loss_func = loss_func.to(device)

    observation_pack = _build_split_observations(setting)
    basis, impute_logs = _run_federated_imputation([observation_pack])
    denoised_train, denoised_val, denoised_test = _datasets_from_observations(observation_pack, basis)
    train_loader = DataLoader(denoised_train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(denoised_val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(denoised_test, batch_size=args.batch_size, shuffle=False)

    best_state = None
    best_val = float("inf")
    patience_count = 0

    if impute_logs:
        last = impute_logs[-1]
        print(
            f"[FedmSSA phase1 local] rounds={len(impute_logs)} rank={basis.size(1)} "
            f"missing_ratio={getattr(args, 'fedmssa_missing_ratio', 0.0):.2f} "
            f"recon={last['recon']:.6f} consensus={last['consensus']:.6f} "
            f"ortho={last['ortho']:.6f} diag={last['diag']:.6f}",
            flush=True,
        )
    print(
        f"[FedmSSA standalone] page_length={args.fedmssa_page_length} rank={basis.size(1)} "
        f"train_batches={len(train_loader)} val_batches={len(val_loader)} test_batches={len(test_loader)}",
        flush=True,
    )

    for epoch in range(args.epochs):
        train_loss, _ = _train_one_epoch(
            model,
            optimizer,
            loss_func,
            train_loader,
            device,
            max_batches=getattr(opts, "max_batches", 0),
        )
        val_metrics = _loader_metrics(model, val_loader, scaler, device)
        val_mae = val_metrics["mae"]

        if val_mae < best_val - getattr(opts, "min_delta", 0.0):
            best_val = val_mae
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= opts.patience:
                print(
                    f"[FedmSSA standalone] Epoch {epoch} | Train Loss(Norm): {train_loss:.4f} "
                    f"| Val MAE: {val_mae:.4f} | best={best_val:.4f} "
                    f"| patience={patience_count}/{opts.patience} | early_stop=True",
                    flush=True,
                )
                break

        print(
            f"[FedmSSA standalone] Epoch {epoch} | Train Loss(Norm): {train_loss:.4f} "
            f"| Val MAE: {val_mae:.4f} | best={best_val:.4f} "
            f"| patience={patience_count}/{opts.patience}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return _loader_metrics(model, test_loader, scaler, device)


def _aggregate_state_dicts(state_dicts, weights):
    total_weight = float(sum(weights))
    result = {}
    for key in state_dicts[0]:
        aggregated = None
        for state_dict, weight in zip(state_dicts, weights):
            contribution = state_dict[key].detach().cpu() * (float(weight) / max(total_weight, 1.0))
            aggregated = contribution if aggregated is None else aggregated + contribution
        result[key] = aggregated
    return result


def train_fedmssa_centralized_sim(settings, opts):
    device = args.device
    observation_packs = [_build_split_observations(setting, rank_offset=rank) for rank, setting in enumerate(settings)]
    shared_basis, impute_logs = _run_federated_imputation(observation_packs)
    client_packages = []

    for rank, setting in enumerate(settings):
        model = setting[3].to(device)
        optimizer = setting[4] or _fresh_optimizer(model)
        loss_func = setting[5] or torch.nn.MSELoss().to(device)
        if hasattr(loss_func, "to"):
            loss_func = loss_func.to(device)
        scaler = setting[9]
        train_ds, val_ds, test_ds = _datasets_from_observations(observation_packs[rank], shared_basis)
        client_packages.append({
            "rank": rank,
            "model": model,
            "optimizer": optimizer,
            "loss_func": loss_func,
            "scaler": scaler,
            "train_loader": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True),
            "val_loader": DataLoader(val_ds, batch_size=args.batch_size, shuffle=False),
            "test_loader": DataLoader(test_ds, batch_size=args.batch_size, shuffle=False),
            "train_size": len(train_ds),
        })

    global_state = copy.deepcopy(client_packages[0]["model"].state_dict())
    best_global_state = copy.deepcopy(global_state)
    best_val = float("inf")
    patience_count = 0

    if impute_logs:
        last = impute_logs[-1]
        print(
            f"[FedmSSA phase1 global] rounds={len(impute_logs)} rank={shared_basis.size(1)} "
            f"missing_ratio={getattr(args, 'fedmssa_missing_ratio', 0.0):.2f} "
            f"recon={last['recon']:.6f} consensus={last['consensus']:.6f} "
            f"ortho={last['ortho']:.6f} diag={last['diag']:.6f}",
            flush=True,
        )
    print(
        f"[FedmSSA global] clients={len(client_packages)} page_length={args.fedmssa_page_length} "
        f"rank={shared_basis.size(1)} local_epochs={args.local_epochs}",
        flush=True,
    )

    for round_idx in range(args.epochs):
        train_losses = []
        local_states = []
        sample_weights = []

        for client in client_packages:
            client["model"].load_state_dict(global_state)
            for _ in range(max(int(args.local_epochs), 1)):
                train_loss, _ = _train_one_epoch(
                    client["model"],
                    client["optimizer"],
                    client["loss_func"],
                    client["train_loader"],
                    device,
                    max_batches=getattr(opts, "max_batches", 0),
                )
            train_losses.append(train_loss)
            local_states.append(copy.deepcopy(client["model"].state_dict()))
            sample_weights.append(client["train_size"])

        global_state = _aggregate_state_dicts(local_states, sample_weights)
        for client in client_packages:
            client["model"].load_state_dict(global_state)

        val_abs_sum, val_elements = 0.0, 0
        for client in client_packages:
            val_metrics = _loader_metrics(client["model"], client["val_loader"], client["scaler"], device)
            val_abs_sum += float(val_metrics["abs_error_sum"])
            val_elements += int(val_metrics["elements"])

        global_val_mae = val_abs_sum / max(val_elements, 1)
        mean_train_loss = sum(train_losses) / max(len(train_losses), 1)

        if global_val_mae < best_val - getattr(opts, "min_delta", 0.0):
            best_val = global_val_mae
            best_global_state = copy.deepcopy(global_state)
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= opts.patience:
                print(
                    f"[FedmSSA global] Round {round_idx} | Train Loss(Norm): {mean_train_loss:.4f} "
                    f"| Val MAE: {global_val_mae:.4f} | best={best_val:.4f} "
                    f"| patience={patience_count}/{opts.patience} | early_stop=True",
                    flush=True,
                )
                break

        print(
            f"[FedmSSA global] Round {round_idx} | Train Loss(Norm): {mean_train_loss:.4f} "
            f"| Val MAE: {global_val_mae:.4f} | best={best_val:.4f} "
            f"| patience={patience_count}/{opts.patience}",
            flush=True,
        )

    total = {
        "abs_error_sum": 0.0,
        "sq_error_sum": 0.0,
        "mape_error_sum": 0.0,
        "elements": 0,
        "mape_elements": 0,
    }
    for client in client_packages:
        client["model"].load_state_dict(best_global_state)
        metrics = _loader_metrics(client["model"], client["test_loader"], client["scaler"], device)
        total["abs_error_sum"] += float(metrics["abs_error_sum"])
        total["sq_error_sum"] += float(metrics["sq_error_sum"])
        total["mape_error_sum"] += float(metrics["mape_error_sum"])
        total["elements"] += int(metrics["elements"])
        total["mape_elements"] += int(metrics["mape_elements"])

    mae = total["abs_error_sum"] / max(total["elements"], 1)
    mse = total["sq_error_sum"] / max(total["elements"], 1)
    rmse = mse ** 0.5
    mape = total["mape_error_sum"] / max(total["mape_elements"], 1) if total["mape_elements"] > 0 else 0.0
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "elements": total["elements"],
        "abs_error_sum": total["abs_error_sum"],
        "sq_error_sum": total["sq_error_sum"],
        "mape_error_sum": total["mape_error_sum"],
        "mape_elements": total["mape_elements"],
    }


def train_fedmssa_task(ctx, args, setting=None, get_setting_fn=None):
    """
    Authentic federated FedmSSA path for run/fate_main.py.
    Phase 1:
      clients optimize local subspaces U_i against the server consensus Z.
    Phase 2:
      clients train the GRU predictor on denoised local data and the server
      performs sample-weighted FedAvg with global early stopping.
    """
    if setting is None and get_setting_fn is not None:
        setting = get_setting_fn(ctx)

    if not ctx.is_on_arbiter and setting is None:
        raise RuntimeError("FedmSSA client requires a setting from get_setting().")

    def _collect_client_payloads(tag):
        guest_payload = ctx.guest.get(tag)
        host_payloads = ctx.hosts.get(tag)
        if not isinstance(host_payloads, list):
            host_payloads = [host_payloads]
        return [guest_payload] + host_payloads

    def _broadcast_to_clients(tag, payloads):
        ctx.guest.put(tag, payloads[0])
        host_payloads = payloads[1:]
        if len(host_payloads) == 1:
            ctx.hosts.put(tag, host_payloads[0])
        else:
            ctx.hosts.put(tag, host_payloads)

    def _server_dp_clip(tag):
        """Return a per-payload clip bound and broadcast it to all clients."""
        configured = float(getattr(args, "dp_clip_norm", 0.0))
        if configured > 0:
            return configured
        norms = _collect_client_payloads(f"{tag}_norm")
        clip = float(torch.quantile(torch.tensor([float(value) for value in norms]), 0.9).item())
        _broadcast_to_clients(f"{tag}_clip", [clip] * args.num_clients)
        print(f"[DPCalibration] FedmSSA arbiter tag={tag} clip_norm={clip:.8f}", flush=True)
        return clip

    if ctx.is_on_arbiter:
        print(f"[FedmSSA Server] starting authentic federated training with {args.num_clients} clients.", flush=True)

        if args.protection == "dp":
            init_clip = _server_dp_clip("fedmssa_dp_init_basis")
            init_payloads = _collect_client_payloads("fedmssa_dp_init_basis")
        else:
            init_clip = None
            init_payloads = _collect_client_payloads("fedmssa_init_basis")
        init_bases = [payload["basis"].to(args.device) for payload in init_payloads]
        shared_basis = orthonormalize_basis(torch.stack(init_bases, dim=0).mean(dim=0), rank=init_bases[0].size(1))
        shared_basis_cpu = shared_basis.detach().cpu()
        _broadcast_to_clients("fedmssa_phase1_basis_0", [shared_basis_cpu] * args.num_clients)

        last_phase1_metrics = None
        phase1_clip = None
        for round_idx in range(max(int(getattr(args, "fedmssa_impute_rounds", 20)), 1)):
            if args.protection == "dp" and phase1_clip is None:
                phase1_clip = _server_dp_clip("fedmssa_dp_phase1_delta")
            phase1_tag = (
                f"fedmssa_dp_phase1_delta_{round_idx}"
                if args.protection == "dp" else f"fedmssa_phase1_local_{round_idx}"
            )
            phase1_payloads = _collect_client_payloads(phase1_tag)
            if args.protection == "dp":
                local_bases = [
                    (shared_basis_cpu + payload["basis"]).to(args.device)
                    for payload in phase1_payloads
                ]
            else:
                local_bases = [payload["basis"].to(args.device) for payload in phase1_payloads]
            shared_basis = orthonormalize_basis(torch.stack(local_bases, dim=0).mean(dim=0), rank=shared_basis.size(1))
            shared_basis_cpu = shared_basis.detach().cpu()

            phase1_metrics = {"recon": 0.0, "consensus": 0.0, "ortho": 0.0, "diag": 0.0, "total": 0.0}
            for payload in phase1_payloads:
                metrics = payload.get("metrics", {})
                for key in phase1_metrics:
                    phase1_metrics[key] += float(metrics.get(key, 0.0))
            client_count = max(len(phase1_payloads), 1)
            last_phase1_metrics = {key: value / client_count for key, value in phase1_metrics.items()}
            print(
                f"[FedmSSA Server] phase1 round={round_idx} "
                f"recon={last_phase1_metrics['recon']:.6f} "
                f"consensus={last_phase1_metrics['consensus']:.6f} "
                f"ortho={last_phase1_metrics['ortho']:.6f} "
                f"diag={last_phase1_metrics['diag']:.6f}",
                flush=True,
            )

            _broadcast_to_clients(
                f"fedmssa_phase1_basis_{round_idx + 1}",
                [shared_basis_cpu] * args.num_clients,
            )

        best_global_state = None
        best_val_mae = float("inf")
        patience_count = 0
        should_stop = False
        actual_rounds = 0
        shared_basis_elements = int(shared_basis.numel())
        phase2_clip = None
        if args.protection == "dp":
            # Model initialization is seed-derived/public.  It is only a
            # reference for reconstructing the first protected local delta.
            initial_states = _collect_client_payloads("fedmssa_dp_initial_model")
            global_state = _clone_state_dict(initial_states[0]["weights"])

        for round_idx in range(args.epochs):
            if args.protection == "dp" and phase2_clip is None:
                phase2_clip = _server_dp_clip("fedmssa_dp_phase2_delta")
            round_tag = (
                f"fedmssa_dp_phase2_delta_{round_idx}"
                if args.protection == "dp" else f"fedmssa_phase2_round_{round_idx}"
            )
            round_payloads = _collect_client_payloads(round_tag)
            if args.protection == "dp":
                for payload in round_payloads:
                    payload["weights"] = _state_add(global_state, payload["weights"])
            state_dicts = [payload["weights"] for payload in round_payloads]
            sample_weights = [int(payload["data_size"]) for payload in round_payloads]
            global_state = _aggregate_state_dicts(state_dicts, sample_weights)

            total_abs = sum(float(payload["val_abs_error_sum"]) for payload in round_payloads)
            total_elements = sum(int(payload["val_elements"]) for payload in round_payloads)
            global_val_mae = total_abs / max(total_elements, 1)
            actual_rounds = round_idx + 1

            if global_val_mae < best_val_mae - float(getattr(args, "fedmssa_min_delta", 0.0)):
                best_val_mae = global_val_mae
                best_global_state = copy.deepcopy(global_state)
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= int(getattr(args, "fedmssa_patience", 20)):
                    should_stop = True

            response_payload = {
                "weights": global_state,
                "should_stop": should_stop,
                "global_val_mae": global_val_mae,
                "best_val_mae": best_val_mae,
                "round": round_idx,
            }
            _broadcast_to_clients(
                f"fedmssa_phase2_response_{round_idx}",
                [response_payload] * args.num_clients,
            )

            print(
                f"[FedmSSA Server] phase2 round={round_idx} "
                f"global_val_mae={global_val_mae:.4f} best={best_val_mae:.4f} "
                f"patience={patience_count}/{getattr(args, 'fedmssa_patience', 20)} "
                f"early_stop={should_stop}",
                flush=True,
            )

            if should_stop:
                break

        if best_global_state is None:
            best_global_state = copy.deepcopy(global_state)

        final_payload = {
            "weights": best_global_state,
            "phase1_basis_elements": shared_basis_elements,
            "phase1_rounds": max(int(getattr(args, "fedmssa_impute_rounds", 20)), 1),
            "phase2_rounds": actual_rounds,
        }
        _broadcast_to_clients("fedmssa_final_payload", [final_payload] * args.num_clients)
        return None

    train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = setting
    device = args.device
    model.to(device)
    if optimizer is None:
        optimizer = _fresh_optimizer(model)
    if loss_func is None:
        loss_func = torch.nn.MSELoss().to(device)
    elif hasattr(loss_func, "to"):
        loss_func = loss_func.to(device)

    observation_pack = _build_split_observations(setting, rank_offset=ctx.rank)
    local_init_basis = build_initial_basis_from_observation(
        observation_pack["train"]["page_matrix_obs"],
        num_columns=observation_pack["train"]["num_columns"],
        explicit_rank=getattr(args, "fedmssa_rank", 0),
        sv_scale=getattr(args, "fedmssa_sv_scale", 2.0),
    ).cpu()
    if args.protection == "dp":
        init_clip = _client_dp_clip(ctx, "fedmssa_dp_init_basis", {"basis": local_init_basis})
        protected_arbiter_put(
            ctx, args, "fedmssa_dp_init_basis", {"basis": local_init_basis}, clip_norm=init_clip
        )
    else:
        ctx.arbiter.put("fedmssa_init_basis", {"basis": local_init_basis})

    consensus_basis = extract_ctx_data(ctx, ctx.arbiter.get("fedmssa_phase1_basis_0")).to(device)
    rank = consensus_basis.size(1)
    last_phase1_metrics = None
    phase1_clip = None
    for round_idx in range(max(int(getattr(args, "fedmssa_impute_rounds", 20)), 1)):
        local_basis, metrics = _optimize_local_basis(observation_pack["train"], consensus_basis)
        last_phase1_metrics = metrics
        if args.protection == "dp":
            basis_delta = local_basis.detach().cpu() - consensus_basis.detach().cpu()
            if phase1_clip is None:
                phase1_clip = _client_dp_clip(ctx, "fedmssa_dp_phase1_delta", {"basis": basis_delta})
            payload = {"basis": basis_delta, "metrics": metrics}
            protected_arbiter_put(
                ctx, args, f"fedmssa_dp_phase1_delta_{round_idx}", payload, clip_norm=phase1_clip
            )
        else:
            payload = {"basis": local_basis.detach().cpu(), "metrics": metrics}
            ctx.arbiter.put(f"fedmssa_phase1_local_{round_idx}", payload)
        consensus_basis = extract_ctx_data(ctx, ctx.arbiter.get(f"fedmssa_phase1_basis_{round_idx + 1}")).to(device)

    if last_phase1_metrics is not None:
        print(
            f"[FedmSSA Client {ctx.rank}] phase1 done rounds={getattr(args, 'fedmssa_impute_rounds', 20)} "
            f"rank={rank} missing_ratio={getattr(args, 'fedmssa_missing_ratio', 0.0):.2f} "
            f"recon={last_phase1_metrics['recon']:.6f} consensus={last_phase1_metrics['consensus']:.6f} "
            f"ortho={last_phase1_metrics['ortho']:.6f} diag={last_phase1_metrics['diag']:.6f}",
            flush=True,
        )

    denoised_train, denoised_val, denoised_test = _datasets_from_observations(observation_pack, consensus_basis.cpu())
    train_loader = DataLoader(denoised_train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(denoised_val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(denoised_test, batch_size=args.batch_size, shuffle=False)
    print(
        f"[FedmSSA Client {ctx.rank}] phase2 train_batches={len(train_loader)} "
        f"val_batches={len(val_loader)} test_batches={len(test_loader)}",
        flush=True,
    )

    total_train_time = 0.0
    total_val_time = 0.0
    actual_rounds = 0
    phase2_clip = None
    if args.protection == "dp":
        ctx.arbiter.put("fedmssa_dp_initial_model", {"weights": _clone_state_dict(model.state_dict())})
    for round_idx in range(args.epochs):
        round_start_state = _clone_state_dict(model.state_dict())
        # The revised monitoring side channel is emitted before any local
        # optimizer step, so its public model snapshot is replayable exactly.
        trace_batch = next(iter(train_loader))
        trace_x = trace_batch[0].to(device)
        with torch.no_grad():
            trace_prediction = model(trace_x)
        capture_revised_quantized_prediction(
            ctx, args, f"fedmssa_prediction_{round_idx}",
            prediction=trace_prediction, model_state_dict=round_start_state,
        )
        train_start = time.time()
        train_loss = 0.0
        for _ in range(max(int(args.local_epochs), 1)):
            epoch_loss, _ = _train_one_epoch(model, optimizer, loss_func, train_loader, device)
            train_loss = epoch_loss
        total_train_time += time.time() - train_start

        val_start = time.time()
        val_metrics = _loader_metrics(model, val_loader, scaler, device)
        total_val_time += time.time() - val_start
        actual_rounds = round_idx + 1

        payload = {
            "weights": _clone_state_dict(model.state_dict()),
            "data_size": len(denoised_train),
            "val_abs_error_sum": float(val_metrics["abs_error_sum"]),
            "val_elements": int(val_metrics["elements"]),
        }
        if args.protection == "dp":
            payload["weights"] = _state_delta(payload["weights"], round_start_state)
            if phase2_clip is None:
                phase2_clip = _client_dp_clip(ctx, "fedmssa_dp_phase2_delta", payload["weights"])
            protected_arbiter_put(
                ctx, args, f"fedmssa_dp_phase2_delta_{round_idx}", payload, clip_norm=phase2_clip
            )
        else:
            ctx.arbiter.put(f"fedmssa_phase2_round_{round_idx}", payload)
        response = extract_ctx_data(ctx, ctx.arbiter.get(f"fedmssa_phase2_response_{round_idx}"))
        model.load_state_dict(response["weights"], strict=True)

        print(
            f"[FedmSSA Client {ctx.rank}] round={round_idx} train_loss={train_loss:.4f} "
            f"local_val_mae={val_metrics['mae']:.4f} global_val_mae={response['global_val_mae']:.4f} "
            f"best={response['best_val_mae']:.4f} early_stop={response['should_stop']}",
            flush=True,
        )

        if response["should_stop"]:
            break

    final_payload = extract_ctx_data(ctx, ctx.arbiter.get("fedmssa_final_payload"))
    model.load_state_dict(final_payload["weights"], strict=True)

    test_start = time.time()
    test_metrics = _loader_metrics(model, test_loader, scaler, device)
    eff_test_time = time.time() - test_start

    params_count = sum(p.numel() for p in model.parameters())
    phase1_basis_elements = int(final_payload.get("phase1_basis_elements", 0))
    phase1_rounds = int(final_payload.get("phase1_rounds", max(int(getattr(args, "fedmssa_impute_rounds", 20)), 1)))
    phase2_rounds = int(final_payload.get("phase2_rounds", actual_rounds))
    phase1_comm_bytes = phase1_basis_elements * 4 * 2 * phase1_rounds
    phase2_comm_bytes = params_count * 4 * 2 * max(phase2_rounds, 1)
    eff_comm_mb = round((phase1_comm_bytes + phase2_comm_bytes) / (1024 * 1024), 4)

    eff_flops = 0.0
    try:
        from thop import profile

        dummy_batch = next(iter(val_loader))
        dummy_x = dummy_batch[0].to(device)
        flops, _ = profile(model, inputs=(dummy_x,), verbose=False)
        eff_flops = round(flops / 1e9, 4)
    except Exception:
        pass

    return (
        max(actual_rounds - 1, 0),
        test_metrics["mae"],
        test_metrics["rmse"],
        test_metrics["mape"],
        test_metrics["elements"],
        test_metrics["abs_error_sum"],
        test_metrics["sq_error_sum"],
        test_metrics["mape_error_sum"],
        test_metrics["mape_elements"],
        total_train_time,
        total_val_time / max(actual_rounds, 1),
        eff_test_time,
        eff_comm_mb,
        actual_rounds,
        eff_flops,
    )
