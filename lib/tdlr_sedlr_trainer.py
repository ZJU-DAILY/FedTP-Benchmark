import copy
import math
import time
from collections import deque

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from lib.utils import align_prediction_and_target, extract_ctx_data, unpack_spatiotemporal_batch
from privacy.protection import l2_norm
from privacy.runtime_protection import protected_arbiter_put
from privacy.attack_trace import (
    capture_he_sa_aggregate,
    capture_he_sa_collusion_residual,
    capture_he_sa_kminus2_hidden_term,
    capture_he_sa_kminus3_hidden_term,
    capture_he_sa_server_aggregate_hidden_term,
    capture_revised_quantized_prediction,
)


def _stream_to_device(tensor, args):
    use_cuda = isinstance(args.device, str) and args.device.startswith("cuda")
    return tensor.to(args.device, non_blocking=use_cuda)


def _clone_batch_to_cpu(batch):
    return tuple(item.detach().cpu().clone() if isinstance(item, torch.Tensor) else item for item in batch)


def _floating_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if value.is_floating_point()
    }


def _parse_tdlr_schedule(schedule_str):
    schedule = []
    if not schedule_str:
        return [(0, 1.0)]

    for chunk in str(schedule_str).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        round_text, scale_text = chunk.split(":")
        schedule.append((int(round_text.strip()), float(scale_text.strip())))

    if not schedule:
        schedule.append((0, 1.0))
    schedule.sort(key=lambda item: item[0])
    return schedule


def _scheduled_lr(base_lr, round_idx, schedule):
    scale = 1.0
    for start_round, start_scale in schedule:
        if round_idx >= start_round:
            scale = start_scale
        else:
            break
    return float(base_lr) * float(scale)


def _set_optimizer_lr(optimizer, lr_value):
    for group in optimizer.param_groups:
        group["lr"] = float(lr_value)


def _run_forward(model, x, y):
    pred = model(x)
    return align_prediction_and_target(pred, y)


def _batch_error_sums(model, batch, args):
    x, y = unpack_spatiotemporal_batch(batch)
    x = _stream_to_device(x, args)
    y = _stream_to_device(y, args)
    with torch.no_grad():
        pred, y = _run_forward(model, x, y)
        diff = pred - y
        abs_sum = float(torch.abs(diff).sum().detach().cpu())
        sq_sum = float((diff ** 2).sum().detach().cpu())
        elements = int(y.numel())
    return abs_sum, sq_sum, elements


def _train_batches(model, optimizer, loss_func, batches, args):
    model.train()
    total_loss = 0.0
    steps = 0
    for batch in batches:
        x, y = unpack_spatiotemporal_batch(batch)
        x = _stream_to_device(x, args)
        y = _stream_to_device(y, args)
        optimizer.zero_grad()
        pred, y = _run_forward(model, x, y)
        loss = loss_func(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        steps += 1
    return total_loss / max(steps, 1)


def _eval_loader_normalized(model, loader, args):
    model.eval()
    total_abs = 0.0
    total_sq = 0.0
    total_elements = 0
    with torch.no_grad():
        for batch in loader:
            x, y = unpack_spatiotemporal_batch(batch)
            x = _stream_to_device(x, args)
            y = _stream_to_device(y, args)
            pred, y = _run_forward(model, x, y)
            diff = pred - y
            total_abs += float(torch.abs(diff).sum().detach().cpu())
            total_sq += float((diff ** 2).sum().detach().cpu())
            total_elements += int(y.numel())

    mae = total_abs / max(total_elements, 1)
    mse = total_sq / max(total_elements, 1)
    rmse = math.sqrt(mse)
    return mae, mse, rmse, total_abs, total_sq, total_elements


def _eval_loader_real(model, loader, scaler, args):
    model.eval()
    total_abs = 0.0
    total_sq = 0.0
    total_mape = 0.0
    total_elements = 0
    total_mape_elements = 0
    with torch.no_grad():
        for batch in loader:
            x, y = unpack_spatiotemporal_batch(batch)
            x = _stream_to_device(x, args)
            y = _stream_to_device(y, args)
            pred, y = _run_forward(model, x, y)

            pred_real = scaler.inverse_transform(pred).detach().cpu().numpy()
            y_real = scaler.inverse_transform(y).detach().cpu().numpy()
            diff = pred_real - y_real

            total_abs += float(np.abs(diff).sum())
            total_sq += float((diff ** 2).sum())
            total_elements += int(y_real.size)

            mask = y_real > 0.5
            valid = int(mask.sum())
            if valid > 0:
                total_mape += float((np.abs(diff[mask]) / y_real[mask]).sum())
                total_mape_elements += valid

    mae = total_abs / max(total_elements, 1)
    mse = total_sq / max(total_elements, 1)
    rmse = math.sqrt(mse)
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


def _weighted_average_state_dict(payloads, sizes):
    total_size = max(float(sum(sizes)), 1.0)
    global_state = {}
    keys = payloads[0]["weights"].keys()
    for key in keys:
        accum = None
        for payload, size in zip(payloads, sizes):
            tensor = payload["weights"][key].float()
            weighted = tensor * (float(size) / total_size)
            accum = weighted if accum is None else accum + weighted
        global_state[key] = accum
    return global_state


def _make_candidate_lrs(low, high, current_lr, max_iter):
    low = max(float(low), 1e-6)
    high = max(float(high), low * 1.01)
    candidates = [float(current_lr), low, high, math.sqrt(low * high)]
    if max_iter > 0:
        log_low, log_high = math.log(low), math.log(high)
        for idx in range(max_iter):
            ratio = (idx + 1) / (max_iter + 1)
            candidates.append(math.exp(log_low + ratio * (log_high - log_low)))
    unique = []
    seen = set()
    for item in candidates:
        rounded = round(float(item), 12)
        if rounded in seen:
            continue
        unique.append(float(item))
        seen.add(rounded)
    return unique


def _surrogate_refine_candidates(trials, low, high):
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    except Exception:
        return []

    if len(trials) < 3:
        return []

    x = np.array([[math.log10(lr)] for lr, _ in trials], dtype=np.float64)
    y = np.array([score for _, score in trials], dtype=np.float64)
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)
    gp.fit(x, y)

    grid = np.linspace(math.log10(low), math.log10(high), 16).reshape(-1, 1)
    mean, std = gp.predict(grid, return_std=True)
    acquisition = mean - 0.25 * std
    order = np.argsort(acquisition)
    refined = []
    for idx in order[:3]:
        refined.append(float(10 ** grid[idx][0]))
    return refined


def _score_lr_candidate(model, state_dict, loss_func, recent_batches, candidate_lr, args):
    if len(recent_batches) < 2:
        return float("inf")

    shadow = copy.deepcopy(model)
    shadow.load_state_dict(state_dict)
    shadow = shadow.to(args.device)
    optimizer = torch.optim.Adam(shadow.parameters(), lr=float(candidate_lr))

    kfold = max(2, min(int(getattr(args, "tdlr_bayes_kfold", 3)), len(recent_batches)))
    fold_scores = []
    for fold_idx in range(kfold):
        train_batches = [batch for i, batch in enumerate(recent_batches) if i % kfold != fold_idx]
        val_batches = [batch for i, batch in enumerate(recent_batches) if i % kfold == fold_idx]
        if not train_batches or not val_batches:
            continue
        shadow.load_state_dict(state_dict)
        _set_optimizer_lr(optimizer, candidate_lr)
        _train_batches(shadow, optimizer, loss_func, train_batches, args)
        total_abs, total_elements = 0.0, 0
        shadow.eval()
        with torch.no_grad():
            for batch in val_batches:
                x, y = unpack_spatiotemporal_batch(batch)
                x = _stream_to_device(x, args)
                y = _stream_to_device(y, args)
                pred, y = _run_forward(shadow, x, y)
                total_abs += float(torch.abs(pred - y).sum().detach().cpu())
                total_elements += int(y.numel())
        fold_scores.append(total_abs / max(total_elements, 1))
    return float(np.mean(fold_scores)) if fold_scores else float("inf")


def _optimize_lr_with_surrogate(model, optimizer, loss_func, recent_batches, args, current_lr):
    low = float(getattr(args, "tdlr_bayes_lb", 1e-4))
    high = float(getattr(args, "tdlr_bayes_ub", 5e-3))
    max_iter = int(getattr(args, "tdlr_bayes_iter", 5))
    if len(recent_batches) < 2:
        return float(current_lr)

    base_state = copy.deepcopy(model.state_dict())
    trials = []
    for candidate_lr in _make_candidate_lrs(low, high, current_lr, max_iter):
        score = _score_lr_candidate(model, base_state, loss_func, recent_batches, candidate_lr, args)
        trials.append((candidate_lr, score))

    for candidate_lr in _surrogate_refine_candidates(trials, low, high):
        score = _score_lr_candidate(model, base_state, loss_func, recent_batches, candidate_lr, args)
        trials.append((candidate_lr, score))

    best_lr, _ = min(trials, key=lambda item: item[1])
    _set_optimizer_lr(optimizer, best_lr)
    return float(best_lr)


def _format_policy_report(payloads):
    reports = []
    for idx, payload in enumerate(payloads):
        trigger_flag = "T" if payload.get("sedlr_triggered", False) else "-"
        bayes_flag = "B" if payload.get("sedlr_bayes_executed", False) else "-"
        tdlr_flag = "TB" if payload.get("tdlr_bayes_executed", False) else "--"
        reports.append(
            f"c{idx}:lr={payload['lr']:.6f}|policy={payload.get('lr_policy','unknown')}|"
            f"trigger={trigger_flag}{bayes_flag}|tdlr={tdlr_flag}"
        )
    return ", ".join(reports)


def _recent_target_mean(batch):
    _, y = unpack_spatiotemporal_batch(batch)
    return float(y.float().mean().item())


def _sedlr_trigger(round_idx, current_batch, recent_target_stats, last_trigger_round, args):
    warmup = int(getattr(args, "sedlr_warmup_rounds", 10))
    cooldown = int(getattr(args, "sedlr_cooldown_rounds", 5))
    if round_idx < warmup:
        return False, "warmup"
    if last_trigger_round >= 0 and (round_idx - last_trigger_round) < cooldown:
        return False, "cooldown"

    current_mean = _recent_target_mean(current_batch)
    feature_type = str(getattr(args, "feature_type", "")).lower()
    if feature_type in ("occ", "occupancy"):
        threshold = float(getattr(args, "sedlr_occ_threshold", 0.8))
        if current_mean >= threshold:
            return True, f"occupancy:{current_mean:.4f}"
        return False, f"occupancy:{current_mean:.4f}"

    if len(recent_target_stats) < 3:
        return False, "history_short"

    history = np.array(recent_target_stats, dtype=np.float64)
    history_mean = float(history.mean())
    history_std = float(history.std()) + 1e-8
    z_score = abs(current_mean - history_mean) / history_std
    sigma = float(getattr(args, "sedlr_anomaly_sigma", 2.0))
    if z_score >= sigma:
        return True, f"anomaly_z:{z_score:.4f}"
    return False, f"anomaly_z:{z_score:.4f}"


def _should_run_sedlr_bayes(round_idx, last_bayes_round, args):
    bayes_warmup = int(getattr(args, "sedlr_bayes_warmup_rounds", getattr(args, "sedlr_warmup_rounds", 10)))
    reopt_every = max(int(getattr(args, "sedlr_bayes_reopt_every", 20)), 1)
    if round_idx < bayes_warmup:
        return False, f"bayes_warmup<{bayes_warmup}"
    if last_bayes_round >= 0 and (round_idx - last_bayes_round) < reopt_every:
        return False, f"bayes_reopt_gap<{reopt_every}"
    return True, "bayes_ready"


def _stream_round_batch(dataset, round_idx, args):
    batch_size = max(1, min(int(getattr(args, "batch_size", 1)), len(dataset)))
    # The paper defines an online mini-batch built from the most recent beta
    # samples after collecting tau new points each FL round. Advancing the
    # batch window by t_in approximates that streaming update cadence.
    round_stride = max(int(getattr(args, "t_in", 1)), 1)
    end = min(batch_size + round_idx * round_stride, len(dataset))
    start = max(0, end - batch_size)
    if end - start < batch_size:
        start = max(0, len(dataset) - batch_size)
        end = len(dataset)
    return default_collate([dataset[idx] for idx in range(start, end)])


def train_tdlr_sedlr_task(ctx, args, setting=None):
    model_name = str(getattr(args, "model", "TDLR_SEDLR"))
    is_he = str(getattr(args, "protection", "plain")).lower() == "he"
    use_tdlr = model_name in ("TDLR", "TDLR_SEDLR")
    use_sedlr = model_name in ("SEDLR", "TDLR_SEDLR")
    total_rounds = max(int(getattr(args, "tdlr_stream_rounds", 0) or getattr(args, "epochs", 1)), 1)

    if ctx.is_on_arbiter:
        size_g = ctx.guest.get("tdlr_size")
        size_h = ctx.hosts.get("tdlr_size")
        size_h = size_h if isinstance(size_h, list) else [size_h]
        all_sizes = [size_g] + size_h
        he_arbiter_context = None
        if is_he:
            from privacy.ckks_backend import (
                decrypt_tree as ckks_decrypt_tree,
                export_public_context as ckks_export_public_context,
                generate_context as ckks_generate_context,
                homomorphic_weighted_sum_tree as ckks_homomorphic_weighted_sum_tree,
            )
            if str(getattr(args, "he_backend", "auto")).lower() not in ("auto", "he_sa"):
                raise ValueError("TDLR_SEDLR supports HE-SA only for its weighted model aggregation.")
            he_arbiter_context = ckks_generate_context(
                int(args.he_ckks_poly_modulus_degree), int(args.he_ckks_scale_bits),
            )
            public_context = ckks_export_public_context(he_arbiter_context)
            # Client-side trace capture needs the same public data-size
            # coefficients that the arbiter applies to encrypted updates.
            ctx.guest.put("__tdlr_sedlr_he_trace_sizes", all_sizes)
            ctx.hosts.put("__tdlr_sedlr_he_trace_sizes", [all_sizes] * len(size_h))
            ctx.guest.put("__tdlr_sedlr_he_sa_ckks_context", public_context)
            ctx.hosts.put("__tdlr_sedlr_he_sa_ckks_context", public_context)
            print(f"[HESA] {model_name} arbiter generated packed CKKS context", flush=True)
        dp_clip_norm = None
        if args.protection == "dp":
            # Initial weights are public common protocol state.  Subsequent
            # messages are protected deltas relative to this global state.
            dp_global_weights = ctx.guest.get("tdlr_dp_initial_state")
            _ = ctx.hosts.get("tdlr_dp_initial_state")
        best_val_mae = float("inf")
        best_round = -1
        best_global_weights = None
        patience = int(getattr(args, "tdlr_patience", 50))
        min_delta = float(getattr(args, "tdlr_min_delta", 0.0))
        patience_counter = 0
        actual_rounds = 0

        print(
            f"[{model_name} Server] start streaming federated training | clients={len(all_sizes)} rounds={total_rounds}",
            flush=True,
        )

        for round_idx in range(total_rounds):
            if args.protection == "dp" and dp_clip_norm is None:
                norm_guest = float(ctx.guest.get(f"tdlr_dp_delta_norm_{round_idx}"))
                norm_hosts = ctx.hosts.get(f"tdlr_dp_delta_norm_{round_idx}")
                norm_hosts = norm_hosts if isinstance(norm_hosts, list) else [norm_hosts]
                dp_clip_norm = float(np.quantile([norm_guest] + [float(value) for value in norm_hosts], 0.9))
                ctx.guest.put(f"tdlr_dp_clip_norm_{round_idx}", dp_clip_norm)
                ctx.hosts.put(f"tdlr_dp_clip_norm_{round_idx}", [dp_clip_norm] * len(norm_hosts))
                print(f"[DPCalibration] {model_name} arbiter epoch={round_idx + 1} clip_norm={dp_clip_norm:.8f}", flush=True)

            payload_tag = (
                f"tdlr_dp_metadata_{round_idx}" if args.protection == "dp"
                else f"tdlr_he_metadata_{round_idx}" if is_he
                else f"tdlr_payload_{round_idx}"
            )
            p_guest = ctx.guest.get(payload_tag)
            p_hosts = ctx.hosts.get(payload_tag)
            p_hosts = p_hosts if isinstance(p_hosts, list) else [p_hosts]
            all_payloads = [p_guest] + p_hosts

            if args.protection == "dp":
                d_guest = ctx.guest.get(f"tdlr_dp_delta_{round_idx}")
                d_hosts = ctx.hosts.get(f"tdlr_dp_delta_{round_idx}")
                d_hosts = d_hosts if isinstance(d_hosts, list) else [d_hosts]
                all_deltas = [d_guest] + d_hosts
                for payload, delta in zip(all_payloads, all_deltas):
                    payload["weights"] = {
                        key: dp_global_weights[key] + value
                        for key, value in delta.items()
                    }

            if is_he:
                weight_guest = ctx.guest.get(f"tdlr_he_weights_{round_idx}")
                weight_hosts = ctx.hosts.get(f"tdlr_he_weights_{round_idx}")
                weight_hosts = weight_hosts if isinstance(weight_hosts, list) else [weight_hosts]
                encrypted_weights = [weight_guest] + weight_hosts
                total_size = max(float(sum(all_sizes)), 1.0)
                coefficients = [float(size) / total_size for size in all_sizes]
                started = time.perf_counter()
                encrypted_global = ckks_homomorphic_weighted_sum_tree(
                    encrypted_weights, coefficients, he_arbiter_context,
                )
                global_weights = ckks_decrypt_tree(encrypted_global, he_arbiter_context)
                capture_he_sa_aggregate(ctx, args, f"tdlr_sedlr_aggregate_{round_idx}", global_weights)
                he_seconds = time.perf_counter() - started
                bytes_guest = int(ctx.guest.get(f"tdlr_he_upload_bytes_{round_idx}"))
                bytes_hosts = ctx.hosts.get(f"tdlr_he_upload_bytes_{round_idx}")
                bytes_hosts = bytes_hosts if isinstance(bytes_hosts, list) else [bytes_hosts]
                print(
                    f"[HESA] {model_name} arbiter round={round_idx + 1} "
                    f"aggregate_decrypt_s={he_seconds:.6f} "
                    f"encrypted_upload_bytes={bytes_guest + sum(int(value) for value in bytes_hosts)}",
                    flush=True,
                )
            else:
                global_weights = _weighted_average_state_dict(all_payloads, all_sizes)
            if args.protection == "dp":
                dp_global_weights = global_weights
            online_abs = sum(float(payload["online_abs_sum"]) for payload in all_payloads)
            online_sq = sum(float(payload["online_sq_sum"]) for payload in all_payloads)
            online_elements = sum(int(payload["online_elements"]) for payload in all_payloads)
            online_mae = online_abs / max(online_elements, 1)
            online_rmse = math.sqrt(online_sq / max(online_elements, 1))
            trigger_clients = [f"c{i}" for i, payload in enumerate(all_payloads) if payload.get("sedlr_triggered", False)]
            sedlr_bayes_clients = [
                f"c{i}" for i, payload in enumerate(all_payloads) if payload.get("sedlr_bayes_executed", False)
            ]
            tdlr_bayes_clients = [
                f"c{i}" for i, payload in enumerate(all_payloads) if payload.get("tdlr_bayes_executed", False)
            ]
            policy_report = _format_policy_report(all_payloads)

            ctx.guest.put(f"tdlr_global_{round_idx}", global_weights)
            ctx.hosts.put(f"tdlr_global_{round_idx}", [global_weights] * len(p_hosts))
            if is_he:
                ctx.guest.put(f"tdlr_he_arbiter_seconds_{round_idx}", he_seconds)
                ctx.hosts.put(f"tdlr_he_arbiter_seconds_{round_idx}", [he_seconds] * len(p_hosts))

            v_guest = ctx.guest.get(f"tdlr_val_{round_idx}")
            v_hosts = ctx.hosts.get(f"tdlr_val_{round_idx}")
            v_hosts = v_hosts if isinstance(v_hosts, list) else [v_hosts]
            all_val = [v_guest] + v_hosts
            val_abs = sum(float(item["val_abs_sum"]) for item in all_val)
            val_sq = sum(float(item["val_sq_sum"]) for item in all_val)
            val_elements = sum(int(item["val_elements"]) for item in all_val)
            val_mae = val_abs / max(val_elements, 1)
            val_rmse = math.sqrt(val_sq / max(val_elements, 1))

            improved = val_mae < (best_val_mae - min_delta)
            if improved:
                best_val_mae = val_mae
                best_round = round_idx
                best_global_weights = copy.deepcopy(global_weights)
                patience_counter = 0
            else:
                patience_counter += 1

            actual_rounds = round_idx + 1
            should_stop = patience_counter >= patience
            control = {"stop": should_stop, "best_round": best_round}
            ctx.guest.put(f"tdlr_control_{round_idx}", control)
            ctx.hosts.put(f"tdlr_control_{round_idx}", [control] * len(p_hosts))

            print(
                f"[{model_name} Server] round={round_idx} online_mae={online_mae:.4f} online_rmse={online_rmse:.4f} "
                f"val_mae={val_mae:.4f} val_rmse={val_rmse:.4f} best_round={best_round} "
                f"patience={patience_counter}/{patience} | sedlr_trigger={trigger_clients or ['none']} "
                f"| sedlr_bayes={sedlr_bayes_clients or ['none']} | tdlr_bayes={tdlr_bayes_clients or ['none']} "
                f"| policy=({policy_report})",
                flush=True,
            )

            if should_stop:
                break

        final_weights = best_global_weights if best_global_weights is not None else global_weights
        final_payload = {"weights": final_weights, "best_round": best_round, "actual_rounds": actual_rounds}
        ctx.guest.put("tdlr_final", final_payload)
        ctx.hosts.put("tdlr_final", [final_payload] * len(size_h))
        return None

    if setting is None:
        raise RuntimeError(f"{model_name} client requires a get_setting(ctx) result.")

    train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = setting
    pin_memory = isinstance(args.device, str) and args.device.startswith("cuda")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)
    schedule = _parse_tdlr_schedule(getattr(args, "tdlr_schedule", "0:1.0"))

    ctx.arbiter.put("tdlr_size", len(train_set))
    he_context = None
    he_upload_bytes = 0
    he_download_bytes = 0
    if is_he:
        from privacy.ckks_backend import (
            ciphertext_bytes as ckks_ciphertext_bytes,
            encrypt_tree as ckks_encrypt_tree,
            import_context as ckks_import_context,
        )
        raw_trace_sizes = ctx.arbiter.get("__tdlr_sedlr_he_trace_sizes")
        # A host broadcast is returned by FATE as several identical nested
        # copies of the public size vector.  Verify and unwrap those copies;
        # do not flatten arbitrary per-client values.
        while isinstance(raw_trace_sizes, (list, tuple)) and raw_trace_sizes and all(
            isinstance(value, (list, tuple)) for value in raw_trace_sizes
        ):
            candidate = list(raw_trace_sizes[0])
            if any(list(value) != candidate for value in raw_trace_sizes[1:]):
                raise RuntimeError(
                    f"{model_name} HE trace broadcasts disagree across recipients: {raw_trace_sizes!r}"
                )
            raw_trace_sizes = candidate
        if not isinstance(raw_trace_sizes, (list, tuple)):
            raise RuntimeError(
                f"{model_name} HE trace weights were not broadcast as a sequence: {type(raw_trace_sizes)!r}"
            )
        trace_sizes = [float(value) for value in raw_trace_sizes]
        if int(ctx.rank) < 0 or int(ctx.rank) >= len(trace_sizes):
            raise RuntimeError(
                f"{model_name} HE trace rank {ctx.rank} is outside broadcast sizes {trace_sizes!r}"
            )
        trace_aggregation_weight = trace_sizes[int(ctx.rank)] / max(sum(trace_sizes), 1.0)
        he_context = ckks_import_context(ctx.arbiter.get("__tdlr_sedlr_he_sa_ckks_context"))
        print(f"[HESA] {model_name} rank={ctx.rank} received packed CKKS public context", flush=True)
    if args.protection == "dp":
        ctx.arbiter.put("tdlr_dp_initial_state", _floating_state_dict(model))

    recent_batches = deque(maxlen=max(int(getattr(args, "tdlr_recent_buffer_size", 8)), 2))
    recent_target_stats = deque(maxlen=max(int(getattr(args, "tdlr_recent_buffer_size", 8)), 2))
    last_trigger_round = -1
    last_sedlr_bayes_round = -1
    total_train_time = 0.0
    total_val_time = 0.0
    actual_rounds = 0
    print(
        f"[{model_name} Client {ctx.rank}] start | train_batches={len(train_loader)} val_batches={len(val_loader)} "
        f"test_batches={len(test_loader)} rounds={total_rounds}",
        flush=True,
    )

    for round_idx in range(total_rounds):
        round_start_weights = _floating_state_dict(model)
        current_batch = _stream_round_batch(train_set, round_idx, args)
        recent_batches.append(_clone_batch_to_cpu(current_batch))
        recent_target_stats.append(_recent_target_mean(current_batch))

        online_abs, online_sq, online_elements = _batch_error_sums(model, current_batch, args)

        base_lr = float(getattr(args, "lr", 1e-3))
        round_lr = base_lr
        if use_tdlr:
            round_lr = _scheduled_lr(round_lr, round_idx, schedule)
        scheduled_lr = round_lr

        sedlr_triggered = False
        sedlr_bayes_executed = False
        tdlr_bayes_executed = False
        trigger_reason = "disabled"
        sedlr_policy = "sedlr_off"
        sedlr_bayes_status = "sedlr_bayes_off"
        lr_policy = "base"
        if use_sedlr:
            sedlr_triggered, trigger_reason = _sedlr_trigger(
                round_idx, current_batch, recent_target_stats, last_trigger_round, args
            )
            if sedlr_triggered:
                last_trigger_round = round_idx
                if getattr(args, "sedlr_bayes_enable", False):
                    bayes_ready, sedlr_bayes_status = _should_run_sedlr_bayes(
                        round_idx, last_sedlr_bayes_round, args
                    )
                    if bayes_ready:
                        round_lr = _optimize_lr_with_surrogate(
                            model, optimizer, loss_func, list(recent_batches), args, round_lr
                        )
                        sedlr_bayes_executed = True
                        last_sedlr_bayes_round = round_idx
                        sedlr_policy = "sedlr_trigger_bayes"
                        lr_policy = "sedlr_trigger_bayes"
                    else:
                        round_lr *= float(getattr(args, "sedlr_aggressive_mult", 2.0))
                        sedlr_policy = "sedlr_trigger_multiplier_fallback"
                        lr_policy = "sedlr_trigger_multiplier_fallback"
                else:
                    round_lr *= float(getattr(args, "sedlr_aggressive_mult", 2.0))
                    sedlr_policy = "sedlr_trigger_multiplier"
                    lr_policy = "sedlr_trigger_multiplier"
            else:
                round_lr *= float(getattr(args, "sedlr_calm_mult", 1.0))
                sedlr_policy = "sedlr_calm_multiplier"
                lr_policy = "sedlr_calm_multiplier"

        if use_tdlr and getattr(args, "tdlr_bayes_enable", False):
            every = max(int(getattr(args, "tdlr_bayes_every", 50)), 1)
            if round_idx > 0 and round_idx % every == 0:
                round_lr = _optimize_lr_with_surrogate(
                    model, optimizer, loss_func, list(recent_batches), args, round_lr
                )
                tdlr_bayes_executed = True
                if sedlr_bayes_executed:
                    lr_policy = "sedlr_bayes_plus_tdlr_bayes"
                elif use_sedlr and sedlr_policy != "sedlr_off":
                    lr_policy = f"{sedlr_policy}_plus_tdlr_bayes"
                else:
                    lr_policy = "tdlr_periodic_bayes"
        elif use_tdlr:
            if lr_policy == "base":
                lr_policy = "tdlr_schedule"
        elif use_sedlr and lr_policy == "base":
            lr_policy = sedlr_policy

        _set_optimizer_lr(optimizer, round_lr)

        train_start = time.time()
        trace_x, _ = unpack_spatiotemporal_batch(current_batch)
        trace_x = _stream_to_device(trace_x, args)
        with torch.no_grad():
            trace_prediction = model(trace_x)
        capture_revised_quantized_prediction(
            ctx, args, f"tdlr_sedlr_prediction_{round_idx}", prediction=trace_prediction,
            model_state_dict=_floating_state_dict(model),
        )
        round_loss = 0.0
        for _ in range(max(int(getattr(args, "local_epochs", 1)), 1)):
            round_loss = _train_batches(model, optimizer, loss_func, [current_batch], args)
        total_train_time += time.time() - train_start

        payload = {
            "weights": _floating_state_dict(model),
            "online_abs_sum": online_abs,
            "online_sq_sum": online_sq,
            "online_elements": online_elements,
            "lr": float(round_lr),
            "scheduled_lr": float(scheduled_lr),
            "base_lr": float(base_lr),
            "sedlr_triggered": bool(sedlr_triggered),
            "sedlr_trigger_reason": str(trigger_reason),
            "sedlr_policy": str(sedlr_policy),
            "sedlr_bayes_status": str(sedlr_bayes_status),
            "sedlr_bayes_executed": bool(sedlr_bayes_executed),
            "tdlr_bayes_executed": bool(tdlr_bayes_executed),
            "lr_policy": str(lr_policy),
            "triggered": bool(sedlr_triggered),
        }
        if args.protection == "dp":
            local_delta = {
                key: payload["weights"][key] - round_start_weights[key]
                for key in round_start_weights
            }
            if float(args.dp_clip_norm) <= 0:
                ctx.arbiter.put(f"tdlr_dp_delta_norm_{round_idx}", float(l2_norm(local_delta).item()))
                calibrated_clip = ctx.arbiter.get(f"tdlr_dp_clip_norm_{round_idx}")
                if isinstance(calibrated_clip, (list, tuple)):
                    calibrated_clip = calibrated_clip[0]
                args.dp_clip_norm = float(calibrated_clip)
                print(
                    f"[DPCalibration] {model_name} rank={ctx.rank} epoch={round_idx + 1} "
                    f"clip_norm={args.dp_clip_norm:.8f}",
                    flush=True,
                )
            protection_started = time.perf_counter()
            protected_arbiter_put(ctx, args, f"tdlr_dp_delta_{round_idx}", local_delta)
            total_train_time += time.perf_counter() - protection_started
            payload.pop("weights")
            ctx.arbiter.put(f"tdlr_dp_metadata_{round_idx}", payload)
        elif is_he:
            aggregate_delta = {
                key: payload["weights"][key] - round_start_weights[key]
                for key in round_start_weights
            }
            capture_he_sa_server_aggregate_hidden_term(
                ctx, args, f"tdlr_sedlr_delta_{round_idx}", payload=aggregate_delta,
                model_state_dict=round_start_weights,
                aggregation_weight=trace_aggregation_weight,
                leak_type="model_update",
            )
            capture_he_sa_collusion_residual(
                ctx, args, f"tdlr_sedlr_delta_{round_idx}", payload=aggregate_delta,
                model_state_dict=round_start_weights,
            )
            capture_he_sa_kminus2_hidden_term(
                ctx, args, f"tdlr_sedlr_delta_{round_idx}", payload=aggregate_delta,
                model_state_dict=round_start_weights,
                aggregation_weight=trace_aggregation_weight,
            )
            capture_he_sa_kminus3_hidden_term(
                ctx, args, f"tdlr_sedlr_delta_{round_idx}", payload=aggregate_delta,
                model_state_dict=round_start_weights,
                aggregation_weight=trace_aggregation_weight,
            )
            protection_started = time.perf_counter()
            encrypted_weights = ckks_encrypt_tree(
                payload.pop("weights"), he_context,
                slot_count=int(args.he_ckks_poly_modulus_degree) // 2,
            )
            upload_bytes = ckks_ciphertext_bytes(encrypted_weights)
            he_upload_bytes += upload_bytes
            total_train_time += time.perf_counter() - protection_started
            ctx.arbiter.put(f"tdlr_he_weights_{round_idx}", encrypted_weights)
            ctx.arbiter.put(f"tdlr_he_upload_bytes_{round_idx}", upload_bytes)
            ctx.arbiter.put(f"tdlr_he_metadata_{round_idx}", payload)
            print(
                f"[HESA] {model_name} rank={ctx.rank} round={round_idx + 1} "
                f"encrypted_upload_bytes={upload_bytes}", flush=True,
            )
        else:
            ctx.arbiter.put(f"tdlr_payload_{round_idx}", payload)

        global_weight_data = ctx.arbiter.get(f"tdlr_global_{round_idx}")
        global_weights = extract_ctx_data(ctx, global_weight_data)
        if is_he:
            he_download_bytes += sum(value.numel() * value.element_size() for value in global_weights.values())
            he_seconds = extract_ctx_data(ctx, ctx.arbiter.get(f"tdlr_he_arbiter_seconds_{round_idx}"))
            total_train_time += float(he_seconds)
        model.load_state_dict(global_weights, strict=(args.protection not in ("dp", "he")))

        val_start = time.time()
        _, _, _, val_abs_sum, val_sq_sum, val_elements = _eval_loader_normalized(model, val_loader, args)
        total_val_time += time.time() - val_start
        ctx.arbiter.put(
            f"tdlr_val_{round_idx}",
            {
                "val_abs_sum": val_abs_sum,
                "val_sq_sum": val_sq_sum,
                "val_elements": val_elements,
            },
        )

        control = extract_ctx_data(ctx, ctx.arbiter.get(f"tdlr_control_{round_idx}"))
        actual_rounds = round_idx + 1
        print(
            f"[{model_name} Client {ctx.rank}] round={round_idx} loss={round_loss:.4f} "
            f"base_lr={base_lr:.6f} scheduled_lr={scheduled_lr:.6f} final_lr={round_lr:.6f} "
            f"lr_policy={lr_policy} sedlr_triggered={int(sedlr_triggered)} "
            f"sedlr_reason={trigger_reason} sedlr_bayes={sedlr_bayes_status} "
            f"tdlr_bayes={int(tdlr_bayes_executed)}",
            flush=True,
        )
        if control["stop"]:
            break

    final_payload = extract_ctx_data(ctx, ctx.arbiter.get("tdlr_final"))
    model.load_state_dict(final_payload["weights"], strict=(args.protection not in ("dp", "he")))
    best_round = int(final_payload.get("best_round", -1))
    actual_rounds = int(final_payload.get("actual_rounds", actual_rounds))

    test_start = time.time()
    test_metrics = _eval_loader_real(model, test_loader, scaler, args)
    eff_test_time = time.time() - test_start

    params_count = sum(param.numel() for param in model.parameters())
    eff_comm_mb = (
        round((he_upload_bytes + he_download_bytes) / (1024 * 1024), 4)
        if is_he else round((params_count * 4 * (actual_rounds * 2)) / (1024 * 1024), 4)
    )

    eff_flops = 0.0
    try:
        from thop import profile

        dummy_batch = next(iter(val_loader))
        dummy_x, _ = unpack_spatiotemporal_batch(dummy_batch)
        flops, _ = profile(model, inputs=(_stream_to_device(dummy_x, args),), verbose=False)
        eff_flops = round(flops / 1e9, 4)
    except Exception as exc:
        print(f"[{model_name} Client {ctx.rank}] FLOPs estimation skipped: {exc}", flush=True)

    return (
        best_round,
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
