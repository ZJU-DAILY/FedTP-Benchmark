import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from lib.utils import synchronize_cuda_for_timing
from privacy.protection import l2_norm
from privacy.runtime_protection import (
    protected_arbiter_put, record_he_ttp_downlink, unprotect_he_ttp_payload,
)
from privacy.attack_trace import capture_he_ttp_insider_upper_bound


def _unwrap_ctx_payload(value):
    if isinstance(value, list):
        if not value:
            return value
        return _unwrap_ctx_payload(value[0])
    return value


def _param_state(model):
    return {
        name: param.detach().cpu().contiguous().clone()
        for name, param in model.named_parameters()
    }


def _load_param_state(model, state, device):
    state_on_device = {
        key: (value.to(device) if torch.is_tensor(value) else value)
        for key, value in state.items()
    }
    model.load_state_dict(state_on_device, strict=False)


def _pred_tensor(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def _align_pred_y(pred, y):
    if y.dim() == 4 and y.shape[-1] == 1:
        y = y.squeeze(-1)
    if pred.dim() == 4 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)
    if pred.shape != y.shape:
        if pred.dim() == 3 and y.dim() == 3 and pred.shape[1] == y.shape[2] and pred.shape[2] == y.shape[1]:
            pred = pred.transpose(1, 2)
        else:
            pred = pred.reshape_as(y)
    return pred, y


def _loss_output(output, pred):
    if isinstance(output, tuple) and len(output) == 2:
        return pred, output[1]
    return pred


def _eval_loader(model, loader, scaler, device):
    model.eval()
    abs_sum = 0.0
    sq_sum = 0.0
    mape_sum = 0.0
    elements = 0
    mape_elements = 0
    batches = 0
    with torch.no_grad():
        for x, y in loader:
            batches += 1
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = _pred_tensor(model(x))
            pred, y = _align_pred_y(pred, y)

            if hasattr(scaler, "inverse_transform"):
                y_real = scaler.inverse_transform(y).detach().cpu().numpy()
                pred_real = scaler.inverse_transform(pred).detach().cpu().numpy()
            else:
                y_real = (y * scaler.std + scaler.mean).detach().cpu().numpy()
                pred_real = (pred * scaler.std + scaler.mean).detach().cpu().numpy()

            diff = pred_real - y_real
            abs_sum += float(np.abs(diff).sum())
            sq_sum += float(np.square(diff).sum())
            elements += int(y_real.size)
            mask = y_real > 0.5
            if np.any(mask):
                mape_sum += float((np.abs(diff[mask]) / y_real[mask]).sum() * 100.0)
                mape_elements += int(mask.sum())

    mse = sq_sum / max(elements, 1)
    return {
        "mae": abs_sum / max(elements, 1),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mape": mape_sum / max(mape_elements, 1) if mape_elements else 0.0,
        "elements": elements,
        "abs_error_sum": abs_sum,
        "sq_error_sum": sq_sum,
        "mape_error_sum": mape_sum,
        "mape_elements": mape_elements,
        "batches": batches,
    }


def _masked_aggregate(payloads, previous_state=None):
    first_state = payloads[0]["weights"]
    total_client_nodes = sum(max(1, int(payload["num_nodes"])) for payload in payloads)
    global_state = {}

    for key, first_tensor in first_state.items():
        if not torch.is_tensor(first_tensor) or not torch.is_floating_point(first_tensor):
            global_state[key] = first_tensor
            continue

        if key == "spatial_emb.weight":
            fallback = (
                previous_state[key].float().clone()
                if previous_state is not None and key in previous_state
                else first_tensor.float().clone()
            )
            agg = torch.zeros_like(fallback)
            counts = torch.zeros((fallback.shape[0], 1), dtype=agg.dtype)
            for payload in payloads:
                nodes = torch.as_tensor(payload["nodes"], dtype=torch.long)
                values = payload["weights"][key].float()
                agg.index_add_(0, nodes, values.index_select(0, nodes))
                counts.index_add_(0, nodes, torch.ones((len(nodes), 1), dtype=agg.dtype))
            mask = counts.squeeze(-1) > 0
            fallback[mask] = agg[mask] / counts[mask]
            global_state[key] = fallback.to(first_tensor.dtype).contiguous()

        elif key == "generator.param_matrix":
            fallback = (
                previous_state[key].float().clone()
                if previous_state is not None and key in previous_state
                else first_tensor.float().clone()
            )
            agg = torch.zeros_like(fallback)
            counts = torch.zeros(fallback.shape[:2], dtype=agg.dtype)
            for payload in payloads:
                nodes = torch.as_tensor(payload["nodes"], dtype=torch.long)
                values = payload["weights"][key].float()
                sub = values.index_select(0, nodes).index_select(1, nodes)
                row_idx = nodes.view(-1, 1).expand(-1, len(nodes)).reshape(-1)
                col_idx = nodes.view(1, -1).expand(len(nodes), -1).reshape(-1)
                agg[row_idx, col_idx] += sub.reshape(-1, sub.shape[-1])
                counts[row_idx, col_idx] += 1.0
            mask = counts > 0
            fallback[mask] = agg[mask] / counts[mask].unsqueeze(-1)
            global_state[key] = fallback.to(first_tensor.dtype).contiguous()

        else:
            agg = torch.zeros_like(first_tensor.float())
            for payload in payloads:
                coef = max(1, int(payload["num_nodes"])) / total_client_nodes
                agg += payload["weights"][key].float() * coef
            global_state[key] = agg.to(first_tensor.dtype).contiguous()

    return global_state


def train_fedagat_masked_global_id(ctx, args, get_setting, log_experiment_results):
    print(f"Rank {ctx.rank}: [FedAGAT Masked] start global-ID aligned masked FedAvg.", flush=True)

    if not ctx.is_on_arbiter:
        train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = get_setting(ctx)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=False, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=False, num_workers=0)

        selected_nodes = list(args.nodes_per[ctx.rank])
        best_state = None
        best_epoch = -1
        total_train_time = 0.0
        total_val_time = 0.0
        last_epoch = 0

        ctx.arbiter.put("fedagat_client_ready", {"rank": ctx.rank, "num_nodes": len(selected_nodes)})
        dp_clip_norm = float(args.dp_clip_norm) if args.protection == "dp" and float(getattr(args, "dp_clip_norm", 0.0)) > 0 else None
        if args.protection == "dp":
            ctx.arbiter.put("fedagat_public_init_state", _param_state(model))
            initial_state = _unwrap_ctx_payload(ctx.arbiter.get("fedagat_public_init_state"))
            _load_param_state(model, initial_state, args.device)

    else:
        guest_ready = _unwrap_ctx_payload(ctx.guest.get("fedagat_client_ready"))
        hosts_ready = ctx.hosts.get("fedagat_client_ready")
        if not isinstance(hosts_ready, list):
            hosts_ready = [hosts_ready]
        hosts_ready = [_unwrap_ctx_payload(item) for item in hosts_ready]
        print(f"[FedAGAT Masked Server] clients ready: {[guest_ready] + hosts_ready}", flush=True)
        best_global_mae = float("inf")
        best_epoch = -1
        wait_count = 0
        previous_global_state = None
        dp_clip_norm = float(args.dp_clip_norm) if args.protection == "dp" and float(getattr(args, "dp_clip_norm", 0.0)) > 0 else None
        if args.protection == "dp":
            init_guest = _unwrap_ctx_payload(ctx.guest.get("fedagat_public_init_state"))
            init_hosts = ctx.hosts.get("fedagat_public_init_state")
            init_hosts = init_hosts if isinstance(init_hosts, list) else [init_hosts]
            previous_global_state = {key: value.detach().cpu().clone() for key, value in init_guest.items()}
            ctx.guest.put("fedagat_public_init_state", previous_global_state)
            ctx.hosts.put("fedagat_public_init_state", [previous_global_state] * len(init_hosts))
            print(f"[FedAGAT-DP] canonical initial state keys={len(previous_global_state)}", flush=True)
        stop_patience = int(getattr(args, "early_stop_patience", 50) or 50)
        min_delta = 1e-5

    for epoch in range(args.epochs):
        if not ctx.is_on_arbiter:
            last_epoch = epoch + 1
            round_start_state = _param_state(model)
            model.train()
            train_start = time.time()
            epoch_loss = 0.0
            steps = 0

            for _ in range(max(1, int(args.local_epochs))):
                for x, y in train_loader:
                    x = x.to(args.device, non_blocking=True)
                    y = y.to(args.device, non_blocking=True)
                    optimizer.zero_grad()
                    output = model(x)
                    pred = _pred_tensor(output)
                    pred, y = _align_pred_y(pred, y)
                    loss = loss_func(_loss_output(output, pred), y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += float(loss.item())
                    steps += 1

            total_train_time += time.time() - train_start
            print(
                f"Rank {ctx.rank}: [FedAGAT Masked] epoch={epoch + 1} "
                f"train_loss={epoch_loss / max(steps, 1):.6f}",
                flush=True,
            )

            current_state = _param_state(model)
            if args.protection == "he":
                capture_he_ttp_insider_upper_bound(
                    ctx, args, f"fedagat_delta_{epoch}",
                    observed_leak={key: current_state[key] - round_start_state[key] for key in current_state},
                    model_state_dict=round_start_state, leak_type="model_update",
                )
            upload_payload = {
                "rank": ctx.rank,
                "nodes": selected_nodes,
                "num_nodes": len(selected_nodes),
                "weights": ({key: current_state[key] - round_start_state[key] for key in current_state} if args.protection == "dp" else current_state),
            }
            # Time the complete client protocol round, not only local SGD.
            protocol_started = time.perf_counter()
            if args.protection == "dp":
                ctx.arbiter.put(f"fedagat_dp_norm_{epoch}", float(l2_norm(upload_payload["weights"]).item()) if dp_clip_norm is None else None)
                if dp_clip_norm is None:
                    dp_clip_norm = _unwrap_ctx_payload(ctx.arbiter.get(f"fedagat_dp_clip_{epoch}"))
                    dp_clip_norm = float(dp_clip_norm)
                protected_arbiter_put(ctx, args, f"fedagat_weights_{epoch}", upload_payload, clip_norm=dp_clip_norm)
            elif args.protection == "he":
                protected_arbiter_put(ctx, args, f"fedagat_weights_{epoch}", upload_payload)
            else:
                ctx.arbiter.put(f"fedagat_weights_{epoch}", upload_payload)

            # Host routing can return a list of all host payloads.  First
            # select/unpack the one addressed to this client; accounting the
            # routing container inflated HE-TTP downlink communication.
            global_state = _unwrap_ctx_payload(
                ctx.arbiter.get(f"fedagat_global_weights_{epoch}")
            )
            global_state = record_he_ttp_downlink(
                args, global_state, tag=f"fedagat_global_weights_{epoch}",
            )
            _load_param_state(model, global_state, args.device)
            total_train_time += time.perf_counter() - protocol_started

            val_start = time.time()
            val_stats = _eval_loader(model, val_loader, scaler, args.device)
            total_val_time += time.time() - val_start
            ctx.arbiter.put(
                f"fedagat_val_{epoch}",
                {"rank": ctx.rank, "mae": val_stats["mae"], "elements": val_stats["elements"]},
            )
            control = _unwrap_ctx_payload(ctx.arbiter.get(f"fedagat_control_{epoch}"))
            print(
                f"Rank {ctx.rank}: [FedAGAT Masked] epoch={epoch + 1} "
                f"val_mae={val_stats['mae']:.6f} global_val_mae={control['global_mae']:.6f} "
                f"best_epoch={control['best_epoch'] + 1}",
                flush=True,
            )
            if control.get("is_best", False):
                best_state = _param_state(model)
                best_epoch = epoch
            if control.get("stop", False):
                break

        else:
            if args.protection == "dp" and dp_clip_norm is None:
                norm_guest = _unwrap_ctx_payload(ctx.guest.get(f"fedagat_dp_norm_{epoch}"))
                norm_hosts = ctx.hosts.get(f"fedagat_dp_norm_{epoch}")
                norm_hosts = norm_hosts if isinstance(norm_hosts, list) else [norm_hosts]
                dp_clip_norm = float(np.quantile([float(v) for v in [norm_guest] + norm_hosts], 0.9))
                ctx.guest.put(f"fedagat_dp_clip_{epoch}", dp_clip_norm)
                ctx.hosts.put(f"fedagat_dp_clip_{epoch}", [dp_clip_norm] * len(norm_hosts))
                print(f"[DPCalibration] FedAGAT target=model_delta clip_norm={dp_clip_norm:.8f}", flush=True)
            guest_payload = _unwrap_ctx_payload(unprotect_he_ttp_payload(
                args, ctx.guest.get(f"fedagat_weights_{epoch}"),
            ))
            hosts_payload = ctx.hosts.get(f"fedagat_weights_{epoch}")
            if not isinstance(hosts_payload, list):
                hosts_payload = [hosts_payload]
            hosts_payload = [
                _unwrap_ctx_payload(unprotect_he_ttp_payload(args, item))
                for item in hosts_payload
            ]
            payloads = [guest_payload] + hosts_payload

            if args.protection == "dp":
                for payload in payloads:
                    payload["weights"] = {
                        key: previous_global_state[key] + value.cpu()
                        for key, value in payload["weights"].items()
                        if key in previous_global_state
                    }

            global_state = _masked_aggregate(payloads, previous_state=previous_global_state)
            previous_global_state = {k: v.detach().cpu().clone() for k, v in global_state.items()}
            ctx.guest.put(f"fedagat_global_weights_{epoch}", global_state)
            ctx.hosts.put(f"fedagat_global_weights_{epoch}", [global_state] * len(hosts_payload))

            guest_val = _unwrap_ctx_payload(ctx.guest.get(f"fedagat_val_{epoch}"))
            hosts_val = ctx.hosts.get(f"fedagat_val_{epoch}")
            if not isinstance(hosts_val, list):
                hosts_val = [hosts_val]
            val_payloads = [guest_val] + [_unwrap_ctx_payload(item) for item in hosts_val]
            total_elements = sum(max(1, int(item["elements"])) for item in val_payloads)
            global_mae = sum(float(item["mae"]) * max(1, int(item["elements"])) for item in val_payloads) / total_elements
            is_best = global_mae < best_global_mae - min_delta
            if is_best:
                best_global_mae = global_mae
                best_epoch = epoch
                wait_count = 0
            else:
                wait_count += 1
            should_stop = wait_count >= stop_patience
            control = {
                "global_mae": global_mae,
                "best_epoch": best_epoch,
                "is_best": is_best,
                "stop": should_stop,
            }
            ctx.guest.put(f"fedagat_control_{epoch}", control)
            ctx.hosts.put(f"fedagat_control_{epoch}", [control] * len(hosts_payload))
            print(
                f"[FedAGAT Masked Server] epoch={epoch + 1} global_val_mae={global_mae:.6f} "
                f"best_epoch={best_epoch + 1} wait={wait_count}/{stop_patience}",
                flush=True,
            )
            if should_stop:
                break

    if ctx.is_on_arbiter:
        print("[FedAGAT Masked Server] finished.", flush=True)
        return

    if best_state is not None:
        _load_param_state(model, best_state, args.device)
    else:
        best_epoch = max(0, last_epoch - 1)

    synchronize_cuda_for_timing(args.device)
    test_start = time.perf_counter()
    test_stats = _eval_loader(model, test_loader, scaler, args.device)
    synchronize_cuda_for_timing(args.device)
    eff_test_time = time.perf_counter() - test_start
    print(
        f"[TestTimingAudit] model=FedAGAT rank={ctx.rank} "
        f"batches={test_stats['batches']} elements={test_stats['elements']} "
        f"seconds={eff_test_time:.6f}",
        flush=True,
    )

    eff_flops = 0.0
    try:
        from thop import profile
        dummy_x, _ = next(iter(val_loader))
        dummy_x = dummy_x.to(args.device, non_blocking=True)
        flops, _ = profile(model, inputs=(dummy_x,), verbose=False)
        eff_flops = round(flops / 1e9, 4)
    except Exception as exc:
        print(f"Rank {ctx.rank}: [FedAGAT Masked] FLOPs fallback to 0.0: {exc}", flush=True)

    comm_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    eff_comm_size_mb = round((comm_params * 4 * 2 * max(1, last_epoch)) / (1024 * 1024), 4)
    avg_val_time = total_val_time / max(1, last_epoch)

    log_experiment_results(
        model_name=args.model,
        dataset_client=f"{args.dataset_name}_client{ctx.rank}",
        feature_type=args.feature_type,
        best_epoch=best_epoch + 1,
        acc_mae=round(test_stats["mae"], 4),
        acc_mse=round(test_stats["mse"], 4),
        acc_rmse=round(test_stats["rmse"], 4),
        acc_mape=round(test_stats["mape"], 4),
        eff_train_time=round(total_train_time, 2),
        eff_val_time=round(avg_val_time, 4),
        eff_test_time=round(eff_test_time, 4),
        eff_comm_size_mb=eff_comm_size_mb,
        eff_train_round=last_epoch,
        eff_flops=eff_flops,
        acc_elements=test_stats["elements"],
        acc_abs_error_sum=test_stats["abs_error_sum"],
        acc_sq_error_sum=test_stats["sq_error_sum"],
        acc_mape_error_sum=test_stats["mape_error_sum"],
        acc_mape_elements=test_stats["mape_elements"],
    )
    print(
        f"Rank {ctx.rank}: [FedAGAT Masked] test_mae={test_stats['mae']:.4f} "
        f"best_epoch={best_epoch + 1} rounds={last_epoch}",
        flush=True,
    )
