import time
import copy
import math
import torch
import numpy as np
from torch.utils.data import DataLoader
from lib.utils import (
    evaluate_client_model,
    ExplicitEarlyStopper,
    EarlyStopSignal,
    extract_ctx_data,
    unpack_spatiotemporal_batch,
    align_prediction_and_target,
)
from privacy.protection import l2_norm
from privacy.runtime_protection import protected_arbiter_put
from privacy.attack_trace import capture_he_sa_aggregate, capture_he_sa_arbiter_insider, capture_he_sa_server_aggregate_hidden_term


def _fedtse_to_device(tensor, args):
    use_cuda = isinstance(args.device, str) and args.device.startswith("cuda")
    return tensor.to(args.device, non_blocking=use_cuda)


def _fedtse_upload_period(args, client_rank):
    """Get a client participation period from an explicit async schedule."""
    raw_schedule = str(getattr(args, "fedtse_upload_periods", "") or "").strip()
    if not raw_schedule:
        return 1
    try:
        periods = [int(value.strip()) for value in raw_schedule.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError(
            "--fedtse_upload_periods must be a comma-separated list of positive integers."
        ) from exc
    if len(periods) != int(args.num_clients) or any(period <= 0 for period in periods):
        raise ValueError(
            f"FedTSE upload schedule must contain {args.num_clients} positive periods; got {raw_schedule!r}."
        )
    return periods[int(client_rank)]


def _floating_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if value.is_floating_point()
    }


def _client_trace_aggregation_weight(ctx, raw_sizes, protocol_name):
    """Return this client's server-side aggregation coefficient.

    ``all_sizes`` is deliberately constructed on the arbiter.  HE trace
    capture also runs on each client, so clients must receive that public
    coefficient list rather than trying to access the arbiter-local variable.
    """
    sizes = raw_sizes
    # FATE returns a direct guest value as ``[s0, ...]`` but may return a
    # host broadcast as ``[[s0, ...], [s0, ...], [s0, ...]]``.  Each nested
    # entry is the same public weight vector, so unwrap it only after checking
    # that they agree.  This must not silently select a different client's
    # vector.
    while isinstance(sizes, (list, tuple)) and sizes and all(
        isinstance(value, (list, tuple)) for value in sizes
    ):
        candidate = list(sizes[0])
        if any(list(value) != candidate for value in sizes[1:]):
            raise RuntimeError(
                f"{protocol_name} HE trace broadcasts disagree across recipients: {sizes!r}"
            )
        sizes = candidate
    if not isinstance(sizes, (list, tuple)):
        raise RuntimeError(f"{protocol_name} HE trace weights were not broadcast as a sequence: {type(sizes)!r}")
    sizes = [float(value) for value in sizes]
    rank = int(ctx.rank)
    if rank < 0 or rank >= len(sizes):
        raise RuntimeError(f"{protocol_name} HE trace rank {rank} is outside broadcast sizes {sizes!r}")
    return sizes[rank] / max(sum(sizes), 1.0)


def train_fedtse_task(ctx, args, setting):
    is_he = str(getattr(args, "protection", "plain")).lower() == "he"
    
    # 1. 步数与数据量同步 (避免 Arbiter setting=None 崩溃)
    if not ctx.is_on_arbiter:
        train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = setting
        
        pin_memory = isinstance(args.device, str) and args.device.startswith("cuda")
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=pin_memory)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)
        
        local_data_size = len(train_set)
        ctx.arbiter.put("data_size", local_data_size)
        ctx.arbiter.put("steps_tr", len(train_loader))
        ctx.arbiter.put("steps_val", len(val_loader))
        ctx.arbiter.put("steps_te", len(test_loader))
        trace_aggregation_weight = None
        
        STEPS_TRAIN = len(train_loader)
        STEPS_VAL = len(val_loader)
        STEPS_TEST = len(test_loader)
        
        stopper = ExplicitEarlyStopper(ctx, args, patience=50, min_delta=1e-4)
        best_norm_mae, best_epoch, best_model_wts = float('inf'), -1, None
        total_train_time, total_val_time = 0.0, 0.0
        actual_epochs = 0
        
        # 平台版工程规则：固定异步参与机制 (避免引入复杂 A3C)
        upload_period = _fedtse_upload_period(args, ctx.rank)
        if args.protection in ('dp', 'he'):
            # The initialization is public and is only used to reconstruct a
            # full model from each protected model delta on the arbiter.
            ctx.arbiter.put("fedtse_dp_initial_state", _floating_state_dict(model))
        print(
            f"[FedTSE Client {ctx.rank}] upload_period={upload_period}; "
            f"schedule={getattr(args, 'fedtse_upload_periods', '') or 'synchronous'}",
            flush=True,
        )
    else:
        def get_min_steps(tag):
            s_g = ctx.guest.get(tag)
            s_h = ctx.hosts.get(tag)
            return min([s_g] + (s_h if isinstance(s_h, list) else [s_h]))
            
        STEPS_TRAIN = get_min_steps("steps_tr")
        STEPS_VAL = get_min_steps("steps_val")
        STEPS_TEST = get_min_steps("steps_te")
        
        size_g = ctx.guest.get("data_size")
        size_h = ctx.hosts.get("data_size")
        all_sizes = [size_g] + (size_h if isinstance(size_h, list) else [size_h])
        total_data_size = sum(all_sizes)
        if is_he:
            # The actual HE aggregation uses these data-size weights on the
            # arbiter. Broadcast the same public coefficients for client-side
            # trace instrumentation; do not access arbiter-local ``all_sizes``
            # from client code.
            ctx.guest.put("__fedtse_he_trace_sizes", all_sizes)
            ctx.hosts.put("__fedtse_he_trace_sizes", [all_sizes] * (len(all_sizes) - 1))
        
        # Arbiter 维护的异步缓存
        cached_weights = [None] * args.num_clients
        cached_e_actual = [0] * args.num_clients
        dp_clip_norm = None
        if args.protection in ('dp', 'he'):
            dp_global_weights = ctx.guest.get("fedtse_dp_initial_state")
            init_hosts = ctx.hosts.get("fedtse_dp_initial_state")
            init_hosts = init_hosts if isinstance(init_hosts, list) else [init_hosts]
            # A delta-based protected path needs a single public reference
            # model.  Previously the arbiter consumed this handshake without
            # returning it, leaving independently initialised clients.
            ctx.guest.put("fedtse_dp_initial_state", dp_global_weights)
            ctx.hosts.put(
                "fedtse_dp_initial_state",
                [dp_global_weights] * len(init_hosts) if len(init_hosts) > 1 else dp_global_weights,
            )

    he_context = None
    he_arbiter_context = None
    he_upload_bytes = he_download_bytes = 0
    if is_he:
        from privacy.ckks_backend import (
            ciphertext_bytes as ckks_ciphertext_bytes,
            decrypt_tree as ckks_decrypt_tree,
            encrypt_tree as ckks_encrypt_tree,
            export_public_context as ckks_export_public_context,
            generate_context as ckks_generate_context,
            homomorphic_weighted_sum_tree as ckks_homomorphic_weighted_sum_tree,
            import_context as ckks_import_context,
        )
        if str(getattr(args, "he_backend", "auto")).lower() not in ("auto", "he_sa"):
            raise ValueError("FedTSE supports HE-SA only for encrypted cached model aggregation.")
        key_tag = "__fedtse_he_sa_ckks_context"
        if ctx.is_on_arbiter:
            he_arbiter_context = ckks_generate_context(
                int(args.he_ckks_poly_modulus_degree), int(args.he_ckks_scale_bits),
            )
            public_context = ckks_export_public_context(he_arbiter_context)
            ctx.guest.put(key_tag, public_context); ctx.hosts.put(key_tag, public_context)
            print("[HESA] FedTSE arbiter generated packed CKKS context", flush=True)
        else:
            trace_aggregation_weight = _client_trace_aggregation_weight(
                ctx, ctx.arbiter.get("__fedtse_he_trace_sizes"), "FedTSE"
            )
            he_context = ckks_import_context(ctx.arbiter.get(key_tag))

    try:
        for epoch in range(args.epochs):
            if not ctx.is_on_arbiter:
                # ================= Client 训练与异步上传 =================
                round_start_weights = _floating_state_dict(model)
                model.train()
                epoch_loss = 0.0
                train_start = time.time()
                for i, batch in enumerate(train_loader):
                    if i >= STEPS_TRAIN: break
                    x, y = unpack_spatiotemporal_batch(batch)
                    x, y = _fedtse_to_device(x, args), _fedtse_to_device(y, args)
                    optimizer.zero_grad()
                    pred = model(x)
                    pred, y = align_prediction_and_target(pred, y)

                    loss = loss_func(pred, y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    
                total_train_time += (time.time() - train_start)
                print(f"Client {ctx.rank} Epoch {epoch}: Train Loss (Norm) = {epoch_loss/max(STEPS_TRAIN,1):.4f}")

                # [关键] 判定是否为“静默轮次”，构建 Payload
                participates = (epoch % upload_period == 0)
                uploaded_weights = _floating_state_dict(model) if participates else None
                if args.protection == 'dp' and participates:
                    uploaded_weights = {
                        key: uploaded_weights[key] - round_start_weights[key]
                        for key in round_start_weights
                    }
                    if float(args.dp_clip_norm) <= 0:
                        ctx.arbiter.put(
                            f"fedtse_dp_delta_norm_{epoch}",
                            float(l2_norm(uploaded_weights).item()),
                        )
                        calibrated_clip = ctx.arbiter.get(f"fedtse_dp_clip_norm_{epoch}")
                        if isinstance(calibrated_clip, (list, tuple)):
                            calibrated_clip = calibrated_clip[0]
                        args.dp_clip_norm = float(calibrated_clip)
                        print(
                            f"[DPCalibration] FedTSE rank={ctx.rank} epoch={epoch + 1} "
                            f"clip_norm={args.dp_clip_norm:.8f}",
                            flush=True,
                        )
                payload = {
                    "participates": participates,
                    "weights": uploaded_weights,
                    "e_actual": epoch if participates else None
                }
                if args.protection == 'dp':
                    protection_start = time.perf_counter()
                    protected_arbiter_put(ctx, args, f"fedtse_dp_payload_{epoch}", payload)
                    total_train_time += time.perf_counter() - protection_start
                elif is_he:
                    upload_bytes = 0
                    if participates:
                        # The HE protocol sends absolute weights.  At epoch 0
                        # every client participates and all cached starts are
                        # shared, so the weighted aggregate is equivalently the
                        # common start plus this weighted delta.  Trace that
                        # delta for server-only aggregate reconstruction.
                        aggregate_delta = {
                            key: uploaded_weights[key] - round_start_weights[key]
                            for key in round_start_weights
                        }
                        capture_he_sa_arbiter_insider(
                            ctx, args, f"fedtse_delta_{epoch}", payload=aggregate_delta,
                            model_state_dict=round_start_weights,
                        )
                        capture_he_sa_server_aggregate_hidden_term(
                            ctx, args, f"fedtse_delta_{epoch}", payload=aggregate_delta,
                            model_state_dict=round_start_weights,
                            aggregation_weight=trace_aggregation_weight,
                            leak_type="model_update",
                        )
                        protection_start = time.perf_counter()
                        encrypted_weights = ckks_encrypt_tree(
                            uploaded_weights, he_context,
                            slot_count=int(args.he_ckks_poly_modulus_degree) // 2,
                        )
                        upload_bytes = ckks_ciphertext_bytes(encrypted_weights)
                        he_upload_bytes += upload_bytes
                        payload["weights"] = encrypted_weights
                        total_train_time += time.perf_counter() - protection_start
                        print(f"[HESA] FedTSE rank={ctx.rank} epoch={epoch + 1} encrypted_upload_bytes={upload_bytes}", flush=True)
                    # Every client must publish this bookkeeping tag; an
                    # asynchronous non-participant contributes zero bytes.
                    ctx.arbiter.put(f"fedtse_he_upload_bytes_{epoch}", upload_bytes)
                    ctx.arbiter.put(f"fedtse_he_payload_{epoch}", payload)
                else:
                    ctx.arbiter.put(f"payload_{epoch}", payload)

                # 使用 extract_ctx_data 优雅解包全局参数
                global_w_data = ctx.arbiter.get(f"gw_{epoch}")
                global_w = extract_ctx_data(ctx, global_w_data)
                if is_he:
                    he_download_bytes += sum(value.numel() * value.element_size() for value in global_w.values())
                    he_seconds = ctx.arbiter.get(f"fedtse_he_arbiter_seconds_{epoch}")
                    while isinstance(he_seconds, (tuple, list)):
                        he_seconds = he_seconds[0]
                    total_train_time += float(he_seconds)
                model.load_state_dict(global_w, strict=(args.protection not in ('dp', 'he')))

                # ================= 本地验证阶段 =================
                model.eval()
                val_start = time.time()
                val_mae, val_rmse, val_elements = 0.0, 0.0, 0
                with torch.no_grad():
                    for i, batch in enumerate(val_loader):
                        if i >= STEPS_VAL: break
                        x, y = unpack_spatiotemporal_batch(batch)
                        x, y = _fedtse_to_device(x, args), _fedtse_to_device(y, args)
                        pred = model(x)
                        pred, y = align_prediction_and_target(pred, y)

                        y_cpu, pred_cpu = y.cpu().numpy(), pred.cpu().numpy()
                        val_mae += np.sum(np.abs(y_cpu - pred_cpu))
                        val_rmse += np.sum((y_cpu - pred_cpu)**2)
                        val_elements += y.numel()

                norm_mae = val_mae / val_elements
                total_val_time += (time.time() - val_start)
                print(f"   ---> [验证(归一化)] Client {ctx.rank} Epoch {epoch} | Norm MAE: {norm_mae:.4f}")

                if norm_mae < best_norm_mae:
                    best_norm_mae, best_epoch = norm_mae, epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                
                actual_epochs = epoch + 1
                should_stop = bool(stopper.check_and_sync(norm_mae))
                # The arbiter is not part of ExplicitEarlyStopper.  Publish
                # the decision after this round's aggregation so it can
                # terminate instead of waiting forever for a next payload.
                ctx.arbiter.put(f"fedtse_stop_{epoch}", should_stop)
                if should_stop:
                    raise EarlyStopSignal("触发早停")
                    
            else:
                # ================= Server AsynWeight 聚合 =================
                if args.protection == 'dp' and dp_clip_norm is None:
                    norm_guest = float(ctx.guest.get(f"fedtse_dp_delta_norm_{epoch}"))
                    norm_hosts = ctx.hosts.get(f"fedtse_dp_delta_norm_{epoch}")
                    norm_hosts = norm_hosts if isinstance(norm_hosts, list) else [norm_hosts]
                    dp_clip_norm = float(np.quantile([norm_guest] + [float(value) for value in norm_hosts], 0.9))
                    ctx.guest.put(f"fedtse_dp_clip_norm_{epoch}", dp_clip_norm)
                    ctx.hosts.put(f"fedtse_dp_clip_norm_{epoch}", [dp_clip_norm] * len(norm_hosts))
                    print(
                        f"[DPCalibration] FedTSE arbiter epoch={epoch + 1} "
                        f"clip_norm={dp_clip_norm:.8f}",
                        flush=True,
                    )
                payload_tag = (
                    f"fedtse_dp_payload_{epoch}" if args.protection == 'dp'
                    else f"fedtse_he_payload_{epoch}" if is_he else f"payload_{epoch}"
                )
                p_guest = ctx.guest.get(payload_tag)
                p_hosts = ctx.hosts.get(payload_tag)
                p_hosts = p_hosts if isinstance(p_hosts, list) else [p_hosts]
                all_payloads = [p_guest] + p_hosts

                # 更新活跃客户端的缓存
                for i, p in enumerate(all_payloads):
                    if p["participates"]:
                        if args.protection == 'dp':
                            cached_weights[i] = {
                                key: dp_global_weights[key] + value
                                for key, value in p["weights"].items()
                            }
                        else:
                            cached_weights[i] = p["weights"]
                        cached_e_actual[i] = p["e_actual"]

                # 完美复现 AsynWeight 公式：p_i * exp(-(e - e_actual))
                if any(weight is None for weight in cached_weights):
                    raise RuntimeError("FedTSE received no initial model from one or more clients.")

                # Paper Eq. (1): |D_i| / |D| * exp(-(e - e_i^actual)).
                # In this cached, round-based simulator the raw coefficients
                # need to be normalized before a parameter average.  Otherwise
                # a silent client makes their sum < 1 and repeatedly shrinks
                # every model parameter toward zero.
                raw_coefs = []
                for i in range(args.num_clients):
                    p_i = all_sizes[i] / total_data_size
                    decay = math.exp(-(epoch - cached_e_actual[i]))
                    raw_coefs.append(p_i * decay)

                raw_coef_sum = sum(raw_coefs)
                if raw_coef_sum <= 0:
                    raise RuntimeError("FedTSE temporal aggregation weights must sum to a positive value.")
                coefs = [coef / raw_coef_sum for coef in raw_coefs]

                # 加权融合：归一化后的时间权重始终构成参数平均。
                if is_he:
                    he_started = time.perf_counter()
                    encrypted_global = ckks_homomorphic_weighted_sum_tree(cached_weights, coefs, he_arbiter_context)
                    global_w = ckks_decrypt_tree(encrypted_global, he_arbiter_context)
                    capture_he_sa_aggregate(ctx, args, f"fedtse_aggregate_{epoch}", global_w)
                    he_seconds = time.perf_counter() - he_started
                    guest_bytes = int(ctx.guest.get(f"fedtse_he_upload_bytes_{epoch}"))
                    host_bytes = ctx.hosts.get(f"fedtse_he_upload_bytes_{epoch}")
                    host_bytes = host_bytes if isinstance(host_bytes, list) else [host_bytes]
                    print(f"[HESA] FedTSE arbiter epoch={epoch + 1} aggregate_decrypt_s={he_seconds:.6f} encrypted_upload_bytes={guest_bytes + sum(int(value) for value in host_bytes)}", flush=True)
                else:
                    global_w = {}
                    for k in cached_weights[0].keys():
                        global_w[k] = sum(coefs[i] * cached_weights[i][k] for i in range(args.num_clients))
                if args.protection == 'dp':
                    dp_global_weights = global_w

                print(
                    f"[FedTSE Server] epoch={epoch} raw_temporal_weights="
                    f"{','.join(f'{coef:.6f}' for coef in raw_coefs)} "
                    f"sum={raw_coef_sum:.6f} normalized_temporal_weights="
                    f"{','.join(f'{coef:.6f}' for coef in coefs)}",
                    flush=True,
                )

                ctx.guest.put(f"gw_{epoch}", global_w)
                ctx.hosts.put(f"gw_{epoch}", [global_w] * len(p_hosts))
                if is_he:
                    ctx.guest.put(f"fedtse_he_arbiter_seconds_{epoch}", he_seconds)
                    ctx.hosts.put(f"fedtse_he_arbiter_seconds_{epoch}", [he_seconds] * len(p_hosts))
                stop_guest = bool(ctx.guest.get(f"fedtse_stop_{epoch}"))
                stop_hosts = ctx.hosts.get(f"fedtse_stop_{epoch}")
                stop_hosts = stop_hosts if isinstance(stop_hosts, list) else [stop_hosts]
                if stop_guest or any(bool(value) for value in stop_hosts):
                    print(f"[FedTSE] arbiter early-stop at epoch={epoch + 1}; returning normally.", flush=True)
                    break

    except EarlyStopSignal:
        if not ctx.is_on_arbiter:
            print(f"Rank {ctx.rank}: ⚔️ 早停触发，实际运行 {actual_epochs} 轮，进入最终 Test 阶段。")

    # ================= 最终测试与严格开销统计 =================
    if not ctx.is_on_arbiter:
        if best_model_wts: model.load_state_dict(best_model_wts)
        
        test_start = time.time()
        total_abs_error_sum = 0.0
        total_sq_error_sum = 0.0
        total_mape_error_sum = 0.0
        total_elements = 0
        total_mape_elements = 0

        with torch.no_grad():
            for batch in test_loader:
                x_test, y_test = unpack_spatiotemporal_batch(batch)
                x_test, y_test = _fedtse_to_device(x_test, args), _fedtse_to_device(y_test, args)
                pred_test = model(x_test)
                pred_test, y_test = align_prediction_and_target(pred_test, y_test)

                y_real = scaler.inverse_transform(y_test).cpu().numpy()
                pred_real = scaler.inverse_transform(pred_test).cpu().numpy()
                diff = pred_real - y_real

                total_abs_error_sum += float(np.sum(np.abs(diff)))
                total_sq_error_sum += float(np.sum(diff ** 2))
                total_elements += int(y_real.size)

                mask = y_real > 0.5
                valid_mape = int(np.sum(mask))
                if valid_mape > 0:
                    total_mape_error_sum += float(np.sum(np.abs(diff[mask]) / y_real[mask]))
                    total_mape_elements += valid_mape

        acc_mae = total_abs_error_sum / max(total_elements, 1)
        acc_mse = total_sq_error_sum / max(total_elements, 1)
        acc_rmse = math.sqrt(acc_mse)
        acc_mape = total_mape_error_sum / max(total_mape_elements, 1) if total_mape_elements > 0 else 0.0
        eff_test_time = time.time() - test_start
        
        # 【要求 1 & 4 满足】严格统一通信量口径：按真实轮数与实际收发频次核算
        params_count = sum(p.numel() for p in model.parameters())
        # 统计真实上传的次数 (is_stale == False 的轮数)
        upload_count = sum(1 for e in range(actual_epochs) if e % upload_period == 0)
        # 假设客户端每轮都需要接收最新全局模型用于验证和下一轮训练
        download_count = actual_epochs  
        
        # 统一公式：(参数量 * 4 字节 * 总收发次数) / 1024 / 1024 -> MB
        eff_comm_mb = (
            round((he_upload_bytes + he_download_bytes) / (1024 * 1024), 4)
            if is_he else round((params_count * 4 * (upload_count + download_count)) / (1024 * 1024), 4)
        )

        # 【要求 1 满足】计算 FLOPs
        eff_flops = 0.0
        try:
            from thop import profile
            dummy_x, _ = unpack_spatiotemporal_batch(next(iter(val_loader)))
            flops, _ = profile(model, inputs=(_fedtse_to_device(dummy_x, args),), verbose=False)
            eff_flops = round(flops / 1e9, 4)
            print(f"Rank {ctx.rank}: FLOPs 估算完成: {eff_flops} G")
        except Exception as e: 
            print(f"Rank {ctx.rank}: FLOPs 计算失败: {e}")

        # 【要求 3 满足】严格返回 Test 集物理指标和实际资源消耗
        return (
                best_epoch, acc_mae, acc_rmse, acc_mape,
                total_elements, total_abs_error_sum, total_sq_error_sum, total_mape_error_sum, total_mape_elements,
                total_train_time, total_val_time / max(actual_epochs, 1), eff_test_time,
                eff_comm_mb, actual_epochs, eff_flops
        )
    
    return None
