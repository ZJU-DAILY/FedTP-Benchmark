import torch
import numpy as np
import time
import copy
import math
from sklearn.cluster import KMeans
from scipy.sparse.linalg import svds  # 引入截断 SVD
from lib.utils import ExplicitEarlyStopper, EarlyStopSignal, synchronize_cuda_for_timing
from privacy.protection import l2_norm
from privacy.runtime_protection import (
    protected_arbiter_put, record_he_ttp_downlink, unprotect_he_ttp_payload,
)
from privacy.attack_trace import capture_revised_quantized_prediction

# ================= 通用安全通信工具 =================
def robust_unpack(data):
    """Client专用：递归剥除FATE多余的列表外壳，直到获取单体Payload"""
    while isinstance(data, list) and len(data) > 0:
        data = data[0]
    return data

def flatten_list(nested_list):
    """Server专用：将FATE收集到的鬼畜嵌套列表彻底展平为一维列表"""
    flat = []
    if not isinstance(nested_list, list):
        return [nested_list]
    for item in nested_list:
        if isinstance(item, list):
            flat.extend(flatten_list(item))
        else:
            flat.append(item)
    return flat

def normalize_tsvd_vector_payload(payload, expected_dim):
    data = payload
    while isinstance(data, list) and len(data) == 1 and isinstance(data[0], (list, tuple, np.ndarray)):
        data = data[0]
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    vector = np.asarray(data, dtype=np.float32).reshape(-1)
    if vector.size != expected_dim:
        raise ValueError(
            f"[STFAM Server] Expected TSVD vector dim {expected_dim}, got {vector.size}"
        )
    return vector


def _payload_bytes(payload):
    """Serialized numeric payload size used by the Plain STFAM protocol.

    This deliberately counts the same tensor trees that HE-TTP wraps.  The
    old parameter-count proxy charged every client for an extractor update,
    even when the protocol sent the public ``SKIP`` token instead.
    """
    if torch.is_tensor(payload):
        return int(payload.numel() * payload.element_size())
    if isinstance(payload, np.ndarray):
        return int(payload.nbytes)
    if isinstance(payload, dict):
        return sum(_payload_bytes(value) for value in payload.values())
    if isinstance(payload, (tuple, list)):
        return sum(_payload_bytes(value) for value in payload)
    return 0

# ================= 联邦主流程 =================
def train_stfam(ctx, model, optimizer, train_loader, val_loader, test_loader, loss_func, scaler, args):
    is_server = ctx.is_on_arbiter
    device = args.device
    # Include the one-time TSVD/KMeans setup in HE-TTP efficiency.  Starting
    # the clock after it made HE appear artificially faster than Plain.
    synchronize_cuda_for_timing(device)
    run_started = time.perf_counter()

    # STFAM exposes an initialization TSVD signature and, when drift is
    # detected, a local feature-extractor state.  These are the two client
    # uploads protected by DP. Parameter calibration can be deferred until a
    # round actually contains a selected uploader.
    stfam_is_dp = str(getattr(args, "protection", "plain")).lower() == "dp"
    stfam_is_he = str(getattr(args, "protection", "plain")).lower() == "he"
    fixed_dp_clip = float(getattr(args, "dp_clip_norm", 0.0))
    stfam_dp_clips = ({
        key: fixed_dp_clip for key in ("tsvd", "params")
    } if stfam_is_dp and fixed_dp_clip > 0 else {})
    # Keep exact bytes rather than a parameter-count proxy.  It is defined
    # before TSVD because STFAM has an initialization-only client upload.
    protocol_comm_bytes = 0

    def _client_dp_upload(payload_key, tag, value):
        if stfam_is_he:
            if value is None:
                return ctx.arbiter.put(tag, "SKIP")
            return protected_arbiter_put(ctx, args, tag, value)
        if not stfam_is_dp:
            return ctx.arbiter.put(tag, value)
        clip_norm = stfam_dp_clips.get(payload_key)
        if clip_norm is None:
            norm_value = 0.0 if value is None else float(l2_norm(value).item())
            ctx.arbiter.put(f"stfam_dp_{payload_key}_norm_{tag}", norm_value)
            calibrated = robust_unpack(ctx.arbiter.get(f"stfam_dp_{payload_key}_clip_{tag}"))
            if calibrated is not None:
                clip_norm = float(calibrated)
                stfam_dp_clips[payload_key] = clip_norm
                print(
                    f"[DPCalibration] STFAM rank={ctx.rank} type={payload_key} "
                    f"tag={tag} clip_norm={clip_norm:.8f}", flush=True,
                )
        if value is None:
            return ctx.arbiter.put(tag, "SKIP")
        if clip_norm is None or clip_norm <= 0:
            raise RuntimeError(f"STFAM DP clip norm unavailable for active {payload_key} upload")
        return protected_arbiter_put(ctx, args, tag, value, clip_norm=clip_norm)

    def _server_dp_calibrate(payload_key, tag):
        if not stfam_is_dp or payload_key in stfam_dp_clips:
            return
        norm_guest = float(ctx.guest.get(f"stfam_dp_{payload_key}_norm_{tag}"))
        norm_hosts = ctx.hosts.get(f"stfam_dp_{payload_key}_norm_{tag}")
        norm_hosts = norm_hosts if isinstance(norm_hosts, list) else [norm_hosts]
        norms = [norm_guest] + [float(value) for value in norm_hosts]
        active_norms = [value for value in norms if value > 0]
        clip_norm = float(np.quantile(active_norms, 0.9)) if active_norms else None
        if clip_norm is not None:
            stfam_dp_clips[payload_key] = clip_norm
            print(
                f"[DPCalibration] STFAM arbiter type={payload_key} tag={tag} "
                f"clip_norm={clip_norm:.8f}", flush=True,
            )
        ctx.guest.put(f"stfam_dp_{payload_key}_clip_{tag}", clip_norm)
        host_clips = [clip_norm] * len(norm_hosts)
        ctx.hosts.put(
            f"stfam_dp_{payload_key}_clip_{tag}",
            host_clips if len(host_clips) > 1 else host_clips[0],
        )

    # ================= Phase 0: 初始化与 TSVD 聚类 =================
    if not is_server:
        print(f"[Client {ctx.rank}] 执行 TSVD 提取本地数据主向量 (启用子采样与截断SVD防爆内存)...")
        all_Tr = []
        max_samples_for_svd = 256 # 截取 256 个样本，足以代表整体时空分布，内存占用极小 (< 1GB)
        current_samples = 0
        
        for Tr_batch, _, _ in train_loader:
            all_Tr.append(Tr_batch.cpu().numpy())
            current_samples += Tr_batch.shape[0]
            if current_samples >= max_samples_for_svd:
                break
                
        all_Tr = np.concatenate(all_Tr, axis=0) 
        Tr_matrix = all_Tr.reshape(-1, all_Tr.shape[-1]).T 
        
        # 核心修复：使用截断 SVD 只算前 K 个奇异值，避免全量 SVD 卡死和内存溢出
        k_svd = min(args.tsvd_dim, min(Tr_matrix.shape) - 1)
        if k_svd > 0:
            _, Sigma, _ = svds(Tr_matrix, k=k_svd)
            Sigma = Sigma[::-1] # svds 返回的是升序奇异值，需翻转为降序
        else:
            _, Sigma, _ = np.linalg.svd(Tr_matrix, full_matrices=False)
            
        principal_vector = np.zeros(args.tsvd_dim)
        fill_len = min(args.tsvd_dim, len(Sigma))
        principal_vector[:fill_len] = Sigma[:fill_len]
        
        if stfam_is_dp:
            _client_dp_upload(
                "tsvd", "tsvd_vector",
                torch.as_tensor(principal_vector, dtype=torch.float32),
            )
        elif stfam_is_he:
            protected_arbiter_put(
                ctx, args, "tsvd_vector",
                torch.as_tensor(principal_vector, dtype=torch.float32),
            )
        else:
            ctx.arbiter.put("tsvd_vector", principal_vector.tolist())
        protocol_comm_bytes += _payload_bytes(principal_vector)
        
    else:
        print("[Server] 接收 TSVD 主向量并执行 K-Means 聚类...")
        _server_dp_calibrate("tsvd", "tsvd_vector")
        h_guest = unprotect_he_ttp_payload(args, ctx.guest.get("tsvd_vector"))
        h_hosts = ctx.hosts.get("tsvd_vector")
        h_hosts = [
            unprotect_he_ttp_payload(args, payload)
            for payload in (h_hosts if isinstance(h_hosts, list) else [h_hosts])
        ]
        
        host_payloads = h_hosts if isinstance(h_hosts, list) else [h_hosts]
        all_vectors = [normalize_tsvd_vector_payload(h_guest, args.tsvd_dim)]
        all_vectors.extend(
            normalize_tsvd_vector_payload(payload, args.tsvd_dim)
            for payload in host_payloads
        )
        sample_count = len(all_vectors)
        if sample_count <= 0:
            raise ValueError("[STFAM Server] No TSVD vectors received from clients.")
        vectors_np = np.stack(all_vectors, axis=0)
        
        vectors_norm = vectors_np / (np.linalg.norm(vectors_np, axis=1, keepdims=True) + 1e-8)
        requested_clusters = int(getattr(args, "num_clusters", 1) or 1)
        effective_clusters = max(1, min(requested_clusters, sample_count))
        if effective_clusters != requested_clusters:
            print(
                f"[STFAM Server] Adjusting num_clusters from {requested_clusters} to "
                f"{effective_clusters} because only {sample_count} client vectors are available.",
                flush=True,
            )
        kmeans = KMeans(n_clusters=effective_clusters, random_state=args.seed).fit(vectors_norm)
        clusters = kmeans.labels_ 
        print(f"[Server] 聚类完成，各客户端归属簇: {clusters}")

    # ================= 联邦训练主循环 =================
    global_state_prev = None
    # DP must bound the data-dependent *update*, not the absolute local
    # parameter state.  Absolute states are dominated by public model values,
    # giving an unnecessarily large C and noise scale for a sample-level
    # reconstruction experiment.  Establish one public common base first;
    # this is ordinary federated-model state, not a client data upload.
    if stfam_is_dp:
        public_base_tag = "stfam_dp_public_base"
        if not is_server:
            local_base = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
                if "local_" in key and torch.is_tensor(value)
            }
            ctx.arbiter.put(public_base_tag, local_base)
            synchronized_base = robust_unpack(ctx.arbiter.get(f"{public_base_tag}_sync"))
            if not isinstance(synchronized_base, dict):
                raise RuntimeError("STFAM DP public base is not a parameter-state dictionary.")
            model.load_state_dict(synchronized_base, strict=False)
        else:
            # A common random initialization is public in this federated
            # protocol.  Use the guest copy as the canonical base and publish
            # it to all clients before their first local update.
            public_base = robust_unpack(ctx.guest.get(public_base_tag))
            if not isinstance(public_base, dict):
                raise RuntimeError("STFAM DP failed to receive the public initial local state.")
            global_state_prev = {
                key: value.detach().cpu().clone()
                for key, value in public_base.items()
                if torch.is_tensor(value)
            }
            ctx.guest.put(f"{public_base_tag}_sync", global_state_prev)
            ctx.hosts.put(
                f"{public_base_tag}_sync",
                [global_state_prev] * max(1, int(args.num_clients) - 1),
            )
        print(
            "[STFAM DP] upload_mode=delta_update; "
            "clip calibration is defined on local_state - public_base.",
            flush=True,
        )
    best_norm_mae, best_epoch = float('inf'), -1
    best_model_wts = None
    total_val_time = 0.0

    stopper = ExplicitEarlyStopper(ctx, args, patience=50, min_delta=1e-4)

    if not is_server:
        actual_epochs = 0
        heartbeat_every = max(0, int(getattr(args, "stfam_heartbeat_batches", 50)))

    try:
        for epoch in range(args.epochs):
            if not is_server:
                actual_epochs = epoch + 1
                model.train()
                old_params = {k: v.clone().detach() for k, v in model.named_parameters() if 'local_' in k}
                old_local_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                    if 'local_' in key and torch.is_tensor(value)
                }
                epoch_loss = 0.0  
                batch_count = 0   
                
                for Tr, U, y in train_loader:
                    if heartbeat_every and batch_count % heartbeat_every == 0:
                        print(
                            f"[STFAM heartbeat] rank={ctx.rank} epoch={epoch} "
                            f"starting_batch={batch_count + 1}",
                            flush=True,
                        )
                    Tr, U, y = Tr.to(device), U.to(device), y.to(device)
                    optimizer.zero_grad()
                    with torch.no_grad():
                        monitor_pred, _, _ = model(Tr, U, args.V_D, args.P_embed)
                    capture_revised_quantized_prediction(
                        ctx, args, f"stfam_prediction_{epoch}_{batch_count}",
                        prediction=monitor_pred, model_state_dict=model.state_dict(),
                    )
                    loss = model.compute_loss(Tr, U, y, args.V_D, args.P_embed) 
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()  
                    batch_count += 1           
                    if heartbeat_every and batch_count % heartbeat_every == 0:
                        # Makes the log a precise liveness signal: completion
                        # of the CUDA work above is required before it prints.
                        synchronize_cuda_for_timing(device)
                        print(
                            f"[STFAM heartbeat] rank={ctx.rank} epoch={epoch} "
                            f"finished_batch={batch_count} loss={loss.item():.6f}",
                            flush=True,
                        )
                
                avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
                print(f"🌟 [Client {ctx.rank}] Epoch {epoch} 本地训练平均 Loss (Norm): {avg_loss:.4f}", flush=True)
                    
                delta_norm = 0.0
                for k, v in model.named_parameters():
                    if 'local_' in k:
                        delta_norm += torch.norm(v - old_params[k], p=2).item() ** 2
                    
                print(f"[STFAM sync] rank={ctx.rank} epoch={epoch} upload_norm", flush=True)
                ctx.arbiter.put(f"norm_{epoch}", delta_norm)
                print(f"[STFAM sync] rank={ctx.rank} epoch={epoch} await_threshold", flush=True)
                threshold = robust_unpack(ctx.arbiter.get(f"threshold_{epoch}"))
                
                if delta_norm >= threshold:
                    if stfam_is_dp:
                        # This is the exact data-dependent payload protected
                        # by DP.  The server adds the aggregated noisy delta
                        # back to the public base below.
                        feature_extractor_payload = {
                            key: (value.detach() - old_local_state[key]).cpu().clone()
                            for key, value in model.state_dict().items()
                            if key in old_local_state and torch.is_tensor(value)
                        }
                    else:
                        feature_extractor_payload = {
                            key: value.cpu().clone()
                            for key, value in model.state_dict().items()
                            if 'local_' in key
                        }
                    _client_dp_upload("params", f"params_{epoch}", feature_extractor_payload)
                    protocol_comm_bytes += _payload_bytes(feature_extractor_payload)
                else:
                    if stfam_is_dp:
                        _client_dp_upload("params", f"params_{epoch}", None)
                    else:
                        ctx.arbiter.put(f"params_{epoch}", "SKIP") 
                    
                # FATE can expose the three host return payloads as a list.
                # They are replicas of the same global state here, so first
                # reduce to this client's one payload and then account bytes.
                # Counting before unpacking inflated host downlink traffic.
                global_state = robust_unpack(ctx.arbiter.get(f"global_params_{epoch}"))
                global_state = record_he_ttp_downlink(
                    args, global_state, tag=f"global_params_{epoch}",
                )
                model.load_state_dict(global_state, strict=False)
                protocol_comm_bytes += _payload_bytes(global_state)

                # ================= 验证环节 =================
                model.eval()
                val_start_time = time.time()
                val_mae, val_rmse = 0.0, 0.0
                total_elements = 0

                with torch.no_grad():
                    for Tr_val, U_val, y_val in val_loader:
                        Tr_val, U_val, y_val = Tr_val.to(device), U_val.to(device), y_val.to(device)
                        pred, _, _ = model(Tr_val, U_val, args.V_D, args.P_embed)
                        if pred.shape != y_val.shape: pred = pred.reshape_as(y_val)

                        total_elements += y_val.numel()
                        y_cpu = y_val.cpu().numpy()
                        pred_cpu = pred.cpu().numpy()

                        val_mae += np.sum(np.abs(y_cpu - pred_cpu))
                        val_rmse += np.sum((y_cpu - pred_cpu) ** 2)

                total_val_time += (time.time() - val_start_time)
                norm_mae = val_mae / total_elements
                norm_rmse = math.sqrt(val_rmse / total_elements)

                print(f"   ---> [STFAM 验证结果(Norm)] Epoch {epoch} | MAE: {norm_mae:.4f} | RMSE: {norm_rmse:.4f}")

                if norm_mae < best_norm_mae:
                    best_norm_mae, best_epoch = norm_mae, epoch
                    best_model_wts = copy.deepcopy(model.state_dict())

                should_stop = stopper.check_and_sync(norm_mae)
                if should_stop:
                    print(f"Rank {ctx.rank}: 🛑 收到全局早停信号，在第 {actual_epochs} 轮提前跳出！")
                    raise EarlyStopSignal("触发早停")

            else:
                print(f"[STFAM Server] epoch={epoch} awaiting norms from all {args.num_clients} clients", flush=True)
                n_guest = ctx.guest.get(f"norm_{epoch}")
                n_hosts = ctx.hosts.get(f"norm_{epoch}")
                all_norms = flatten_list([n_guest, n_hosts])
                
                threshold = float(np.mean(all_norms))
                ctx.guest.put(f"threshold_{epoch}", threshold)
                ctx.hosts.put(f"threshold_{epoch}", [threshold] * len(all_norms[1:]))
                
                _server_dp_calibrate("params", f"params_{epoch}")
                p_guest = unprotect_he_ttp_payload(args, ctx.guest.get(f"params_{epoch}"))
                p_hosts = ctx.hosts.get(f"params_{epoch}")
                p_hosts = [
                    unprotect_he_ttp_payload(args, payload)
                    for payload in (p_hosts if isinstance(p_hosts, list) else [p_hosts])
                ]
                all_params = flatten_list([p_guest, p_hosts])
                
                uploaded_clients, skipped_clients = [], []
                for i, p in enumerate(all_params):
                    if isinstance(p, str) and p == "SKIP": skipped_clients.append(i)
                    else: uploaded_clients.append((i, p))
                        
                final_params_list = [None] * args.num_clients
                for client_idx, param in uploaded_clients:
                    final_params_list[client_idx] = param
                    
                for missing_idx in skipped_clients:
                    my_cluster = clusters[missing_idx]
                    peer_params = [all_params[i] for i in range(args.num_clients) if clusters[i] == my_cluster and not isinstance(all_params[i], str)]
                    if len(peer_params) > 0:
                        sub_param = {k: torch.stack([p[k] for p in peer_params]).mean(dim=0) for k in peer_params[0].keys()}
                        final_params_list[missing_idx] = sub_param
                    else:
                        if stfam_is_dp:
                            # A non-uploading client contributes a zero delta
                            # when it has no cluster peer to impute from.
                            reference_delta = uploaded_clients[0][1]
                            final_params_list[missing_idx] = {
                                key: torch.zeros_like(value)
                                for key, value in reference_delta.items()
                            }
                        else:
                            final_params_list[missing_idx] = global_state_prev if global_state_prev is not None else uploaded_clients[0][1]

                mean_payload = {
                    key: torch.stack([p[key] for p in final_params_list]).mean(dim=0)
                    for key in final_params_list[0].keys()
                }
                if stfam_is_dp:
                    if global_state_prev is None:
                        raise RuntimeError("STFAM DP global base is unavailable for delta aggregation.")
                    global_state = {
                        key: global_state_prev[key].to(mean_delta.device, dtype=mean_delta.dtype) + mean_delta
                        for key, mean_delta in mean_payload.items()
                    }
                else:
                    global_state = mean_payload
                global_state_prev = global_state 
                ctx.guest.put(f"global_params_{epoch}", global_state)
                ctx.hosts.put(f"global_params_{epoch}", [global_state] * len(all_norms[1:]))

    except EarlyStopSignal:
        if not is_server:
            print(f"Rank {ctx.rank}: ⚔️ 早停触发，实际运行 {actual_epochs} 轮，进入测试。")

    # ================= 终极物理尺度 Test 与 落盘计算 =================
    if not is_server:
        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)
            print(f"Rank {ctx.rank}: [STFAM] 已成功回滚至第 {best_epoch + 1} 轮的最优模型！")

        print(f"Rank {ctx.rank}: 🚀 启动最终物理尺度 Test 评估...")
        model.eval()
        synchronize_cuda_for_timing(device)
        test_start_time = time.perf_counter()
        test_mae, test_rmse, test_mape = 0.0, 0.0, 0.0
        test_elements, valid_mape_count = 0, 0
        test_batches = 0
        per_horizon_abs_error_sum = None
        per_horizon_sq_error_sum = None
        per_horizon_mape_error_sum = None
        per_horizon_elements = None
        per_horizon_mape_elements = None

        with torch.no_grad():
            for Tr_test, U_test, y_test in test_loader:
                test_batches += 1
                Tr_test, U_test, y_test = Tr_test.to(device), U_test.to(device), y_test.to(device)

                pred, _, _ = model(Tr_test, U_test, args.V_D, args.P_embed)
                if pred.shape != y_test.shape: pred = pred.reshape_as(y_test)

                y_real = scaler.inverse_transform(y_test).cpu().numpy()
                pred_real = scaler.inverse_transform(pred).cpu().numpy()
                if y_real.ndim == 3:
                    y_real = np.expand_dims(y_real, axis=-1)
                    pred_real = np.expand_dims(pred_real, axis=-1)

                abs_error = np.abs(y_real - pred_real)
                sq_error = (y_real - pred_real) ** 2

                test_mae += np.sum(abs_error)
                test_rmse += np.sum(sq_error)
                test_elements += y_real.size

                horizon_steps = abs_error.shape[2]
                if per_horizon_abs_error_sum is None:
                    per_horizon_abs_error_sum = np.zeros(horizon_steps, dtype=np.float64)
                    per_horizon_sq_error_sum = np.zeros(horizon_steps, dtype=np.float64)
                    per_horizon_mape_error_sum = np.zeros(horizon_steps, dtype=np.float64)
                    per_horizon_elements = np.zeros(horizon_steps, dtype=np.int64)
                    per_horizon_mape_elements = np.zeros(horizon_steps, dtype=np.int64)

                reduce_axes = (0, 1, 3)
                per_horizon_abs_error_sum += abs_error.sum(axis=reduce_axes)
                per_horizon_sq_error_sum += sq_error.sum(axis=reduce_axes)
                per_horizon_elements += np.full(
                    horizon_steps,
                    abs_error.shape[0] * abs_error.shape[1] * abs_error.shape[3],
                    dtype=np.int64,
                )

                mask = y_real > 0.5
                if np.sum(mask) > 0:
                    test_mape += np.sum(abs_error[mask] / y_real[mask])
                    valid_mape_count += np.sum(mask)
                    per_horizon_mape_error_sum += np.where(
                        mask,
                        abs_error / np.maximum(y_real, 1e-12),
                        0.0,
                    ).sum(axis=reduce_axes)
                    per_horizon_mape_elements += mask.sum(axis=reduce_axes)

        acc_mae = test_mae / test_elements
        acc_rmse = math.sqrt(test_rmse / test_elements)
        acc_mape = (test_mape / valid_mape_count) * 100 if valid_mape_count > 0 else 0.0
        synchronize_cuda_for_timing(device)
        eff_test_time = time.perf_counter() - test_start_time
        print(
            f"[TestTimingAudit] model=STFAM rank={ctx.rank} batches={test_batches} "
            f"elements={test_elements} seconds={eff_test_time:.6f}",
            flush=True,
        )

        if per_horizon_abs_error_sum is not None:
            print(f"Rank {ctx.rank}: [STFAM Per-Horizon Test Metrics]")
            for h_idx in range(len(per_horizon_abs_error_sum)):
                h_mae = per_horizon_abs_error_sum[h_idx] / max(per_horizon_elements[h_idx], 1)
                h_rmse = math.sqrt(per_horizon_sq_error_sum[h_idx] / max(per_horizon_elements[h_idx], 1))
                h_mape = (
                    per_horizon_mape_error_sum[h_idx] / max(per_horizon_mape_elements[h_idx], 1) * 100.0
                    if per_horizon_mape_elements[h_idx] > 0 else 0.0
                )
                print(
                    f"  Horizon {h_idx + 1:02d} | "
                    f"MAE: {h_mae:.4f} | RMSE: {h_rmse:.4f} | MAPE: {h_mape:.4f}"
                )

            early_end = min(3, len(per_horizon_abs_error_sum))
            late_start = max(0, len(per_horizon_abs_error_sum) - 3)
            early_mae = per_horizon_abs_error_sum[:early_end].sum() / max(per_horizon_elements[:early_end].sum(), 1)
            late_mae = per_horizon_abs_error_sum[late_start:].sum() / max(per_horizon_elements[late_start:].sum(), 1)
            early_rmse = math.sqrt(
                per_horizon_sq_error_sum[:early_end].sum() / max(per_horizon_elements[:early_end].sum(), 1)
            )
            late_rmse = math.sqrt(
                per_horizon_sq_error_sum[late_start:].sum() / max(per_horizon_elements[late_start:].sum(), 1)
            )
            early_mape = (
                per_horizon_mape_error_sum[:early_end].sum() / max(per_horizon_mape_elements[:early_end].sum(), 1) * 100.0
                if per_horizon_mape_elements[:early_end].sum() > 0 else 0.0
            )
            late_mape = (
                per_horizon_mape_error_sum[late_start:].sum() / max(per_horizon_mape_elements[late_start:].sum(), 1) * 100.0
                if per_horizon_mape_elements[late_start:].sum() > 0 else 0.0
            )
            print(
                f"Rank {ctx.rank}: [STFAM Horizon Summary] "
                f"early(1-{early_end}) MAE/RMSE/MAPE = "
                f"{early_mae:.4f}/{early_rmse:.4f}/{early_mape:.4f} | "
                f"late({late_start + 1}-{len(per_horizon_abs_error_sum)}) MAE/RMSE/MAPE = "
                f"{late_mae:.4f}/{late_rmse:.4f}/{late_mape:.4f}"
            )

        eff_flops = 0.0
        try:
            from thop import profile
            class STFAM_Wrapper(torch.nn.Module):
                def __init__(self, m, vd, p):
                    super().__init__()
                    self.m = m
                    self.vd = vd
                    self.p = p
                def forward(self, tr, u):
                    return self.m(tr, u, self.vd, self.p)[0]

            wrapper = STFAM_Wrapper(model, args.V_D, args.P_embed).to(device)
            dummy_tr, dummy_u, _ = next(iter(val_loader))
            flops, _ = profile(wrapper, inputs=(dummy_tr.to(device), dummy_u.to(device)), verbose=False)
            eff_flops = flops / 1e9
        except Exception as e:
            pass

        synchronize_cuda_for_timing(device)
        total_train_time = time.perf_counter() - run_started - total_val_time - eff_test_time
        return best_epoch, acc_mae, acc_rmse, acc_mape, total_train_time, total_val_time, eff_test_time, protocol_comm_bytes, actual_epochs, eff_flops
    else:
        return [None]*10
