# Keep the timer module under a private name: some legacy Fed4TP variants use
# ``time`` as a loop/local variable, which otherwise shadows this import.
import time as _time
import pywt
import torch
import numpy as np
import math
import copy
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from lib.fed4tp_utils import slice_data_for_twt
from lib.utils import (
    evaluate_client_model, ExplicitEarlyStopper, EarlyStopSignal,
    synchronize_cuda_for_timing,
)
from privacy.protection import l2_norm
from privacy.runtime_protection import (
    protected_arbiter_put, record_he_ttp_downlink, unprotect_he_ttp_payload,
)
from privacy.attack_trace import capture_he_ttp_insider_upper_bound

try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

def apply_wavelet_denoising(dataset):
    X_tensor, Y_tensor = dataset.tensors
    data_np = X_tensor.cpu().numpy()
    try:
        coeffs = pywt.wavedec(data_np, 'db4', level=3, axis=1) 
        coeffs_filtered = [pywt.threshold(c, 0.5 * np.max(np.abs(c)), mode='soft') for c in coeffs]
        data_denoised = pywt.waverec(coeffs_filtered, 'db4', axis=1)
        if data_denoised.shape != data_np.shape:
            data_denoised = data_denoised[:, :data_np.shape[1], :]
        X_new = torch.from_numpy(data_denoised).float().to(X_tensor.device)
        print("  -> [GLD] 本地小波去噪完成！")
        return TensorDataset(X_new, Y_tensor)
    except Exception as e:
        print(f"  -> [GLD Error] 去噪失败，保持原数据: {e}")
        return dataset

def flatten_weights(weights_dict):
    return torch.cat([v.contiguous().view(-1) for v in weights_dict.values()])


def _floating_state_dict(model):
    """Return only federatable floating-point state.

    Some graph backbones additionally contain node-indexed parameters (for
    example DyHSL's node embedding).  Their shape legitimately differs between
    the 9-node and 10-node METIS clients.  The arbiter performs a second,
    shape-compatibility filter before aggregation, while ``strict=False`` on
    reception preserves every client-local tensor.
    """
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key != "adj" and value.is_floating_point()
    }

def train_fed4tp_task(ctx, args, setting=None):
    """
    Fed4TP 终极严谨版: TWT(嵌套循环+先右后左搜索) + MPL(参数交集) + GLD(余弦相似度)
    """
    # ==========================================
    #            Server 端逻辑 
    # ==========================================
    if ctx.is_on_arbiter:
        print("[Fed4TP Server] 启动...")
        num_clients = args.num_clients
        gd_use = getattr(args, 'fed4tp_gd_use', False)
        mg_use = getattr(args, 'fed4tp_mg_use', False)
        rho = getattr(args, 'rho', 0.1)
        
        historical_avg_model = None
        model_count = 0
        dp_clip_norm = None
        if args.protection == "dp":
            initial_guest = ctx.guest.get("fed4tp_dp_initial_state")
            initial_hosts = ctx.hosts.get("fed4tp_dp_initial_state")
            initial_hosts = initial_hosts if isinstance(initial_hosts, list) else [initial_hosts]
            dp_client_weights = [initial_guest] + initial_hosts
        
        # 【TWT 修复】：外层 Time Window，内层 Epoch
        for t in range(args.time_window_num):
            for e in range(args.epochs):
                tag = f"{t}_{e}"
                print(f"\n[Server] Window {t} | Epoch {e} Start...")
                
                # --- 1. 接收权重与 TWT 标志 ---
                if args.protection == "dp" and dp_clip_norm is None:
                    norm_guest = float(ctx.guest.get(f"fed4tp_dp_delta_norm_{tag}"))
                    norm_hosts = ctx.hosts.get(f"fed4tp_dp_delta_norm_{tag}")
                    norm_hosts = norm_hosts if isinstance(norm_hosts, list) else [norm_hosts]
                    dp_clip_norm = float(np.quantile([norm_guest] + [float(value) for value in norm_hosts], 0.9))
                    ctx.guest.put(f"fed4tp_dp_clip_norm_{tag}", dp_clip_norm)
                    ctx.hosts.put(f"fed4tp_dp_clip_norm_{tag}", [dp_clip_norm] * len(norm_hosts))
                    print(f"[DPCalibration] Fed4TP arbiter epoch={tag} clip_norm={dp_clip_norm:.8f}", flush=True)

                payload_tag = f"fed4tp_dp_payload_{tag}" if args.protection == "dp" else f"weights_{tag}"
                w_guest = unprotect_he_ttp_payload(args, ctx.guest.get(payload_tag))
                w_hosts = ctx.hosts.get(payload_tag)
                if not isinstance(w_hosts, list): w_hosts = [w_hosts]
                w_hosts = [unprotect_he_ttp_payload(args, payload) for payload in w_hosts]
                all_payloads = [w_guest] + w_hosts
                if args.protection == "dp":
                    for client_id, payload in enumerate(all_payloads):
                        payload["weights"] = {
                            key: dp_client_weights[client_id][key] + value
                            for key, value in payload["weights"].items()
                        }
                
                # --- 2. GLD 模型检测 ---
                unreliable_flags = []
                if gd_use and historical_avg_model is not None:
                    flat_h = flatten_weights(historical_avg_model)
                    for client_id, payload in enumerate(all_payloads):
                        is_unreliable = False
                        if not payload["skip"]:
                            flat_w = flatten_weights({
                                key: payload["weights"][key]
                                for key in historical_avg_model
                                if key in payload["weights"]
                                and tuple(payload["weights"][key].shape) == tuple(historical_avg_model[key].shape)
                            })
                            cos_sim = F.cosine_similarity(flat_w.unsqueeze(0), flat_h.unsqueeze(0)).item()
                            if cos_sim < rho:
                                is_unreliable = True
                                print(f"  -> 🚨 [GLD] Client {client_id} 模型异常 (CosSim: {cos_sim:.4f} < 阈值 {rho})")
                        unreliable_flags.append(is_unreliable)
                else:
                    unreliable_flags = [False] * num_clients
                    
                ctx.guest.put(f"flag_{tag}", unreliable_flags[0])
                ctx.hosts.put(f"flag_{tag}", unreliable_flags[1:] if len(unreliable_flags)>2 else unreliable_flags[1])
                
                # --- 3. 过滤有效客户端 ---
                active_clients = []
                total_global_data_size = 0
                for i, payload in enumerate(all_payloads):
                    if not payload["skip"]:
                        active_clients.append(i)
                        total_global_data_size += payload["data_size"]
                
                personalized_global_weights = [None] * num_clients
                fallback_global_weights = None
                
                if len(active_clients) == 0:
                    print(f"  -> [TWT 警告] 本轮全部空窗，跳过全局聚合！")
                    for i in range(num_clients):
                        personalized_global_weights[i] = {"weights": all_payloads[i]["weights"], "global_size": 0}
                else:
                    # --- 4. MPL: Top-K Mask 交集 ---
                    # A METIS split of 307 PeMS04 sensors creates 9-node and
                    # 10-node clients.  Only parameters with the same shape on
                    # every active client are valid for cross-client averaging.
                    # Node-indexed tensors are intentionally left local.
                    first_active = active_clients[0]
                    first_weights = all_payloads[first_active]["weights"]
                    shared_keys = [
                        key for key, value in first_weights.items()
                        if all(
                            key in all_payloads[cid]["weights"]
                            and key in all_payloads[cid]["top_k_mask"]
                            and tuple(all_payloads[cid]["weights"][key].shape) == tuple(value.shape)
                            and tuple(all_payloads[cid]["top_k_mask"][key].shape) == tuple(value.shape)
                            for cid in active_clients
                        )
                    ]
                    if not shared_keys:
                        raise RuntimeError(
                            "Fed4TP found no shape-compatible parameters across active clients."
                        )
                    global_mask_intersection = {}
                    for key in shared_keys:
                        intersect = all_payloads[first_active]["top_k_mask"][key]
                        for cid in active_clients[1:]:
                            intersect = intersect & all_payloads[cid]["top_k_mask"][key]
                        global_mask_intersection[key] = intersect

                    fallback_global_weights = {}
                    for key in shared_keys:
                        fallback_global_weights[key] = sum([all_payloads[c]["weights"][key] * all_payloads[c]["data_size"] for c in active_clients]) / total_global_data_size

                    # --- 5. 分组聚合 ---
                    if mg_use and len(active_clients) > 1:
                        client_vectors = [
                            flatten_weights({key: all_payloads[idx]["weights"][key] for key in shared_keys})
                            for idx in active_clients
                        ]
                        sim_matrix = np.zeros((len(active_clients), len(active_clients)))
                        for i in range(len(active_clients)):
                            for j in range(len(active_clients)):
                                sim_matrix[i, j] = F.cosine_similarity(client_vectors[i].unsqueeze(0), client_vectors[j].unsqueeze(0)).item()
                        
                        num_groups = max(1, len(active_clients) // 2)
                        labels = KMeans(n_clusters=num_groups, random_state=0, n_init='auto').fit(sim_matrix).labels_ if HAS_SKLEARN else np.zeros(len(active_clients))
                            
                        for act_idx, client_id in enumerate(active_clients):
                            group_id = labels[act_idx]
                            group_members = [active_clients[i] for i, l in enumerate(labels) if l == group_id]
                            group_data_size = sum([all_payloads[m]["data_size"] for m in group_members])
                            
                            client_final_weights = {}
                            for key in shared_keys:
                                is_global = global_mask_intersection[key]
                                w_global = fallback_global_weights[key]
                                w_personal = sum([all_payloads[m]["weights"][key] * all_payloads[m]["data_size"] for m in group_members]) / group_data_size
                                client_final_weights[key] = torch.where(is_global, w_global, w_personal)
                                
                            personalized_global_weights[client_id] = {"weights": client_final_weights, "global_size": total_global_data_size}
                    else:
                        for client_id in active_clients:
                            personalized_global_weights[client_id] = {"weights": fallback_global_weights, "global_size": total_global_data_size}

                    for i in range(num_clients):
                        if personalized_global_weights[i] is None:
                            personalized_global_weights[i] = {"weights": fallback_global_weights, "global_size": total_global_data_size}

                # --- 6. GLD：更新历史平均模型 ---
                if gd_use and fallback_global_weights is not None:
                    if historical_avg_model is None:
                        historical_avg_model = {k: v.clone() for k, v in fallback_global_weights.items()}
                        model_count = 1
                    else:
                        model_count += 1
                        for k in historical_avg_model.keys():
                            historical_avg_model[k] = (historical_avg_model[k] * (model_count - 1) + fallback_global_weights[k]) / model_count

                # --- 7. 分发权重 ---
                # With MPL grouping disabled (the default), every active
                # client receives the same fallback state.  Broadcast that one
                # state instead of a 31-element duplicate list: the latter
                # crosses FATE's table threshold at 32 clients.
                if not mg_use and fallback_global_weights is not None:
                    shared_downlink = {
                        "weights": fallback_global_weights,
                        "global_size": total_global_data_size,
                    }
                    ctx.guest.put(f"global_weights_{tag}", shared_downlink)
                    ctx.hosts.put(f"global_weights_{tag}", shared_downlink)
                else:
                    ctx.guest.put(f"global_weights_{tag}", personalized_global_weights[0])
                    ctx.hosts.put(f"global_weights_{tag}", personalized_global_weights[1:] if len(personalized_global_weights)>2 else personalized_global_weights[1])
                if args.protection == "dp":
                    # This is exactly the client state that will be used as
                    # the reference point for its next protected delta.
                    next_states = []
                    for client_id, payload in enumerate(all_payloads):
                        server_state = personalized_global_weights[client_id]["weights"]
                        global_size = personalized_global_weights[client_id]["global_size"]
                        if payload["skip"] and global_size > 0:
                            local_size = payload["data_size"]
                            next_states.append({
                                key: (
                                    (global_size * server_state[key] + local_size * payload["weights"][key])
                                    / (global_size + local_size)
                                    if key in server_state else payload["weights"][key]
                                )
                                for key in payload["weights"]
                            })
                        else:
                            next_states.append({
                                key: (server_state[key] if key in server_state else value).detach().cpu().clone()
                                for key, value in payload["weights"].items()
                            })
                    dp_client_weights = next_states

                stop_guest = bool(ctx.guest.get(f"fed4tp_stop_{tag}"))
                stop_hosts = ctx.hosts.get(f"fed4tp_stop_{tag}")
                stop_hosts = stop_hosts if isinstance(stop_hosts, list) else [stop_hosts]
                if stop_guest or any(bool(value) for value in stop_hosts):
                    print(f"[Fed4TP] arbiter early-stop at window={t}, epoch={e}; returning normally.", flush=True)
                    return None
            
        return None

    # ==========================================
    #            Client 端逻辑 
    # ==========================================
    else:
        print(f"[Fed4TP Client {ctx.rank}] 启动...")
        import copy
        train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = setting
        
        twt_data_list = slice_data_for_twt(train_set, args.time_window_num)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        
        is_unreliable = False
        best_norm_mae = float('inf')
        best_global_epoch = -1
        best_model_wts = None
        total_train_time, total_val_time = 0.0, 0.0
        
        stopper = ExplicitEarlyStopper(ctx, args, patience=50, min_delta=1e-4)
        global_epoch_counter = 0 
        if args.protection == "dp":
            ctx.arbiter.put("fed4tp_dp_initial_state", _floating_state_dict(model))
        
        try:
            # 【TWT 修复】：外层 Time Window，内层 Epoch
            for t in range(args.time_window_num):
                
                # --- 1. TWT 核心逻辑：空窗判定与回退 (基于时间窗粒度) ---
                current_dataset = twt_data_list[t]
                local_data_size = len(current_dataset)
                skip_upload = False
                
                if local_data_size == 0:
                    skip_upload = True
                    found = False
                    for offset in range(1, len(twt_data_list)):
                        right, left = t + offset, t - offset
                        # 【TWT 修复】：严格按照原著，先向右找，再向左找
                        if right < len(twt_data_list) and len(twt_data_list[right]) > 0:
                            current_dataset = twt_data_list[right]; found = True; break
                        if left >= 0 and len(twt_data_list[left]) > 0:
                            current_dataset = twt_data_list[left]; found = True; break
                    local_data_size = len(current_dataset)
                    print(f"[Client {ctx.rank}] Window {t} 为空！向 {'右' if right < len(twt_data_list) and len(twt_data_list[right]) > 0 else '左'} 借用数据，隔离上传。")
                
                for e in range(args.epochs):
                    tag = f"{t}_{e}"
                    round_start_weights = _floating_state_dict(model)
                    protocol_started = _time.perf_counter()
                    model.train()
                    
                    # --- 2. GLD 去噪 (根据上一轮 Server 判定) ---
                    if is_unreliable and getattr(args, 'fed4tp_gd_use', False):
                        current_dataset = apply_wavelet_denoising(current_dataset)
                        is_unreliable = False 
                        
                    train_loader = DataLoader(current_dataset, batch_size=args.batch_size, shuffle=True)
                    
                    # --- 3. 本地训练 (加入全 Epoch 梯度累加器) ---
                    epoch_loss = 0.0
                    num_batches = 0
                    
                    # 初始化梯度累加器 (仅针对需要求导的参数)
                    epoch_grads_accum = {name: torch.zeros_like(p.data) for name, p in model.named_parameters() if p.requires_grad}
                    
                    for x, y in train_loader:
                        x, y = x.to(args.device), y.to(args.device)
                        optimizer.zero_grad()
                        pred = model(x)
                        if isinstance(pred, tuple): pred = pred[0] 
                        if y.dim() == 4 and y.shape[-1] == 1: y = y.squeeze(-1)
                        if pred.dim() == 4 and pred.shape[-1] == 1: pred = pred.squeeze(-1)
                        if pred.shape != y.shape: pred = pred.reshape_as(y)
                        
                        loss = loss_func(pred, y)
                        loss.backward()
                        
                        # 【核心修复】：在 step 之前，累加当前 batch 的绝对梯度
                        with torch.no_grad():
                            for name, p in model.named_parameters():
                                if p.grad is not None:
                                    # 累加绝对值，使用 detach() 防止显存泄漏
                                    epoch_grads_accum[name] += p.grad.abs().detach()
                        
                        optimizer.step()
                        epoch_loss += loss.item()
                        num_batches += 1
                        
                    epoch_loss_avg = epoch_loss / max(len(train_loader), 1)
                    train_log = {
                        'loss': round(epoch_loss_avg, 4),
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'epoch': float(global_epoch_counter)
                    }
                    print(str(train_log))
                        
                    # --- 4. MPL: Top-K 梯度 Mask (基于全 Epoch 平均梯度) ---
                    top_k_mask = {}
                    with torch.no_grad():
                        all_grads = []
                        # 计算全 Epoch 平均梯度
                        for name, p in model.named_parameters():
                            if p.requires_grad:
                                avg_grad = epoch_grads_accum[name] / max(num_batches, 1)
                                all_grads.append(avg_grad.cpu().flatten())
                                
                        if len(all_grads) > 0:
                            concat_grads = torch.cat(all_grads)
                            k_idx = int(concat_grads.numel() * 0.2) # 取 Top 20%
                            threshold = torch.topk(concat_grads, k_idx).values[-1]
                            
                            # 生成 Mask
                            for name, p in model.named_parameters():
                                if p.requires_grad:
                                    avg_grad = epoch_grads_accum[name] / max(num_batches, 1)
                                    top_k_mask[name] = (avg_grad.cpu() >= threshold)
                                else:
                                    top_k_mask[name] = torch.zeros_like(p.data.cpu(), dtype=torch.bool)

                    # --- 5. 权重交互 ---
                    local_weights = _floating_state_dict(model)
                    payload = {"weights": local_weights, "skip": skip_upload, "data_size": local_data_size, "top_k_mask": top_k_mask}
                    if args.protection == "dp":
                        local_delta = {
                            key: local_weights[key] - round_start_weights[key]
                            for key in round_start_weights
                        }
                        if float(args.dp_clip_norm) <= 0:
                            ctx.arbiter.put(f"fed4tp_dp_delta_norm_{tag}", float(l2_norm(local_delta).item()))
                            calibrated_clip = ctx.arbiter.get(f"fed4tp_dp_clip_norm_{tag}")
                            if isinstance(calibrated_clip, (list, tuple)):
                                calibrated_clip = calibrated_clip[0]
                            args.dp_clip_norm = float(calibrated_clip)
                            print(f"[DPCalibration] Fed4TP rank={ctx.rank} epoch={tag} clip_norm={args.dp_clip_norm:.8f}", flush=True)
                        payload["weights"] = local_delta
                        protected_arbiter_put(ctx, args, f"fed4tp_dp_payload_{tag}", payload)
                    elif args.protection == "he":
                        capture_he_ttp_insider_upper_bound(
                            ctx, args, f"fed4tp_delta_{tag}",
                            observed_leak={key: local_weights[key] - round_start_weights[key] for key in round_start_weights},
                            model_state_dict=round_start_weights, leak_type="model_update",
                        )
                        protected_arbiter_put(ctx, args, f"weights_{tag}", payload)
                    else:
                        ctx.arbiter.put(f"weights_{tag}", payload)
                    
                    # 接收 GLD Flag
                    flag_data = ctx.arbiter.get(f"flag_{tag}")
                    is_unreliable = flag_data[ctx.rank - 1] if isinstance(flag_data, list) else flag_data
                    
                    # 接收权重
                    server_payload_data = ctx.arbiter.get(f"global_weights_{tag}")
                    server_payload = (
                        server_payload_data[ctx.rank - 1]
                        if isinstance(server_payload_data, list)
                        else server_payload_data
                    )
                    server_payload = record_he_ttp_downlink(
                        args, server_payload, tag=f"global_weights_{tag}",
                    )
                    global_w, global_size = server_payload["weights"], server_payload["global_size"]
                    
                    # --- 6. TWT 补偿更新 (公式 10) ---
                    updated_weights = {}
                    if skip_upload and global_size > 0:
                        for k in local_weights.keys():
                            if k in global_w:
                                updated_weights[k] = (global_size * global_w[k].to(args.device) + local_data_size * local_weights[k].to(args.device)) / (global_size + local_data_size)
                            else:
                                updated_weights[k] = local_weights[k].to(args.device)
                    else:
                        for k in global_w.keys():
                            updated_weights[k] = global_w[k].to(args.device)
                    model.load_state_dict(updated_weights, strict=False)
                    # Include local optimisation, upload, arbiter wait and
                    # returned global state in one protocol-round timing.
                    total_train_time += _time.perf_counter() - protocol_started
                    
                    # --- 7. 验证集评估与标准 HF 日志输出 ---
                    val_start = _time.time()
                    model.eval()
                    val_loss, val_mae, val_rmse, val_mape = 0.0, 0.0, 0.0, 0.0
                    total_val_elements = 0
                    valid_val_mape_count = 0
                    
                    with torch.no_grad():
                        for x_val, y_val in val_loader:
                            x_val, y_val = x_val.to(args.device), y_val.to(args.device)
                            pred_val = model(x_val)
                            if isinstance(pred_val, tuple): pred_val = pred_val[0]
                            if y_val.dim() == 4 and y_val.shape[-1] == 1: y_val = y_val.squeeze(-1)
                            if pred_val.dim() == 4 and pred_val.shape[-1] == 1: pred_val = pred_val.squeeze(-1)
                            if pred_val.shape != y_val.shape: pred_val = pred_val.reshape_as(y_val)
                            
                            # 计算归一化 Loss
                            loss = loss_func(pred_val, y_val)
                            val_loss += loss.item()
                            
                            # 计算归一化 MAE / RMSE
                            y_cpu = y_val.cpu().numpy()
                            pred_cpu = pred_val.cpu().numpy()
                            val_mae += np.sum(np.abs(y_cpu - pred_cpu))
                            val_rmse += np.sum((y_cpu - pred_cpu) ** 2)
                            total_val_elements += y_val.numel()

                            # 计算反归一化 MAPE (*100 转换为百分比格式)
                            y_real = scaler.inverse_transform(y_val).cpu().numpy()
                            pred_real = scaler.inverse_transform(pred_val).cpu().numpy()
                            mask = y_real > 0.5
                            if np.sum(mask) > 0:
                                val_mape += np.sum(np.abs(y_real[mask] - pred_real[mask]) / y_real[mask]) * 100.0
                                valid_val_mape_count += np.sum(mask)

                    # 结算各指标
                    norm_val_mae = val_mae / total_val_elements
                    norm_val_rmse = math.sqrt(val_rmse / total_val_elements)
                    norm_val_mse = norm_val_rmse ** 2
                    final_val_mape = (val_mape / valid_val_mape_count) if valid_val_mape_count > 0 else 0.0
                    
                    total_val_time += (_time.time() - val_start)
                    
                    # 【新增】：严格对齐 HF Trainer 的验证日志格式
                    eval_log = {
                        'eval_mae': float(norm_val_mae),
                        'eval_rmse': float(norm_val_rmse),
                        'eval_mse': float(norm_val_mse),
                        'eval_mape': float(final_val_mape),
                        'epoch': float(global_epoch_counter)
                    }
                    print(str(eval_log))
                    
                    # 使用归一化的 MAE 记录最优权重 (防抖动)
                    if norm_val_mae < best_norm_mae:
                        best_norm_mae, best_global_epoch = norm_val_mae, global_epoch_counter
                        best_model_wts = copy.deepcopy(model.state_dict())

                    # --- 8. 早停检测 ---
                    should_stop = bool(stopper.check_and_sync(float(norm_val_mae)))
                    ctx.arbiter.put(f"fed4tp_stop_{tag}", should_stop)
                    global_epoch_counter += 1
                    
                    if should_stop:
                        print(f"Rank {ctx.rank}: 🛑 收到全局早停信号，抛出中断异常！")
                        raise EarlyStopSignal("触发早停机制")

        except EarlyStopSignal:
            print(f"Rank {ctx.rank}: 🎉 成功跳出训练循环 (实际运行 {global_epoch_counter} 轮)！")

        # --- 9. 测试集验收 ---
        print(f"Rank {ctx.rank}: 🚀 开始独立测试集最终评估 (反归一化)...")
        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)
            
        synchronize_cuda_for_timing(args.device)
        test_start = _time.perf_counter()
        model.eval()
        test_mae, test_rmse, test_mape = 0.0, 0.0, 0.0
        test_elements, valid_mape_count = 0, 0
        test_batches = 0
        
        with torch.no_grad():
            for x_t, y_t in test_loader:
                test_batches += 1
                x_t, y_t = x_t.to(args.device), y_t.to(args.device)
                pred_t = model(x_t)
                if isinstance(pred_t, tuple): pred_t = pred_t[0]
                if y_t.dim() == 4 and y_t.shape[-1] == 1: y_t = y_t.squeeze(-1)
                if pred_t.dim() == 4 and pred_t.shape[-1] == 1: pred_t = pred_t.squeeze(-1)
                if pred_t.shape != y_t.shape: pred_t = pred_t.reshape_as(y_t)
                
                y_real = scaler.inverse_transform(y_t).cpu().numpy()
                pred_real = scaler.inverse_transform(pred_t).cpu().numpy()
                
                test_mae += np.sum(np.abs(y_real - pred_real))
                test_rmse += np.sum((y_real - pred_real) ** 2)
                test_elements += y_real.size
                
                mask = y_real > 0.5
                if np.sum(mask) > 0:
                    test_mape += np.sum(np.abs(y_real[mask] - pred_real[mask]) / y_real[mask])
                    valid_mape_count += np.sum(mask)

        acc_mae = test_mae / test_elements
        acc_rmse = math.sqrt(test_rmse / test_elements)
        acc_mape = (test_mape / valid_mape_count) if valid_mape_count > 0 else 0.0
        synchronize_cuda_for_timing(args.device)
        eff_test_time = round(_time.perf_counter() - test_start, 4)
        print(
            f"[TestTimingAudit] model=Fed4TP rank={ctx.rank} batches={test_batches} "
            f"elements={test_elements} seconds={eff_test_time:.6f}",
            flush=True,
        )
        
        comm_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        eff_flops = 0.0
        try:
            from thop import profile
            dummy_x, _ = next(iter(val_loader))
            flops, _ = profile(model, inputs=(dummy_x.to(args.device),), verbose=False)
            eff_flops = round(flops / 1e9, 4)
        except Exception: pass
        
        eff_val_time_per_epoch = round(total_val_time / max(global_epoch_counter, 1), 4)
        
        return (best_global_epoch + 1, round(acc_mae, 4), round(acc_rmse, 4), round(acc_mape, 4), 
                round(total_train_time, 2), eff_val_time_per_epoch, eff_test_time, comm_params, global_epoch_counter, eff_flops)
