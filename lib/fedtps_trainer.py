import time
import math
import copy
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from lib.utils import ExplicitEarlyStopper, EarlyStopSignal
from privacy.protection import l2_norm
from privacy.runtime_protection import (
    protected_arbiter_put, record_he_ttp_downlink, unprotect_he_ttp_payload,
)
from privacy.attack_trace import (
    capture_he_ttp_insider_upper_bound,
    capture_revised_quantized_prediction,
)

def train_fedtps_task(ctx, args, get_setting_fn):
    """
    FedTPS 专用联邦训练流程
    核心：个性化联邦学习 (PFL)，仅共享 Traffic Pattern Repository (Patterns 矩阵)，
    服务端使用 Top-k 余弦相似度感知聚合 (Similarity-aware Aggregation)。
    """
    if ctx.is_on_arbiter:
        print("[FedTPS Server] 启动 FedTPS 服务端相似度感知聚合中心...")
        STEPS_PER_EPOCH = 0
    else:
        print(f"[FedTPS Client {ctx.rank}] 启动...")
        # 提取各个组件（注意这里我们把 test_set 也取出来了）
        train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = get_setting_fn(ctx)
        
        # 数据设备转移
        if hasattr(train_set, 'data') and isinstance(train_set.data, np.ndarray):
            train_set.data = torch.from_numpy(train_set.data).to(args.device)
        if hasattr(val_set, 'data') and isinstance(val_set.data, np.ndarray):
            val_set.data = torch.from_numpy(val_set.data).to(args.device)
        if hasattr(test_set, 'data') and isinstance(test_set.data, np.ndarray):
            test_set.data = torch.from_numpy(test_set.data).to(args.device)
            
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        
        STEPS_PER_EPOCH = len(train_loader)
        ctx.arbiter.put("steps", STEPS_PER_EPOCH)
        
        # 初始化最优指标和监控变量
        best_norm_mae = float('inf')
        best_epoch = -1
        best_model_wts = None
        total_train_time, total_val_time = 0.0, 0.0
        actual_epochs = 0
        
        # 挂载上帝裁判（早停机制）
        stopper = ExplicitEarlyStopper(ctx, args, patience=50, min_delta=1e-4)
        if args.protection == "dp":
            ctx.arbiter.put("fedtps_dp_initial_patterns", model.Patterns.detach().cpu().clone())

    # ---------------- 步数同步 ----------------
    if ctx.is_on_arbiter:
        s_guest = ctx.guest.get("steps")
        s_hosts = ctx.hosts.get("steps")
        if not isinstance(s_hosts, list): s_hosts = [s_hosts]
        STEPS_PER_EPOCH = min([s_guest] + s_hosts)
        dp_clip_norm = None
        if args.protection == "dp":
            initial_guest = ctx.guest.get("fedtps_dp_initial_patterns")
            initial_hosts = ctx.hosts.get("fedtps_dp_initial_patterns")
            initial_hosts = initial_hosts if isinstance(initial_hosts, list) else [initial_hosts]
            dp_client_patterns = [initial_guest] + initial_hosts

    top_k = getattr(args, 'fedtps_k', 2)

    # ---------------- 联邦训练主循环 ----------------
    try:
        for epoch in range(args.epochs):
            
            # ==================== Client 端逻辑 ====================
            if not ctx.is_on_arbiter: 
                round_start_patterns = model.Patterns.detach().cpu().clone()
                model.train()
                epoch_loss = 0.0
                train_start_t = time.time()
                
                # 1. 本地训练 (不进行反归一化)
                for i, batch in enumerate(train_loader):
                    if i >= STEPS_PER_EPOCH: break
                    
                    x, y = batch[0].to(args.device), batch[1].to(args.device)
                    # 动态适配 y_cov
                    y_cov = batch[2].to(args.device) if len(batch) > 2 else None

                    # This belongs to the explicitly revised protocol: a
                    # public fixed-point prediction is returned in addition
                    # to the unchanged HE-protected Patterns upload.  Capture
                    # it before the first local optimizer step so the stored
                    # round-start model state replays the prediction exactly.
                    if i == 0:
                        trace_state = copy.deepcopy(model.state_dict())
                        with torch.no_grad():
                            # A returned monitoring prediction is generated
                            # on the serving path, without teacher forcing.
                            trace_prediction = model(x, y_cov=y_cov)
                        capture_revised_quantized_prediction(
                            ctx,
                            args,
                            f"fedtps_prediction_{epoch}",
                            prediction=trace_prediction,
                            model_state_dict=trace_state,
                        )
                    
                    optimizer.zero_grad()
                    pred = model(x, y_cov=y_cov, labels=y)
                    
                    # 维度对齐兜底
                    if y.dim() == 4 and y.shape[-1] == 1: y = y.squeeze(-1)
                    if pred.dim() == 4 and pred.shape[-1] == 1: pred = pred.squeeze(-1)
                    if pred.shape != y.shape:
                        if pred.dim() == 3 and y.dim() == 3 and pred.shape[1] == y.shape[2] and pred.shape[2] == y.shape[1]:
                            pred = pred.transpose(1, 2)
                        else:
                            pred = pred.reshape_as(y)
                    
                    loss = loss_func(pred, y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                total_train_time += (time.time() - train_start_t)
                
                # 【日志规范】：强制打印 Normalized Loss
                print(f"[Client {ctx.rank}] Epoch {epoch} | Train Loss (Norm): {epoch_loss/max(1, STEPS_PER_EPOCH):.4f}")    
                
                # 2. 提取公共交通模式矩阵并上传
                local_patterns = model.Patterns.detach().cpu().clone()
                if args.protection == "he":
                    capture_he_ttp_insider_upper_bound(
                        ctx, args, f"fedtps_patterns_delta_{epoch}",
                        observed_leak=(local_patterns - round_start_patterns,),
                        model_state_dict={
                            name: (round_start_patterns if name == "Patterns" else param.detach().cpu().clone())
                            for name, param in model.named_parameters()
                        },
                        leak_type="model_update",
                    )
                if args.protection == "dp":
                    local_delta = local_patterns - round_start_patterns
                    if float(args.dp_clip_norm) <= 0:
                        ctx.arbiter.put(f"fedtps_dp_delta_norm_{epoch}", float(l2_norm(local_delta).item()))
                        calibrated_clip = ctx.arbiter.get(f"fedtps_dp_clip_norm_{epoch}")
                        if isinstance(calibrated_clip, (list, tuple)):
                            calibrated_clip = calibrated_clip[0]
                        args.dp_clip_norm = float(calibrated_clip)
                        print(f"[DPCalibration] FedTPS rank={ctx.rank} epoch={epoch + 1} clip_norm={args.dp_clip_norm:.8f}", flush=True)
                    protection_started = time.perf_counter()
                    protected_arbiter_put(ctx, args, f"fedtps_dp_delta_{epoch}", local_delta)
                    total_train_time += time.perf_counter() - protection_started
                elif args.protection == "he":
                    protected_arbiter_put(ctx, args, f"patterns_{epoch}", local_patterns)
                else:
                    ctx.arbiter.put(f"patterns_{epoch}", local_patterns)
                
                # 3. 接收 Server 聚合后的 Pattern，更新本地模型
                agg_patterns_data = record_he_ttp_downlink(
                    args, ctx.arbiter.get(f"agg_patterns_{epoch}"),
                )
                if isinstance(agg_patterns_data, list): 
                    agg_patterns_data = agg_patterns_data[ctx.rank - 1] if len(agg_patterns_data) > 1 else agg_patterns_data[0]
                
                model.Patterns.data.copy_(agg_patterns_data.to(args.device))
                
                # 4. 验证环节 (严格不使用 inverse_transform)
                model.eval()
                val_start_t = time.time()
                val_mae, val_rmse = 0.0, 0.0
                val_elements = 0

                with torch.no_grad():
                    for batch in val_loader:
                        x, y = batch[0].to(args.device), batch[1].to(args.device)
                        y_cov = batch[2].to(args.device) if len(batch) > 2 else None

                        pred = model(x, y_cov=y_cov)

                        if y.dim() == 4 and y.shape[-1] == 1: y = y.squeeze(-1)
                        if pred.dim() == 4 and pred.shape[-1] == 1: pred = pred.squeeze(-1)
                        if pred.shape != y.shape:
                            if pred.dim() == 3 and y.dim() == 3 and pred.shape[1] == y.shape[2] and pred.shape[2] == y.shape[1]:
                                pred = pred.transpose(1, 2)
                            else:
                                pred = pred.reshape_as(y)

                        y_cpu = y.cpu().numpy()
                        pred_cpu = pred.cpu().numpy()

                        val_mae += np.sum(np.abs(y_cpu - pred_cpu))
                        val_rmse += np.sum((y_cpu - pred_cpu) ** 2)
                        val_elements += y.numel()

                norm_val_mae = val_mae / val_elements
                norm_val_rmse = math.sqrt(val_rmse / val_elements)
                total_val_time += (time.time() - val_start_t)
                actual_epochs = epoch + 1 # 真实运行轮数更新

                # 【日志规范】：打印 Normalized Val Metrics
                print(f"   ---> [验证集] Client {ctx.rank} Epoch {epoch} | Val MAE (Norm): {norm_val_mae:.4f} | Val RMSE (Norm): {norm_val_rmse:.4f}")

                # 根据归一化的 Val MAE 保存最优模型权重
                if norm_val_mae < best_norm_mae:
                    best_norm_mae = norm_val_mae
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                
                # 早停裁判
                should_stop = bool(stopper.check_and_sync(norm_val_mae))
                ctx.arbiter.put(f"fedtps_stop_{epoch}", should_stop)
                if should_stop:
                    print(f"Rank {ctx.rank}: 🛑 收到全局早停信号，抛出中断异常！")
                    raise EarlyStopSignal("触发早停")
                
            # ==================== Server 端逻辑 ====================
            else: 
                if args.protection == "dp" and dp_clip_norm is None:
                    norm_guest = float(ctx.guest.get(f"fedtps_dp_delta_norm_{epoch}"))
                    norm_hosts = ctx.hosts.get(f"fedtps_dp_delta_norm_{epoch}")
                    norm_hosts = norm_hosts if isinstance(norm_hosts, list) else [norm_hosts]
                    dp_clip_norm = float(np.quantile([norm_guest] + [float(value) for value in norm_hosts], 0.9))
                    ctx.guest.put(f"fedtps_dp_clip_norm_{epoch}", dp_clip_norm)
                    ctx.hosts.put(f"fedtps_dp_clip_norm_{epoch}", [dp_clip_norm] * len(norm_hosts))
                    print(f"[DPCalibration] FedTPS arbiter epoch={epoch + 1} clip_norm={dp_clip_norm:.8f}", flush=True)
                if args.protection == "dp":
                    patterns_guest = dp_client_patterns[0] + ctx.guest.get(f"fedtps_dp_delta_{epoch}")
                    delta_hosts = ctx.hosts.get(f"fedtps_dp_delta_{epoch}")
                    delta_hosts = delta_hosts if isinstance(delta_hosts, list) else [delta_hosts]
                    patterns_hosts = [
                        dp_client_patterns[index + 1] + value
                        for index, value in enumerate(delta_hosts)
                    ]
                else:
                    patterns_guest = unprotect_he_ttp_payload(args, ctx.guest.get(f"patterns_{epoch}"))
                    patterns_hosts = ctx.hosts.get(f"patterns_{epoch}")
                    patterns_hosts = [
                        unprotect_he_ttp_payload(args, payload)
                        for payload in (patterns_hosts if isinstance(patterns_hosts, list) else [patterns_hosts])
                    ]
                if not isinstance(patterns_hosts, list): patterns_hosts = [patterns_hosts]
                
                all_patterns = [patterns_guest] + patterns_hosts 
                M = len(all_patterns)
                aggregated_patterns_list = []
                
                for m in range(M):
                    Wm_p = all_patterns[m]
                    pattern_num, pattern_dim = Wm_p.shape
                    new_Wm_p = torch.zeros_like(Wm_p)
                    
                    for i in range(pattern_num):
                        query = Wm_p[i:i+1]
                        sum_similar_patterns = torch.zeros_like(query)
                        
                        for n in range(M):
                            Wn_p = all_patterns[n]
                            sim = F.cosine_similarity(query, Wn_p)
                            actual_k = min(top_k, pattern_num) 
                            _, topk_indices = torch.topk(sim, actual_k)
                            sum_similar_patterns += Wn_p[topk_indices].sum(dim=0, keepdim=True)
                        
                        new_Wm_p[i] = sum_similar_patterns / (M * actual_k)
                    
                    aggregated_patterns_list.append(new_Wm_p)
                
                ctx.guest.put(f"agg_patterns_{epoch}", aggregated_patterns_list[0])
                host_aggs = aggregated_patterns_list[1:]
                if len(host_aggs) == 1:
                    ctx.hosts.put(f"agg_patterns_{epoch}", host_aggs[0])
                elif len(host_aggs) > 1:
                    ctx.hosts.put(f"agg_patterns_{epoch}", host_aggs)
                if args.protection == "dp":
                    dp_client_patterns = [pattern.detach().cpu().clone() for pattern in aggregated_patterns_list]
                stop_guest = bool(ctx.guest.get(f"fedtps_stop_{epoch}"))
                stop_hosts = ctx.hosts.get(f"fedtps_stop_{epoch}")
                stop_hosts = stop_hosts if isinstance(stop_hosts, list) else [stop_hosts]
                if stop_guest or any(bool(value) for value in stop_hosts):
                    print(f"[FedTPS] arbiter early-stop at epoch={epoch + 1}; returning normally.", flush=True)
                    break

    except EarlyStopSignal:
        if not ctx.is_on_arbiter:
            print(f"Rank {ctx.rank}: ⚔️ 早停触发，实际运行 {actual_epochs} 轮，进入最终 Test 阶段。")

    # ---------------- 训练结束，测试评估与开销统计 ----------------
    if not ctx.is_on_arbiter:
        # 回滚最优权重
        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)
            print(f"Rank {ctx.rank}: 已回滚至 Epoch {best_epoch} 的最佳权重。")
            
        print(f"Rank {ctx.rank}: 🚀 启动最终物理尺度 Test Set 评估...")
        model.eval()
        test_start_t = time.time()
        test_mae, test_rmse, test_mape = 0.0, 0.0, 0.0
        test_elements, valid_mape_count = 0, 0

        with torch.no_grad():
            for batch in test_loader:
                x, y = batch[0].to(args.device), batch[1].to(args.device)
                y_cov = batch[2].to(args.device) if len(batch) > 2 else None

                pred = model(x, y_cov=y_cov)

                if y.dim() == 4 and y.shape[-1] == 1: y = y.squeeze(-1)
                if pred.dim() == 4 and pred.shape[-1] == 1: pred = pred.squeeze(-1)
                if pred.shape != y.shape:
                    if pred.dim() == 3 and y.dim() == 3 and pred.shape[1] == y.shape[2] and pred.shape[2] == y.shape[1]:
                        pred = pred.transpose(1, 2)
                    else:
                        pred = pred.reshape_as(y)

                # 【核心】：仅在最终 Test Set 评估时，执行物理尺度反归一化！
                y_real = scaler.inverse_transform(y).cpu().numpy()
                pred_real = scaler.inverse_transform(pred).cpu().numpy()

                test_mae += np.sum(np.abs(y_real - pred_real))
                test_rmse += np.sum((y_real - pred_real) ** 2)
                test_elements += y_real.size

                mask = y_real > 0.5
                if np.sum(mask) > 0:
                    test_mape += np.sum(np.abs(y_real[mask] - pred_real[mask]) / y_real[mask])
                    valid_mape_count += np.sum(mask)

        # 结算物理真实指标
        acc_mae = test_mae / test_elements
        acc_rmse = math.sqrt(test_rmse / test_elements)
        acc_mape = (test_mape / valid_mape_count) * 100 if valid_mape_count > 0 else 0.0

        eff_test_time = time.time() - test_start_t
        eff_train_time = total_train_time
        eff_val_time = total_val_time / max(actual_epochs, 1)

        # 统一通信量基准：返回单次通信涉及的参数量 (FedTPS 只通信 Patterns)
        comm_params = model.Patterns.numel()
        
        # 动态捕捉 FLOPs 算力开销
        eff_flops = 0.0
        try:
            from thop import profile
            dummy_batch = next(iter(val_loader))
            dummy_x = dummy_batch[0].to(args.device)
            # 兼容 y_cov 评估
            dummy_ycov = dummy_batch[2].to(args.device) if len(dummy_batch) > 2 else None
            
            # 使用简单的 Wrapper 给 profile 传参
            class FLOPsWrapper(torch.nn.Module):
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, x, y_cov): return self.m(x, y_cov=y_cov)
            
            flops, _ = profile(FLOPsWrapper(model), inputs=(dummy_x, dummy_ycov), verbose=False)
            eff_flops = round(flops / 1e9, 4)
        except Exception as e:
            print(f"Rank {ctx.rank}: FLOPs 估算失败，回退为 0.0。原因: {e}")

        # 传出 10 个返回值，完美适配外层统一落盘逻辑
        return best_epoch, acc_mae, acc_rmse, acc_mape, eff_train_time, eff_val_time, eff_test_time, comm_params, actual_epochs, eff_flops
    else:
        return None
