import time
import math
import copy
import os
import signal
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _flatten_state_for_similarity(state_dict):
    tensors = []
    for value in state_dict.values():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            tensors.append(value.detach().float().cpu().reshape(-1))
    if not tensors:
        return torch.zeros(1)
    return torch.cat(tensors)


def _build_fpass_weights(flat_weights, metric, zeta, self_weight):
    num_clients = len(flat_weights)
    scores = torch.zeros((num_clients, num_clients), dtype=torch.float32)

    for i in range(num_clients):
        for j in range(num_clients):
            if metric == "cosine":
                sim = F.cosine_similarity(flat_weights[i].unsqueeze(0), flat_weights[j].unsqueeze(0), dim=1).item()
                distance = max(0.0, 1.0 - sim)
            elif metric == "sum_sq":
                distance = torch.sum((flat_weights[i] - flat_weights[j]) ** 2).item()
            else:
                distance = torch.mean((flat_weights[i] - flat_weights[j]) ** 2).item()
            scores[i, j] = -zeta * distance

    W_matrix = torch.softmax(scores, dim=1)
    if self_weight > 0:
        self_weight = min(max(self_weight, 0.0), 1.0)
        W_matrix = W_matrix * (1.0 - self_weight)
        W_matrix += torch.eye(num_clients, dtype=W_matrix.dtype) * self_weight

    if not torch.isfinite(W_matrix).all():
        W_matrix = torch.eye(num_clients, dtype=torch.float32)
    return W_matrix


def _aggregate_state_for_client(all_weights, client_idx, weights_row):
    agg_w = {}
    template = all_weights[client_idx]
    for key, value in template.items():
        if not torch.is_tensor(value) or not torch.is_floating_point(value):
            agg_w[key] = value.clone() if torch.is_tensor(value) else value
            continue
        mixed = torch.zeros_like(value, dtype=value.dtype)
        for src_idx, src_state in enumerate(all_weights):
            mixed = mixed + weights_row[src_idx].item() * src_state[key].to(dtype=value.dtype)
        agg_w[key] = mixed
    return agg_w


def _configure_twomgtcn_runtime(model, args, optimizer=None):
    topk = _env_int("TWOMGTCN_TOPK", 32)
    if hasattr(getattr(model, "gcn", None), "set_sparse_topk"):
        model.gcn.set_sparse_topk(topk)
        nnz = model.gcn.A_sparse._nnz() if model.gcn.A_sparse is not None else model.gcn.A.numel()
        print(
            f"[2MGTCN runtime] sparse_topk={topk} adj_shape={tuple(model.gcn.A.shape)} nnz={nnz}",
            flush=True,
        )

    if os.environ.get("TWOMGTCN_FAST_TEMPORAL", "1") != "0" and hasattr(model, "enable_fast_temporal"):
        if getattr(model, "fast_temporal", None) is None:
            model.enable_fast_temporal()
            print("[2MGTCN runtime] fast_temporal=enabled", flush=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    return optimizer


def _twomgtcn_to_device(tensor, args):
    use_cuda = isinstance(args.device, str) and args.device.startswith("cuda")
    return tensor.to(args.device, non_blocking=use_cuda)
from lib.utils import ExplicitEarlyStopper, EarlyStopSignal # 去掉了 evaluate_client_model，我们自己写

def compute_lfac(features):
    """
    计算特征分解损失 L_Fac (绝对防爆版)
    """
    B_N, D = features.shape[0] * features.shape[1], features.shape[2]
    r = features.reshape(B_N, D)
    
    r_mean = r.mean(dim=0, keepdim=True)
    
    # 👉 核心修复：手动计算方差，先加上 1e-6，再开根号！
    # 这样无论是前向还是反向求导，根号里面永远大于 0，绝对不会触发底层的 NaN Bug
    r_var = r.var(dim=0, keepdim=True, unbiased=False)
    r_std = torch.sqrt(r_var + 1e-6)
    
    r_norm = (r - r_mean) / r_std
    
    C = torch.matmul(r_norm.T, r_norm) / (B_N - 1)
    I = torch.eye(D, device=features.device)
    
    L_fac = 0.5 * torch.norm(C - I, p='fro') ** 2
    return L_fac

def train_twomgtcn_task(ctx, args, get_setting):
    """
    2MGTCN 专属的联邦训练与 FPASS 聚合引擎
    """
    if ctx.is_on_arbiter:
        print(f"[2MGTCN Server] 启动 FPASS (基于空间相似度) 聚合中心...")
        zeta = _env_float("TWOMGTCN_ZETA", 1.0)
        self_weight = _env_float("TWOMGTCN_SELF_WEIGHT", 0.6)
        warmup_epochs = _env_int("TWOMGTCN_WARMUP_EPOCHS", 5)
        metric = os.environ.get("TWOMGTCN_FPASS_METRIC", "cosine").strip().lower()
        log_every = _env_int("TWOMGTCN_FPASS_LOG_EVERY", 10)
        print(
            f"[2MGTCN Server] FPASS metric={metric} zeta={zeta} "
            f"self_weight={self_weight} warmup_epochs={warmup_epochs}",
            flush=True,
        )
        
        for epoch in range(args.epochs):
            w_guest = ctx.guest.get(f"w_{epoch}")
            w_hosts = ctx.hosts.get(f"w_{epoch}")
            if not isinstance(w_hosts, list): w_hosts = [w_hosts]
            all_weights = [w_guest] + w_hosts
            num_clients = len(all_weights)
            
            if epoch < warmup_epochs:
                W_matrix = torch.eye(num_clients, dtype=torch.float32)
            else:
                flat_weights = [_flatten_state_for_similarity(w) for w in all_weights]
                W_matrix = _build_fpass_weights(flat_weights, metric, zeta, self_weight)

            if log_every > 0 and (epoch < 3 or (epoch + 1) % log_every == 0):
                print(f"[2MGTCN Server] epoch={epoch} W={W_matrix.numpy().round(4).tolist()}", flush=True)

            personalized_weights = [
                _aggregate_state_for_client(all_weights, i, W_matrix[i])
                for i in range(num_clients)
            ]
            
            ctx.guest.put(f"agg_w_{epoch}", personalized_weights[0])
            ctx.hosts.put(f"agg_w_{epoch}", personalized_weights[1:] if len(personalized_weights) > 2 else personalized_weights[1])
            
        return None

    else:
        # ================== Client 本地训练 ==================
        print(f"[2MGTCN Client {ctx.rank}] 启动多模态训练...")
        print(f"[2MGTCN Client {ctx.rank}] stage=before_get_setting", flush=True)
        train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = get_setting(ctx)
        print(
            f"[2MGTCN Client {ctx.rank}] stage=after_get_setting "
            f"train_len={len(train_set)} val_len={len(val_set)} test_len={len(test_set)}",
            flush=True,
        )
        optimizer = _configure_twomgtcn_runtime(model, args, optimizer)
        lfac_alpha = _env_float("TWOMGTCN_LFAC_ALPHA", 0.001)
        print(f"[2MGTCN Client {ctx.rank}] lfac_alpha={lfac_alpha}", flush=True)
        
        for dset in [train_set, val_set, test_set]:
            if hasattr(dset, 'data') and isinstance(dset.data, np.ndarray):
                dset.data = torch.from_numpy(dset.data)

        pin_memory = isinstance(args.device, str) and args.device.startswith("cuda")
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=pin_memory)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)
        print(
            f"[2MGTCN Client {ctx.rank}] stage=after_dataloader "
            f"train_steps={len(train_loader)} val_steps={len(val_loader)} test_steps={len(test_loader)} "
            f"pin_memory={pin_memory}",
            flush=True,
        )
        
        best_mae, best_rmse, best_mape, best_epoch = float('inf'), float('inf'), float('inf'), -1
        best_model_wts = None
        
        total_train_time, total_val_time = 0.0, 0.0
        actual_epochs = 0
        stopper = ExplicitEarlyStopper(ctx, args, patience=50, min_delta=1e-4)

        try:
            for epoch in range(args.epochs):
                print(f"[2MGTCN Client {ctx.rank}] stage=before_epoch epoch={epoch}", flush=True)
                model.train()
                epoch_loss = 0.0
                train_start = time.time()
                
                for batch in train_loader:
                    x_c, x_p, x_t, x_ext, y = batch
                    x_c = _twomgtcn_to_device(x_c, args)
                    x_ext = _twomgtcn_to_device(x_ext, args)
                    y = _twomgtcn_to_device(y, args)

                    optimizer.zero_grad()
                    pred, fused_feat = model(x_c, x_ext)
                    
                    if y.dim() == 4 and y.shape[-1] == 1: y = y.squeeze(-1)
                    if pred.dim() == 4 and pred.shape[-1] == 1: pred = pred.squeeze(-1)
                    if pred.shape != y.shape:
                        pred = pred.reshape_as(y)
                    
                    base_loss = loss_func(pred, y)
                    l_fac = compute_lfac(fused_feat)
                    
                    loss = base_loss + lfac_alpha * l_fac

                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"[警告] Client {ctx.rank} 在 Epoch {epoch} 发现 NaN Loss！已紧急跳过此 Batch 的参数更新。")
                        optimizer.zero_grad() # 清空脏数据
                        continue # 跳过这一轮，不执行 backward
                    
                    loss.backward()

                    has_bad_grad = False
                    for param in model.parameters():
                        if param.grad is not None and not torch.isfinite(param.grad).all():
                            has_bad_grad = True
                            break
                    
                    if has_bad_grad:
                        print(f"[安检拦截] Client {ctx.rank} Epoch {epoch} 捕获到异常梯度(NaN/Inf)！已丢弃该 Batch 防止污染模型。")
                        optimizer.zero_grad()
                        continue # 跳过这一轮参数更
                        
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    
                total_train_time += (time.time() - train_start)
                
                # ==== 与 Server 通信 (FPASS) ====
                local_w = {k: v.cpu().clone().detach() for k, v in model.state_dict().items()}
                ctx.arbiter.put(f"w_{epoch}", local_w)
                
                agg_w_cpu = ctx.arbiter.get(f"agg_w_{epoch}")
                if isinstance(agg_w_cpu, list): 
                    agg_w_cpu = agg_w_cpu[ctx.rank - 1] if len(agg_w_cpu) > 1 else agg_w_cpu[0]
                model.load_state_dict({k: v.to(args.device) for k, v in agg_w_cpu.items()})
                
                # ========================================================
                # 🛡️ 本地验证 (Validation)：输出完整的 eval 指标 🛡️
                # ========================================================
                model.eval()
                val_start = time.time()
                val_mae, val_mse, val_rmse, val_mape = 0.0, 0.0, 0.0, 0.0
                total_val_elements = 0
                valid_val_mape_count = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        x_c, x_p, x_t, x_ext, y_val = batch
                        x_c = _twomgtcn_to_device(x_c, args)
                        x_ext = _twomgtcn_to_device(x_ext, args)
                        y_val = _twomgtcn_to_device(y_val, args)
                        
                        pred_val, _ = model(x_c, x_ext)
                        if y_val.dim() == 4 and y_val.shape[-1] == 1: y_val = y_val.squeeze(-1)
                        if pred_val.dim() == 4 and pred_val.shape[-1] == 1: pred_val = pred_val.squeeze(-1)
                        if pred_val.shape != y_val.shape: pred_val = pred_val.reshape_as(y_val)
                        
                        # 👉 1. 在归一化数据上算 MAE 和 MSE (速度快，用于早停判定)
                        y_cpu, pred_cpu = y_val.cpu().numpy(), pred_val.cpu().numpy()
                        val_mae += np.sum(np.abs(y_cpu - pred_cpu))
                        val_mse += np.sum((y_cpu - pred_cpu) ** 2)
                        total_val_elements += y_val.numel()

                        # 👉 2. 单独为 MAPE 进行反归一化 (保证分母有物理意义)
                        y_real = scaler.inverse_transform(y_val).cpu().numpy()
                        pred_real = scaler.inverse_transform(pred_val).cpu().numpy()
                        
                        # 过滤掉极小值(比如流量<0.5辆车)，防止除零错乱
                        mask = y_real > 0.5
                        if np.sum(mask) > 0:
                            val_mape += np.sum(np.abs(y_real[mask] - pred_real[mask]) / y_real[mask])
                            valid_val_mape_count += np.sum(mask)

                # 结算 4 大指标
                norm_val_mae = float(val_mae / total_val_elements)
                norm_val_mse = float(val_mse / total_val_elements)
                norm_val_rmse = math.sqrt(norm_val_mse)
                real_val_mape = float(val_mape / valid_val_mape_count) if valid_val_mape_count > 0 else 0.0
                
                total_val_time += (time.time() - val_start)
                actual_epochs = epoch + 1
                
             
                print(f"   ---> [验证结果] Client {ctx.rank} Epoch {epoch} | eval_mae: {norm_val_mae:.4f} | eval_rmse: {norm_val_rmse:.4f} | eval_mse: {norm_val_mse:.4f} | eval_mape: {real_val_mape:.4f}")

                if norm_val_mae < best_mae:
                    best_mae, best_rmse, best_epoch = norm_val_mae, norm_val_rmse, epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                
                if stopper.check_and_sync(norm_val_mae):
                    print(f"Rank {ctx.rank}: 🛑 收到全局早停信号，抛出中断异常！")
                    raise EarlyStopSignal("触发早停机制")
                
        except EarlyStopSignal:
            print(f"Rank {ctx.rank}: 🎉 成功穿透黑盒！跳出训练循环，实际运行 {actual_epochs} 轮。")

        # ========================================================
        # 🚀 最终测试 (Test)：强制执行反归一化 (Inverse Transform) 🚀
        # ========================================================
        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)
            
        print(f"Rank {ctx.rank}: 🚀 开始最终测试集评估 (执行反归一化还原真实物理量纲)...")
        model.eval()
        test_start = time.time()
        
        test_mae, test_rmse, test_mape = 0.0, 0.0, 0.0
        test_elements, valid_test_mape_count = 0, 0
        
        with torch.no_grad():
            for batch in test_loader:
                x_c, x_p, x_t, x_ext, y_test = batch
                x_c = _twomgtcn_to_device(x_c, args)
                x_ext = _twomgtcn_to_device(x_ext, args)
                y_test = _twomgtcn_to_device(y_test, args)
                
                pred_test, _ = model(x_c, x_ext)
                if y_test.dim() == 4 and y_test.shape[-1] == 1: y_test = y_test.squeeze(-1)
                if pred_test.dim() == 4 and pred_test.shape[-1] == 1: pred_test = pred_test.squeeze(-1)
                if pred_test.shape != y_test.shape: pred_test = pred_test.reshape_as(y_test)
                
                # 👉 核心：只在这里调用 scaler.inverse_transform()！
                y_real = scaler.inverse_transform(y_test).cpu().numpy()
                pred_real = scaler.inverse_transform(pred_test).cpu().numpy()
                
                # 计算物理量纲下的真实误差
                test_mae += np.sum(np.abs(y_real - pred_real))
                test_rmse += np.sum((y_real - pred_real) ** 2)
                test_elements += y_real.size
                
                # 过滤极小值，计算真实 MAPE
                mask = y_real > 0.5
                if np.sum(mask) > 0:
                    test_mape += np.sum(np.abs(y_real[mask] - pred_real[mask]) / y_real[mask])
                    valid_test_mape_count += np.sum(mask)

        # 结算最终的物理指标 (写进 CSV 的数据)
        acc_mae = test_mae / test_elements
        acc_rmse = math.sqrt(test_rmse / test_elements)
        acc_mape = (test_mape / valid_test_mape_count) if valid_test_mape_count > 0 else 0.0
        
        eff_test_time = round(time.time() - test_start, 4)
        comm_params = sum(p.numel() for p in model.parameters())
        
        return best_epoch + 1, acc_mae, acc_rmse, acc_mape, total_train_time, (total_val_time / max(actual_epochs, 1)), eff_test_time, comm_params, actual_epochs
