import copy
import time
import os
import signal
import csv
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


# =========================================================
# 基础工具与通信
# =========================================================

def _as_list(x):
    if isinstance(x, list):
        return x
    return [x]

def _collect_all_from_clients(ctx, tag: str) -> List[Any]:
    guest_obj = ctx.guest.get(tag)
    host_objs = ctx.hosts.get(tag)
    host_objs = _as_list(host_objs)
    return [guest_obj] + host_objs

def _broadcast_same_to_all_clients(ctx, tag: str, value: Any, num_clients: int):
    ctx.guest.put(tag, value)
    num_hosts = max(0, num_clients - 1)
    if num_hosts == 0:
        return
    elif num_hosts == 1:
        ctx.hosts.put(tag, value)
    else:
        ctx.hosts.put(tag, [copy.deepcopy(value) for _ in range(num_hosts)])

def _safe_get(ctx, tag: str):
    val = ctx.arbiter.get(tag)
    if isinstance(val, list):
        my_idx = ctx.rank - 1
        if 0 <= my_idx < len(val):
            return val[my_idx]
        return val[0]
    return val

def _broadcast_ranked_to_all_clients(ctx, tag: str, ranked_values: List[Any]):
    if len(ranked_values) == 0:
        raise ValueError("ranked_values is empty")
    ctx.guest.put(tag, ranked_values[0])
    host_values = ranked_values[1:]
    if len(host_values) == 0:
        return
    elif len(host_values) == 1:
        ctx.hosts.put(tag, host_values[0])
    else:
        ctx.hosts.put(tag, host_values)


# =========================================================
# 模型权重与状态管理
# =========================================================

def _clone_model_state_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in model.state_dict().items():
        if k == "edge_index":
            continue
        out[k] = v.detach().cpu().clone()
    return out

def _load_model_state_keep_local_graph(model: nn.Module, state: Dict[str, torch.Tensor]):
    current = model.state_dict()
    patched = {}
    for k, v in current.items():
        if k in state:
            src = state[k]
            if isinstance(v, torch.Tensor) and isinstance(src, torch.Tensor) and v.shape == src.shape:
                patched[k] = src.to(v.device, dtype=v.dtype)
            else:
                patched[k] = v
        else:
            patched[k] = v
    model.load_state_dict(patched, strict=False)

def _average_state_dicts(weighted_states: List[Tuple[Dict[str, torch.Tensor], float]]) -> Dict[str, torch.Tensor]:
    if len(weighted_states) == 0:
        raise ValueError("No weighted states to average.")
    total_weight = float(sum(w for _, w in weighted_states))
    if total_weight <= 0:
        raise ValueError(f"Invalid total weight: {total_weight}")

    base_state = copy.deepcopy(weighted_states[0][0])
    for k in base_state.keys():
        acc = None
        for state, w in weighted_states:
            val = state[k].float()
            scaled = val * (float(w) / total_weight)
            acc = scaled if acc is None else (acc + scaled)
        base_state[k] = acc.to(dtype=weighted_states[0][0][k].dtype)
    return base_state


# =========================================================
# 数据与评估逻辑
# =========================================================

def _prepare_loss_fn(loss_func, device):
    if isinstance(loss_func, nn.Module):
        return loss_func.to(device)
    return loss_func

def _extract_xy_from_batch(batch, device):
    if not isinstance(batch, (list, tuple)):
        raise ValueError(f"Unsupported batch type: {type(batch)}")
    if len(batch) < 2:
        raise ValueError(f"Batch should have at least 2 items, got len={len(batch)}")
    x = batch[0]
    y = batch[-1]
    if isinstance(x, torch.Tensor):
        x = x.to(device)
    if isinstance(y, torch.Tensor):
        y = y.to(device)
    return x, y

def _align_target_to_pred(y: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    if y.shape == pred.shape:
        return y
    if y.dim() == 4 and pred.dim() == 4:
        if y.shape[0] == pred.shape[0] and y.shape[-1] == pred.shape[-1]:
            if y.shape[1] == pred.shape[2] and y.shape[2] == pred.shape[1]:
                return y.permute(0, 2, 1, 3).contiguous()
    if y.dim() == 3 and pred.dim() == 3:
        if y.shape[0] == pred.shape[0] and y.shape[1] == pred.shape[2] and y.shape[2] == pred.shape[1]:
            return y.permute(0, 2, 1).contiguous()
    raise ValueError(f"Cannot align target shape {tuple(y.shape)} to pred shape {tuple(pred.shape)}")

def _inverse_transform_np(arr: np.ndarray, scaler) -> np.ndarray:
    if hasattr(scaler, "inverse_transform"):
        return scaler.inverse_transform(arr)
    if hasattr(scaler, "std") and hasattr(scaler, "mean"):
        return arr * scaler.std + scaler.mean
    return arr

def _evaluate_regression(model: nn.Module, loader: DataLoader, scaler, device: str, inverse: bool = False) -> Dict[str, float]:
    model.eval()
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for batch in loader:
            x, y = _extract_xy_from_batch(batch, device)
            pred = model(x)
            y = _align_target_to_pred(y, pred)
            y_true_list.append(y.detach().cpu().numpy())
            y_pred_list.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    if inverse and scaler is not None:
        y_true = _inverse_transform_np(y_true, scaler)
        y_pred = _inverse_transform_np(y_pred, scaler)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))

    mape = 0.0
    if inverse:
        mask = y_true > 0.5
        if np.sum(mask) > 0:
            mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

    return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape}


# =========================================================
# 训练 Epoch 辅助函数
# =========================================================

def _train_supervised_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_func, device: str
) -> float:
    model.train()
    loss_meter = []
    for batch in loader:
        x, y = _extract_xy_from_batch(batch, device)
        optimizer.zero_grad()
        pred = model(x)
        y = _align_target_to_pred(y, pred)
        loss = loss_func(pred, y)
        loss.backward()
        optimizer.step()
        loss_meter.append(loss.item())
    return float(np.mean(loss_meter)) if len(loss_meter) > 0 else 0.0

def _train_target_adapt_epoch(
    target_model: nn.Module, source_model: nn.Module, loader: DataLoader,
    optimizer: torch.optim.Optimizer, loss_func, device: str, lambda_mmd: float,
) -> float:
    target_model.train()
    source_model.eval()
    loss_meter = []
    for batch in loader:
        x, y = _extract_xy_from_batch(batch, device)
        optimizer.zero_grad()
        with torch.no_grad():
            _, z_s = source_model(x, return_feature=True)

        pred_t, z_t = target_model(x, return_feature=True)
        y = _align_target_to_pred(y, pred_t)

        loss_sup = loss_func(pred_t, y)
        
        # 避免 OOM，沿着 Time 维度进行均值池化
        z_s_pool = z_s.mean(dim=2).reshape(-1, z_s.size(-1))
        z_t_pool = z_t.mean(dim=2).reshape(-1, z_t.size(-1))
        
        loss_mmd = _mmd_rbf_loss(z_s_pool, z_t_pool)
        loss = loss_sup + lambda_mmd * loss_mmd
        loss.backward()
        optimizer.step()
        loss_meter.append(loss.item())
    return float(np.mean(loss_meter)) if len(loss_meter) > 0 else 0.0

def _build_target_sparse_subset(dataset, ratio: float, seed: int = 0):
    n_total = len(dataset)
    n_keep = max(1, int(round(n_total * float(ratio))))
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_total)[:n_keep]
    indices = sorted(indices.tolist())
    return Subset(dataset, indices)

def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    dist2 = torch.cdist(x, y, p=2) ** 2
    return torch.exp(-gamma * dist2)

def _mmd_rbf_loss(xs: torch.Tensor, xt: torch.Tensor, gamma: Optional[float] = None) -> torch.Tensor:
    xs = xs.reshape(xs.size(0), -1)
    xt = xt.reshape(xt.size(0), -1)
    if gamma is None:
        with torch.no_grad():
            all_feat = torch.cat([xs, xt], dim=0)
            pairwise = torch.cdist(all_feat, all_feat, p=2) ** 2
            gamma = 1.0 / pairwise.mean().clamp(min=1e-6)
    k_xx = _rbf_kernel(xs, xs, gamma).mean()
    k_tt = _rbf_kernel(xt, xt, gamma).mean()
    k_xt = _rbf_kernel(xs, xt, gamma).mean()
    return k_xx + k_tt - 2.0 * k_xt

def _select_target_client_by_min_nodes(ctx, args) -> int:
    tag_local_nodes = "tistgnn_ic_local_num_nodes"
    tag_target_id = "tistgnn_ic_target_client_id"

    if not ctx.is_on_arbiter:
        local_num_nodes = len(args.nodes_per[ctx.rank])
        ctx.arbiter.put(tag_local_nodes, int(local_num_nodes))
        target_id = _safe_get(ctx, tag_target_id)
        return int(target_id)

    all_num_nodes = _collect_all_from_clients(ctx, tag_local_nodes)
    all_num_nodes = [int(x) for x in all_num_nodes]
    target_id = int(min(range(len(all_num_nodes)), key=lambda i: all_num_nodes[i]))

    _broadcast_same_to_all_clients(ctx, tag_target_id, target_id, args.num_clients)
    print(f"[Arbiter][T-ISTGNN(i-c)] Client node counts = {all_num_nodes}, target_client_id = {target_id}")
    return target_id


# =========================================================
# 训练主流程
# =========================================================

def train_tistgnn_ic(ctx, args, setting=None):
    target_init_epochs = int(getattr(args, "target_init_epochs", 10))
    transfer_epochs = int(getattr(args, "transfer_epochs", 50))
    patience_limit = 50
    lambda_mmd = float(getattr(args, "lambda_mmd", 0.1))
    freeze_predictor = bool(getattr(args, "freeze_predictor", True))
    target_train_ratio = float(getattr(args, "target_train_ratio", 0.05))

    target_client_id = _select_target_client_by_min_nodes(ctx, args)

    # =====================================================
    # Arbiter
    # =====================================================
    if ctx.is_on_arbiter:
        # --- Stage A ---
        init_state = _collect_all_from_clients(ctx, "tistgnn_ic_target_init_state")
        init_state = [x for x in init_state if x is not None][0]
        _broadcast_same_to_all_clients(ctx, "tistgnn_ic_init_global_state", init_state, args.num_clients)

        # --- Stage B ---
        best_global_mae = float("inf")
        patience_counter = 0
        actual_stageB_rounds = args.epochs
        global_state = init_state

        for rnd in range(args.epochs):
            _broadcast_same_to_all_clients(ctx, f"tistgnn_ic_global_state_{rnd}", global_state, args.num_clients)
            payloads = _collect_all_from_clients(ctx, f"tistgnn_ic_client_payload_{rnd}")
            payloads = [p for p in payloads if p is not None]

            weighted_states = [(p["state"], float(p["weight"])) for p in payloads]
            global_state = _average_state_dicts(weighted_states)

            total_weight = sum(p["weight"] for p in payloads)
            avg_val_mae = sum(p["val_mae_norm"] * p["weight"] for p in payloads) / total_weight

            should_stop = False
            if avg_val_mae < best_global_mae - 1e-4:
                best_global_mae, patience_counter = avg_val_mae, 0
            else:
                patience_counter += 1
                if patience_counter >= patience_limit: should_stop = True

            # 记录统一格式到轨迹 CSV
            guest_loss_str = f'{payloads[0]["val_mae_norm"]:.4f}'
            host_losses_str = ";".join([f'{p["val_mae_norm"]:.4f}' for p in payloads[1:]])
            task_prefix = f"{args.model}_{args.dataset_name}_{args.feature_type}"
            
            current_row = [
                task_prefix, rnd + 1, guest_loss_str, host_losses_str, 
                f'{avg_val_mae:.4f}', f'{best_global_mae:.4f}', patience_counter
            ]

            csv_path = "lib/early_stop_losses.csv"
            file_exists = os.path.isfile(csv_path)
            try:
                with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["Model_Info", "Epoch", "Guest_Loss", "Host_Losses_List", "Global_Avg_Loss", "Best_Loss", "Patience_Counter"])
                    writer.writerow(current_row)
            except Exception as e:
                print(f"[Arbiter][Stage B] ⚠️ 实时追加 CSV 失败: {e}")

            _broadcast_same_to_all_clients(ctx, f"tistgnn_ic_stop_info_{rnd}", {"stop": should_stop, "patience": patience_counter}, args.num_clients)
            
            print(f"[Arbiter][Stage B] Epoch {rnd + 1}/{args.epochs} aggregated. Avg MAE(Norm): {avg_val_mae:.4f} | Patience: {patience_counter}/{patience_limit}")
            
            if should_stop or rnd == args.epochs - 1:
                actual_stageB_rounds = rnd + 1
                if should_stop:
                    print(f"[Arbiter][Stage B] 🛑 触发全局早停，实际运行 {actual_stageB_rounds} 轮！")
                break

        _broadcast_same_to_all_clients(ctx, "tistgnn_ic_source_global_state", global_state, args.num_clients)
        _broadcast_same_to_all_clients(ctx, "tistgnn_ic_actual_stageB_rounds", actual_stageB_rounds, args.num_clients)
        
        final_metrics = _collect_all_from_clients(ctx, "tistgnn_ic_final_metrics")
        return [m for m in final_metrics if m is not None][0]

    # =====================================================
    # Client
    # =====================================================
    else:
        train_set, val_set, test_set, model, _, loss_func, _, _, _, scaler, _ = setting
        is_target = (ctx.rank == target_client_id)
        
        # [核心修复 1]：全局秒表，记录最真实的挂机耗时
        client_start_time = time.time()
        total_val_time = 0.0
        val_calls = 0

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

        target_sparse_train_set = None
        target_sparse_loader = None

        # --- Stage A ---
        if is_target:
            sparse_set = _build_target_sparse_subset(train_set, ratio=target_train_ratio, seed=args.seed)
            sparse_loader = DataLoader(sparse_set, batch_size=args.batch_size, shuffle=True)
            m_a = model.to(args.device)
            opt_a = torch.optim.Adam(m_a.parameters(), lr=args.lr)
            best_state_a = None
            best_mae_a = float('inf')
            
            for ep in range(target_init_epochs):
                train_loss = _train_supervised_epoch(m_a, sparse_loader, opt_a, loss_func.to(args.device), args.device)
                
                # 记录验证耗时
                val_start = time.time()
                metrics = _evaluate_regression(m_a, val_loader, scaler, args.device, inverse=False)
                total_val_time += (time.time() - val_start)
                val_calls += 1

                if metrics['mae'] < best_mae_a:
                    best_mae_a, best_state_a = metrics['mae'], _clone_model_state_cpu(m_a)
                print(f"[Client {ctx.rank}][Stage A][Epoch {ep + 1}] train_loss(norm)={train_loss:.4f}, val_mae(norm)={metrics['mae']:.4f}")
            ctx.arbiter.put("tistgnn_ic_target_init_state", best_state_a)
        else:
            ctx.arbiter.put("tistgnn_ic_target_init_state", None)

        init_global = _safe_get(ctx, "tistgnn_ic_init_global_state")

        # --- Stage B ---
        actual_stageB_rounds = args.epochs
        if not is_target:
            m_b = model.to(args.device)
            for rnd in range(args.epochs):
                _load_model_state_keep_local_graph(m_b, _safe_get(ctx, f"tistgnn_ic_global_state_{rnd}"))
                opt_b = torch.optim.Adam(m_b.parameters(), lr=args.lr)
                
                local_losses = []
                for _ in range(args.local_epochs):
                    local_losses.append(_train_supervised_epoch(m_b, train_loader, opt_b, loss_func.to(args.device), args.device))
                
                # 记录验证耗时
                val_start = time.time()
                v_met = _evaluate_regression(m_b, val_loader, scaler, args.device, inverse=False)
                total_val_time += (time.time() - val_start)
                val_calls += 1

                ctx.arbiter.put(f"tistgnn_ic_client_payload_{rnd}", {"state": _clone_model_state_cpu(m_b), "weight": len(train_set), "val_mae_norm": v_met["mae"]})
                
                stop_info = _safe_get(ctx, f"tistgnn_ic_stop_info_{rnd}")
                print(f"[Client {ctx.rank}][Stage B][Epoch {rnd + 1}] local_loss(norm)={np.mean(local_losses):.4f}, val_mae(norm)={v_met['mae']:.4f} | Patience: {stop_info['patience']}/{patience_limit}")
                
                if stop_info["stop"]: 
                    actual_stageB_rounds = rnd + 1
                    break
        else:
            for rnd in range(args.epochs):
                _safe_get(ctx, f"tistgnn_ic_global_state_{rnd}")
                ctx.arbiter.put(f"tistgnn_ic_client_payload_{rnd}", None)
                if _safe_get(ctx, f"tistgnn_ic_stop_info_{rnd}")["stop"]: 
                    actual_stageB_rounds = rnd + 1
                    break

        source_global = _safe_get(ctx, "tistgnn_ic_source_global_state")
        actual_stageB_rounds = _safe_get(ctx, "tistgnn_ic_actual_stageB_rounds")

        # --- Stage C ---
        if is_target:
            m_src = copy.deepcopy(model).to(args.device)
            _load_model_state_keep_local_graph(m_src, source_global)
            m_src.eval()
            
            m_tgt = copy.deepcopy(model).to(args.device)
            _load_model_state_keep_local_graph(m_tgt, source_global)
            if freeze_predictor: m_tgt.freeze_predictor()
            opt_c = torch.optim.Adam(m_tgt.parameters(), lr=args.lr)
            
            best_state_c, best_mae_c, best_epoch_c = None, float('inf'), 0
            p_count_c, actual_stageC_epochs = 0, transfer_epochs

            for ep in range(transfer_epochs):
                train_loss = _train_target_adapt_epoch(m_tgt, m_src, sparse_loader, opt_c, loss_func.to(args.device), args.device, lambda_mmd)

                # 记录验证耗时
                val_start = time.time()
                v_m = _evaluate_regression(m_tgt, val_loader, scaler, args.device, inverse=False)
                total_val_time += (time.time() - val_start)
                val_calls += 1

                if v_m['mae'] < best_mae_c - 1e-4:
                    best_mae_c, best_epoch_c, best_state_c, p_count_c = v_m['mae'], ep + 1, _clone_model_state_cpu(m_tgt), 0
                else:
                    p_count_c += 1
                
                print(f"[Client {ctx.rank}][Stage C][Epoch {ep+1}] train_loss(norm)={train_loss:.4f}, val_mae(norm): {v_m['mae']:.4f} | Patience: {p_count_c}/{patience_limit}")
                if p_count_c >= patience_limit:
                    actual_stageC_epochs = ep + 1
                    print(f"[Client {ctx.rank}][Stage C] 🛑 触发本地早停，实际运行 {actual_stageC_epochs} 轮！")
                    break

            # ---------------------------------------------
            # [核心修复 2]：最真实的联邦总耗时计算
            # ---------------------------------------------
            # 用现在的总时间 - 第一秒打下的时间戳 - 纯验证集的时间 = 纯联邦训练与通信挂机总时间！
            total_train_time = time.time() - client_start_time - total_val_time

            _load_model_state_keep_local_graph(m_tgt, best_state_c)
            t_start = time.time()
            test_metrics = _evaluate_regression(m_tgt, test_loader, scaler, args.device, inverse=True)
            eff_test_time = time.time() - t_start

            # 计算通信与FLOPs
            comm_mb = (sum(p.numel() for p in m_tgt.parameters()) * 4 * 2 * actual_stageB_rounds) / (1024 * 1024)
            eff_flops = 0.0
            try:
                from thop import profile
                dx, _ = _extract_xy_from_batch(next(iter(val_loader)), args.device)
                f, _ = profile(m_tgt, inputs=(dx,), verbose=False)
                eff_flops = round(f / 1e9, 4)
            except: pass

            final = {
                "best_epoch": int(best_epoch_c), 
                "actual_rounds": int(actual_stageB_rounds), 
                "mae": test_metrics['mae'], "mse": test_metrics['mse'], "rmse": test_metrics['rmse'], "mape": test_metrics['mape'],
                
                "train_t": total_train_time, 
                "val_t": total_val_time / max(val_calls, 1), # 平均每次验证集调用的耗时
                "test_t": eff_test_time,
                
                "comm_mb": comm_mb, "flops": eff_flops
            }
            ctx.arbiter.put("tistgnn_ic_final_metrics", final)
            
            print(f"[Client {ctx.rank}](Target) 阶段 C 跑完啦！数据已发送给 Arbiter。")
            return final
            
        else:
            ctx.arbiter.put("tistgnn_ic_final_metrics", None)
            
            print(f"[Client {ctx.rank}](Source) 阶段 B 已完成，安静待机等待 Target 跑完...")
            return None