import math
import time
import numpy as np
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

# 引入早停与异常工具
from lib.utils import ExplicitEarlyStopper, EarlyStopSignal

def _unwrap_ctx_payload(payload: Any) -> Any:
    if isinstance(payload, list):
        return payload[0]
    return payload

def _align_pred_target(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if target.dim() == 4 and target.shape[-1] == 1:
        target = target.squeeze(-1)
    if pred.dim() == 4 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)
    if (pred.shape != target.shape and pred.dim() == 3 and target.dim() == 3 and pred.shape[1] == target.shape[2]):
        pred = pred.transpose(1, 2)
    return pred, target

def _slice_dataset_by_ratio(dataset, ratio: float):
    total = len(dataset)
    keep = max(1, math.ceil(total * ratio))
    if keep >= total:
        return dataset, total
    if hasattr(dataset, "tensors"):
        tensors = tuple(t[:keep] for t in dataset.tensors)
        return TensorDataset(*tensors), keep
    indices = list(range(keep))
    return Subset(dataset, indices), keep

def _clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}

def _average_state_dicts(state_dicts):
    avg_state = {}
    num_clients = len(state_dicts)
    for key in state_dicts[0].keys():
        avg_tensor = state_dicts[0][key].clone()
        for i in range(1, num_clients):
            avg_tensor += state_dicts[i][key]
        avg_state[key] = avg_tensor / num_clients
    return avg_state

def _adaptive_interpolate_shared(local_module, global_state: Dict[str, torch.Tensor]):
    with torch.no_grad():
        local_state = local_module.state_dict()
        updated_state = {}
        for name, local_param in local_state.items():
            if name not in global_state:
                updated_state[name] = local_param
                continue
            global_param = global_state[name].to(local_param.device, dtype=local_param.dtype)
            if local_param.numel() == 1:
                sim = torch.tensor(1.0, device=local_param.device, dtype=local_param.dtype)
            else:
                sim = F.cosine_similarity(local_param.reshape(1, -1), global_param.reshape(1, -1), dim=1).mean()
                sim = sim.clamp(0.0, 1.0)
            updated_state[name] = local_param + sim * (global_param - local_param)
        local_module.load_state_dict(updated_state, strict=False)

def train_pfedctp(ctx, args, get_setting_fn):
    print(f"Rank {ctx.rank}: [pFedCTP] 启动严谨联邦评估模式...")

    target_rank = None
    is_target_client = False
    
    actual_stage1_epochs = 0 
    actual_stage2_epochs = 0
    total_val_time = 0.0 # 【新增】全程记录验证耗时

    if not ctx.is_on_arbiter:
        train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = get_setting_fn(ctx)

        local_sample_count = len(train_set)
        ctx.arbiter.put("pfedctp_sample_count", int(local_sample_count))

        target_rank_payload = ctx.arbiter.get("pfedctp_target_rank")
        target_rank = int(_unwrap_ctx_payload(target_rank_payload))
        is_target_client = (ctx.rank == target_rank)

        if is_target_client:
            scarce_ratio = float(getattr(args, "target_train_ratio", 0.05))
            train_set, kept_samples = _slice_dataset_by_ratio(train_set, scarce_ratio)
            print(f"Rank {ctx.rank}: [pFedCTP] 靶向客户端触发数据稀缺限制 ({kept_samples}/{local_sample_count})")

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False) 

        stage1_ratio = float(getattr(args, "pfedctp_stage1_batch_ratio", 0.2))
        stage1_steps = max(1, math.ceil(len(train_loader) * stage1_ratio))
        stage1_steps = min(stage1_steps, len(train_loader))
        ctx.arbiter.put("pfedctp_stage1_steps", int(stage1_steps))

        best_norm_mae, best_epoch = float("inf"), -1
        best_model_wts = None
        start_time = time.time()

        init_shared_weights = _clone_state_dict(model.shareModel.state_dict())
        ctx.arbiter.put("pfedctp_bootstrap_shared", init_shared_weights)
        
        stopper = ExplicitEarlyStopper(ctx, args, patience=50, min_delta=1e-4)

    else:
        sample_guest = int(ctx.guest.get("pfedctp_sample_count"))
        sample_hosts = ctx.hosts.get("pfedctp_sample_count")
        if not isinstance(sample_hosts, list): sample_hosts = [sample_hosts]
        all_counts = [sample_guest] + [int(v) for v in sample_hosts]
        
        target_rank = int(min(range(len(all_counts)), key=lambda i: (all_counts[i], i)))
        ctx.guest.put("pfedctp_target_rank", target_rank)
        ctx.hosts.put("pfedctp_target_rank", [target_rank] * len(sample_hosts))

        init_guest = ctx.guest.get("pfedctp_bootstrap_shared")
        init_hosts = ctx.hosts.get("pfedctp_bootstrap_shared")
        if not isinstance(init_hosts, list): init_hosts = [init_hosts]
        init_all = [init_guest] + init_hosts
        global_shared_0 = _average_state_dicts(init_all)

        ctx.guest.put("pfedctp_global_shared_0", global_shared_0)
        ctx.hosts.put("pfedctp_global_shared_0", [global_shared_0] * len(init_hosts))

    # =========================================================
    # Stage 1: 联邦共享阶段
    # =========================================================
    last_global_shared = None 

    try:
        for epoch in range(args.epochs):
            if not ctx.is_on_arbiter:
                global_shared_data = ctx.arbiter.get(f"pfedctp_global_shared_{epoch}")
                global_shared_data = _unwrap_ctx_payload(global_shared_data)
                last_global_shared = global_shared_data 

                _adaptive_interpolate_shared(model.shareModel, global_shared_data)

                # 训练
                model.train()
                total_loss = 0.0
                actual_steps = 0
                for step, (x, y) in enumerate(train_loader):
                    if step >= stage1_steps: break
                    x, y = x.to(args.device), y.to(args.device)
                    optimizer.zero_grad()
                    pred = model(x)
                    pred, y = _align_pred_target(pred, y)
                    loss = loss_func(pred, y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    actual_steps += 1

                avg_loss_norm = total_loss / max(1, actual_steps)
                
                # 验证：精准记录耗时
                model.eval()
                val_start_t = time.time() # 【新增】计时开始
                val_mae_norm, val_elements = 0.0, 0
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        x_val, y_val = x_val.to(args.device), y_val.to(args.device)
                        pred_val = model(x_val)
                        pred_val, y_val = _align_pred_target(pred_val, y_val)
                        val_mae_norm += torch.abs(pred_val - y_val).sum().item()
                        val_elements += y_val.numel()
                
                total_val_time += (time.time() - val_start_t) # 【新增】累加耗时
                avg_val_mae_norm = val_mae_norm / max(1, val_elements)
                
                print(f"Client {ctx.rank} Stage-1 Epoch {epoch} | Train Loss(Norm): {avg_loss_norm:.4f} | Val MAE(Norm): {avg_val_mae_norm:.4f}")

                if avg_val_mae_norm < best_norm_mae:
                    best_norm_mae = avg_val_mae_norm
                    best_epoch = epoch
                    best_model_wts = _clone_state_dict(model.state_dict())

                if stopper.check_and_sync(float(avg_val_mae_norm)):
                    print(f"Rank {ctx.rank}: 🛑 收到全局早停信号，在第 {epoch} 轮斩断循环！")
                    raise EarlyStopSignal("Stage-1 联邦早停")

                local_shared_weights = _clone_state_dict(model.shareModel.state_dict())
                ctx.arbiter.put(f"pfedctp_shared_{epoch}", local_shared_weights)
                actual_stage1_epochs = epoch + 1

            else:
                w_guest = ctx.guest.get(f"pfedctp_shared_{epoch}")
                w_hosts = ctx.hosts.get(f"pfedctp_shared_{epoch}")
                if not isinstance(w_hosts, list): w_hosts = [w_hosts]
                all_weights = [w_guest] + w_hosts
                global_shared_weights = _average_state_dicts(all_weights)
                ctx.guest.put(f"pfedctp_global_shared_{epoch + 1}", global_shared_weights)
                ctx.hosts.put(f"pfedctp_global_shared_{epoch + 1}", [global_shared_weights] * len(w_hosts))

    except EarlyStopSignal:
        if not ctx.is_on_arbiter:
            print(f"Rank {ctx.rank}: 🎉 平稳跳出 Stage-1 黑盒，准备微调！")

    # =========================================================
    # Stage 2: 靶向微调阶段 (Target Client)
    # =========================================================
    if not ctx.is_on_arbiter:
        run_finetune = is_target_client or bool(getattr(args, "finetune_all", False))

        if run_finetune:
            if last_global_shared is not None:
                _adaptive_interpolate_shared(model.shareModel, last_global_shared)

            print(f"Client {ctx.rank}: 进入 Stage-2 单兵微调阶段.")
            
            local_patience = 20
            local_no_improve = 0

            for ft_epoch in range(args.target_epochs):
                model.train()
                for x, y in train_loader:
                    x, y = x.to(args.device), y.to(args.device)
                    optimizer.zero_grad()
                    pred = model(x)
                    pred, y = _align_pred_target(pred, y)
                    loss = loss_func(pred, y)
                    loss.backward()
                    optimizer.step()

                # 微调期验证：同样精准记录耗时
                model.eval()
                val_start_t = time.time() # 【新增】计时开始
                val_mae_norm, val_elements = 0.0, 0
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        x_val, y_val = x_val.to(args.device), y_val.to(args.device)
                        pred_val = model(x_val)
                        pred_val, y_val = _align_pred_target(pred_val, y_val)
                        val_mae_norm += torch.abs(pred_val - y_val).sum().item()
                        val_elements += y_val.numel()
                
                total_val_time += (time.time() - val_start_t) # 【新增】累加耗时
                avg_val_mae_norm = val_mae_norm / max(1, val_elements)
                
                print(f"Client {ctx.rank}: [微调] Epoch {ft_epoch} | Val MAE(Norm): {avg_val_mae_norm:.4f}")
                
                if avg_val_mae_norm < best_norm_mae:
                    best_norm_mae = avg_val_mae_norm
                    best_epoch = actual_stage1_epochs + ft_epoch
                    best_model_wts = _clone_state_dict(model.state_dict())
                    local_no_improve = 0
                else:
                    local_no_improve += 1
                    
                actual_stage2_epochs = ft_epoch + 1
                if local_no_improve >= local_patience:
                    print(f"Client {ctx.rank}: 🛑 Stage 2 微调触发本地早停。")
                    break
        else:
            print(f"Client {ctx.rank}: 非目标客户端(target_rank={target_rank})，跳过微调。")


        # =========================================================
        # Stage 3: 最终极验收测试 (反归一化物理尺度)
        # =========================================================
        print(f"Rank {ctx.rank}: 🚀 启动基于 Test Set 的终极物理评估...")
        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)
        
        model.eval()
        test_mae, test_rmse, test_mape = 0.0, 0.0, 0.0
        test_elements, valid_mape_count = 0, 0
        test_start_t = time.time()
        
        with torch.no_grad():
            for x_t, y_t in test_loader:
                x_t, y_t = x_t.to(args.device), y_t.to(args.device)
                pred_t = model(x_t)
                pred_t, y_t = _align_pred_target(pred_t, y_t)

                y_real = scaler.inverse_transform(y_t).cpu().numpy()
                pred_real = scaler.inverse_transform(pred_t).cpu().numpy()
                
                test_mae += np.sum(np.abs(y_real - pred_real))
                test_rmse += np.sum((y_real - pred_real) ** 2)
                test_elements += y_real.size
                
                mask = y_real > 0.5
                if np.sum(mask) > 0:
                    test_mape += np.sum(np.abs(y_real[mask] - pred_real[mask]) / y_real[mask])
                    valid_mape_count += np.sum(mask)

        final_acc_mae = test_mae / max(1, test_elements)
        final_acc_rmse = math.sqrt(test_rmse / max(1, test_elements))
        final_acc_mape = (test_mape / valid_mape_count) * 100 if valid_mape_count > 0 else 0.0
        
        eff_test_time = time.time() - test_start_t
        total_train_time = time.time() - start_time - eff_test_time - total_val_time # 精确扣除测试和验证时间
        
        # 【新增】计算平均单轮验证耗时
        total_actual_epochs = max(1, actual_stage1_epochs + actual_stage2_epochs)
        eff_val_time = total_val_time / total_actual_epochs
        
        comm_params_per_round = sum(p.numel() for p in model.shareModel.parameters() if p.requires_grad)
        
        eff_flops = 0.0
        try:
            from thop import profile
            dummy_x = next(iter(val_loader))[0].to(args.device)
            flops, _ = profile(model, inputs=(dummy_x,), verbose=False)
            eff_flops = round(flops / 1e9, 4)
        except Exception:
            pass

        # 【修改】将 eff_val_time 加入返回值，严格对齐 fate_main 解包
        return (
            best_epoch,
            final_acc_mae,
            final_acc_rmse,
            final_acc_mape,
            total_train_time,
            eff_val_time,    # <--- 这里是你要的验证耗时
            eff_test_time,
            comm_params_per_round,
            actual_stage1_epochs,
            total_actual_epochs,
            eff_flops,
            run_finetune,
            int(target_rank),
        )
