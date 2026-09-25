import time
import copy
import math
import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.FGNNEH import FGNNEH_Server
# ⚠️ 这里补上了上一版不小心漏掉的 evaluate_client_model
from lib.utils import ExplicitEarlyStopper, EarlyStopSignal, evaluate_client_model
from privacy.protection import l2_norm
from privacy.runtime_protection import (
    protected_arbiter_put, record_he_ttp_downlink, unprotect_he_ttp_payload,
)
from privacy.attack_trace import capture_he_ttp_insider_upper_bound


def _partial_hypernode_view(hypernode, args):
    """Fixed-prefix leakage for the explicitly labelled partial-view study."""
    ratio = float(os.environ.get("FGNNEH_HYPERNODE_LEAK_RATIO", "1.0"))
    if not 0.0 < ratio <= 1.0:
        raise ValueError("FGNNEH_HYPERNODE_LEAK_RATIO must be in (0, 1].")
    count = max(1, min(hypernode.shape[-1], int(round(hypernode.shape[-1] * ratio))))
    return hypernode[..., :count]


def _quantize_public_prediction(prediction):
    """Fixed public quantizer for the explicitly labelled side-channel study."""
    bits = int(os.environ.get("FGNNEH_QUANT_PRED_BITS", "4"))
    clip = float(os.environ.get("FGNNEH_QUANT_PRED_CLIP", "3.0"))
    if bits < 2 or clip <= 0:
        raise ValueError("FGNNEH_QUANT_PRED_BITS must be >=2 and FGNNEH_QUANT_PRED_CLIP must be >0.")
    levels = float((1 << (bits - 1)) - 1)
    return (prediction.clamp(-clip, clip) * (levels / clip)).round().mul(clip / levels)

def extract_ctx_data(ctx, data_list):
    """FATE 通信列表解包工具"""
    if isinstance(data_list, list):
        if len(data_list) == 1:
            return data_list[0]
        else:
            my_idx = ctx.rank - 1 
            if 0 <= my_idx < len(data_list):
                return data_list[my_idx]
            return data_list[0] 
    return data_list

def train_fgnneh_task(ctx, args, get_setting_fn):
    if ctx.is_on_arbiter:
        # 加上 flush=True，强制立刻把日志写到文件，绝不憋在缓冲区
        print(f"🚀 [FGNNEH Server Rank {ctx.rank}] Starting Strict Algorithm 2 Evolution...", flush=True)
        server_model = FGNNEH_Server(args.num_clients, args.hidden_dim).to(args.device)
        optimizer_server = torch.optim.Adam(server_model.parameters(), lr=args.lr)
        
        hyper_adj = torch.eye(args.num_clients, device=args.device)
        if args.num_clients > 1:
            for i in range(args.num_clients):
                next_idx = (i + 1) % args.num_clients
                hyper_adj[i, next_idx] = 0.5
                hyper_adj[next_idx, i] = 0.5
            
        prev_perfs = [float('inf')] * args.num_clients
        delta_prime = getattr(args, 'fgnneh_delta', 0.5)  
        alpha = getattr(args, 'fgnneh_alpha', 0.1)  
        beta = getattr(args, 'fgnneh_beta', 0.05)    
        gamma = 1e-5                                
        epsilon_weight = 0.5                        
        dp_hyper_clip = None
        dp_ctx_grad_clip = None
        
    else:
        print(f"🚀 [FGNNEH Client {ctx.rank}] Starting with End-to-End Gradients...", flush=True)
        train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = get_setting_fn(ctx)
        
        for dset in [train_set, val_set, test_set]:
            if hasattr(dset, 'data') and isinstance(dset.data, np.ndarray):
                dset.data = torch.from_numpy(dset.data).to(args.device)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        
        steps = len(train_loader)
        print(f"✅ [Client {ctx.rank}] Data Loaders ready, sending steps ({steps}) to Server...", flush=True)
        ctx.arbiter.put("steps", steps)
        
        best_norm_mae, best_epoch = float('inf'), -1
        best_model_wts = None
        total_train_time, total_val_time = 0.0, 0.0
        stopper = ExplicitEarlyStopper(ctx, args, patience=50, min_delta=1e-4)
        dp_hyper_clip = None
        dp_ctx_grad_clip = None

    # ================= 步数同步 =================
    if ctx.is_on_arbiter:
        print(f"⏳ [Server] Waiting for 'steps' from Guest and Hosts...", flush=True)
        s_guest = ctx.guest.get("steps")
        s_hosts = ctx.hosts.get("steps")
        if not isinstance(s_hosts, list): s_hosts = [s_hosts]
        STEPS = min([s_guest] + s_hosts)
        print(f"✅ [Server] Synced STEPS: {STEPS}. Entering training loop...", flush=True)
    else:
        STEPS = steps 

    actual_epochs = 0

    try:
        for epoch in range(args.epochs):
            
            if not ctx.is_on_arbiter: # ==== Client 逻辑 ====
                model.train()
                epoch_loss = 0.0
                train_start_t = time.time()
                
                for i, (x, y) in enumerate(train_loader):
                    if i >= STEPS: break
                    
                    # 探针：只在第一轮的第一个 Batch 打印，看是不是卡在 PyTorch 前向传播
                    if i == 0 and epoch == 0:
                        print(f"   ---> [Client {ctx.rank}] Entered Batch 0. Starting Forward Local...", flush=True)

                    x, y = x.to(args.device), y.to(args.device)
                    optimizer.zero_grad()
                    tag = f"e{epoch}_s{i}"
                    
                    h, h_backbone = model.forward_local(x, model.backbone_extractor.adj)
                    hyper_node_vec = model.generate_hypernode(h_backbone) 
                    hyper_node_vec.retain_grad()
                    
                    if i == 0 and epoch == 0:
                        print(f"   ---> [Client {ctx.rank}] Forward done. Sending hypernode to Server...", flush=True)
                        
                    hyper_upload = hyper_node_vec.detach().cpu()
                    if args.protection == "dp":
                        ctx.arbiter.put(f"fgnneh_dp_hyper_norm_{tag}", float(l2_norm(hyper_upload).item()) if dp_hyper_clip is None else None)
                        if dp_hyper_clip is None:
                            dp_hyper_clip = ctx.arbiter.get(f"fgnneh_dp_hyper_clip_{tag}")
                            if isinstance(dp_hyper_clip, (list, tuple)): dp_hyper_clip = dp_hyper_clip[0]
                            dp_hyper_clip = float(dp_hyper_clip)
                        protected_arbiter_put(ctx, args, f"hyper_{tag}", hyper_upload, clip_norm=dp_hyper_clip)
                    elif args.protection == "he":
                        # The quantized-prediction study intentionally does
                        # not save the hypernode. Its only target-dependent
                        # observation is captured after prediction below.
                        if not bool(os.environ.get("PRIVACY_TRACE_FGNNEH_QUANTIZED_PREDICTION", "")):
                            observed_hypernode = _partial_hypernode_view(hyper_upload, args)
                            leak_ratio = float(os.environ.get("FGNNEH_HYPERNODE_LEAK_RATIO", "1.0"))
                            capture_he_ttp_insider_upper_bound(
                                ctx, args, f"fgnneh_hyper_{tag}",
                                observed_leak=observed_hypernode,
                                model_state_dict=model.state_dict(),
                                leak_type="activation",
                                threat_model=(
                                    "HE-TTP trusted-Arbiter partial-hypernode upper bound "
                                    f"(deterministic disclosed ratio={leak_ratio:.6g})"
                                ),
                            )
                        protected_arbiter_put(ctx, args, f"hyper_{tag}", hyper_upload)
                    else:
                        ctx.arbiter.put(f"hyper_{tag}", hyper_upload)
                    
                    if i == 0 and epoch == 0:
                        print(f"   ---> [Client {ctx.rank}] Waiting for global context from Server...", flush=True)
                        
                    context_data = extract_ctx_data(ctx, ctx.arbiter.get(f"ctx_{tag}"))
                    context_data = record_he_ttp_downlink(args, context_data)
                    context_vec = context_data.to(args.device).requires_grad_()
                    
                    pred = model.forward_predict(h, context_vec)
                    if y.dim() == 4 and y.shape[-1] == 1: y = y.squeeze(-1)
                    if pred.dim() == 4 and pred.shape[-1] == 1: pred = pred.squeeze(-1)

                    if args.protection == "he" and bool(os.environ.get("PRIVACY_TRACE_FGNNEH_QUANTIZED_PREDICTION", "")):
                        capture_he_ttp_insider_upper_bound(
                            ctx, args, f"fgnneh_quantized_prediction_{tag}",
                            observed_leak=(
                                _quantize_public_prediction(pred.detach().cpu()),
                                context_data.detach().cpu(),
                            ),
                            model_state_dict=model.state_dict(),
                            leak_type="activation",
                            threat_model=(
                                "Server-plus-public-model quantized-prediction side-channel: "
                                "server observes the returned context and a fixed-point client prediction; "
                                "no hypernode plaintext is exposed."
                            ),
                        )
                    
                    loss = loss_func(pred, y)
                    loss.backward(retain_graph=True) 
                    
                    ctx_grad_upload = context_vec.grad.detach().cpu()
                    if args.protection == "dp":
                        ctx.arbiter.put(f"fgnneh_dp_ctx_grad_norm_{tag}", float(l2_norm(ctx_grad_upload).item()) if dp_ctx_grad_clip is None else None)
                        if dp_ctx_grad_clip is None:
                            dp_ctx_grad_clip = ctx.arbiter.get(f"fgnneh_dp_ctx_grad_clip_{tag}")
                            if isinstance(dp_ctx_grad_clip, (list, tuple)): dp_ctx_grad_clip = dp_ctx_grad_clip[0]
                            dp_ctx_grad_clip = float(dp_ctx_grad_clip)
                        protected_arbiter_put(ctx, args, f"g_ctx_{tag}", ctx_grad_upload, clip_norm=dp_ctx_grad_clip)
                    elif args.protection == "he":
                        protected_arbiter_put(ctx, args, f"g_ctx_{tag}", ctx_grad_upload)
                    else:
                        ctx.arbiter.put(f"g_ctx_{tag}", ctx_grad_upload)
                    g_hyper_data = extract_ctx_data(ctx, ctx.arbiter.get(f"g_hyper_{tag}"))
                    g_hyper_data = record_he_ttp_downlink(args, g_hyper_data)
                    g_hyper = g_hyper_data.to(args.device)
                    hyper_node_vec.backward(g_hyper) 
                    
                    optimizer.step()
                    epoch_loss += loss.item()
                    
                    if i == 0 and epoch == 0:
                        print(f"   ---> [Client {ctx.rank}] Batch 0 completed successfully! The pipeline is unblocked.", flush=True)
                
                total_train_time += (time.time() - train_start_t)
                
                # ----------------- 2. 验证阶段 (归一化数据) -----------------
                model.eval()
                val_start_t = time.time()
                val_mae, val_rmse, val_mape = 0.0, 0.0, 0.0  # 新增 val_mape
                val_elements, valid_val_mape_count = 0, 0    # 新增 valid_val_mape_count
                
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        x_val, y_val = x_val.to(args.device), y_val.to(args.device)
                        h_val, h_backbone_val = model.forward_local(x_val, model.backbone_extractor.adj)
                        eval_ctx = model.generate_hypernode(h_backbone_val)
                        pred_val = model.forward_predict(h_val, eval_ctx)
                        
                        if y_val.dim() == 4 and y_val.shape[-1] == 1: y_val = y_val.squeeze(-1)
                        if pred_val.dim() == 4 and pred_val.shape[-1] == 1: pred_val = pred_val.squeeze(-1)
                        if pred_val.shape != y_val.shape:
                            pred_val = pred_val.reshape_as(y_val)
                        
                        y_cpu = y_val.cpu().numpy()
                        pred_cpu = pred_val.cpu().numpy()
                        
                        val_mae += np.sum(np.abs(y_cpu - pred_cpu))
                        val_rmse += np.sum((y_cpu - pred_cpu) ** 2)
                        val_elements += y_cpu.size
                        
                        # 【新增】：计算归一化尺度下的 MAPE
                        # 加上绝对值和微小阈值防除零（因为 std 归一化后会有很多接近0的数据）
                        mask = np.abs(y_cpu) > 1e-3
                        if np.sum(mask) > 0:
                            val_mape += np.sum(np.abs(y_cpu[mask] - pred_cpu[mask]) / np.abs(y_cpu[mask]))
                            valid_val_mape_count += np.sum(mask)
                
                norm_val_mae = val_mae / val_elements
                norm_val_rmse = math.sqrt(val_rmse / val_elements)
                # 【新增】：计算百分比 MAPE
                norm_val_mape = (val_mape / valid_val_mape_count * 100) if valid_val_mape_count > 0 else 0.0
                
                total_val_time += (time.time() - val_start_t)
                
                # 【修改】：在 print 中加上 Norm Val MAPE 的打印
                print(f"[Client {ctx.rank}] Epoch {epoch} | Train Loss: {epoch_loss/max(1, STEPS):.4f} | Norm Val MAE: {norm_val_mae:.4f} | Norm Val RMSE: {norm_val_rmse:.4f} | Norm Val MAPE: {norm_val_mape:.2f}%", flush=True)
                
                if norm_val_mae < best_norm_mae:
                    best_norm_mae = norm_val_mae
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())


                ctx.arbiter.put(f"perf_{epoch}", float(norm_val_mae))

                should_stop = stopper.check_and_sync(norm_val_mae)
                if should_stop:
                    print(f"Rank {ctx.rank}: 🛑 收到全局早停信号，抛出中断异常！", flush=True)
                    raise EarlyStopSignal("触发早停")

            else: # ==== Server 逻辑 ====
                for step in range(STEPS):
                    tag = f"e{epoch}_s{step}"
                    
                    if step == 0 and epoch == 0:
                        print(f"   ---> [Server] Waiting for Client hypernodes (Batch 0)...", flush=True)
                        
                    if args.protection == "dp" and dp_hyper_clip is None:
                        hyper_norm_guest = ctx.guest.get(f"fgnneh_dp_hyper_norm_{tag}")
                        hyper_norm_hosts = ctx.hosts.get(f"fgnneh_dp_hyper_norm_{tag}")
                        hyper_norm_hosts = hyper_norm_hosts if isinstance(hyper_norm_hosts, list) else [hyper_norm_hosts]
                        dp_hyper_clip = float(np.quantile([float(v) for v in [hyper_norm_guest] + hyper_norm_hosts], 0.9))
                        ctx.guest.put(f"fgnneh_dp_hyper_clip_{tag}", dp_hyper_clip)
                        ctx.hosts.put(f"fgnneh_dp_hyper_clip_{tag}", [dp_hyper_clip] * len(hyper_norm_hosts))
                        print(f"[DPCalibration] FGNNEH target=hypernode clip_norm={dp_hyper_clip:.8f}", flush=True)
                    hyper_guest = unprotect_he_ttp_payload(args, ctx.guest.get(f"hyper_{tag}"))
                    hyper_hosts = ctx.hosts.get(f"hyper_{tag}")
                    hyper_hosts = hyper_hosts if isinstance(hyper_hosts, list) else [hyper_hosts]
                    hyper_hosts = [unprotect_he_ttp_payload(args, value) for value in hyper_hosts]
                    
                    if step == 0 and epoch == 0:
                        print(f"   ---> [Server] Received hypernodes! Computing global context...", flush=True)
                        
                    all_embs = [hyper_guest] + hyper_hosts
                    all_embs = [e.cpu() if isinstance(e, torch.Tensor) else torch.tensor(e) for e in all_embs]
                    hyper_embs_tensor = torch.stack(all_embs).to(args.device).requires_grad_()
                    
                    optimizer_server.zero_grad()
                    global_context = server_model(hyper_embs_tensor, hyper_adj) 
                    
                    ctx.guest.put(f"ctx_{tag}", global_context[0].detach().cpu())
                    host_ctxs = [c.detach().cpu() for c in global_context[1:]]
                    ctx.hosts.put(f"ctx_{tag}", host_ctxs)
                    
                    if args.protection == "dp" and dp_ctx_grad_clip is None:
                        grad_norm_guest = ctx.guest.get(f"fgnneh_dp_ctx_grad_norm_{tag}")
                        grad_norm_hosts = ctx.hosts.get(f"fgnneh_dp_ctx_grad_norm_{tag}")
                        grad_norm_hosts = grad_norm_hosts if isinstance(grad_norm_hosts, list) else [grad_norm_hosts]
                        dp_ctx_grad_clip = float(np.quantile([float(v) for v in [grad_norm_guest] + grad_norm_hosts], 0.9))
                        ctx.guest.put(f"fgnneh_dp_ctx_grad_clip_{tag}", dp_ctx_grad_clip)
                        ctx.hosts.put(f"fgnneh_dp_ctx_grad_clip_{tag}", [dp_ctx_grad_clip] * len(grad_norm_hosts))
                        print(f"[DPCalibration] FGNNEH target=context_gradient clip_norm={dp_ctx_grad_clip:.8f}", flush=True)
                    g_ctx_guest = unprotect_he_ttp_payload(args, ctx.guest.get(f"g_ctx_{tag}"))
                    g_ctx_hosts = ctx.hosts.get(f"g_ctx_{tag}")
                    g_ctx_hosts = g_ctx_hosts if isinstance(g_ctx_hosts, list) else [g_ctx_hosts]
                    g_ctx_hosts = [unprotect_he_ttp_payload(args, value) for value in g_ctx_hosts]
                    all_g_ctx = [g_ctx_guest] + g_ctx_hosts
                    g_global_context = torch.stack([g.to(args.device) for g in all_g_ctx])
                    
                    global_context.backward(g_global_context)
                    optimizer_server.step()
                    
                    g_hyper = hyper_embs_tensor.grad.detach().cpu()
                    ctx.guest.put(f"g_hyper_{tag}", g_hyper[0])
                    ctx.hosts.put(f"g_hyper_{tag}", [g for g in g_hyper[1:]] if len(g_hyper) > 2 else g_hyper[1])
                
                # Algorithm 2 超图演化
                p_guest = ctx.guest.get(f"perf_{epoch}")
                p_hosts = ctx.hosts.get(f"perf_{epoch}")
                curr_perfs = [p_guest] + p_hosts if isinstance(p_hosts, list) else [p_guest, p_hosts]
                
                if epoch > 0:
                    sorted_indices = np.argsort(curr_perfs) 
                    mid = len(sorted_indices) // 2
                    G_good = sorted_indices[:mid]
                    G_poor = sorted_indices[mid:]
                    
                    for i in range(args.num_clients):
                        delta_p = prev_perfs[i] - curr_perfs[i] 
                        for j in range(args.num_clients):
                            if i == j: continue
                            if hyper_adj[i, j] > 0:
                                if delta_p > delta_prime:
                                    hyper_adj[i, j] += alpha * delta_p
                                else:
                                    hyper_adj[i, j] -= beta * (abs(delta_p) + gamma)
                                
                                if hyper_adj[i, j] < epsilon_weight:
                                    hyper_adj[i, j] = 0.0
                    
                    for poor_node in G_poor:
                        unconnected_good = [g for g in G_good if hyper_adj[poor_node, g] == 0]
                        if unconnected_good:
                            target = np.random.choice(unconnected_good)
                            hyper_adj[poor_node, target] = 0.5
                            hyper_adj[target, poor_node] = 0.5
                    
                    hyper_adj = torch.clamp(hyper_adj, 0.0, 1.0)
                    hyper_adj = (hyper_adj + hyper_adj.T) / 2.0 
                    hyper_adj.fill_diagonal_(1.0)
                    print(f"Server Epoch {epoch}: Topology Updated. Current G_poor: {G_poor}", flush=True)
                
                prev_perfs = curr_perfs

            if not ctx.is_on_arbiter:
                actual_epochs = epoch + 1

    except EarlyStopSignal:
        if not ctx.is_on_arbiter:
            actual_epochs = epoch + 1
            print(f"Rank {ctx.rank}: 🎉 成功穿透死循环 (实际运行 {actual_epochs} 轮)！准备回滚权重...", flush=True)

    if not ctx.is_on_arbiter:
        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)
            
        print(f"Rank {ctx.rank}: 🚀 开始测试集评估 (反归一化)...", flush=True)
        model.eval()
        test_start_t = time.time()
        test_mae, test_rmse, test_mape = 0.0, 0.0, 0.0
        test_elements, valid_mape_count = 0, 0
        
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test, y_test = x_test.to(args.device), y_test.to(args.device)
                
                h_val, h_backbone_test = model.forward_local(x_test, model.backbone_extractor.adj)
                eval_ctx = model.generate_hypernode(h_backbone_test)
                pred_test = model.forward_predict(h_val, eval_ctx)
                
                if y_test.dim() == 4 and y_test.shape[-1] == 1: y_test = y_test.squeeze(-1)
                if pred_test.dim() == 4 and pred_test.shape[-1] == 1: pred_test = pred_test.squeeze(-1)
                if pred_test.shape != y_test.shape:
                    pred_test = pred_test.reshape_as(y_test)
                
                y_real = scaler.inverse_transform(y_test).cpu().numpy()
                pred_real = scaler.inverse_transform(pred_test).cpu().numpy()
                
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
        
        eff_test_time = time.time() - test_start_t
        eff_val_time = total_val_time / max(actual_epochs, 1) 
        
        eff_flops = 0.0
        try:
            from thop import profile
            class FGNNEH_Wrapper(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def forward(self, x_in):
                    h_val, h_backbone_val = self.m.forward_local(x_in, self.m.backbone_extractor.adj)
                    eval_ctx = self.m.generate_hypernode(h_backbone_val)
                    return self.m.forward_predict(h_val, eval_ctx)

            wrapper = FGNNEH_Wrapper(model).to(args.device)
            dummy_x, _ = next(iter(val_loader))
            flops, _ = profile(wrapper, inputs=(dummy_x.to(args.device),), verbose=False)
            eff_flops = round(flops / 1e9, 4)
        except Exception:
            pass
            
        comm_params_per_batch = 4 * args.hidden_dim
        total_comm_params = comm_params_per_batch * STEPS * actual_epochs

        return best_epoch, acc_mae, acc_rmse, acc_mape, total_train_time, eff_val_time, eff_test_time, actual_epochs, eff_flops, total_comm_params
