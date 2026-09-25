import time
import torch
import numpy as np
import math
import copy
from torch.utils.data import DataLoader
from lib.utils import ExplicitEarlyStopper, EarlyStopSignal
from privacy.protection import l2_norm
from privacy.runtime_protection import protected_arbiter_put
from privacy.attack_trace import (
    capture_he_sa_aggregate,
    capture_he_sa_arbiter_insider,
    capture_he_sa_collusion_residual,
    capture_he_sa_kminus2_hidden_term,
    capture_he_sa_kminus3_hidden_term,
    capture_he_sa_server_aggregate_hidden_term,
)


def _is_dp_zero_equivalence(args):
    """Whether this run must reproduce the historical Plain protocol exactly.

    A fixed, effectively inactive radius and sigma=0 are used only for the
    diagnostic equivalence run.  In that case the first FCFedGCN round has to
    aggregate complete client states, just as the original Plain trainer did;
    a delta protocol is not algebraically equivalent when worker processes
    happened to initialise their models differently.
    """
    return (
        str(getattr(args, 'protection', 'plain')).lower() == 'dp'
        and float(getattr(args, 'dp_sigma', 0.0)) == 0.0
        and float(getattr(args, 'dp_clip_norm', 0.0)) > 0.0
    )


def _public_federated_state(model):
    """State tensors that are safe and shape-compatible to synchronise once."""
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if 'adj' not in key and 'fca_features' not in key and value.is_floating_point()
    }

def train_fcfedgcn_task(ctx, args, get_setting_func):
    """
    FC-FedGCN 严格复现版联邦训练引擎
    特色：
    1. 严格分离 通信轮数(epochs) 与 本地迭代(local_epochs)
    2. 严格执行 Eq.13 节点加权聚合
    3. 全局早停机制 + 异常拦截跳出
    4. 纯归一化验证 + 最终反归一化物理指标测试
    """
    if ctx.is_on_arbiter:
        print("[FC-FedGCN Server] 正在启动全局参数聚合中心...")
        STEPS_PER_EPOCH = 0
    else:
        print(f"[FC-FedGCN Client {ctx.rank}] 正在启动本地训练客户端...")
        # 1. 加载本地数据与模型
        train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = get_setting_func(ctx)
        
        # 确保数据在设备上
        for dset in [train_set, val_set, test_set]:
            if hasattr(dset, 'data') and isinstance(dset.data, np.ndarray):
                dset.data = torch.from_numpy(dset.data).to(args.device)
            
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        
        STEPS_PER_EPOCH = len(train_loader)
        ctx.arbiter.put("steps", STEPS_PER_EPOCH)
        
        N_local = len(args.nodes_per[ctx.rank])
        ctx.arbiter.put("node_count", N_local)
        
        # 追踪变量
        best_norm_mae = float('inf')
        best_epoch = -1
        best_model_wts = None
        actual_epochs = 0
        
        # 耗时统计
        total_train_time = 0.0
        total_val_time = 0.0
        
        # 实例化上帝视角裁判 (早停)
        stopper = ExplicitEarlyStopper(ctx, args, patience=50, min_delta=1e-4)

    # ================= 1. 初始化同步阶段 =================
    # A positive --dp_clip_norm is already a fixed, globally agreed C.  Only
    # auto-C mode (0) should make the Arbiter wait for first-round norm
    # messages.  Initialising this to None unconditionally deadlocks fixed-C
    # equivalence runs: clients correctly skip the norm upload while the
    # Arbiter incorrectly waits for it.
    dp_zero_equivalence = _is_dp_zero_equivalence(args)
    dp_clip_norm = (
        float(args.dp_clip_norm)
        if args.protection == 'dp' and float(getattr(args, 'dp_clip_norm', 0.0)) > 0
        else None
    )
    he_public_context = None
    he_arbiter_context = None
    he_upload_bytes = 0
    he_download_bytes = 0

    if ctx.is_on_arbiter:
        s_guest = ctx.guest.get("steps")
        s_hosts = ctx.hosts.get("steps")
        if not isinstance(s_hosts, list): s_hosts = [s_hosts]
        STEPS_PER_EPOCH = min([s_guest] + s_hosts)
        
        v_guest = ctx.guest.get("node_count")
        v_hosts = ctx.hosts.get("node_count")
        if not isinstance(v_hosts, list): v_hosts = [v_hosts]
        all_nodes_count = [v_guest] + v_hosts
        total_nodes = sum(all_nodes_count)
        
        # 【严格复现】：计算 Eq.13 中的权重: |V_i| / sum(|V_i|)
        agg_weights = [n / total_nodes for n in all_nodes_count]
        print(f"[Server] 各客户端节点数: {all_nodes_count}, 聚合权重: {[round(w, 4) for w in agg_weights]}")

    # ================= 2. 联邦训练主循环 =================
    if args.protection == 'he':
        # Keep the DP/Plain import path independent of optional HE helpers.
        # This also permits DP jobs to run while a server has not yet received
        # the newest HE backend module.
        from privacy.ckks_backend import (
            ciphertext_bytes as ckks_ciphertext_bytes,
            decrypt_tree as ckks_decrypt_tree,
            encrypt_tree as ckks_encrypt_tree,
            export_public_context as ckks_export_public_context,
            generate_context as ckks_generate_context,
            homomorphic_weighted_sum_tree as ckks_homomorphic_weighted_sum_tree,
            import_context as ckks_import_context,
        )
        if str(getattr(args, 'he_backend', 'auto')).lower() not in ('auto', 'he_sa'):
            raise ValueError('FCFedGCN supports HE-SA only; use --he_backend he_sa.')
        if ctx.is_on_arbiter:
            he_arbiter_context = ckks_generate_context(
                int(args.he_ckks_poly_modulus_degree), int(args.he_ckks_scale_bits),
            )
            public_context_payload = ckks_export_public_context(he_arbiter_context)
            ctx.guest.put('__fcfedgcn_he_sa_ckks_context', public_context_payload)
            ctx.hosts.put('__fcfedgcn_he_sa_ckks_context', public_context_payload)
            print(f"[HESA] FCFedGCN arbiter generated packed CKKS context poly_degree={args.he_ckks_poly_modulus_degree}", flush=True)
        else:
            he_public_context = ckks_import_context(ctx.arbiter.get('__fcfedgcn_he_sa_ckks_context'))
            print(f"[HESA] FCFedGCN rank={ctx.rank} received packed CKKS public context", flush=True)

    # A delta is meaningful only relative to the same public model.  The
    # historical Plain loop happened to aggregate full states in round one,
    # which hides per-process initialisation differences.  For an actual DP
    # run, first establish a public shared reference without perturbation and
    # then protect only data-dependent deltas.  The sigma=0 diagnostic instead
    # intentionally skips this synchronisation and reproduces the historical
    # full-state first aggregation below.
    if args.protection == 'dp' and not dp_zero_equivalence:
        if ctx.is_on_arbiter:
            init_guest = ctx.guest.get('fcfedgcn_dp_public_init')
            init_hosts = ctx.hosts.get('fcfedgcn_dp_public_init')
            if not isinstance(init_hosts, list):
                init_hosts = [init_hosts]
            initial_states = [init_guest] + init_hosts
            public_initial_state = {}
            for key in init_guest:
                public_initial_state[key] = sum(
                    agg_weights[index] * initial_states[index][key]
                    for index in range(len(initial_states))
                )
            ctx.guest.put('fcfedgcn_dp_public_init', public_initial_state)
            ctx.hosts.put(
                'fcfedgcn_dp_public_init',
                [public_initial_state] * len(init_hosts),
            )
            print('[FCFedGCN-DP] Arbiter broadcast a one-time public initial model reference.', flush=True)
        else:
            ctx.arbiter.put('fcfedgcn_dp_public_init', _public_federated_state(model))
            public_initial_state = ctx.arbiter.get('fcfedgcn_dp_public_init')
            if isinstance(public_initial_state, list):
                public_initial_state = public_initial_state[0]
            model.load_state_dict(public_initial_state, strict=False)
            print(f'[FCFedGCN-DP] rank={ctx.rank} loaded one-time public initial model reference.', flush=True)

    try: 
        # 外层循环：联邦通信轮数 (Global Communication Rounds)
        for epoch in range(args.epochs):
            
            if not ctx.is_on_arbiter: 
                # ---------------- Client 训练 ----------------
                round_start_weights = _public_federated_state(model)
                model.train()
                epoch_train_loss = 0.0
                train_start = time.time()
                
                # 【严格复现】：内层循环为本地模型迭代次数 (Local Epochs)
                for le in range(args.local_epochs):
                    for i, (x, y) in enumerate(train_loader):
                        if i >= STEPS_PER_EPOCH: break
                        
                        optimizer.zero_grad()
                        x, y = x.to(args.device), y.to(args.device)
                        
                        pred = model(x)
                        
                        if y.dim() == 4 and y.shape[-1] == 1: y = y.squeeze(-1)
                        if pred.dim() == 4 and pred.shape[-1] == 1: pred = pred.squeeze(-1)
                        if pred.shape != y.shape:
                            pred = pred.reshape_as(y)
                            
                        loss = loss_func(pred, y)
                        loss.backward()
                        optimizer.step()
                        epoch_train_loss += loss.item()
                    
                total_train_time += (time.time() - train_start)
                # 打印平均到每个 step 的 Loss
                avg_step_loss = epoch_train_loss / max(STEPS_PER_EPOCH * args.local_epochs, 1)
                print(f"[Client {ctx.rank}] Round {epoch} | Local Train Loss: {avg_step_loss:.4f}")
                
                # ---------------- Client 验证 (使用归一化数据) ----------------
                model.eval()
                val_start = time.time()
                val_mae, val_rmse = 0.0, 0.0
                val_elements = 0
                
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        x_val, y_val = x_val.to(args.device), y_val.to(args.device)
                        pred_val = model(x_val)
                        
                        if y_val.dim() == 4 and y_val.shape[-1] == 1: y_val = y_val.squeeze(-1)
                        if pred_val.dim() == 4 and pred_val.shape[-1] == 1: pred_val = pred_val.squeeze(-1)
                        if pred_val.shape != y_val.shape: pred_val = pred_val.reshape_as(y_val)
                        
                        # 严格验证：直接用模型输出算误差，绝对不反归一化
                        y_cpu = y_val.cpu().numpy()
                        pred_cpu = pred_val.cpu().numpy()
                        val_mae += np.sum(np.abs(y_cpu - pred_cpu))
                        val_rmse += np.sum((y_cpu - pred_cpu)**2)
                        val_elements += y_cpu.size
                        
                norm_val_mae = val_mae / val_elements
                norm_val_rmse = math.sqrt(val_rmse / val_elements)
                total_val_time += (time.time() - val_start)
                
                print(f"   ---> [验证集] Client {ctx.rank} Round {epoch} | Norm MAE: {norm_val_mae:.4f} | Norm RMSE: {norm_val_rmse:.4f}")
                
                # 保存最佳模型
                if norm_val_mae < best_norm_mae:
                    best_norm_mae = norm_val_mae
                    best_epoch = epoch
                    best_model_wts = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
                actual_epochs = epoch + 1  # 记录实际跑的通信轮数
                
                # 早停判定 (异常拦截跳出)
                # Complete this round before stopping.  Exiting before the
                # upload leaves the Arbiter blocked on weights_{epoch}.
                should_stop = bool(stopper.check_and_sync(norm_val_mae))
                
                # ---------------- 参数上传与联邦聚合 ----------------
                # 【严格复现】：因为维度通过 Padding 统一了，所以所有 GCN 权重都可以上传。
                # 只需剔除掉不需要更新的静态缓冲 (adj, fca_features) 以节省带宽
                local_state = {k: v.cpu().clone() for k, v in model.state_dict().items() if 'adj' not in k and 'fca_features' not in k}
                local_delta = {key: local_state[key] - round_start_weights[key] for key in round_start_weights}
                # For the no-noise equivalence audit use the exact historical
                # Plain payload (complete state) in round one and every later
                # round.  Formal DP/HE runs protect updates relative to their
                # now-common public initial/global reference.
                local_weights = (
                    local_state
                    if (args.protection == 'plain' or dp_zero_equivalence)
                    else local_delta
                )

                if args.protection == 'dp' and float(args.dp_clip_norm) <= 0:
                    ctx.arbiter.put(f"fcfedgcn_dp_delta_norm_{epoch}", float(l2_norm(local_delta).item()))
                    calibrated_clip = ctx.arbiter.get(f"fcfedgcn_dp_clip_norm_{epoch}")
                    if isinstance(calibrated_clip, (list, tuple)):
                        calibrated_clip = calibrated_clip[0]
                    args.dp_clip_norm = float(calibrated_clip)
                    print(f"[DPCalibration] FCFedGCN rank={ctx.rank} epoch={epoch + 1} clip_norm={args.dp_clip_norm:.8f}", flush=True)

                if args.protection == 'he':
                    capture_he_sa_arbiter_insider(
                        ctx, args, f"fcfedgcn_weights_{epoch}",
                        payload=local_weights,
                        model_state_dict=round_start_weights,
                    )
                    capture_he_sa_collusion_residual(
                        ctx, args, f"fcfedgcn_weights_{epoch}",
                        payload=local_weights,
                        model_state_dict=round_start_weights,
                    )
                    capture_he_sa_kminus2_hidden_term(
                        ctx, args, f"fcfedgcn_weights_{epoch}",
                        payload=local_weights,
                        model_state_dict=round_start_weights,
                        # all_nodes_count exists only in the Arbiter process;
                        # the client has its own public node count already.
                        aggregation_weight=float(N_local),
                    )
                    capture_he_sa_kminus3_hidden_term(
                        ctx, args, f"fcfedgcn_weights_{epoch}",
                        payload=local_weights,
                        model_state_dict=round_start_weights,
                        aggregation_weight=float(N_local),
                    )
                    capture_he_sa_server_aggregate_hidden_term(
                        ctx, args, f"fcfedgcn_weights_{epoch}",
                        payload=local_weights,
                        model_state_dict=round_start_weights,
                        aggregation_weight=float(N_local),
                    )
                    he_started = time.perf_counter()
                    encrypted_weights = ckks_encrypt_tree(
                        local_weights, he_public_context,
                        slot_count=int(args.he_ckks_poly_modulus_degree) // 2,
                    )
                    upload_bytes = ckks_ciphertext_bytes(encrypted_weights)
                    he_upload_bytes += upload_bytes
                    total_train_time += time.perf_counter() - he_started
                    ctx.arbiter.put(f"weights_{epoch}", encrypted_weights)
                    ctx.arbiter.put(f"fcfedgcn_he_upload_bytes_{epoch}", upload_bytes)
                    print(f"[HESA] FCFedGCN rank={ctx.rank} epoch={epoch + 1} encrypted_upload_bytes={upload_bytes}", flush=True)
                else:
                    protection_started = time.perf_counter()
                    if epoch < 3 or (epoch + 1) % 50 == 0:
                        print(
                            f"[FCFedGCN-DP Sync] rank={ctx.rank} round={epoch + 1} "
                            "sending protected delta to Arbiter",
                            flush=True,
                        )
                    protected_arbiter_put(ctx, args, f"weights_{epoch}", local_weights)
                    if args.protection == 'dp':
                        total_train_time += time.perf_counter() - protection_started
                    if epoch < 3 or (epoch + 1) % 50 == 0:
                        print(
                            f"[FCFedGCN-DP Sync] rank={ctx.rank} round={epoch + 1} "
                            "upload complete; waiting for global aggregate",
                            flush=True,
                        )
                
                global_weights_data = ctx.arbiter.get(f"global_weights_{epoch}")
                if epoch < 3 or (epoch + 1) % 50 == 0:
                    print(
                        f"[FCFedGCN-DP Sync] rank={ctx.rank} round={epoch + 1} "
                        "received global aggregate",
                        flush=True,
                    )
                if isinstance(global_weights_data, list): global_weights_data = global_weights_data[0]
                if args.protection in ('dp', 'he') and not dp_zero_equivalence:
                    global_weights_data = {key: round_start_weights[key] + value.cpu() for key, value in global_weights_data.items()}
                if args.protection == 'he':
                    he_download_bytes += sum(value.numel() * value.element_size() for value in global_weights_data.values())
                    he_arbiter_seconds = ctx.arbiter.get(f"fcfedgcn_he_arbiter_seconds_{epoch}")
                    # FATE may wrap a broadcast scalar in one or more lists.
                    while isinstance(he_arbiter_seconds, (list, tuple)):
                        if not he_arbiter_seconds:
                            raise RuntimeError("FCFedGCN HE received an empty Arbiter timing payload")
                        he_arbiter_seconds = he_arbiter_seconds[0]
                    total_train_time += float(he_arbiter_seconds)
                model.load_state_dict(global_weights_data, strict=False)
                if ctx.is_on_guest:
                    ctx.arbiter.put(f"fcfedgcn_stop_{epoch}", should_stop)
                if should_stop:
                    print(f"[FCFedGCN] rank={ctx.rank} received global early-stop at round {epoch + 1}", flush=True)
                    raise EarlyStopSignal("global early stop")
                    
            else:
                # ---------------- Server 逻辑 ----------------
                if args.protection == 'dp' and dp_clip_norm is None:
                    norm_guest = float(ctx.guest.get(f"fcfedgcn_dp_delta_norm_{epoch}"))
                    norm_hosts = ctx.hosts.get(f"fcfedgcn_dp_delta_norm_{epoch}")
                    norm_hosts = norm_hosts if isinstance(norm_hosts, list) else [norm_hosts]
                    dp_clip_norm = float(np.quantile([norm_guest] + [float(value) for value in norm_hosts], 0.9))
                    ctx.guest.put(f"fcfedgcn_dp_clip_norm_{epoch}", dp_clip_norm)
                    ctx.hosts.put(f"fcfedgcn_dp_clip_norm_{epoch}", [dp_clip_norm] * len(norm_hosts))
                    print(f"[DPCalibration] FCFedGCN arbiter epoch={epoch + 1} clip_norm={dp_clip_norm:.8f}", flush=True)

                if epoch < 3 or (epoch + 1) % 50 == 0:
                    print(
                        f"[FCFedGCN-DP Sync] arbiter round={epoch + 1} waiting for guest upload", flush=True,
                    )
                w_guest = ctx.guest.get(f"weights_{epoch}")
                if epoch < 3 or (epoch + 1) % 50 == 0:
                    print(
                        f"[FCFedGCN-DP Sync] arbiter round={epoch + 1} received guest upload; waiting for hosts", flush=True,
                    )
                w_hosts = ctx.hosts.get(f"weights_{epoch}")
                if not isinstance(w_hosts, list): w_hosts = [w_hosts]
                all_weights = [w_guest] + w_hosts
                if epoch < 3 or (epoch + 1) % 50 == 0:
                    print(
                        f"[FCFedGCN-DP Sync] arbiter round={epoch + 1} received {len(all_weights)} uploads; aggregating", flush=True,
                    )
                
                if args.protection == 'he':
                    he_started = time.perf_counter()
                    encrypted_sum = ckks_homomorphic_weighted_sum_tree(
                        all_weights, [float(value) for value in all_nodes_count], he_arbiter_context,
                    )
                    summed_delta = ckks_decrypt_tree(encrypted_sum, he_arbiter_context)
                    capture_he_sa_aggregate(ctx, args, f"fcfedgcn_aggregate_{epoch}", summed_delta)
                    global_weights = {key: value / total_nodes for key, value in summed_delta.items()}
                    he_seconds = time.perf_counter() - he_started
                    byte_guest = int(ctx.guest.get(f"fcfedgcn_he_upload_bytes_{epoch}"))
                    byte_hosts = ctx.hosts.get(f"fcfedgcn_he_upload_bytes_{epoch}")
                    byte_hosts = byte_hosts if isinstance(byte_hosts, list) else [byte_hosts]
                    print(f"[HESA] FCFedGCN arbiter epoch={epoch + 1} aggregate_decrypt_s={he_seconds:.6f} encrypted_upload_bytes={byte_guest + sum(int(value) for value in byte_hosts)}", flush=True)
                else:
                    global_weights = {}
                    for key in all_weights[0].keys():
                        global_weights[key] = sum(agg_weights[i] * all_weights[i][key] for i in range(len(all_weights)))
                    he_seconds = None
                    
                ctx.guest.put(f"global_weights_{epoch}", global_weights)
                ctx.hosts.put(f"global_weights_{epoch}", [global_weights] * len(w_hosts))
                if epoch < 3 or (epoch + 1) % 50 == 0:
                    print(
                        f"[FCFedGCN-DP Sync] arbiter round={epoch + 1} broadcast global aggregate", flush=True,
                    )
                if args.protection == 'he':
                    ctx.guest.put(f"fcfedgcn_he_arbiter_seconds_{epoch}", he_seconds)
                    ctx.hosts.put(f"fcfedgcn_he_arbiter_seconds_{epoch}", [he_seconds] * len(w_hosts))
                if bool(ctx.guest.get(f"fcfedgcn_stop_{epoch}")):
                    print(f"[FCFedGCN] arbiter received early-stop at round {epoch + 1}", flush=True)
                    break

    except EarlyStopSignal:
        if not ctx.is_on_arbiter:
            print(f"Rank {ctx.rank}: 🎉 成功跳出训练循环 (实际通信 {actual_epochs} 轮)！准备进行最终反归一化物理测试...")

    # ================= 3. 最终测试与数据落盘 =================
    if not ctx.is_on_arbiter:
        # 回滚至最优权重
        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)
            
        model.eval()
        test_start = time.time()
        test_mae, test_rmse, test_mape = 0.0, 0.0, 0.0
        test_elements, valid_mape_count = 0, 0
        test_mape_error_sum = 0.0
        
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test, y_test = x_test.to(args.device), y_test.to(args.device)
                pred_test = model(x_test)
                
                if y_test.dim() == 4 and y_test.shape[-1] == 1: y_test = y_test.squeeze(-1)
                if pred_test.dim() == 4 and pred_test.shape[-1] == 1: pred_test = pred_test.squeeze(-1)
                if pred_test.shape != y_test.shape: pred_test = pred_test.reshape_as(y_test)
                
                # 【严格复现】：仅在测试阶段调用 scaler.inverse_transform 还原为物理尺度真实值
                y_real = scaler.inverse_transform(y_test).cpu().numpy()
                pred_real = scaler.inverse_transform(pred_test).cpu().numpy()
                
                diff = pred_real - y_real
                abs_error_sum = np.sum(np.abs(diff))
                sq_error_sum = np.sum(diff ** 2)

                test_mae += abs_error_sum
                test_rmse += sq_error_sum
                test_elements += y_real.size
                
                # 过滤极小值，计算真实 MAPE
                mask = y_real > 0.5
                if np.sum(mask) > 0:
                    mape_error_sum = np.sum(np.abs(diff[mask]) / y_real[mask])
                    test_mape += mape_error_sum
                    test_mape_error_sum += mape_error_sum * 100.0
                    valid_mape_count += np.sum(mask)

        # 结算真实物理指标
        # 结算真实物理指标 (增加防御性编程)
        acc_mae = (test_mae / test_elements) if test_elements > 0 else 0.0
        acc_rmse = math.sqrt(test_rmse / test_elements) if test_elements > 0 else 0.0
        acc_mape = (test_mape / valid_mape_count) * 100.0 if valid_mape_count > 0 else 0.0
        
        eff_test_time = time.time() - test_start
        eff_train_time = total_train_time
        eff_val_time = total_val_time / max(actual_epochs, 1) # 平均每轮验证耗时
        
        # 统计通信负担 (参数数量)
        comm_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if args.protection == 'he':
            # For HE this return field carries measured bytes, not parameter count.
            comm_params = he_upload_bytes + he_download_bytes
        
        # 动态计算 FLOPs
        eff_flops = 0.0
        try:
            from thop import profile
            dummy_x = next(iter(val_loader))[0].to(args.device)
            flops, _ = profile(model, inputs=(dummy_x,), verbose=False)
            eff_flops = round(flops / 1e9, 4)
        except Exception as e:
            print(f"Rank {ctx.rank}: FLOPs 计算失败，已回退为 0.0。")

        # 返回指标给 fate_main.py 写入 CSV；末尾几项用于样本级加权汇总。
        return (
            best_epoch, acc_mae, acc_rmse, acc_mape,
            eff_train_time, eff_val_time, eff_test_time,
            comm_params, actual_epochs, eff_flops,
            test_elements, test_mae, test_rmse,
            test_mape_error_sum, valid_mape_count,
        )
