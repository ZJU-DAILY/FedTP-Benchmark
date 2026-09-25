import time
import copy
import torch
import pandas as pd
import numpy as np
import scipy.stats
import math
import os
import signal
import sys
import traceback
import hashlib
from model.refol_nets import AttGCN, GRU
from lib.utils import ExplicitEarlyStopper, synchronize_cuda_for_timing
from privacy.protection import l2_norm
from privacy.runtime_protection import (
    protected_arbiter_put, record_he_ttp_downlink, unprotect_he_ttp_payload,
)
from privacy.attack_trace import capture_revised_quantized_prediction


def _state_fingerprint(state_dict):
    """Stable short fingerprint for paired REFOL reproducibility audits."""
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        if torch.is_tensor(value):
            digest.update(key.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def _refol_state_weights(active_clients_info):
    """Return the historical REFOL client weights.

    REFOL's submitted Plain experiments used the number of physical road
    nodes held by every currently participating client as its aggregation
    weight.  Do not silently replace this with an unweighted graph diffusion:
    doing so changes the baseline even when DP has ``sigma=0``.
    """
    weights = [max(1, len(info.get("nodes", []))) for info in active_clients_info]
    total = float(sum(weights)) if weights else 1.0
    return [float(weight) / total for weight in weights]


def _average_state_dicts(state_dicts, weights):
    """Weighted CPU FedAvg preserving parameter dtypes and non-float entries."""
    averaged = {}
    first_state = state_dicts[0]
    for name, first_tensor in first_state.items():
        if not torch.is_tensor(first_tensor) or not first_tensor.is_floating_point():
            averaged[name] = first_tensor.clone() if torch.is_tensor(first_tensor) else copy.deepcopy(first_tensor)
            continue
        acc = torch.zeros_like(first_tensor, dtype=torch.float32, device="cpu")
        for state, weight in zip(state_dicts, weights):
            acc += state[name].detach().cpu().float() * float(weight)
        averaged[name] = acc.to(dtype=first_tensor.dtype).contiguous()
    return averaged


def _blend_state_dicts(base_state, residual_state, alpha):
    """Historical residual REFOL aggregation: ``(1-alpha)*FedAvg + alpha*GCN``."""
    alpha = max(0.0, min(1.0, float(alpha)))
    blended = {}
    for name, base_tensor in base_state.items():
        if (
            torch.is_tensor(base_tensor)
            and base_tensor.is_floating_point()
            and name in residual_state
            and torch.is_tensor(residual_state[name])
        ):
            mixed = (1.0 - alpha) * base_tensor.detach().cpu().float()
            mixed += alpha * residual_state[name].detach().cpu().float()
            blended[name] = mixed.to(dtype=base_tensor.dtype).contiguous()
        else:
            blended[name] = base_tensor.clone() if torch.is_tensor(base_tensor) else copy.deepcopy(base_tensor)
    return blended


def train_refol_distributed(ctx, args, setting=None):
    """
    FATE 分布式 REFOL 训练引擎 (流式在线学习版)
    """
    # 【核弹级异常捕获】：只要有报错，立刻砸在屏幕上并强杀进程，拒绝任何无声死锁！
    try:
        # ``fate_main`` seeds every FATE rank before model construction.  Do
        # not reset it again here: clients have already constructed their GRU
        # while the arbiter has not, which creates asymmetric RNG histories.
        total_rounds = args.epochs  
        if ctx.is_on_arbiter:
            run_refol_server(ctx, args, total_rounds)
        else:
            run_refol_client(ctx, args, setting, total_rounds)
    except Exception as e:
        error_msg = f"\n{'='*50}\n🔥 CRITICAL ERROR on Rank {ctx.rank} 🔥\n{traceback.format_exc()}\n{'='*50}\n"
        print(error_msg, file=sys.stderr, flush=True)
        os._exit(1) # 强行结束，打破 FATE 的无限等待


# ==========================================
# 0. Server 专属工具：根据物理距离构建 Client 拓扑图
# ==========================================
def get_client_edge_index(active_clients_info, dataset_name, legacy_refol=False):
    """
    根据活跃 Client 拥有的物理节点，从 distance.csv 动态推导 Client 级别的空间邻接关系
    并加入 Server 的虚拟节点 [cite: 331-337]。

    ``legacy_refol`` reproduces the directed graph used by the original
    four-client REFOL runner: road edges keep their file direction, only
    client-to-server edges are added, and only the virtual server has an
    explicit self-loop.  The later bidirectional/self-loop topology is kept
    for the newer FedAvg-based variants.
    """
    num_active = len(active_clients_info)
    dist_file = f'data/{dataset_name}/distance.csv'
    
    try:
        dist_df = pd.read_csv(dist_file)
    except Exception as e:
        print(f"[Warning] 无法读取 {dist_file}，将退化为全连接图进行聚合。原因: {e}")
        src, dst = [], []
        for i in range(num_active + 1):
            for j in range(num_active + 1):
                src.append(i)
                dst.append(j)
        return torch.tensor([src, dst], dtype=torch.long, device="cpu")

    # 1. 建立 物理节点ID -> 活跃 Client 索引(0 到 num_active-1) 的映射表
    node_to_active_idx = {}
    for idx, info in enumerate(active_clients_info):
        for node in info['nodes']:
            node_to_active_idx[int(node)] = idx

    # 2. 遍历路网，如果连通的两个节点分别属于两个活跃 Client，则这两个 Client 相连
    client_edges = set()
    for _, row in dist_df.iterrows():
        u, v, cost = int(float(row['from'])), int(float(row['to'])), float(row['cost'])
        if cost > 0 and u in node_to_active_idx and v in node_to_active_idx:
            idx_u = node_to_active_idx[u]
            idx_v = node_to_active_idx[v]
            client_edges.add((idx_u, idx_v))
            if not legacy_refol:
                client_edges.add((idx_v, idx_u)) # 无向图双向传递
            
    # 3. 补齐 Client 自身的自环
    if not legacy_refol:
        for i in range(num_active):
            client_edges.add((i, i))

    # 4. 核心：引入 Server 的虚拟节点 (索引为 num_active) 
    # 虚拟节点与所有活跃 Client 相连，用于汇总全局空间信息
    virtual_node_idx = num_active
    client_edges.add((virtual_node_idx, virtual_node_idx)) # Server 自环
    for i in range(num_active):
        client_edges.add((i, virtual_node_idx))
        if not legacy_refol:
            client_edges.add((virtual_node_idx, i))
        
    src = [e[0] for e in client_edges]
    dst = [e[1] for e in client_edges]
    
    return torch.tensor([src, dst], dtype=torch.long, device="cpu")


# ==========================================
# 1. Server (Arbiter) 逻辑：图聚合中心
# ==========================================
def run_refol_server(ctx, args, total_rounds):
    print("[REFOL Server] 启动，构建动态图结构并等待握手...")
    # Server 仅做聚合，强制用 CPU 计算，避免多个进程争抢 GPU 导致底层挂起
    gcn_aggregator = AttGCN().to("cpu")
    print(
        f"[REFOL-Reproducibility] arbiter_attgcn_sha256="
        f"{_state_fingerprint(gcn_aggregator.state_dict())}",
        flush=True,
    )
    
    # Keep the same per-client state semantics as Plain REFOL.  A client that
    # skips a drift round retains its own previous local model; DP deltas must
    # therefore be decoded against that client's reference, not one shared
    # current global model.
    global_model_state = None
    # The scalable variant defaults to a node-count weighted FedAvg residual.
    # The archived REFOL trainer used pure GCN aggregation.
    aggregation_mode = str(getattr(args, "refol_aggregation", "fedavg_residual")).lower()
    residual_alpha = float(getattr(args, "refol_gcn_residual_alpha", 0.1))
    topology_mode = str(getattr(args, "refol_topology", "bidirectional")).lower()
    legacy_topology = topology_mode == "legacy_directed"
    print(
        f"[REFOL Server] aggregation={aggregation_mode} "
        f"gcn_residual_alpha={residual_alpha:.6f}",
        flush=True,
    )
    # This is only a protocol-equivalence diagnostic.  With sigma=0 and a
    # fixed non-binding C, use REFOL's original absolute-state payload exactly;
    # it isolates the DP transport wrapper from the separate delta protocol.
    dp_identity_diagnostic = (
        args.protection == "dp"
        and float(getattr(args, "dp_sigma", 0.0)) == 0.0
        and float(getattr(args, "dp_clip_norm", 0.0)) > 0.0
    )
    dp_client_reference_states = None
    dp_clip_norm = (
        float(args.dp_clip_norm)
        if args.protection == "dp" and float(getattr(args, "dp_clip_norm", 0.0)) > 0
        else None
    )
    if args.protection == "dp" and not dp_identity_diagnostic:
        init_guest = ctx.guest.get("refol_public_init_state")
        init_hosts = ctx.hosts.get("refol_public_init_state")
        init_hosts = init_hosts if isinstance(init_hosts, list) else [init_hosts]
        init_states = [init_guest] + init_hosts
        if any(not isinstance(state, dict) for state in init_states):
            raise ValueError("REFOL DP requires one initial state dictionary per client.")
        dp_client_reference_states = [
            {key: value.detach().cpu().clone() for key, value in state.items()}
            for state in init_states
        ]
        print(
            f"[REFOL-DP] registered {len(dp_client_reference_states)} per-client "
            "initial references; skipped clients retain their Plain REFOL state.",
            flush=True,
        )
    elif dp_identity_diagnostic:
        print(
            "[REFOL-DP Equivalence] sigma=0 fixed-C: using original absolute-state "
            "REFOL aggregation; DP transport is numerically identity.",
            flush=True,
        )

    for rround in range(1, total_rounds + 1):
        # --- 阶段 1：接收 Client 的参与状态与节点分布 ---
        status_guest = ctx.guest.get(f"status_{rround}")
        status_hosts = ctx.hosts.get(f"status_{rround}")
        if not isinstance(status_hosts, list): status_hosts = [status_hosts]
        if args.protection == "dp" and dp_clip_norm is None:
            norm_guest = ctx.guest.get(f"refol_dp_norm_{rround}")
            norm_hosts = ctx.hosts.get(f"refol_dp_norm_{rround}")
            norm_hosts = norm_hosts if isinstance(norm_hosts, list) else [norm_hosts]
            active_norms = [float(v) for v in [norm_guest] + norm_hosts if v is not None]
            dp_clip_norm = float(np.quantile(active_norms, 0.9)) if active_norms else 1.0
            ctx.guest.put(f"refol_dp_clip_{rround}", dp_clip_norm)
            ctx.hosts.put(f"refol_dp_clip_{rround}", [dp_clip_norm] * len(norm_hosts))
            print(f"[DPCalibration] REFOL round={rround} clip_norm={dp_clip_norm:.8f}", flush=True)
        
        # --- 阶段 2：接收 Client 的本地模型权重 ---
        weights_guest = unprotect_he_ttp_payload(
            args, ctx.guest.get(f"weights_{rround}")
        )
        weights_hosts = ctx.hosts.get(f"weights_{rround}")
        if not isinstance(weights_hosts, list): weights_hosts = [weights_hosts]
        weights_hosts = [
            unprotect_he_ttp_payload(args, value) for value in weights_hosts
        ]
        
        all_statuses = [status_guest] + status_hosts
        all_weights = [weights_guest] + weights_hosts
        
        valid_states = []
        active_indices = []
        active_clients_info = []
        
        # 筛选出本轮检测到概念漂移并上传了权重的 Client
        for idx, (st, wt) in enumerate(zip(all_statuses, all_weights)):
            if wt is not None and st is not None and st.get("participate", False):
                if args.protection == "dp" and not dp_identity_diagnostic:
                    reference = dp_client_reference_states[idx]
                    reconstructed = {
                        key: (
                            reference[key] + wt[key].detach().cpu()
                            if key in wt and reference[key].is_floating_point()
                            else reference[key].detach().cpu().clone()
                        )
                        for key in reference
                    }
                    valid_states.append(reconstructed)
                else:
                    valid_states.append(wt)
                active_indices.append(idx)
                active_clients_info.append({
                    'rank_idx': idx, 
                    'nodes': st.get("global_node_ids", [])
                })

        # --- 阶段 3：执行基于空间图卷积的参数聚合  ---
        if len(valid_states) > 0:
            num_active = len(valid_states)
            agg_weights = _refol_state_weights(active_clients_info)
            fedavg_state = _average_state_dicts(valid_states, agg_weights)
            
            # 动态生成符合真实物理路网的连通图
            final_edge_index = get_client_edge_index(
                active_clients_info,
                args.dataset_name,
                legacy_refol=legacy_topology,
            )
            
            if global_model_state is None:
                # Original REFOL seeds the virtual server node from the first
                # active client before applying its pure GCN aggregation.  The
                # residual/FedAvg variants instead use the weighted mean.
                global_model_state = copy.deepcopy(
                    valid_states[0] if aggregation_mode == "gcn" else fedavg_state
                )
            
            # 将活跃 Client 和 Server(上一轮全局模型) 的特征打包
            features_list = valid_states + [global_model_state]
            flat_features = []
            for state in features_list:
                state_flat = []
                for name in state:
                    state_flat += state[name].flatten().tolist()
                flat_features.append(state_flat)
            
            feature_tensor = torch.Tensor(flat_features).to("cpu")

            # 喂入 2 层 GCN 聚合器 [cite: 339-349]
            agg_output = gcn_aggregator(feature_tensor, final_edge_index)
            
            # 提取最后一行 (虚拟节点) 作为新的全局模型
            new_global_flat = agg_output[-1]
            new_global_state = {}
            len_start = 0
            for name in global_model_state:
                length = len(global_model_state[name].flatten().tolist())
                new_global_state[name] = new_global_flat[len_start:len_start + length].reshape_as(global_model_state[name]).cpu()
                len_start += length
                
            if aggregation_mode == "gcn":
                global_model_state = new_global_state
            elif aggregation_mode == "fedavg":
                global_model_state = fedavg_state
            else:
                global_model_state = _blend_state_dicts(
                    fedavg_state, new_global_state, residual_alpha
                )
            print(
                f"Server: Round {rround} REFOL aggregation={aggregation_mode} "
                f"| topology={topology_mode} "
                f"| active_clients={num_active} | edges={final_edge_index.shape[1]} "
                f"| weights={[round(weight, 4) for weight in agg_weights]}",
                flush=True,
            )
            if args.protection == "dp" and not dp_identity_diagnostic:
                # Only participating clients load this state below.  Leave
                # skipped-client references untouched to preserve Plain REFOL.
                for idx in active_indices:
                    dp_client_reference_states[idx] = {
                        key: value.detach().cpu().clone()
                        for key, value in global_model_state.items()
                    }
            print(f"Server: Round {rround} 图聚合成功 | 活跃 Client: {num_active} 个 | 边数量: {final_edge_index.shape[1]}")
            
        ctx.guest.put(f"global_{rround}", global_model_state)
        ctx.hosts.put(f"global_{rround}", [global_model_state] * len(status_hosts))

        # --- 阶段 4：接收 Client 的早停信号 ---
        is_stop_guest = ctx.guest.get(f"early_stop_{rround}")
        if is_stop_guest:
            print(f"Server: 🛑 收到 Client 早停信号 (Round {rround})，终止聚合！")
            break


# ==========================================
# 2. Client 逻辑：本地在线学习与漂移检测
# ==========================================
def run_refol_client(ctx, args, setting, total_rounds):
    print(f"[REFOL Client {ctx.rank}] 启动本地流式学习...")
    
    train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = setting
    local_model = model.to(args.device)
    print(
        f"[REFOL-Reproducibility] rank={ctx.rank} initial_gru_sha256="
        f"{_state_fingerprint(local_model.state_dict())}",
        flush=True,
    )
    historical_data_flat = None 
    global_node_ids = args.nodes_per[ctx.rank]
    
    # drop_last=True，丢弃尾部残缺 batch，保证和历史 batch 尺寸完美一致用于计算 KL 散度
    data_stream_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    stream_iterator = iter(data_stream_loader)
    stopper = ExplicitEarlyStopper(ctx, args, patience=50, min_delta=1e-4) 
    legacy_teacher_forcing_eval = bool(
        getattr(args, "refol_legacy_teacher_forcing_eval", False)
    )
    if legacy_teacher_forcing_eval:
        print(
            "[REFOL Evaluation] protocol=legacy_teacher_forcing: validation and "
            "test receive true future labels. Historical reproduction only; not "
            "strict autoregressive forecasting.",
            flush=True,
        )
    dp_identity_diagnostic = (
        args.protection == "dp"
        and float(getattr(args, "dp_sigma", 0.0)) == 0.0
        and float(getattr(args, "dp_clip_norm", 0.0)) > 0.0
    )
    dp_clip_norm = float(args.dp_clip_norm) if args.protection == "dp" and float(getattr(args, "dp_clip_norm", 0.0)) > 0 else None
    if args.protection == "dp" and not dp_identity_diagnostic:
        # The server needs this public protocol initialisation only to decode
        # a later delta.  Do not overwrite the local model: Plain REFOL starts
        # each client from its own state and keeps stale states when it skips.
        ctx.arbiter.put("refol_public_init_state", {k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()})
        print(f"[REFOL-DP] rank={ctx.rank} submitted initial reference without replacing local state", flush=True)
    elif dp_identity_diagnostic:
        print(
            f"[REFOL-DP Equivalence] rank={ctx.rank} will upload original absolute states.",
            flush=True,
        )

    actual_epochs = 0
    total_train_time = 0.0
    total_val_time = 0.0
    best_state = None
    best_epoch = 0
    best_norm_mae = float("inf")

    for rround in range(1, total_rounds + 1):
        try:
            batch = next(stream_iterator)
        except StopIteration:
            print(f"Client {ctx.rank}: 数据流已遍历完毕 (共 {rround-1} 步)，自然结束流式学习。")
            actual_epochs = rround - 1
            break  
            
        x, y = batch[0].to(args.device), batch[1].to(args.device)
        
        participate = False
        current_data_flat = x.cpu().numpy().flatten()
        
        # --- 概念漂移检测 (Concept Drift Detection) [cite: 208-216] ---
        if historical_data_flat is None:
            participate = True
        else:
            if hasattr(scaler, 'inverse_transform'):
                data_now_real = scaler.inverse_transform(current_data_flat) + 1e-8
                data_h_real = scaler.inverse_transform(historical_data_flat) + 1e-8
            else:
                data_now_real = current_data_flat * scaler.std + scaler.mean + 1e-8
                data_h_real = historical_data_flat * scaler.std + scaler.mean + 1e-8
                
            kld = scipy.stats.entropy(data_now_real, data_h_real)
            if kld > args.kl_threshold:
                participate = True

        train_start = time.time()
        
        # 将本地的节点分布发给 Server，供其构建路网拓扑图
        ctx.arbiter.put(f"status_{rround}", {"participate": participate, "global_node_ids": global_node_ids})
        
        if participate:
            round_start_state = {k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()}
            local_model.train()
            x_refol = x.transpose(1, 2)
            y_refol = y.transpose(1, 2)
            monitor_x_attr = torch.empty(x_refol.shape[:-1] + (0,), device=args.device)
            monitor_y_attr = torch.empty(y_refol.shape[:-1] + (0,), device=args.device)
            with torch.no_grad():
                monitor_prediction = local_model({
                    "x": x_refol, "x_attr": monitor_x_attr,
                    "y": torch.zeros_like(y_refol), "y_attr": monitor_y_attr,
                    "teacher_forcing": False,
                }).transpose(1, 2)
            capture_revised_quantized_prediction(
                ctx, args, f"refol_prediction_{rround}", prediction=monitor_prediction,
                model_state_dict=round_start_state,
            )
            for _ in range(args.local_epochs if hasattr(args, 'local_epochs') else 1): 
                optimizer.zero_grad()
                x_refol = x.transpose(1, 2)
                y_refol = y.transpose(1, 2)
                dummy_x_attr = torch.empty(x_refol.shape[:-1] + (0,), device=args.device)
                dummy_y_attr = torch.empty(y_refol.shape[:-1] + (0,), device=args.device)
                
                out = local_model({
                    "x": x_refol,
                    "x_attr": dummy_x_attr,
                    "y": y_refol,
                    "y_attr": dummy_y_attr,
                    "teacher_forcing": True,
                })
                out = out.transpose(1, 2)
                
                if out.shape != y.shape: out = out.reshape_as(y)
                loss = loss_func(out, y)
                loss.backward()
                optimizer.step()
                
            local_weights_cpu = {k: v.cpu().clone() for k, v in local_model.state_dict().items()}
            if args.protection == "dp" and dp_identity_diagnostic:
                # Preserve the exact Plain REFOL payload in this diagnostic.
                # The runtime wrapper sees sigma=0 and a non-binding C, so it
                # serializes an unchanged copy of the absolute local state.
                protected_arbiter_put(ctx, args, f"weights_{rround}", local_weights_cpu, clip_norm=dp_clip_norm)
            elif args.protection == "dp":
                local_payload = {k: local_weights_cpu[k] - round_start_state[k] for k in round_start_state if local_weights_cpu[k].is_floating_point()}
                ctx.arbiter.put(f"refol_dp_norm_{rround}", float(l2_norm(local_payload).item()) if dp_clip_norm is None else None)
                if dp_clip_norm is None:
                    dp_clip_norm = ctx.arbiter.get(f"refol_dp_clip_{rround}")
                    if isinstance(dp_clip_norm, (tuple, list)): dp_clip_norm = dp_clip_norm[0]
                    dp_clip_norm = float(dp_clip_norm)
                protected_arbiter_put(ctx, args, f"weights_{rround}", local_payload, clip_norm=dp_clip_norm)
            elif args.protection == "he":
                protected_arbiter_put(ctx, args, f"weights_{rround}", local_weights_cpu)
            else:
                ctx.arbiter.put(f"weights_{rround}", local_weights_cpu)
            
            new_global_state = ctx.arbiter.get(f"global_{rround}")
            if new_global_state is not None:
                if isinstance(new_global_state, list):
                    new_global_state = new_global_state[0]
                new_global_state = record_he_ttp_downlink(args, new_global_state)
                local_model.load_state_dict(new_global_state)
                
            historical_data_flat = copy.deepcopy(current_data_flat)
        else:
            # 未发生漂移，复用历史模型，不产生通信负担 [cite: 288-290, 302-304]
            if args.protection == "dp" and not dp_identity_diagnostic:
                ctx.arbiter.put(f"refol_dp_norm_{rround}", None)
                if dp_clip_norm is None:
                    dp_clip_norm = ctx.arbiter.get(f"refol_dp_clip_{rround}")
                    if isinstance(dp_clip_norm, (tuple, list)): dp_clip_norm = dp_clip_norm[0]
                    dp_clip_norm = float(dp_clip_norm)
            ctx.arbiter.put(f"weights_{rround}", None)
            ignored_global_state = ctx.arbiter.get(f"global_{rround}")
            if isinstance(ignored_global_state, list):
                ignored_global_state = ignored_global_state[0]
            record_he_ttp_downlink(args, ignored_global_state)

        total_train_time += (time.time() - train_start)

        val_start = time.time()
        local_model.eval()
        val_mae_norm = 0.0
        val_elements = 0
        
        with torch.no_grad():
            for v_batch in val_loader:
                vx, vy = v_batch[0].to(args.device), v_batch[1].to(args.device)
                vx_refol = vx.transpose(1, 2)
                vy_refol = vy.transpose(1, 2)
                dummy_vx_attr = torch.empty(vx_refol.shape[:-1] + (0,), device=args.device)
                dummy_vy_attr = torch.empty(vy_refol.shape[:-1] + (0,), device=args.device)
                
                v_out = local_model({
                    "x": vx_refol,
                    "x_attr": dummy_vx_attr,
                    "y": vy_refol,
                    "y_attr": dummy_vy_attr,
                    "teacher_forcing": legacy_teacher_forcing_eval,
                })
                v_out = v_out.transpose(1, 2)
                
                if v_out.shape != vy.shape: v_out = v_out.reshape_as(vy)
                val_mae_norm += torch.sum(torch.abs(v_out - vy)).item()
                val_elements += vy.numel()
                
        local_norm_mae = float(val_mae_norm / val_elements)
        total_val_time += (time.time() - val_start)
        
        if rround % 10 == 0 or rround == 1:
            print(f"Client {ctx.rank} Round {rround} | 归一化 Val MAE: {local_norm_mae:.4f}")

        stop_decision = stopper.check_and_sync(local_norm_mae)
        is_best = bool(getattr(stop_decision, "is_best", False))
        should_stop = bool(stop_decision)
        # Historical REFOL evaluates the local model from its best validation
        # round, rather than silently evaluating the final streamed update.
        if is_best or best_state is None or local_norm_mae < best_norm_mae:
            best_norm_mae = local_norm_mae
            best_epoch = rround
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in local_model.state_dict().items()
            }
            print(
                f"Client {ctx.rank}: REFOL saved best federated checkpoint "
                f"round={best_epoch} val_mae_norm={best_norm_mae:.6f}",
                flush=True,
            )
        ctx.arbiter.put(f"early_stop_{rround}", bool(should_stop))

        if should_stop:
            actual_epochs = rround
            print(f"Client {ctx.rank}: 🛑 收到全局早停信号 (Round {rround})，优雅跳出训练循环！")
            break
        
        actual_epochs = rround

    print(f"Rank {ctx.rank}: 🚀 进入测试集评估阶段...")
    if bool(getattr(args, "refol_restore_best", True)) and best_state is not None:
        local_model.load_state_dict({key: value.to(args.device) for key, value in best_state.items()})
        print(
            f"Rank {ctx.rank}: REFOL restored best federated checkpoint "
            f"round={best_epoch} val_mae_norm={best_norm_mae:.6f}",
            flush=True,
        )
    elif best_state is None:
        print(f"Rank {ctx.rank}: REFOL no best checkpoint recorded; testing final state.", flush=True)

    local_model.eval()
    synchronize_cuda_for_timing(args.device)
    test_start_t = time.perf_counter()
    test_mae, test_rmse, test_mape = 0.0, 0.0, 0.0
    test_elements, valid_test_mape_count = 0, 0
    test_batches = 0
    
    with torch.no_grad():
        for t_batch in test_loader:
            test_batches += 1
            tx, ty = t_batch[0].to(args.device), t_batch[1].to(args.device)
            tx_refol = tx.transpose(1, 2)
            ty_refol = ty.transpose(1, 2)
            dummy_tx_attr = torch.empty(tx_refol.shape[:-1] + (0,), device=args.device)
            dummy_ty_attr = torch.empty(ty_refol.shape[:-1] + (0,), device=args.device)
            
            t_out = local_model({
                "x": tx_refol,
                "x_attr": dummy_tx_attr,
                "y": ty_refol,
                "y_attr": dummy_ty_attr,
                "teacher_forcing": legacy_teacher_forcing_eval,
            })
            t_out = t_out.transpose(1, 2)
            
            if t_out.shape != ty.shape: t_out = t_out.reshape_as(ty)
            
            if hasattr(scaler, 'inverse_transform'):
                y_real = scaler.inverse_transform(ty).cpu().numpy()
                pred_real = scaler.inverse_transform(t_out).cpu().numpy()
            else:
                y_real = ty.cpu().numpy() * scaler.std + scaler.mean
                pred_real = t_out.cpu().numpy() * scaler.std + scaler.mean
            
            test_mae += np.sum(np.abs(y_real - pred_real))
            test_rmse += np.sum((y_real - pred_real) ** 2)
            test_elements += y_real.size
            
            mask = y_real > 0.5
            if np.sum(mask) > 0:
                test_mape += np.sum(np.abs(y_real[mask] - pred_real[mask]) / y_real[mask])
                valid_test_mape_count += np.sum(mask)
                
    acc_mae = round(test_mae / test_elements, 4)
    acc_rmse = round(math.sqrt(test_rmse / test_elements), 4)
    acc_mse = round(acc_rmse ** 2, 4)
    acc_mape = round((test_mape / valid_test_mape_count) if valid_test_mape_count > 0 else 0.0, 4)
    synchronize_cuda_for_timing(args.device)
    eff_test_time = round(time.perf_counter() - test_start_t, 4)
    print(
        f"[TestTimingAudit] model=REFOL rank={ctx.rank} batches={test_batches} "
        f"elements={test_elements} seconds={eff_test_time:.6f}",
        flush=True,
    )
    
    # 通信开销
    comm_params = sum(p.numel() for p in local_model.parameters())
    eff_comm_size_mb = round((comm_params * 4 * 2 * actual_epochs) / (1024 * 1024), 4)

    # 计算 FLOPs
    eff_flops = 0.0
    try:
        from thop import profile
        class RefolFlopsWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x_in, x_attr_in, y_in, y_attr_in):
                return self.m({"x": x_in, "x_attr": x_attr_in, "y": y_in, "y_attr": y_attr_in})

        wrapper = RefolFlopsWrapper(local_model).to(args.device)
        dummy_batch = next(iter(test_loader))
        dx, dy = dummy_batch[0].to(args.device), dummy_batch[1].to(args.device)
        dx_refol = dx.transpose(1, 2)
        dy_refol = dy.transpose(1, 2)
        dummy_x_attr = torch.empty(dx_refol.shape[:-1] + (0,), device=args.device)
        dummy_y_attr = torch.empty(dy_refol.shape[:-1] + (0,), device=args.device)
        
        flops, _ = profile(wrapper, inputs=(dx_refol, dummy_x_attr, dy_refol, dummy_y_attr), verbose=False)
        eff_flops = round(flops / 1e9, 4) 
    except Exception:
        eff_flops = 0.0

    from fate_main import log_experiment_results
    dataset_client_name = f"{args.dataset_name}_client{ctx.rank}"
    
    log_experiment_results(
        model_name=args.model, 
        dataset_client=dataset_client_name, 
        feature_type=args.feature_type,
        best_epoch=best_epoch if best_epoch > 0 else actual_epochs,
        acc_mae=acc_mae, 
        acc_mse=acc_mse, 
        acc_rmse=acc_rmse, 
        acc_mape=acc_mape,
        eff_train_time=round(total_train_time, 2), 
        eff_val_time=round(total_val_time / actual_epochs, 4) if actual_epochs > 0 else 0.0, 
        eff_test_time=eff_test_time,
        eff_comm_size_mb=eff_comm_size_mb,
        eff_train_round=actual_epochs,
        eff_flops=eff_flops,
        dp_noise=getattr(args, 'dp_noise', 0.0)
    )
    print(f"🎉 Client {ctx.rank} 物理指标已写入 CSV：MAE={acc_mae}, RMSE={acc_rmse}")

    if ctx.is_on_guest and not bool(getattr(args, "cross_device_multiplex", False)):
        print("Guest 节点：数据已安全保存，准备发送终止信号清理 Arbiter...")
        time.sleep(1.5)
        os.kill(os.getppid(), signal.SIGTERM)
