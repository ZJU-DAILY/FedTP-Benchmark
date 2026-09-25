import math
import copy
import torch
import numpy as np
from torch.utils.data import DataLoader
import os
import time

from lib.load_dataset import load_dataset
from lib.utils import ExplicitEarlyStopper, EarlyStopSignal
from model.CNFGNN import CNFGNN
from privacy.protection import l2_norm
from privacy.runtime_protection import (
    protected_arbiter_put, record_he_ttp_downlink, unprotect_he_ttp_payload,
)
from privacy.attack_trace import capture_revised_quantized_prediction

# 获取项目根目录 (由于此文件在 lib/ 下，向上退一级即可)
current_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.dirname(current_dir)


def unwrap_payload(ctx, payload):
    """
    【核心修复】智能解包函数：
    处理 FATE 框架中 Server 发送 List 给多个 Host 时的切片问题。
    让每个 Client 拿属于自己 shape 的数据，彻底告别 5760 和 5696 的冲突。
    """
    if isinstance(payload, list):
        if ctx.is_on_guest:
            return payload[0]
        else:
            # Host 收到的 list 是针对所有 Host 的，按照 rank-1 取自己那份
            my_idx = ctx.rank - 1
            if 0 <= my_idx < len(payload):
                return payload[my_idx]
            return payload[0]  # 兜底
    return payload


def get_cnfgnn_setup(ctx, args):
    """CNFGNN 专用的数据和模型加载逻辑"""
    total_nodes_map = {
        'PeMS03': 358,
        'PeMS04': 307,
        'PeMSD7': 228,
        'PeMS08': 170,
    }

    matched_key = next((key for key in total_nodes_map if key in args.dataset_name), None)
    if matched_key is None:
        raise ValueError(f"[CNFGNN Error] 未知的数据集节点数: {args.dataset_name}")

    args.total_nodes_global = total_nodes_map[matched_key]

    if not getattr(args, "nodes_per", None):
        quotient, remainder = divmod(args.total_nodes_global, args.num_clients)
        nodes_per = []
        start_idx = 0
        for i in range(args.num_clients):
            count = quotient + (1 if i < remainder else 0)
            nodes_per.append(list(range(start_idx, start_idx + count)))
            start_idx += count
        args.nodes_per = nodes_per

    if ctx.is_on_arbiter:
        selected_nodes = list(range(args.total_nodes_global))
        print(f"[CNFGNN Server] Rank {ctx.rank}: Loading GLOBAL graph with {args.total_nodes_global} nodes...", flush=True)
    else:
        selected_nodes = args.nodes_per[ctx.rank]
        print(f"[CNFGNN Client] Rank {ctx.rank}: Loading LOCAL graph with {len(selected_nodes)} nodes...", flush=True)

    train_set, val_set, test_set, edge_index, scaler = load_dataset(
        dataset_name=args.dataset_name,
        feature_type=args.feature_type,
        normalizer=args.normalizer,
        T_in=args.t_in,
        T_out=args.t_out,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        return_edge_index=True,
        device=args.device,
        selected_nodes=selected_nodes
    )

    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.LongTensor(edge_index)
    edge_index = edge_index.to(args.device)

    dist_file = os.path.join(file_dir, 'data', args.dataset_name, 'distance.csv')
    if os.path.exists(dist_file):
        import pandas as pd
        dist_df = pd.read_csv(dist_file)
        dist_dict = {(int(row['from']), int(row['to'])): float(row['cost']) for _, row in dist_df.iterrows()}

        weights = []
        row_idx, col_idx = edge_index.cpu().numpy()
        sigma_square = 100.0

        for u, v in zip(row_idx, col_idx):
            global_u = selected_nodes[u]
            global_v = selected_nodes[v]
            cost = dist_dict.get((global_u, global_v), 1.0)
            w = cost if cost <= 1.0 else np.exp(-(cost ** 2) / sigma_square)
            weights.append(w)

        edge_weight = torch.tensor(weights, dtype=torch.float32)
    else:
        edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)

    edge_weight = edge_weight.to(args.device)

    num_init_nodes = len(selected_nodes) if not ctx.is_on_arbiter else args.total_nodes_global

    model = CNFGNN(
        num_nodes=num_init_nodes,
        in_dim=args.t_in,
        out_dim=args.t_out,
        hidden_dim=args.hidden_dim,
        edge_index=edge_index,
        edge_weight=edge_weight,
        dropout=0.1
    ).to(args.device)

    return train_set, val_set, test_set, model, scaler


def train_cnfgnn_task(ctx, args):
    """独立的 CNFGNN 训练入口，返回核心指标，供主程序统一写入 CSV"""
    train_set, val_set, test_set, model, scaler = get_cnfgnn_setup(ctx, args)
    loss_func = torch.nn.MSELoss().to(args.device)

    R_c, R_s = 1, 2

    # CNFGNN combines a client-model FedAvg phase with split-learning
    # encoder/gradient exchanges.  Protect all client -> Arbiter tensors,
    # using an independent first-round q90 clipping threshold for each
    # semantic payload type.
    cnfgnn_is_dp = str(getattr(args, "protection", "plain")).lower() == "dp"
    cnfgnn_is_he = str(getattr(args, "protection", "plain")).lower() == "he"
    fixed_dp_clip = float(getattr(args, "dp_clip_norm", 0.0))
    cnfgnn_dp_clips = ({
        key: fixed_dp_clip for key in ("model", "encoding", "spatial_gradient")
    } if cnfgnn_is_dp and fixed_dp_clip > 0 else {})

    def _client_dp_upload(payload_key, tag, value):
        if cnfgnn_is_he:
            return protected_arbiter_put(ctx, args, tag, value)
        if not cnfgnn_is_dp:
            return ctx.arbiter.put(tag, value)
        clip_norm = cnfgnn_dp_clips.get(payload_key)
        if clip_norm is None:
            ctx.arbiter.put(
                f"cnfgnn_dp_{payload_key}_norm_{tag}",
                float(l2_norm(value).item()),
            )
            calibrated = ctx.arbiter.get(f"cnfgnn_dp_{payload_key}_clip_{tag}")
            calibrated = unwrap_payload(ctx, calibrated)
            clip_norm = float(calibrated)
            cnfgnn_dp_clips[payload_key] = clip_norm
            print(
                f"[DPCalibration] CNFGNN rank={ctx.rank} type={payload_key} "
                f"tag={tag} clip_norm={clip_norm:.8f}", flush=True,
            )
        return protected_arbiter_put(ctx, args, tag, value, clip_norm=clip_norm)

    def _server_he_inputs(guest_value, hosts_value):
        host_values = hosts_value if isinstance(hosts_value, list) else [hosts_value]
        return [unprotect_he_ttp_payload(args, guest_value)] + [
            unprotect_he_ttp_payload(args, value) for value in host_values
        ]

    def _server_dp_calibrate(payload_key, tag):
        if not cnfgnn_is_dp or payload_key in cnfgnn_dp_clips:
            return
        norm_guest = float(ctx.guest.get(f"cnfgnn_dp_{payload_key}_norm_{tag}"))
        norm_hosts = ctx.hosts.get(f"cnfgnn_dp_{payload_key}_norm_{tag}")
        norm_hosts = [norm_hosts] if not isinstance(norm_hosts, list) else norm_hosts
        clip_norm = float(np.quantile(
            [norm_guest] + [float(value) for value in norm_hosts], 0.9
        ))
        cnfgnn_dp_clips[payload_key] = clip_norm
        print(
            f"[DPCalibration] CNFGNN arbiter type={payload_key} tag={tag} "
            f"clip_norm={clip_norm:.8f}", flush=True,
        )
        ctx.guest.put(f"cnfgnn_dp_{payload_key}_clip_{tag}", clip_norm)
        host_clips = [clip_norm] * len(norm_hosts)
        ctx.hosts.put(
            f"cnfgnn_dp_{payload_key}_clip_{tag}",
            host_clips if len(host_clips) > 1 else host_clips[0],
        )

    if not ctx.is_on_arbiter:
        optimizer = torch.optim.Adam(model.client_model.parameters(), lr=args.lr, weight_decay=args.wd)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

        ctx.arbiter.put("cnfgnn_init_steps", len(train_loader))
        ctx.arbiter.put("cnfgnn_val_steps", len(val_loader))

        STEPS_PER_EPOCH = len(train_loader)
        VAL_STEPS = len(val_loader)

        best_client_wts = None
        best_epoch = -1
        total_train_time, total_val_time = 0.0, 0.0
        comm_history = []
    else:
        optimizer = torch.optim.Adam(model.server_model.parameters(), lr=args.lr, weight_decay=args.wd)

        s_guest = ctx.guest.get("cnfgnn_init_steps")
        s_hosts = ctx.hosts.get("cnfgnn_init_steps")
        s_hosts = [s_hosts] if not isinstance(s_hosts, list) else s_hosts
        STEPS_PER_EPOCH = min([s_guest] + s_hosts)

        v_guest = ctx.guest.get("cnfgnn_val_steps")
        v_hosts = ctx.hosts.get("cnfgnn_val_steps")
        v_hosts = [v_hosts] if not isinstance(v_hosts, list) else v_hosts
        VAL_STEPS = min([v_guest] + v_hosts)

        best_server_wts = None
        global_best_norm_mae = float("inf")
        global_best_epoch = -1
        patience_counter = 0
        patience = 50

    # A revised-prediction trace needs one synchronized protocol batch only.
    # Apply this limit after every rank has exchanged its real loader length so
    # the clients and Arbiter use the identical message schedule.  Normal
    # training is unchanged unless the trace-only environment switch is set.
    trace_steps = int(getattr(args, "cnfgnn_trace_max_steps", 0) or 0)
    if bool(getattr(args, "privacy_trace_only", False)) and trace_steps > 0:
        STEPS_PER_EPOCH = min(STEPS_PER_EPOCH, trace_steps)
        VAL_STEPS = min(VAL_STEPS, trace_steps)
        print(
            f"[CNFGNN TraceOnly] rank={ctx.rank} limiting synchronized steps to {STEPS_PER_EPOCH}",
            flush=True,
        )

    actual_epochs = 0

    try:
        for epoch in range(args.epochs):
            actual_epochs = epoch + 1
            epoch_start_time = time.time()
            epoch_comm_bytes = 0

            # 仅客户端统计 train loss
            epoch_train_loss_sum = 0.0
            epoch_train_loss_steps = 0

            # ================= Phase 1: FedAvg =================
            if not ctx.is_on_arbiter:
                local_weights = {k: v.detach().cpu() for k, v in model.client_model.state_dict().items()}
                _client_dp_upload("model", f"cnfgnn_fedavg_up_{epoch}", local_weights)

                global_weights = unwrap_payload(
                    ctx, ctx.arbiter.get(f"cnfgnn_fedavg_down_{epoch}")
                )
                global_weights = record_he_ttp_downlink(args, global_weights)
                model.client_model.load_state_dict(global_weights)

                params_cnt = sum(p.numel() for p in model.client_model.parameters())
                epoch_comm_bytes += params_cnt * 4 * 2
            else:
                _server_dp_calibrate("model", f"cnfgnn_fedavg_up_{epoch}")
                all_weights = _server_he_inputs(
                    ctx.guest.get(f"cnfgnn_fedavg_up_{epoch}"),
                    ctx.hosts.get(f"cnfgnn_fedavg_up_{epoch}"),
                )
                global_weights = {
                    key: sum(w[key] for w in all_weights) / len(all_weights)
                    for key in all_weights[0].keys()
                }

                ctx.guest.put(f"cnfgnn_fedavg_down_{epoch}", global_weights)
                ctx.hosts.put(
                    f"cnfgnn_fedavg_down_{epoch}",
                    [global_weights] * (len(all_weights) - 1)
                    if len(all_weights) > 2 else global_weights
                )

            # ================= Phase 2: Client update =================
            if not ctx.is_on_arbiter:
                model.client_model.eval()
                cached_h_spatial = []

                with torch.no_grad():
                    for step, (x, _) in enumerate(train_loader):
                        if step >= STEPS_PER_EPOCH:
                            break
                        x = x.to(args.device)
                        h_encode, _ = model.forward_client_encoder(x)
                        payload = h_encode.squeeze(0).cpu()
                        _client_dp_upload("encoding", f"cnfgnn_p2_req_{epoch}_{step}", payload)

                        h_sp = unwrap_payload(
                            ctx, ctx.arbiter.get(f"cnfgnn_p2_rep_{epoch}_{step}")
                        )
                        h_sp = record_he_ttp_downlink(args, h_sp)
                        cached_h_spatial.append(h_sp.to(args.device))

                        epoch_comm_bytes += payload.numel() * 4 + h_sp.numel() * 4

                model.client_model.train()
                for rc in range(R_c):
                    for step, (x, y) in enumerate(train_loader):
                        if step >= STEPS_PER_EPOCH:
                            break
                        x, y = x.to(args.device), y.to(args.device)
                        optimizer.zero_grad()
                        h_encode, x_reshaped = model.forward_client_encoder(x)
                        pred = model.forward_client_decoder(x_reshaped, y, h_encode, cached_h_spatial[step])
                        loss = loss_func(pred, y)
                        loss.backward()
                        optimizer.step()

                        epoch_train_loss_sum += float(loss.item())
                        epoch_train_loss_steps += 1
            else:
                model.server_model.eval()
                with torch.no_grad():
                    for step in range(STEPS_PER_EPOCH):
                        _server_dp_calibrate("encoding", f"cnfgnn_p2_req_{epoch}_{step}")
                        all_encodes = _server_he_inputs(
                            ctx.guest.get(f"cnfgnn_p2_req_{epoch}_{step}"),
                            ctx.hosts.get(f"cnfgnn_p2_req_{epoch}_{step}"),
                        )
                        h_guest = all_encodes[0]
                        curr_bs = int(h_guest.shape[0] / len(args.nodes_per[0]))

                        reshaped = [
                            h.view(curr_bs, len(args.nodes_per[k]), -1).to(args.device)
                            for k, h in enumerate(all_encodes)
                        ]
                        h_global = torch.cat(reshaped, dim=1).reshape(-1, args.hidden_dim)
                        h_spatial_flat = model.forward_server_gnn(h_global, curr_bs, args.total_nodes_global)
                        h_spatial_batch = h_spatial_flat.view(curr_bs, args.total_nodes_global, -1)

                        splits, cur = [], 0
                        for k in range(args.num_clients):
                            Ni = len(args.nodes_per[k])
                            splits.append(
                                h_spatial_batch[:, cur:cur + Ni, :].contiguous().view(-1, args.hidden_dim).cpu()
                            )
                            cur += Ni

                        ctx.guest.put(f"cnfgnn_p2_rep_{epoch}_{step}", splits[0])
                        ctx.hosts.put(
                            f"cnfgnn_p2_rep_{epoch}_{step}",
                            splits[1:] if len(splits) > 2 else splits[1]
                        )

            # ================= Phase 3: Server update =================
            if not ctx.is_on_arbiter:
                model.client_model.eval()
                for step, (x, _) in enumerate(train_loader):
                    if step >= STEPS_PER_EPOCH:
                        break
                    x = x.to(args.device)
                    with torch.no_grad():
                        h_encode, _ = model.forward_client_encoder(x)
                    payload = h_encode.squeeze(0).cpu()
                    _client_dp_upload("encoding", f"cnfgnn_p3_hc_{epoch}_{step}", payload)
                    epoch_comm_bytes += payload.numel() * 4

                model.client_model.train()
                for rs in range(R_s):
                    for step, (x, y) in enumerate(train_loader):
                        if step >= STEPS_PER_EPOCH:
                            break
                        x, y = x.to(args.device), y.to(args.device)

                        h_sp = unwrap_payload(
                            ctx, ctx.arbiter.get(f"cnfgnn_p3_hsp_{epoch}_{rs}_{step}")
                        )
                        h_spatial = record_he_ttp_downlink(args, h_sp).to(args.device).requires_grad_()

                        with torch.no_grad():
                            h_encode, x_reshaped = model.forward_client_encoder(x)

                        # The revised monitoring message uses a public zero
                        # decoder-start token rather than the private target.
                        monitor_pred = model.forward_client_decoder(
                            x_reshaped, torch.zeros_like(y), h_encode, h_spatial
                        )
                        capture_revised_quantized_prediction(
                            ctx, args, f"cnfgnn_prediction_{epoch}_{rs}_{step}",
                            prediction=monitor_pred, model_state_dict=model.state_dict(),
                        )
                        pred = model.forward_client_decoder(x_reshaped, y, h_encode, h_spatial)
                        loss = loss_func(pred, y)
                        loss.backward()

                        grad_payload = h_spatial.grad.detach().cpu()
                        _client_dp_upload("spatial_gradient", f"cnfgnn_p3_gsp_{epoch}_{rs}_{step}", grad_payload)
                        epoch_comm_bytes += h_spatial.numel() * 4 + grad_payload.numel() * 4
            else:
                cached_h_global = []
                for step in range(STEPS_PER_EPOCH):
                    all_encodes = _server_he_inputs(
                        ctx.guest.get(f"cnfgnn_p3_hc_{epoch}_{step}"),
                        ctx.hosts.get(f"cnfgnn_p3_hc_{epoch}_{step}"),
                    )
                    h_guest = all_encodes[0]
                    curr_bs = int(h_guest.shape[0] / len(args.nodes_per[0]))
                    reshaped = [
                        h.view(curr_bs, len(args.nodes_per[k]), -1).to(args.device)
                        for k, h in enumerate(all_encodes)
                    ]
                    cached_h_global.append((torch.cat(reshaped, dim=1).reshape(-1, args.hidden_dim), curr_bs))

                model.server_model.train()
                for rs in range(R_s):
                    for step in range(STEPS_PER_EPOCH):
                        optimizer.zero_grad()
                        h_global, curr_bs = cached_h_global[step]
                        h_global = h_global.detach()

                        h_spatial_flat = model.forward_server_gnn(h_global, curr_bs, args.total_nodes_global)
                        h_spatial_batch = h_spatial_flat.view(curr_bs, args.total_nodes_global, -1)

                        splits, cur = [], 0
                        for k in range(args.num_clients):
                            Ni = len(args.nodes_per[k])
                            splits.append(
                                h_spatial_batch[:, cur:cur + Ni, :].contiguous().view(-1, args.hidden_dim).cpu()
                            )
                            cur += Ni

                        ctx.guest.put(f"cnfgnn_p3_hsp_{epoch}_{rs}_{step}", splits[0])
                        ctx.hosts.put(
                            f"cnfgnn_p3_hsp_{epoch}_{rs}_{step}",
                            splits[1:] if len(splits) > 2 else splits[1]
                        )

                        _server_dp_calibrate("spatial_gradient", f"cnfgnn_p3_gsp_{epoch}_{rs}_{step}")
                        reshaped_grads = [
                            g.view(curr_bs, len(args.nodes_per[k]), -1).to(args.device)
                            for k, g in enumerate(_server_he_inputs(
                                ctx.guest.get(f"cnfgnn_p3_gsp_{epoch}_{rs}_{step}"),
                                ctx.hosts.get(f"cnfgnn_p3_gsp_{epoch}_{rs}_{step}"),
                            ))
                        ]
                        g_global = torch.cat(reshaped_grads, dim=1).reshape(-1, args.hidden_dim)
                        h_spatial_flat.backward(g_global)
                        optimizer.step()

            # ================= Validation & Early Stop =================
            if not ctx.is_on_arbiter:
                total_train_time += (time.time() - epoch_start_time)
                val_start = time.time()
                model.client_model.eval()
                val_err_sum, val_count = 0.0, 0

                with torch.no_grad():
                    for step, (x_val, y_val) in enumerate(val_loader):
                        if step >= VAL_STEPS:
                            break
                        x_val, y_val = x_val.to(args.device), y_val.to(args.device)

                        h_encode, x_reshaped = model.forward_client_encoder(x_val)
                        payload = h_encode.squeeze(0).cpu()
                        _client_dp_upload("encoding", f"cnfgnn_val_req_{epoch}_{step}", payload)

                        h_sp = unwrap_payload(
                            ctx, ctx.arbiter.get(f"cnfgnn_val_rep_{epoch}_{step}")
                        )
                        h_sp = record_he_ttp_downlink(args, h_sp)

                        pred_val = model.forward_client_decoder(
                            x_reshaped,
                            None,
                            h_encode,
                            h_sp.to(args.device)
                        )

                        val_err_sum += float(torch.abs(pred_val - y_val).sum().item())
                        val_count += int(y_val.numel())
                        epoch_comm_bytes += payload.numel() * 4 + h_sp.numel() * 4

                total_val_time += (time.time() - val_start)

                ctx.arbiter.put(
                    f"cnfgnn_val_report_{epoch}",
                    {"err_sum": float(val_err_sum), "count": int(val_count)}
                )

                sync_info = ctx.arbiter.get(f"cnfgnn_val_sync_{epoch}")
                sync_info = unwrap_payload(ctx, sync_info)

                best_epoch = sync_info["best_epoch"]
                if sync_info["is_best"]:
                    best_client_wts = copy.deepcopy(model.client_model.state_dict())

                comm_history.append(epoch_comm_bytes)

                avg_train_loss = epoch_train_loss_sum / max(epoch_train_loss_steps, 1)
                local_val_mae = val_err_sum / max(val_count, 1)

                print(
                    f"Client {ctx.rank} | Epoch {epoch} | "
                    f"Train Loss={avg_train_loss:.6f} | "
                    f"Val MAE={local_val_mae:.4f} | "
                    f"Global MAE={sync_info['global_mae']:.4f} | "
                    f"Patience={sync_info['patience_counter']}/{sync_info['patience']}",
                    flush=True
                )

                if sync_info["should_stop"]:
                    raise EarlyStopSignal("CNFGNN early stopping triggered")

            else:
                model.server_model.eval()
                with torch.no_grad():
                    for step in range(VAL_STEPS):
                        _server_dp_calibrate("encoding", f"cnfgnn_val_req_{epoch}_{step}")
                        all_encodes = _server_he_inputs(
                            ctx.guest.get(f"cnfgnn_val_req_{epoch}_{step}"),
                            ctx.hosts.get(f"cnfgnn_val_req_{epoch}_{step}"),
                        )
                        h_guest = all_encodes[0]
                        curr_bs = int(h_guest.shape[0] / len(args.nodes_per[0]))
                        reshaped = [
                            h.view(curr_bs, len(args.nodes_per[k]), -1).to(args.device)
                            for k, h in enumerate(all_encodes)
                        ]
                        h_global = torch.cat(reshaped, dim=1).reshape(-1, args.hidden_dim)

                        h_spatial_batch = model.forward_server_gnn(
                            h_global, curr_bs, args.total_nodes_global
                        ).view(curr_bs, args.total_nodes_global, -1)

                        splits, cur = [], 0
                        for k in range(args.num_clients):
                            Ni = len(args.nodes_per[k])
                            splits.append(
                                h_spatial_batch[:, cur:cur + Ni, :].contiguous().view(-1, args.hidden_dim).cpu()
                            )
                            cur += Ni

                        ctx.guest.put(f"cnfgnn_val_rep_{epoch}_{step}", splits[0])
                        ctx.hosts.put(
                            f"cnfgnn_val_rep_{epoch}_{step}",
                            splits[1:] if len(splits) > 2 else splits[1]
                        )

                val_g = ctx.guest.get(f"cnfgnn_val_report_{epoch}")
                val_h = ctx.hosts.get(f"cnfgnn_val_report_{epoch}")
                all_reports = [val_g] + ([val_h] if not isinstance(val_h, list) else val_h)

                global_mae = sum(r["err_sum"] for r in all_reports) / max(sum(r["count"] for r in all_reports), 1)

                is_best, should_stop = False, False
                if global_mae < global_best_norm_mae - 1e-4:
                    global_best_norm_mae, global_best_epoch = global_mae, epoch
                    best_server_wts = copy.deepcopy(model.server_model.state_dict())
                    patience_counter, is_best = 0, True
                else:
                    patience_counter += 1
                    should_stop = patience_counter >= patience

                sync_payload = {
                    "global_mae": float(global_mae),
                    "is_best": is_best,
                    "should_stop": should_stop,
                    "best_epoch": int(global_best_epoch),
                    "patience_counter": int(patience_counter),
                    "patience": int(patience)
                }

                print(
                    f"[CNFGNN Server] Epoch {epoch} | "
                    f"Global MAE={global_mae:.4f} | "
                    f"Best Epoch={global_best_epoch} | "
                    f"Patience={patience_counter}/{patience}",
                    flush=True
                )

                ctx.guest.put(f"cnfgnn_val_sync_{epoch}", sync_payload)
                ctx.hosts.put(
                    f"cnfgnn_val_sync_{epoch}",
                    [copy.deepcopy(sync_payload) for _ in range(args.num_clients - 1)]
                )

                if should_stop:
                    print(f"Rank {ctx.rank} (Server): 🛑 触发全局早停！", flush=True)
                    raise EarlyStopSignal("CNFGNN Server early stopping triggered")

    except EarlyStopSignal:
        pass

    # ================= Test =================
    if not ctx.is_on_arbiter:
        if best_client_wts is not None:
            model.client_model.load_state_dict(best_client_wts)

        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        ctx.arbiter.put("cnfgnn_test_steps", len(test_loader))

        model.client_model.eval()
        test_start_t = time.time()
        test_mae, test_rmse, test_mape = 0.0, 0.0, 0.0
        total_test_elements, valid_mape_count = 0, 0

        with torch.no_grad():
            for step, (x, y) in enumerate(test_loader):
                x, y = x.to(args.device), y.to(args.device)

                h_encode, x_reshaped = model.forward_client_encoder(x)
                payload = h_encode.squeeze(0).cpu()
                _client_dp_upload("encoding", f"cnfgnn_test_req_{step}", payload)

                h_sp = unwrap_payload(
                    ctx, ctx.arbiter.get(f"cnfgnn_test_rep_{step}")
                )
                h_sp = record_he_ttp_downlink(args, h_sp)

                pred = model.forward_client_decoder(
                    x_reshaped,
                    None,
                    h_encode,
                    h_sp.to(args.device)
                )

                y_real = scaler.inverse_transform(y).detach().cpu().numpy()
                pred_real = scaler.inverse_transform(pred).detach().cpu().numpy()

                test_mae += np.sum(np.abs(y_real - pred_real))
                test_rmse += np.sum((y_real - pred_real) ** 2)
                total_test_elements += y_real.size

                mask = y_real > 0.5
                if np.sum(mask) > 0:
                    test_mape += np.sum(np.abs(y_real[mask] - pred_real[mask]) / y_real[mask])
                    valid_mape_count += np.sum(mask)

        acc_mae = test_mae / total_test_elements
        acc_rmse = math.sqrt(test_rmse / total_test_elements)
        acc_mape = (test_mape / valid_mape_count) * 100 if valid_mape_count > 0 else 0.0

        eff_test_time = time.time() - test_start_t
        eff_comm_size_mb = round(sum(comm_history[:actual_epochs]) / (1024 * 1024), 4)

        # FLOPs 探测
        eff_flops = 0.0
        try:
            from thop import profile

            class CNFGNN_Wrapper(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, x, dh):
                    he, xr = self.m.forward_client_encoder(x)
                    return self.m.forward_client_decoder(xr, None, he, dh)

            dummy_x, _ = next(iter(val_loader))
            dummy_x = dummy_x.to(args.device)
            dummy_h = torch.zeros(
                dummy_x.shape[0] * len(args.nodes_per[ctx.rank]), args.hidden_dim
            ).to(args.device)
            flops, _ = profile(
                CNFGNN_Wrapper(model).to(args.device),
                inputs=(dummy_x, dummy_h),
                verbose=False
            )
            eff_flops = round(flops / 1e9, 4)
        except Exception:
            eff_flops = 0.0

        return (
            best_epoch,
            acc_mae,
            acc_rmse,
            acc_mape,
            total_train_time,
            total_val_time / max(actual_epochs, 1),
            eff_test_time,
            eff_comm_size_mb,
            actual_epochs,
            eff_flops
        )
    else:
        if best_server_wts is not None:
            model.server_model.load_state_dict(best_server_wts)
        model.server_model.eval()

        s_guest = ctx.guest.get("cnfgnn_test_steps")
        s_hosts = ctx.hosts.get("cnfgnn_test_steps")
        TEST_STEPS = min([s_guest] + ([s_hosts] if not isinstance(s_hosts, list) else s_hosts))

        with torch.no_grad():
            for step in range(TEST_STEPS):
                _server_dp_calibrate("encoding", f"cnfgnn_test_req_{step}")
                all_encodes = _server_he_inputs(
                    ctx.guest.get(f"cnfgnn_test_req_{step}"),
                    ctx.hosts.get(f"cnfgnn_test_req_{step}"),
                )
                h_guest = all_encodes[0]
                curr_bs = int(h_guest.shape[0] / len(args.nodes_per[0]))
                reshaped = [
                    h.view(curr_bs, len(args.nodes_per[k]), -1).to(args.device)
                    for k, h in enumerate(all_encodes)
                ]
                h_global = torch.cat(reshaped, dim=1).reshape(-1, args.hidden_dim)

                h_spatial_batch = model.forward_server_gnn(
                    h_global, curr_bs, args.total_nodes_global
                ).view(curr_bs, args.total_nodes_global, -1)

                splits, cur = [], 0
                for k in range(args.num_clients):
                    Ni = len(args.nodes_per[k])
                    splits.append(
                        h_spatial_batch[:, cur:cur + Ni, :].contiguous().view(-1, args.hidden_dim).cpu()
                    )
                    cur += Ni

                ctx.guest.put(f"cnfgnn_test_rep_{step}", splits[0])
                ctx.hosts.put(
                    f"cnfgnn_test_rep_{step}",
                    splits[1:] if len(splits) > 2 else splits[1]
                )
