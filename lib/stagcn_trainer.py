import copy
import os
import signal
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import data.dividing as dividing
from lib.load_dataset import load_dataset
from lib.utils import evaluate_client_model, synchronize_cuda_for_timing
from privacy.protection import l2_norm
from privacy.runtime_protection import (
    protected_arbiter_put, record_he_ttp_downlink, unprotect_he_ttp_payload,
)
from privacy.attack_trace import capture_he_ttp_insider_upper_bound


def _total_nodes_for(dataset_name):
    total_nodes_map = {
        "PeMS03": 358,
        "PeMS04": 307,
        "PeMSD7": 228,
        "PeMS07": 883,
        "PeMS08": 170,
        "TaxiBJ": 1024,
        "TaxiNYC": 75,
        "METR-LA": 207,
        "PEMS-BAY": 325,
    }
    for key, value in total_nodes_map.items():
        if key in dataset_name:
            return value
    return 307


def _learnable_state_dict(model):
    param_names = {name for name, _ in model.named_parameters()}
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if name in param_names
    }


def _filter_state_for_model(model, weights_dict, device):
    param_names = {name for name, _ in model.named_parameters()}
    current = model.state_dict()
    return {
        name: tensor.to(device)
        for name, tensor in weights_dict.items()
        if name in param_names and name in current and current[name].shape == tensor.shape
    }


def _normalize_neighbor_candidates(payload):
    if payload is None:
        return []
    if isinstance(payload, dict):
        return [("neighbor", payload)]
    if (
        isinstance(payload, (list, tuple))
        and len(payload) == 2
        and isinstance(payload[1], dict)
        and not isinstance(payload[0], (list, tuple, dict))
    ):
        return [(payload[0], payload[1])]
    if isinstance(payload, list):
        if len(payload) == 0:
            return []
        if all(
            isinstance(item, (list, tuple))
            and len(item) == 2
            and isinstance(item[1], dict)
            for item in payload
        ):
            return [(item[0], item[1]) for item in payload]
        if all(isinstance(item, dict) for item in payload):
            return [(idx, item) for idx, item in enumerate(payload)]
        if len(payload) == 1:
            return _normalize_neighbor_candidates(payload[0])
    return []


def _select_host_payload_for_rank(payload, rank, num_clients):
    if rank <= 0 or not isinstance(payload, list):
        return payload
    if len(payload) != max(num_clients - 1, 0):
        return payload
    if all(
        isinstance(item, list)
        and not (
            len(item) == 2
            and isinstance(item[1], dict)
            and not isinstance(item[0], (list, tuple, dict))
        )
        for item in payload
    ):
        host_idx = rank - 1
        if 0 <= host_idx < len(payload):
            return payload[host_idx]
    return payload


def train_stagcn_ec_task(ctx, args, get_setting_func):
    """STAGCN-EC federated edge training with neighbor pre-training selection."""
    from fate_main import EarlyStopSignal, ExplicitEarlyStopper

    total_nodes = _total_nodes_for(args.dataset_name)

    if ctx.is_on_arbiter:
        metis_partition = getattr(args, "nodes_per", None)
        if not metis_partition:
            split_var_name = f"{args.dataset_name}FLOW_{args.num_clients}p_metis"
            try:
                metis_partition = getattr(dividing, split_var_name)
            except AttributeError as exc:
                raise ValueError(f"Missing STAGCN-EC split in data.dividing: {split_var_name}") from exc
        print(f"[STAGCN Server] loaded partition strategy={getattr(args, 'partition_strategy', 'legacy')}", flush=True)
        _, _, _, full_edge_index, _ = load_dataset(
            dataset_name=args.dataset_name,
            feature_type=args.feature_type,
            normalizer=args.normalizer,
            T_in=args.t_in,
            T_out=args.t_out,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            return_edge_index=True,
            device="cpu",
            selected_nodes=list(range(total_nodes)),
        )

        node_to_client = {}
        for client_idx, nodes in enumerate(metis_partition):
            for node in nodes:
                node_to_client[int(node)] = client_idx

        adjacency_matrix = np.zeros((args.num_clients, args.num_clients), dtype=np.float32)
        edge_array = full_edge_index.cpu().numpy()
        for edge_idx in range(edge_array.shape[1]):
            src, dst = int(edge_array[0, edge_idx]), int(edge_array[1, edge_idx])
            if src in node_to_client and dst in node_to_client:
                c_src, c_dst = node_to_client[src], node_to_client[dst]
                if c_src != c_dst:
                    adjacency_matrix[c_src, c_dst] = 1.0
                    adjacency_matrix[c_dst, c_src] = 1.0

        adj_matrix_list = adjacency_matrix.tolist()
        ctx.guest.put("adj_matrix", adj_matrix_list)
        ctx.hosts.put("adj_matrix", [adj_matrix_list] * (args.num_clients - 1))

        print("[STAGCN Server] routing all neighbor parameter candidates", flush=True)
        init_tag = "init_weights"
        if args.protection == "dp":
            configured_clip = float(getattr(args, "dp_clip_norm", 0.0))
            if configured_clip <= 0:
                norm_guest = float(ctx.guest.get("stagcn_dp_init_norm"))
                norm_hosts = ctx.hosts.get("stagcn_dp_init_norm")
                norm_hosts = norm_hosts if isinstance(norm_hosts, list) else [norm_hosts]
                configured_clip = float(np.quantile([norm_guest] + [float(value) for value in norm_hosts], 0.9))
                ctx.guest.put("stagcn_dp_init_clip", configured_clip)
                ctx.hosts.put("stagcn_dp_init_clip", [configured_clip] * len(norm_hosts))
                print(f"[DPCalibration] STAGCN arbiter clip_norm={configured_clip:.8f}", flush=True)
            args.dp_clip_norm = configured_clip
            init_tag = "stagcn_dp_init_weights"
        w_guest = unprotect_he_ttp_payload(args, ctx.guest.get(init_tag))
        w_hosts = ctx.hosts.get(init_tag)
        if not isinstance(w_hosts, list):
            w_hosts = [w_hosts]
        w_hosts = [unprotect_he_ttp_payload(args, payload) for payload in w_hosts]
        all_weights = [w_guest] + w_hosts

        guest_neighbor_weights = []
        host_neighbor_weights = []
        for client_idx in range(args.num_clients):
            neighbors = [
                int(neighbor_idx)
                for neighbor_idx in np.where(adjacency_matrix[client_idx] == 1)[0]
                if all_weights[int(neighbor_idx)] is not None
            ]
            payload = [(neighbor_idx, all_weights[neighbor_idx]) for neighbor_idx in neighbors]
            if client_idx == 0:
                guest_neighbor_weights = payload
            else:
                host_neighbor_weights.append(payload)

        ctx.guest.put("neighbor_weights", guest_neighbor_weights)
        if len(host_neighbor_weights) == 1:
            ctx.hosts.put("neighbor_weights", host_neighbor_weights[0])
        else:
            ctx.hosts.put("neighbor_weights", host_neighbor_weights)

        stopper = ExplicitEarlyStopper(
            ctx,
            args,
            patience=getattr(args, "patience", 50),
            min_delta=1e-4,
        )
        try:
            for _ in range(args.epochs):
                if stopper.check_and_sync(eval_loss=0.0):
                    print("[STAGCN Server] received global early-stop signal", flush=True)
                    break
        except Exception:
            pass

        return None, None, None, None, None, None

    print(f"[STAGCN Client {ctx.rank}] receiving adjacency and neighbor candidates", flush=True)
    # STAGCN-EC exchanges neighbor candidates once before local training.
    # Keep this separate from the repeated edge-training rounds: multiplying
    # it by the early-stop round count is mathematically incorrect.
    initialization_started = time.perf_counter()
    initialization_comm_bytes = 0
    _ = ctx.arbiter.get("adj_matrix")

    train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = get_setting_func(ctx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    local_init_weights = _learnable_state_dict(model)
    if args.protection == "dp":
        clip_norm = float(getattr(args, "dp_clip_norm", 0.0))
        if clip_norm <= 0:
            ctx.arbiter.put("stagcn_dp_init_norm", float(l2_norm(local_init_weights).item()))
            clip_norm = ctx.arbiter.get("stagcn_dp_init_clip")
            if isinstance(clip_norm, (list, tuple)):
                clip_norm = clip_norm[0]
            clip_norm = float(clip_norm)
            args.dp_clip_norm = clip_norm
            print(f"[DPCalibration] STAGCN rank={ctx.rank} clip_norm={clip_norm:.8f}", flush=True)
        protected_arbiter_put(ctx, args, "stagcn_dp_init_weights", local_init_weights, clip_norm=clip_norm)
    else:
        if args.protection == "he":
            protected_arbiter_put(ctx, args, "init_weights", local_init_weights)
        else:
            ctx.arbiter.put("init_weights", local_init_weights)
    initialization_comm_bytes += sum(
        int(tensor.numel() * tensor.element_size())
        for tensor in local_init_weights.values() if torch.is_tensor(tensor)
    )
    raw_neighbor_payload = ctx.arbiter.get("neighbor_weights")
    rank_payload = _select_host_payload_for_rank(
        raw_neighbor_payload,
        ctx.rank,
        args.num_clients,
    )
    # A host can receive a list containing all host payloads.  Count only the
    # payload addressed to this rank, not that entire routing list.
    rank_payload = record_he_ttp_downlink(
        args, rank_payload, tag="neighbor_weights",
    )
    def _payload_bytes(payload):
        if torch.is_tensor(payload):
            return int(payload.numel() * payload.element_size())
        if isinstance(payload, dict):
            return sum(_payload_bytes(item) for item in payload.values())
        if isinstance(payload, (tuple, list)):
            return sum(_payload_bytes(item) for item in payload)
        return 0
    initialization_comm_bytes += _payload_bytes(rank_payload)
    neighbor_candidates = _normalize_neighbor_candidates(rank_payload)
    print(
        f"[STAGCN Client {ctx.rank}] neighbor_candidates={len(neighbor_candidates)}",
        flush=True,
    )

    if neighbor_candidates:
        print(
            f"[STAGCN Client {ctx.rank}] evaluating {len(neighbor_candidates)} neighbor candidates",
            flush=True,
        )

        def eval_weights(weights_dict, tag):
            temp_model = copy.deepcopy(model)
            filtered_weights = _filter_state_for_model(temp_model, weights_dict, args.device)
            temp_model.load_state_dict(filtered_weights, strict=False)

            temp_optimizer = torch.optim.Adam(temp_model.parameters(), lr=args.lr)
            temp_model.train()
            loss_sum = 0.0
            steps = 0
            eval_steps = int(os.environ.get("STAGCN_PRETRAIN_STEPS", "10"))
            for batch_idx, (x, y) in enumerate(train_loader):
                if batch_idx >= eval_steps:
                    break
                x, y = x.to(args.device), y.to(args.device)
                temp_optimizer.zero_grad()
                pred = temp_model(x)
                if pred.shape != y.shape:
                    pred = pred.reshape_as(y)
                loss = loss_func(pred, y)
                loss.backward()
                temp_optimizer.step()
                loss_sum += float(loss.item())
                steps += 1
            avg_loss = loss_sum / max(steps, 1)
            return avg_loss, _learnable_state_dict(temp_model)

        loss_local, local_updated_weights = eval_weights(local_init_weights, "local")
        choices = [("local", loss_local, local_updated_weights)]
        for source_rank, candidate_weights in neighbor_candidates:
            loss_neighbor, neighbor_updated_weights = eval_weights(
                candidate_weights,
                f"neighbor{source_rank}",
            )
            choices.append((f"neighbor{source_rank}", loss_neighbor, neighbor_updated_weights))

        selected_label, selected_loss, selected_weights = min(choices, key=lambda item: item[1])
        print(
            f"[STAGCN Client {ctx.rank}] pretrain_select={selected_label} "
            f"loss={selected_loss:.6f} candidates="
            + ",".join(f"{label}:{loss:.6f}" for label, loss, _ in choices),
            flush=True,
        )
        model.load_state_dict(
            _filter_state_for_model(model, selected_weights, args.device),
            strict=False,
        )

    initialization_seconds = time.perf_counter() - initialization_started
    print(
        f"[ProtocolTiming] model=STAGCN-EC rank={ctx.rank} "
        f"phase=initialization seconds={initialization_seconds:.6f} "
        f"plain_payload_bytes={initialization_comm_bytes}",
        flush=True,
    )
    print(f"[STAGCN Client {ctx.rank}] start local edge training", flush=True)

    total_train_time = 0.0
    total_val_time = 0.0
    epochs_run = 0
    best_mae = float("inf")
    best_epoch = -1
    best_model_wts = None

    stopper = ExplicitEarlyStopper(
        ctx,
        args,
        patience=getattr(args, "patience", 50),
        min_delta=1e-4,
    )

    try:
        for epoch in range(args.epochs):
            epochs_run += 1

            train_start = time.time()
            model.train()
            for x, y in train_loader:
                x, y = x.to(args.device), y.to(args.device)
                optimizer.zero_grad()
                capture_client_state_oracle = (
                    bool(os.environ.get("PRIVACY_TRACE_SAVE_STAGCN_CLIENT_STATE_ORACLE", ""))
                    and bool(getattr(args, "privacy_trace_save_insider", False))
                    and ctx.rank == int(getattr(args, "privacy_trace_client", 0))
                    and int(getattr(args, "_privacy_he_ttp_insider_upper_records", 0)) < 1
                )
                if capture_client_state_oracle:
                    round_start_weights = _learnable_state_dict(model)
                pred = model(x)
                if pred.shape != y.shape:
                    pred = pred.reshape_as(y)
                loss = loss_func(pred, y)
                loss.backward()
                optimizer.step()
                if capture_client_state_oracle:
                    local_weights = _learnable_state_dict(model)
                    capture_he_ttp_insider_upper_bound(
                        ctx, args, "stagcn_ec_first_local_adam_update",
                        observed_leak={
                            key: local_weights[key] - round_start_weights[key]
                            for key in round_start_weights
                        },
                        model_state_dict=round_start_weights,
                        leak_type="model_update",
                        observer="malicious_client_state_oracle_upper_bound",
                        threat_model=(
                            "STAGCN-EC client-state oracle upper bound: attacker receives "
                            "the target client's first local Adam update, round-start model, "
                            "and the target label. This is not HE-TTP Arbiter-visible."
                        ),
                    )
            total_train_time += time.time() - train_start

            val_start = time.time()
            model.eval()
            val_abs = 0.0
            val_elements = 0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(args.device), y_val.to(args.device)
                    pred_val = model(x_val)
                    if pred_val.shape != y_val.shape:
                        pred_val = pred_val.reshape_as(y_val)
                    val_abs += torch.abs(pred_val - y_val).sum().item()
                    val_elements += y_val.numel()
            final_mae = float(val_abs / max(val_elements, 1))
            total_val_time += time.time() - val_start

            if final_mae < best_mae:
                best_mae = final_mae
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())

            if stopper.check_and_sync(final_mae):
                print(f"Rank {ctx.rank}: received global early-stop signal", flush=True)
                raise EarlyStopSignal("STAGCN-EC early stop")
    except EarlyStopSignal:
        print(f"Rank {ctx.rank}: leaving training loop for final test", flush=True)

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    print(f"[STAGCN Client {ctx.rank}] evaluating test set", flush=True)
    synchronize_cuda_for_timing(args.device)
    test_start = time.perf_counter()
    _, final_test_mae, final_test_rmse, final_test_mape = evaluate_client_model(
        model,
        test_loader,
        loss_func,
        scaler,
        args.device,
    )
    synchronize_cuda_for_timing(args.device)
    eff_test_time = time.perf_counter() - test_start
    print(
        f"[TestTimingAudit] model=STAGCN-EC rank={ctx.rank} "
        f"batches={len(test_loader)} seconds={eff_test_time:.6f}",
        flush=True,
    )
    avg_val_time = total_val_time / max(epochs_run, 1)

    eff_flops = 0.0
    try:
        from thop import profile

        dummy_x, _ = next(iter(test_loader))
        dummy_x = dummy_x.to(args.device)
        flops, _ = profile(model, inputs=(dummy_x,), verbose=False)
        eff_flops = round(flops / 1e9, 4)
    except Exception as exc:
        print(f"[STAGCN Client {ctx.rank}] FLOPs profiling failed: {exc}", flush=True)

    if ctx.is_on_guest and (
        getattr(args, "force_exit_after_run", False)
        or os.environ.get("STAGCN_EC_KILL_PARENT", "0") == "1"
    ):
        print("[STAGCN Client 0] force-exit cleanup requested", flush=True)
        time.sleep(2)
        os.kill(os.getppid(), signal.SIGTERM)

    # STAGCN-EC performs encrypted neighbour discovery and candidate
    # pre-training once before the first local epoch.  It is protocol work
    # required by HE-TTP, hence it belongs in the measured run rather than
    # being silently excluded from its one-round efficiency record.
    total_eff_train_time = initialization_seconds + total_train_time
    args._he_init_time_s = float(initialization_seconds)
    args._he_train_compute_time_s = float(total_train_time)

    return (
        best_epoch,
        round(final_test_mae, 4),
        round(final_test_rmse, 4),
        round(final_test_mape, 4),
        round(total_eff_train_time, 2),
        round(avg_val_time, 4),
        round(eff_test_time, 4),
        initialization_comm_bytes,
        eff_flops,
        epochs_run,
    )
