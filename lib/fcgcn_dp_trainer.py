"""Explicit, auditable DP loop shared by model-update FedAvg baselines."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.utils import (
    EarlyStopSignal,
    ExplicitEarlyStopper,
    align_prediction_and_target,
    unpack_spatiotemporal_batch,
    synchronize_cuda_for_timing,
)
from lib.plain_protocol_eval import evaluate_with_plain_client
from privacy.protection import l2_norm
from privacy.runtime_protection import protected_arbiter_put
from privacy.attack_trace import (
    capture_he_sa_aggregate,
    capture_he_sa_server_aggregate_hidden_term,
    capture_revised_quantized_prediction,
)


def _floating_trainable_state(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        # ``adj`` is a fixed local graph buffer, never a federated update.
        if name != "adj" and value.is_floating_point()
    }


def _prediction_and_target(model, x, y):
    prediction = model(x)
    return align_prediction_and_target(prediction, y)


def _unpack_xy(batch):
    """Accept both ordinary ``(x, y)`` and grid-data multi-field batches.

    FedGRU on TaxiBJ/TaxiNYC/BikeNYC receives temporal/external auxiliary
    fields from the grid dataset.  The shared DP loop only needs model input
    and target, whose positions are normalized by this project helper.
    """
    return unpack_spatiotemporal_batch(batch)


def train_fedavg_dp_task(ctx, args, get_setting_func):
    """Run a model-delta FedAvg loop with either DP or real CKKS HE-SA.

    The HE branch deliberately exists beside the DP branch instead of relying
    on FATE's ``secure_aggregate`` setting: clients encrypt their *delta*
    locally, the arbiter adds ciphertext chunks, and only that aggregate is
    decrypted.  This loop is consequently suitable only for baselines whose
    federated dependency is a shared model-update mean.
    """
    label = str(args.model)
    is_he = str(getattr(args, "protection", "plain")).lower() == "he"
    mode = "he" if is_he else "dp"
    tag_prefix = f"{label.lower()}_{mode}"
    if ctx.is_on_arbiter:
        steps = 0
    else:
        (
            train_set, val_set, test_set, model, optimizer, loss_func,
            scheduler, train_args, fed_args, scaler, _,
        ) = get_setting_func(ctx)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        steps = len(train_loader)
        ctx.arbiter.put(f"{tag_prefix}_steps", steps)
        best_norm_mae = float("inf")
        best_epoch = -1
        best_model_wts = None
        actual_epochs = 0
        total_train_time = 0.0
        total_val_time = 0.0
        stopper = ExplicitEarlyStopper(ctx, args, patience=50, min_delta=1e-5)

    he_context = None
    he_arbiter_context = None
    he_upload_bytes = 0
    he_download_bytes = 0
    if is_he:
        from privacy.ckks_backend import (
            ciphertext_bytes as ckks_ciphertext_bytes,
            decrypt_tree as ckks_decrypt_tree,
            encrypt_tree as ckks_encrypt_tree,
            export_public_context as ckks_export_public_context,
            generate_context as ckks_generate_context,
            homomorphic_sum_tree as ckks_homomorphic_sum_tree,
            import_context as ckks_import_context,
        )
        if str(getattr(args, "he_backend", "auto")).lower() not in ("auto", "he_sa"):
            raise ValueError(f"{label} supports HE-SA only in this model-update loop.")
        key_tag = f"__{label.lower()}_he_sa_ckks_context"
        if ctx.is_on_arbiter:
            he_arbiter_context = ckks_generate_context(
                int(args.he_ckks_poly_modulus_degree), int(args.he_ckks_scale_bits),
            )
            public_context = ckks_export_public_context(he_arbiter_context)
            ctx.guest.put(key_tag, public_context)
            ctx.hosts.put(key_tag, public_context)
            print(f"[HESA] {label} arbiter generated packed CKKS context", flush=True)
        else:
            he_context = ckks_import_context(ctx.arbiter.get(key_tag))
            print(f"[HESA] {label} rank={ctx.rank} received packed CKKS public context", flush=True)

    # Delta aggregation is valid only when all clients start each round from
    # the same shared state.  Public model initialisation contains no training
    # data, so it is intentionally not encrypted.
    if is_he:
        init_tag = f"{tag_prefix}_public_init"
        if ctx.is_on_arbiter:
            guest_init = ctx.guest.get(init_tag)
            host_init = ctx.hosts.get(init_tag)
            host_init = host_init if isinstance(host_init, list) else [host_init]
            canonical = guest_init
            ctx.guest.put(init_tag, canonical)
            ctx.hosts.put(init_tag, [canonical] * len(host_init))
        else:
            ctx.arbiter.put(init_tag, _floating_trainable_state(model))
            canonical = ctx.arbiter.get(init_tag)
            if isinstance(canonical, list):
                canonical = canonical[0]
            model.load_state_dict(canonical, strict=False)

    dp_clip_norm = None
    if ctx.is_on_arbiter:
        guest_steps = int(ctx.guest.get(f"{tag_prefix}_steps"))
        host_steps = ctx.hosts.get(f"{tag_prefix}_steps")
        host_steps = host_steps if isinstance(host_steps, list) else [host_steps]
        steps = min([guest_steps] + [int(value) for value in host_steps])
        print(f"[{label}-DP] arbiter steps_per_round={steps}", flush=True)

    try:
        for epoch in range(args.epochs):
            if not ctx.is_on_arbiter:
                round_start = _floating_trainable_state(model)
                model.train()
                local_started = time.perf_counter()
                loss_sum = 0.0
                for _ in range(args.local_epochs):
                    for step, batch in enumerate(train_loader):
                        if step >= steps:
                            break
                        x, y = _unpack_xy(batch)
                        x, y = x.to(args.device), y.to(args.device)
                        optimizer.zero_grad()
                        prediction, y = _prediction_and_target(model, x, y)
                        capture_revised_quantized_prediction(
                            ctx, args, f"{label.lower()}_prediction_{epoch}_{step}",
                            prediction=prediction, model_state_dict=round_start,
                        )
                        loss = loss_func(prediction, y)
                        loss.backward()
                        optimizer.step()
                        loss_sum += float(loss.item())
                total_train_time += time.perf_counter() - local_started
                print(
                    f"[{label}-DP] rank={ctx.rank} round={epoch + 1} "
                    f"local_loss={loss_sum / max(steps * args.local_epochs, 1):.8f}",
                    flush=True,
                )

                model.eval()
                validation_started = time.perf_counter()
                absolute_error = squared_error = 0.0
                element_count = 0
                with torch.no_grad():
                    for batch in val_loader:
                        x, y = _unpack_xy(batch)
                        x, y = x.to(args.device), y.to(args.device)
                        prediction, y = _prediction_and_target(model, x, y)
                        diff = prediction - y
                        absolute_error += float(diff.abs().sum().item())
                        squared_error += float(diff.square().sum().item())
                        element_count += int(y.numel())
                norm_mae = absolute_error / max(element_count, 1)
                norm_rmse = math.sqrt(squared_error / max(element_count, 1))
                total_val_time += time.perf_counter() - validation_started
                print(
                    f"[{label}-DP] rank={ctx.rank} round={epoch + 1} "
                    f"norm_mae={norm_mae:.8f} norm_rmse={norm_rmse:.8f}",
                    flush=True,
                )
                if norm_mae < best_norm_mae:
                    best_norm_mae = norm_mae
                    best_epoch = epoch
                    best_model_wts = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                actual_epochs = epoch + 1
                should_stop = bool(stopper.check_and_sync(norm_mae))

                local_end = _floating_trainable_state(model)
                delta = {name: local_end[name] - round_start[name] for name in round_start}
                if not is_he and float(args.dp_clip_norm) <= 0:
                    ctx.arbiter.put(f"{tag_prefix}_delta_norm_{epoch}", float(l2_norm(delta).item()))
                    calibrated_clip = ctx.arbiter.get(f"{tag_prefix}_clip_norm_{epoch}")
                    if isinstance(calibrated_clip, (list, tuple)):
                        calibrated_clip = calibrated_clip[0]
                    args.dp_clip_norm = float(calibrated_clip)
                    print(f"[DPCalibration] {label} rank={ctx.rank} epoch={epoch + 1} clip_norm={args.dp_clip_norm:.8f}", flush=True)

                protection_started = time.perf_counter()
                if is_he:
                    capture_he_sa_server_aggregate_hidden_term(
                        ctx, args, f"{tag_prefix}_delta_{epoch}", payload=delta,
                        model_state_dict=round_start, aggregation_weight=1.0,
                        leak_type="model_update",
                    )
                    encrypted_delta = ckks_encrypt_tree(
                        delta, he_context,
                        slot_count=int(args.he_ckks_poly_modulus_degree) // 2,
                    )
                    upload_bytes = ckks_ciphertext_bytes(encrypted_delta)
                    he_upload_bytes += upload_bytes
                    ctx.arbiter.put(f"{tag_prefix}_delta_{epoch}", encrypted_delta)
                    ctx.arbiter.put(f"{tag_prefix}_upload_bytes_{epoch}", upload_bytes)
                    print(
                        f"[HESA] {label} rank={ctx.rank} round={epoch + 1} "
                        f"encrypted_upload_bytes={upload_bytes}", flush=True,
                    )
                else:
                    protected_arbiter_put(ctx, args, f"{tag_prefix}_delta_{epoch}", delta)
                total_train_time += time.perf_counter() - protection_started
                global_delta = ctx.arbiter.get(f"{tag_prefix}_global_delta_{epoch}")
                if isinstance(global_delta, list):
                    global_delta = global_delta[0]
                if is_he:
                    he_download_bytes += sum(value.numel() * value.element_size() for value in global_delta.values())
                    he_seconds = ctx.arbiter.get(f"{tag_prefix}_arbiter_seconds_{epoch}")
                    while isinstance(he_seconds, (list, tuple)):
                        he_seconds = he_seconds[0]
                    total_train_time += float(he_seconds)
                updated = {name: round_start[name] + value.cpu() for name, value in global_delta.items()}
                model.load_state_dict(updated, strict=False)
                if ctx.is_on_guest:
                    ctx.arbiter.put(f"{tag_prefix}_stop_{epoch}", should_stop)
                if should_stop:
                    print(f"[{label}-DP] rank={ctx.rank} early-stop at round {epoch + 1}", flush=True)
                    raise EarlyStopSignal("global early stop")
            else:
                if not is_he and dp_clip_norm is None:
                    guest_norm = float(ctx.guest.get(f"{tag_prefix}_delta_norm_{epoch}"))
                    host_norms = ctx.hosts.get(f"{tag_prefix}_delta_norm_{epoch}")
                    host_norms = host_norms if isinstance(host_norms, list) else [host_norms]
                    dp_clip_norm = float(np.quantile([guest_norm] + [float(value) for value in host_norms], 0.9))
                    ctx.guest.put(f"{tag_prefix}_clip_norm_{epoch}", dp_clip_norm)
                    ctx.hosts.put(f"{tag_prefix}_clip_norm_{epoch}", [dp_clip_norm] * len(host_norms))
                    print(f"[DPCalibration] {label} arbiter epoch={epoch + 1} clip_norm={dp_clip_norm:.8f}", flush=True)
                guest_delta = ctx.guest.get(f"{tag_prefix}_delta_{epoch}")
                host_deltas = ctx.hosts.get(f"{tag_prefix}_delta_{epoch}")
                host_deltas = host_deltas if isinstance(host_deltas, list) else [host_deltas]
                all_deltas = [guest_delta] + host_deltas
                if is_he:
                    started = time.perf_counter()
                    encrypted_sum = ckks_homomorphic_sum_tree(all_deltas, he_arbiter_context)
                    summed_delta = ckks_decrypt_tree(encrypted_sum, he_arbiter_context)
                    capture_he_sa_aggregate(ctx, args, f"{tag_prefix}_aggregate_{epoch}", summed_delta)
                    global_delta = {name: value / len(all_deltas) for name, value in summed_delta.items()}
                    he_seconds = time.perf_counter() - started
                    guest_bytes = int(ctx.guest.get(f"{tag_prefix}_upload_bytes_{epoch}"))
                    host_bytes = ctx.hosts.get(f"{tag_prefix}_upload_bytes_{epoch}")
                    host_bytes = host_bytes if isinstance(host_bytes, list) else [host_bytes]
                    print(
                        f"[HESA] {label} arbiter round={epoch + 1} "
                        f"aggregate_decrypt_s={he_seconds:.6f} "
                        f"encrypted_upload_bytes={guest_bytes + sum(int(v) for v in host_bytes)}",
                        flush=True,
                    )
                else:
                    global_delta = {
                        name: sum(delta[name] for delta in all_deltas) / len(all_deltas)
                        for name in all_deltas[0]
                    }
                ctx.guest.put(f"{tag_prefix}_global_delta_{epoch}", global_delta)
                ctx.hosts.put(
                    f"{tag_prefix}_global_delta_{epoch}",
                    [global_delta] * len(host_deltas) if len(host_deltas) > 1 else global_delta,
                )
                if is_he:
                    ctx.guest.put(f"{tag_prefix}_arbiter_seconds_{epoch}", he_seconds)
                    ctx.hosts.put(
                        f"{tag_prefix}_arbiter_seconds_{epoch}",
                        [he_seconds] * len(host_deltas) if len(host_deltas) > 1 else he_seconds,
                    )
                if bool(ctx.guest.get(f"{tag_prefix}_stop_{epoch}")):
                    print(f"[{label}-DP] arbiter early-stop at round {epoch + 1}", flush=True)
                    break
    except EarlyStopSignal:
        pass

    if ctx.is_on_arbiter:
        return None

    # Keep HE aggregation, but use the exact FATE client predictor used by
    # the Plain baseline for final test timing and metric accounting.
    from fate.ml.nn.homo.fedavg import FedAVGClient
    test_metrics, test_time = evaluate_with_plain_client(
        ctx=ctx, client_cls=FedAVGClient, model=model,
        train_set=train_set, val_set=val_set, test_set=test_set,
        optimizer=optimizer, loss_fn=loss_func, scheduler=scheduler,
        training_args=train_args, fed_args=fed_args, scaler=scaler,
        model_label=label, checkpoint_state=best_model_wts,
    )
    mae = test_metrics["mae"]
    rmse = test_metrics["rmse"]
    mape = test_metrics["mape"]
    elements = test_metrics["elements"]
    abs_sum = test_metrics["abs_sum"]
    sq_sum = test_metrics["sq_sum"]
    mape_sum = test_metrics["mape_sum"]
    mape_elements = test_metrics["mape_elements"]
    # For HE this value is actual bytes observed by this client (ciphertext
    # upload plus plaintext global-model download), rather than a proxy based
    # on parameter count.  The caller branches on protection when logging it.
    comm_params = (
        he_upload_bytes + he_download_bytes
        if is_he else sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    flops = 0.0
    try:
        from thop import profile
        sample_x, _ = _unpack_xy(next(iter(val_loader)))
        flops, _ = profile(model, inputs=(sample_x.to(args.device),), verbose=False)
        flops = round(flops / 1e9, 4)
    except Exception as exc:
        print(f"[{label}-DP] rank={ctx.rank} FLOPs unavailable: {exc}", flush=True)
    return (
        best_epoch, mae, rmse, mape,
        total_train_time, total_val_time / max(actual_epochs, 1), test_time,
        comm_params, actual_epochs, flops,
        elements, abs_sum, sq_sum, mape_sum, mape_elements,
    )


# Kept as a stable name for the existing FCGCN dispatch.
def train_fcgcn_dp_task(ctx, args, get_setting_func):
    return train_fedavg_dp_task(ctx, args, get_setting_func)
