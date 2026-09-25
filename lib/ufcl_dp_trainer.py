"""Explicit DP-FedAvg loop that preserves UFCL local continual learning."""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
from torch.utils.data import default_collate

from lib.ufcl_trainer import UFCLTrainer
from lib.ufcl_client import UFCLFedAVGClient
from lib.plain_protocol_eval import evaluate_with_plain_client
from lib.utils import (
    EarlyStopSignal, ExplicitEarlyStopper, align_prediction_and_target,
    unpack_spatiotemporal_batch, synchronize_cuda_for_timing,
)
from privacy.protection import l2_norm
from privacy.runtime_protection import protected_arbiter_put
from privacy.attack_trace import capture_he_sa_aggregate, capture_he_sa_server_aggregate_hidden_term


def _trainable_state(model):
    """Return exactly the tensors FATE's FedAVG wrapper aggregates.

    FATE's ``BaseAggregatorClient._process_model`` iterates over
    ``model.parameters()`` and filters ``requires_grad``.  In particular it
    does *not* aggregate floating buffers from ``state_dict`` (e.g.
    BatchNorm running statistics).  Keeping this distinction is essential for
    UFCL: averaging the buffers changes the second and every later local
    trajectory even when DP sigma is zero.
    """
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _as_rank_payload(ctx, value):
    if isinstance(value, list):
        return value[ctx.rank - 1] if ctx.rank > 0 and len(value) > 1 else value[0]
    return value


def _weighted_mean_states(states, weights):
    """FATE-compatible weighted mean of trainable parameter trees.

    The bundled FATE implementation converts trainable parameters to
    float64, applies the per-client sample-count weight, and copies the
    aggregate back to the parameter tensor.  This reproduces that numerical
    order while intentionally leaving non-parameter buffers client-local.
    """
    if not states:
        raise ValueError("Cannot average an empty UFCL state list.")
    if len(states) != len(weights) or not weights or sum(weights) <= 0:
        raise ValueError("UFCL aggregation requires one positive weight per client state.")
    reference = states[0]
    result = {}
    for key, value in reference.items():
        values = [state.get(key) for state in states]
        if not all(torch.is_tensor(item) and item.shape == value.shape for item in values):
            raise ValueError(f"UFCL trainable parameter mismatch for key={key!r}.")
        aggregate = sum(item.detach().double() * float(weight) for item, weight in zip(values, weights))
        result[key] = aggregate.div(float(sum(weights))).to(value.dtype)
    return result


def _load_trainable_state(model, state):
    """Copy an aggregate into parameters only, matching FATE's recovery path."""
    parameters = dict(model.named_parameters())
    missing = [name for name in state if name not in parameters]
    if missing:
        raise ValueError(f"UFCL aggregate contains unknown trainable parameters: {missing[:3]}")
    with torch.no_grad():
        for name, value in state.items():
            parameter = parameters[name]
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def _prediction(model, x, y):
    output = model(x)
    return align_prediction_and_target(output, y)


def train_ufcl_dp_task(ctx, args, get_setting_func):
    """Run UFCL replay/mixup/KD with DP or CKKS HE-SA model deltas."""
    is_he = str(getattr(args, "protection", "plain")).lower() == "he"
    prefix = "ufcl_he" if is_he else "ufcl_dp"
    if ctx.is_on_arbiter:
        steps = 0
        dp_clip = float(args.dp_clip_norm) if float(args.dp_clip_norm) > 0 else None

    he_context = None
    he_arbiter_context = None
    he_upload_bytes = he_download_bytes = 0
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
            raise ValueError("UFCL supports HE-SA only for its shared model-delta aggregation.")
        key_tag = "__ufcl_he_sa_ckks_context"
        if ctx.is_on_arbiter:
            he_arbiter_context = ckks_generate_context(
                int(args.he_ckks_poly_modulus_degree), int(args.he_ckks_scale_bits),
            )
            payload = ckks_export_public_context(he_arbiter_context)
            ctx.guest.put(key_tag, payload); ctx.hosts.put(key_tag, payload)
            print("[HESA] UFCL arbiter generated packed CKKS context", flush=True)
        else:
            he_context = ckks_import_context(ctx.arbiter.get(key_tag))
    # Client model/data initialisation is independent of the protection
    # backend.  In particular, HE clients first receive the CKKS public
    # context above and must then still build their local UFCL model.
    if not ctx.is_on_arbiter:
        (
            train_set, val_set, test_set, model, optimizer, loss_func,
            scheduler, train_args, fed_args, scaler, _,
        ) = get_setting_func(ctx)
        # The native Trainer's sampler must be constructed before the first
        # client/Arbiter rendezvous: the Arbiter receives ``steps`` first.
        # Sending it after waiting for initial_global_state creates a circular
        # wait between every client and the Arbiter.
        ufcl_trainer = UFCLTrainer(
            model=model, args=train_args, train_dataset=train_set, eval_dataset=val_set,
            compute_metrics=None, optimizers=(optimizer, scheduler),
            # UFCL datasets return positional (x, y) tuples.  Transformers'
            # default collator assumes dataclass/dict records and calls
            # vars(tuple), whereas the native torch loader stacks these tuples
            # with PyTorch's default collate implementation.
            data_collator=default_collate,
            ufcl_max_nodes=None,
        )
        train_loader = ufcl_trainer.get_train_dataloader()
        val_loader = ufcl_trainer.get_eval_dataloader()
        steps = len(train_loader)
        ctx.arbiter.put(f"{prefix}_steps", steps)
        ctx.arbiter.put(f"{prefix}_sample_count", int(len(train_set)))
        print(
            f"[UFCL-DP] rank={ctx.rank} using native UFCLTrainer dataloaders "
            f"train_steps={len(train_loader)} val_steps={len(val_loader)}",
            flush=True,
        )
        # The historical Plain UFCL implementation lets every client complete
        # its *first* local epoch from the model that its own process built,
        # then FATE averages complete model states.  Forcing every DP client
        # to start from the guest state changes the first local trajectory,
        # and that divergence persists even when sigma=0.  Preserve that
        # native first-round behaviour for DP: initial weights are public
        # randomness, while only each client's delta from its own initial
        # reference is privacy-protected.  HE-SA retains its canonical shared
        # initialization because its encrypted delta protocol requires it.
        initial_reference = _trainable_state(model)
        if is_he:
            ctx.arbiter.put(f"{prefix}_public_init", initial_reference)
            public_state = _as_rank_payload(ctx, ctx.arbiter.get(f"{prefix}_public_init"))
            _load_trainable_state(model, public_state)
        else:
            ctx.arbiter.put(f"{prefix}_initial_reference", initial_reference)
            initial_global_state = _as_rank_payload(ctx, ctx.arbiter.get(f"{prefix}_initial_global_state"))
            print(
                f"[UFCL-DP] rank={ctx.rank} retained native local initial state; "
                f"received public first-round reference keys={len(initial_global_state)}",
                flush=True,
            )

        # With ``scheduler=None`` the native HuggingFace Trainer creates its
        # default linear schedule from the full configured optimization-step
        # count.  The explicit DP loop must do the same; otherwise it keeps
        # LR=0.001 forever whereas the Plain baseline decays it every batch.
        # This is a training-protocol detail, not a privacy mechanism.
        if scheduler is None:
            total_optimizer_steps = max(
                1,
                int(args.epochs) * max(int(args.local_epochs), 1) * max(int(steps), 1),
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: max(0.0, 1.0 - (float(step) / float(total_optimizer_steps))),
            )
            print(
                f"[UFCL-DP] rank={ctx.rank} installed native-equivalent linear LR schedule "
                f"total_optimizer_steps={total_optimizer_steps}",
                flush=True,
            )
        best_state = None
        best_epoch = -1
        best_val = float("inf")
        total_train = total_val = 0.0
        actual_rounds = 0
        stopper = ExplicitEarlyStopper(ctx, args, patience=50, min_delta=1e-5)
        dp_clip = float(args.dp_clip_norm) if float(args.dp_clip_norm) > 0 else None

    if ctx.is_on_arbiter:
        guest_steps = int(ctx.guest.get(f"{prefix}_steps"))
        host_steps = ctx.hosts.get(f"{prefix}_steps")
        host_steps = host_steps if isinstance(host_steps, list) else [host_steps]
        steps = min([guest_steps] + [int(value) for value in host_steps])
        guest_samples = int(ctx.guest.get(f"{prefix}_sample_count"))
        host_samples = ctx.hosts.get(f"{prefix}_sample_count")
        host_samples = host_samples if isinstance(host_samples, list) else [host_samples]
        client_weights = [guest_samples] + [int(value) for value in host_samples]
        if any(weight <= 0 for weight in client_weights):
            raise ValueError(f"UFCL received non-positive FedAVG sample counts: {client_weights}")
        if is_he:
            init_guest = ctx.guest.get(f"{prefix}_public_init")
            init_hosts = ctx.hosts.get(f"{prefix}_public_init")
            init_hosts = init_hosts if isinstance(init_hosts, list) else [init_hosts]
            # Initialization is public/synchronization-only; use guest as the
            # shared reference state and do not add DP noise.
            ctx.guest.put(f"{prefix}_public_init", init_guest)
            ctx.hosts.put(
                f"{prefix}_public_init",
                [init_guest] * len(init_hosts) if len(init_hosts) > 1 else init_guest,
            )
            initial_global_state = None
        else:
            init_guest = ctx.guest.get(f"{prefix}_initial_reference")
            init_hosts = ctx.hosts.get(f"{prefix}_initial_reference")
            init_hosts = init_hosts if isinstance(init_hosts, list) else [init_hosts]
            initial_global_state = _weighted_mean_states([init_guest] + init_hosts, client_weights)
            ctx.guest.put(f"{prefix}_initial_global_state", initial_global_state)
            ctx.hosts.put(
                f"{prefix}_initial_global_state",
                [initial_global_state] * len(init_hosts) if len(init_hosts) > 1 else initial_global_state,
            )
            print(
                f"[UFCL-DP] arbiter retained native first-round initial states; "
                f"initial_global_keys={len(initial_global_state)} sample_counts={client_weights}",
                flush=True,
            )

    try:
        for epoch in range(args.epochs):
            if not ctx.is_on_arbiter:
                start_state = _trainable_state(model)
                # Keep the native UFCL teacher/replay lifecycle.  The original
                # FATE UFCL trainer creates its teacher on the first local
                # batch and keeps both the teacher and replay buffer alive for
                # the rest of the run.  Resetting it here once per explicit
                # aggregation round changed the objective even at sigma=0,
                # which made a DP-vs-Plain accuracy comparison invalid.
                model.train()
                started = time.perf_counter()
                loss_sum = 0.0
                for _ in range(max(int(args.local_epochs), 1)):
                    for step, batch in enumerate(train_loader):
                        if step >= steps:
                            break
                        x, y = unpack_spatiotemporal_batch(batch)
                        optimizer.zero_grad()
                        loss = ufcl_trainer.compute_loss(model, (x.to(args.device), y.to(args.device)))
                        loss.backward()
                        # ``UFCLFedAVGClient`` delegates optimisation to
                        # HuggingFace ``Trainer``.  Its default
                        # ``max_grad_norm=1.0`` clips gradients before every
                        # optimizer step.  The former explicit DP loop called
                        # ``optimizer.step`` directly and omitted that clip,
                        # so it followed a different optimization trajectory
                        # even at sigma=0 (its first-round model deltas grew to
                        # L2 13--16).  Reproduce the native Trainer behaviour
                        # before forming the client update.
                        max_grad_norm = float(getattr(train_args, "max_grad_norm", 0.0) or 0.0)
                        if max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        loss_sum += float(loss.detach().cpu())
                total_train += time.perf_counter() - started

                end_state = _trainable_state(model)
                delta = {key: end_state[key] - start_state[key] for key in start_state}
                if not is_he and dp_clip is None:
                    ctx.arbiter.put(f"{prefix}_delta_norm_{epoch}", float(l2_norm(delta).item()))
                    dp_clip = float(_as_rank_payload(ctx, ctx.arbiter.get(f"{prefix}_clip_{epoch}")))
                    args.dp_clip_norm = dp_clip
                    print(f"[DPCalibration] UFCL rank={ctx.rank} epoch={epoch + 1} clip_norm={dp_clip:.8f}", flush=True)
                protection_started = time.perf_counter()
                if is_he:
                    capture_he_sa_server_aggregate_hidden_term(
                        ctx, args, f"{prefix}_delta_{epoch}", payload=delta,
                        model_state_dict=start_state, aggregation_weight=1.0,
                        leak_type="model_update",
                    )
                    encrypted_delta = ckks_encrypt_tree(
                        delta, he_context,
                        slot_count=int(args.he_ckks_poly_modulus_degree) // 2,
                    )
                    upload_bytes = ckks_ciphertext_bytes(encrypted_delta)
                    he_upload_bytes += upload_bytes
                    ctx.arbiter.put(f"{prefix}_delta_{epoch}", encrypted_delta)
                    ctx.arbiter.put(f"{prefix}_upload_bytes_{epoch}", upload_bytes)
                    print(f"[HESA] UFCL rank={ctx.rank} round={epoch + 1} encrypted_upload_bytes={upload_bytes}", flush=True)
                else:
                    # Retain the DP accounting record for a compact training
                    # diagnostic.  In a high-dimensional model, a seemingly
                    # small per-coordinate sigma can still have a substantial
                    # vector L2 norm (roughly sigma * C * sqrt(d)).  Logging
                    # this at round 1--3 and then every 50 rounds makes a
                    # genuine noise-dominates-signal failure distinguishable
                    # from an incorrect UFCL protocol implementation.
                    _, _, dp_info = protected_arbiter_put(
                        ctx, args, f"{prefix}_delta_{epoch}", delta,
                        clip_norm=dp_clip, return_protected_payload=True,
                    )
                    if dp_info is not None and (epoch < 3 or (epoch + 1) % 50 == 0):
                        clipped_signal_l2 = float(dp_info["upload_l2_norm"]) * float(dp_info["clip_coefficient"])
                        ratio = float(dp_info["noise_l2_norm"]) / max(clipped_signal_l2, 1e-12)
                        print(
                            f"[UFCL-DP SNR] rank={ctx.rank} round={epoch + 1} "
                            f"raw_delta_l2={dp_info['upload_l2_norm']:.8f} "
                            f"clipped_signal_l2={clipped_signal_l2:.8f} "
                            f"noise_l2={dp_info['noise_l2_norm']:.8f} "
                            f"noise_to_signal={ratio:.8f}",
                            flush=True,
                        )
                total_train += time.perf_counter() - protection_started
                global_delta = _as_rank_payload(ctx, ctx.arbiter.get(f"{prefix}_global_delta_{epoch}"))
                if is_he:
                    he_download_bytes += sum(value.numel() * value.element_size() for value in global_delta.values())
                    he_seconds = _as_rank_payload(ctx, ctx.arbiter.get(f"{prefix}_arbiter_seconds_{epoch}"))
                    total_train += float(he_seconds)
                if not is_he and epoch == 0:
                    # Match native FATE's first aggregation exactly:
                    # mean(initial_client_state + client_delta).  Subsequent
                    # rounds begin from this common global state and can use
                    # ordinary mean-delta aggregation.
                    global_state = _as_rank_payload(ctx, ctx.arbiter.get(f"{prefix}_global_state_0"))
                    _load_trainable_state(model, global_state)
                else:
                    _load_trainable_state(
                        model,
                        {key: start_state[key] + value.cpu() for key, value in global_delta.items()},
                    )

                model.eval()
                val_started = time.perf_counter()
                abs_sum = sq_sum = 0.0
                elements = 0
                with torch.no_grad():
                    for batch in val_loader:
                        x, y = unpack_spatiotemporal_batch(batch)
                        prediction, target = _prediction(model, x.to(args.device), y.to(args.device))
                        diff = prediction - target
                        abs_sum += float(diff.abs().sum())
                        sq_sum += float(diff.square().sum())
                        elements += int(target.numel())
                total_val += time.perf_counter() - val_started
                val_mae = abs_sum / max(elements, 1)
                if val_mae < best_val:
                    best_val, best_epoch = val_mae, epoch
                    best_state = copy.deepcopy(model.state_dict())
                actual_rounds = epoch + 1
                should_stop = bool(stopper.check_and_sync(val_mae))
                if ctx.is_on_guest:
                    ctx.arbiter.put(f"{prefix}_stop_{epoch}", should_stop)
                current_lr = float(optimizer.param_groups[0]["lr"])
                print(
                    f"[UFCL-DP] rank={ctx.rank} round={epoch + 1} "
                    f"local_loss={loss_sum/max(steps,1):.6f} val_mae={val_mae:.8f} lr={current_lr:.10f}",
                    flush=True,
                )
                if should_stop:
                    raise EarlyStopSignal("UFCL DP early stop")
            else:
                if not is_he and dp_clip is None:
                    guest_norm = float(ctx.guest.get(f"{prefix}_delta_norm_{epoch}"))
                    host_norms = ctx.hosts.get(f"{prefix}_delta_norm_{epoch}")
                    host_norms = host_norms if isinstance(host_norms, list) else [host_norms]
                    dp_clip = float(np.quantile([guest_norm] + [float(value) for value in host_norms], 0.9))
                    ctx.guest.put(f"{prefix}_clip_{epoch}", dp_clip)
                    ctx.hosts.put(f"{prefix}_clip_{epoch}", [dp_clip] * len(host_norms))
                    print(f"[DPCalibration] UFCL arbiter epoch={epoch + 1} clip_norm={dp_clip:.8f}", flush=True)
                guest_delta = ctx.guest.get(f"{prefix}_delta_{epoch}")
                host_deltas = ctx.hosts.get(f"{prefix}_delta_{epoch}")
                host_deltas = host_deltas if isinstance(host_deltas, list) else [host_deltas]
                deltas = [guest_delta] + host_deltas
                if is_he:
                    started = time.perf_counter()
                    encrypted_sum = ckks_homomorphic_sum_tree(deltas, he_arbiter_context)
                    summed = ckks_decrypt_tree(encrypted_sum, he_arbiter_context)
                    capture_he_sa_aggregate(ctx, args, f"ufcl_aggregate_{epoch}", summed)
                    averaged = {key: value / len(deltas) for key, value in summed.items()}
                    he_seconds = time.perf_counter() - started
                    guest_bytes = int(ctx.guest.get(f"{prefix}_upload_bytes_{epoch}"))
                    host_bytes = ctx.hosts.get(f"{prefix}_upload_bytes_{epoch}")
                    host_bytes = host_bytes if isinstance(host_bytes, list) else [host_bytes]
                    print(f"[HESA] UFCL arbiter round={epoch + 1} aggregate_decrypt_s={he_seconds:.6f} encrypted_upload_bytes={guest_bytes + sum(int(v) for v in host_bytes)}", flush=True)
                else:
                    averaged = _weighted_mean_states(deltas, client_weights)
                ctx.guest.put(f"{prefix}_global_delta_{epoch}", averaged)
                ctx.hosts.put(f"{prefix}_global_delta_{epoch}", [averaged] * len(host_deltas) if len(host_deltas) > 1 else averaged)
                if not is_he and epoch == 0:
                    first_global_state = {
                        key: initial_global_state[key] + value
                        for key, value in averaged.items()
                    }
                    ctx.guest.put(f"{prefix}_global_state_0", first_global_state)
                    ctx.hosts.put(
                        f"{prefix}_global_state_0",
                        [first_global_state] * len(host_deltas) if len(host_deltas) > 1 else first_global_state,
                    )
                if is_he:
                    ctx.guest.put(f"{prefix}_arbiter_seconds_{epoch}", he_seconds)
                    ctx.hosts.put(f"{prefix}_arbiter_seconds_{epoch}", [he_seconds] * len(host_deltas) if len(host_deltas) > 1 else he_seconds)
                if bool(ctx.guest.get(f"{prefix}_stop_{epoch}")):
                    break
    except EarlyStopSignal:
        pass

    if ctx.is_on_arbiter:
        return None
    # The Plain UFCL baseline evaluates via UFCLFedAVGClient/UFCLTrainer,
    # including its baseline-specific padding and prediction step.
    test_metrics, test_time = evaluate_with_plain_client(
        ctx=ctx, client_cls=UFCLFedAVGClient, model=model,
        train_set=train_set, val_set=val_set, test_set=test_set,
        optimizer=optimizer, loss_fn=loss_func, scheduler=scheduler,
        training_args=train_args, fed_args=fed_args, scaler=scaler,
        model_label="UFCL", checkpoint_state=best_state,
    )
    abs_sum = test_metrics["abs_sum"]
    sq_sum = test_metrics["sq_sum"]
    mape_sum = test_metrics["mape_sum"]
    elements = test_metrics["elements"]
    mape_elements = test_metrics["mape_elements"]
    flops = 0.0
    try:
        from thop import profile
        x, _ = unpack_spatiotemporal_batch(next(iter(val_loader)))
        flops, _ = profile(model, inputs=(x.to(args.device),), verbose=False)
        flops = round(flops / 1e9, 4)
    except Exception:
        pass
    return (best_epoch, abs_sum/max(elements,1), math.sqrt(sq_sum/max(elements,1)),
            mape_sum/max(mape_elements,1) if mape_elements else 0.0,
            total_train, total_val/max(actual_rounds,1), test_time,
            (he_upload_bytes + he_download_bytes) if is_he else sum(p.numel() for p in model.parameters() if p.requires_grad), actual_rounds,
            flops, elements, abs_sum, sq_sum, mape_sum, mape_elements)
