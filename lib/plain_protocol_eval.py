"""Final-test evaluation through the same client predictor as Plain runs.

HE training loops customize only encrypted model-delta aggregation.  Their
final test must use the original baseline client predictor so that
``Eff_TestTime`` has the same protocol/Trainer boundary as Plain.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import torch


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _plain_compute_metrics(prediction_output, scaler):
    """Equivalent to the non-testing metric work in ``fate_main.train``."""
    y_true_norm = _as_numpy(prediction_output.label_ids)
    y_pred_norm = _as_numpy(prediction_output.predictions)
    if hasattr(scaler, "inverse_transform"):
        y_true_real = _as_numpy(scaler.inverse_transform(y_true_norm))
        y_pred_real = _as_numpy(scaler.inverse_transform(y_pred_norm))
    else:
        y_true_real = y_true_norm * scaler.std + scaler.mean
        y_pred_real = y_pred_norm * scaler.std + scaler.mean
    diff = y_pred_real - y_true_real
    mask = y_true_real > 0.5
    return {
        "mae": float(np.mean(np.abs(diff))),
        "mse": float(np.mean(np.square(diff))),
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "mape": float(np.mean(np.abs(diff[mask]) / y_true_real[mask]) * 100.0) if np.any(mask) else 0.0,
    }


def evaluate_with_plain_client(
    *, ctx, client_cls, model, train_set, val_set, test_set, optimizer,
    loss_fn, scheduler, training_args, fed_args, scaler, model_label: str,
    checkpoint_state=None,
):
    """Call the original FATE client ``predict`` API and reproduce Plain timing.

    Plain runs persist a validation-best checkpoint during training, then load
    it *inside* the final-test timer.  Custom HE loops retain best weights in
    memory, so materialize the same checkpoint before timing and restore it
    inside the timed section.
    """
    checkpoint_dir = os.path.join(
        str(getattr(training_args, "output_dir", "./checkpoints")),
        "_he_plainpath_best",
    )
    checkpoint_path = os.path.join(checkpoint_dir, f"rank_{ctx.rank}.pt")
    os.makedirs(checkpoint_dir, exist_ok=True)
    state_to_persist = checkpoint_state or model.state_dict()
    cpu_state = {
        key: value.detach().cpu().contiguous().clone() if torch.is_tensor(value) else value
        for key, value in state_to_persist.items()
    }
    # Equivalent to Plain's checkpoint save during training; intentionally
    # outside final-test timing.
    torch.save(cpu_state, checkpoint_path)

    model.eval()
    started = time.time()
    client = client_cls(
        ctx=ctx, model=model, train_set=train_set, val_set=val_set,
        optimizer=optimizer, loss_fn=loss_fn, scheduler=scheduler,
        training_args=training_args, fed_args=fed_args,
        compute_metrics=lambda output: _plain_compute_metrics(output, scaler),
    )
    restore_state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(restore_state, strict=False)
    model.to(next(model.parameters()).device)
    if hasattr(client, "model"):
        client.model = model
    print(
        f"[PlainPathCheckpoint] model={model_label} rank={ctx.rank} restored={checkpoint_path}",
        flush=True,
    )
    prediction_output = client.predict(test_set)
    elapsed = time.time() - started

    y_true_real = _as_numpy(scaler.inverse_transform(_as_numpy(prediction_output.label_ids)))
    y_pred_real = _as_numpy(scaler.inverse_transform(_as_numpy(prediction_output.predictions)))
    diff = y_pred_real - y_true_real
    mask = y_true_real > 0.5
    elements = int(y_true_real.size)
    mape_elements = int(mask.sum())
    abs_sum = float(np.abs(diff).sum())
    sq_sum = float(np.square(diff).sum())
    mape_sum = float((np.abs(diff[mask]) / y_true_real[mask]).sum() * 100.0) if mape_elements else 0.0
    print(
        f"[PlainPathTestTiming] model={model_label} rank={ctx.rank} "
        f"test_samples={len(test_set)} "
        f"configured_eval_batch={getattr(training_args, 'per_device_eval_batch_size', None)} "
        f"seconds={elapsed:.6f}",
        flush=True,
    )
    return {
        "mae": abs_sum / max(elements, 1),
        "rmse": math.sqrt(sq_sum / max(elements, 1)),
        "mape": mape_sum / max(mape_elements, 1) if mape_elements else 0.0,
        "elements": elements,
        "abs_sum": abs_sum,
        "sq_sum": sq_sum,
        "mape_sum": mape_sum,
        "mape_elements": mape_elements,
    }, elapsed
