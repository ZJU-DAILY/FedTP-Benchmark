from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def inverse_transform_tensor(x: torch.Tensor, scaler: Any) -> np.ndarray:
    x_detached = x.detach()
    if scaler is None or not hasattr(scaler, "inverse_transform"):
        return _to_numpy(x_detached)
    with torch.no_grad():
        return _to_numpy(scaler.inverse_transform(x_detached))


def reconstruction_metrics(
    reconstructed_x: torch.Tensor,
    real_x: torch.Tensor,
    scaler: Any = None,
    mape_eps: float = 10.0,
) -> Dict[str, float]:
    pred = inverse_transform_tensor(reconstructed_x, scaler).reshape(-1).astype(np.float64)
    true = inverse_transform_tensor(real_x, scaler).reshape(-1).astype(np.float64)

    diff = pred - true
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))

    abs_true = np.abs(true)
    abs_diff = np.abs(diff)
    mask = abs_true > mape_eps
    if np.any(mask):
        mape = float(np.mean(abs_diff[mask] / abs_true[mask]))
    else:
        mape = 0.0
    mape_valid_ratio = float(np.mean(mask))

    denominator = float(np.sum(abs_true))
    if denominator > mape_eps:
        wmape = float(np.sum(abs_diff) / denominator)
    else:
        wmape = 0.0

    pred_std = float(np.std(pred))
    true_std = float(np.std(true))
    if pred_std < 1e-12 or true_std < 1e-12:
        pcc = 0.0
    else:
        pcc = float(np.corrcoef(pred, true)[0, 1])

    return {
        "PCC": pcc,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "WMAPE": wmape,
        "MAPE_VALID_RATIO": mape_valid_ratio,
        "MAPE_EPS": float(mape_eps),
    }
