"""Shared privacy-protection primitives for experiment runners and attacks.

The functions are deliberately payload-agnostic: a federated upload can be a
tensor, a state-dict, or a nested combination of those.  DP is applied once to
the *complete* client upload, not independently to every parameter tensor.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch


TensorTree = Any


def _tensor_items(value: TensorTree):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for key in sorted(value):
            yield from _tensor_items(value[key])
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensor_items(item)


def l2_norm(value: TensorTree) -> torch.Tensor:
    tensors = [tensor for tensor in _tensor_items(value) if tensor.is_floating_point()]
    if not tensors:
        raise ValueError("Protected upload contains no tensors.")
    return torch.sqrt(sum(t.detach().float().pow(2).sum() for t in tensors))


def _map_tree(value: TensorTree, fn):
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, Mapping):
        return {key: _map_tree(item, fn) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_map_tree(item, fn) for item in value)
    if isinstance(value, list):
        return [_map_tree(item, fn) for item in value]
    return value


def _zip_map_tree(left: TensorTree, right: TensorTree, fn):
    if torch.is_tensor(left):
        return fn(left, right)
    if isinstance(left, Mapping):
        return {key: _zip_map_tree(left[key], right[key], fn) for key in left}
    if isinstance(left, tuple):
        return tuple(_zip_map_tree(a, b, fn) for a, b in zip(left, right))
    if isinstance(left, list):
        return [_zip_map_tree(a, b, fn) for a, b in zip(left, right)]
    return left


def gaussian_dp_upload(
    upload: TensorTree,
    *,
    clip_norm: float,
    sigma: float,
    post_clip_ratio: float = 0.0,
    experimental_noise_clip_ratio: float = 0.0,
    experimental_skip_update_clip: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[TensorTree, dict[str, float]]:
    """Clip a complete upload globally, add Gaussian noise, then optionally
    post-clip the complete protected upload.

    This is an update-level mechanism.  ``sigma`` is the noise multiplier and
    the per-coordinate noise standard deviation is ``sigma * clip_norm``.
    """
    if clip_norm <= 0:
        raise ValueError("dp_clip_norm must be > 0 when DP is enabled.")
    if sigma < 0:
        raise ValueError("dp_sigma must be >= 0.")
    if post_clip_ratio < 0 or experimental_noise_clip_ratio < 0:
        raise ValueError("dp_post_clip_ratio must be >= 0.")
    norm = l2_norm(upload)
    coefficient = (
        1.0 if experimental_skip_update_clip
        else min(1.0, float(clip_norm / (norm.item() + 1e-12)))
    )

    def scaled_signal(t: torch.Tensor) -> torch.Tensor:
        if not t.is_floating_point():
            # Counters/buffers are protocol metadata, not continuous DP payloads.
            return t.detach().clone()
        return t.detach().clone().mul_(coefficient)

    def gaussian_noise(t: torch.Tensor) -> torch.Tensor:
        if not t.is_floating_point():
            return t.detach().clone()
        if sigma <= 0:
            return torch.zeros_like(t)
        # Generate on CPU so one deterministic seed works for CPU and CUDA
        # attack runs alike, then move to the upload's device.
        return torch.randn(
            t.shape, device="cpu", dtype=t.dtype, generator=generator
        ).to(t.device).mul_(sigma * clip_norm)

    signal = _map_tree(upload, scaled_signal)
    noise = _map_tree(upload, gaussian_noise)
    noise_l2 = float(l2_norm(noise).item())
    noise_clip_norm = float(experimental_noise_clip_ratio * clip_norm)
    noise_clip_coefficient = 1.0
    if noise_clip_norm > 0:
        # EXPERIMENTAL ONLY: clipping the sampled noise before addition makes
        # it a truncated, non-Gaussian mechanism.  This is intentionally kept
        # separate from DP-safe post-processing and must not be reported as
        # standard Gaussian DP.
        noise_clip_coefficient = min(1.0, noise_clip_norm / (noise_l2 + 1e-12))
        noise = _map_tree(
            noise,
            lambda tensor: (
                tensor.detach().clone().mul_(noise_clip_coefficient)
                if tensor.is_floating_point() else tensor.detach().clone()
            ),
        )

    def add_signal_and_noise(signal_tensor: torch.Tensor, noise_tensor: torch.Tensor) -> torch.Tensor:
        if not signal_tensor.is_floating_point():
            return signal_tensor.detach().clone()
        return signal_tensor.detach().clone().add_(noise_tensor)

    protected = _zip_map_tree(signal, noise, add_signal_and_noise)
    post_noise_l2 = float(l2_norm(protected).item())
    post_clip_norm = float(post_clip_ratio * clip_norm)
    post_clip_coefficient = 1.0
    if post_clip_norm > 0:
        # This is DP-safe post-processing: the standard Gaussian mechanism
        # has already been applied above.  We clip the *complete noised
        # upload*, not the Gaussian noise alone, so that unusually large
        # parameter jumps cannot destabilise the following global update.
        post_clip_coefficient = min(1.0, post_clip_norm / (post_noise_l2 + 1e-12))
        protected = _map_tree(
            protected,
            lambda tensor: (
                tensor.detach().clone().mul_(post_clip_coefficient)
                if tensor.is_floating_point() else tensor.detach().clone()
            ),
        )
    final_upload_l2 = float(l2_norm(protected).item())

    return protected, {
        "dp_clip_norm": float(clip_norm),
        "dp_sigma": float(sigma),
        "upload_l2_norm": float(norm.item()),
        "clip_coefficient": coefficient,
        "noise_std": float(sigma * clip_norm),
        "experimental_skip_update_clip": float(bool(experimental_skip_update_clip)),
        "noise_l2_norm": noise_l2,
        "experimental_noise_clip_norm": noise_clip_norm,
        "experimental_noise_clip_coefficient": noise_clip_coefficient,
        "post_clip_norm": post_clip_norm,
        "post_noise_l2_norm": post_noise_l2,
        "post_clip_coefficient": post_clip_coefficient,
        "final_upload_l2_norm": final_upload_l2,
    }
