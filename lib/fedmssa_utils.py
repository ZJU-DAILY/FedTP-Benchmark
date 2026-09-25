import math
from typing import Optional, Tuple

import torch


def reconstruct_series_from_windows(window_dataset):
    """
    Recover an approximate split-level raw series.

    Supported inputs:
      1. TensorDataset-like object with tensors=(X, Y), where
         X: [samples, nodes, t_in, features]
      2. UnifiedTrafficDataset-like object with flow_split already storing
         the normalized split-level sequence as [time, nodes, features]
      3. torch.utils.data.Subset wrapping one of the above
      4. Custom dataset exposing x/y tensors directly

    Returns:
      raw_series: [time, nodes, features]
    """
    if hasattr(window_dataset, "dataset"):
        base_dataset = window_dataset.dataset
        if hasattr(window_dataset, "indices"):
            indices = window_dataset.indices
            if hasattr(base_dataset, "tensors") and len(base_dataset.tensors) >= 2:
                x_tensor = base_dataset.tensors[0][indices]
                y_tensor = base_dataset.tensors[1][indices]
                window_dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
            else:
                window_dataset = base_dataset
        else:
            window_dataset = base_dataset

    if hasattr(window_dataset, "flow_split"):
        raw = window_dataset.flow_split
        if isinstance(raw, torch.Tensor):
            return raw.detach().cpu().float()
        return torch.as_tensor(raw, dtype=torch.float32)

    if hasattr(window_dataset, "x"):
        x_tensor = window_dataset.x
        if not torch.is_tensor(x_tensor):
            x_tensor = torch.as_tensor(x_tensor, dtype=torch.float32)
        if x_tensor.dim() != 4:
            raise ValueError(
                f"FedmSSA expects X windows shaped [samples, nodes, time, features], got {tuple(x_tensor.shape)}"
            )
        if x_tensor.size(0) == 0:
            raise ValueError("FedmSSA received an empty split and cannot reconstruct raw series.")
        x_cpu = x_tensor.detach().cpu()
        first_window = x_cpu[0].permute(1, 0, 2).contiguous()
        tail = x_cpu[1:, :, -1, :].contiguous()
        raw = first_window if tail.numel() == 0 else torch.cat([first_window, tail], dim=0)
        return raw.float()

    if not hasattr(window_dataset, "tensors") or len(window_dataset.tensors) < 2:
        raise ValueError(
            "FedmSSA currently expects either a TensorDataset with (X, Y) tensors "
            "or a dataset exposing flow_split."
        )

    x_tensor = window_dataset.tensors[0]
    if x_tensor.dim() != 4:
        raise ValueError(
            f"FedmSSA expects X windows shaped [samples, nodes, time, features], got {tuple(x_tensor.shape)}"
        )
    if x_tensor.size(0) == 0:
        raise ValueError("FedmSSA received an empty split and cannot reconstruct raw series.")

    x_cpu = x_tensor.detach().cpu()
    first_window = x_cpu[0].permute(1, 0, 2).contiguous()
    tail = x_cpu[1:, :, -1, :].contiguous()
    if tail.numel() == 0:
        raw = first_window
    else:
        raw = torch.cat([first_window, tail], dim=0)
    return raw.float()


def prepare_series_for_page_matrix(raw_series: torch.Tensor) -> torch.Tensor:
    if raw_series.dim() != 3:
        raise ValueError(f"Expected raw series [time, nodes, features], got {tuple(raw_series.shape)}")
    time_steps, num_nodes, num_features = raw_series.shape
    return raw_series.permute(1, 2, 0).contiguous().view(num_nodes * num_features, time_steps)


def build_page_matrix(multivariate_series: torch.Tensor, page_length: int) -> torch.Tensor:
    if multivariate_series.dim() != 2:
        raise ValueError(
            f"Expected multivariate series [variables, time], got {tuple(multivariate_series.shape)}"
        )
    page_length = int(page_length)
    variables, time_steps = multivariate_series.shape
    usable = (time_steps // page_length) * page_length
    if usable < page_length:
        raise ValueError(
            f"Sequence too short for FedmSSA page_length={page_length}: time_steps={time_steps}"
        )
    trimmed = multivariate_series[:, :usable]
    num_pages = usable // page_length
    blocks = []
    for var_idx in range(variables):
        var_series = trimmed[var_idx].view(num_pages, page_length).transpose(0, 1).contiguous()
        blocks.append(var_series)
    return torch.cat(blocks, dim=1)


def infer_rank_from_singular_values(
    singular_values: torch.Tensor,
    num_columns: int,
    explicit_rank: Optional[int] = None,
    sv_scale: float = 2.0,
) -> int:
    if explicit_rank is not None and int(explicit_rank) > 0:
        return min(int(explicit_rank), int(singular_values.numel()))
    threshold = float(sv_scale) * math.sqrt(max(int(num_columns), 1))
    rank = int((singular_values >= threshold).sum().item())
    return max(rank, 1)


def orthonormalize_basis(matrix: torch.Tensor, rank: Optional[int] = None) -> torch.Tensor:
    if matrix.dim() != 2:
        raise ValueError(f"Expected basis matrix with 2 dims, got shape={tuple(matrix.shape)}")
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    if rank is not None:
        q = q[:, : int(rank)]
    return q.contiguous()


def page_matrix_to_series(
    page_matrix: torch.Tensor,
    page_length: int,
    num_nodes: int,
    num_features: int,
) -> torch.Tensor:
    page_length = int(page_length)
    variables = int(num_nodes) * int(num_features)
    if page_matrix.dim() != 2 or page_matrix.size(0) != page_length:
        raise ValueError(
            f"Invalid page matrix shape {tuple(page_matrix.shape)} for page_length={page_length}"
        )
    total_cols = int(page_matrix.size(1))
    if total_cols % variables != 0:
        raise ValueError(
            f"Page matrix columns={total_cols} not divisible by variables={variables}."
        )
    num_pages = total_cols // variables
    var_series = []
    for var_idx in range(variables):
        block = page_matrix[:, var_idx * num_pages:(var_idx + 1) * num_pages]
        series = block.transpose(0, 1).contiguous().view(-1)
        var_series.append(series)
    stacked = torch.stack(var_series, dim=0).view(num_nodes, num_features, num_pages * page_length)
    return stacked.permute(2, 0, 1).contiguous()


def create_window_tensors_from_series(raw_series: torch.Tensor, t_in: int, t_out: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if raw_series.dim() != 3:
        raise ValueError(f"Expected raw series [time, nodes, features], got {tuple(raw_series.shape)}")
    time_steps = int(raw_series.size(0))
    total = time_steps - int(t_in) - int(t_out) + 1
    if total <= 0:
        raise ValueError(
            f"FedmSSA split too short for t_in={t_in}, t_out={t_out}, time_steps={time_steps}"
        )
    xs, ys = [], []
    for start in range(total):
        xs.append(raw_series[start:start + t_in])
        ys.append(raw_series[start + t_in:start + t_in + t_out])
    x_tensor = torch.stack(xs, dim=0).permute(0, 2, 1, 3).contiguous()
    y_tensor = torch.stack(ys, dim=0).permute(0, 2, 1, 3).contiguous()
    return x_tensor.float(), y_tensor.float()


def tensor_dataset_from_series(raw_series: torch.Tensor, t_in: int, t_out: int, device: str):
    """Build a CPU-backed window dataset for Fed-mSSA.

    FATE launches all 32 clients as separate processes on one GPU.  Moving an
    entire denoised TaxiBJ split to CUDA here multiplies the full train/val/test
    storage by 32 and exhausts VRAM before phase-2 training starts.  Trainers
    already move only their current mini-batch to ``device``.
    """
    x_tensor, y_tensor = create_window_tensors_from_series(raw_series, t_in=t_in, t_out=t_out)
    return torch.utils.data.TensorDataset(x_tensor.cpu(), y_tensor.cpu())


def build_page_observation(
    raw_series: torch.Tensor,
    page_length: int,
    missing_ratio: float = 0.0,
    seed: int = 0,
) -> dict:
    """
    Returns:
      {
        raw_series,
        multivariate_series,
        page_matrix_true,
        page_matrix_obs,
        obs_mask,
        usable,
        num_nodes,
        num_features,
        num_columns,
      }
    """
    if raw_series.dim() != 3:
        raise ValueError(f"Expected raw series [time, nodes, features], got {tuple(raw_series.shape)}")
    time_steps, num_nodes, num_features = raw_series.shape
    multivariate_series = prepare_series_for_page_matrix(raw_series)
    usable = (multivariate_series.size(1) // int(page_length)) * int(page_length)
    trimmed_series = multivariate_series[:, :usable]
    page_matrix_true = build_page_matrix(trimmed_series, page_length=page_length).float()

    obs_mask = torch.ones_like(page_matrix_true, dtype=torch.float32)
    ratio = float(max(0.0, min(1.0, missing_ratio)))
    if ratio > 0.0:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        sampled = torch.rand(page_matrix_true.shape, generator=generator)
        obs_mask = (sampled >= ratio).float()
        for row in range(obs_mask.size(0)):
            if obs_mask[row].sum() <= 0:
                idx = int(row % max(obs_mask.size(1), 1))
                obs_mask[row, idx] = 1.0

    page_matrix_obs = page_matrix_true * obs_mask
    return {
        "raw_series": raw_series.float(),
        "multivariate_series": trimmed_series.float(),
        "page_matrix_true": page_matrix_true,
        "page_matrix_obs": page_matrix_obs,
        "obs_mask": obs_mask,
        "usable": usable,
        "num_nodes": int(num_nodes),
        "num_features": int(num_features),
        "num_columns": int(page_matrix_true.size(1)),
    }


def low_rank_project(page_matrix: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    page_matrix = page_matrix.to(basis.device)
    return basis @ (basis.transpose(0, 1) @ page_matrix)


def masked_reconstruction_loss(page_obs: torch.Tensor, obs_mask: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    page_obs = page_obs.to(basis.device)
    obs_mask = obs_mask.to(basis.device)
    reconstruction = low_rank_project(page_obs, basis)
    diff = obs_mask * (page_obs - reconstruction)
    denom = obs_mask.sum().clamp_min(1.0)
    return (diff.pow(2).sum()) / denom


def orthogonality_penalty(basis: torch.Tensor) -> torch.Tensor:
    rank = int(basis.size(1))
    identity = torch.eye(rank, device=basis.device, dtype=basis.dtype)
    diff = basis.transpose(0, 1) @ basis - identity
    return diff.pow(2).mean()


def decorrelation_penalty(page_obs: torch.Tensor, obs_mask: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    weighted = page_obs * obs_mask
    covariance = weighted @ weighted.transpose(0, 1)
    projected = basis.transpose(0, 1) @ covariance @ basis
    off_diag = projected - torch.diag(torch.diagonal(projected))
    return off_diag.pow(2).mean()


def build_initial_basis_from_observation(
    page_obs: torch.Tensor,
    num_columns: int,
    explicit_rank: Optional[int] = None,
    sv_scale: float = 2.0,
) -> torch.Tensor:
    covariance = page_obs @ page_obs.transpose(0, 1)
    covariance = covariance.float()
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0.0)
    eigenvectors = eigenvectors[:, order]
    singular_values = torch.sqrt(eigenvalues)
    rank = infer_rank_from_singular_values(
        singular_values,
        num_columns=num_columns,
        explicit_rank=explicit_rank,
        sv_scale=sv_scale,
    )
    return orthonormalize_basis(eigenvectors[:, :rank], rank=rank)


def reconstruct_page_with_mask(page_obs: torch.Tensor, obs_mask: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    low_rank = low_rank_project(page_obs, basis)
    return obs_mask * page_obs + (1.0 - obs_mask) * low_rank


def denoise_raw_series_with_observation(observation: dict, basis: torch.Tensor, page_length: int) -> torch.Tensor:
    reconstructed_page = reconstruct_page_with_mask(
        observation["page_matrix_obs"],
        observation["obs_mask"],
        basis,
    )
    reconstructed_trimmed = page_matrix_to_series(
        reconstructed_page,
        page_length=page_length,
        num_nodes=observation["num_nodes"],
        num_features=observation["num_features"],
    )
    raw_series = observation["raw_series"]
    usable = int(observation["usable"])
    if usable == raw_series.size(0):
        return reconstructed_trimmed
    return torch.cat([reconstructed_trimmed, raw_series[usable:]], dim=0)
