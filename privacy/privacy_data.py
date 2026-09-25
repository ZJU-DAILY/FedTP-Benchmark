from __future__ import annotations

import importlib
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence

import torch


TOTAL_NODES_MAP = {
    "PeMS03": 358,
    "PeMS04": 307,
    "PeMSD7": 228,
    "PeMS07": 228,
    "PeMS08": 170,
    "TaxiBJ": 1024,
    "TaxiNYC": 75,
    "BikeNYC": 128,
}

GRID_HW_MAP = {
    "TaxiBJ": (32, 32),
    "TaxiNYC": (15, 5),
    "BikeNYC": (16, 8),
}


@dataclass
class PrivacyBatch:
    real_x: torch.Tensor
    real_y: torch.Tensor
    x_ext: torch.Tensor | None
    scaler: Any
    edge_index: Any
    selected_nodes: Sequence[int]
    nodes_per: List[List[int]]
    split: str
    sample_index: int
    dataset: Any
    ufcl_sequence_steps: int = 1
    ufcl_train_batch_size: int = 0


def normalize_dataset_name(dataset_name: str) -> str:
    if dataset_name == "PeMS07":
        return "PeMSD7"
    return dataset_name


def _equal_node_split(total_nodes: int, num_clients: int) -> List[List[int]]:
    quotient, remainder = divmod(total_nodes, num_clients)
    nodes_per = []
    start_idx = 0
    for client_id in range(num_clients):
        count = quotient + (1 if client_id < remainder else 0)
        nodes_per.append(list(range(start_idx, start_idx + count)))
        start_idx += count
    return nodes_per


def _grid_quadrant_split(dataset_name: str) -> List[List[int]]:
    height, width = GRID_HW_MAP[dataset_name]
    half_h, half_w = math.ceil(height / 2), math.ceil(width / 2)
    nodes_per = [[], [], [], []]
    for row in range(height):
        for col in range(width):
            node_id = row * width + col
            if row < half_h and col < half_w:
                nodes_per[0].append(node_id)
            elif row < half_h and col >= half_w:
                nodes_per[1].append(node_id)
            elif row >= half_h and col < half_w:
                nodes_per[2].append(node_id)
            else:
                nodes_per[3].append(node_id)
    return nodes_per


def _load_training_partition_if_available(
    dataset_name: str,
    num_clients: int,
    seed: int,
    strategy: str | None,
) -> List[List[int]] | None:
    """Load the exact partition artifact used by ``run/fate_main.py``.

    Privacy replay must select the same client node IDs as the run that wrote
    its trace.  In particular, the legacy ``data.dividing`` METIS constant is
    not necessarily identical to the audited FATE partition artifact.
    """
    supported_counts = {"PeMS04": {2, 4, 8, 16, 32}, "TaxiBJ": {4, 32}}
    if dataset_name not in supported_counts or int(num_clients) not in supported_counts[dataset_name]:
        return None

    from lib.partitioning import load_partition_artifact, resolve_strategy

    project_root = Path(__file__).resolve().parents[1]
    selected = resolve_strategy(dataset_name, strategy, num_clients)
    artifact = project_root / "partition" / selected / (
        f"{dataset_name}_flow_{num_clients}clients_seed{seed}.json"
    )
    if not artifact.exists():
        return None
    groups, _ = load_partition_artifact(
        dataset_name=dataset_name,
        strategy=strategy,
        num_clients=num_clients,
        seed=seed,
        project_root=project_root,
    )
    return [list(nodes) for nodes in groups]


def resolve_nodes_per(
    dataset_name: str,
    num_clients: int,
    *,
    seed: int = 42,
    partition_strategy: str | None = "auto",
) -> List[List[int]]:
    dataset_name = normalize_dataset_name(dataset_name)

    # Prefer the artifact consumed by the FATE launcher.  This prevents a
    # replay trace with e.g. 77 client nodes from being matched against a
    # legacy 76-node privacy batch.
    artifact_nodes = _load_training_partition_if_available(
        dataset_name, num_clients, seed, partition_strategy
    )
    if artifact_nodes is not None:
        return artifact_nodes

    split_var_name = f"{dataset_name}FLOW_{num_clients}p_metis"

    try:
        dividing = importlib.import_module("data.dividing")
        split = getattr(dividing, split_var_name)
        return [list(nodes) for nodes in split]
    except Exception:
        pass

    if dataset_name in GRID_HW_MAP and num_clients == 4:
        return _grid_quadrant_split(dataset_name)

    total_nodes = TOTAL_NODES_MAP.get(dataset_name)
    if total_nodes is None:
        raise ValueError(
            f"Unknown node count for dataset {dataset_name!r}. "
            "Add it to TOTAL_NODES_MAP or provide a data.dividing split."
        )
    return _equal_node_split(total_nodes, num_clients)


def _get_dataset_by_split(train_set: Any, val_set: Any, test_set: Any, split: str) -> Any:
    if split == "train":
        return train_set
    if split == "val":
        return val_set
    if split == "test":
        return test_set
    raise ValueError(f"Unsupported split: {split}")


def _load_graph_sets(args: Any, selected_nodes: Sequence[int]):
    try:
        from lib.load_dataset import load_dataset
    except ModuleNotFoundError as exc:
        if exc.name != "data.dataset":
            raise

        # lib.load_dataset imports the grid dataset class at module import time.
        # Graph-only privacy runs do not use it, so this lightweight shim lets
        # PeMS experiments reuse the existing graph loader in repos where the
        # grid dataset package is not present.
        data_pkg = sys.modules.setdefault("data", types.ModuleType("data"))
        if not hasattr(data_pkg, "__path__"):
            data_pkg.__path__ = []
        dataset_mod = types.ModuleType("data.dataset")

        class _MissingUnifiedTrafficDataset:
            def __init__(self, *_, **__):
                raise ModuleNotFoundError(
                    "data.dataset.UnifiedTrafficDataset is required for grid datasets."
                )

        dataset_mod.UnifiedTrafficDataset = _MissingUnifiedTrafficDataset
        sys.modules["data.dataset"] = dataset_mod
        from lib.load_dataset import load_dataset

    return load_dataset(
        dataset_name=normalize_dataset_name(args.dataset_name),
        feature_type=args.feature_type,
        normalizer=args.normalizer,
        T_in=args.t_in,
        T_out=args.t_out,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        return_edge_index=True,
        device="cpu",
        selected_nodes=selected_nodes,
    )


def _load_grid_sets(args: Any, selected_nodes: Sequence[int]):
    from lib.load_dataset import load_grid_dataset_for_fedstn

    return load_grid_dataset_for_fedstn(
        dataset_name=args.dataset_name,
        t_in=args.t_in,
        t_out=args.t_out,
        device="cpu",
        selected_nodes=selected_nodes,
        model_name=args.model,
    )


def _stack_graph_or_grid_samples(samples: list[Any], dataset_name: str, args: Any):
    if not samples:
        raise ValueError("Cannot stack an empty sample list.")

    x_ext = None
    if dataset_name in GRID_HW_MAP and len(samples[0]) >= 5:
        x_list = [sample[0].permute(1, 0, 2).contiguous() for sample in samples]
        y_list = [sample[-1].permute(1, 0, 2).contiguous() for sample in samples]
        x_ext_list = [sample[3].contiguous() for sample in samples]
        x = torch.stack(x_list, dim=0)
        y = torch.stack(y_list, dim=0)
        x_ext = torch.stack(x_ext_list, dim=0)
        args.input_dim = int(x.shape[-1])
        args.output_dim = int(y.shape[-1])
        args.ext_dim = int(x_ext.shape[-1])
        return x, y, x_ext

    x_list = []
    y_list = []
    for sample in samples:
        x_item = sample[0]
        y_item = sample[-1]
        if x_item.dim() == 4 and x_item.shape[0] == 1:
            x_item = x_item.squeeze(0)
        if y_item.dim() == 4 and y_item.shape[0] == 1:
            y_item = y_item.squeeze(0)
        x_list.append(x_item.contiguous())
        y_list.append(y_item.contiguous())
    x = torch.stack(x_list, dim=0)
    y = torch.stack(y_list, dim=0)
    return x, y, None


def load_privacy_batch(args: Any) -> PrivacyBatch:
    dataset_name = normalize_dataset_name(args.dataset_name)
    nodes_per = resolve_nodes_per(
        dataset_name,
        args.num_clients,
        seed=int(getattr(args, "seed", 42)),
        partition_strategy=getattr(args, "partition_strategy", "auto"),
    )

    if args.client_rank < 0 or args.client_rank >= len(nodes_per):
        raise ValueError(
            f"client_rank={args.client_rank} is outside available clients 0..{len(nodes_per) - 1}"
        )

    selected_nodes = nodes_per[args.client_rank]
    if dataset_name in GRID_HW_MAP:
        train_set, val_set, test_set, edge_index, scaler = _load_grid_sets(args, selected_nodes)
    else:
        train_set, val_set, test_set, edge_index, scaler = _load_graph_sets(args, selected_nodes)

    dataset = _get_dataset_by_split(train_set, val_set, test_set, args.split)
    if len(dataset) == 0:
        raise RuntimeError(f"{args.split} dataset is empty for {dataset_name} client {args.client_rank}")
    batch_size = max(1, int(getattr(args, "batch_size", 1)))
    # UFCL uploads an update after a sequence of local mini-batches.  One
    # mini-batch only seeds its synthetic replay buffer; the replay first
    # changes the optimization on the following mini-batch.  Keep this local
    # sequence explicit so the attack matches that ordering.
    is_ufcl = str(getattr(args, "model", "")).upper() == "UFCL"
    # The protocol-faithful UFCL model-update attack needs an ordered local
    # trajectory.  The diagnostic per-batch gradient upper bound observes one
    # ordinary training batch instead, so it must not silently append replay
    # seed batches to the victim batch.
    is_ufcl_update_attack = is_ufcl and str(getattr(args, "attack", "auto")).lower() != "gradient"
    ufcl_sequence_steps = (
        max(1, int(getattr(args, "local_update_steps", 1))) if is_ufcl_update_attack else 1
    )
    requested_samples = batch_size * ufcl_sequence_steps
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(
            f"sample_index={args.sample_index} is outside {args.split} dataset length {len(dataset)}"
        )
    end_index = args.sample_index + requested_samples
    if end_index > len(dataset):
        raise IndexError(
            f"Requested batch sample range [{args.sample_index}, {end_index}) exceeds "
            f"{args.split} dataset length {len(dataset)}"
        )

    samples = [dataset[idx] for idx in range(args.sample_index, end_index)]
    sample = samples[0]
    if not isinstance(sample, (tuple, list)) or len(sample) < 2:
        raise TypeError(
            "Expected dataset sample to be a tuple/list with at least (x, y). "
            f"Got {type(sample)!r}."
        )

    x, y, x_ext = _stack_graph_or_grid_samples(samples, dataset_name, args)

    x = x.to(args.device).float().contiguous()
    y = y.to(args.device).float().contiguous()
    if x_ext is not None:
        x_ext = x_ext.to(args.device).float().contiguous()

    if getattr(args, "attack_node_index", -1) >= 0:
        node_idx = int(args.attack_node_index)
        if x.dim() != 4 or y.dim() != 4:
            raise ValueError("attack_node_index expects 4D x/y tensors.")
        if node_idx >= x.shape[1]:
            raise IndexError(
                f"attack_node_index={node_idx} is outside local node dimension {x.shape[1]}"
            )
        x = x[:, node_idx:node_idx + 1, :, :].contiguous()
        y = y[:, node_idx:node_idx + 1, :, :].contiguous()
        selected_nodes = [selected_nodes[node_idx]]

    return PrivacyBatch(
        real_x=x,
        real_y=y,
        x_ext=x_ext,
        scaler=scaler,
        edge_index=edge_index,
        selected_nodes=selected_nodes,
        nodes_per=nodes_per,
        split=args.split,
        sample_index=args.sample_index,
        dataset=dataset,
        ufcl_sequence_steps=ufcl_sequence_steps,
        ufcl_train_batch_size=batch_size if is_ufcl_update_attack else 0,
    )
