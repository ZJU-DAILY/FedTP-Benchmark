"""Deterministic client-node partitions used by the FATE entry point.

The module deliberately has no FATE dependency so partitions can be inspected
or tested before an experiment starts.
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import SpectralEmbedding

import data.dividing as dividing
from lib.grid_partition import grid_hw_for_dataset, grid_rectangular_split


SUPPORTED_DATASETS = {"PeMS04": 307, "TaxiBJ": 1024}
# The PeMS04 scalability study uses these client counts.  TaxiBJ remains a
# four-client grid experiment because its fixed artifacts were designed for
# that setting.
SUPPORTED_CLIENT_COUNTS = {"PeMS04": {2, 4, 8, 16, 32}, "TaxiBJ": {4, 32}}
STRATEGIES = {"metis", "grid", "louvain", "geographic", "sensorid"}


def resolve_strategy(dataset_name: str, strategy: str | None, num_clients: int = 4) -> str:
    """Resolve the intentionally dataset-aware default strategy."""
    value = (strategy or "auto").lower()
    if value == "auto":
        if dataset_name == "PeMS04":
            # Keep the supplied 4/8-way METIS splits.  Larger power-of-two
            # settings refine the audited 4-way METIS root with deterministic
            # topology-aware bisection; 2 clients keeps the geographic split.
            return "metis" if int(num_clients) in {4, 8, 16, 32} else "geographic"
        if dataset_name == "TaxiBJ":
            return "grid"
    if value not in STRATEGIES:
        raise ValueError(f"Unsupported partition strategy: {strategy}")
    return value


def _validate_dataset_and_strategy(dataset_name: str, strategy: str, num_clients: int) -> int:
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            "partition_strategy is currently supported only for PeMS04 and TaxiBJ; "
            f"got {dataset_name}."
        )
    allowed_counts = SUPPORTED_CLIENT_COUNTS[dataset_name]
    if int(num_clients) not in allowed_counts:
        raise ValueError(
            f"{dataset_name} supports client counts {sorted(allowed_counts)}, got {num_clients}."
        )
    if strategy == "metis" and dataset_name != "PeMS04":
        raise ValueError("partition_strategy=metis is available only for PeMS04.")
    if strategy == "metis" and int(num_clients) not in {4, 8, 16, 32}:
        raise ValueError(
            "PeMS04 supports METIS-based splits only for 4, 8, 16, and 32 clients. "
            "Use partition_strategy=geographic, louvain, or sensorid for 2 clients."
        )
    if strategy == "grid" and dataset_name != "TaxiBJ":
        raise ValueError("partition_strategy=grid is available only for TaxiBJ.")
    return SUPPORTED_DATASETS[dataset_name]


def validate_partition(nodes_per: Sequence[Sequence[int]], num_nodes: int, num_clients: int = 4) -> List[List[int]]:
    """Return canonical groups or raise instead of silently changing a split."""
    if len(nodes_per) != num_clients:
        raise ValueError(f"Expected {num_clients} client partitions, got {len(nodes_per)}.")
    groups = [sorted(int(node) for node in group) for group in nodes_per]
    if any(not group for group in groups):
        raise ValueError("Client partition contains an empty node group.")
    flat = [node for group in groups for node in group]
    expected = list(range(num_nodes))
    if sorted(flat) != expected:
        duplicates = len(flat) - len(set(flat))
        missing = sorted(set(expected) - set(flat))
        unexpected = sorted(set(flat) - set(expected))
        raise ValueError(
            "Invalid partition: nodes must cover the global graph exactly once; "
            f"duplicates={duplicates}, missing={missing[:10]}, unexpected={unexpected[:10]}."
        )
    return sorted(groups, key=lambda group: group[0])


def _project_root(project_root: str | os.PathLike | None = None) -> Path:
    return Path(project_root) if project_root else Path(__file__).resolve().parents[1]


def _pems04_graph(project_root: Path) -> nx.Graph:
    distance_path = project_root / "data" / "PeMS04" / "distance.csv"
    if not distance_path.exists():
        raise FileNotFoundError(
            f"PeMS04 partitioning requires {distance_path}. "
            "Run this on the server that contains the PeMS04 dataset."
        )
    graph = nx.Graph()
    graph.add_nodes_from(range(SUPPORTED_DATASETS["PeMS04"]))
    with distance_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"from", "to", "cost"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"{distance_path} must contain columns from,to,cost.")
        for row in reader:
            src, dst = int(row["from"]), int(row["to"])
            if not (0 <= src < 307 and 0 <= dst < 307) or src == dst:
                continue
            cost = max(float(row["cost"]), 1e-8)
            similarity = 1.0 / cost
            if graph.has_edge(src, dst):
                graph[src][dst]["weight"] = max(graph[src][dst]["weight"], similarity)
            else:
                graph.add_edge(src, dst, weight=similarity, cost=cost)
    return graph


def _taxibj_graph() -> nx.Graph:
    height, width = grid_hw_for_dataset("TaxiBJ")
    graph = nx.Graph()
    graph.add_nodes_from(range(height * width))
    for row in range(height):
        for col in range(width):
            node = row * width + col
            if row + 1 < height:
                graph.add_edge(node, (row + 1) * width + col, weight=1.0)
            if col + 1 < width:
                graph.add_edge(node, row * width + col + 1, weight=1.0)
    return graph


def _graph_for_dataset(dataset_name: str, project_root: Path) -> nx.Graph:
    return _pems04_graph(project_root) if dataset_name == "PeMS04" else _taxibj_graph()


def _canonical(
    groups: Iterable[Iterable[int]], num_nodes: int, num_clients: int = 4
) -> List[List[int]]:
    return validate_partition(list(groups), num_nodes, num_clients)


def _balanced_id_split(nodes: Iterable[int]) -> Tuple[set[int], set[int]]:
    ordered = sorted(nodes)
    midpoint = len(ordered) // 2
    if midpoint == 0:
        raise ValueError("Cannot split a singleton community.")
    return set(ordered[:midpoint]), set(ordered[midpoint:])


def _split_group(graph: nx.Graph, group: set[int], seed: int) -> Tuple[set[int], set[int]]:
    if len(group) < 2:
        raise ValueError("Cannot split a singleton Louvain community.")
    induced = graph.subgraph(group).copy()
    if induced.number_of_edges() == 0:
        return _balanced_id_split(group)
    try:
        left, right = nx.algorithms.community.kernighan_lin_bisection(
            induced, weight="weight", seed=seed
        )
        if left and right:
            return set(left), set(right)
    except Exception:
        pass
    return _balanced_id_split(group)


def _merge_to_num_clients(graph: nx.Graph, groups: List[set[int]], num_clients: int) -> List[set[int]]:
    while len(groups) > num_clients:
        source_idx = min(range(len(groups)), key=lambda index: (len(groups[index]), min(groups[index])))
        source = groups[source_idx]
        scores = []
        for target_idx, target in enumerate(groups):
            if target_idx == source_idx:
                continue
            score = sum(
                float(data.get("weight", 1.0))
                for left in source
                for right, data in graph[left].items()
                if right in target
            )
            scores.append((score, -min(target), target_idx))
        # Highest affinity; when disconnected, the smallest node-id group wins.
        target_idx = max(scores)[2]
        groups[target_idx] = groups[target_idx] | source
        del groups[source_idx]
    return groups


def _merge_to_four(graph: nx.Graph, groups: List[set[int]]) -> List[set[int]]:
    """Backward-compatible alias for callers from the original 4-client study."""
    return _merge_to_num_clients(graph, groups, 4)


def _size_bounds(num_nodes: int, num_clients: int = 4, tolerance: float = 0.20) -> Tuple[int, int]:
    """Return inclusive client-size bounds around the average client size."""
    average = num_nodes / num_clients
    return math.ceil(average * (1.0 - tolerance)), math.floor(average * (1.0 + tolerance))


def _move_score(graph: nx.Graph, node: int, donor: set[int], receiver: set[int]) -> Tuple[float, float, int]:
    """Prefer boundary nodes that are more compatible with the receiver."""
    to_receiver = sum(
        float(data.get("weight", 1.0)) for neighbor, data in graph[node].items() if neighbor in receiver
    )
    to_donor = sum(
        float(data.get("weight", 1.0)) for neighbor, data in graph[node].items() if neighbor in donor
    )
    # Maximize receiver affinity, then minimize loss of donor affinity, then
    # choose the smallest ID to keep the result deterministic.
    return to_receiver, -to_donor, -node


def _balance_groups(graph: nx.Graph, groups: List[set[int]], num_nodes: int) -> List[set[int]]:
    """Keep community-based client sizes within +/-20% of average, deterministically.

    Louvain and geographic clustering do not optimize FL client size. Only
    communities outside the permitted range are adjusted, preserving natural
    variation. Boundary nodes are preferred; ID ordering is the deterministic
    fallback for disconnected graphs.
    """
    groups = [set(group) for group in groups]
    min_size, max_size = _size_bounds(num_nodes, len(groups))
    groups.sort(key=lambda group: min(group))

    # First ensure no client is too small.  This is separate from the upper
    # bound pass because four groups can all be <= max_size while one remains
    # below min_size.
    while True:
        receivers = [index for index, group in enumerate(groups) if len(group) < min_size]
        if not receivers:
            break
        donors = [index for index, group in enumerate(groups) if len(group) > min_size]
        if not donors:
            raise RuntimeError("Louvain balancing cannot satisfy the minimum client-size bound.")
        donor_idx = max(donors, key=lambda index: (len(groups[index]), -min(groups[index])))
        receiver_idx = min(receivers, key=lambda index: (len(groups[index]), min(groups[index])))
        donor, receiver = groups[donor_idx], groups[receiver_idx]
        candidates = [node for node in donor if len(donor) > 1]
        if not candidates:
            raise RuntimeError("Louvain balancing attempted to empty a client partition.")
        node = max(candidates, key=lambda item: _move_score(graph, item, donor, receiver))
        donor.remove(node)
        receiver.add(node)

    # Then reduce only groups that exceed the permitted upper bound.  Receivers
    # remain below that bound, rather than being forced to an exact average.
    while True:
        donors = [index for index, group in enumerate(groups) if len(group) > max_size]
        if not donors:
            break
        receivers = [index for index, group in enumerate(groups) if len(group) < max_size]
        if not receivers:
            raise RuntimeError("Louvain balancing cannot satisfy the maximum client-size bound.")
        donor_idx = max(donors, key=lambda index: (len(groups[index]), -min(groups[index])))
        receiver_idx = min(receivers, key=lambda index: (len(groups[index]), min(groups[index])))
        donor, receiver = groups[donor_idx], groups[receiver_idx]
        node = max(donor, key=lambda item: _move_score(graph, item, donor, receiver))
        donor.remove(node)
        receiver.add(node)
    return groups


def _louvain_partition(
    graph: nx.Graph, num_nodes: int, num_clients: int, seed: int
) -> List[List[int]]:
    try:
        import community as community_louvain
    except ImportError as exc:
        raise ImportError("Louvain partitioning requires the python-louvain package.") from exc

    labels = community_louvain.best_partition(graph, weight="weight", random_state=seed)
    buckets: Dict[int, set[int]] = defaultdict(set)
    for node in range(num_nodes):
        buckets[int(labels[node])].add(node)
    groups = list(buckets.values())
    groups = _merge_to_num_clients(graph, groups, num_clients)
    split_seed = seed
    while len(groups) < num_clients:
        candidates = [index for index, group in enumerate(groups) if len(group) >= 2]
        if not candidates:
            raise ValueError(f"Louvain could not create {num_clients} non-empty client partitions.")
        source_idx = max(candidates, key=lambda index: (len(groups[index]), -min(groups[index])))
        left, right = _split_group(graph, groups[source_idx], split_seed)
        split_seed += 1
        groups[source_idx] = left
        groups.append(right)
    return _canonical(_balance_groups(graph, groups, num_nodes), num_nodes, num_clients)


def _geographic_partition(
    dataset_name: str, graph: nx.Graph, num_nodes: int, num_clients: int, seed: int
) -> List[List[int]]:
    if dataset_name == "TaxiBJ":
        height, width = grid_hw_for_dataset(dataset_name)
        coordinates = np.array([(node // width, node % width) for node in range(height * width)], dtype=float)
    else:
        affinity = nx.to_numpy_array(graph, nodelist=range(num_nodes), weight="weight", dtype=float)
        np.fill_diagonal(affinity, 1.0)
        try:
            coordinates = SpectralEmbedding(
                n_components=2, affinity="precomputed", random_state=seed, eigen_solver="arpack"
            ).fit_transform(affinity)
        except Exception as exc:
            raise RuntimeError("Could not derive PeMS04 road-distance spatial embedding.") from exc
    labels = KMeans(n_clusters=num_clients, random_state=seed, n_init=20).fit_predict(coordinates)
    return _canonical(
        [[node for node, label in enumerate(labels) if label == client] for client in range(num_clients)],
        num_nodes,
        num_clients,
    )


def _fixed_metis_partition(num_clients: int, graph: nx.Graph | None = None, seed: int = 42) -> List[List[int]]:
    """Return a supplied PeMS04 METIS split, including its historical alias."""
    names = (
        f"PeMS04FLOW_{num_clients}p_metis",
        # Older files spelled the dataset as PeMSD4.  Retain that artifact
        # rather than silently replacing its published 8-client assignment.
        f"PeMSD4FLOW_{num_clients}p_metis",
    )
    for name in names:
        split = getattr(dividing, name, None)
        if split is not None:
            return [list(group) for group in split]
    if num_clients in {16, 32}:
        if graph is None:
            raise ValueError(
                f"PeMS04 {num_clients}-client METIS refinement requires the road graph."
            )
        # Refine the audited 4-way METIS assignment with deterministic graph
        # bisection, retaining the original METIS communities as its root.
        groups = [set(group) for group in _fixed_metis_partition(4)]
        split_seed = int(seed)
        while len(groups) < num_clients:
            index = max(range(len(groups)), key=lambda i: (len(groups[i]), -min(groups[i])))
            left, right = _split_group(graph, groups[index], split_seed)
            split_seed += 1
            groups[index] = left
            groups.append(right)
        return [sorted(group) for group in groups]
    raise ValueError(f"No fixed PeMS04 METIS split is available for {num_clients} clients.")


def _sensor_id_partition(num_nodes: int, num_clients: int) -> List[List[int]]:
    quotient, remainder = divmod(num_nodes, num_clients)
    groups, start = [], 0
    for client in range(num_clients):
        end = start + quotient + (1 if client < remainder else 0)
        groups.append(list(range(start, end)))
        start = end
    return groups


def build_partition(
    dataset_name: str,
    strategy: str | None,
    num_clients: int = 4,
    seed: int = 42,
    project_root: str | os.PathLike | None = None,
) -> Tuple[List[List[int]], Dict[str, object]]:
    """Build one validated, deterministic partition and its audit metadata."""
    selected = resolve_strategy(dataset_name, strategy, num_clients)
    num_nodes = _validate_dataset_and_strategy(dataset_name, selected, num_clients)
    root = _project_root(project_root)
    graph = None
    if selected == "metis":
        if num_clients in {16, 32}:
            graph = _graph_for_dataset(dataset_name, root)
        groups = _fixed_metis_partition(num_clients, graph=graph, seed=seed)
    elif selected == "grid":
        groups = grid_rectangular_split(dataset_name, num_clients)
    elif selected == "sensorid":
        groups = _sensor_id_partition(num_nodes, num_clients)
    else:
        graph = _graph_for_dataset(dataset_name, root)
        groups = (
            _louvain_partition(graph, num_nodes, num_clients, seed)
            if selected == "louvain"
            else _geographic_partition(dataset_name, graph, num_nodes, num_clients, seed)
        )
        if selected == "geographic":
            groups = _canonical(
                _balance_groups(graph, [set(group) for group in groups], num_nodes),
                num_nodes,
                num_clients,
            )
    groups = validate_partition(groups, num_nodes, num_clients)
    if graph is None:
        graph = _graph_for_dataset(dataset_name, root)
    node_to_client = {node: client for client, group in enumerate(groups) for node in group}
    total_edges = graph.number_of_edges()
    cross_edges = sum(1 for left, right in graph.edges() if node_to_client[left] != node_to_client[right])
    metadata: Dict[str, object] = {
        # Geographic v4 adds the same +/-20% size bound as Louvain. Other
        # strategies retain their existing v3 artifacts.
        "partition_version": 4 if selected == "geographic" else 3,
        "dataset": dataset_name,
        "feature": "flow",
        "strategy": selected,
        "seed": int(seed),
        "num_clients": int(num_clients),
        "num_nodes": num_nodes,
        "nodes_per": groups,
        "client_sizes": [len(group) for group in groups],
        "total_edges": int(total_edges),
        "cross_client_edges": int(cross_edges),
        "cross_client_edge_ratio": (float(cross_edges) / total_edges) if total_edges else 0.0,
    }
    if selected == "metis" and num_clients in {16, 32}:
        metadata["partition_algorithm"] = "hierarchical_metis_seeded_topology_bisection"
    if selected in {"louvain", "geographic"}:
        min_size, max_size = _size_bounds(num_nodes, num_clients)
        metadata["balance_policy"] = "deterministic_boundary_transfer_within_20_percent"
        metadata["client_size_bounds"] = {"min": min_size, "max": max_size}
    return groups, metadata


def write_partition_artifacts(
    metadata: Dict[str, object], project_root: str | os.PathLike | None = None,
    train_ratio: float = 0.7, val_ratio: float = 0.1, test_ratio: float = 0.2, t_in: int = 12, t_out: int = 3,
) -> Tuple[Path, Path]:
    root = _project_root(project_root)
    strategy = str(metadata["strategy"])
    output_dir = root / "partition" / strategy
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{metadata['dataset']}_{metadata['feature']}_{metadata['num_clients']}clients_seed{metadata['seed']}"
    payload = dict(metadata)
    payload.update({"train_ratio": train_ratio, "val_ratio": val_ratio, "test_ratio": test_ratio, "t_in": t_in, "t_out": t_out})
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}_nodes.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["client_id", "node_id"])
        for client, group in enumerate(metadata["nodes_per"]):
            writer.writerows((client, node) for node in group)
    return json_path, csv_path


def load_partition_artifact(
    dataset_name: str,
    strategy: str | None,
    num_clients: int = 4,
    seed: int = 42,
    project_root: str | os.PathLike | None = None,
) -> Tuple[List[List[int]], Dict[str, object]]:
    """Load a previously audited partition instead of recomputing it for training."""
    selected = resolve_strategy(dataset_name, strategy, num_clients)
    num_nodes = _validate_dataset_and_strategy(dataset_name, selected, num_clients)
    root = _project_root(project_root)
    stem = f"{dataset_name}_flow_{num_clients}clients_seed{seed}"
    json_path = root / "partition" / selected / f"{stem}.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Partition artifact is missing: {json_path}. "
            "Generate it first with `python preprocess_partitions.py`."
        )
    with json_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected = {
        "partition_version": 4 if selected == "geographic" else 3,
        "dataset": dataset_name,
        "feature": "flow",
        "strategy": selected,
        "seed": int(seed),
        "num_clients": int(num_clients),
        "num_nodes": int(num_nodes),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Partition artifact metadata mismatch for {key}: "
                f"expected {value!r}, got {metadata.get(key)!r} ({json_path})."
            )
    groups = validate_partition(metadata.get("nodes_per", []), num_nodes, num_clients)
    metadata["nodes_per"] = groups
    return groups, metadata
