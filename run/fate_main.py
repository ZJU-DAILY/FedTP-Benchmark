import json
import io
import os
import sys
import math
import copy
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
# Do not force CUDA_VISIBLE_DEVICES here. Device selection is controlled by
# the caller through CUDA_VISIBLE_DEVICES and/or --device.

for _thread_env_key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "TBB_NUM_THREADS",
):
    os.environ.setdefault(_thread_env_key, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["WANDB_DISABLED"] = "true"

file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)
from lib.utils import init_seed

import networkx as nx
import community as community_louvain 
import time
import torch
torch.backends.cudnn.benchmark = True
import numpy as np
import torch.nn as nn
from datetime import datetime
from fate.arch import Context


def _log_cuda_environment(prefix="[DeviceCheck]"):
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    try:
        cuda_available = torch.cuda.is_available()
    except Exception as exc:
        cuda_available = f"error:{exc}"
    try:
        device_count = torch.cuda.device_count()
    except Exception as exc:
        device_count = f"error:{exc}"
    print(
        f"{prefix} requested_device={getattr(args, 'device', '<unknown>')} "
        f"CUDA_VISIBLE_DEVICES={cuda_visible} "
        f"cuda_available={cuda_available} "
        f"device_count={device_count}",
        flush=True,
    )
from fate.arch.launchers.multiprocess_launcher import launch
from fate.ml.nn.homo.fedavg import FedAVGArguments, FedAVGClient, FedAVGServer, TrainingArguments
from transformers.trainer import Trainer
from torch.utils.data import TensorDataset, DataLoader
from lib.ufcl_trainer import UFCLTrainer
from lib.ufcl_client import UFCLFedAVGClient
from lib.ufcl_dp_trainer import train_ufcl_dp_task
from lib.fedgru_trainer import FedGRUFedAVGClient
from lib.fed4tp_trainer import train_fed4tp_task
from lib.fgnneh_trainer import train_fgnneh_task
from lib.custom_loss import FedAGATLoss
from lib.stagcn_trainer import train_stagcn_ec_task
from lib.fcfedgcn_trainer import train_fcfedgcn_task
from lib.fcgcn_dp_trainer import train_fedavg_dp_task
from lib.fedtps_trainer import train_fedtps_task
from lib.refol_trainer import train_refol_distributed
from lib.fed4tp_trainer import train_fed4tp_task 

from config_args import args
if bool(getattr(args, "privacy_trace_only", False)):
    # Existing HE efficiency scripts use their benchmark batch size.  A
    # reconstruction trace must instead describe exactly one deterministic
    # local record, irrespective of that script's normal batch setting.
    args.batch_size = 1
from privacy.protection import l2_norm
from privacy.runtime_protection import (
    install_runtime_protection, protected_arbiter_put,
    record_he_ttp_downlink, unprotect_he_ttp_payload,
)
from privacy.attack_trace import capture_he_sa_aggregate, capture_he_sa_arbiter_insider, capture_dp_reconstruction_trace, capture_revised_quantized_prediction
from privacy.he_backend import (
    ciphertext_bytes, decrypt_tree, encrypt_tree, export_public_key,
    generate_keypair, homomorphic_sum_tree, import_public_key,
)
from privacy.ckks_backend import (
    ciphertext_bytes as ckks_ciphertext_bytes,
    decrypt_tree as ckks_decrypt_tree,
    encrypt_tree as ckks_encrypt_tree,
    export_public_context as ckks_export_public_context,
    generate_context as ckks_generate_context,
    homomorphic_sum_tree as ckks_homomorphic_sum_tree,
    homomorphic_weighted_sum_tree as ckks_homomorphic_weighted_sum_tree,
    import_context as ckks_import_context,
)

from data.dividing import *


class _PrivacyTraceSingleSampleDataset:
    """A one-sample view that preserves custom dataset attributes for trainers."""

    def __init__(self, dataset, sample_index):
        if len(dataset) <= 0:
            raise ValueError("Cannot create a privacy trace view from an empty dataset.")
        index = int(sample_index)
        if index < 0 or index >= len(dataset):
            raise IndexError(
                f"privacy_trace_sample_index={index} is outside dataset length {len(dataset)}"
            )
        self._dataset = dataset
        self._index = index

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if int(index) != 0:
            raise IndexError(index)
        return self._dataset[self._index]

    def __getattr__(self, name):
        return getattr(self._dataset, name)


# Names used in experiment tables are not always the names of the registered
# model classes.  Normalize them at the entry point as well as in the shell
# launcher so direct `python run/fate_main.py` invocations behave identically.
_MODEL_ALIASES = {
    "FedAGAT": "ASTGAT",
}


_CROSS_DEVICE_NODE_CLIENT_MODELS = {
    "CNFGNN",
    "REFOL",
    "FedOSTC",
    "FedGODE",
    "FedSTG",
}

_CROSS_DEVICE_TOTAL_NODES = {
    "PeMS03": 358,
    "PeMS04": 307,
    "PeMSD7": 228,
    "PeMS08": 170,
}


def _uses_cross_device_node_clients():
    """Return whether this run uses one graph node per logical client."""
    model_name = _MODEL_ALIASES.get(getattr(args, "model", ""), getattr(args, "model", ""))
    return (
        model_name in _CROSS_DEVICE_NODE_CLIENT_MODELS
        or str(getattr(args, "trainer_mode", "")).lower() == "sfl"
    )


def _cross_device_dataset_node_count(dataset_name):
    matched = next(
        (count for name, count in _CROSS_DEVICE_TOTAL_NODES.items() if name in str(dataset_name)),
        None,
    )
    if matched is None:
        raise ValueError(
            "Cross-device node-client multiplexing currently supports "
            f"{sorted(_CROSS_DEVICE_TOTAL_NODES)}; got dataset={dataset_name!r}."
        )
    return int(matched)


def _configure_cross_device_node_clients():
    """Separate logical node-clients from physical FATE worker processes.

    Existing launch scripts keep using ``--num_clients`` to select their
    physical process count.  Only the six cross-device methods opt into this
    translation.  Once configured, their trainers continue to see
    ``args.num_clients == K`` and ``args.nodes_per == [[0], ..., [K-1]]``.
    """
    if not _uses_cross_device_node_clients():
        args.cross_device_multiplex = False
        args.num_workers = int(args.num_clients)
        return

    if bool(getattr(args, "cross_device_multiplex", False)):
        return

    requested_workers = int(getattr(args, "cross_device_workers", 0) or args.num_clients)
    logical_clients = _cross_device_dataset_node_count(args.dataset_name)
    if requested_workers < 1:
        raise ValueError("Cross-device worker count must be at least 1.")
    if requested_workers > logical_clients:
        raise ValueError(
            f"Cross-device worker count P={requested_workers} exceeds logical clients K={logical_clients}."
        )

    args.cross_device_multiplex = True
    args.requested_num_clients = int(args.num_clients)
    args.num_workers = requested_workers
    args.logical_num_clients = logical_clients
    args.num_clients = logical_clients
    args.nodes_per = [[node_id] for node_id in range(logical_clients)]
    print(
        f"[CrossDeviceMux] model={args.model} logical_clients(K)={logical_clients} "
        f"workers(P)={requested_workers} mapping=one_graph_node_per_logical_client",
        flush=True,
    )


_CDMUX_GUEST_UPLOAD_PREFIX = "__cdmux_guest_up__"
_CDMUX_HOST_UPLOAD_PREFIX = "__cdmux_host_up__"
_CDMUX_GUEST_DOWN_PREFIX = "__cdmux_guest_down__"
_CDMUX_HOST_DOWN_PREFIX = "__cdmux_host_down__"
_CDMUX_PEER_HOST_UP_PREFIX = "__cdmux_peer_host_up__"
_CDMUX_PEER_HOST_DOWN_PREFIX = "__cdmux_peer_host_down__"
_CDMUX_PACKET_MARKER = "cross_device_logical_clients_v1"


def _cross_device_worker_assignments(logical_clients, workers):
    """Return a deterministic, balanced contiguous assignment of client ids."""
    logical_clients, workers = int(logical_clients), int(workers)
    if logical_clients < 1 or workers < 1 or workers > logical_clients:
        raise ValueError(
            f"Invalid cross-device assignment K={logical_clients}, P={workers}."
        )
    quotient, remainder = divmod(logical_clients, workers)
    assignments, start = [], 0
    for worker_id in range(workers):
        count = quotient + (1 if worker_id < remainder else 0)
        assignments.append(list(range(start, start + count)))
        start += count
    return assignments


def _cdmux_packet(worker_id, values):
    return {
        "marker": _CDMUX_PACKET_MARKER,
        "worker_id": int(worker_id),
        "values": {int(client_id): value for client_id, value in values.items()},
    }


def _cdmux_unpack_packet(packet, expected_worker_id=None):
    if not isinstance(packet, dict) or packet.get("marker") != _CDMUX_PACKET_MARKER:
        raise RuntimeError(
            "Cross-device multiplexing received an invalid worker packet. "
            f"Expected marker={_CDMUX_PACKET_MARKER!r}, got type={type(packet)!r}."
        )
    worker_id = int(packet.get("worker_id", -1))
    if expected_worker_id is not None and worker_id != int(expected_worker_id):
        raise RuntimeError(
            f"Cross-device packet worker mismatch: expected {expected_worker_id}, got {worker_id}."
        )
    values = packet.get("values")
    if not isinstance(values, dict):
        raise RuntimeError("Cross-device worker packet has no logical-client mapping.")
    return {int(client_id): value for client_id, value in values.items()}


class _CrossDeviceWorkerMux:
    """Multiplex logical-client traffic through one physical FATE party."""

    def __init__(self, physical_ctx, worker_id, logical_client_ids, assignments):
        self.physical_ctx = physical_ctx
        self.worker_id = int(worker_id)
        self.logical_client_ids = tuple(int(value) for value in logical_client_ids)
        self.host_client_ids = tuple(value for value in self.logical_client_ids if value != 0)
        self.assignments = assignments
        self._upload_lock = threading.Lock()
        self._host_uploads = {}
        self._download_lock = threading.Lock()
        self._host_downloads = {}
        self._peer_upload_lock = threading.Lock()
        self._peer_uploads = {}
        self._peer_down_lock = threading.Lock()
        self._peer_downs = {}

    def put(self, logical_client_id, tag, value):
        logical_client_id = int(logical_client_id)
        if logical_client_id == 0:
            return self.physical_ctx.arbiter.put(
                f"{_CDMUX_GUEST_UPLOAD_PREFIX}{tag}", value
            )

        should_send = False
        with self._upload_lock:
            state = self._host_uploads.get(tag)
            if state is None:
                state = {
                    "values": {},
                    "event": threading.Event(),
                    "error": None,
                    "remaining": len(self.host_client_ids),
                }
                self._host_uploads[tag] = state
            values = state["values"]
            if logical_client_id in values:
                raise RuntimeError(
                    f"Logical client {logical_client_id} uploaded tag {tag!r} more than once."
                )
            values[logical_client_id] = value
            if len(values) == len(self.host_client_ids):
                missing = set(self.host_client_ids).difference(values)
                if missing:
                    raise RuntimeError(
                        f"Worker {self.worker_id} is missing logical clients {sorted(missing)} for tag {tag!r}."
                    )
                should_send = True

        if should_send:
            try:
                self.physical_ctx.arbiter.put(
                    f"{_CDMUX_HOST_UPLOAD_PREFIX}{tag}",
                    _cdmux_packet(self.worker_id, values),
                )
            except BaseException as exc:
                state["error"] = exc
            finally:
                state["event"].set()
        else:
            # Do not let a fast logical client advance to its receive phase
            # before this worker's complete upload packet has been emitted.
            state["event"].wait()

        if state["error"] is not None:
            raise state["error"]
        with self._upload_lock:
            state["remaining"] -= 1
            if state["remaining"] == 0:
                self._host_uploads.pop(tag, None)
        return None

    def get(self, logical_client_id, tag):
        logical_client_id = int(logical_client_id)
        if logical_client_id == 0:
            return self.physical_ctx.arbiter.get(
                f"{_CDMUX_GUEST_DOWN_PREFIX}{tag}"
            )

        with self._download_lock:
            state = self._host_downloads.get(tag)
            if state is None:
                state = {
                    "event": threading.Event(),
                    "fetching": True,
                    "value": None,
                    "error": None,
                    "remaining": len(self.host_client_ids),
                }
                self._host_downloads[tag] = state
                is_fetcher = True
            else:
                is_fetcher = False

        if is_fetcher:
            try:
                state["value"] = self.physical_ctx.arbiter.get(
                    f"{_CDMUX_HOST_DOWN_PREFIX}{tag}"
                )
            except BaseException as exc:
                state["error"] = exc
            finally:
                state["event"].set()
        else:
            state["event"].wait()

        if state["error"] is not None:
            raise state["error"]
        value = state["value"]
        with self._download_lock:
            state["remaining"] -= 1
            if state["remaining"] == 0:
                self._host_downloads.pop(tag, None)
        return value

    def peer_host_put(self, logical_client_id, tag, value):
        """Upload one logical host value to logical guest 0."""
        logical_client_id = int(logical_client_id)
        if logical_client_id == 0:
            raise RuntimeError("Logical guest cannot use the host-to-guest upload endpoint.")

        should_publish = False
        with self._peer_upload_lock:
            state = self._peer_uploads.get(tag)
            if state is None:
                state = {
                    "values": {},
                    "event": threading.Event(),
                    "error": None,
                    "remaining": len(self.host_client_ids),
                }
                self._peer_uploads[tag] = state
            values = state["values"]
            if logical_client_id in values:
                raise RuntimeError(
                    f"Logical host {logical_client_id} sent peer tag {tag!r} more than once."
                )
            values[logical_client_id] = value
            if len(values) == len(self.host_client_ids):
                should_publish = True

        if should_publish:
            try:
                if self.worker_id == 0:
                    # Logical guest 0 lives in this same process and consumes
                    # the completed mapping directly in peer_guest_get().
                    pass
                else:
                    self.physical_ctx.guest.put(
                        f"{_CDMUX_PEER_HOST_UP_PREFIX}{tag}",
                        _cdmux_packet(self.worker_id, values),
                    )
            except BaseException as exc:
                state["error"] = exc
            finally:
                state["event"].set()
        else:
            state["event"].wait()

        if state["error"] is not None:
            raise state["error"]
        if self.worker_id != 0:
            with self._peer_upload_lock:
                state["remaining"] -= 1
                if state["remaining"] == 0:
                    self._peer_uploads.pop(tag, None)
        return None

    def peer_guest_get(self, tag):
        """Collect K-1 logical-host values for logical guest 0."""
        logical_values = {}
        if self.host_client_ids:
            with self._peer_upload_lock:
                state = self._peer_uploads.get(tag)
                if state is None:
                    state = {
                        "values": {},
                        "event": threading.Event(),
                        "error": None,
                        "remaining": len(self.host_client_ids),
                    }
                    self._peer_uploads[tag] = state
            state["event"].wait()
            if state["error"] is not None:
                raise state["error"]
            logical_values.update(state["values"])
            with self._peer_upload_lock:
                self._peer_uploads.pop(tag, None)

        if len(self.assignments) > 1:
            packets = self.physical_ctx.hosts.get(
                f"{_CDMUX_PEER_HOST_UP_PREFIX}{tag}"
            )
            packets = packets if isinstance(packets, list) else [packets]
            if len(packets) != len(self.assignments) - 1:
                raise RuntimeError(
                    f"Expected {len(self.assignments) - 1} peer packets for tag {tag!r}, "
                    f"received {len(packets)}."
                )
            for worker_id, packet in enumerate(packets, start=1):
                logical_values.update(
                    _cdmux_unpack_packet(packet, expected_worker_id=worker_id)
                )

        expected_ids = set(range(1, sum(len(group) for group in self.assignments)))
        if set(logical_values) != expected_ids:
            raise RuntimeError(
                f"Peer upload mismatch for tag {tag!r}: "
                f"expected={sorted(expected_ids)}, got={sorted(logical_values)}."
            )
        return [logical_values[client_id] for client_id in sorted(expected_ids)]

    def peer_guest_put(self, tag, value):
        """Broadcast a logical-guest response to every logical host."""
        if self.host_client_ids:
            with self._peer_down_lock:
                state = self._peer_downs.get(tag)
                if state is None:
                    state = {
                        "event": threading.Event(),
                        "value": None,
                        "error": None,
                        "remaining": len(self.host_client_ids),
                    }
                    self._peer_downs[tag] = state
                state["value"] = value
                state["event"].set()
        if len(self.assignments) > 1:
            return self.physical_ctx.hosts.put(
                f"{_CDMUX_PEER_HOST_DOWN_PREFIX}{tag}", value
            )
        return None

    def peer_host_get(self, logical_client_id, tag):
        """Receive the logical-guest response for one logical host."""
        logical_client_id = int(logical_client_id)
        if self.worker_id == 0:
            with self._peer_down_lock:
                state = self._peer_downs.get(tag)
                if state is None:
                    state = {
                        "event": threading.Event(),
                        "value": None,
                        "error": None,
                        "remaining": len(self.host_client_ids),
                    }
                    self._peer_downs[tag] = state
            state["event"].wait()
        else:
            with self._peer_down_lock:
                state = self._peer_downs.get(tag)
                if state is None:
                    state = {
                        "event": threading.Event(),
                        "value": None,
                        "error": None,
                        "remaining": len(self.host_client_ids),
                    }
                    self._peer_downs[tag] = state
                    is_fetcher = True
                else:
                    is_fetcher = False
            if is_fetcher:
                try:
                    state["value"] = self.physical_ctx.guest.get(
                        f"{_CDMUX_PEER_HOST_DOWN_PREFIX}{tag}"
                    )
                except BaseException as exc:
                    state["error"] = exc
                finally:
                    state["event"].set()
            else:
                state["event"].wait()

        if state["error"] is not None:
            raise state["error"]
        value = state["value"]
        with self._peer_down_lock:
            state["remaining"] -= 1
            if state["remaining"] == 0:
                self._peer_downs.pop(tag, None)
        return value


class _CrossDeviceClientArbiterEndpoint:
    def __init__(self, mux, logical_client_id):
        self._mux = mux
        self._logical_client_id = int(logical_client_id)

    def put(self, tag, value, *unused_args, **unused_kwargs):
        return self._mux.put(self._logical_client_id, tag, value)

    def get(self, tag, *unused_args, **unused_kwargs):
        return self._mux.get(self._logical_client_id, tag)


class _CrossDeviceLogicalGuestHostsEndpoint:
    """Logical guest-0 view of all K-1 logical hosts."""

    def __init__(self, mux):
        self._mux = mux

    def get(self, tag, *unused_args, **unused_kwargs):
        return self._mux.peer_guest_get(tag)

    def put(self, tag, value, *unused_args, **unused_kwargs):
        return self._mux.peer_guest_put(tag, value)


class _CrossDeviceLogicalHostGuestEndpoint:
    """One logical host's view of logical guest 0."""

    def __init__(self, mux, logical_client_id):
        self._mux = mux
        self._logical_client_id = int(logical_client_id)

    def put(self, tag, value, *unused_args, **unused_kwargs):
        return self._mux.peer_host_put(self._logical_client_id, tag, value)

    def get(self, tag, *unused_args, **unused_kwargs):
        return self._mux.peer_host_get(self._logical_client_id, tag)


class _CrossDeviceClientContext:
    def __init__(self, physical_ctx, mux, logical_client_id):
        self._physical_ctx = physical_ctx
        self.rank = int(logical_client_id)
        self.is_on_arbiter = False
        self.is_on_guest = self.rank == 0
        self.is_on_host = self.rank != 0
        self.arbiter = _CrossDeviceClientArbiterEndpoint(mux, self.rank)
        if self.is_on_guest:
            self.hosts = _CrossDeviceLogicalGuestHostsEndpoint(mux)
        else:
            self.guest = _CrossDeviceLogicalHostGuestEndpoint(mux, self.rank)

    def __getattr__(self, name):
        return getattr(self._physical_ctx, name)


class _CrossDeviceServerGuestEndpoint:
    def __init__(self, server_ctx):
        self._server_ctx = server_ctx

    def get(self, tag, *unused_args, **unused_kwargs):
        return self._server_ctx._physical_ctx.guest.get(
            f"{_CDMUX_GUEST_UPLOAD_PREFIX}{tag}"
        )

    def put(self, tag, value, *unused_args, **unused_kwargs):
        return self._server_ctx._physical_ctx.guest.put(
            f"{_CDMUX_GUEST_DOWN_PREFIX}{tag}", value
        )


class _CrossDeviceServerHostsEndpoint:
    def __init__(self, server_ctx):
        self._server_ctx = server_ctx

    @staticmethod
    def _as_packet_list(value):
        return value if isinstance(value, list) else [value]

    def get(self, tag, *unused_args, **unused_kwargs):
        physical_ctx = self._server_ctx._physical_ctx
        assignments = self._server_ctx.assignments
        logical_values = {}

        # Worker 0 is the FATE guest.  Its logical client 0 travels on the
        # guest channel; any additional clients assigned to that same process
        # travel on this second, host-logical-client channel.
        worker_zero_hosts = [value for value in assignments[0] if value != 0]
        if worker_zero_hosts:
            packet = physical_ctx.guest.get(f"{_CDMUX_HOST_UPLOAD_PREFIX}{tag}")
            logical_values.update(_cdmux_unpack_packet(packet, expected_worker_id=0))

        if self._server_ctx.num_workers > 1:
            packets = physical_ctx.hosts.get(f"{_CDMUX_HOST_UPLOAD_PREFIX}{tag}")
            packets = self._as_packet_list(packets)
            if len(packets) != self._server_ctx.num_workers - 1:
                raise RuntimeError(
                    f"Expected {self._server_ctx.num_workers - 1} physical host packets "
                    f"for tag {tag!r}, received {len(packets)}."
                )
            for worker_id, packet in enumerate(packets, start=1):
                logical_values.update(
                    _cdmux_unpack_packet(packet, expected_worker_id=worker_id)
                )

        expected_ids = set(range(1, self._server_ctx.logical_clients))
        actual_ids = set(logical_values)
        if actual_ids != expected_ids:
            missing = sorted(expected_ids.difference(actual_ids))
            extra = sorted(actual_ids.difference(expected_ids))
            raise RuntimeError(
                f"Logical-client upload mismatch for tag {tag!r}: missing={missing}, extra={extra}."
            )
        return [logical_values[client_id] for client_id in range(1, self._server_ctx.logical_clients)]

    def put(self, tag, value, *unused_args, **unused_kwargs):
        physical_ctx = self._server_ctx._physical_ctx
        results = []
        if any(client_id != 0 for client_id in self._server_ctx.assignments[0]):
            results.append(
                physical_ctx.guest.put(f"{_CDMUX_HOST_DOWN_PREFIX}{tag}", value)
            )
        if self._server_ctx.num_workers > 1:
            results.append(
                physical_ctx.hosts.put(f"{_CDMUX_HOST_DOWN_PREFIX}{tag}", value)
            )
        return results[-1] if results else None


class _CrossDeviceServerContext:
    def __init__(self, physical_ctx, logical_clients, num_workers, assignments):
        self._physical_ctx = physical_ctx
        self.logical_clients = int(logical_clients)
        self.num_workers = int(num_workers)
        self.assignments = assignments
        self.rank = getattr(physical_ctx, "rank", self.num_workers)
        self.is_on_arbiter = True
        self.is_on_guest = False
        self.is_on_host = False
        self.guest = _CrossDeviceServerGuestEndpoint(self)
        self.hosts = _CrossDeviceServerHostsEndpoint(self)

    def __getattr__(self, name):
        return getattr(self._physical_ctx, name)


def _normalize_model_alias() -> None:
    requested = getattr(args, "model", "")
    resolved = _MODEL_ALIASES.get(requested, requested)
    if resolved != requested:
        args.model = resolved
        print(f"[ModelAlias] baseline={requested} -> model={resolved}", flush=True)

from model.TwoMGTCN import TwoMGTCN
from model.istgnn import ISTGNN
from model.FedOSTC import FedOSTC
from model.STGCN_EC import SpatioTemporalModel # 原 STGCN-EC.py 改名后
from model.AGAT import ASTGAT # 原 AGAT.py
from model.FedTSE import TrafficLSTM
from model.FedTPS import DCRNN_TP
from model.FedGODE import ODEGCN 
from model.FedGTP import FedGTP_Model
import model.FedGODE
from model.FedSTN import FedSTN  
from model.CNFGNN import CNFGNN
from lib.refol_strategy import REFOL
from model.FCGCN import FCGCN 
from model.FGNNEH import FGNNEH_Client, FGNNEH_Server
from model.FedMetro import FedMetro_Client_Model
from model.FCFedGCN import FC_FedGCN_Traffic
from model.FedSTG import FedSTG_Client, FedSTG_Server
from model.FUELS import FUELS_Model
from model.FedGRU import FedGRU_Model
from model.SFL_RNN import SFLPureRNN
from model.UFCL_GWN import LightGraphWaveNet
from model.DyHSL import DyHSL
from model.FedmSSA import FedmSSA_Model
from model.TDLR_SEDLR import StreamingTrafficLSTM
from model.ST_Net import STNET_pFedCTP


from lib.fedstg_client import FedSTGClientManager
import torch.nn.functional as F
# 在 fate_main.py 开头添加
import concepts 
import pandas as pd
from lib.utils import get_normalized_adj 
from lib.dtw import compute_dtw_matrix
from lib.fca import get_equiconcept_matrix
from lib.fgnneh_algo import FGNNEH_Backbone
from lib.fed4tp_utils import slice_data_for_twt
from lib.utils import init_seed, evaluate_client_model,extract_ctx_data,GlobalEarlyStopping, federated_early_stopping,FATEGlobalEarlyStoppingCallback,ExplicitEarlyStopper, EarlyStopSignal, unpack_spatiotemporal_batch, align_prediction_and_target, synchronize_cuda_for_timing
import torch_geometric.utils as pyg_utils
import matplotlib.pyplot as plt
from lib.load_dataset import load_dataset, spectral_community_detection,load_grid_dataset_for_fedstn ,read_st_dataset_file
from lib.grid_partition import grid_rectangular_split, grid_hw_for_dataset
from lib.partitioning import load_partition_artifact
from lib.twomgtcn_trainer import train_twomgtcn_task
# FedmSSA is optional for all other baselines.  Keep its missing/corrupted
# implementation from preventing FCGCN/UFCL (and every unrelated model) from
# even importing this entry point.
try:
    from lib.fedmssa_trainer import train_fedmssa_task
except (ImportError, AttributeError):
    train_fedmssa_task = None
from lib.tdlr_sedlr_trainer import train_tdlr_sedlr_task
import time



import csv
import os
import time


_BENCHMARK_RUN_ID = os.environ.get("BENCHMARK_RUN_ID") or os.environ.get("FCGCN_RUN_ID") or time.strftime("%Y%m%d_%H%M%S")


def _benchmark_run_id():
    return _BENCHMARK_RUN_ID


def _float_or_zero(value):
    if value in ("", None):
        return 0.0
    return float(value)


def _int_or_zero(value):
    if value in ("", None):
        return 0
    return int(float(value))


def _sample_metrics_from_sums(mae, mse, rmse, mape, elements, abs_sum, sq_sum, mape_sum, mape_elements):
    total_elements = _int_or_zero(elements)
    if total_elements <= 0:
        return {
            "Sample_MAE": mae,
            "Sample_MSE": mse,
            "Sample_RMSE": rmse,
            "Sample_MAPE": mape,
            "Sample_Elements": "",
        }

    total_abs = _float_or_zero(abs_sum)
    total_sq = _float_or_zero(sq_sum)
    total_mape = _float_or_zero(mape_sum)
    total_mape_elements = _int_or_zero(mape_elements)
    sample_mse = total_sq / max(total_elements, 1)
    return {
        "Sample_MAE": round(total_abs / max(total_elements, 1), 4),
        "Sample_MSE": round(sample_mse, 4),
        "Sample_RMSE": round(math.sqrt(sample_mse), 4),
        "Sample_MAPE": round(total_mape / total_mape_elements, 4) if total_mape_elements else 0.0,
        "Sample_Elements": total_elements,
    }


def _estimate_fedgru_flops(model, sample_x):
    """
    THOP often does not count nn.GRU reliably, so FedGRU falls back to a
    simple analytical estimate based on the actual batch shape.
    """
    x = sample_x.detach()
    if x.dim() == 3:
        x = x.unsqueeze(-1)
    if x.dim() != 4:
        raise ValueError(f"FedGRU FLOPs fallback expects 4D input, got {tuple(x.shape)}")

    if x.shape[1] < x.shape[2]:
        x = x.transpose(1, 2).contiguous()

    batch_size, num_nodes, time_steps, feature_dim = x.shape
    hidden_dim = int(getattr(model, "hidden_dim"))
    num_layers = int(getattr(model.gru, "num_layers", 1))
    input_dim = int(getattr(model, "input_dim", feature_dim))
    out_features = int(getattr(model, "pre_len")) * int(getattr(model, "out_dim"))

    batch_nodes = batch_size * num_nodes
    total_flops = 0.0
    current_input_dim = input_dim

    for _ in range(num_layers):
        # 3 GRU gates, each with input and hidden affine transforms.
        gate_flops = 2.0 * 3.0 * (current_input_dim * hidden_dim + hidden_dim * hidden_dim)
        # Bias/activation terms are small but included for stability.
        gate_flops += 6.0 * hidden_dim
        total_flops += batch_nodes * time_steps * gate_flops
        current_input_dim = hidden_dim

    total_flops += batch_nodes * (2.0 * hidden_dim * out_features + out_features)
    return round(total_flops / 1e9, 4)


def _federated_log_root(project_root, t_out):
    return os.path.join(project_root, "logs12" if int(t_out) == 12 else "logs")


def _federated_result_dir(project_root, model_name, t_out):
    custom_root = str(getattr(args, "result_root", "") or "").strip()
    if custom_root:
        return custom_root
    log_root = _federated_log_root(project_root, t_out)
    safe_model_name = "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_"
        for ch in str(model_name)
    ).strip("_") or "unknown_model"
    return os.path.join(log_root, f"{safe_model_name}_Logs")


class _InterProcessFileLock:
    def __init__(self, target_path, timeout=60.0, poll_interval=0.1):
        self.lock_path = f"{target_path}.lock"
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.fd = None

    def __enter__(self):
        start = time.time()
        while True:
            try:
                self.fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(self.fd, str(os.getpid()).encode("utf-8", errors="ignore"))
                return self
            except FileExistsError:
                if time.time() - start >= self.timeout:
                    raise TimeoutError(f"Timed out waiting for CSV lock: {self.lock_path}")
                time.sleep(self.poll_interval)

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.fd is not None:
                os.close(self.fd)
        finally:
            self.fd = None
            try:
                os.remove(self.lock_path)
            except FileNotFoundError:
                pass



def _log_experiment_results_unlocked(
    model_name, dataset_client, feature_type,
    best_epoch, # <--- 新增：最优轮次
    # 1. Accuracy (准确度)
    acc_mae, acc_mse, acc_rmse, acc_mape,
    # 2. Efficiency (效率)
    eff_train_time, eff_val_time, eff_test_time, eff_comm_size_mb, eff_train_round, eff_flops,
    acc_elements="", acc_abs_error_sum="", acc_sq_error_sum="", acc_mape_error_sum="", acc_mape_elements="",
    **kwargs # 吸收多余参数防报错
):
    """自动记录 Accuracy 和 Efficiency 维度的实验结果到 CSV"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_log_dir = getattr(args, 'privacy_result_dir', '') or _federated_result_dir(project_root, model_name, args.t_out)
    os.makedirs(model_log_dir, exist_ok=True)
    file_path = os.path.join(model_log_dir, "benchmark_results_acc_eff.csv")
    fieldnames = [
        'Model', 'Dataset_Client', 'Feature', 'Best_Epoch',
        'Client_MAE', 'Client_MSE', 'Client_RMSE', 'Client_MAPE',
        'Acc_MAE', 'Acc_MSE', 'Acc_RMSE', 'Acc_MAPE',
        'Acc_Elements', 'Acc_AbsErrorSum', 'Acc_SqErrorSum', 'Acc_MAPEErrorSum', 'Acc_MAPEElements',
        'Sample_MAE', 'Sample_MSE', 'Sample_RMSE', 'Sample_MAPE', 'Sample_Elements',
        'Eff_TrainTime(s)', 'Eff_ValTime(s)', 'Eff_TestTime(s)', 'Eff_CommSize(MB)', 'Eff_TrainRound', 'Eff_FLOPs(G)',
        'Protection', 'DP_Sigma', 'DP_ClipNorm', 'HE_Backend', 'Run_ID'
    ]
    sample_metrics = _sample_metrics_from_sums(
        acc_mae, acc_mse, acc_rmse, acc_mape,
        acc_elements, acc_abs_error_sum, acc_sq_error_sum,
        acc_mape_error_sum, acc_mape_elements,
    )
    row = {
        'Model': model_name,
        'Dataset_Client': dataset_client,
        'Feature': feature_type,
        'Best_Epoch': best_epoch,
        'Client_MAE': acc_mae,
        'Client_MSE': acc_mse,
        'Client_RMSE': acc_rmse,
        'Client_MAPE': acc_mape,
        'Acc_MAE': acc_mae,
        'Acc_MSE': acc_mse,
        'Acc_RMSE': acc_rmse,
        'Acc_MAPE': acc_mape,
        'Acc_Elements': acc_elements,
        'Acc_AbsErrorSum': acc_abs_error_sum,
        'Acc_SqErrorSum': acc_sq_error_sum,
        'Acc_MAPEErrorSum': acc_mape_error_sum,
        'Acc_MAPEElements': acc_mape_elements,
        'Sample_MAE': sample_metrics["Sample_MAE"],
        'Sample_MSE': sample_metrics["Sample_MSE"],
        'Sample_RMSE': sample_metrics["Sample_RMSE"],
        'Sample_MAPE': sample_metrics["Sample_MAPE"],
        'Sample_Elements': sample_metrics["Sample_Elements"],
        'Eff_TrainTime(s)': eff_train_time,
        'Eff_ValTime(s)': eff_val_time,
        'Eff_TestTime(s)': eff_test_time,
        'Eff_CommSize(MB)': eff_comm_size_mb,
        'Eff_TrainRound': eff_train_round,
        'Eff_FLOPs(G)': eff_flops,
        'Protection': kwargs.get('protection', getattr(args, 'protection', 'plain')),
        'HE_Backend': kwargs.get('he_backend', getattr(args, 'he_backend', '')),
        'DP_Sigma': kwargs.get('dp_sigma', getattr(args, 'dp_sigma', '')),
        'DP_ClipNorm': kwargs.get('dp_clip_norm', getattr(args, 'dp_clip_norm', '')),
        'Run_ID': _benchmark_run_id(),
    }
    
    file_exists = os.path.isfile(file_path)
    existing_rows = []
    if file_exists:
        with open(file_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and reader.fieldnames != fieldnames:
                existing_rows = list(reader)
                file_exists = False
    
    try:
        mode = 'a' if file_exists else 'w'
        with open(file_path, mode=mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                # 加上 Best_Epoch 和 Eff_ValTime(s) 表头
                writer.writerow([
                    'Model', 'Dataset_Client', 'Feature', 'Best_Epoch',
                    'Client_MAE', 'Client_MSE', 'Client_RMSE', 'Client_MAPE',
                    'Acc_MAE', 'Acc_MSE', 'Acc_RMSE', 'Acc_MAPE',
                    'Acc_Elements', 'Acc_AbsErrorSum', 'Acc_SqErrorSum', 'Acc_MAPEErrorSum', 'Acc_MAPEElements',
                    'Sample_MAE', 'Sample_MSE', 'Sample_RMSE', 'Sample_MAPE', 'Sample_Elements',
                    'Eff_TrainTime(s)', 'Eff_ValTime(s)', 'Eff_TestTime(s)', 'Eff_CommSize(MB)', 'Eff_TrainRound', 'Eff_FLOPs(G)',
                    'Protection', 'DP_Sigma', 'DP_ClipNorm', 'HE_Backend', 'Run_ID'
                ])
                for old_row in existing_rows:
                    writer.writerow([old_row.get(name, "") for name in fieldnames])
            
            # 写入数据
            writer.writerow([
                model_name, dataset_client, feature_type, best_epoch,
                acc_mae, acc_mse, acc_rmse, acc_mape,
                acc_mae, acc_mse, acc_rmse, acc_mape,
                acc_elements, acc_abs_error_sum, acc_sq_error_sum, acc_mape_error_sum, acc_mape_elements,
                sample_metrics["Sample_MAE"], sample_metrics["Sample_MSE"], sample_metrics["Sample_RMSE"],
                sample_metrics["Sample_MAPE"], sample_metrics["Sample_Elements"],
                eff_train_time, eff_val_time, eff_test_time, eff_comm_size_mb, eff_train_round, eff_flops,
                kwargs.get('protection', getattr(args, 'protection', 'plain')),
                kwargs.get('dp_sigma', getattr(args, 'dp_sigma', '')),
                kwargs.get('dp_clip_norm', getattr(args, 'dp_clip_norm', '')),
                kwargs.get('he_backend', getattr(args, 'he_backend', '')),
                _benchmark_run_id()
            ])
        print(f"CSV result saved to: {file_path}")
        write_benchmark_sample_summary(file_path)
    except Exception as e:
        print(f"CSV 写入失败，报错: {e}")


_LOG_RESULTS_THREAD_LOCK = threading.Lock()


def log_experiment_results(*call_args, **call_kwargs):
    """Serialize result writes from multiplexed logical-client threads/processes."""
    if not bool(getattr(args, "cross_device_multiplex", False)):
        return _log_experiment_results_unlocked(*call_args, **call_kwargs)
    if call_args:
        model_name = call_args[0]
    else:
        model_name = call_kwargs.get("model_name", getattr(args, "model", "unknown"))
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_log_dir = (
        getattr(args, "privacy_result_dir", "")
        or _federated_result_dir(project_root, model_name, args.t_out)
    )
    os.makedirs(model_log_dir, exist_ok=True)
    result_path = os.path.join(model_log_dir, "benchmark_results_acc_eff.csv")
    with _LOG_RESULTS_THREAD_LOCK:
        with _InterProcessFileLock(result_path):
            return _log_experiment_results_unlocked(*call_args, **call_kwargs)


def write_benchmark_sample_summary(raw_path):
    summary_path = raw_path.replace(".csv", "_summary.csv")
    summary_fields = [
        "Model", "Dataset", "Feature", "Clients",
        "Client_MAE", "Client_MSE", "Client_RMSE", "Client_MAPE",
        "Client_Mean_MAE", "Client_Mean_MSE", "Client_Mean_RMSE", "Client_Mean_MAPE",
        "Sample_MAE", "Sample_MSE", "Sample_RMSE", "Sample_MAPE", "Sample_Elements",
        "Run_ID",
    ]
    if not os.path.isfile(raw_path):
        return

    with open(raw_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    grouped = {}
    for row in rows:
        dataset_client = row.get("Dataset_Client", "")
        if "_client" not in dataset_client:
            continue
        dataset = dataset_client.rsplit("_client", 1)[0]
        run_id = row.get("Run_ID") or "legacy"
        key = (run_id, row.get("Model", ""), dataset, row.get("Feature", ""))
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (run_id, model_name, dataset, feature), group_rows in grouped.items():
        client_mae = [float(r.get("Acc_MAE") or 0.0) for r in group_rows]
        client_mse = [float(r.get("Acc_MSE") or 0.0) for r in group_rows]
        client_rmse = [float(r.get("Acc_RMSE") or 0.0) for r in group_rows]
        client_mape = [float(r.get("Acc_MAPE") or 0.0) for r in group_rows]
        total_elements = sum(_int_or_zero(r.get("Acc_Elements")) for r in group_rows)
        total_abs = sum(_float_or_zero(r.get("Acc_AbsErrorSum")) for r in group_rows)
        total_sq = sum(_float_or_zero(r.get("Acc_SqErrorSum")) for r in group_rows)
        total_mape = sum(_float_or_zero(r.get("Acc_MAPEErrorSum")) for r in group_rows)
        total_mape_elements = sum(_int_or_zero(r.get("Acc_MAPEElements")) for r in group_rows)

        if total_elements > 0:
            sample_mae = total_abs / total_elements
            sample_mse = total_sq / total_elements
            sample_rmse = math.sqrt(sample_mse)
            sample_mape = total_mape / total_mape_elements if total_mape_elements else 0.0
        else:
            sample_mae = sum(client_mae) / max(len(client_mae), 1)
            sample_mse = sum(client_mse) / max(len(client_mse), 1)
            sample_rmse = sum(client_rmse) / max(len(client_rmse), 1)
            sample_mape = sum(client_mape) / max(len(client_mape), 1)

        summary_rows.append({
            "Model": model_name,
            "Dataset": dataset,
            "Feature": feature,
            "Clients": len(group_rows),
            "Client_MAE": round(sum(client_mae) / max(len(client_mae), 1), 4),
            "Client_MSE": round(sum(client_mse) / max(len(client_mse), 1), 4),
            "Client_RMSE": round(sum(client_rmse) / max(len(client_rmse), 1), 4),
            "Client_MAPE": round(sum(client_mape) / max(len(client_mape), 1), 4),
            "Client_Mean_MAE": round(sum(client_mae) / max(len(client_mae), 1), 4),
            "Client_Mean_MSE": round(sum(client_mse) / max(len(client_mse), 1), 4),
            "Client_Mean_RMSE": round(sum(client_rmse) / max(len(client_rmse), 1), 4),
            "Client_Mean_MAPE": round(sum(client_mape) / max(len(client_mape), 1), 4),
            "Sample_MAE": round(sample_mae, 4),
            "Sample_MSE": round(sample_mse, 4),
            "Sample_RMSE": round(sample_rmse, 4),
            "Sample_MAPE": round(sample_mape, 4),
            "Sample_Elements": total_elements,
            "Run_ID": run_id,
        })

    with open(summary_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def _log_experiment_results_with_lock(
    model_name, dataset_client, feature_type,
    best_epoch, acc_mae, acc_mse, acc_rmse, acc_mape,
    eff_train_time, eff_val_time, eff_test_time, eff_comm_size_mb, eff_train_round, eff_flops,
    acc_elements="", acc_abs_error_sum="", acc_sq_error_sum="", acc_mape_error_sum="", acc_mape_elements="",
    **kwargs
):
    # Keep the all-message quantity for audit, but publish a training
    # communication metric with a consistent scope: setup once plus messages
    # sent during one local training round. Validation/test exchanges must not
    # be multiplied by the historical number of training rounds.
    he_init_comm_mb = ""
    he_train_comm_mb = ""
    he_val_comm_mb = ""
    he_test_comm_mb = ""
    he_total_comm_mb = ""
    if (
        str(getattr(args, 'protection', 'plain')).lower() == 'he'
        and str(getattr(args, 'he_backend', '')).lower() == 'he_ttp'
        and hasattr(args, '_he_ttp_upload_bytes')
    ):
        he_upload = int(getattr(args, '_he_ttp_upload_bytes', 0))
        he_download = int(getattr(args, '_he_ttp_download_bytes', 0))
        he_upload_by_phase = dict(getattr(args, '_he_ttp_upload_bytes_by_phase', {}) or {})
        he_downlink_by_phase = dict(getattr(args, '_he_ttp_downlink_bytes_by_phase', {}) or {})
        he_encrypt_s = float(getattr(args, '_he_ttp_encrypt_seconds', 0.0))
        he_encrypt_by_phase = dict(getattr(args, '_he_ttp_encrypt_seconds_by_phase', {}) or {})
        phase_mb = lambda phase: round(
            (int(he_upload_by_phase.get(phase, 0)) + int(he_downlink_by_phase.get(phase, 0)))
            / (1024 * 1024),
            4,
        )
        he_init_comm_mb = phase_mb("initialization")
        he_train_comm_mb = phase_mb("train")
        he_val_comm_mb = phase_mb("validation")
        he_test_comm_mb = phase_mb("test")
        he_total_comm_mb = round((he_upload + he_download) / (1024 * 1024), 4)
        eff_comm_size_mb = round(float(he_init_comm_mb) + float(he_train_comm_mb), 4)
        # Do not add encryption seconds here. Some trainers already time the
        # upload/get interval, whereas others measure local work only. A
        # global addition silently double-counted several HE-TTP methods.
        # The raw encryption totals remain in the HETTP audit log below.
        print(
            f"[HETTPMetrics] upload_bytes={he_upload} downlink_bytes={he_download} "
            f"encrypt_s={he_encrypt_s:.6f} train_scope_comm_mb={eff_comm_size_mb:.4f} "
            f"phase_comm_mb={{'initialization': {he_init_comm_mb:.4f}, 'train': {he_train_comm_mb:.4f}, "
            f"'validation': {he_val_comm_mb:.4f}, 'test': {he_test_comm_mb:.4f}, 'all': {he_total_comm_mb:.4f}}} "
            f"upload_by_phase={he_upload_by_phase} downlink_by_phase={he_downlink_by_phase} "
            f"encrypt_s_by_phase={he_encrypt_by_phase}",
            flush=True,
        )
    if bool(getattr(args, 'privacy_trace_only', False)):
        print(
            f"[PrivacyTrace] CSV output suppressed for trace-only run: "
            f"{model_name} {dataset_client}",
            flush=True,
        )
        return
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_log_dir = getattr(args, 'privacy_result_dir', '') or _federated_result_dir(project_root, model_name, args.t_out)
    os.makedirs(model_log_dir, exist_ok=True)
    file_path = os.path.join(model_log_dir, "benchmark_results_acc_eff.csv")
    fieldnames = [
        'Model', 'Dataset_Client', 'Feature', 'Best_Epoch',
        'Client_MAE', 'Client_MSE', 'Client_RMSE', 'Client_MAPE',
        'Acc_MAE', 'Acc_MSE', 'Acc_RMSE', 'Acc_MAPE',
        'Acc_Elements', 'Acc_AbsErrorSum', 'Acc_SqErrorSum', 'Acc_MAPEErrorSum', 'Acc_MAPEElements',
        'Sample_MAE', 'Sample_MSE', 'Sample_RMSE', 'Sample_MAPE', 'Sample_Elements',
        'Eff_TrainTime(s)', 'Eff_ValTime(s)', 'Eff_TestTime(s)', 'Eff_CommSize(MB)', 'Eff_TrainRound', 'Eff_FLOPs(G)',
        'Protection', 'DP_Sigma', 'DP_ClipNorm', 'HE_Backend',
        'HE_InitComm(MB)', 'HE_TrainComm(MB)', 'HE_ValComm(MB)',
        'HE_TestComm(MB)', 'HE_AllPhaseComm(MB)', 'Run_ID'
    ]
    sample_metrics = _sample_metrics_from_sums(
        acc_mae, acc_mse, acc_rmse, acc_mape,
        acc_elements, acc_abs_error_sum, acc_sq_error_sum,
        acc_mape_error_sum, acc_mape_elements,
    )

    try:
        with _InterProcessFileLock(file_path):
            file_exists = os.path.isfile(file_path)
            existing_rows = []
            if file_exists:
                with open(file_path, newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames and reader.fieldnames != fieldnames:
                        existing_rows = list(reader)
                        file_exists = False

            mode = 'a' if file_exists else 'w'
            with open(file_path, mode=mode, newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(fieldnames)
                    for old_row in existing_rows:
                        writer.writerow([old_row.get(name, "") for name in fieldnames])

                writer.writerow([
                    model_name, dataset_client, feature_type, best_epoch,
                    acc_mae, acc_mse, acc_rmse, acc_mape,
                    acc_mae, acc_mse, acc_rmse, acc_mape,
                    acc_elements, acc_abs_error_sum, acc_sq_error_sum, acc_mape_error_sum, acc_mape_elements,
                    sample_metrics["Sample_MAE"], sample_metrics["Sample_MSE"], sample_metrics["Sample_RMSE"],
                    sample_metrics["Sample_MAPE"], sample_metrics["Sample_Elements"],
                    eff_train_time, eff_val_time, eff_test_time, eff_comm_size_mb, eff_train_round, eff_flops,
                    kwargs.get('protection', getattr(args, 'protection', 'plain')),
                    kwargs.get('dp_sigma', getattr(args, 'dp_sigma', '')),
                    kwargs.get('dp_clip_norm', getattr(args, 'dp_clip_norm', '')),
                    kwargs.get('he_backend', getattr(args, 'he_backend', '')),
                    he_init_comm_mb, he_train_comm_mb, he_val_comm_mb,
                    he_test_comm_mb, he_total_comm_mb,
                    _benchmark_run_id()
                ])

            write_benchmark_sample_summary(file_path)
        print(f"CSV result saved to: {file_path}")
    except Exception as e:
        print(f"CSV result write failed: {e}")


log_experiment_results = _log_experiment_results_with_lock


def _restore_best_checkpoint_for_prediction(trainer, model, device, rank=None):
    """Load the validation-best checkpoint before final test prediction."""
    state = getattr(trainer, "state", None)
    train_args = getattr(trainer, "args", None) or getattr(trainer, "training_args", None)
    best_checkpoint = getattr(state, "best_model_checkpoint", None) if state is not None else None
    best_metric = getattr(state, "best_metric", None) if state is not None else None

    if not best_checkpoint and state is not None and train_args is not None:
        eval_rows = [
            row for row in getattr(state, "log_history", [])
            if "eval_mae" in row and row.get("step") is not None
        ]
        if eval_rows:
            best_row = min(eval_rows, key=lambda row: float(row["eval_mae"]))
            best_checkpoint = os.path.join(train_args.output_dir, f"checkpoint-{int(best_row['step'])}")
            best_metric = best_row.get("eval_mae")
            state.best_model_checkpoint = best_checkpoint
            state.best_metric = best_metric

    rank_prefix = f"Rank {rank}: " if rank is not None else ""
    if not best_checkpoint:
        print(f"{rank_prefix}[BestCheckpoint] no validation-best checkpoint found; using current model.", flush=True)
        return False
    if not os.path.isdir(best_checkpoint):
        print(f"{rank_prefix}[BestCheckpoint] checkpoint path not found: {best_checkpoint}; using current model.", flush=True)
        return False

    try:
        load_best = getattr(trainer, "_load_best_model", None)
        if callable(load_best):
            load_best()
            model.to(device)
            print(
                f"{rank_prefix}[BestCheckpoint] restored via Trainer: {best_checkpoint} "
                f"(best_eval_mae_norm={best_metric})",
                flush=True,
            )
            return True
    except Exception as exc:
        print(f"{rank_prefix}[BestCheckpoint] Trainer restore failed: {exc}; trying state_dict load.", flush=True)

    state_path = os.path.join(best_checkpoint, "pytorch_model.bin")
    safe_path = os.path.join(best_checkpoint, "model.safetensors")
    try:
        if os.path.isfile(state_path):
            state_dict = torch.load(state_path, map_location=device)
        elif os.path.isfile(safe_path):
            from safetensors.torch import load_file
            state_dict = load_file(safe_path, device=str(device))
        else:
            print(f"{rank_prefix}[BestCheckpoint] model weights not found under {best_checkpoint}; using current model.", flush=True)
            return False

        if isinstance(state_dict, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in state_dict and isinstance(state_dict[key], dict):
                    state_dict = state_dict[key]
                    break
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        if hasattr(trainer, "model"):
            trainer.model = model
        print(
            f"{rank_prefix}[BestCheckpoint] restored state_dict: {best_checkpoint} "
            f"(best_eval_mae_norm={best_metric})",
            flush=True,
        )
        return True
    except Exception as exc:
        print(f"{rank_prefix}[BestCheckpoint] state_dict restore failed: {exc}; using current model.", flush=True)
        return False


_PARTITION_CACHE = None


def _initialize_partition(ctx: Context):
    """Prepare the scoped partition for every FATE role, including arbiter."""
    global _PARTITION_CACHE
    _configure_cross_device_node_clients()
    if bool(getattr(args, "cross_device_multiplex", False)):
        expected_clients = _cross_device_dataset_node_count(args.dataset_name)
        if int(args.num_clients) != expected_clients:
            raise RuntimeError(
                f"Cross-device logical client count changed unexpectedly: "
                f"expected {expected_clients}, got {args.num_clients}."
            )
        args.nodes_per = [[node_id] for node_id in range(expected_clients)]
        return True
    if args.dataset_name not in {"PeMS04", "TaxiBJ"}:
        if getattr(args, "partition_strategy", "auto") != "auto":
            raise ValueError("partition_strategy currently supports only PeMS04 and TaxiBJ.")
        return False
    if args.feature_type != "flow":
        raise ValueError("The partition experiment suite supports flow only.")
    allowed_client_counts = {"PeMS04": {2, 4, 8, 16, 32}, "TaxiBJ": {4, 32}}
    if int(args.num_clients) not in allowed_client_counts[args.dataset_name] or int(args.t_out) != 3:
        raise ValueError(
            "The partition experiment suite requires t_out=3 and client counts "
            f"{sorted(allowed_client_counts[args.dataset_name])} for {args.dataset_name}."
        )
    if int(args.seed) != 42:
        raise ValueError("The partition experiment suite requires seed=42.")
    # ``auto`` is the normal, fixed client partition used by every benchmark
    # condition.  Only an explicitly selected alternative strategy is the
    # separate partition-study experiment, which remains Plain-only.  The old
    # condition treated the default ``auto`` as that study and consequently
    # blocked every PeMS04/TaxiBJ DP or HE run before training started.
    is_explicit_partition_study = getattr(args, "partition_strategy", "auto") != "auto"
    if is_explicit_partition_study and (
        str(getattr(args, "protection", "plain")).lower() != "plain"
        or float(getattr(args, "dp_noise", 0.0)) != 0.0
    ):
        raise ValueError("Explicit partition-strategy experiments support plain training only (no DP or HE).")
    expected_ratios = (0.7, 0.1, 0.2)
    actual_ratios = (float(args.train_ratio), float(args.val_ratio), float(args.test_ratio))
    if any(abs(actual - expected) > 1e-9 for actual, expected in zip(actual_ratios, expected_ratios)):
        raise ValueError("The partition experiment suite requires train/val/test = 0.7/0.1/0.2.")
    if _PARTITION_CACHE is None:
        _PARTITION_CACHE = load_partition_artifact(
            dataset_name=args.dataset_name,
            strategy=getattr(args, "partition_strategy", "auto"),
            num_clients=args.num_clients,
            seed=args.seed,
            project_root=file_dir,
        )
    nodes_per, metadata = _PARTITION_CACHE
    args.nodes_per = [list(group) for group in nodes_per]
    # The legacy get_setting implementation still resolves this symbol with
    # eval().  Supplying the selected partition preserves its downstream setup.
    globals()[f"{args.dataset_name}FLOW_{args.num_clients}p_metis"] = args.nodes_per
    if ctx.rank == 0:
        selected = metadata['strategy']
        json_path = Path(file_dir) / "partition" / selected / (
            f"{args.dataset_name}_flow_{args.num_clients}clients_seed{args.seed}.json"
        )
        print(
            f"[Partition] loaded strategy={selected} sizes={metadata['client_sizes']} "
            f"artifact={json_path}", flush=True,
        )
    return True


def get_setting(ctx: Context):
    _initialize_partition(ctx)
    return _get_setting_legacy(ctx)


def _get_setting_legacy(ctx: Context):
    

    # ================= [开始修改：全自动切分逻辑] =================

    total_nodes_map = {
        'PeMS03': 358,    # 补充 PeMS03
        'PeMS04': 307,
        'PeMSD7': 228,    # 补充 PeMS07
        'PeMS08': 170,
        'TaxiBJ': 1024,
        'TaxiNYC': 75,
        'BikeNYC': 128    # 补充 BikeNYC (请根据实际 Grid 节点数修正)
    }  
    # 2. 自动匹配当前数据集的节点数
    current_total_nodes = 0
    for key, val in total_nodes_map.items():
        if key in args.dataset_name:
            current_total_nodes = val
            break
            
    try:
        # 尝试读取预定义的 Metis 切分 (老代码逻辑)
        split_var_name = f"{args.dataset_name}FLOW_{args.num_clients}p_metis"
        args.nodes_per = eval(split_var_name)
        
    except (NameError, SyntaxError):
        if any(grid_name in args.dataset_name for grid_name in ['TaxiBJ', 'TaxiNYC', 'BikeNYC']):
            H, W = grid_hw_for_dataset(args.dataset_name)
            if ctx.rank == 0:
                print(
                    f"Rank {ctx.rank}: using rectangular grid split for {args.dataset_name} "
                    f"({H}x{W}, num_clients={args.num_clients})..."
                )
            args.nodes_per = grid_rectangular_split(args.dataset_name, args.num_clients)

        elif current_total_nodes > 0:
            if ctx.rank == 0: 
                print(f"Rank {ctx.rank}: 正在执行【自动均匀切分】...")
            quotient, remainder = divmod(current_total_nodes, args.num_clients)
            nodes_per = []
            start_idx = 0
            for i in range(args.num_clients):
                count = quotient + (1 if i < remainder else 0)
                nodes_per.append(list(range(start_idx, start_idx + count)))
                start_idx += count
            args.nodes_per = nodes_per
            
        else:
            # 如果连总节点数都不知道，那就真的没办法了
            raise ValueError(f"严重错误：不知道数据集 {args.dataset_name} 有多少个节点！请在 fate_main.py 的 total_nodes_map 里补充配置。")
            
    # ================= [修改结束] =================
    max_num_nodes_global = max([len(nodes) for nodes in args.nodes_per])

    
    selected_nodes = args.nodes_per[ctx.rank]
    dataset_name = args.dataset_name

    
    if any(grid_name in args.dataset_name for grid_name in ['TaxiBJ', 'TaxiNYC', 'BikeNYC']):
        if args.model == 'TwoMGTCN':
            print(f"[2MGTCN get_setting][rank {ctx.rank}] stage=before_grid_dataset", flush=True)
        from lib.load_dataset import load_grid_dataset_for_fedstn
        train_set, val_set, test_set, edge_index, scaler = load_grid_dataset_for_fedstn(
            dataset_name=args.dataset_name,
            t_in=args.t_in,
            t_out=args.t_out,
            device=args.device,
            selected_nodes=selected_nodes,
            model_name=args.model
        )
        if args.model == 'TwoMGTCN':
            print(
                f"[2MGTCN get_setting][rank {ctx.rank}] stage=after_grid_dataset "
                f"N={len(selected_nodes)} ext_dim={len(train_set.ext_names)}",
                flush=True,
            )
        N = len(selected_nodes) 
        args.input_dim = 2 # 双通道
        args.output_dim = 2 
        args.ext_dim = len(train_set.ext_names) # 比如 21维
    else:
        # 走原来的路
        train_set, val_set, test_set, edge_index, scaler = load_dataset(dataset_name=dataset_name, 
            feature_type=args.feature_type, 
            normalizer=args.normalizer,
            T_in=args.t_in,
            T_out=args.t_out, 
            train_ratio=args.train_ratio, 
            val_ratio=args.val_ratio, 
            test_ratio=args.test_ratio,  # <--- 新增传入参数
            return_edge_index=True, 
            device=args.device, 
            selected_nodes=selected_nodes,
            norm_scope=getattr(args, "norm_scope", "global"),
            scaler_fit_scope=getattr(args, "scaler_fit_scope", "selected")
        )
        N = len(selected_nodes)

    if bool(getattr(args, "privacy_trace_only", False)):
        trace_index = int(getattr(args, "privacy_trace_sample_index", 0))
        train_set = _PrivacyTraceSingleSampleDataset(train_set, trace_index)
        val_set = _PrivacyTraceSingleSampleDataset(val_set, min(trace_index, len(val_set) - 1))
        test_set = _PrivacyTraceSingleSampleDataset(test_set, min(trace_index, len(test_set) - 1))
        print(
            f"[PrivacyTrace] rank={ctx.rank} using one deterministic sample "
            f"train={trace_index}, val/test={min(trace_index, len(val_set._dataset) - 1)}",
            flush=True,
        )

    # ================= 模型动态选择区域 =================
    model_name = args.model
    print(f"Loading Model: {model_name}")

    if model_name == 'ISTGNN':
        model = ISTGNN(edge_index, 1, args.hidden_dim, args.hidden_dim, num_nodes=N, pre_len=args.t_out)

    elif model_name == 'TwoMGTCN':
        #  1. 拒绝使用丢失权重的 edge_index，直接从 dataset 取出原汁原味的语义矩阵
        A_raw = torch.tensor(train_set.adj, dtype=torch.float32).to(args.device)
        
        #  2. 对图矩阵进行度归一化 (Row Normalization)，彻底镇压梯度爆炸
        # 让每一行的权重和等于 1.0，无论节点有多少，卷积后数值大小都不会膨胀
        row_sum = A_raw.sum(dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1.0  # 防止除以 0
        A_norm = A_raw / row_sum
        
        model = TwoMGTCN(
            A=A_norm, 
            T_in=args.t_in, 
            T_out=args.t_out, 
            hidden_size=args.hidden_dim, 
            num_layers=3, 
            nb_flow=args.input_dim,                # 双通道流量
            ext_dim=getattr(args, 'ext_dim', 21)   # 21维气象特征
        )

    elif model_name == 'ASTGAT':

        if edge_index is None:
            print(f"Rank {ctx.rank}: [ASTGAT Graph Check] edge_index is None, fallback to identity adjacency.")
            A_dense = torch.eye(N, dtype=torch.float32, device=args.device)
        else:
            if not isinstance(edge_index, torch.Tensor):
                edge_index = torch.LongTensor(edge_index)
            edge_index = edge_index.long().to(args.device)

            # 1. 基础索引检查
            num_edges = edge_index.shape[1] if edge_index.dim() == 2 else 0

            print("=" * 80)
            print(f"Rank {ctx.rank}: [ASTGAT Graph Check - BEFORE FIX]")
            print(f"Rank {ctx.rank}: Dataset={args.dataset_name}, Feature={args.feature_type}")
            print(f"Rank {ctx.rank}: Local nodes N={N}, selected_nodes_len={len(selected_nodes)}")
            print(f"Rank {ctx.rank}: edge_index shape={tuple(edge_index.shape)}, num_edges={num_edges}")

            if num_edges > 0:
                edge_min = int(edge_index.min().item())
                edge_max = int(edge_index.max().item())
                print(f"Rank {ctx.rank}: edge_index min={edge_min}, max={edge_max}")

                # 如果这里触发，说明 load_dataset 返回的 edge_index 不是局部 0..N-1 编号
                if edge_min < 0 or edge_max >= N:
                    print(
                        f"Rank {ctx.rank}: [严重警告] ASTGAT 收到的 edge_index 可能不是局部编号！"
                        f" edge_min={edge_min}, edge_max={edge_max}, N={N}。"
                        f" 这通常意味着 selected_nodes 映射或 distance.csv 节点编号有问题。"
                    )
            else:
                print(f"Rank {ctx.rank}: [警告] 当前客户端没有任何边，ASTGAT 将退化为只依赖自环。")

            # 2. 原始邻接矩阵
            A_raw = pyg_utils.to_dense_adj(edge_index, max_num_nodes=N)[0].to(args.device)
            A_raw = (A_raw > 0).float()

            deg_raw = A_raw.sum(dim=1)
            zero_deg_raw = int((deg_raw == 0).sum().item())
            density_raw = float(A_raw.sum().item() / max(N * N, 1))

            print(
                f"Rank {ctx.rank}: BEFORE_FIX density={density_raw:.6f}, "
                f"zero_degree_nodes={zero_deg_raw}, "
                f"min_deg={float(deg_raw.min().item()):.2f}, "
                f"max_deg={float(deg_raw.max().item()):.2f}"
            )

            # 3. 第四步：最小图修复
            #    3.1 有向边转无向边
            #    3.2 加自环，避免 GTHA 中某些节点没有可注意的邻居
            #    3.3 二值化，保证 mask 语义清晰
            A_dense = ((A_raw + A_raw.T) > 0).float()
            A_dense.fill_diagonal_(1.0)

            # 4. 修复后再次打印
            deg_fixed = A_dense.sum(dim=1)
            zero_deg_fixed = int((deg_fixed == 0).sum().item())
            density_fixed = float(A_dense.sum().item() / max(N * N, 1))

            print(f"Rank {ctx.rank}: [ASTGAT Graph Check - AFTER FIX]")
            print(
                f"Rank {ctx.rank}: AFTER_FIX density={density_fixed:.6f}, "
                f"zero_degree_nodes={zero_deg_fixed}, "
                f"min_deg={float(deg_fixed.min().item()):.2f}, "
                f"max_deg={float(deg_fixed.max().item()):.2f}"
            )
            print("=" * 80)

        max_num_nodes_global = max(current_total_nodes, max([max(nodes) for nodes in args.nodes_per if nodes]) + 1)
        print(f"Global node slots for ASTGAT ID-aligned params: {max_num_nodes_global}")

        model = ASTGAT(
            num_nodes=N,
            in_dim=args.t_in,
            pred_len=args.t_out,
            adj=A_dense,
            emb_dim=args.hidden_dim,
            max_nodes=max_num_nodes_global,
            node_ids=selected_nodes
        )

    elif model_name == 'STGCN': # 对应 STGCN-EC
        max_num_nodes_global = max([len(nodes) for nodes in args.nodes_per])
        
        model = SpatioTemporalModel(
            feat_dim=1, 
            hidden_dim=args.hidden_dim, 
            time_steps=args.t_in,
            output_dim=args.t_out,
            K=3, # 对应论文中的多项式阶数，默认为2或3
            max_nodes=max_num_nodes_global # 传入全局最大节点数
        )
        model.register_buffer('edge_index', edge_index)

    elif model_name == 'DyHSL' or (model_name == 'FedTSE' and args.trainer_mode == 'fed4tp'):
        scale_values = tuple(
            int(item.strip())
            for item in str(getattr(args, 'dyhsl_scales', '1,3,6,12')).split(',')
            if item.strip()
        )
        model = DyHSL(
            num_nodes=N,
            t_in=args.t_in,
            t_out=args.t_out,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            hidden_dim=args.hidden_dim,
            dropout=getattr(args, 'dyhsl_dropout', 0.1),
            num_backbone_layers=getattr(args, 'dyhsl_num_backbone_layers', 2),
            num_head_layers=getattr(args, 'dyhsl_num_head_layers', 2),
            num_hyper_edge=getattr(args, 'dyhsl_num_hyper_edge', 32),
            winsize=getattr(args, 'dyhsl_winsize', 3),
            scales=scale_values,
        )
        model.set_edge_index(edge_index)

    elif model_name == 'FedTSE':
        model = TrafficLSTM(
            num_nodes=N, 
            t_in=args.t_in, 
            input_size=args.input_dim, 
            hidden_size=args.hidden_dim, 
            output_size=args.t_out,
            output_dim=args.output_dim
        )

    elif model_name == 'FedOSTC':
        model = FedOSTC(
            enc_dim=args.hidden_dim, 
            gat_dim=args.hidden_dim, 
            pred_steps=args.t_out
        )
        model.register_buffer('edge_index', edge_index)

    elif model_name == 'pFedCTP':
    
        model = STNET_pFedCTP(
            num_nodes=N,
            hidden_dim=args.hidden_dim,
            his_num=args.t_in,
            pred_num=args.t_out,
            message_dim=args.input_dim,   # 比固定写 1 更稳
            gcn_layers=1,                 # 更接近官方仓库/论文默认设置
            edge_index=edge_index.to(args.device)
        )

    elif model_name == 'FedTPS':
        wave_mapping = {
            'PeMS03': 'haar',      # 或 'db2' (Daubechies)
            'PeMS04': 'coif1',     # Coiflets
            'PeMSD7': 'bior1.3',   # Biorthogonal (pytorch_wavelets 支持 bior1.x - bior6.x)
            'PeMS08': 'haar'       # 或 'db2'
        }
        chosen_wave = wave_mapping.get(args.dataset_name, 'coif1')
        if ctx.rank == 0:
            print(f"👉 [FedTPS] 自动匹配数据集 {args.dataset_name} 的最优小波基: {chosen_wave}")

        model = DCRNN_TP(
            num_nodes=N,                 
            input_dim=1,               
            output_dim=1,
            horizon=args.t_out,        
            rnn_units=64, 
            num_layers=2, 
            cheb_k=2,
            ycov_dim=1,                
            wave=chosen_wave           # <--- 替换掉硬编码的 "coif1"
        )
        model = model.to(args.device)

    elif model_name == 'FedGODE':
        
        node_indices = list(selected_nodes)
        dist_file = os.path.join(file_dir, 'data', args.dataset_name, 'distance.csv')
        A_sp_raw = np.zeros((N, N))
        
        if os.path.exists(dist_file):
            dist_df = pd.read_csv(dist_file)
            node_map = {global_id: i for i, global_id in enumerate(node_indices)}
            
            for _, row in dist_df.iterrows():
                u, v, cost = int(row['from']), int(row['to']), float(row['cost'])
                if u in node_map and v in node_map:
                    idx_u, idx_v = node_map[u], node_map[v]
                    # 计算高斯权重
                    weight = np.exp(- (cost**2) / (args.sigma2**2))
                    # 阈值过滤
                    if weight >= args.thres2:
                        A_sp_raw[idx_u, idx_v] = weight
                        A_sp_raw[idx_v, idx_u] = weight # 确保对称
        else:
            print(f"Rank {ctx.rank}: [警告] 未找到 distance.csv，回退至连通图归一化！")
            adj_dense = pyg_utils.to_dense_adj(edge_index, max_num_nodes=N)[0].numpy()
            A_sp_raw = adj_dense
            
        np.fill_diagonal(A_sp_raw, 1.0) # 添加自环
        A_sp_wave = get_normalized_adj(A_sp_raw).to(args.device)

        # ==========================================
        # 2. 制作语义矩阵 (A_se) - 局部DTW转换相似度 + 阈值过滤
        # ==========================================
        from lib.dtw import compute_dtw_matrix 
        dtw_filename = f"{args.dataset_name.lower()}_dtw_distance.npy"
        dtw_path = os.path.join(file_dir, 'data', args.dataset_name, dtw_filename)
        
        if os.path.exists(dtw_path):
            dtw_adj_full = np.load(dtw_path)
            dtw_adj_numpy = dtw_adj_full[np.ix_(node_indices, node_indices)] # 切片获取局部距离
        else:
            print(f"Rank {ctx.rank}: 未找到DTW缓存文件，正在计算局部 DTW...")
            dtw_adj_numpy = compute_dtw_matrix(train_set, num_nodes=N, limit_samples=100, epsilon=0.6)

        # DTW 距离转相似度
        A_se_raw = np.exp(- (dtw_adj_numpy**2) / (args.sigma1**2))
        A_se_raw[A_se_raw < args.thres1] = 0.0 # 阈值过滤
        np.fill_diagonal(A_se_raw, 1.0)
        A_se_wave = get_normalized_adj(A_se_raw).to(args.device)

        model = ODEGCN(
            num_nodes=N,
            num_features=1,  
            num_timesteps_input=args.t_in,
            num_timesteps_output=args.t_out,
            A_sp_hat=A_sp_wave,
            A_se_hat=A_se_wave
        )
        
    elif model_name == 'FedGTP':
        num_local_nodes = len(selected_nodes)
        max_num_nodes_global = max([len(nodes) for nodes in args.nodes_per])
        print(f"Rank {ctx.rank}: Local Nodes {num_local_nodes}, Global Max Padding to {max_num_nodes_global}")

        model = FedGTP_Model(
            num_nodes=num_local_nodes,      
            max_nodes=max_num_nodes_global, 
            in_dim=args.t_in,
            out_dim=args.t_out,
            feature_dim=args.input_dim,  
            hidden_dim=args.hidden_dim,    
            emb_dim=args.node_emb_dim,
            poly_k=args.poly_k   
        )
    
    elif model_name == 'FedSTN':
        # N 是在上面几行定义的 (N = len(selected_nodes))，代表当前客户端的节点数
        print(f"Rank {ctx.rank}: Initializing FedSTN with {N} nodes.")

        # 确保 edge_index 是 Tensor
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.LongTensor(edge_index)
        edge_index = edge_index.to(args.device)

        # 初始化模型
        model = FedSTN(
            num_nodes=N,                # <--- 这里直接用 N，不要用未定义的 num_nodes
            input_dim=args.input_dim,   # 记得在 config_args.py 里加这个参数
            hidden_dim=args.hidden_dim,
            out_dim=args.t_out,
            output_features=args.output_dim, # 记得在 config_args.py 里加这个参数
            edge_index=edge_index
        )

    elif model_name == 'CNFGNN':
        # 1. 获取当前客户端的真实节点数
        num_local_nodes = len(selected_nodes)
        print(f"Rank {ctx.rank}: Initializing CNFGNN with {num_local_nodes} nodes.")

        # 2. 确保 edge_index 是 Tensor 且在正确的设备上
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.LongTensor(edge_index)
        edge_index = edge_index.to(args.device)

        # =================【请复制这部分代码替换原有的 subgraph】=================
        # 彻底弃用 pyg_utils.subgraph，改用手动查表，保证索引绝对正确
        
        # A. 创建映射表: mapper[全局ID] = 本地ID
        # 找出可能的最大 ID (防止越界)
        max_id_in_edges = edge_index.max().item() if edge_index.numel() > 0 else 0
        max_global_id = max(max_id_in_edges, max(selected_nodes))
        
        # 初始化映射表为 -1 (表示该节点不存在于本地)
        mapper = torch.full((max_global_id + 1,), -1, dtype=torch.long, device=args.device)
        
        # 填入映射关系：selected_nodes 中的第 i 个节点 -> 本地 ID i
        # 这样能严格保证与 feature 矩阵 x 的行顺序一致
        nodes_tensor = torch.tensor(selected_nodes, dtype=torch.long, device=args.device)
        mapper[nodes_tensor] = torch.arange(len(selected_nodes), device=args.device)
        
        # B. 过滤掉不属于本地节点的边
        row, col = edge_index
        # 只有当 源节点 和 目标节点 都在映射表中(值 >= 0)时，才保留这条边
        mask = (mapper[row] >= 0) & (mapper[col] >= 0)
        
        valid_row = row[mask]
        valid_col = col[mask]
        
        # C. 转换为本地 ID (从全局 ID 变成 0, 1, 2...)
        new_row = mapper[valid_row]
        new_col = mapper[valid_col]
        
        # 重组 edge_index
        edge_index = torch.stack([new_row, new_col], dim=0)
        
        print(f"Rank {ctx.rank}: Edge Index Fixed. Valid Edges: {edge_index.shape[1]}")
        # =================【替换结束】=================

        # 3. 初始化模型
        model = CNFGNN(
            num_nodes=num_local_nodes,
            in_dim=args.t_in, 
            out_dim=args.t_out,
            hidden_dim=args.hidden_dim,
            edge_index=edge_index, # 使用修复后的 edge_index
            dropout=0.1
        )

    elif model_name == 'FCGCN':
        total_nodes_global = current_total_nodes 
        num_local_nodes = len(selected_nodes)
        node_to_idx = {global_id: i for i, global_id in enumerate(selected_nodes)}
        selected_nodes_set = set(selected_nodes)
        local_adj = torch.zeros((num_local_nodes, num_local_nodes), device=args.device)

        def _add_edges_from_local_edge_index(weight=1.0):
            if edge_index is None:
                return 0
            edge_tensor = edge_index
            if not isinstance(edge_tensor, torch.Tensor):
                edge_tensor = torch.LongTensor(edge_tensor)
            edge_tensor = edge_tensor.detach().cpu().long()
            if edge_tensor.numel() == 0 or edge_tensor.dim() != 2 or edge_tensor.shape[0] != 2:
                return 0

            added_entries = 0
            row_idx, col_idx = edge_tensor
            for u, v in zip(row_idx.tolist(), col_idx.tolist()):
                u, v = int(u), int(v)
                if u == v:
                    continue
                if 0 <= u < num_local_nodes and 0 <= v < num_local_nodes:
                    if local_adj[u, v].item() <= 0:
                        added_entries += 1
                    if local_adj[v, u].item() <= 0:
                        added_entries += 1
                    local_adj[u, v] = weight
                    local_adj[v, u] = weight
            return added_entries
    
        dist_file = os.path.join(file_dir, 'data', args.dataset_name, 'distance.csv')

        if os.path.exists(dist_file):
            print(f"Rank {ctx.rank}: [FCGCN 严格模式] 正在加载 distance.csv 并构建客户端内部无向加权图.")

            dist_df = pd.read_csv(dist_file)

            if 'from' not in dist_df.columns or 'to' not in dist_df.columns or 'cost' not in dist_df.columns:
                raise ValueError(
                    f"[FCGCN] distance.csv 必须包含 from/to/cost 三列，当前列为: {list(dist_df.columns)}"
                )

            graph_dist_df = dist_df.copy()
            raw_ids = pd.concat([graph_dist_df['from'], graph_dist_df['to']], ignore_index=True).dropna().astype(int)
            unique_raw_ids = sorted(raw_ids.unique().tolist())
            uses_compact_ids = (
                len(unique_raw_ids) > 0
                and min(unique_raw_ids) >= 0
                and max(unique_raw_ids) < total_nodes_global
            )
            if not uses_compact_ids and len(unique_raw_ids) == total_nodes_global:
                raw_to_compact = {raw_id: idx for idx, raw_id in enumerate(unique_raw_ids)}
                graph_dist_df['from'] = graph_dist_df['from'].astype(int).map(raw_to_compact)
                graph_dist_df['to'] = graph_dist_df['to'].astype(int).map(raw_to_compact)
                graph_dist_df = graph_dist_df.dropna(subset=['from', 'to']).copy()
                graph_dist_df['from'] = graph_dist_df['from'].astype(int)
                graph_dist_df['to'] = graph_dist_df['to'].astype(int)
                print(
                    f"Rank {ctx.rank}: [FCGCN Graph ID Map] distance.csv raw node ids mapped "
                    f"to compact ids 0..{total_nodes_global - 1}."
                )
            elif not uses_compact_ids:
                print(
                    f"Rank {ctx.rank}: [FCGCN Graph ID Warning] distance.csv node ids are not compact and "
                    f"unique_nodes={len(unique_raw_ids)} != total_nodes={total_nodes_global}; "
                    "will fall back to edge_index if no client edges are found."
                )

            is_already_weight = graph_dist_df['cost'].max() <= 1.0
            K = 8

            if is_already_weight:

                # 1. 先过滤当前客户端内部边
                local_dist_df = graph_dist_df[
                    graph_dist_df['from'].isin(selected_nodes_set) &
                    graph_dist_df['to'].isin(selected_nodes_set)
                ].copy()

                # 2. 去掉自环，避免 Top-K 把自己算进去
                local_dist_df = local_dist_df[local_dist_df['from'] != local_dist_df['to']]

                # 3. 在客户端内部按权重从大到小取 Top-K
                topk_dist = (
                    local_dist_df
                    .sort_values(['from', 'cost'], ascending=[True, False])
                    .groupby('from', group_keys=False)
                    .head(K)
                )

                # 4. 对称化
                rev_dist = topk_dist.copy()
                rev_dist['from'], rev_dist['to'] = topk_dist['to'].values, topk_dist['from'].values

                relevant_dists = (
                    pd.concat([topk_dist, rev_dist], ignore_index=True)
                    .drop_duplicates(subset=['from', 'to'])
                )

                print(
                    f"Rank {ctx.rank}: [FCGCN-D7] 客户端内部 Top-{K} 构图完成 | "
                    f"local_nodes={num_local_nodes} | local_edges={len(relevant_dists)}"
                )

            else:
                # ==========================================================
                # PeMS04/PeMS08 这类物理距离：
                # 保持原始逻辑：只取当前客户端内部边，用 1/cost 作为权重
                # ==========================================================
                relevant_dists = graph_dist_df[
                    graph_dist_df['from'].isin(selected_nodes_set) &
                    graph_dist_df['to'].isin(selected_nodes_set)
                ].copy()

                if ctx.rank == 0:
                    print("👉 [FCGCN] 检测到物理距离，将使用 1/cost 作为权重。")

            # 5. 写入 local_adj
            for _, row in relevant_dists.iterrows():
                u, v, cost = int(row['from']), int(row['to']), float(row['cost'])

                if u == v:
                    continue

                if cost > 0 and u in node_to_idx and v in node_to_idx:
                    idx_u, idx_v = node_to_idx[u], node_to_idx[v]

                    weight = cost if is_already_weight else 1.0 / cost

                    local_adj[idx_u, idx_v] = weight
                    local_adj[idx_v, idx_u] = weight

            # 6. 打印构图质量，方便你判断 D7 是否还异常
            deg_before_loop = local_adj.sum(dim=1)
            zero_deg = int((deg_before_loop == 0).sum().item())
            edge_num = int((local_adj > 0).sum().item())

            if edge_num == 0:
                added_entries = _add_edges_from_local_edge_index(weight=1.0)
                if added_entries > 0:
                    deg_before_loop = local_adj.sum(dim=1)
                    zero_deg = int((deg_before_loop == 0).sum().item())
                    edge_num = int((local_adj > 0).sum().item())
                    print(
                        f"Rank {ctx.rank}: [FCGCN Graph Fallback] distance.csv produced 0 edges; "
                        f"rebuilt adjacency from load_dataset edge_index with added_entries={added_entries}."
                    )

            print(
                f"Rank {ctx.rank}: [FCGCN Graph Check] "
                f"nodes={num_local_nodes} | nonzero_adj_entries={edge_num} | "
                f"zero_degree_before_self_loop={zero_deg} | "
                f"deg_min={float(deg_before_loop.min().item()):.4f} | "
                f"deg_max={float(deg_before_loop.max().item()):.4f} | "
                f"deg_mean={float(deg_before_loop.mean().item()):.4f}"
            )

        else:
            print(f"Rank {ctx.rank}: [警告] 未找到距离文件，使用 edge_index 构建连通矩阵。")
            added_entries = _add_edges_from_local_edge_index(weight=1.0)
            deg_before_loop = local_adj.sum(dim=1)
            print(
                f"Rank {ctx.rank}: [FCGCN Graph Check] "
                f"nodes={num_local_nodes} | nonzero_adj_entries={int((local_adj > 0).sum().item())} | "
                f"zero_degree_before_self_loop={int((deg_before_loop == 0).sum().item())} | "
                f"fallback_added_entries={added_entries}"
            )
        
        local_adj.fill_diagonal_(1.0)
    
        row_sum = local_adj.sum(dim=1)
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    
        a_hat = d_inv_sqrt.view(-1, 1) * local_adj * d_inv_sqrt.view(1, -1)

        if args.model == "FCGCN" and args.dataset_name == "PeMSD7":
            alpha = 1  

            I = torch.eye(
                num_local_nodes,
                dtype=a_hat.dtype,
                device=a_hat.device
            )

            a_hat = alpha * I + (1.0 - alpha) * a_hat

            a_hat = 0.5 * (a_hat + a_hat.T)

            print(
                f"Rank {ctx.rank}: [FCGCN-D7 ResidualAdj] "
                f"A_mix = {alpha:.2f} * I + {1.0 - alpha:.2f} * A_hat"
            )
        elif args.model == "FCGCN":
            alpha = float(getattr(args, "fcgcn_adj_residual_alpha", 0.9))
            alpha = max(0.0, min(1.0, alpha))
            if alpha > 0.0:
                I = torch.eye(
                    num_local_nodes,
                    dtype=a_hat.dtype,
                    device=a_hat.device
                )
                a_hat = alpha * I + (1.0 - alpha) * a_hat
                a_hat = 0.5 * (a_hat + a_hat.T)
                print(
                    f"Rank {ctx.rank}: [FCGCN ResidualAdj] "
                    f"A_mix = {alpha:.2f} * I + {1.0 - alpha:.2f} * A_hat"
                )


        model = FCGCN(
            num_nodes=num_local_nodes,
            in_dim=args.t_in,
            out_dim=args.t_out,
            hidden_dim=args.hidden_dim,
            adj_matrix=a_hat
        )
    
        del local_adj, row_sum, d_inv_sqrt
    
    elif model_name == 'FCFedGCN':
        print(f"Rank {ctx.rank}: [FC-FedGCN_Paper] 正在初始化严格复现版模型(搭载智能加权图)...")
        
        num_local_nodes = len(selected_nodes)
        
        # 1. FCA 拓扑特征提取
        MAX_FCA_DIM = 100
        fca_features = get_equiconcept_matrix(edge_index, num_local_nodes, args.device, max_fca_dim=MAX_FCA_DIM)
        
        # 2. GCN 距离加权图构造
        local_adj = torch.zeros((num_local_nodes, num_local_nodes), device=args.device)
        node_to_idx = {global_id: i for i, global_id in enumerate(selected_nodes)}
        selected_nodes_set = set(selected_nodes)

        dist_file = os.path.join(file_dir, 'data', args.dataset_name, 'distance.csv')
        
        if os.path.exists(dist_file):
            print(f"Rank {ctx.rank}: [严格模式] 正在加载物理距离构建无向加权图...")
            dist_df = pd.read_csv(dist_file)
            
            is_already_weight = dist_df['cost'].max() <= 1.0
            
            # 【核心修复：Top-K 图稀疏化同步】
            if is_already_weight:
                K = 8
                topk_dist = dist_df.sort_values(['from', 'cost'], ascending=[True, False]).groupby('from').head(K)
                rev_dist = topk_dist.copy()
                rev_dist['from'], rev_dist['to'] = topk_dist['to'], topk_dist['from']
                dist_df = pd.concat([topk_dist, rev_dist]).drop_duplicates(subset=['from', 'to'])
                
                if ctx.rank == 0:
                    print(f"👉 检测到完全图权重，已执行 Top-{K} 严格稀疏化与对称化！")
            elif ctx.rank == 0:
                print("👉 检测到物理距离，将使用 1/cost 作为权重")

            # 过滤出属于当前客户端的边
            mask = dist_df['from'].isin(selected_nodes_set) & dist_df['to'].isin(selected_nodes_set)
            relevant_dists = dist_df[mask]
                    
        else:
            print(f"Rank {ctx.rank}: [警告] 未找到 distance.csv，退化为普通 edge_index")
            row, col = edge_index
            for u, v in zip(row.tolist(), col.tolist()):
                local_adj[u, v] = 1.0
                local_adj[v, u] = 1.0

        local_adj.diagonal().add_(1.0)
        
        row_sum = local_adj.sum(dim=1)
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        a_hat = d_inv_sqrt.view(-1, 1) * local_adj * d_inv_sqrt.view(1, -1)

        # 3. 初始化模型
        model = FC_FedGCN_Traffic(
            num_nodes=num_local_nodes,
            in_dim=args.t_in,
            out_dim=args.t_out,
            hidden_dim=args.hidden_dim,
            fca_dim=MAX_FCA_DIM,
            adj_matrix=a_hat
        )
        model.set_fca_features(fca_features)
        print(f"✅ 模型初始化完成! FCA维度统一对齐至: {MAX_FCA_DIM}")
 
    elif model_name == 'FGNNEH':
        print(f"Rank {ctx.rank}: Initializing FGNNEH Model...")
        
        # 1. 确保 edge_index 是 LongTensor 且在正确的设备上
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.LongTensor(edge_index)
        edge_index = edge_index.to(args.device)
        
        # 2. 生成 Dense Matrix (这是 Float 类型，用于 matmul)
        adj_dense = pyg_utils.to_dense_adj(edge_index, max_num_nodes=N)[0].to(args.device)
        
        # 3. 初始化算法组件
        backbone_extractor = FGNNEH_Backbone(
            adj_matrix=adj_dense, 
            P=args.fgnneh_P, 
            gamma=args.fgnneh_gamma, 
            n_components=args.fgnneh_n_components, 
            device=args.device
        )
        
        # 4. 初始化模型
        model = FGNNEH_Client(
            num_nodes=N,
            in_dim=args.t_in * args.input_dim,
            out_dim=args.t_out,
            hidden_dim=args.hidden_dim,
            backbone_extractor=backbone_extractor
        )

    elif model_name == 'FedMetro':
        # 1. 获取当前客户端的真实节点数
        num_local_nodes = len(selected_nodes)
        print(f"Rank {ctx.rank}: Initializing FedMetro with {num_local_nodes} local nodes.")
        
        # 2. 传入配置好的参数实例化 FedMetro 模型
        model = FedMetro_Client_Model(
            num_nodes=num_local_nodes,
            t_in=args.t_in,
            t_out=args.t_out,
            input_dim=args.input_dim,   # 比如 1 (流量)
            hidden_dim=args.hidden_dim, # 比如 32 或 64
            d_E=args.node_emb_dim,      # 复用 node_emb_dim (默认 2-4)
            K=args.poly_k               # 多项式分解的阶数 K (默认 4)
        )
        
        # 将配置中的掩码参数直接传给 HardConcreteMask
        model.dyn_emb_mask.mask_generator.beta = args.fedmetro_mask_beta
        model.dyn_emb_mask.mask_generator.gamma = args.fedmetro_mask_gamma
        model.dyn_emb_mask.mask_generator.zeta = args.fedmetro_mask_zeta

    elif model_name == 'STFAM':
        loss_func = torch.nn.L1Loss().to(args.device) if args.loss_func == 'mae' else torch.nn.MSELoss().to(args.device)
        return train_set, val_set, test_set, None, None, loss_func, None, None, None, scaler, args

    elif model_name == 'FedSTG':
        if ctx.is_on_arbiter:
            # Server 端只负责图结构学习，不需要本地节点数，但需要全局维度
            model = FedSTG_Server(
                hidden_dim=args.hidden_dim, 
                beta=getattr(args, 'fedstg_beta', 0.5)
            )
            print(f"Rank {ctx.rank}: [FedSTG Server] Initialized.")
        else:
            # Client 端负责本地时序模式提取
            num_local_nodes = len(selected_nodes)
            model = FedSTG_Client(
                in_dim=args.input_dim,
                out_dim=args.t_out,
                hidden_dim=args.hidden_dim,
                K=getattr(args, 'fedstg_K', 10),
                d=getattr(args, 'node_emb_dim', 32), # 借用 node_emb_dim 作为模式维度 d
                seq_len=args.t_in,
                num_nodes=num_local_nodes
            )
            print(f"Rank {ctx.rank}: [FedSTG Client] Initialized with {num_local_nodes} nodes.")

    elif model_name == 'FUELS':
        model = FUELS_Model(
            num_nodes=N,
            in_dim=args.input_dim,
            out_dim=args.output_dim,
            hidden_dim=args.hidden_dim,
            dr=args.fuels_dr,
            batch_size=args.batch_size,
            seq_len=args.t_in,
            pred_len=args.t_out,
            device=args.device,
            fuels_c=args.fuels_c,
            fuels_q=args.fuels_q,
            aug_noise_std=getattr(args, 'fuels_aug_noise_std', 0.01),
            aug_mask_ratio=getattr(args, 'fuels_aug_mask_ratio', 0.10),
            aug_shift_prob=getattr(args, 'fuels_aug_shift_prob', 0.50),
            aug_shift_pad_mode=getattr(args, 'fuels_aug_shift_pad_mode', 'edge'),
        )

    elif model_name == 'UFCL_GWN':
        model = LightGraphWaveNet(
            num_nodes=N,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            horizon=args.t_out,
            residual_channels=args.ufcl_gwn_channels,
            skip_channels=args.ufcl_gwn_skip_channels,
            end_channels=args.ufcl_gwn_end_channels,
            blocks=args.ufcl_gwn_blocks,
            layers=args.ufcl_gwn_layers,
            dropout=args.ufcl_gwn_dropout,
        )
        model.set_edge_index(edge_index)

    elif model_name == 'FedGRU':
        # 实例化 FedGRU，默认采用论文推荐的 2 层结构
        model = FedGRU_Model(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,  # 推荐设为 100
            out_dim=args.output_dim,
            pre_len=args.t_out,
            num_layers=2
        )

    elif model_name == 'SFL_RNN':
        # SFL client backbone: explicitly a pure, one-layer vanilla RNN.
        # The SFL relation graph is built by train_sfl on the arbiter, not by
        # this local predictor.
        model = SFLPureRNN(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            t_in=args.t_in,
            t_out=args.t_out,
        )

    elif model_name == 'FedmSSA':
        model = FedmSSA_Model(
            num_nodes=N,
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            out_dim=args.output_dim,
            pre_len=args.t_out,
            page_length=args.fedmssa_page_length,
            selected_nodes=selected_nodes,
            num_layers=args.fedmssa_gru_layers,
            dropout=args.fedmssa_dropout,
        )

    elif model_name in ('TDLR', 'SEDLR', 'TDLR_SEDLR'):
        model = StreamingTrafficLSTM(
            num_nodes=N,
            t_in=args.t_in,
            input_size=args.input_dim,
            hidden_size=args.hidden_dim,
            output_size=args.t_out,
            output_dim=args.output_dim,
            num_layers=args.tdlr_lstm_layers,
            dropout=args.tdlr_dropout,
        )

    elif model_name == 'REFOL':
        from model.refol_nets import GRU
        model = GRU(
            input_size=args.input_dim,
            hidden_size=args.hidden_dim,
            output_size=args.output_dim,
            dropout=getattr(args, 'dropout', 0.0),
            gru_num_layers=1
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}. Please choose from [ISTGNN, TwoMGTCN, ASTGAT, STGCN, UFCL_GWN, DyHSL, FedTSE, FedOSTC, STNET, FedTPS, FedmSSA, TDLR, SEDLR, TDLR_SEDLR]")

    
    if model_name == 'TwoMGTCN':
        print(f"[2MGTCN get_setting][rank {ctx.rank}] stage=before_model_to_device", flush=True)
    model = model.to(args.device)
    if model_name == 'TwoMGTCN':
        print(f"[2MGTCN get_setting][rank {ctx.rank}] stage=after_model_to_device", flush=True)
   
    # Keep full datasets on CPU for multi-process FATE baselines.  FedGRU and
    # FedmSSA move their current mini-batch to the model device inside their
    # trainers; moving every 32-client TaxiBJ split here exhausts one 3090
    # before the first training step.
    if model_name not in (
        'FedTSE', 'TwoMGTCN', 'TDLR', 'SEDLR', 'TDLR_SEDLR', 'FedGRU', 'FedmSSA'
    ):
        if hasattr(train_set, 'data'):
            # 假设 data 是 Tensor。如果是 numpy，先转 Tensor
            if isinstance(train_set.data, np.ndarray):
                train_set.data = torch.from_numpy(train_set.data)
            train_set.data = train_set.data.to(args.device)
        # 同理搬运 val_set
        if hasattr(val_set, 'data'):
            if isinstance(val_set.data, np.ndarray):
                val_set.data = torch.from_numpy(val_set.data)
            val_set.data = val_set.data.to(args.device)
    
    # ================= [新增结束] =================



    if args.loss_func == 'mae':
        base_loss = torch.nn.L1Loss().to(args.device)
    elif args.loss_func == 'mse':
        base_loss = torch.nn.MSELoss().to(args.device)
    elif args.loss_func in ('smoothl1', 'huber'):
        base_loss = torch.nn.SmoothL1Loss().to(args.device)
    else:
        base_loss = torch.nn.MSELoss().to(args.device)

    if model_name == 'ASTGAT':
        loss = FedAGATLoss(base_loss_func=base_loss, max_epochs=args.epochs).to(args.device)
        loss.batches_per_epoch = max(1, len(train_set) // args.batch_size) # 加个 max(1, x) 防止除 0
    else:
        loss = base_loss

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = None
    use_cpu = args.device == 'cpu'

   
    if args.model == 'REFOL':
        train_args = None
    else:
        checkpoint_tag = "_".join(
            str(part).replace("/", "_").replace("\\", "_").replace(" ", "_")
            for part in (
                args.model,
                args.dataset_name,
                args.feature_type,
                _benchmark_run_id(),
                f"client_{ctx.rank}",
            )
        )
        train_args = TrainingArguments(
            output_dir=os.path.join("./checkpoints", checkpoint_tag),
            save_total_limit=2,
            num_train_epochs=args.epochs,
            evaluation_strategy="epoch",
            # FATE's native FedAVG trainer performs its epoch-level protocol
            # bookkeeping through this cadence.  Keep it for UFCL, but lower
            # the Transformers logger itself below so it does not flood logs.
            logging_strategy="epoch",
            log_level="error" if getattr(args, 'trainer_mode', '') == 'ufcl' else "passive",
            save_strategy="epoch",
            per_device_train_batch_size=args.batch_size, 
            dataloader_pin_memory=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_mae",
            greater_is_better=False
        )
        train_args.ufcl_max_nodes = max_num_nodes_global 
        if getattr(args, 'trainer_mode', '') != 'ufcl':
            print(f"Rank {ctx.rank}: [Config] Successfully injected ufcl_max_nodes={max_num_nodes_global} into TrainArgs")
   

   
    if hasattr(args, 'dp_noise') and args.dp_noise > 0 and args.model == 'FedGODE':
        if ctx.rank != 0: 
            print(f"Rank {ctx.rank}: 🛡️ 已激活 LDP (局部差分隐私) 机制，Laplace 噪声强度: {args.dp_noise}")
            clip_value = 1.0 
            
            for p in model.parameters():
                if p.requires_grad:
                    def dp_hook(grad, clip=clip_value, noise_std=args.dp_noise):
                        grad_norm = torch.norm(grad, p=2)
                        clip_coef = clip / (grad_norm + 1e-6)
                        if clip_coef < 1:
                            grad = grad * clip_coef
                        noise = torch.distributions.Laplace(0, noise_std).sample(grad.shape).to(grad.device)
                        return grad + noise
                        
                    p.register_hook(dp_hook)

    if args.protection == 'he' and args.he_backend == 'he_sa' and args.model not in ('FedGODE', 'FCFedGCN', 'FedGTP'):
        fed_arg = FedAVGArguments(
            aggregate_strategy='epoch',
            aggregate_freq=args.local_epochs,
            aggregator='secure_aggregate',
        )
        print(f"Rank {ctx.rank}: [{args.model}] FATE secure_aggregate requested for HE experiment.")
    elif args.model in ['FedGODE']:
        fed_arg = FedAVGArguments(
            aggregate_strategy='epoch',
            aggregate_freq=args.local_epochs,
        )
        print(f"Rank {ctx.rank}: [{args.model}] 使用标准 FedAvg 聚合（未启用 HE/安全聚合）！")
    else:
        fed_arg = FedAVGArguments(
            aggregate_strategy='epoch',
            aggregate_freq=args.local_epochs
        )

    return train_set, val_set, test_set, model, optimizer, loss, lr_scheduler, train_args, fed_arg, scaler, args


def train(ctx: Context,
          train_data=None,
          val_data=None,
          test_data=None,
          model=None,
          optimizer=None,
          loss_func=None,
          lr_scheduler=None,
          train_args: TrainingArguments = None,
          fed_args: FedAVGArguments = None,
          scaler=None,
          extra_args=None
          ):

    # 早停关键点
    stopper = ExplicitEarlyStopper(ctx, args, patience=50, min_delta=1e-5)
    phase_flag = {"is_testing": False}
    def metrics(pred):
        # 取出归一化后的数据
        y_true_norm = pred.label_ids
        y_pred_norm = pred.predictions
        
        # 1. 计算 MAE/MSE (基于归一化数据，后续在外面乘 scaler.metrics_coef 还原，保持 FATE 原逻辑)
        mae = np.mean(np.abs(y_true_norm - y_pred_norm))
        mse = np.mean((y_true_norm - y_pred_norm) ** 2)
        rmse = np.sqrt(mse)
        
        # 2. 计算 MAPE (必须先反归一化还原为真实交通流)
        if hasattr(scaler, 'inverse_transform'):
            y_true_real = scaler.inverse_transform(y_true_norm)
            y_pred_real = scaler.inverse_transform(y_pred_norm)
        else:
            # 如果你的 scaler 只有 mean/std 属性
            y_true_real = y_true_norm * scaler.std + scaler.mean
            y_pred_real = y_pred_norm * scaler.std + scaler.mean

        # 过滤真实流量小于 0.5 的车次 (交通流通常是整数，<0.5基本就是无车)
        mask = y_true_real > 0.5
        
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((y_true_real[mask] - y_pred_real[mask]) / y_true_real[mask])) * 100
        else:
            mape = 0.0

        # 【修改】：如果不是在最后的测试阶段，才去执行早停检测
        if not phase_flag["is_testing"]:
            if stopper.check_and_sync(mae):
                print(f"Rank {ctx.rank}: 🛑 收到全局早停信号，抛出中断异常！")
                raise EarlyStopSignal("触发早停机制，切断框架死循环")
        return {
            'mae': mae,
            'rmse': rmse,
            'mse': mse,
            'mape': mape
        }

    def prediction_metric_sums(prediction_output):
        def _to_np(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        y_true_norm = _to_np(prediction_output.label_ids)
        y_pred_norm = _to_np(prediction_output.predictions)
        if hasattr(scaler, 'inverse_transform'):
            y_true_real = _to_np(scaler.inverse_transform(y_true_norm))
            y_pred_real = _to_np(scaler.inverse_transform(y_pred_norm))
        else:
            y_true_real = y_true_norm * scaler.std + scaler.mean
            y_pred_real = y_pred_norm * scaler.std + scaler.mean

        diff = y_pred_real - y_true_real
        mask = y_true_real > 0.5
        return {
            "elements": int(y_true_real.size),
            "abs_error_sum": float(np.abs(diff).sum()),
            "sq_error_sum": float(np.square(diff).sum()),
            "mape_error_sum": float((np.abs(diff[mask]) / y_true_real[mask]).sum() * 100.0) if np.any(mask) else 0.0,
            "mape_elements": int(mask.sum()) if np.any(mask) else 0,
        }

    
    
    if ctx.is_on_guest or ctx.is_on_host:
        if args.model == 'FedGRU':
            client_class = FedGRUFedAVGClient
        elif hasattr(args, 'trainer_mode') and args.trainer_mode == 'ufcl':
            client_class = UFCLFedAVGClient
        else:
            client_class = FedAVGClient
    
        trainer = client_class(
            ctx=ctx, model=model, train_set=train_data, val_set=val_data,
            optimizer=optimizer, loss_fn=loss_func, scheduler=lr_scheduler,
            training_args=train_args, fed_args=fed_args, compute_metrics=metrics
        )
        # Used only by the explicitly enabled revised quantized-prediction
        # trace; normal FedAVG/FedGRU training does not expose this message.
        trainer.privacy_args = args
        if hasattr(trainer, "trainer"):
            trainer.trainer.privacy_args = args
    else:
        trainer = FedAVGServer(ctx)

    # 3. 捕捉异常，让程序“平稳落地”并继续写入你的 benchmark_results.csv
    start_train_time = time.time()

    try:
        trainer.train()
    except EarlyStopSignal:
        print(f"Rank {ctx.rank}: 🎉 成功穿透黑盒跳出训练循环！准备写入测试评估数据...")

    
    actual_train_time = time.time() - start_train_time

   
    if len(trainer.state.log_history) > 0:
        trainer.state.log_history[-1]["train_runtime"] = actual_train_time
    else:
        trainer.state.log_history.append({"train_runtime": actual_train_time})

    print(f"Rank {ctx.rank}: 本次联邦训练实际耗时 {actual_train_time:.2f} 秒。")
    

    
    # 写csv
    if not ctx.is_on_arbiter:
        # 1. Accuracy 指标提取
        eval_maes = [d["eval_mae"] * scaler.metrics_coef for d in trainer.state.log_history if "eval_mae" in d]
        eval_rmses = [d["eval_rmse"] * scaler.metrics_coef for d in trainer.state.log_history if "eval_rmse" in d]
        eval_mses = [d["eval_mse"] * (scaler.metrics_coef ** 2) for d in trainer.state.log_history if "eval_mse" in d]
        eval_mapes = [d["eval_mape"] for d in trainer.state.log_history if "eval_mape" in d]
        
        if eval_maes and eval_rmses:
            best_mae = min(eval_maes)
            # Python 列表索引是从 0 开始的，所以真实的轮次要 +1
            best_epoch_idx = eval_maes.index(best_mae) 
            best_epoch = best_epoch_idx + 1  # <--- 这就是你要的 best_epoch
            
            acc_mae = round(best_mae, 4)
            acc_rmse = round(eval_rmses[best_epoch_idx], 4)
            acc_mse = round(eval_mses[best_epoch_idx], 4) if eval_mses else 0.0
            acc_mape = round(eval_mapes[best_epoch_idx], 4) if eval_mapes else 0.0
            
            # 2. Efficiency 指标计算
            # (1) 训练总时间
            eff_train_time = round(trainer.state.log_history[-1].get("train_runtime", 0.0), 2)
            
            # (2) 验证集耗时 (Validation Time) - 提取平均每轮的验证耗时
            eval_runtimes = [d["eval_runtime"] for d in trainer.state.log_history if "eval_runtime" in d]
            eff_val_time = round(sum(eval_runtimes) / len(eval_runtimes), 4) if eval_runtimes else 0.0
            
            # (3) 测试集耗时 (Test Time) 
            model.eval()
            start_test_time = time.time()
            acc_elements = ""
            acc_abs_error_sum = ""
            acc_sq_error_sum = ""
            acc_mape_error_sum = ""
            acc_mape_elements = ""

            if test_data is not None:
                _restore_best_checkpoint_for_prediction(trainer, model, args.device, rank=ctx.rank)
                phase_flag["is_testing"] = True
                test_output = trainer.predict(test_data)
                sample_stats = prediction_metric_sums(test_output)
                acc_elements = sample_stats["elements"]
                acc_abs_error_sum = sample_stats["abs_error_sum"]
                acc_sq_error_sum = sample_stats["sq_error_sum"]
                acc_mape_error_sum = sample_stats["mape_error_sum"]
                acc_mape_elements = sample_stats["mape_elements"]
                if acc_elements:
                    acc_mae = round(acc_abs_error_sum / max(acc_elements, 1), 4)
                    acc_mse = round(acc_sq_error_sum / max(acc_elements, 1), 4)
                    acc_rmse = round(math.sqrt(acc_sq_error_sum / max(acc_elements, 1)), 4)
                    acc_mape = round(
                        acc_mape_error_sum / max(acc_mape_elements, 1) if acc_mape_elements else 0.0,
                        4,
                    )

                if args.model == "FCGCN" and args.dataset_name == "PeMSD7" and args.feature_type == "flow":

                    def _to_np(z):
                        if isinstance(z, torch.Tensor):
                            return z.detach().cpu().numpy()
                        return np.asarray(z)

                    y_true_norm = _to_np(test_output.label_ids)
                    y_pred_norm = _to_np(test_output.predictions)

                    x_test_tensor, y_test_tensor = test_data.tensors
                    naive_norm = x_test_tensor[:, :, -1:, :].repeat(1, 1, args.t_out, 1)
                    naive_norm = _to_np(naive_norm)

                    y_true_real = _to_np(scaler.inverse_transform(y_true_norm))
                    y_pred_real = _to_np(scaler.inverse_transform(y_pred_norm))
                    naive_real = _to_np(scaler.inverse_transform(naive_norm))

                    print("=" * 100)
                    print(f"Rank {ctx.rank}: [FCGCN-D7 Prediction Distribution Check]")

                    print(
                        f"y_true_norm | mean={y_true_norm.mean():.6f}, std={y_true_norm.std():.6f}, "
                        f"min={y_true_norm.min():.6f}, max={y_true_norm.max():.6f}"
                    )
                    print(
                        f"y_pred_norm | mean={y_pred_norm.mean():.6f}, std={y_pred_norm.std():.6f}, "
                        f"min={y_pred_norm.min():.6f}, max={y_pred_norm.max():.6f}"
                    )
                    print(
                        f"naive_norm  | mean={naive_norm.mean():.6f}, std={naive_norm.std():.6f}, "
                        f"min={naive_norm.min():.6f}, max={naive_norm.max():.6f}"
                    )

                    fcgcn_mae = np.mean(np.abs(y_true_real - y_pred_real))
                    naive_mae = np.mean(np.abs(y_true_real - naive_real))

                    print(f"FCGCN_REAL_MAE={fcgcn_mae:.4f}")
                    print(f"NAIVE_REAL_MAE={naive_mae:.4f}")
                    print("=" * 100)
                
               
                if (not acc_elements) and hasattr(test_output, 'metrics'):
                    test_metrics = test_output.metrics
                    acc_mae = round(test_metrics.get("test_mae", 0.0) * scaler.metrics_coef, 4)
                    acc_rmse = round(test_metrics.get("test_rmse", 0.0) * scaler.metrics_coef, 4)
                    acc_mse = round(test_metrics.get("test_mse", 0.0) * (scaler.metrics_coef ** 2), 4)
                    acc_mape = round(test_metrics.get("test_mape", 0.0), 4) 

            eff_test_time = round(time.time() - start_test_time, 4)
            
            actual_epochs_ran = int(trainer.state.epoch) if trainer.state.epoch is not None else args.epochs
            eff_train_round = actual_epochs_ran 
            
            model_comm_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            eff_comm_size_mb = round((model_comm_params * 4 * 2 * actual_epochs_ran) / (1024 * 1024), 4)
          
            
            # (6) FLOPs (浮点运算次数) 
            eff_flops = 0.0
            try:
                from thop import profile
                dummy_x, _ = next(iter(DataLoader(val_data, batch_size=args.batch_size)))
                dummy_x = dummy_x.to(args.device)
                flops, _ = profile(model, inputs=(dummy_x,), verbose=False)
                eff_flops = round(flops / 1e9, 4) 
            except Exception as e:
                print(f"Rank {ctx.rank}: FLOPs 计算失败，已回退为 0.0。原因: {e}", flush=True)
                eff_flops = 0.0
            if eff_flops == 0.0 and args.model == "FedGRU":
                try:
                    if 'dummy_x' not in locals():
                        dummy_x, _ = next(iter(DataLoader(val_data, batch_size=args.batch_size)))
                    eff_flops = _estimate_fedgru_flops(model, dummy_x)
                    print(f"Rank {ctx.rank}: FedGRU FLOPs 使用解析估算: {eff_flops} G", flush=True)
                except Exception as e:
                    print(f"Rank {ctx.rank}: FedGRU FLOPs 解析估算失败，仍回退为 0.0。原因: {e}", flush=True)
                    eff_flops = 0.0
            
            dataset_client_name = f"{args.dataset_name}_client{ctx.rank}"
            result_model_name = "UFCL" if getattr(args, "trainer_mode", "") == "ufcl" else args.model
            
            # 写入终极版 CSV
            log_experiment_results(
                model_name=result_model_name, 
                dataset_client=dataset_client_name, 
                feature_type=args.feature_type,
                best_epoch=best_epoch, # <--- 传入 best_epoch
                
                # Accuracy
                acc_mae=acc_mae, acc_mse=acc_mse, acc_rmse=acc_rmse, acc_mape=acc_mape,
                
                # Efficiency
                eff_train_time=eff_train_time, 
                eff_val_time=eff_val_time,   # <--- 传入 val_time
                eff_test_time=eff_test_time, 
                eff_comm_size_mb=eff_comm_size_mb, eff_train_round=eff_train_round, eff_flops=eff_flops,
                acc_elements=acc_elements,
                acc_abs_error_sum=acc_abs_error_sum,
                acc_sq_error_sum=acc_sq_error_sum,
                acc_mape_error_sum=acc_mape_error_sum,
                acc_mape_elements=acc_mape_elements
            )
            print(f"🎉 Client {ctx.rank} ({args.model} - {args.feature_type}) 实验完成！")

    # print(ctx.rank)
    #
    # if ctx.is_on_guest:
    #     loss_record = [d["loss"] for d in trainer.state.log_history[:-1:2]]
    #
    #     plt.plot(list(range(len(loss_record))), loss_record)
    #     plt.show()
    # train_end_time = time.time()
    if ctx.is_on_guest:
        train_loss_record = [d["loss"] * scaler.metrics_coef for d in trainer.state.log_history[:-1:2]]
        eval_mae_record = [d["eval_mae"] * scaler.metrics_coef for d in trainer.state.log_history[1:-1:2]]
        # eval_rmse_record = [d["eval_rmse"] * scaler.metrics_coef for d in trainer.state.log_history[1:-1:2]]
        #
        #
        #
        with open('./eval_mae.txt', 'w') as f:
            for num in eval_mae_record:
                f.write(f"{num:.4f}\n")
        #
        # with open('./eval_rmse.txt', 'w') as f:
        #     for num in eval_rmse_record:
        #         f.write(f"{num:.4f}\n")

        source_train_time = trainer.state.log_history[-1]["train_runtime"]


        # dataset_name = args.data_list[-1]
        # train_set, val_set, A_norm, scaler = load_dataset(dataset_name=dataset_name,
        #                                                             feature_type=args.feature_type,
        #                                                             normalizer=args.normalizer,
        #                                                             T_in=args.t_in,
        #                                                             T_out=args.t_out,
        #                                                             train_ratio=args.target_train_ratio,
        #                                                             val_ratio=args.target_test_ratio, device=args.device)

        # train_set, val_set, edge_index, scaler = load_dataset(dataset_name=dataset_name, feature_type=args.feature_type,
        #                                                       normalizer=args.normalizer,
        #                                                       T_in=args.t_in,
        #                                                       T_out=args.t_out, train_ratio=args.train_ratio,
        #                                                       val_ratio=args.val_ratio, return_edge_index=True,
        #                                                       device=args.device)
        # N = train_set[0][0].shape[0]
        #
        #
        # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        # val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
        #
        # target_model = ISTGNN(edge_index, 1, args.hidden_dim, args.hidden_dim, num_nodes=N, pre_len=args.t_out)
        # # target_model = TwoMGTCN(A_norm, args.t_in, args.t_out, args.hidden_dim, 1)
        # target_model.load_state_dict(trainer.model.state_dict())
        # target_model.to(args.device)
        #
        # target_model.return_embedding = True
        # trainer.model.return_embedding = True
        # trainer.model.eval()
        #
        #
        # new_optimizer = torch.optim.Adam(params=target_model.parameters(), lr=args.lr, eps=1.0e-8,
        #                                  weight_decay=args.wd, amsgrad=False)
        #
        #
        # def GS_MMD_loss(embedding_s, embedding_t):
        #     def kf(a, b):
        #         mu = 0.1
        #         return torch.exp(-mu * torch.sum((a - b) ** 2, dim=1))
        #
        #     # (batch_size, num_nodes, seq_length, embedding_dim)
        #     es = embedding_s.reshape(-1, embedding_s.shape[-1]) # (b, h)
        #     et = embedding_t.reshape(-1, embedding_t.shape[-1]) # (b, h)
        #
        #     assert es.shape[0] == et.shape[0]
        #     b = es.shape[0]
        #
        #     diff_ss = 0
        #     diff_st = 0
        #     diff_tt = 0
        #
        #     for i in range(b):
        #         for j in range(b):
        #             if i != j:
        #                 diff_ss += kf(es[i], es[j])
        #                 diff_st += kf(es[i], et[j])
        #                 diff_tt += kf(et[j], et[j])
        #
        #     diff_ss /= b * (b - 1)
        #     diff_st /= b * (b - 1) // 2
        #     diff_tt /= b * (b - 1)
        #
        #     # ss = torch.outer(diff_ss, diff_ss)
        #     # mask = ~np.eye(b, dtype=bool)
        #     #
        #     # # 应用掩码并调整形状为b x (b-1)
        #     # result = sim_matrix[mask].reshape(b, b - 1)
        #     return diff_ss + diff_st + diff_tt
        #
        # beta = 0.8
        #
        # start_time = time.time()
        # target_model.train()
        # for ep in range(args.target_epochs):
        #     epoch_loss = []
        #     for i, (data, y) in enumerate(train_loader):
        #         ypred, et = target_model(data)
        #         _, es = trainer.model(data)
        #
        #         loss = beta * ((ypred - y) ** 2).mean() + (1 - beta) * GS_MMD_loss(es, et)
        #         new_optimizer.zero_grad()
        #         loss.backward()
        #         new_optimizer.step()
        #         epoch_loss.append(loss.item() * y.shape[0])
        #         if i % 50 == 0:
        #             print("Train [%.2fs] ep %d it %d, loss %.4f" % (time.time() - start_time, ep, i, loss.item()))
        #     target_train_loss = sum(epoch_loss) / len(train_set)
        #     print("Train [%.2fs] ep %d, loss %.4f" % (time.time() - start_time, ep, target_train_loss))
        #
        # target_train_time = time.time() - start_time
        #
        # epoch_loss = []
        # target_model.eval()
        #
        #
        # metrics_d = {'mae': 0,
        #              'rmse': 0,
        #              'mse': 0,
        #              'mape': 0}
        #
        # for i, (data, y) in enumerate(val_loader):
        #     ypred, _ = target_model(data)
        #     loss = ((ypred - y) ** 2).mean() #B*N*3
        #     epoch_loss.append(loss.item())
        #     if i % 50 == 0:
        #         print("Test [%.2fs] it %d, loss %.4f" % (time.time() - start_time, i, loss.item()))
        #
        #     y_cpu = y.cpu().detach().numpy()
        #     y_pred_cpu = ypred.cpu().detach().numpy()
        #     metrics_d['mae'] += np.sum(np.abs(y_cpu - y_pred_cpu)).item() / args.t_out / N
        #     metrics_d['rmse'] += np.sum((y_cpu - y_pred_cpu) ** 2).item() / args.t_out / N
        #     metrics_d['mse'] += np.sum((y_cpu - y_pred_cpu) ** 2).item() / args.t_out / N
        #     metrics_d['mape'] += np.sum(np.abs((y_cpu - y_pred_cpu) / y_cpu) * 100).item() / args.t_out / N
        #
        #
        # target_eval_loss = np.mean(np.array(epoch_loss)).item()
        #
        # metrics_d['mae'] = metrics_d['mae'] / len(val_set) * scaler.metrics_coef
        # metrics_d['rmse'] = math.sqrt(metrics_d['rmse'] / len(val_set)) * scaler.metrics_coef
        # metrics_d['mse'] = metrics_d['mse'] / len(val_set) * scaler.metrics_coef * scaler.metrics_coef
        # metrics_d['mape'] = metrics_d['mape'] / len(val_set)
        #
        # print("Test [%.2fs], loss %.4f" % (time.time() - start_time, target_eval_loss))
        #
        # current_dir = os.path.dirname(os.path.realpath(__file__))
        # log_dir = os.path.join(current_dir, 'record_new')
        # os.makedirs(log_dir, exist_ok=True)
        # data = dict(metrics_d.items())
        # data["target_train_time"] = target_train_time
        # data["source_train_time"] = source_train_time
        #
        # data["batch_size"] = args.batch_size
        # data["target_train_size"] = args.target_train_ratio
        # data["t_in"] = args.t_in
        # data["t_out"] = args.t_out
        # data["epochs"] = args.epochs
        #
        # def print_model_parameters(model):
        #     # if not only_num:
        #     for name, param in model.named_parameters():
        #         print('{} {} {}'.format(name, param.shape, param.requires_grad))
        #     total_num = sum([param.nelement() for param in model.parameters()])
        #     return total_num
        #
        # total_num = print_model_parameters(model)
        # data["target_model_size"] = total_num
        #
        # city_str = "_".join(args.data_list)
        # log_path = os.path.join(log_dir, f"{city_str}_{args.t_in}_{args.t_out}_{args.target_train_ratio}_{args.feature_type}_{args.lr}.json")
        # with open(log_path, 'w', encoding='utf-8') as f:
        #     json.dump(data, f, ensure_ascii=False, indent=4)

        # learning rate decay
        new_lr_scheduler = None
        # if args.lr_decay:
        #     lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
        #     new_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
        #                                                         milestones=lr_decay_steps,
        #                                                         gamma=args.lr_decay_rate)

        # train_args = TrainingArguments(
        #     num_train_epochs=args.epochs * args.local_epochs,
        #     per_device_train_batch_size=args.batch_size,
        #     per_device_eval_batch_size=args.batch_size,
        #     logging_steps=args.epochs * args.local_epochs + 1
        # )

        # new_train_args = TrainingArguments(
        #     num_train_epochs=args.target_epochs,
        #     per_device_train_batch_size=args.batch_size,
        #     eval_strategy='epoch'
        # )

        # new_optimizer = torch.optim.Adam(params=trainer.model.parameters(), lr=args.lr, eps=1.0e-8,
        #                                  weight_decay=args.wd, amsgrad=False)
        #
        # new_trainer = Trainer(model=trainer.model,
        #                       train_dataset=train_data,
        #                       eval_dataset=val_data,
        #                       optimizers=(new_optimizer, new_lr_scheduler),
        #                       compute_loss_func=loss_func,
        #                       args=new_train_args,
        #                       compute_metrics=metrics)
        #
        # # TODO: time stat
        # # target_train_start_time = time.time()
        # # new_trainer.train()
        # # target_train_time = time.time() - target_train_start_time
        #
        # new_trainer.train()
        #
        # new_train_loss_record = [d["loss"] * scaler.metrics_coef for d in new_trainer.state.log_history[:-1:2]]
        # new_eval_mae_record = [d["eval_mae"] * scaler.metrics_coef for d in new_trainer.state.log_history[1:-1:2]]
        # new_eval_rmse_record = [d["eval_rmse"] * scaler.metrics_coef for d in new_trainer.state.log_history[1:-1:2]]
        # new_eval_mse_record = [d["eval_mse"] * scaler.metrics_coef for d in new_trainer.state.log_history[1:-1:2]]
        # new_eval_mape_record = [d["eval_mape"] * scaler.metrics_coef for d in new_trainer.state.log_history[1:-1:2]]
        #
        # with open('./new_eval_mae.txt', 'w') as f:
        #     for num in new_eval_mae_record:
        #         f.write(f"{num:.4f}\n")
        #
        # with open('./new_eval_mse.txt', 'w') as f:
        #     for num in new_eval_mse_record:
        #         f.write(f"{num:.4f}\n")
        #
        # with open('./new_eval_rmse.txt', 'w') as f:
        #     for num in new_eval_rmse_record:
        #         f.write(f"{num:.4f}\n")
        #
        # with open('./new_eval_mape.txt', 'w') as f:
        #     for num in new_eval_mape_record:
        #         f.write(f"{num:.4f}\n")
        #
        plt.plot(list(range(1, len(train_loss_record) + 1)), train_loss_record, label="train_loss")
        plt.plot(list(range(1, len(eval_mae_record) + 1)), eval_mae_record, label="eval_mae")
        # plt.plot(list(range(1, len(eval_rmse_record) + 1)), eval_rmse_record, label="eval_rmse")
        plt.legend()
        # plt.show()
        print("Guest 节点：所有数据与日志已安全保存，准备清理后台 Arbiter 释放进程...")
        time.sleep(2) # 停顿2秒，确保其他 Host 小弟的 CSV 也彻底写入硬盘
        import signal
        os.kill(os.getppid(), signal.SIGTERM) # 强制向父进程(Launcher)发送终止信号

    return trainer

def train_fedgtp(ctx):
    import traceback

    print(f"Rank {ctx.rank}: [FedGTP] Starting Patched Federated Training (Debug + Safe EarlyStop + Safe Exit).", flush=True)

    VAL_EVERY = 5

    # ===== 调试开关 =====
    COMM_DEBUG = True          # True: 打印通信日志
    COMM_DEBUG_FULL = False    # False: 只打印每个 batch 的首尾通信；True: 每个 comm_step 都打印
    SAFE_EXIT = True           # True: 不再 kill 父进程
    NUM_WORKERS = 0            # 调试期先别开多进程 DataLoader，避免额外不确定性

    # ===== 训练/验证/测试轮次统计 =====
    actual_global_epochs = 0
    actual_val_rounds = 0
    should_stop_training = False

    def _log(msg):
        print(msg, flush=True)

    if not ctx.is_on_arbiter:
        train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = get_setting(ctx)

        # 可选：把调试标志注入模型，便于模型内部打印
        if hasattr(model, "set_debug"):
            model.set_debug(enabled=COMM_DEBUG, full=COMM_DEBUG_FULL, flush=True)

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=NUM_WORKERS
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=NUM_WORKERS
        )
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=NUM_WORKERS
        ) 

        ctx.arbiter.put("init_steps", len(train_loader))
        ctx.arbiter.put("val_steps", len(val_loader))
        ctx.arbiter.put("test_steps_local", len(test_loader))

        STEPS_PER_EPOCH = len(train_loader)
        VAL_STEPS_PER_EPOCH = len(val_loader)
        TEST_STEPS = len(test_loader)

        LOCAL_NUM_NODES = len(args.nodes_per[ctx.rank])

        best_norm_mae = float("inf")
        best_epoch = -1
        best_model_wts = None

        total_train_time = 0.0
        total_val_time = 0.0

        # 动态通信量统计：记录一次 EH hook 的真实字节数
        eh_comm_bytes_per_step = 0

    else:
        s_guest = ctx.guest.get("init_steps")
        s_hosts = ctx.hosts.get("init_steps")
        if not isinstance(s_hosts, list):
            s_hosts = [s_hosts]
        STEPS_PER_EPOCH = min([s_guest] + s_hosts)

        v_guest = ctx.guest.get("val_steps")
        v_hosts = ctx.hosts.get("val_steps")
        if not isinstance(v_hosts, list):
            v_hosts = [v_hosts]
        VAL_STEPS_PER_EPOCH = min([v_guest] + v_hosts)

        t_guest = ctx.guest.get("test_steps_local")
        t_hosts = ctx.hosts.get("test_steps_local")
        if not isinstance(t_hosts, list):
            t_hosts = [t_hosts]
        TEST_STEPS = min([t_guest] + t_hosts)

        # 让 arbiter 也参与早停同步，避免 client 早停后 arbiter 继续等训练通信
        global_es = GlobalEarlyStopping(patience=10, verbose=True, delta=1e-4)

        _log(f"[FedGTP Server] Synced! Train batches={STEPS_PER_EPOCH}, Val batches={VAL_STEPS_PER_EPOCH}, Test batches={TEST_STEPS}")

    # 当前模型是 1 layer，因此这里仍然是 1 * t_in * 2
    NUM_COMM_PER_BATCH = 1 * args.t_in * 2
    # EH has one tensor per client for every communication step.  Use the
    # number of local road nodes as the federation weight, rather than giving
    # a 62-node partition the same contribution as a 92-node partition.
    # ``args.nodes_per`` is initialized for the arbiter before this trainer
    # starts, so the order matches [guest, host1, host2, host3].
    fedgtp_client_node_counts = [len(nodes) for nodes in args.nodes_per]
    fedgtp_total_nodes = sum(fedgtp_client_node_counts)
    if fedgtp_total_nodes <= 0:
        raise ValueError("FedGTP requires non-empty client node partitions.")
    if ctx.is_on_arbiter:
        _log(
            f"[FedGTP Server] EH aggregation=node_count_weighted_mean "
            f"client_nodes={fedgtp_client_node_counts}",
        )

    def _parse_comm_idx(tag):
        try:
            return int(str(tag).rsplit("_c", 1)[1])
        except Exception:
            return -1

    def _should_log_tag(tag):
        if COMM_DEBUG_FULL:
            return True
        idx = _parse_comm_idx(tag)
        return idx in (0, NUM_COMM_PER_BATCH - 1)

    def _make_client_comm_hook(phase_name):
        def client_comm_hook(EH_list, tag):
            nonlocal eh_comm_bytes_per_step

            if eh_comm_bytes_per_step == 0:
                eh_comm_bytes_per_step = sum(t.numel() * t.element_size() for t in EH_list)

            if COMM_DEBUG and _should_log_tag(tag):
                _log(f"[FedGTP Client {ctx.rank}] [{phase_name}] -> put EH_{tag}")

            ctx.arbiter.put(f"EH_{tag}", {"val": EH_list})

            if COMM_DEBUG and _should_log_tag(tag):
                _log(f"[FedGTP Client {ctx.rank}] [{phase_name}] -> wait sumEH_{tag}")

            resp = ctx.arbiter.get(f"sumEH_{tag}")

            if COMM_DEBUG and _should_log_tag(tag):
                _log(f"[FedGTP Client {ctx.rank}] [{phase_name}] <- got sumEH_{tag}")

            if isinstance(resp, list):
                resp = resp[0]
            return resp["val"]
        return client_comm_hook

    def _server_sum_eh_for_tag(tag, phase_name):
        if COMM_DEBUG and _should_log_tag(tag):
            _log(f"[FedGTP Server] [{phase_name}] -> wait EH_{tag}")

        guest_resp = ctx.guest.get(f"EH_{tag}")
        eh_guest = guest_resp[0]["val"] if isinstance(guest_resp, list) else guest_resp["val"]

        hosts_resp = ctx.hosts.get(f"EH_{tag}")
        if not isinstance(hosts_resp, list):
            hosts_resp = [hosts_resp]
        eh_hosts = [h["val"] for h in hosts_resp]

        all_eh = [eh_guest] + eh_hosts

        weighted_mean_eh = []
        for k in range(args.poly_k + 1):
            weighted_value = sum(
                client_eh[k].float() * (node_count / fedgtp_total_nodes)
                for client_eh, node_count in zip(all_eh, fedgtp_client_node_counts)
            )
            weighted_mean_eh.append(weighted_value.to(torch.float16).contiguous())

        payload = {"val": weighted_mean_eh}

        if COMM_DEBUG and _should_log_tag(tag):
            _log(f"[FedGTP Server] [{phase_name}] <- broadcast sumEH_{tag}")

        ctx.guest.put(f"sumEH_{tag}", payload)
        ctx.hosts.put(f"sumEH_{tag}", [payload] * len(eh_hosts))

    try:
        # =========================
        # 1. Global rounds
        # =========================
        for global_epoch in range(args.epochs):
            actual_global_epochs = global_epoch + 1
            do_val = ((global_epoch + 1) % VAL_EVERY == 0) or (global_epoch == args.epochs - 1)

            # -----------------------------------------
            # 1.1 每轮开始：客户端加载上一轮全局共享参数
            # -----------------------------------------
            if not ctx.is_on_arbiter and global_epoch > 0:
                _log(f"Rank {ctx.rank}: [FedGTP] waiting global_weights_{global_epoch-1}")
                global_weights_data = ctx.arbiter.get(f"global_weights_{global_epoch-1}")
                if isinstance(global_weights_data, list):
                    global_weights_data = global_weights_data[0]

                global_weights_data = {
                    k: (v.to(args.device) if torch.is_tensor(v) else v)
                    for k, v in global_weights_data.items()
                }

                if hasattr(model, "load_shared_params"):
                    model.load_shared_params(global_weights_data)
                else:
                    model.load_state_dict(global_weights_data, strict=False)

                _log(f"Rank {ctx.rank}: [FedGTP] loaded global_weights_{global_epoch-1}")

            # -----------------------------------------
            # 1.2 Local epochs
            # -----------------------------------------
            for local_ep in range(args.local_epochs):
                if not ctx.is_on_arbiter:
                    model.train()
                    epoch_train_loss = 0.0
                    train_start_time = time.time()

                    train_comm_hook = _make_client_comm_hook("TRAIN")

                    # For the explicitly labelled HE-SA Arbiter-insider upper
                    # bound, retain the complete first-batch EH upload and
                    # the public round-start state.  Normal HE-SA traces
                    # remain aggregate-only and never enter this branch.
                    capture_eh_upper = (
                        is_he
                        and bool(getattr(args, "privacy_trace_save_insider", False))
                        and ctx.rank == int(getattr(args, "privacy_trace_client", 0))
                        and int(getattr(args, "_privacy_he_sa_insider_records", 0)) < 1
                    )
                    if capture_eh_upper:
                        eh_round_start_state = {
                            key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()
                        }

                    for i, (x, y) in enumerate(train_loader):
                        if i >= STEPS_PER_EPOCH:
                            break

                        optimizer.zero_grad()
                        x = x.to(args.device, non_blocking=True)
                        y = y.to(args.device, non_blocking=True)

                        batch_tag = f"ge{global_epoch}_le{local_ep}_train_{i}"
                        model.privacy_capture_eh = bool(capture_eh_upper and i == 0)
                        if model.privacy_capture_eh:
                            model.privacy_eh_records = []
                        pred = model(x, comm_hook=train_comm_hook, batch_tag=batch_tag)
                        if model.privacy_capture_eh:
                            capture_he_sa_arbiter_insider(
                                ctx, args, f"fedgtp_eh_{batch_tag}",
                                # Match the actual transport: each EH tensor
                                # is serialized as CPU FP16 before encryption.
                                payload=tuple(
                                    tuple(item.detach().to(torch.float16).cpu().contiguous() for item in record)
                                    for record in model.privacy_eh_records
                                ),
                                model_state_dict=eh_round_start_state,
                                leak_type="activation",
                            )
                            model.privacy_capture_eh = False
                            model.privacy_eh_records = None
                            capture_eh_upper = False

                        if y.dim() == 4 and y.shape[-1] == 1:
                            y = y.squeeze(-1)
                        if pred.dim() == 4 and pred.shape[-1] == 1:
                            pred = pred.squeeze(-1)
                        if pred.shape != y.shape:
                            if pred.dim() == 3 and y.dim() == 3 and pred.shape[1] == y.shape[2] and pred.shape[2] == y.shape[1]:
                                pred = pred.transpose(1, 2)
                            else:
                                pred = pred.reshape_as(y)

                        loss = loss_func(pred, y)
                        loss.backward()
                        optimizer.step()

                        epoch_train_loss += loss.item()

                    total_train_time += (time.time() - train_start_time)
                    _log(
                        f"Client {ctx.rank} Global Epoch {global_epoch} | Local Epoch {local_ep} "
                        f"| Train Loss (Norm) = {epoch_train_loss / max(1, STEPS_PER_EPOCH):.4f}"
                    )

                else:
                    for i in range(STEPS_PER_EPOCH):
                        batch_tag = f"ge{global_epoch}_le{local_ep}_train_{i}"
                        for c in range(NUM_COMM_PER_BATCH):
                            tag = f"{batch_tag}_c{c}"
                            _server_sum_eh_for_tag(tag, "TRAIN")

            # -----------------------------------------
            # 1.3 Global round 结束：Weighted PartialFedAvg
            # -----------------------------------------
            if not ctx.is_on_arbiter:
                shared_weights = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                    if "node_embeddings" not in k
                }
                upload_payload = {
                    "num_nodes": LOCAL_NUM_NODES,
                    "weights": shared_weights
                }

                _log(f"Rank {ctx.rank}: [FedGTP] upload weights_{global_epoch}")
                ctx.arbiter.put(f"weights_{global_epoch}", upload_payload)

            else:
                _log(f"[FedGTP Server] waiting weights_{global_epoch}")

                guest_payload = ctx.guest.get(f"weights_{global_epoch}")
                hosts_payload = ctx.hosts.get(f"weights_{global_epoch}")
                if not isinstance(hosts_payload, list):
                    hosts_payload = [hosts_payload]

                all_payloads = [guest_payload] + hosts_payload
                total_nodes = sum(p["num_nodes"] for p in all_payloads)

                first_weights = all_payloads[0]["weights"]
                global_weights = {}

                for key in first_weights.keys():
                    first_tensor = first_weights[key]
                    if torch.is_tensor(first_tensor) and torch.is_floating_point(first_tensor):
                        agg = None
                        for p in all_payloads:
                            coef = p["num_nodes"] / total_nodes
                            contrib = p["weights"][key].float() * coef
                            agg = contrib if agg is None else agg + contrib
                        global_weights[key] = agg.to(first_tensor.dtype).contiguous()
                    else:
                        global_weights[key] = first_tensor

                ctx.guest.put(f"global_weights_{global_epoch}", global_weights)
                ctx.hosts.put(f"global_weights_{global_epoch}", [global_weights] * len(hosts_payload))

                _log(f"[FedGTP Server] broadcast global_weights_{global_epoch}")

            # -----------------------------------------
            # 1.4 低频验证 + 安全早停（arbiter 参与）
            # -----------------------------------------
            if do_val:
                actual_val_rounds += 1

                if not ctx.is_on_arbiter:
                    model.eval()
                    val_mae_norm, val_rmse_norm = 0.0, 0.0
                    actual_elements = 0
                    val_start_time = time.time()

                    val_comm_hook = _make_client_comm_hook("VAL")

                    with torch.no_grad():
                        for i, (x, y) in enumerate(val_loader):
                            if i >= VAL_STEPS_PER_EPOCH:
                                break

                            x = x.to(args.device, non_blocking=True)
                            y = y.to(args.device, non_blocking=True)

                            batch_tag = f"ge{global_epoch}_val_{i}"
                            pred = model(x, comm_hook=val_comm_hook, batch_tag=batch_tag)

                            if y.dim() == 4 and y.shape[-1] == 1:
                                y = y.squeeze(-1)
                            if pred.dim() == 4 and pred.shape[-1] == 1:
                                pred = pred.squeeze(-1)
                            if pred.shape != y.shape:
                                if pred.dim() == 3 and y.dim() == 3 and pred.shape[1] == y.shape[2] and pred.shape[2] == y.shape[1]:
                                    pred = pred.transpose(1, 2)
                                else:
                                    pred = pred.reshape_as(y)

                            diff = pred - y
                            val_mae_norm += diff.abs().sum().item()
                            val_rmse_norm += (diff ** 2).sum().item()
                            actual_elements += y.numel()

                    total_val_time += (time.time() - val_start_time)

                    final_val_mae_norm = val_mae_norm / actual_elements
                    final_val_rmse_norm = math.sqrt(val_rmse_norm / actual_elements)

                    _log(
                        f"   ---> [验证结果(归一化)] Client {ctx.rank} Global Epoch {global_epoch} "
                        f"| MAE: {final_val_mae_norm:.4f} | RMSE: {final_val_rmse_norm:.4f}"
                    )

                    if final_val_mae_norm < best_norm_mae:
                        best_norm_mae = final_val_mae_norm
                        best_epoch = global_epoch
                        best_model_wts = copy.deepcopy(model.state_dict())

                    # ===== 关键修复：早停通过 arbiter 同步 =====
                    _, should_stop_training = federated_early_stopping(
                        ctx,
                        actual_val_rounds,
                        local_metric=final_val_mae_norm
                    )

                    if should_stop_training:
                        _log(f"Rank {ctx.rank}: [FedGTP] 收到 arbiter 同步的全局早停信号，准备安全结束训练。")

                else:
                    for i in range(VAL_STEPS_PER_EPOCH):
                        batch_tag = f"ge{global_epoch}_val_{i}"
                        for c in range(NUM_COMM_PER_BATCH):
                            tag = f"{batch_tag}_c{c}"
                            _server_sum_eh_for_tag(tag, "VAL")

                    # ===== 关键修复：arbiter 也参与早停 =====
                    _, should_stop_training = federated_early_stopping(
                        ctx,
                        actual_val_rounds,
                        global_es=global_es
                    )

                    if should_stop_training:
                        _log("[FedGTP Server] 触发全局早停，arbiter 将与各客户端一起安全结束训练。")

                if should_stop_training:
                    break

        # for global_epoch 结束

    except Exception as e:
        _log(f"Rank {ctx.rank}: [FedGTP] 运行异常: {e}")
        traceback.print_exc()
        raise

    # =========================
    # 2. 训练结束：Test
    # =========================
    if not ctx.is_on_arbiter:
        if best_model_wts is not None:
            model.load_state_dict(best_model_wts, strict=False)
            _log(f"Rank {ctx.rank}: 完整回滚至第 {best_epoch + 1} 个 Global Epoch 的最优模型！")

        _log(f"Rank {ctx.rank}: 🚀 启动基于 Test Set 的最终物理评估。")
        model.eval()
        synchronize_cuda_for_timing(args.device)
        test_start_t = time.perf_counter()

        test_mae_real, test_rmse_real, test_mape_real = 0.0, 0.0, 0.0
        total_test_elements, valid_mape_count = 0, 0

        test_comm_hook = _make_client_comm_hook("TEST")

        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                if i >= TEST_STEPS:
                    break

                x = x.to(args.device, non_blocking=True)
                y = y.to(args.device, non_blocking=True)
                batch_tag = f"test_{i}"

                pred = model(x, comm_hook=test_comm_hook, batch_tag=batch_tag)

                if y.dim() == 4 and y.shape[-1] == 1:
                    y = y.squeeze(-1)
                if pred.dim() == 4 and pred.shape[-1] == 1:
                    pred = pred.squeeze(-1)
                if pred.shape != y.shape:
                    if pred.dim() == 3 and y.dim() == 3 and pred.shape[1] == y.shape[2] and pred.shape[2] == y.shape[1]:
                        pred = pred.transpose(1, 2)
                    else:
                        pred = pred.reshape_as(y)

                y_real = scaler.inverse_transform(y).cpu().numpy()
                pred_real = scaler.inverse_transform(pred).cpu().numpy()

                test_mae_real += np.sum(np.abs(y_real - pred_real))
                test_rmse_real += np.sum((y_real - pred_real) ** 2)
                total_test_elements += y_real.size

                mask = y_real > 0.5
                if np.sum(mask) > 0:
                    test_mape_real += np.sum(np.abs(y_real[mask] - pred_real[mask]) / y_real[mask])
                    valid_mape_count += np.sum(mask)

        acc_mae = test_mae_real / total_test_elements
        acc_rmse = math.sqrt(test_rmse_real / total_test_elements)
        acc_mape = (test_mape_real / valid_mape_count) * 100 if valid_mape_count > 0 else 0.0
        acc_mse = acc_rmse ** 2
        eff_test_time = time.time() - test_start_t

        # =========================
        # 3. 通信量统计
        # =========================
        eh_comm_per_step = eh_comm_bytes_per_step * NUM_COMM_PER_BATCH
        eh_comm_train_per_global_epoch = eh_comm_per_step * (STEPS_PER_EPOCH * args.local_epochs)
        eh_comm_val_total = eh_comm_per_step * VAL_STEPS_PER_EPOCH * actual_val_rounds

        weights_params = sum(p.numel() for n, p in model.named_parameters() if 'node_embeddings' not in n)
        weights_comm_per_global_epoch = weights_params * 4 * 2  # 上传 + 下发，按 float32 参数计

        total_comm_bytes = (
            (eh_comm_train_per_global_epoch + weights_comm_per_global_epoch) * actual_global_epochs
            + eh_comm_val_total
            + (eh_comm_per_step * TEST_STEPS)
        )
        eff_comm = round(total_comm_bytes / (1024 * 1024), 4)

        # =========================
        # 4. FLOPs
        # =========================
        eff_flops = 0.0
        try:
            from thop import profile

            class FLOPsWrapper(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, x):
                    return self.m(
                        x,
                        comm_hook=lambda eh, tag: [t.clone() for t in eh],
                        batch_tag="flops"
                    )

            dummy_x, _ = next(iter(val_loader))
            dummy_x = dummy_x.to(args.device, non_blocking=True)

            wrapper = FLOPsWrapper(model).to(args.device)
            flops, _ = profile(wrapper, inputs=(dummy_x,), verbose=False)
            eff_flops = round(flops / 1e9, 4)
            _log(f"Rank {ctx.rank}: 成功计算 FLOPs: {eff_flops} G")

        except Exception as e:
            _log(f"Rank {ctx.rank}: FLOPs 计算失败，已回退为 0.0。原因: {e}")
            eff_flops = 0.0

        dataset_client_name = f"{args.dataset_name}_client{ctx.rank}"
        eff_train_time = round(total_train_time, 2)
        eff_val_time = round(total_val_time / max(actual_val_rounds, 1), 4)

        final_csv_string = (
            f"{args.model},{dataset_client_name},{args.feature_type},{best_epoch + 1},"
            f"{round(acc_mae, 4)},{round(acc_mse, 4)},{round(acc_rmse, 4)},{round(acc_mape, 4)},"
            f"{eff_train_time},{eff_val_time},{round(eff_test_time, 4)},{eff_comm},{actual_global_epochs},{eff_flops}"
        )

        _log("\n📊 [最终物理指标] Model,Dataset_Client,Feature,Best_Epoch,Acc_MAE,Acc_MSE,Acc_RMSE,Acc_MAPE,Eff_TrainTime(s),Eff_ValTime(s),Eff_TestTime(s),Eff_CommSize(MB),Eff_TrainRound,Eff_FLOPs(G)")
        _log(f"✅ {final_csv_string}\n")

        log_experiment_results(
            model_name=args.model,
            dataset_client=dataset_client_name,
            feature_type=args.feature_type,
            best_epoch=best_epoch + 1,
            acc_mae=round(acc_mae, 4),
            acc_mse=round(acc_mse, 4),
            acc_rmse=round(acc_rmse, 4),
            acc_mape=round(acc_mape, 4),
            eff_train_time=eff_train_time,
            eff_val_time=eff_val_time,
            eff_test_time=round(eff_test_time, 4),
            eff_comm_size_mb=eff_comm,
            eff_train_round=actual_global_epochs,
            eff_flops=eff_flops,
            dp_noise=getattr(args, 'dp_noise', 0.0)
        )

        if ctx.is_on_guest:
            if SAFE_EXIT:
                _log("Guest 节点：FedGTP 数据已安全保存，当前补丁不再强制 kill 父进程，直接自然退出。")
            else:
                _log("Guest 节点：数据已安全保存，准备清理后台 Arbiter 释放进程。")
                time.sleep(2)
                os.kill(os.getppid(), signal.SIGTERM)

    else:
        _log(f"[FedGTP Server] 进入最终测试配合阶段，同步 Test batches: {TEST_STEPS}")
        for i in range(TEST_STEPS):
            batch_tag = f"test_{i}"
            for c in range(NUM_COMM_PER_BATCH):
                tag = f"{batch_tag}_c{c}"
                _server_sum_eh_for_tag(tag, "TEST")
def train_fedgtp(ctx):
    print(f"Rank {ctx.rank}: [FedGTP] Starting Silent Federated Training.", flush=True)

    # HE-SA covers both FedGTP communication surfaces: EH polynomial terms
    # exchanged during forward passes and the shared model parameters.
    is_he = getattr(args, "protection", "plain") == "he"
    is_dp = getattr(args, "protection", "plain") == "dp"
    if is_he and str(getattr(args, "he_backend", "auto")).lower() not in ("auto", "he_sa"):
        raise ValueError("FedGTP HE requires --he_backend he_sa.")

    VAL_EVERY = 5
    actual_global_epochs = 0
    actual_val_rounds = 0
    should_stop_training = False

    if not ctx.is_on_arbiter:
        train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = get_setting(ctx)

        # 完全关闭 model 内部 comm_hook 日志
        if hasattr(model, "set_debug"):
            model.set_debug(enabled=False, full=False, flush=False)

        # 你的 dataset 现在本来就是 GPU tensor，所以 pin_memory 必须关掉
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=0
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=0
        )
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=0
        )

        ctx.arbiter.put("init_steps", len(train_loader))
        ctx.arbiter.put("val_steps", len(val_loader))
        ctx.arbiter.put("test_steps_local", len(test_loader))

        STEPS_PER_EPOCH = len(train_loader)
        VAL_STEPS_PER_EPOCH = len(val_loader)
        TEST_STEPS = len(test_loader)

        LOCAL_NUM_NODES = len(args.nodes_per[ctx.rank])

        best_norm_mae = float("inf")
        best_epoch = -1
        best_model_wts = None

        total_train_time = 0.0
        total_val_time = 0.0

        eh_comm_bytes_per_step = 0
        he_upload_bytes = 0
        he_download_bytes = 0
        # FedGTP has two distinct client-to-Arbiter privacy surfaces:
        # (1) EH polynomial terms used by the cross-client forward pass and
        # (2) the round-end shared-model delta.  Their scales differ greatly,
        # so DP calibrates a q90 clipping radius for each surface once.
        fixed_dp_clip = float(getattr(args, "dp_clip_norm", 0.0))
        fedgtp_dp_clips = (
            {key: fixed_dp_clip for key in ("eh", "weights")}
            if is_dp and fixed_dp_clip > 0 else {}
        )
        dp_global_weights = None

    else:
        s_guest = ctx.guest.get("init_steps")
        s_hosts = ctx.hosts.get("init_steps")
        if not isinstance(s_hosts, list):
            s_hosts = [s_hosts]
        STEPS_PER_EPOCH = min([s_guest] + s_hosts)

        v_guest = ctx.guest.get("val_steps")
        v_hosts = ctx.hosts.get("val_steps")
        if not isinstance(v_hosts, list):
            v_hosts = [v_hosts]
        VAL_STEPS_PER_EPOCH = min([v_guest] + v_hosts)

        t_guest = ctx.guest.get("test_steps_local")
        t_hosts = ctx.hosts.get("test_steps_local")
        if not isinstance(t_hosts, list):
            t_hosts = [t_hosts]
        TEST_STEPS = min([t_guest] + t_hosts)

        global_es = GlobalEarlyStopping(patience=10, verbose=True, delta=1e-4)

        if ctx.rank == args.num_clients:
            print(f"[FedGTP Server] Synced! Train={STEPS_PER_EPOCH}, Val={VAL_STEPS_PER_EPOCH}, Test={TEST_STEPS}", flush=True)
        fixed_dp_clip = float(getattr(args, "dp_clip_norm", 0.0))
        fedgtp_dp_clips = (
            {key: fixed_dp_clip for key in ("eh", "weights")}
            if is_dp and fixed_dp_clip > 0 else {}
        )
        dp_global_weights = None

    he_context = None
    he_arbiter_context = None
    if is_he:
        context_tag = "__fedgtp_he_sa_ckks_context"
        if ctx.is_on_arbiter:
            he_arbiter_context = ckks_generate_context(
                int(args.he_ckks_poly_modulus_degree), int(args.he_ckks_scale_bits),
            )
            public_context = ckks_export_public_context(he_arbiter_context)
            ctx.guest.put(context_tag, public_context)
            ctx.hosts.put(context_tag, public_context)
            print("[HESA] FedGTP arbiter generated CKKS context", flush=True)
        else:
            he_context = ckks_import_context(ctx.arbiter.get(context_tag))

    # 当前平台版 FedGTP 是 1 layer
    NUM_COMM_PER_BATCH = 1 * args.t_in * 2

    def _unwrap_rank_payload(value):
        """FATE returns a broadcast list to hosts; select this host's entry."""
        if not isinstance(value, list):
            return value
        if ctx.is_on_guest:
            return value[0]
        index = max(0, int(ctx.rank) - 1)
        return value[index] if index < len(value) else value[0]

    def _client_dp_upload(payload_key, tag, value):
        """Calibrate one FedGTP upload type, then add update-level DP once."""
        if not is_dp:
            return ctx.arbiter.put(tag, value)
        clip_norm = fedgtp_dp_clips.get(payload_key)
        if clip_norm is None:
            ctx.arbiter.put(
                f"fedgtp_dp_{payload_key}_norm_{tag}",
                float(l2_norm(value).item()),
            )
            calibrated = _unwrap_rank_payload(
                ctx.arbiter.get(f"fedgtp_dp_{payload_key}_clip_{tag}")
            )
            clip_norm = float(calibrated)
            fedgtp_dp_clips[payload_key] = clip_norm
            print(
                f"[DPCalibration] FedGTP rank={ctx.rank} type={payload_key} "
                f"tag={tag} clip_norm={clip_norm:.8f}",
                flush=True,
            )
        return protected_arbiter_put(ctx, args, tag, value, clip_norm=clip_norm)

    def _server_dp_calibrate(payload_key, tag):
        """Arbiter side of the one-time q90 clipping calibration."""
        if not is_dp or payload_key in fedgtp_dp_clips:
            return
        guest_norm = float(ctx.guest.get(f"fedgtp_dp_{payload_key}_norm_{tag}"))
        host_norms = ctx.hosts.get(f"fedgtp_dp_{payload_key}_norm_{tag}")
        host_norms = host_norms if isinstance(host_norms, list) else [host_norms]
        clip_norm = float(np.quantile([guest_norm] + [float(item) for item in host_norms], 0.9))
        fedgtp_dp_clips[payload_key] = clip_norm
        print(
            f"[DPCalibration] FedGTP arbiter type={payload_key} tag={tag} "
            f"clip_norm={clip_norm:.8f}",
            flush=True,
        )
        ctx.guest.put(f"fedgtp_dp_{payload_key}_clip_{tag}", clip_norm)
        host_clips = [clip_norm] * len(host_norms)
        ctx.hosts.put(
            f"fedgtp_dp_{payload_key}_clip_{tag}",
            host_clips if len(host_clips) > 1 else host_clips[0],
        )

    def _make_client_comm_hook():
        def client_comm_hook(EH_list, tag):
            nonlocal eh_comm_bytes_per_step, he_upload_bytes, he_download_bytes
            if eh_comm_bytes_per_step == 0:
                eh_comm_bytes_per_step = sum(t.numel() * t.element_size() for t in EH_list)

            if is_he:
                encrypted_eh = ckks_encrypt_tree(
                    EH_list, he_context,
                    slot_count=int(args.he_ckks_poly_modulus_degree) // 2,
                )
                he_upload_bytes += ckks_ciphertext_bytes(encrypted_eh)
                ctx.arbiter.put(f"EH_{tag}", {"val": encrypted_eh})
            else:
                _client_dp_upload("eh", f"EH_{tag}", {"val": EH_list})
            resp = ctx.arbiter.get(f"sumEH_{tag}")
            if isinstance(resp, list):
                resp = resp[0]
            if is_he:
                # This is an aggregate, not an individual client EH payload.
                he_download_bytes += sum(
                    item.numel() * item.element_size()
                    for item in resp["val"] if torch.is_tensor(item)
                )
            return resp["val"]
        return client_comm_hook

    def _server_sum_eh_for_tag(tag):
        _server_dp_calibrate("eh", f"EH_{tag}")
        guest_resp = ctx.guest.get(f"EH_{tag}")
        eh_guest = guest_resp[0]["val"] if isinstance(guest_resp, list) else guest_resp["val"]

        hosts_resp = ctx.hosts.get(f"EH_{tag}")
        if not isinstance(hosts_resp, list):
            hosts_resp = [hosts_resp]
        eh_hosts = [h["val"] for h in hosts_resp]

        all_eh = [eh_guest] + eh_hosts

        if is_he:
            # Keep the original FedGTP unweighted EH-sum rule.  The arbiter
            # decrypts only the homomorphically aggregated list.
            encrypted_sum = ckks_homomorphic_sum_tree(all_eh, he_arbiter_context)
            sum_EH = ckks_decrypt_tree(encrypted_sum, he_arbiter_context)
            capture_he_sa_aggregate(ctx, args, f"fedgtp_eh_aggregate_{tag}", sum_EH)
            sum_EH = [item.to(torch.float16).contiguous() for item in sum_EH]
        else:
            sum_EH = []
            for k in range(args.poly_k + 1):
                sum_val = sum([client_eh[k].float() for client_eh in all_eh])
                sum_EH.append(sum_val.to(torch.float16).contiguous())

        payload = {"val": sum_EH}
        ctx.guest.put(f"sumEH_{tag}", payload)
        ctx.hosts.put(f"sumEH_{tag}", [payload] * len(eh_hosts))

    # DP is applied to deltas relative to a public common initial model, never
    # to full absolute weights.  This keeps the sensitivity meaning of C
    # coherent and matches the other FedAvg-style DP trainers in this project.
    if is_dp:
        if not ctx.is_on_arbiter:
            dp_global_weights = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
                if "node_embeddings" not in key
                and torch.is_tensor(value)
                and torch.is_floating_point(value)
            }
            ctx.arbiter.put("fedgtp_dp_public_initial_state", dp_global_weights)
        else:
            dp_global_weights = ctx.guest.get("fedgtp_dp_public_initial_state")

    try:
        # =========================
        # 1. Global rounds
        # =========================
        for global_epoch in range(args.epochs):
            actual_global_epochs = global_epoch + 1
            do_val = ((global_epoch + 1) % VAL_EVERY == 0) or (global_epoch == args.epochs - 1)

            # -----------------------------------------
            # 1.1 每轮开始：客户端加载上一轮全局共享参数
            # -----------------------------------------
            if not ctx.is_on_arbiter and global_epoch > 0:
                global_weights_data = ctx.arbiter.get(f"global_weights_{global_epoch-1}")
                if isinstance(global_weights_data, list):
                    global_weights_data = global_weights_data[0]

                global_weights_data = {
                    k: (v.to(args.device) if torch.is_tensor(v) else v)
                    for k, v in global_weights_data.items()
                }

                if hasattr(model, "load_shared_params"):
                    model.load_shared_params(global_weights_data)
                else:
                    model.load_state_dict(global_weights_data, strict=False)
                if is_dp:
                    dp_global_weights = {
                        key: value.detach().cpu().clone()
                        for key, value in global_weights_data.items()
                        if torch.is_tensor(value) and torch.is_floating_point(value)
                    }

            # -----------------------------------------
            # 1.2 Local epochs
            # -----------------------------------------
            for local_ep in range(args.local_epochs):
                if not ctx.is_on_arbiter:
                    model.train()
                    epoch_train_loss = 0.0
                    train_start_time = time.time()

                    train_comm_hook = _make_client_comm_hook()

                    # Explicit upper-bound instrumentation only: retain the
                    # target client's complete first-batch EH payload before
                    # aggregation, together with the round-start state.
                    capture_eh_upper = (
                        is_he
                        and bool(getattr(args, "privacy_trace_save_insider", False))
                        and ctx.rank == int(getattr(args, "privacy_trace_client", 0))
                        and int(getattr(args, "_privacy_he_sa_insider_records", 0)) < 1
                    )
                    if capture_eh_upper:
                        eh_round_start_state = {
                            key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()
                        }

                    for i, (x, y) in enumerate(train_loader):
                        if i >= STEPS_PER_EPOCH:
                            break

                        optimizer.zero_grad()
                        x = x.to(args.device, non_blocking=True)
                        y = y.to(args.device, non_blocking=True)

                        batch_tag = f"ge{global_epoch}_le{local_ep}_train_{i}"
                        model.privacy_capture_eh = bool(capture_eh_upper and i == 0)
                        if model.privacy_capture_eh:
                            model.privacy_eh_records = []
                        pred = model(x, comm_hook=train_comm_hook, batch_tag=batch_tag)
                        if model.privacy_capture_eh:
                            capture_he_sa_arbiter_insider(
                                ctx, args, f"fedgtp_eh_{batch_tag}",
                                # Match the actual transport: each EH tensor
                                # is serialized as CPU FP16 before encryption.
                                payload=tuple(
                                    tuple(item.detach().to(torch.float16).cpu().contiguous() for item in record)
                                    for record in model.privacy_eh_records
                                ),
                                model_state_dict=eh_round_start_state,
                                leak_type="activation",
                            )
                            model.privacy_capture_eh = False
                            model.privacy_eh_records = None
                            capture_eh_upper = False

                        if y.dim() == 4 and y.shape[-1] == 1:
                            y = y.squeeze(-1)
                        if pred.dim() == 4 and pred.shape[-1] == 1:
                            pred = pred.squeeze(-1)
                        if pred.shape != y.shape:
                            if pred.dim() == 3 and y.dim() == 3 and pred.shape[1] == y.shape[2] and pred.shape[2] == y.shape[1]:
                                pred = pred.transpose(1, 2)
                            else:
                                pred = pred.reshape_as(y)

                        loss = loss_func(pred, y)
                        loss.backward()
                        optimizer.step()

                        epoch_train_loss += loss.item()

                    total_train_time += (time.time() - train_start_time)
                    print(
                        f"Client {ctx.rank} Global Epoch {global_epoch} | Local Epoch {local_ep} "
                        f"| Train Loss (Norm) = {epoch_train_loss / max(1, STEPS_PER_EPOCH):.4f}",
                        flush=True
                    )

                else:
                    for i in range(STEPS_PER_EPOCH):
                        batch_tag = f"ge{global_epoch}_le{local_ep}_train_{i}"
                        for c in range(NUM_COMM_PER_BATCH):
                            tag = f"{batch_tag}_c{c}"
                            _server_sum_eh_for_tag(tag)

            # -----------------------------------------
            # 1.3 Global round 结束：Weighted PartialFedAvg
            # -----------------------------------------
            if not ctx.is_on_arbiter:
                shared_weights = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                    if "node_embeddings" not in k
                    and (not (is_he or is_dp) or (torch.is_tensor(v) and torch.is_floating_point(v)))
                }
                if is_he:
                    encrypted_weights = ckks_encrypt_tree(
                        shared_weights, he_context,
                        slot_count=int(args.he_ckks_poly_modulus_degree) // 2,
                    )
                    he_upload_bytes += ckks_ciphertext_bytes(encrypted_weights)
                    upload_weights = encrypted_weights
                    # The one-round efficiency protocol still broadcasts this
                    # global state; count its plaintext downlink.
                    he_download_bytes += sum(
                        item.numel() * item.element_size()
                        for item in shared_weights.values() if torch.is_tensor(item)
                    )
                elif is_dp:
                    if dp_global_weights is None:
                        raise RuntimeError("FedGTP DP global reference weights are unavailable.")
                    upload_weights = {
                        key: value - dp_global_weights[key]
                        for key, value in shared_weights.items()
                    }
                else:
                    upload_weights = shared_weights
                upload_payload = {
                    "num_nodes": LOCAL_NUM_NODES,
                    "weights": upload_weights
                }
                if is_dp:
                    _client_dp_upload("weights", f"weights_{global_epoch}", upload_payload["weights"])
                    # Keep client-size metadata public and send it separately;
                    # only the tensor delta is the DP-protected upload.
                    ctx.arbiter.put(
                        f"weights_metadata_{global_epoch}",
                        {"num_nodes": LOCAL_NUM_NODES},
                    )
                else:
                    ctx.arbiter.put(f"weights_{global_epoch}", upload_payload)

            else:
                _server_dp_calibrate("weights", f"weights_{global_epoch}")
                if is_dp:
                    guest_delta = ctx.guest.get(f"weights_{global_epoch}")
                    host_deltas = ctx.hosts.get(f"weights_{global_epoch}")
                    host_deltas = host_deltas if isinstance(host_deltas, list) else [host_deltas]
                    guest_meta = ctx.guest.get(f"weights_metadata_{global_epoch}")
                    host_meta = ctx.hosts.get(f"weights_metadata_{global_epoch}")
                    host_meta = host_meta if isinstance(host_meta, list) else [host_meta]
                    all_payloads = [
                        {"num_nodes": guest_meta["num_nodes"], "weights": guest_delta}
                    ] + [
                        {"num_nodes": meta["num_nodes"], "weights": delta}
                        for meta, delta in zip(host_meta, host_deltas)
                    ]
                else:
                    guest_payload = ctx.guest.get(f"weights_{global_epoch}")
                    hosts_payload = ctx.hosts.get(f"weights_{global_epoch}")
                    if not isinstance(hosts_payload, list):
                        hosts_payload = [hosts_payload]
                    all_payloads = [guest_payload] + hosts_payload
                total_nodes = sum(p["num_nodes"] for p in all_payloads)

                first_weights = all_payloads[0]["weights"]
                if is_he:
                    coefficients = [p["num_nodes"] / total_nodes for p in all_payloads]
                    encrypted_global = ckks_homomorphic_weighted_sum_tree(
                        [p["weights"] for p in all_payloads], coefficients, he_arbiter_context,
                    )
                    global_weights = ckks_decrypt_tree(encrypted_global, he_arbiter_context)
                    capture_he_sa_aggregate(ctx, args, f"fedgtp_weights_aggregate_{global_epoch}", global_weights)
                    global_weights = {
                        key: value.contiguous() if torch.is_tensor(value) else value
                        for key, value in global_weights.items()
                    }
                else:
                    global_weights = {}
                    for key in first_weights.keys():
                        first_tensor = first_weights[key]
                        if torch.is_tensor(first_tensor) and torch.is_floating_point(first_tensor):
                            agg = None
                            for p in all_payloads:
                                coef = p["num_nodes"] / total_nodes
                                contrib = p["weights"][key].float() * coef
                                agg = contrib if agg is None else agg + contrib
                            if is_dp:
                                global_weights[key] = (dp_global_weights[key] + agg).to(first_tensor.dtype).contiguous()
                            else:
                                global_weights[key] = agg.to(first_tensor.dtype).contiguous()
                        else:
                            global_weights[key] = first_tensor

                if is_dp:
                    dp_global_weights = {
                        key: value.detach().cpu().clone()
                        for key, value in global_weights.items()
                        if torch.is_tensor(value) and torch.is_floating_point(value)
                    }
                ctx.guest.put(f"global_weights_{global_epoch}", global_weights)
                ctx.hosts.put(f"global_weights_{global_epoch}", [global_weights] * (len(all_payloads) - 1))

            # -----------------------------------------
            # 1.4 低频验证 + 安全早停
            # -----------------------------------------
            if do_val:
                actual_val_rounds += 1

                if not ctx.is_on_arbiter:
                    model.eval()
                    val_mae_norm, val_rmse_norm = 0.0, 0.0
                    actual_elements = 0
                    val_start_time = time.time()

                    val_comm_hook = _make_client_comm_hook()

                    with torch.no_grad():
                        for i, (x, y) in enumerate(val_loader):
                            if i >= VAL_STEPS_PER_EPOCH:
                                break

                            x = x.to(args.device, non_blocking=True)
                            y = y.to(args.device, non_blocking=True)

                            batch_tag = f"ge{global_epoch}_val_{i}"
                            pred = model(x, comm_hook=val_comm_hook, batch_tag=batch_tag)

                            if y.dim() == 4 and y.shape[-1] == 1:
                                y = y.squeeze(-1)
                            if pred.dim() == 4 and pred.shape[-1] == 1:
                                pred = pred.squeeze(-1)
                            if pred.shape != y.shape:
                                if pred.dim() == 3 and y.dim() == 3 and pred.shape[1] == y.shape[2] and pred.shape[2] == y.shape[1]:
                                    pred = pred.transpose(1, 2)
                                else:
                                    pred = pred.reshape_as(y)

                            diff = pred - y
                            val_mae_norm += diff.abs().sum().item()
                            val_rmse_norm += (diff ** 2).sum().item()
                            actual_elements += y.numel()

                    total_val_time += (time.time() - val_start_time)

                    final_val_mae_norm = val_mae_norm / actual_elements
                    final_val_rmse_norm = math.sqrt(val_rmse_norm / actual_elements)

                    print(
                        f"   ---> [验证结果(归一化)] Client {ctx.rank} Global Epoch {global_epoch} "
                        f"| MAE: {final_val_mae_norm:.4f} | RMSE: {final_val_rmse_norm:.4f}",
                        flush=True
                    )

                    if final_val_mae_norm < best_norm_mae:
                        best_norm_mae = final_val_mae_norm
                        best_epoch = global_epoch
                        best_model_wts = copy.deepcopy(model.state_dict())

                    _, should_stop_training = federated_early_stopping(
                        ctx,
                        actual_val_rounds,
                        local_metric=final_val_mae_norm
                    )

                    if should_stop_training:
                        print(f"Rank {ctx.rank}: [FedGTP] 收到全局早停信号。", flush=True)

                else:
                    for i in range(VAL_STEPS_PER_EPOCH):
                        batch_tag = f"ge{global_epoch}_val_{i}"
                        for c in range(NUM_COMM_PER_BATCH):
                            tag = f"{batch_tag}_c{c}"
                            _server_sum_eh_for_tag(tag)

                    _, should_stop_training = federated_early_stopping(
                        ctx,
                        actual_val_rounds,
                        global_es=global_es
                    )

                    if should_stop_training and ctx.rank == args.num_clients:
                        print("[FedGTP Server] 触发全局早停。", flush=True)

                if should_stop_training:
                    break

    except Exception as e:
        print(f"Rank {ctx.rank}: [FedGTP] 运行异常: {e}", flush=True)
        raise

    # =========================
    # 2. 训练结束：Test
    # =========================
    if not ctx.is_on_arbiter:
        if best_model_wts is not None:
            model.load_state_dict(best_model_wts, strict=False)
            print(f"Rank {ctx.rank}: 回滚至第 {best_epoch + 1} 个 Global Epoch 的最优模型。", flush=True)

        print(f"Rank {ctx.rank}: 启动最终 Test 评估。", flush=True)
        model.eval()
        test_start_t = time.time()

        test_mae_real, test_rmse_real, test_mape_real = 0.0, 0.0, 0.0
        total_test_elements, valid_mape_count = 0, 0

        test_comm_hook = _make_client_comm_hook()

        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                if i >= TEST_STEPS:
                    break

                x = x.to(args.device, non_blocking=True)
                y = y.to(args.device, non_blocking=True)
                batch_tag = f"test_{i}"

                pred = model(x, comm_hook=test_comm_hook, batch_tag=batch_tag)

                if y.dim() == 4 and y.shape[-1] == 1:
                    y = y.squeeze(-1)
                if pred.dim() == 4 and pred.shape[-1] == 1:
                    pred = pred.squeeze(-1)
                if pred.shape != y.shape:
                    if pred.dim() == 3 and y.dim() == 3 and pred.shape[1] == y.shape[2] and pred.shape[2] == y.shape[1]:
                        pred = pred.transpose(1, 2)
                    else:
                        pred = pred.reshape_as(y)

                y_real = scaler.inverse_transform(y).cpu().numpy()
                pred_real = scaler.inverse_transform(pred).cpu().numpy()

                test_mae_real += np.sum(np.abs(y_real - pred_real))
                test_rmse_real += np.sum((y_real - pred_real) ** 2)
                total_test_elements += y_real.size

                mask = y_real > 0.5
                if np.sum(mask) > 0:
                    test_mape_real += np.sum(np.abs(y_real[mask] - pred_real[mask]) / y_real[mask])
                    valid_mape_count += np.sum(mask)

        acc_mae = test_mae_real / total_test_elements
        acc_rmse = math.sqrt(test_rmse_real / total_test_elements)
        acc_mape = (test_mape_real / valid_mape_count) * 100 if valid_mape_count > 0 else 0.0
        acc_mse = acc_rmse ** 2
        eff_test_time = time.time() - test_start_t

        # =========================
        # 3. 通信量统计
        # =========================
        if is_he:
            total_comm_bytes = he_upload_bytes + he_download_bytes
        else:
            eh_comm_per_step = eh_comm_bytes_per_step * NUM_COMM_PER_BATCH
            eh_comm_train_per_global_epoch = eh_comm_per_step * (STEPS_PER_EPOCH * args.local_epochs)
            eh_comm_val_total = eh_comm_per_step * VAL_STEPS_PER_EPOCH * actual_val_rounds
            weights_params = sum(p.numel() for n, p in model.named_parameters() if 'node_embeddings' not in n)
            weights_comm_per_global_epoch = weights_params * 4 * 2
            total_comm_bytes = (
                (eh_comm_train_per_global_epoch + weights_comm_per_global_epoch) * actual_global_epochs
                + eh_comm_val_total
                + (eh_comm_per_step * TEST_STEPS)
            )
        eff_comm = round(total_comm_bytes / (1024 * 1024), 4)

        # =========================
        # 4. FLOPs
        # =========================
        eff_flops = 0.0
        try:
            from thop import profile

            class FLOPsWrapper(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, x):
                    return self.m(
                        x,
                        comm_hook=lambda eh, tag: [t.clone() for t in eh],
                        batch_tag="flops"
                    )

            dummy_x, _ = next(iter(val_loader))
            dummy_x = dummy_x.to(args.device, non_blocking=True)

            wrapper = FLOPsWrapper(model).to(args.device)
            flops, _ = profile(wrapper, inputs=(dummy_x,), verbose=False)
            eff_flops = round(flops / 1e9, 4)

        except Exception:
            eff_flops = 0.0

        dataset_client_name = f"{args.dataset_name}_client{ctx.rank}"
        eff_train_time = round(total_train_time, 2)
        eff_val_time = round(total_val_time / max(actual_val_rounds, 1), 4)

        final_csv_string = (
            f"{args.model},{dataset_client_name},{args.feature_type},{best_epoch + 1},"
            f"{round(acc_mae, 4)},{round(acc_mse, 4)},{round(acc_rmse, 4)},{round(acc_mape, 4)},"
            f"{eff_train_time},{eff_val_time},{round(eff_test_time, 4)},{eff_comm},{actual_global_epochs},{eff_flops}"
        )

        print("\n📊 [最终物理指标] Model,Dataset_Client,Feature,Best_Epoch,Acc_MAE,Acc_MSE,Acc_RMSE,Acc_MAPE,Eff_TrainTime(s),Eff_ValTime(s),Eff_TestTime(s),Eff_CommSize(MB),Eff_TrainRound,Eff_FLOPs(G)", flush=True)
        print(f"✅ {final_csv_string}\n", flush=True)

        log_experiment_results(
            model_name=args.model,
            dataset_client=dataset_client_name,
            feature_type=args.feature_type,
            best_epoch=best_epoch + 1,
            acc_mae=round(acc_mae, 4),
            acc_mse=round(acc_mse, 4),
            acc_rmse=round(acc_rmse, 4),
            acc_mape=round(acc_mape, 4),
            eff_train_time=eff_train_time,
            eff_val_time=eff_val_time,
            eff_test_time=round(eff_test_time, 4),
            eff_comm_size_mb=eff_comm,
            eff_train_round=actual_global_epochs,
            eff_flops=eff_flops,
            dp_noise=getattr(args, 'dp_noise', 0.0)
        )

        if ctx.is_on_guest:
            # Do not kill the multiprocess launcher here.  The HE
            # reconstruction shell script waits for fate_main.py to return
            # before it evaluates the exported aggregate trace.  Sending
            # SIGTERM to the parent aborts that shell and silently prevents
            # the PCC runner from ever starting.
            print("Guest 节点：FedGTP 数据已安全保存，正常退出 launcher。", flush=True)
            # This guest has completed its final test handshake.  Return
            # explicitly so the FATE worker reports completion to the
            # multiprocess launcher; otherwise trace-only FedGTP runs can
            # leave rank 0 alive after every other party has exited.
            return

    else:
        if ctx.rank == args.num_clients:
            print(f"[FedGTP Server] 进入最终测试配合阶段，Test batches={TEST_STEPS}", flush=True)

        for i in range(TEST_STEPS):
            batch_tag = f"test_{i}"
            for c in range(NUM_COMM_PER_BATCH):
                tag = f"{batch_tag}_c{c}"
                _server_sum_eh_for_tag(tag)

def train_fedmetro_fixed(ctx):
    print(f"Rank {ctx.rank}: [FedMetro] start federated training with matched eval aggregation.")

    def _as_list(value):
        return value if isinstance(value, list) else [value]

    def _mean_payloads(payloads):
        return sum(payloads) / max(len(payloads), 1)

    def _is_finite_tensor(value):
        return torch.is_tensor(value) and torch.isfinite(value).all().item()

    def _model_grads_are_finite(model):
        for param in model.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all().item():
                return False
        return True

    # FedMetro exposes a split representation, its corresponding split
    # gradient, and a per-round parameter state. DP is applied explicitly to
    # these client -> Arbiter uploads; each class calibrates its own q90 C.
    fedmetro_is_dp = str(getattr(args, "protection", "plain")).lower() == "dp"
    # FedMetro needs per-client split activations/gradients, so HE-SA would
    # change its protocol.  HE-TTP encrypts each such upload to the trusted
    # arbiter, which explicitly decrypts before retaining the original logic.
    fedmetro_is_he = str(getattr(args, "protection", "plain")).lower() == "he"
    fixed_dp_clip = float(getattr(args, "dp_clip_norm", 0.0))
    fedmetro_dp_clips = ({
        key: fixed_dp_clip for key in ("agg", "g_agg", "weights")
    } if fedmetro_is_dp and fixed_dp_clip > 0 else {})

    def _fedmetro_phase(tag):
        """Classify FedMetro's explicit protocol messages for the audit log."""
        normalized = str(tag).lower()
        if "test" in normalized:
            return "test"
        if "val" in normalized or "valid" in normalized:
            return "validation"
        if normalized.startswith("init"):
            return "initialization"
        return "train"

    def _fedmetro_plain_bytes(value):
        """Serialized size of the *single-client* plaintext protocol payload."""
        if torch.is_tensor(value):
            return int(value.numel() * value.element_size())
        if isinstance(value, dict):
            return sum(_fedmetro_plain_bytes(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(_fedmetro_plain_bytes(item) for item in value)
        return 0

    def _fedmetro_record_plain_equivalent(tag, value):
        """Record the Plain counterpart of an HE-TTP message.

        During HE it reconstructs the comparable Plain communication volume
        from identical tensors and batch counts.  During a future Plain run,
        the same counter replaces the former shape proxy.
        """
        if str(getattr(args, "protection", "plain")).lower() not in ("plain", "he"):
            return
        phase = _fedmetro_phase(tag)
        totals = dict(getattr(args, "_fedmetro_plain_protocol_bytes_by_phase", {}) or {})
        totals[phase] = int(totals.get(phase, 0)) + _fedmetro_plain_bytes(value)
        args._fedmetro_plain_protocol_bytes_by_phase = totals

    def _fedmetro_select_client_payload(value):
        """Extract this client's broadcast value before any byte accounting.

        FATE can expose a host broadcast as a list containing one item per
        host.  Counting that container first wrongly charges a client for its
        peers' downlinks.  The protocol payloads here are broadcast values,
        but selecting the rank-specific item also keeps the accounting correct
        should that change later.
        """
        while isinstance(value, (list, tuple)):
            if not value:
                raise RuntimeError("FedMetro received an empty broadcast payload")
            host_index = int(getattr(ctx, "rank", 0)) - 1
            index = host_index if 0 <= host_index < len(value) else 0
            value = value[index]
        return value

    def _fedmetro_receive_downlink(tag):
        """Receive, select, audit and then account for one client downlink."""
        value = _fedmetro_select_client_payload(ctx.arbiter.get(tag))
        _fedmetro_record_plain_equivalent(tag, value)
        return record_he_ttp_downlink(args, value, tag=tag)

    def _client_dp_upload(payload_key, tag, value):
        # The raw tensor is the same logical message Plain would transmit.
        # Record it before HE serialisation changes its byte representation.
        _fedmetro_record_plain_equivalent(tag, value)
        if fedmetro_is_he:
            # FedMetro's sample-dependent AGG representation is encrypted to
            # the trusted Arbiter.  Export a separately labelled replay trace
            # only for the explicit Arbiter-insider upper-bound experiment.
            # Evaluation AGG messages are excluded: they are not the target
            # training upload used by privacy_main.py.
            if payload_key == "agg" and str(tag).startswith("agg_"):
                from privacy.attack_trace import capture_he_ttp_insider_upper_bound
                replay_model = getattr(model, "_orig_mod", model)
                capture_he_ttp_insider_upper_bound(
                    ctx, args, f"fedmetro_agg_{tag}",
                    observed_leak=value,
                    model_state_dict=replay_model.state_dict(),
                    leak_type="activation",
                )
            return protected_arbiter_put(ctx, args, tag, value)
        if not fedmetro_is_dp:
            return ctx.arbiter.put(tag, value)
        clip_norm = fedmetro_dp_clips.get(payload_key)
        if clip_norm is None:
            ctx.arbiter.put(
                f"fedmetro_dp_{payload_key}_norm_{tag}",
                float(l2_norm(value).item()),
            )
            calibrated = ctx.arbiter.get(f"fedmetro_dp_{payload_key}_clip_{tag}")
            if isinstance(calibrated, list):
                calibrated = calibrated[ctx.rank - 1] if len(calibrated) > 1 else calibrated[0]
            clip_norm = float(calibrated)
            fedmetro_dp_clips[payload_key] = clip_norm
            print(
                f"[DPCalibration] FedMetro rank={ctx.rank} type={payload_key} "
                f"tag={tag} clip_norm={clip_norm:.8f}", flush=True,
            )
        return protected_arbiter_put(ctx, args, tag, value, clip_norm=clip_norm)

    def _server_dp_calibrate(payload_key, tag):
        if not fedmetro_is_dp or payload_key in fedmetro_dp_clips:
            return
        norm_guest = float(ctx.guest.get(f"fedmetro_dp_{payload_key}_norm_{tag}"))
        norm_hosts = _as_list(ctx.hosts.get(f"fedmetro_dp_{payload_key}_norm_{tag}"))
        clip_norm = float(np.quantile(
            [norm_guest] + [float(value) for value in norm_hosts], 0.9
        ))
        fedmetro_dp_clips[payload_key] = clip_norm
        print(
            f"[DPCalibration] FedMetro arbiter type={payload_key} tag={tag} "
            f"clip_norm={clip_norm:.8f}", flush=True,
        )
        ctx.guest.put(f"fedmetro_dp_{payload_key}_clip_{tag}", clip_norm)
        host_clips = [clip_norm] * len(norm_hosts)
        ctx.hosts.put(
            f"fedmetro_dp_{payload_key}_clip_{tag}",
            host_clips if len(host_clips) > 1 else host_clips[0],
        )

    def _serve_eval_split(split_name, num_steps, epoch=None):
        for step in range(num_steps):
            tag = f"{split_name}_e{epoch}_s{step}" if epoch is not None else f"{split_name}_s{step}"
            _server_dp_calibrate("agg", f"eval_agg_{tag}")
            agg_guest = unprotect_he_ttp_payload(args, ctx.guest.get(f"eval_agg_{tag}"))
            agg_hosts = [
                unprotect_he_ttp_payload(args, item)
                for item in _as_list(ctx.hosts.get(f"eval_agg_{tag}"))
            ]
            agg_global = _mean_payloads([agg_guest] + agg_hosts)
            ctx.guest.put(f"eval_agg_global_{tag}", agg_global)
            ctx.hosts.put(f"eval_agg_global_{tag}", [agg_global] * len(agg_hosts))

    if not ctx.is_on_arbiter:
        train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = get_setting(ctx)
        # Keep an independent, uncompiled copy solely for FLOPs measurement.
        # THOP mutates modules by registering `total_ops`/`total_params`; profiling
        # the torch.compile wrapper can expose the same module twice and causes
        # `attribute 'total_ops' already exists` on FedMetro.
        fedmetro_flops_model = copy.deepcopy(model).to(args.device).eval()
        model = torch.compile(model)

        for dataset in [train_set, val_set, test_set]:
            if hasattr(dataset, 'data') and isinstance(dataset.data, np.ndarray):
                dataset.data = torch.from_numpy(dataset.data).to(args.device)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

        ctx.arbiter.put("init_train_steps", len(train_loader))
        ctx.arbiter.put("init_val_steps", len(val_loader))
        ctx.arbiter.put("init_test_steps", len(test_loader))
        STEPS_PER_EPOCH = len(train_loader)
        VAL_STEPS = len(val_loader)
        TEST_STEPS = len(test_loader)

        best_norm_mae = float('inf')
        best_epoch = -1
        best_model_wts = None
        total_train_time = 0.0
        total_val_time = 0.0
        actual_epochs = 0
        stopper = ExplicitEarlyStopper(ctx, args, patience=getattr(args, 'patience', 50), min_delta=1e-4)
        grad_clip = float(getattr(args, 'fedmetro_grad_clip', 5.0))

        def _eval_loader_with_fed_agg(loader, split_name, max_steps, epoch=None, inverse=False):
            model.eval()
            mae_sum, sq_sum, mape_sum = 0.0, 0.0, 0.0
            element_count, mape_count = 0, 0

            with torch.no_grad():
                for step, (x_eval, y_eval) in enumerate(loader):
                    if step >= max_steps:
                        break
                    tag = f"{split_name}_e{epoch}_s{step}" if epoch is not None else f"{split_name}_s{step}"
                    x_eval, y_eval = x_eval.to(args.device), y_eval.to(args.device)
                    agg_local, feat_E, pure_E = model.forward_phase1(x_eval)
                    _client_dp_upload("agg", f"eval_agg_{tag}", agg_local.detach().cpu())

                    agg_global_data = _fedmetro_receive_downlink(
                        f"eval_agg_global_{tag}"
                    )
                    agg_global = agg_global_data.to(args.device)

                    pred_eval = model.forward_phase2(x_eval, feat_E, pure_E, agg_global)
                    if y_eval.dim() == 4 and y_eval.shape[-1] == 1:
                        y_eval = y_eval.squeeze(-1)
                    if pred_eval.dim() == 4 and pred_eval.shape[-1] == 1:
                        pred_eval = pred_eval.squeeze(-1)
                    if pred_eval.shape != y_eval.shape:
                        pred_eval = pred_eval.transpose(1, 2) if pred_eval.dim() == 3 and y_eval.dim() == 3 and pred_eval.shape[1] == y_eval.shape[2] else pred_eval.reshape_as(y_eval)

                    if inverse:
                        y_np = scaler.inverse_transform(y_eval).cpu().numpy()
                        pred_np = scaler.inverse_transform(pred_eval).cpu().numpy()
                        mask = y_np > 0.5
                        if np.sum(mask) > 0:
                            mape_sum += np.sum(np.abs(y_np[mask] - pred_np[mask]) / y_np[mask])
                            mape_count += np.sum(mask)
                    else:
                        y_np = y_eval.cpu().numpy()
                        pred_np = pred_eval.cpu().numpy()

                    mae_sum += np.sum(np.abs(y_np - pred_np))
                    sq_sum += np.sum((y_np - pred_np) ** 2)
                    element_count += y_np.size

            mse = sq_sum / max(element_count, 1)
            return {
                "mae": mae_sum / max(element_count, 1),
                "rmse": math.sqrt(mse),
                "mape": (mape_sum / mape_count) * 100 if inverse and mape_count > 0 else 0.0,
                "elements": element_count,
                "abs_error_sum": mae_sum,
                "sq_error_sum": sq_sum,
                "mape_error_sum": mape_sum,
                "mape_elements": mape_count,
            }
    else:
        s_guest = ctx.guest.get("init_train_steps")
        s_hosts = _as_list(ctx.hosts.get("init_train_steps"))
        STEPS_PER_EPOCH = min([s_guest] + s_hosts)

        v_guest = ctx.guest.get("init_val_steps")
        v_hosts = _as_list(ctx.hosts.get("init_val_steps"))
        VAL_STEPS = min([v_guest] + v_hosts)

        t_guest = ctx.guest.get("init_test_steps")
        t_hosts = _as_list(ctx.hosts.get("init_test_steps"))
        TEST_STEPS = min([t_guest] + t_hosts)

    try:
        for epoch in range(args.epochs):
            actual_epochs = epoch + 1

            if not ctx.is_on_arbiter:
                model.train()
                epoch_train_loss = 0.0
                epoch_train_start = time.time()

                for step, (x, y) in enumerate(train_loader):
                    if step >= STEPS_PER_EPOCH:
                        break
                    tag = f"e{epoch}_s{step}"
                    optimizer.zero_grad()
                    x, y = x.to(args.device), y.to(args.device)

                    AGG_i_t, F_E_t, E_t = model.forward_phase1(x)
                    _client_dp_upload("agg", f"agg_{tag}", AGG_i_t.detach().cpu())

                    agg_global_data = _fedmetro_receive_downlink(
                        f"agg_global_{tag}"
                    )
                    AGG_global = agg_global_data.to(args.device).requires_grad_()

                    pred = model.forward_phase2(x, F_E_t, E_t, AGG_global)
                    if y.dim() == 4 and y.shape[-1] == 1:
                        y = y.squeeze(-1)

                    main_loss = loss_func(pred, y)
                    reg_loss = getattr(args, 'lambda_reg', 0.001) * torch.mean(model.dyn_emb_mask.mask_generator.last_m)
                    loss = main_loss + reg_loss

                    if not torch.isfinite(loss).item():
                        print(f"Rank {ctx.rank}: FedMetro non-finite loss at epoch={epoch}, step={step}; skipping local update.", flush=True)
                        _client_dp_upload("g_agg", f"g_agg_{tag}", torch.zeros_like(AGG_global).detach().cpu())
                        _ = _fedmetro_receive_downlink(f"g_total_{tag}")
                        continue

                    loss.backward(retain_graph=True)
                    g_agg = AGG_global.grad
                    if g_agg is None or not _is_finite_tensor(g_agg):
                        print(f"Rank {ctx.rank}: FedMetro non-finite AGG grad at epoch={epoch}, step={step}; sending zero grad.", flush=True)
                        g_agg = torch.zeros_like(AGG_global)
                    _client_dp_upload("g_agg", f"g_agg_{tag}", g_agg.detach().cpu())

                    g_total_data = _fedmetro_receive_downlink(f"g_total_{tag}")
                    g_total = g_total_data.to(args.device)

                    if _is_finite_tensor(g_total):
                        torch.autograd.backward(AGG_i_t, g_total)
                        if grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                        if _model_grads_are_finite(model):
                            optimizer.step()
                        else:
                            print(f"Rank {ctx.rank}: FedMetro non-finite model grad at epoch={epoch}, step={step}; skipping optimizer step.", flush=True)
                    else:
                        print(f"Rank {ctx.rank}: FedMetro non-finite global AGG grad at epoch={epoch}, step={step}; skipping optimizer step.", flush=True)

                    epoch_train_loss += float(loss.item())

                total_train_time += (time.time() - epoch_train_start)
                print(f"Client {ctx.rank} Epoch {epoch}: Train Loss(Norm) {epoch_train_loss/max(1, STEPS_PER_EPOCH):.4f}")

            else:
                for step in range(STEPS_PER_EPOCH):
                    tag = f"e{epoch}_s{step}"
                    _server_dp_calibrate("agg", f"agg_{tag}")
                    agg_guest = unprotect_he_ttp_payload(args, ctx.guest.get(f"agg_{tag}"))
                    agg_hosts = [
                        unprotect_he_ttp_payload(args, item)
                        for item in _as_list(ctx.hosts.get(f"agg_{tag}"))
                    ]
                    AGG_global = _mean_payloads([agg_guest] + agg_hosts)
                    ctx.guest.put(f"agg_global_{tag}", AGG_global)
                    ctx.hosts.put(f"agg_global_{tag}", [AGG_global] * len(agg_hosts))

                    _server_dp_calibrate("g_agg", f"g_agg_{tag}")
                    g_guest = unprotect_he_ttp_payload(args, ctx.guest.get(f"g_agg_{tag}"))
                    g_hosts = [
                        unprotect_he_ttp_payload(args, item)
                        for item in _as_list(ctx.hosts.get(f"g_agg_{tag}"))
                    ]
                    g_total = _mean_payloads([g_guest] + g_hosts)
                    ctx.guest.put(f"g_total_{tag}", g_total)
                    ctx.hosts.put(f"g_total_{tag}", [g_total] * len(g_hosts))

            if not ctx.is_on_arbiter:
                local_weights = {k: v.cpu() for k, v in model.state_dict().items() if 'static_E' not in k and 'node_embeddings' not in k}
                _client_dp_upload("weights", f"weights_{epoch}", local_weights)

                global_weights_data = _fedmetro_receive_downlink(
                    f"global_weights_{epoch}"
                )
                model.load_state_dict(global_weights_data, strict=False)

                epoch_val_start = time.time()
                val_metrics = _eval_loader_with_fed_agg(val_loader, "val", VAL_STEPS, epoch=epoch, inverse=False)
                total_val_time += (time.time() - epoch_val_start)
                final_val_mae_norm = val_metrics["mae"]
                final_val_rmse_norm = val_metrics["rmse"]

                print(f"   ---> [FedMetro federated val] Client {ctx.rank} Epoch {epoch} | MAE: {final_val_mae_norm:.4f} | RMSE: {final_val_rmse_norm:.4f}")

                if final_val_mae_norm < best_norm_mae:
                    best_norm_mae = final_val_mae_norm
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())

                should_stop = stopper.check_and_sync(float(final_val_mae_norm))
                if ctx.is_on_guest:
                    ctx.arbiter.put(f"fedmetro_stop_{epoch}", bool(should_stop))
                if should_stop:
                    print(f"Rank {ctx.rank}: FedMetro received global early-stop at epoch {actual_epochs}.")
                    raise EarlyStopSignal("FedMetro early stop")

            else:
                _server_dp_calibrate("weights", f"weights_{epoch}")
                w_guest = unprotect_he_ttp_payload(args, ctx.guest.get(f"weights_{epoch}"))
                w_hosts = [
                    unprotect_he_ttp_payload(args, item)
                    for item in _as_list(ctx.hosts.get(f"weights_{epoch}"))
                ]
                all_weights = [w_guest] + w_hosts

                global_weights = {}
                for key in all_weights[0].keys():
                    global_weights[key] = sum([w[key] for w in all_weights]) / len(all_weights)

                ctx.guest.put(f"global_weights_{epoch}", global_weights)
                ctx.hosts.put(f"global_weights_{epoch}", [global_weights] * len(w_hosts))

                _serve_eval_split("val", VAL_STEPS, epoch=epoch)
                should_stop = bool(ctx.guest.get(f"fedmetro_stop_{epoch}"))
                if should_stop:
                    print(f"Rank {ctx.rank}: FedMetro arbiter received early-stop at epoch {epoch + 1}.")
                    break

    except EarlyStopSignal:
        if not ctx.is_on_arbiter:
            print(f"Rank {ctx.rank}: FedMetro leaves training loop for final federated test.")

    if ctx.is_on_arbiter:
        _serve_eval_split("test", TEST_STEPS)
        return

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
        print(f"Rank {ctx.rank}: FedMetro restored best epoch {best_epoch + 1}.")

    print(f"Rank {ctx.rank}: FedMetro starts final federated test.")
    test_start_t = time.time()
    test_metrics = _eval_loader_with_fed_agg(test_loader, "test", TEST_STEPS, inverse=True)
    eff_test_time = time.time() - test_start_t

    acc_mae = test_metrics["mae"]
    acc_rmse = test_metrics["rmse"]
    acc_mape = test_metrics["mape"]
    dataset_client_name = f"{args.dataset_name}_client{ctx.rank}"

    step_comm = args.batch_size * args.node_emb_dim * args.poly_k * args.hidden_dim * 2 * STEPS_PER_EPOCH
    epoch_comm = sum(p.numel() for name, p in model.named_parameters() if 'static_E' not in name) * 2
    eval_comm = args.batch_size * args.node_emb_dim * args.poly_k * args.hidden_dim * (VAL_STEPS * max(actual_epochs, 1) + TEST_STEPS)
    total_comm_params = (step_comm + epoch_comm) * max(actual_epochs, 1) + eval_comm
    eff_comm_size_mb = round((total_comm_params * 4) / (1024 * 1024), 4)

    plain_protocol_by_phase = dict(
        getattr(args, "_fedmetro_plain_protocol_bytes_by_phase", {}) or {}
    )
    if str(getattr(args, "protection", "plain")).lower() == "plain" and plain_protocol_by_phase:
        # Count all four split-learning directions with their true tensor
        # shapes (including t_in), rather than the former scalar proxy.
        eff_comm_size_mb = round(sum(plain_protocol_by_phase.values()) / (1024 * 1024), 4)

    if fedmetro_is_he:
        plain_protocol_mb = {
            phase: round(int(plain_protocol_by_phase.get(phase, 0)) / (1024 * 1024), 4)
            for phase in ("initialization", "train", "validation", "test")
        }
        plain_protocol_mb["all"] = round(sum(plain_protocol_mb.values()), 4)
        print(
            f"[FedMetroPlainProtocolAudit] rank={ctx.rank} "
            f"phase_comm_mb={plain_protocol_mb} "
            f"note=single_client_plain_equivalent_same_messages",
            flush=True,
        )

    eff_flops = 0.0
    try:
        from thop import profile
        dummy_x, _ = next(iter(val_loader))

        class FLOPsWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                agg_l, f_e, p_e = self.m.forward_phase1(x)
                return self.m.forward_phase2(x, f_e, p_e, agg_l)

        # Profile the fresh uncompiled copy, never the trained torch.compile
        # wrapper.  Clear defensive remnants in case this code is invoked again
        # within the same worker process.
        for module in fedmetro_flops_model.modules():
            module._buffers.pop("total_ops", None)
            module._buffers.pop("total_params", None)
        flops, _ = profile(
            FLOPsWrapper(fedmetro_flops_model).to(args.device),
            inputs=(dummy_x.to(args.device),),
            verbose=False,
        )
        eff_flops = round(flops / 1e9, 4)
        print(f"Rank {ctx.rank}: FedMetro FLOPs computed: {eff_flops} G", flush=True)
    except Exception as e:
        print(f"Rank {ctx.rank}: FedMetro FLOPs probe failed, fallback to 0.0. Reason: {e}")

    log_experiment_results(
        model_name="FedMetro",
        dataset_client=dataset_client_name,
        feature_type=args.feature_type,
        best_epoch=best_epoch + 1,
        acc_mae=round(acc_mae, 4),
        acc_mse=round(acc_rmse**2, 4),
        acc_rmse=round(acc_rmse, 4),
        acc_mape=round(acc_mape, 4),
        acc_elements=test_metrics["elements"],
        acc_abs_error_sum=round(test_metrics["abs_error_sum"], 4),
        acc_sq_error_sum=round(test_metrics["sq_error_sum"], 4),
        acc_mape_error_sum=round(test_metrics["mape_error_sum"], 4),
        acc_mape_elements=test_metrics["mape_elements"],
        eff_train_time=round(total_train_time, 2),
        eff_val_time=round(total_val_time / max(actual_epochs, 1), 4),
        eff_test_time=round(eff_test_time, 4),
        eff_comm_size_mb=eff_comm_size_mb,
        eff_train_round=actual_epochs,
        eff_flops=eff_flops,
        dp_noise=getattr(args, 'dp_noise', 0.0)
    )
    print(f"Rank {ctx.rank}: FedMetro final federated metrics saved after {actual_epochs} epochs.")

    if ctx.is_on_guest:
        print("Guest node saved FedMetro metrics; waiting for other clients to finish naturally.")


def train_fedmetro(ctx):

    print(f"Rank {ctx.rank}: [FedMetro] 正在初始化联邦训练 (严谨评估模式)...")
    
    if not ctx.is_on_arbiter:
        train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = get_setting(ctx)
        #图编译加速了
        model = torch.compile(model)
        
        # 数据转入 GPU
        for dataset in [train_set, val_set, test_set]:
            if hasattr(dataset, 'data') and isinstance(dataset.data, np.ndarray): 
                dataset.data = torch.from_numpy(dataset.data).to(args.device)
            
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False) 
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        
        ctx.arbiter.put("init_steps", len(train_loader))
        STEPS_PER_EPOCH = len(train_loader)
        
        # 追踪指标与计时 (这里追踪的是归一化状态下的最佳验证集 MAE)
        best_norm_mae = float('inf')
        best_epoch = -1
        best_model_wts = None
        
        total_train_time = 0.0
        total_val_time = 0.0
        actual_epochs = 0 # 记录实际跑了多少轮
        
        # 早停在这
        stopper = ExplicitEarlyStopper(ctx, args, patience=50, min_delta=1e-4) # 建议把 patience 改回 50
    else:
        # Server 步数同步
        s_guest = ctx.guest.get("init_steps")
        s_hosts = ctx.hosts.get("init_steps")
        if not isinstance(s_hosts, list): s_hosts = [s_hosts]
        STEPS_PER_EPOCH = min([s_guest] + s_hosts)

    # ==========================================
    # 核心训练循环
    # ==========================================
    try:
        for epoch in range(args.epochs):
            actual_epochs = epoch + 1
            
            # ---------------- A. 训练阶段 ----------------
            if not ctx.is_on_arbiter:
                model.train()
                epoch_train_loss = 0.0
                epoch_train_start = time.time()
                
                for i, (x, y) in enumerate(train_loader):
                    if i >= STEPS_PER_EPOCH: break
                    tag = f"e{epoch}_s{i}"
                    optimizer.zero_grad()
                    x, y = x.to(args.device), y.to(args.device)

                    # 前向传播 (解包纯净 E_t)
                    AGG_i_t, F_E_t, E_t = model.forward_phase1(x) 
                    ctx.arbiter.put(f"agg_{tag}", AGG_i_t.detach().cpu())
                    
                    agg_global_data = ctx.arbiter.get(f"agg_global_{tag}")
                    if isinstance(agg_global_data, list): agg_global_data = agg_global_data[0]
                    AGG_global = agg_global_data.to(args.device).requires_grad_()
                    
                    pred = model.forward_phase2(x, F_E_t, E_t, AGG_global)
                    if y.dim() == 4 and y.shape[-1] == 1: y = y.squeeze(-1) 
                    
                    # 损失计算与反向传播 (此处使用的是归一化数据的 Loss)
                    main_loss = loss_func(pred, y)
                    reg_loss = getattr(args, 'lambda_reg', 0.001) * torch.mean(model.dyn_emb_mask.mask_generator.last_m)
                    loss = main_loss + reg_loss
                    
                    loss.backward(retain_graph=True)
                    ctx.arbiter.put(f"g_agg_{tag}", AGG_global.grad.detach().cpu())
                    
                    g_total_data = ctx.arbiter.get(f"g_total_{tag}")
                    if isinstance(g_total_data, list): g_total_data = g_total_data[0]
                    g_total = g_total_data.to(args.device)
                    
                    torch.autograd.backward(AGG_i_t, g_total)
                    optimizer.step()
                    epoch_train_loss += loss.item()
                    
                total_train_time += (time.time() - epoch_train_start)
                # 【约束 2 满足】：打印归一化训练 Loss
                print(f"Client {ctx.rank} Epoch {epoch}: Train Loss(Norm) {epoch_train_loss/max(1, STEPS_PER_EPOCH):.4f}")

            else: # SERVER 训练逻辑
                for step in range(STEPS_PER_EPOCH):
                    tag = f"e{epoch}_s{step}"
                    agg_guest = ctx.guest.get(f"agg_{tag}")
                    agg_hosts = ctx.hosts.get(f"agg_{tag}")
                    if not isinstance(agg_hosts, list): agg_hosts = [agg_hosts]
                    AGG_global = sum([agg_guest] + agg_hosts) 
                    
                    ctx.guest.put(f"agg_global_{tag}", AGG_global)
                    ctx.hosts.put(f"agg_global_{tag}", [AGG_global] * len(agg_hosts))
                    
                    g_guest = ctx.guest.get(f"g_agg_{tag}")
                    g_hosts = ctx.hosts.get(f"g_agg_{tag}")
                    if not isinstance(g_hosts, list): g_hosts = [g_hosts]
                    
                    g_total = sum([g_guest] + g_hosts)
                    ctx.guest.put(f"g_total_{tag}", g_total)
                    ctx.hosts.put(f"g_total_{tag}", [g_total] * len(g_hosts))

            # ---------------- B. 验证与早停阶段 ----------------
            if not ctx.is_on_arbiter:
                # 权重聚合
                local_weights = {k: v.cpu() for k, v in model.state_dict().items() if 'static_E' not in k and 'node_embeddings' not in k}
                ctx.arbiter.put(f"weights_{epoch}", local_weights)
                
                global_weights_data = ctx.arbiter.get(f"global_weights_{epoch}")
                if isinstance(global_weights_data, list): global_weights_data = global_weights_data[0]
                model.load_state_dict(global_weights_data, strict=False)
                
                # 验证集评估 (纯归一化状态)
                epoch_val_start = time.time()
                val_start = time.time()
                val_metrics = _evaluate_sfl_loader(model, val_loader, scaler, args.device)
                norm_val_mae = val_metrics["normalized_mae"]
                norm_val_rmse = val_metrics["normalized_rmse"]
                total_val_time += (time.time() - val_start)
                print(f"   ---> [SFL validation] Client {ctx.rank} Epoch {epoch} | Norm MAE: {norm_val_mae:.4f} | Norm RMSE: {norm_val_rmse:.4f}")

                if norm_val_mae < best_norm_mae:
                    best_norm_mae = norm_val_mae
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())

                if stopper.check_and_sync(norm_val_mae):
                    print(f"Rank {ctx.rank}: received global early-stop signal.")
                    raise EarlyStopSignal("triggered early stopping")

                local_v = _sfl_param_state(model)
                ctx.arbiter.put(f"v_{epoch}", local_v)
                actual_epochs = epoch + 1
                continue
                val_start = time.time()
                val_metrics = _evaluate_sfl_loader(model, val_loader, scaler, args.device)
                norm_val_mae = val_metrics["normalized_mae"]
                norm_val_rmse = val_metrics["normalized_rmse"]
                total_val_time += (time.time() - val_start)
                print(f"   ---> [SFL validation] Client {ctx.rank} Epoch {epoch} | Norm MAE: {norm_val_mae:.4f} | Norm RMSE: {norm_val_rmse:.4f}")

                if norm_val_mae < best_norm_mae:
                    best_norm_mae = norm_val_mae
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())

                if stopper.check_and_sync(norm_val_mae):
                    print(f"Rank {ctx.rank}: received global early-stop signal.")
                    raise EarlyStopSignal("triggered early stopping")

                local_v = _sfl_param_state(model)
                ctx.arbiter.put(f"v_{epoch}", local_v)
                actual_epochs = epoch + 1
                continue
                model.eval()
                val_mae_norm, val_rmse_norm = 0.0, 0.0
                total_val_elements = 0
                
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        x_val, y_val = x_val.to(args.device), y_val.to(args.device)
                        agg_local, feat_E, pure_E = model.forward_phase1(x_val)
                        pred_val = model.forward_phase2(x_val, feat_E, pure_E, agg_local)
                        
                        if y_val.dim() == 4 and y_val.shape[-1] == 1: y_val = y_val.squeeze(-1)
                        if pred_val.dim() == 4 and pred_val.shape[-1] == 1: pred_val = pred_val.squeeze(-1)
                        if pred_val.shape != y_val.shape:
                            pred_val = pred_val.transpose(1, 2) if pred_val.dim() == 3 and y_val.dim() == 3 and pred_val.shape[1] == y_val.shape[2] else pred_val.reshape_as(y_val)
                        
                        # 【约束 2 满足】：直接使用归一化数据做绝对值差异计算
                        y_cpu = y_val.cpu().numpy()
                        pred_cpu = pred_val.cpu().numpy()
                        val_mae_norm += np.sum(np.abs(y_cpu - pred_cpu))
                        val_rmse_norm += np.sum((y_cpu - pred_cpu) ** 2)
                        total_val_elements += y_val.numel()
                        
                final_val_mae_norm = val_mae_norm / total_val_elements
                final_val_rmse_norm = math.sqrt(val_rmse_norm / total_val_elements)
                total_val_time += (time.time() - epoch_val_start)
                
                # 【约束 2 满足】：打印归一化验证指标
                print(f"   ---> [验证结果(归一化)] Client {ctx.rank} Epoch {epoch} | MAE: {final_val_mae_norm:.4f} | RMSE: {final_val_rmse_norm:.4f}")

                if final_val_mae_norm < best_norm_mae:
                    best_norm_mae = final_val_mae_norm
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                
                # 早停裁判 (基于归一化 MAE 判断)
                should_stop = stopper.check_and_sync(float(final_val_mae_norm))
                if should_stop:
                    print(f"Rank {ctx.rank}: 🛑 收到全局早停信号，在第 {actual_epochs} 轮跳出循环！")
                    raise EarlyStopSignal("触发早停机制")

            else: # SERVER 权重聚合
                w_guest = ctx.guest.get(f"weights_{epoch}")
                w_hosts = ctx.hosts.get(f"weights_{epoch}")
                if not isinstance(w_hosts, list): w_hosts = [w_hosts]
                all_weights = [w_guest] + w_hosts
                
                global_weights = {}
                for key in all_weights[0].keys():
                    global_weights[key] = sum([w[key] for w in all_weights]) / len(all_weights)
                    
                ctx.guest.put(f"global_weights_{epoch}", global_weights)
                ctx.hosts.put(f"global_weights_{epoch}", [global_weights] * len(w_hosts))

    except EarlyStopSignal:
        if not ctx.is_on_arbiter:
            print(f"Rank {ctx.rank}: 🎉 成功穿透黑盒！跳出训练，进入绝对安全的物理测试区...")


    if not ctx.is_on_arbiter:
        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)
            print(f"Rank {ctx.rank}: 已回滚至第 {best_epoch + 1} 轮的最优模型！")
            
        print(f"Rank {ctx.rank}: 🚀 启动基于 Test Set 的最终物理评估...")
        test_start_t = time.time()
        
        model.eval()
        test_mae_real, test_rmse_real, test_mape_real = 0.0, 0.0, 0.0
        total_test_elements, valid_mape_count = 0, 0
        
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test, y_test = x_test.to(args.device), y_test.to(args.device)
                agg_local, feat_E, pure_E = model.forward_phase1(x_test)
                pred_test = model.forward_phase2(x_test, feat_E, pure_E, agg_local)
                
                if y_test.dim() == 4 and y_test.shape[-1] == 1: y_test = y_test.squeeze(-1)
                if pred_test.dim() == 4 and pred_test.shape[-1] == 1: pred_test = pred_test.squeeze(-1)
                if pred_test.shape != y_test.shape:
                    pred_test = pred_test.transpose(1, 2) if pred_test.dim() == 3 and y_test.dim() == 3 and pred_test.shape[1] == y_test.shape[2] else pred_test.reshape_as(y_test)
                
                # 【约束 3 满足】：在 Test Set 上严格执行反归一化，恢复真实流量/速度尺度
                y_real = scaler.inverse_transform(y_test).cpu().numpy()
                pred_real = scaler.inverse_transform(pred_test).cpu().numpy()
                
                test_mae_real += np.sum(np.abs(y_real - pred_real))
                test_rmse_real += np.sum((y_real - pred_real) ** 2)
                total_test_elements += y_real.size
                
                # 过滤无客流/无车流的数据点 (物理值 < 0.5 视为 0)
                mask = y_real > 0.5
                if np.sum(mask) > 0:
                    test_mape_real += np.sum(np.abs(y_real[mask] - pred_real[mask]) / y_real[mask])
                    valid_mape_count += np.sum(mask)
                    
        # 计算真实的物理指标
        acc_mae = test_mae_real / total_test_elements
        acc_rmse = math.sqrt(test_rmse_real / total_test_elements)
        acc_mape = (test_mape_real / valid_mape_count) * 100 if valid_mape_count > 0 else 0.0
        eff_test_time = time.time() - test_start_t
        
       
        dataset_client_name = f"{args.dataset_name}_client{ctx.rank}"
       
        step_comm = args.batch_size * args.node_emb_dim * args.poly_k * args.hidden_dim * 2 * STEPS_PER_EPOCH
        
       
        epoch_comm = sum(p.numel() for name, p in model.named_parameters() if 'static_E' not in name) * 2
        
        total_comm_params = (step_comm + epoch_comm) * actual_epochs
        eff_comm_size_mb = round((total_comm_params * 4) / (1024 * 1024), 4)

        eff_flops = 0.0
        try:
            from thop import profile
            dummy_x, _ = next(iter(val_loader))
            # 临时包装器用于计算 FLOPs
            class FLOPsWrapper(torch.nn.Module):
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, x):
                    agg_l, f_e, p_e = self.m.forward_phase1(x)
                    return self.m.forward_phase2(x, f_e, p_e, agg_l)
            
            flops, _ = profile(FLOPsWrapper(model).to(args.device), inputs=(dummy_x.to(args.device),), verbose=False)
            eff_flops = round(flops / 1e9, 4) # 转换为 GFLOPs
        except Exception as e:
            print(f"Rank {ctx.rank}: FLOPs 探测失败，回退为 0.0。原因: {e}")

        # 写入最终 CSV
        log_experiment_results(
            model_name="FedMetro", 
            dataset_client=dataset_client_name, 
            feature_type=args.feature_type,
            best_epoch=best_epoch + 1, 
            acc_mae=round(final_metrics["client_mae"], 4), 
            acc_mse=round(final_metrics["client_mse"], 4), 
            acc_rmse=round(final_metrics["client_rmse"], 4), 
            acc_mape=round(final_metrics["client_mape"], 4),
            eff_train_time=round(total_train_time, 2), 
            eff_val_time=round(total_val_time / actual_epochs, 4) if actual_epochs > 0 else 0.0,
            eff_test_time=round(eff_test_time, 4),
            eff_comm_size_mb=eff_comm_size_mb, 
            eff_train_round=actual_epochs, # 绝对真实的运行轮数
            eff_flops=eff_flops,
            dp_noise=getattr(args, 'dp_noise', 0.0)
        )
        print(f"🎉 Client {ctx.rank} 物理指标落盘完成！(已记录实际 {actual_epochs} 轮的开销)")
        
        if ctx.is_on_guest:
            print("Guest 节点：数据已安全保存，准备清理后台傻等的 Arbiter，释放进程...")
            time.sleep(2) 
            os.kill(os.getppid(), signal.SIGTERM)

def train_stfam_task(ctx):
    """
    STFAM 专用训练入口函数 (完全兼容 Grid 双通道底座，修复维度切片)
    """
    from lib.stfam_loader import build_global_data, load_stfam_client_dataset, unified_dataset_to_stfam_xy
    from model.STFAM import STFAM_Client_Model, Global_1D_CNN_Autoencoder, Global_2D_CNN_Autoencoder
    from lib.load_dataset import load_dataset
    import os
    import torch
    from torch.utils.data import DataLoader
    from lib.stfam_strategy import train_stfam, robust_unpack

    # ================= 修复 1：彻底切断硬编码隐患 =================
    total_nodes_map = {
        'PeMS03': 358, 'PeMS04': 307, 'PeMSD7': 228, 'PeMS08': 170,
        'TaxiBJ': 1024, 'TaxiNYC': 75, 'BikeNYC': 128
    }
    num_global_nodes = None 
    for key, val in total_nodes_map.items():
        if key in args.dataset_name:
            num_global_nodes = val
            break
    
    if num_global_nodes is None:
        raise ValueError(f"未找到数据集 {args.dataset_name} 的节点数映射，请在 total_nodes_map 补充！")

    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path))
    
    if ctx.is_on_arbiter:
        print(f"[STFAM Server] 初始化全局特征提取器 (全局节点数: {num_global_nodes})...")
        D_tensor, P_tensor = build_global_data(args.dataset_name, num_global_nodes, project_root)
        global_2d_cnn = Global_2D_CNN_Autoencoder(1, args.stfam_embed_dim).to(args.device)
        global_1d_cnn = Global_1D_CNN_Autoencoder(1, args.stfam_embed_dim).to(args.device)
        
        print("[STFAM Server] 🚀 开始全局空间 Autoencoder 预训练...")
        optimizer_global = torch.optim.Adam(list(global_2d_cnn.parameters()) + list(global_1d_cnn.parameters()), lr=args.lr)
        mse_loss = torch.nn.MSELoss()

        D_input = D_tensor.to(args.device).unsqueeze(0)    
        P_input = P_tensor.to(args.device).view(1, 1, -1)  

        global_2d_cnn.train()
        global_1d_cnn.train()

        pretrain_epochs = max(0, int(getattr(args, 'stfam_global_pretrain_epochs', 200)))
        for ep in range(pretrain_epochs):
            optimizer_global.zero_grad()
            _, D_recon = global_2d_cnn(D_input)
            _, P_recon = global_1d_cnn(P_input)
            loss = mse_loss(D_recon, D_input) + mse_loss(P_recon, P_input)
            loss.backward()
            optimizer_global.step()
            if (ep + 1) % 10 == 0 or ep == 0 or ep + 1 == pretrain_epochs:
                print(
                    f"[STFAM Server] global pretrain {ep + 1}/{pretrain_epochs} "
                    f"loss={loss.item():.6f}",
                    flush=True,
                )

        print("[STFAM Server] ✅ 预训练完成！下发全局特征权重...")
        wts_payload = {
            'cnn2d': {k: v.cpu() for k, v in global_2d_cnn.state_dict().items()},
            'cnn1d': {k: v.cpu() for k, v in global_1d_cnn.state_dict().items()}
        }
        ctx.guest.put("global_cnn_wts", wts_payload)
        ctx.hosts.put("global_cnn_wts", [wts_payload] * (args.num_clients - 1))

        return train_stfam(ctx, None, None, None, None, None, None, None, args)

    else:
        setting = get_setting(ctx)
        train_set_raw, val_set_raw, test_set_raw = setting[0], setting[1], setting[2]
        loss_func, scaler = setting[5], setting[9]

        selected_nodes = args.nodes_per[ctx.rank]
        max_local_nodes = max([len(nodes) for nodes in args.nodes_per])

        is_grid = any(g in args.dataset_name for g in ['TaxiBJ', 'TaxiNYC', 'BikeNYC'])
        if is_grid:
            print(f"[Client {ctx.rank}] 检测到 Grid 数据集，挂载 UnifiedDataset 转接头...")
            adj_matrix = torch.tensor(train_set_raw.adj, dtype=torch.float32)
            edge_index = adj_matrix.nonzero(as_tuple=False).t().contiguous().to(args.device)
            x_tra, y_tra = unified_dataset_to_stfam_xy(train_set_raw, args.t_in, args.t_out)
            x_val, y_val = unified_dataset_to_stfam_xy(val_set_raw, args.t_in, args.t_out)
            x_test, y_test = unified_dataset_to_stfam_xy(test_set_raw, args.t_in, args.t_out)
        else:
            print(f"[Client {ctx.rank}] 检测到标准 Graph 数据集，使用常规加载...")
            _, _, edge_index, _ = load_dataset(
                dataset_name=args.dataset_name, feature_type=args.feature_type, 
                normalizer=args.normalizer, T_in=args.t_in, T_out=args.t_out, 
                train_ratio=args.train_ratio, val_ratio=args.val_ratio, 
                return_edge_index=True, device=args.device, selected_nodes=selected_nodes
            )
            x_tra, y_tra = train_set_raw.tensors[0], train_set_raw.tensors[1]
            x_val, y_val = val_set_raw.tensors[0], val_set_raw.tensors[1]
            x_test, y_test = test_set_raw.tensors[0], test_set_raw.tensors[1]

        def build_loader(x_data, y_data, is_train):
            stfam_data = load_stfam_client_dataset(x_data, y_data, edge_index, max_local_nodes, args.device)
            return DataLoader(stfam_data, batch_size=args.batch_size, shuffle=is_train)

        stfam_train_loader = build_loader(x_tra, y_tra, True)
        stfam_val_loader = build_loader(x_val, y_val, False)
        stfam_test_loader = build_loader(x_test, y_test, False)
        
        global_wts = robust_unpack(ctx.arbiter.get("global_cnn_wts"))
        local_g2d = Global_2D_CNN_Autoencoder(1, args.stfam_embed_dim).to(args.device)
        local_g1d = Global_1D_CNN_Autoencoder(1, args.stfam_embed_dim).to(args.device)
        local_g2d.load_state_dict(global_wts['cnn2d'])
        local_g1d.load_state_dict(global_wts['cnn1d'])
        local_g2d.eval()
        local_g1d.eval()
        
        full_D, full_P = build_global_data(args.dataset_name, num_global_nodes, project_root)
        local_idx = torch.tensor(selected_nodes, dtype=torch.long)
        
        # ================= 修复 2：绝对安全的张量切片 =================
        # full_D 是 [1, N, N]，我们取出 NxN，切片后扩充为 [1, 1, n_local, n_local]
        sub_D = full_D[0][local_idx][:, local_idx].unsqueeze(0).unsqueeze(0).to(args.device)
        
        # full_P 是 [N, F]，取行方向节点，铺平为 [1, 1, -1] 对齐 Conv1d
        sub_P = full_P[local_idx, :].to(args.device)
        sub_P_input = sub_P.view(1, 1, -1)
        
        with torch.no_grad():
            V_D, _ = local_g2d(sub_D)
            P_embed, _ = local_g1d(sub_P_input)
            
        args.V_D, args.P_embed = V_D, P_embed

        in_channels = getattr(args, 'input_dim', 1)
        out_channels = getattr(args, 'output_dim', 1)
        client_model = STFAM_Client_Model(
            t_in=args.t_in, 
            num_local_nodes=max_local_nodes, 
            embed_dim=args.stfam_embed_dim, 
            in_channels=in_channels,
            pred_steps=args.t_out,
            out_channels=out_channels
        ).to(args.device)
        
        optimizer = torch.optim.Adam(client_model.parameters(), lr=args.lr, weight_decay=args.wd)

        return train_stfam(ctx, client_model, optimizer, stfam_train_loader, stfam_val_loader, stfam_test_loader, loss_func, scaler, args)


def train_fedostc_task(ctx, args, get_setting):
    DEBUG_FIRST_STEP = True

    def is_first_train_step(epoch, step):
        return DEBUG_FIRST_STEP and epoch == 0 and step == 0

    def is_first_val_step(epoch, step):
        return DEBUG_FIRST_STEP and epoch == 0 and step == 0

    def is_first_test_step(step):
        return DEBUG_FIRST_STEP and step == 0

    # FedOSTC is a split-learning protocol: clients expose only their time
    # hidden state and the gradient of the server-returned spatial state.
    # DP protects both client -> Arbiter tensor types, with separate first
    # round q90 clipping calibration.
    fedostc_is_dp = str(getattr(args, "protection", "plain")).lower() == "dp"
    fedostc_is_he = str(getattr(args, "protection", "plain")).lower() == "he"
    fixed_dp_clip = float(getattr(args, "dp_clip_norm", 0.0))
    fedostc_dp_clips = ({
        key: fixed_dp_clip for key in ("h_time", "g_spatio")
    } if fedostc_is_dp and fixed_dp_clip > 0 else {})

    def _client_dp_upload(payload_key, tag, value):
        if fedostc_is_he:
            return protected_arbiter_put(ctx, args, tag, value)
        if not fedostc_is_dp:
            return ctx.arbiter.put(tag, value)
        clip_norm = fedostc_dp_clips.get(payload_key)
        if clip_norm is None:
            ctx.arbiter.put(
                f"fedostc_dp_{payload_key}_norm_{tag}",
                float(l2_norm(value).item()),
            )
            calibrated = ctx.arbiter.get(f"fedostc_dp_{payload_key}_clip_{tag}")
            if isinstance(calibrated, list):
                calibrated = calibrated[ctx.rank - 1] if len(calibrated) > 1 else calibrated[0]
            clip_norm = float(calibrated)
            fedostc_dp_clips[payload_key] = clip_norm
            print(
                f"[DPCalibration] FedOSTC rank={ctx.rank} type={payload_key} "
                f"tag={tag} clip_norm={clip_norm:.8f}", flush=True,
            )
        return protected_arbiter_put(ctx, args, tag, value, clip_norm=clip_norm)

    def _server_dp_calibrate(payload_key, tag):
        if not fedostc_is_dp or payload_key in fedostc_dp_clips:
            return
        norm_guest = float(ctx.guest.get(f"fedostc_dp_{payload_key}_norm_{tag}"))
        norm_hosts = ctx.hosts.get(f"fedostc_dp_{payload_key}_norm_{tag}")
        norm_hosts = norm_hosts if isinstance(norm_hosts, list) else [norm_hosts]
        clip_norm = float(np.quantile(
            [norm_guest] + [float(value) for value in norm_hosts], 0.9
        ))
        fedostc_dp_clips[payload_key] = clip_norm
        print(
            f"[DPCalibration] FedOSTC arbiter type={payload_key} tag={tag} "
            f"clip_norm={clip_norm:.8f}", flush=True,
        )
        ctx.guest.put(f"fedostc_dp_{payload_key}_clip_{tag}", clip_norm)
        host_clips = [clip_norm] * len(norm_hosts)
        ctx.hosts.put(
            f"fedostc_dp_{payload_key}_clip_{tag}",
            host_clips if len(host_clips) > 1 else host_clips[0],
        )

    if ctx.is_on_arbiter:
        print(f"[FedOSTC Server] 启动全局图注意力聚合中心...")

        total_nodes_map = {
            'PeMS03': 358,
            'PeMS04': 307,
            'PeMSD7': 228,
            'PeMS08': 170
        }
        N_total = 307
        for key, val in total_nodes_map.items():
            if key in args.dataset_name:
                N_total = val
                break

        global_nodes = list(range(N_total))

        # 加载全局路网
        _, global_edge_index = read_st_dataset_file(
            args.dataset_name,
            args.feature_type,
            selected_nodes=global_nodes
        )
        global_edge_index = global_edge_index.to(args.device)

        server_model = FedOSTC(
            enc_dim=args.hidden_dim,
            gat_dim=args.hidden_dim,
            pred_steps=args.t_out
        ).to(args.device)
        optimizer_server = torch.optim.Adam(server_model.gat.parameters(), lr=args.lr)

        # 全局 best
        global_best_norm_mae = float('inf')
        global_best_epoch = -1
        server_best_wts = None

        patience = 50
        min_delta = 1e-4
        patience_counter = 0

    else:
        print(f"[FedOSTC Client {ctx.rank}] 启动...")
        train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = get_setting(ctx)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        # Plain and HE must count the same split-learning messages.  Plain
        # therefore records its actual activation/gradient bytes below rather
        # than using a FedAvg model-parameter proxy.
        plain_protocol_bytes = 0

        best_epoch = -1
        best_model_wts = None
        total_train_time, total_val_time = 0.0, 0.0
        actual_epochs = 0

    # ==========================================
    # 1. 步数同步（防死锁基石）
    # ==========================================
    if not ctx.is_on_arbiter:
        ctx.arbiter.put("train_steps", len(train_loader))
        ctx.arbiter.put("val_steps", len(val_loader))
        ctx.arbiter.put("test_steps", len(test_loader))
        STEPS_TRAIN, STEPS_VAL, STEPS_TEST = len(train_loader), len(val_loader), len(test_loader)
    else:
        s_g_tr, s_h_tr = ctx.guest.get("train_steps"), ctx.hosts.get("train_steps")
        s_g_va, s_h_va = ctx.guest.get("val_steps"), ctx.hosts.get("val_steps")
        s_g_te, s_h_te = ctx.guest.get("test_steps"), ctx.hosts.get("test_steps")

        s_h_tr = s_h_tr if isinstance(s_h_tr, list) else [s_h_tr]
        s_h_va = s_h_va if isinstance(s_h_va, list) else [s_h_va]
        s_h_te = s_h_te if isinstance(s_h_te, list) else [s_h_te]

        STEPS_TRAIN = min([s_g_tr] + s_h_tr)
        STEPS_VAL = min([s_g_va] + s_h_va)
        STEPS_TEST = min([s_g_te] + s_h_te)
        print(f"[Server] 步数同步完毕: Train {STEPS_TRAIN}, Val {STEPS_VAL}, Test {STEPS_TEST}")

    # ==========================================
    # 2. 核心训练与验证循环
    # ==========================================
    try:
        for epoch in range(args.epochs):
            actual_epochs = epoch + 1
            should_stop = False

            # ---------------- A. 训练阶段 ----------------
            if not ctx.is_on_arbiter:
                model.train()
                epoch_train_loss = 0.0
                train_start = time.time()

                for i, (x, y) in enumerate(train_loader):
                    if i >= STEPS_TRAIN:
                        break

                    tag = f"tr_e{epoch}_s{i}"
                    if is_first_train_step(epoch, i):
                        print(f"[Client {ctx.rank}] >>> 进入首个训练 step: {tag}")

                    x, y = x.to(args.device), y.to(args.device)
                    optimizer.zero_grad()

                    # 1. 客户端编码并上传
                    h_time = model.forward_encoder(x)
                    if is_first_train_step(epoch, i):
                        print(f"[Client {ctx.rank}] forward_encoder 完成, h_time.shape={tuple(h_time.shape)}")

                    h_time_upload = h_time.detach().cpu().contiguous()
                    if args.protection == "plain":
                        plain_protocol_bytes += h_time_upload.numel() * h_time_upload.element_size()
                    _client_dp_upload("h_time", f"h_time_{tag}", h_time_upload)
                    if is_first_train_step(epoch, i):
                        print(f"[Client {ctx.rank}] 已发送 h_time -> arbiter")

                    # 2. 接收 Server 融合特征
                    if is_first_train_step(epoch, i):
                        print(f"[Client {ctx.rank}] 等待接收 h_spatio ...")
                    h_spatio_data = ctx.arbiter.get(f"h_spatio_{tag}")
                    if is_first_train_step(epoch, i):
                        print(f"[Client {ctx.rank}] 已接收 h_spatio")

                    h_spatio_local = (
                        h_spatio_data[ctx.rank - 1]
                        if isinstance(h_spatio_data, list)
                        else h_spatio_data
                    )
                    if args.protection == "he":
                        h_spatio_local = record_he_ttp_downlink(args, h_spatio_local, tag=f"h_spatio_{tag}")
                    elif args.protection == "plain":
                        plain_protocol_bytes += h_spatio_local.numel() * h_spatio_local.element_size()
                    h_spatio_local = h_spatio_local.to(args.device).requires_grad_()

                    # 3. 客户端解码并计算归一化 Loss
                    pred = model.forward_decoder(h_spatio_local)
                    if is_first_train_step(epoch, i):
                        print(f"[Client {ctx.rank}] forward_decoder 完成, pred.shape={tuple(pred.shape)}")

                    if y.dim() == 4 and y.shape[-1] == 1:
                        y = y.squeeze(-1)
                    if pred.dim() == 4 and pred.shape[-1] == 1:
                        pred = pred.squeeze(-1)
                    if pred.shape != y.shape:
                        pred = pred.reshape_as(y)
                    capture_revised_quantized_prediction(
                        ctx, args, f"fedostc_prediction_{tag}",
                        prediction=pred, model_state_dict=model.state_dict(),
                    )

                    loss = loss_func(pred, y)
                    loss.backward()

                    if is_first_train_step(epoch, i):
                        print(f"[Client {ctx.rank}] backward(loss) 完成, 准备上传 g_spatio")

                    # 4. 梯度回传
                    g_spatio_upload = h_spatio_local.grad.cpu().contiguous()
                    if args.protection == "plain":
                        plain_protocol_bytes += g_spatio_upload.numel() * g_spatio_upload.element_size()
                    _client_dp_upload("g_spatio", f"g_spatio_{tag}", g_spatio_upload)
                    if is_first_train_step(epoch, i):
                        print(f"[Client {ctx.rank}] 已发送 g_spatio -> arbiter, 等待 g_time ...")

                    g_time_data = ctx.arbiter.get(f"g_time_{tag}")
                    if is_first_train_step(epoch, i):
                        print(f"[Client {ctx.rank}] 已接收 g_time")

                    g_time = (
                        g_time_data[ctx.rank - 1]
                        if isinstance(g_time_data, list)
                        else g_time_data
                    )
                    if args.protection == "he":
                        g_time = record_he_ttp_downlink(args, g_time, tag=f"g_time_{tag}")
                    elif args.protection == "plain":
                        plain_protocol_bytes += g_time.numel() * g_time.element_size()
                    g_time = g_time.to(args.device)

                    torch.autograd.backward(h_time, g_time)
                    optimizer.step()

                    if is_first_train_step(epoch, i):
                        print(f"[Client {ctx.rank}] 首个训练 step 完成")

                    epoch_train_loss += loss.item()

                total_train_time += (time.time() - train_start)
                print(f"Client {ctx.rank} Epoch {epoch}: Train Loss(Norm) {epoch_train_loss / max(1, STEPS_TRAIN):.4f}")

            else:
                # Server 训练逻辑
                server_model.train()
                for step in range(STEPS_TRAIN):
                    tag = f"tr_e{epoch}_s{step}"
                    optimizer_server.zero_grad()

                    if is_first_train_step(epoch, step):
                        print(f"[Server] >>> 进入首个训练 step: {tag}")
                        print("[Server] 等待接收所有客户端 h_time ...")

                    _server_dp_calibrate("h_time", f"h_time_{tag}")
                    h_guest = unprotect_he_ttp_payload(args, ctx.guest.get(f"h_time_{tag}"))
                    h_hosts = ctx.hosts.get(f"h_time_{tag}")
                    h_hosts = h_hosts if isinstance(h_hosts, list) else [h_hosts]
                    h_hosts = [unprotect_he_ttp_payload(args, item) for item in h_hosts]
                    all_h_time = [h_guest] + h_hosts

                    if is_first_train_step(epoch, step):
                        shapes = [tuple(h.shape) for h in all_h_time]
                        print(f"[Server] 已收到全部 h_time, shapes={shapes}")

                    h_global = torch.cat([h.to(args.device) for h in all_h_time], dim=1).requires_grad_()

                    # 全局 GAT
                    h_spatio_global = server_model.forward_server_gat(h_global, global_edge_index)
                    if is_first_train_step(epoch, step):
                        print(f"[Server] forward_server_gat 完成, h_spatio_global.shape={tuple(h_spatio_global.shape)}")

                    # 切片下发
                    splits, curr = [], 0
                    for k in range(args.num_clients):
                        Ni = all_h_time[k].shape[1]
                        splits.append(h_spatio_global[:, curr:curr + Ni, :].detach().cpu())
                        curr += Ni

                    ctx.guest.put(f"h_spatio_{tag}", splits[0])
                    host_payload = splits[1:] if len(splits) > 2 else splits[1]
                    ctx.hosts.put(f"h_spatio_{tag}", host_payload)

                    if is_first_train_step(epoch, step):
                        print("[Server] 已下发全部 h_spatio, 等待 g_spatio ...")

                    # 接收梯度并反传
                    _server_dp_calibrate("g_spatio", f"g_spatio_{tag}")
                    g_guest = unprotect_he_ttp_payload(args, ctx.guest.get(f"g_spatio_{tag}"))
                    g_hosts = ctx.hosts.get(f"g_spatio_{tag}")
                    g_hosts = g_hosts if isinstance(g_hosts, list) else [g_hosts]
                    g_hosts = [unprotect_he_ttp_payload(args, item) for item in g_hosts]
                    all_g_spatio = [g_guest] + g_hosts

                    if is_first_train_step(epoch, step):
                        g_shapes = [tuple(g.shape) for g in all_g_spatio]
                        print(f"[Server] 已收到全部 g_spatio, shapes={g_shapes}")

                    g_global = torch.cat([g.to(args.device) for g in all_g_spatio], dim=1)
                    h_spatio_global.backward(g_global)

                    # 下发时间梯度
                    g_time_global = h_global.grad
                    g_splits, curr = [], 0
                    for k in range(args.num_clients):
                        Ni = all_h_time[k].shape[1]
                        g_splits.append(g_time_global[:, curr:curr + Ni, :].cpu())
                        curr += Ni

                    ctx.guest.put(f"g_time_{tag}", g_splits[0])
                    host_payload = g_splits[1:] if len(g_splits) > 2 else g_splits[1]
                    ctx.hosts.put(f"g_time_{tag}", host_payload)

                    optimizer_server.step()

                    if is_first_train_step(epoch, step):
                        print("[Server] 首个训练 step 完成")

            # ---------------- B. 验证阶段 ----------------
            if not ctx.is_on_arbiter:
                model.eval()
                val_start = time.time()
                val_mae, val_rmse, total_val_elements = 0.0, 0.0, 0

                with torch.no_grad():
                    for i, (x_val, y_val) in enumerate(val_loader):
                        if i >= STEPS_VAL:
                            break

                        tag = f"val_e{epoch}_s{i}"
                        if is_first_val_step(epoch, i):
                            print(f"[Client {ctx.rank}] >>> 进入首个验证 step: {tag}")

                        x_val, y_val = x_val.to(args.device), y_val.to(args.device)

                        h_time = model.forward_encoder(x_val)
                        if is_first_val_step(epoch, i):
                            print(f"[Client {ctx.rank}] val forward_encoder 完成, h_time.shape={tuple(h_time.shape)}")

                        h_time_upload = h_time.cpu().contiguous()
                        if args.protection == "plain":
                            plain_protocol_bytes += h_time_upload.numel() * h_time_upload.element_size()
                        _client_dp_upload("h_time", f"h_time_{tag}", h_time_upload)
                        if is_first_val_step(epoch, i):
                            print(f"[Client {ctx.rank}] val 已发送 h_time")

                        h_spatio_data = ctx.arbiter.get(f"h_spatio_{tag}")
                        if is_first_val_step(epoch, i):
                            print(f"[Client {ctx.rank}] val 已接收 h_spatio")

                        h_spatio_local = (
                            h_spatio_data[ctx.rank - 1]
                            if isinstance(h_spatio_data, list)
                            else h_spatio_data
                        )
                        if args.protection == "he":
                            h_spatio_local = record_he_ttp_downlink(args, h_spatio_local, tag=f"h_spatio_{tag}")
                        elif args.protection == "plain":
                            plain_protocol_bytes += h_spatio_local.numel() * h_spatio_local.element_size()
                        h_spatio_local = h_spatio_local.to(args.device)

                        pred_val = model.forward_decoder(h_spatio_local)

                        if y_val.dim() == 4 and y_val.shape[-1] == 1:
                            y_val = y_val.squeeze(-1)
                        if pred_val.dim() == 4 and pred_val.shape[-1] == 1:
                            pred_val = pred_val.squeeze(-1)
                        if pred_val.shape != y_val.shape:
                            pred_val = pred_val.reshape_as(y_val)

                        y_cpu = y_val.detach().cpu().numpy()
                        pred_cpu = pred_val.detach().cpu().numpy()

                        val_mae += np.sum(np.abs(y_cpu - pred_cpu))
                        val_rmse += np.sum((y_cpu - pred_cpu) ** 2)
                        total_val_elements += y_val.numel()

                norm_val_mae = val_mae / total_val_elements
                norm_val_rmse = math.sqrt(val_rmse / total_val_elements)
                total_val_time += (time.time() - val_start)

                print(
                    f"   ---> [验证结果] Client {ctx.rank} Epoch {epoch} "
                    f"| Norm MAE: {norm_val_mae:.4f} | Norm RMSE: {norm_val_rmse:.4f}"
                )

                # 上报本地验证 MAE
                ctx.arbiter.put(f"local_val_mae_{epoch}", float(norm_val_mae))

                sync_info = ctx.arbiter.get(f"val_sync_{epoch}")
                if isinstance(sync_info, list):
                    sync_info = sync_info[ctx.rank - 1]

                global_val_mae = sync_info["global_val_mae"]
                is_best = sync_info["is_best"]
                should_stop = sync_info["should_stop"]
                best_epoch = sync_info["best_epoch"]

                if is_best:
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print(
                        f"Rank {ctx.rank}: ✅ 已保存全局最优 Epoch {best_epoch + 1} 的 client 权重 "
                        f"(Global Val MAE={global_val_mae:.4f})"
                    )

            else:
                # Server 验证逻辑
                server_model.eval()
                with torch.no_grad():
                    for step in range(STEPS_VAL):
                        tag = f"val_e{epoch}_s{step}"

                        if is_first_val_step(epoch, step):
                            print(f"[Server] >>> 进入首个验证 step: {tag}")
                            print("[Server] val 等待接收所有 h_time ...")

                        _server_dp_calibrate("h_time", f"h_time_{tag}")
                        h_guest = unprotect_he_ttp_payload(args, ctx.guest.get(f"h_time_{tag}"))
                        h_hosts = ctx.hosts.get(f"h_time_{tag}")
                        h_hosts = h_hosts if isinstance(h_hosts, list) else [h_hosts]
                        h_hosts = [unprotect_he_ttp_payload(args, item) for item in h_hosts]
                        all_h_time = [h_guest] + h_hosts

                        if is_first_val_step(epoch, step):
                            shapes = [tuple(h.shape) for h in all_h_time]
                            print(f"[Server] val 已收到全部 h_time, shapes={shapes}")

                        h_global = torch.cat([h.to(args.device) for h in all_h_time], dim=1)
                        h_spatio_global = server_model.forward_server_gat(h_global, global_edge_index)

                        splits, curr = [], 0
                        for k in range(args.num_clients):
                            Ni = all_h_time[k].shape[1]
                            splits.append(h_spatio_global[:, curr:curr + Ni, :].cpu())
                            curr += Ni

                        ctx.guest.put(f"h_spatio_{tag}", splits[0])
                        host_payload = splits[1:] if len(splits) > 2 else splits[1]
                        ctx.hosts.put(f"h_spatio_{tag}", host_payload)

                        if is_first_val_step(epoch, step):
                            print("[Server] val 已下发全部 h_spatio")

                # 收集所有 client 的本地验证 MAE
                val_guest = ctx.guest.get(f"local_val_mae_{epoch}")
                val_hosts = ctx.hosts.get(f"local_val_mae_{epoch}")
                val_hosts = val_hosts if isinstance(val_hosts, list) else [val_hosts]
                all_val_mae = [val_guest] + val_hosts

                global_val_mae = float(sum(all_val_mae) / len(all_val_mae))

                if global_val_mae < global_best_norm_mae - min_delta:
                    global_best_norm_mae = global_val_mae
                    global_best_epoch = epoch
                    server_best_wts = copy.deepcopy(server_model.state_dict())
                    patience_counter = 0
                    is_best = True
                else:
                    patience_counter += 1
                    is_best = False

                should_stop = patience_counter >= patience

                model_name_for_log = getattr(args, "model_name", "FedOSTC")
                print(
                    f"--- [Server 裁判] {model_name_for_log}_{args.dataset_name}_{args.feature_type} "
                    f"| Epoch {epoch + 1} | Global Val MAE: {global_val_mae:.4f} "
                    f"| Best: {global_best_norm_mae:.4f} | Patience: {patience_counter}/{patience} ---"
                )

                sync_payload = {
                    "global_val_mae": global_val_mae,
                    "is_best": is_best,
                    "should_stop": should_stop,
                    "best_epoch": global_best_epoch,
                }

                ctx.guest.put(f"val_sync_{epoch}", sync_payload)
                host_payload = [copy.deepcopy(sync_payload) for _ in range(args.num_clients - 1)]
                ctx.hosts.put(
                    f"val_sync_{epoch}",
                    host_payload if len(host_payload) > 1 else host_payload[0]
                )

                if should_stop:
                    print(f"[Server] 收到全局早停条件，在第 {actual_epochs} 轮停止训练。")
                    break

            if not ctx.is_on_arbiter and should_stop:
                print(f"Rank {ctx.rank}: 🛑 收到全局早停信号，在第 {actual_epochs} 轮跳出！")
                raise EarlyStopSignal("触发早停机制")

    except EarlyStopSignal:
        pass

    # ==========================================
    # 3. 最终测试与落盘
    # ==========================================
    if not ctx.is_on_arbiter:
        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)
            print(f"Rank {ctx.rank}: ✅ 已回滚到全局最优 Epoch {best_epoch + 1} 的 client 权重")

        print(f"Rank {ctx.rank}: 🚀 启动最终物理尺度 Test 评估...")
        model.eval()
        synchronize_cuda_for_timing(args.device)
        test_start_t = time.perf_counter()

        test_mae, test_rmse, test_mape = 0.0, 0.0, 0.0
        test_elements, valid_mape_count = 0, 0

        with torch.no_grad():
            for i, (x_test, y_test) in enumerate(test_loader):
                if i >= STEPS_TEST:
                    break

                tag = f"test_s{i}"
                if is_first_test_step(i):
                    print(f"[Client {ctx.rank}] >>> 进入首个测试 step: {tag}")

                x_test, y_test = x_test.to(args.device), y_test.to(args.device)

                h_time = model.forward_encoder(x_test)
                if is_first_test_step(i):
                    print(f"[Client {ctx.rank}] test forward_encoder 完成, h_time.shape={tuple(h_time.shape)}")

                h_time_upload = h_time.cpu().contiguous()
                if args.protection == "plain":
                    plain_protocol_bytes += h_time_upload.numel() * h_time_upload.element_size()
                _client_dp_upload("h_time", f"h_time_{tag}", h_time_upload)
                if is_first_test_step(i):
                    print(f"[Client {ctx.rank}] test 已发送 h_time")

                h_spatio_data = ctx.arbiter.get(f"h_spatio_{tag}")
                if is_first_test_step(i):
                    print(f"[Client {ctx.rank}] test 已接收 h_spatio")

                h_spatio_local = (
                    h_spatio_data[ctx.rank - 1]
                    if isinstance(h_spatio_data, list)
                    else h_spatio_data
                )
                if args.protection == "he":
                    h_spatio_local = record_he_ttp_downlink(args, h_spatio_local, tag=f"h_spatio_{tag}")
                elif args.protection == "plain":
                    plain_protocol_bytes += h_spatio_local.numel() * h_spatio_local.element_size()
                h_spatio_local = h_spatio_local.to(args.device)

                pred_test = model.forward_decoder(h_spatio_local)

                if y_test.dim() == 4 and y_test.shape[-1] == 1:
                    y_test = y_test.squeeze(-1)
                if pred_test.dim() == 4 and pred_test.shape[-1] == 1:
                    pred_test = pred_test.squeeze(-1)

                assert pred_test.shape == y_test.shape, \
                    f"Test shape mismatch on client {ctx.rank}: pred={pred_test.shape}, y={y_test.shape}"

                y_real = scaler.inverse_transform(y_test).detach().cpu().numpy()
                pred_real = scaler.inverse_transform(pred_test).detach().cpu().numpy()

                test_mae += np.sum(np.abs(y_real - pred_real))
                test_rmse += np.sum((y_real - pred_real) ** 2)
                test_elements += y_real.size

                mask = y_real > 0.5
                if np.sum(mask) > 0:
                    test_mape += np.sum(np.abs(y_real[mask] - pred_real[mask]) / y_real[mask])
                    valid_mape_count += np.sum(mask)

        acc_mae = test_mae / test_elements
        acc_rmse = math.sqrt(test_rmse / test_elements)
        acc_mse = acc_rmse ** 2
        acc_mape = (test_mape / valid_mape_count) * 100 if valid_mape_count > 0 else 0.0
        synchronize_cuda_for_timing(args.device)
        eff_test_time = time.perf_counter() - test_start_t
        print(
            f"[TestTimingAudit] model=FedOSTC rank={ctx.rank} "
            f"batches={min(len(test_loader), STEPS_TEST)} elements={test_elements} "
            f"seconds={eff_test_time:.6f}",
            flush=True,
        )

        eff_comm_size_mb = round(plain_protocol_bytes / (1024 * 1024), 4)

        eff_flops = 0.0
        try:
            from thop import profile

            class FedOSTC_Client_Wrapper(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, x):
                    h_time = self.m.forward_encoder(x)
                    pseudo_h_spatio = h_time
                    return self.m.forward_decoder(pseudo_h_spatio)

            wrapper = FedOSTC_Client_Wrapper(model).to(args.device)
            dummy_x, _ = next(iter(val_loader))
            flops, _ = profile(wrapper, inputs=(dummy_x.to(args.device),), verbose=False)
            eff_flops = round(flops / 1e9, 4)
            print(f"Rank {ctx.rank}: Client 本地 FLOPs 计算成功: {eff_flops} G")
        except Exception as e:
            print(f"Rank {ctx.rank}: FLOPs 探测失败，设为 0.0。原因: {e}")

        dataset_client_name = f"{args.dataset_name}_client{ctx.rank}"
        log_experiment_results(
            model_name="FedOSTC",
            dataset_client=dataset_client_name,
            feature_type=args.feature_type,
            best_epoch=best_epoch + 1,
            acc_mae=round(acc_mae, 4),
            acc_mse=round(acc_mse, 4),
            acc_rmse=round(acc_rmse, 4),
            acc_mape=round(acc_mape, 4),
            eff_train_time=round(total_train_time, 2),
            eff_val_time=round(total_val_time / max(actual_epochs, 1), 4),
            eff_test_time=round(eff_test_time, 4),
            eff_comm_size_mb=eff_comm_size_mb,
            eff_train_round=actual_epochs,
            eff_flops=eff_flops,
            dp_noise=getattr(args, 'dp_noise', 0.0)
        )
        print(f"🎉 Client {ctx.rank} Test 物理指标已精准落盘！")

        if ctx.is_on_guest and not bool(getattr(args, "cross_device_multiplex", False)):
            time.sleep(2)
            os.kill(os.getppid(), signal.SIGTERM)

    else:
        if server_best_wts is not None:
            server_model.load_state_dict(server_best_wts)
            print(f"[Server] ✅ 已回滚到全局最优 Epoch {global_best_epoch + 1} 的 server 权重")

        server_model.eval()
        with torch.no_grad():
            for step in range(STEPS_TEST):
                tag = f"test_s{step}"

                if is_first_test_step(step):
                    print(f"[Server] >>> 进入首个测试 step: {tag}")
                    print("[Server] test 等待接收所有 h_time ...")

                _server_dp_calibrate("h_time", f"h_time_{tag}")
                h_guest = unprotect_he_ttp_payload(args, ctx.guest.get(f"h_time_{tag}"))
                h_hosts = ctx.hosts.get(f"h_time_{tag}")
                h_hosts = h_hosts if isinstance(h_hosts, list) else [h_hosts]
                h_hosts = [unprotect_he_ttp_payload(args, item) for item in h_hosts]
                all_h_time = [h_guest] + h_hosts

                if is_first_test_step(step):
                    shapes = [tuple(h.shape) for h in all_h_time]
                    print(f"[Server] test 已收到全部 h_time, shapes={shapes}")

                h_global = torch.cat([h.to(args.device) for h in all_h_time], dim=1)
                h_spatio_global = server_model.forward_server_gat(h_global, global_edge_index)

                splits, curr = [], 0
                for k in range(args.num_clients):
                    Ni = all_h_time[k].shape[1]
                    splits.append(h_spatio_global[:, curr:curr + Ni, :].cpu())
                    curr += Ni

                ctx.guest.put(f"h_spatio_{tag}", splits[0])
                host_payload = splits[1:] if len(splits) > 2 else splits[1]
                ctx.hosts.put(f"h_spatio_{tag}", host_payload)

                if is_first_test_step(step):
                    print("[Server] test 已下发全部 h_spatio")

def train_fedstg(ctx):
    """
    FedSTG 完整训练流程：支持图特征融合 + 验证早停 + 严谨的物理指标测试 + 防死锁收尾
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    import time
    import copy
    import io
    import os
    import math
    import numpy as np
    from lib.load_dataset import build_fedstg_static_adj
    from lib.utils import ExplicitEarlyStopper, EarlyStopSignal

    fedstg_chunk_bytes = 512 * 1024

    def _server_send_host_splits(tag, values):
        if not fedstg_chunked_host_downlink:
            ctx.hosts.put(tag, values if len(values) > 1 else values[0])
            return
        buffer = io.BytesIO()
        torch.save(values, buffer)
        payload = buffer.getvalue()
        chunks = [payload[offset: offset + fedstg_chunk_bytes]
                  for offset in range(0, len(payload), fedstg_chunk_bytes)]
        # A scalar is broadcast identically by FATE; each individual byte
        # message stays below the table-serialization threshold.
        ctx.hosts.put(f"{tag}__chunk_count", len(chunks))
        for index, chunk in enumerate(chunks):
            ctx.hosts.put(f"{tag}__chunk_{index}", chunk)

    def _client_receive_host_split(tag):
        if ctx.is_on_guest:
            return _client_he_downlink(ctx.arbiter.get(tag), tag)
        if not fedstg_chunked_host_downlink:
            payload = _client_he_downlink(ctx.arbiter.get(tag), tag)
            return safe_get_hosts_data(payload)[ctx.rank - 1]
        count = ctx.arbiter.get(f"{tag}__chunk_count")
        if isinstance(count, (list, tuple)):
            count = count[0]
        chunks = [ctx.arbiter.get(f"{tag}__chunk_{index}")
                  for index in range(int(count))]
        values = torch.load(io.BytesIO(b"".join(chunks)), map_location="cpu")
        return _client_he_downlink(values[ctx.rank - 1], tag)

    # 安全的 FATE 通信数据提取器 (防重放报错核心补丁)
    def safe_get_hosts_data(hosts_data):
        return hosts_data if isinstance(hosts_data, list) else [hosts_data]

    # FedSTG is a split-learning protocol.  Its privacy boundary comprises
    # client activations (h_tau), pattern embeddings (z_tau), the split
    # gradient (g_hG), and the per-round GRU/pattern summaries.  Each payload
    # class receives an independent first-round q90 clipping calibration.
    fedstg_is_dp = str(getattr(args, "protection", "plain")).lower() == "dp"
    # FedSTG is split learning: the Arbiter requires per-client tensors, so
    # protect its uplinks with HE-TTP rather than an incompatible HE-SA sum.
    fedstg_is_he = str(getattr(args, "protection", "plain")).lower() == "he"
    # A 32-client h_G list at batch 64 is larger than FATE's 1 MiB object
    # threshold.  Its table fallback repeatedly forks copy workers and was the
    # source of the BrokenProcessPool failure.  Keep the training batch at 64;
    # only split plain host broadcasts into sub-threshold byte messages.
    fedstg_chunked_host_downlink = (
        int(args.num_clients) >= 16 and not fedstg_is_dp and not fedstg_is_he
    )
    fixed_dp_clip = float(getattr(args, "dp_clip_norm", 0.0))
    fedstg_dp_clips = ({
        key: fixed_dp_clip for key in ("htau", "ztau", "ghg", "gru", "client_z")
    } if fedstg_is_dp and fixed_dp_clip > 0 else {})

    def _client_dp_upload(payload_key, tag, value):
        """Calibrate one FedSTG client-upload type, then DP-protect it."""
        if fedstg_is_he:
            return protected_arbiter_put(ctx, args, tag, value)
        if not fedstg_is_dp:
            return ctx.arbiter.put(tag, value)
        clip_norm = fedstg_dp_clips.get(payload_key)
        if clip_norm is None:
            ctx.arbiter.put(
                f"fedstg_dp_{payload_key}_norm_{tag}",
                float(l2_norm(value).item()),
            )
            calibrated = ctx.arbiter.get(f"fedstg_dp_{payload_key}_clip_{tag}")
            if isinstance(calibrated, list):
                calibrated = calibrated[ctx.rank - 1] if len(calibrated) > 1 else calibrated[0]
            clip_norm = float(calibrated)
            fedstg_dp_clips[payload_key] = clip_norm
            print(
                f"[DPCalibration] FedSTG rank={ctx.rank} type={payload_key} "
                f"tag={tag} clip_norm={clip_norm:.8f}", flush=True,
            )
        return protected_arbiter_put(ctx, args, tag, value, clip_norm=clip_norm)

    def _server_he_inputs(guest_value, hosts_value):
        """Decode explicit HE-TTP client payloads on the trusted Arbiter."""
        return [unprotect_he_ttp_payload(args, guest_value)] + [
            unprotect_he_ttp_payload(args, item)
            for item in safe_get_hosts_data(hosts_value)
        ]

    def _client_he_downlink(value, tag):
        """Account an Arbiter-to-client HE payload in its original phase."""
        return record_he_ttp_downlink(args, value, tag=tag)

    def _server_dp_calibrate(payload_key, tag):
        """Arbiter side of the one-time per-payload FedSTG calibration."""
        if not fedstg_is_dp or payload_key in fedstg_dp_clips:
            return
        norm_guest = float(ctx.guest.get(f"fedstg_dp_{payload_key}_norm_{tag}"))
        norm_hosts = safe_get_hosts_data(
            ctx.hosts.get(f"fedstg_dp_{payload_key}_norm_{tag}")
        )
        clip_norm = float(np.quantile(
            [norm_guest] + [float(value) for value in norm_hosts], 0.9
        ))
        fedstg_dp_clips[payload_key] = clip_norm
        print(
            f"[DPCalibration] FedSTG arbiter type={payload_key} tag={tag} "
            f"clip_norm={clip_norm:.8f}", flush=True,
        )
        ctx.guest.put(f"fedstg_dp_{payload_key}_clip_{tag}", clip_norm)
        host_clips = [clip_norm] * len(norm_hosts)
        ctx.hosts.put(
            f"fedstg_dp_{payload_key}_clip_{tag}",
            host_clips if len(host_clips) > 1 else host_clips[0],
        )

    # ==========================================
    # 1. 联邦环境与数据加载初始化
    # ==========================================
    if ctx.is_on_arbiter:
        print("[FedSTG Server] 正在初始化服务端组件与全局静态图...")
        
        # 【补丁1】接收带头大哥发来的全局节点数，告别 AttributeError
        N_total = ctx.guest.get("N_total_nodes")
        print(f"[FedSTG Server] 已从 Guest 接收到全局总节点数: {N_total}")
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 全局静态图仅在循环外构建一次
        A_S_real = build_fedstg_static_adj(
            dataset_name=args.dataset_name, 
            num_nodes=N_total, 
            sigma=getattr(args, 'fedstg_sigma', 0.0), 
            kappa=getattr(args, 'fedstg_kappa', 0.0),
            project_root=project_root
        ).to(args.device)

        server_model = FedSTG_Server(
            hidden_dim=args.hidden_dim, 
            beta=getattr(args, 'fedstg_beta', 0.5)
        ).to(args.device)
        optimizer_server = torch.optim.Adam(server_model.parameters(), lr=args.lr)
        server_best_wts = None
        
    else:
        print(f"[FedSTG Client {ctx.rank}] 正在加载本地数据与管理器...")
        train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = get_setting(ctx)
        
        # 【补丁1】带头大哥计算节点数并发送给 Arbiter
        if ctx.is_on_guest:
            N_total = sum(len(nodes) for nodes in args.nodes_per)
            ctx.arbiter.put("N_total_nodes", N_total)
            print(f"[FedSTG Guest] 已计算全局总节点数 {N_total} 并发送给 Server")

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        
        manager = FedSTGClientManager(
            ctx=ctx, model=model, optimizer=optimizer, 
            loss_fn=loss_func, alpha=getattr(args, 'fedstg_alpha', 0.01), device=args.device
        )
        
        stopper = ExplicitEarlyStopper(ctx, args, patience=50, min_delta=1e-4)
        best_norm_mae, best_epoch, best_model_wts = float('inf'), -1, None
        total_train_time, total_val_time = 0.0, 0.0
        actual_epochs = 0 

        ctx.arbiter.put("steps_tr", len(train_loader))
        ctx.arbiter.put("steps_val", len(val_loader))
        ctx.arbiter.put("steps_te", len(test_loader))

    # --- 步骤对齐防死锁 ---
    if ctx.is_on_arbiter:
        # 【补丁2】使用安全提取器，防止 duplicate tag 报错
        def get_steps(tag):
            return min([ctx.guest.get(tag)] + safe_get_hosts_data(ctx.hosts.get(tag)))
        STEPS_TRAIN = get_steps("steps_tr")
        STEPS_VAL = get_steps("steps_val")
        STEPS_TEST = get_steps("steps_te")
    else:
        STEPS_TRAIN, STEPS_VAL, STEPS_TEST = len(train_loader), len(val_loader), len(test_loader)

    # ==========================================
    # 2. 核心训练与验证循环
    # ==========================================
    try:
        for epoch in range(args.epochs):
            
            # ---------------- A. Train 阶段 ----------------
            if not ctx.is_on_arbiter:
                actual_epochs = epoch + 1
                manager.step_epoch_start()
                manager.model.train()
                epoch_loss = 0
                epoch_reg = 0.0
                epoch_z_tau_list = []
                train_start_t = time.time()
                
                for i, (x, y) in enumerate(train_loader):
                    if i >= STEPS_TRAIN: break
                    x, y = x.to(args.device), y.to(args.device)
                    manager.optimizer.zero_grad()
                    tag = f"tr_e{epoch}_s{i}"
                    
                    h_tau, z_tau, L_k = manager.model.forward_encode(x)
                    h_tau.retain_grad()
                    _client_dp_upload("htau", f"htau_{tag}", h_tau.detach().cpu())
                    _client_dp_upload("ztau", f"ztau_{tag}", z_tau.mean(dim=0).detach().cpu())
                    
                    h_G = _client_receive_host_split(f"hG_{tag}")
                    h_G = h_G.to(args.device).requires_grad_()
                    
                    pred = manager.model.forward_predict(h_tau, z_tau, h_G)
                    capture_revised_quantized_prediction(ctx, args, f"fedstg_prediction_{tag}", prediction=pred, model_state_dict=manager.model.state_dict())
                    loss, _, reg_loss = manager.compute_total_loss(pred, y, L_k)
                    loss.backward(retain_graph=True)
                    
                    _client_dp_upload("ghg", f"g_hG_{tag}", h_G.grad.cpu())
                    g_htau = _client_receive_host_split(f"g_htau_{tag}")
                    g_htau = g_htau.to(args.device)
                    h_tau.backward(g_htau)
                    
                    torch.nn.utils.clip_grad_norm_(manager.model.parameters(), max_norm=5.0)
                    manager.optimizer.step()
                    epoch_loss += loss.item()
                    epoch_reg += reg_loss.item()
                    epoch_z_tau_list.append(z_tau.mean(dim=(0, 1)).detach().cpu())
                
                total_train_time += (time.time() - train_start_t)
                print(f"Client {ctx.rank} Epoch {epoch}: Train Loss (Norm) {epoch_loss/max(STEPS_TRAIN, 1):.4f}")
                
                local_gru = manager.extract_gru_parameters()
                client_z_mean = torch.stack(epoch_z_tau_list).mean(dim=0)
                _client_dp_upload("gru", f"local_gru_{epoch}", local_gru)
                _client_dp_upload("client_z", f"client_z_{epoch}", client_z_mean)
                global_gru = _client_he_downlink(
                    ctx.arbiter.get(f"global_gru_{epoch}"), f"global_gru_{epoch}"
                )
                manager.load_global_gru_parameters(safe_get_hosts_data(global_gru)[ctx.rank - 1])
            else:
                server_model.train()
                for step in range(STEPS_TRAIN):
                    tag = f"tr_e{epoch}_s{step}"
                    optimizer_server.zero_grad()

                    _server_dp_calibrate("htau", f"htau_{tag}")
                    _server_dp_calibrate("ztau", f"ztau_{tag}")
                    
                    h_htaus = _server_he_inputs(
                        ctx.guest.get(f"htau_{tag}"), ctx.hosts.get(f"htau_{tag}")
                    )
                    h_c = torch.cat([h.to(args.device) for h in h_htaus], dim=1).requires_grad_()
                    
                    h_ztaus = _server_he_inputs(
                        ctx.guest.get(f"ztau_{tag}"), ctx.hosts.get(f"ztau_{tag}")
                    )
                    z_c = torch.cat([z.to(args.device) for z in h_ztaus], dim=0)
                    
                    h_G_global = server_model(h_c, z_c, A_S_real)
                    
                    splits = [h_G_global[:, sum([h.shape[1] for h in h_htaus[:k]]) : sum([h.shape[1] for h in h_htaus[:k+1]]), :].detach().cpu() for k in range(args.num_clients)]
                    ctx.guest.put(f"hG_{tag}", splits[0])
                    _server_send_host_splits(f"hG_{tag}", splits[1:])
                    
                    _server_dp_calibrate("ghg", f"g_hG_{tag}")
                    g_hG = _server_he_inputs(
                        ctx.guest.get(f"g_hG_{tag}"), ctx.hosts.get(f"g_hG_{tag}")
                    )
                    g_global = torch.cat([g.to(args.device) for g in g_hG], dim=1)
                    h_G_global.backward(g_global)
                    
                    g_splits = [h_c.grad[:, sum([h.shape[1] for h in h_htaus[:k]]) : sum([h.shape[1] for h in h_htaus[:k+1]]), :].cpu() for k in range(args.num_clients)]
                    ctx.guest.put(f"g_htau_{tag}", g_splits[0])
                    _server_send_host_splits(f"g_htau_{tag}", g_splits[1:])
                    optimizer_server.step()

                _server_dp_calibrate("gru", f"local_gru_{epoch}")
                _server_dp_calibrate("client_z", f"client_z_{epoch}")
                all_grus = _server_he_inputs(
                    ctx.guest.get(f"local_gru_{epoch}"), ctx.hosts.get(f"local_gru_{epoch}")
                )
                all_zs = torch.stack(_server_he_inputs(
                    ctx.guest.get(f"client_z_{epoch}"), ctx.hosts.get(f"client_z_{epoch}")
                ))
                W_ij = F.softmax(torch.matmul(F.normalize(all_zs, dim=1), F.normalize(all_zs, dim=1).T), dim=1)
                
                personalized_grus = [{key: sum(W_ij[i, j].item() * all_grus[j][key] for j in range(args.num_clients)) for key in all_grus[0].keys()} for i in range(args.num_clients)]
                ctx.guest.put(f"global_gru_{epoch}", personalized_grus[0])
                ctx.hosts.put(f"global_gru_{epoch}", personalized_grus[1:] if len(personalized_grus) > 2 else personalized_grus[1])

            # ---------------- B. Val 阶段与上帝裁判机制 ----------------
            if not ctx.is_on_arbiter:
                manager.model.eval()
                val_mae, val_rmse, val_elements = 0.0, 0.0, 0
                val_start_t = time.time()
                
                with torch.no_grad():
                    for i, (x, y) in enumerate(val_loader):
                        if i >= STEPS_VAL: break
                        tag = f"va_e{epoch}_s{i}"
                        x, y = x.to(args.device), y.to(args.device)
                        
                        h_tau, z_tau, _ = manager.model.forward_encode(x)
                        _client_dp_upload("htau", f"htau_{tag}", h_tau.cpu())
                        _client_dp_upload("ztau", f"ztau_{tag}", z_tau.mean(dim=0).cpu())
                        
                        h_G = _client_receive_host_split(f"hG_{tag}")
                        h_G = h_G.to(args.device)
                        
                        pred = manager.model.forward_predict(h_tau, z_tau, h_G)
                        if y.dim() == 4 and y.shape[-1] == 1: y = y.squeeze(-1)
                        if pred.dim() == 4 and pred.shape[-1] == 1: pred = pred.squeeze(-1)
                        if pred.shape != y.shape: pred = pred.reshape_as(y)

                        y_cpu, pred_cpu = y.cpu().numpy(), pred.cpu().numpy()
                        val_mae += np.sum(np.abs(y_cpu - pred_cpu))
                        val_rmse += np.sum((y_cpu - pred_cpu) ** 2)
                        val_elements += y.numel()

                total_val_time += (time.time() - val_start_t)
                norm_val_mae = val_mae / val_elements
                norm_val_rmse = math.sqrt(val_rmse / val_elements)
                
                print(f"   ---> [验证集] Client {ctx.rank} Epoch {epoch} | Norm MAE: {norm_val_mae:.4f} | Norm RMSE: {norm_val_rmse:.4f}")

                is_best, should_stop = stopper.check_and_sync(norm_val_mae)

                if is_best:
                    best_norm_mae, best_epoch = norm_val_mae, epoch
                    best_model_wts = copy.deepcopy(manager.model.state_dict())
                
                if ctx.is_on_guest:
                    ctx.arbiter.put(f"server_signal_{epoch}", (bool(is_best), bool(should_stop)))

                if should_stop:
                    raise EarlyStopSignal("收到全局早停指令，跳出循环！")

            else:
                server_model.eval()
                with torch.no_grad():
                    for step in range(STEPS_VAL):
                        tag = f"va_e{epoch}_s{step}"
                        _server_dp_calibrate("htau", f"htau_{tag}")
                        _server_dp_calibrate("ztau", f"ztau_{tag}")
                        h_htaus = _server_he_inputs(
                            ctx.guest.get(f"htau_{tag}"), ctx.hosts.get(f"htau_{tag}")
                        )
                        h_c = torch.cat([h.to(args.device) for h in h_htaus], dim=1)
                        
                        h_ztaus = _server_he_inputs(
                            ctx.guest.get(f"ztau_{tag}"), ctx.hosts.get(f"ztau_{tag}")
                        )
                        z_c = torch.cat([z.to(args.device) for z in h_ztaus], dim=0)
                        
                        h_G_global = server_model(h_c, z_c, A_S_real)
                        splits = [h_G_global[:, sum([h.shape[1] for h in h_htaus[:k]]) : sum([h.shape[1] for h in h_htaus[:k+1]]), :].cpu() for k in range(args.num_clients)]
                        ctx.guest.put(f"hG_{tag}", splits[0])
                        _server_send_host_splits(f"hG_{tag}", splits[1:])

                server_signal = ctx.guest.get(f"server_signal_{epoch}")
                if isinstance(server_signal, tuple):
                    server_is_best, server_should_stop = server_signal
                else:
                    server_is_best, server_should_stop = False, bool(server_signal)
                if server_is_best:
                    server_best_wts = copy.deepcopy(server_model.state_dict())
                if bool(server_should_stop):
                    print("[Server] 收到 Guest 裁判早停下车信号，同步跳出！")
                    break

    except EarlyStopSignal:
        if not ctx.is_on_arbiter:
            print(f"Rank {ctx.rank}: ⚔️ 触发早停机制，记录实际训练轮数为 {actual_epochs}，进入 Test 阶段...")

    # ==========================================
    # 3. 真实物理测试与“同归于尽”落盘阶段
    # ==========================================
    if not ctx.is_on_arbiter:
        if best_model_wts is not None:
            manager.model.load_state_dict(best_model_wts)
            
        print(f"Rank {ctx.rank}: 🚀 启动最终 Test Set 物理尺度评估...")
        manager.model.eval()
        synchronize_cuda_for_timing(args.device)
        test_start_t = time.perf_counter()
        test_mae, test_rmse, test_mape = 0.0, 0.0, 0.0
        test_elements, valid_mape_count = 0, 0
        
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                if i >= STEPS_TEST: break
                tag = f"te_s{i}"
                x, y = x.to(args.device), y.to(args.device)
                
                h_tau, z_tau, _ = manager.model.forward_encode(x)
                _client_dp_upload("htau", f"htau_{tag}", h_tau.cpu())
                _client_dp_upload("ztau", f"ztau_{tag}", z_tau.mean(dim=0).cpu())
                
                h_G = _client_receive_host_split(f"hG_{tag}")
                h_G = h_G.to(args.device)
                
                pred = manager.model.forward_predict(h_tau, z_tau, h_G)
                if y.dim() == 4 and y.shape[-1] == 1: y = y.squeeze(-1)
                if pred.dim() == 4 and pred.shape[-1] == 1: pred = pred.squeeze(-1)
                if pred.shape != y.shape: pred = pred.reshape_as(y)

                y_real = scaler.inverse_transform(y).cpu().numpy()
                pred_real = scaler.inverse_transform(pred).cpu().numpy()
                
                test_mae += np.sum(np.abs(y_real - pred_real))
                test_rmse += np.sum((y_real - pred_real) ** 2)
                test_elements += y_real.size
                
                mask = y_real > 0.5
                if np.sum(mask) > 0:
                    test_mape += np.sum(np.abs(y_real[mask] - pred_real[mask]) / y_real[mask])
                    valid_mape_count += np.sum(mask)

        acc_mae = round(test_mae / test_elements, 4)
        acc_rmse = round(math.sqrt(test_rmse / test_elements), 4)
        acc_mse = round(acc_rmse ** 2, 4)
        acc_mape = round((test_mape / valid_mape_count) * 100 if valid_mape_count > 0 else 0.0, 4)
        synchronize_cuda_for_timing(args.device)
        eff_test_time = round(time.perf_counter() - test_start_t, 4)
        print(
            f"[TestTimingAudit] model=FedSTG rank={ctx.rank} "
            f"batches={min(len(test_loader), STEPS_TEST)} elements={test_elements} "
            f"seconds={eff_test_time:.6f}",
            flush=True,
        )
        
        N_local = len(args.nodes_per[ctx.rank])
        B = args.batch_size
        H = args.hidden_dim
        d = getattr(args, 'node_emb_dim', 32) 
        
        tensor_size = B * N_local * H
        train_step_params = 4 * tensor_size + d
        eval_step_params = 2 * tensor_size + d
        gru_params_count = sum(p.numel() for p in manager.extract_gru_parameters().values())
        epoch_params = 2 * gru_params_count + d
        
        total_comm_params = (
            (train_step_params * STEPS_TRAIN + eval_step_params * STEPS_VAL + epoch_params) * actual_epochs
            + (eval_step_params * STEPS_TEST) 
        )
        eff_comm_size_mb = round((total_comm_params * 4) / (1024 * 1024), 4)
        
        eff_flops = 0.0
        try:
            from thop import profile
            class FedSTG_FLOPs_Wrapper(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def forward(self, x):
                    h_tau, z_tau, _ = self.m.forward_encode(x)
                    h_G_mock = torch.zeros_like(h_tau)
                    return self.m.forward_predict(h_tau, z_tau, h_G_mock)

            wrapper = FedSTG_FLOPs_Wrapper(manager.model).to(args.device)
            dummy_x, _ = next(iter(val_loader))
            flops, _ = profile(wrapper, inputs=(dummy_x.to(args.device),), verbose=False)
            eff_flops = round(flops / 1e9, 4)
        except Exception: pass

        log_experiment_results(
            model_name="FedSTG", 
            dataset_client=f"{args.dataset_name}_client{ctx.rank}", 
            feature_type=args.feature_type,
            best_epoch=best_epoch + 1, 
            acc_mae=acc_mae, acc_mse=acc_mse, acc_rmse=acc_rmse, acc_mape=acc_mape,
            acc_elements=test_elements,
            acc_abs_error_sum=float(test_mae),
            acc_sq_error_sum=float(test_rmse),
            acc_mape_error_sum=float(test_mape * 100.0),
            acc_mape_elements=int(valid_mape_count),
            eff_train_time=round(total_train_time, 2), 
            eff_val_time=round(total_val_time / max(actual_epochs, 1), 4), 
            eff_test_time=eff_test_time, 
            eff_comm_size_mb=eff_comm_size_mb, 
            eff_train_round=actual_epochs, 
            eff_flops=eff_flops 
        )

        if ctx.is_on_guest and not bool(getattr(args, "cross_device_multiplex", False)):
            import os, signal
            print("👑 Guest 节点：数据已写入，执行同归于尽清理后台...")
            time.sleep(2) 
            os.kill(os.getppid(), signal.SIGTERM) 
            
    else:
        if server_best_wts is not None:
            server_model.load_state_dict(server_best_wts)
        server_model.eval()
        with torch.no_grad():
            for step in range(STEPS_TEST):
                tag = f"te_s{step}"
                _server_dp_calibrate("htau", f"htau_{tag}")
                _server_dp_calibrate("ztau", f"ztau_{tag}")
                h_htaus = _server_he_inputs(
                    ctx.guest.get(f"htau_{tag}"), ctx.hosts.get(f"htau_{tag}")
                )
                h_c = torch.cat([h.to(args.device) for h in h_htaus], dim=1)
                
                h_ztaus = _server_he_inputs(
                    ctx.guest.get(f"ztau_{tag}"), ctx.hosts.get(f"ztau_{tag}")
                )
                z_c = torch.cat([z.to(args.device) for z in h_ztaus], dim=0)
                
                h_G_global = server_model(h_c, z_c, A_S_real)
                splits = [h_G_global[:, sum([h.shape[1] for h in h_htaus[:k]]) : sum([h.shape[1] for h in h_htaus[:k+1]]), :].cpu() for k in range(args.num_clients)]
                ctx.guest.put(f"hG_{tag}", splits[0])
                _server_send_host_splits(f"hG_{tag}", splits[1:])

def train_fedgode(ctx):
    print(f"Rank {ctx.rank}: [FedGODE-Platform] 启动专属白盒训练循环...")
    
    if not ctx.is_on_arbiter:
        train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = get_setting(ctx)
        
        # 数据转入设备
        for dset in [train_set, val_set, test_set]:
            if hasattr(dset, 'data') and isinstance(dset.data, np.ndarray):
                dset.data = torch.from_numpy(dset.data).to(args.device)
                
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        
        ctx.arbiter.put("steps_tr", len(train_loader))
        ctx.arbiter.put("steps_val", len(val_loader))
        ctx.arbiter.put("steps_te", len(test_loader))
        
        best_norm_mae = float('inf')
        best_epoch = -1
        best_model_wts = None
        total_train_time, total_val_time = 0.0, 0.0
        
        # 【约束 1】追踪实际运行轮数
        actual_epochs = 0 
        
        # 挂载你之前写好的上帝裁判早停器
        stopper = ExplicitEarlyStopper(ctx, args, patience=50, min_delta=1e-4)
        
    # ================= 步数防死锁同步 =================
    if ctx.is_on_arbiter:
        def get_steps(tag):
            res_g = ctx.guest.get(tag)
            res_h = ctx.hosts.get(tag)
            res_h = res_h if isinstance(res_h, list) else [res_h]
            return min([res_g] + res_h)
        STEPS_TRAIN = get_steps("steps_tr")
        STEPS_VAL = get_steps("steps_val")
        STEPS_TEST = get_steps("steps_te")
    else:
        STEPS_TRAIN, STEPS_VAL, STEPS_TEST = len(train_loader), len(val_loader), len(test_loader)

    # FATE's standalone backend turns messages larger than 1 MiB into tables.
    # With 32 concurrent FedGODE clients that table path forks copy workers for
    # every complete model upload and can exhaust host RAM.  Split every state
    # into sub-1-MiB byte messages so the plain protocol stays on direct bytes.
    _FEDGODE_STATE_CHUNK_BYTES = 512 * 1024

    def _pack_fedgode_state(state):
        buffer = io.BytesIO()
        torch.save(state, buffer)
        return buffer.getvalue()

    def _unpack_fedgode_state(payload):
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError(f"FedGODE expected a byte state payload, got {type(payload)!r}")
        return torch.load(io.BytesIO(payload), map_location="cpu")

    def _unwrap_fedgode_payload(value, label):
        # ``ctx.arbiter.get`` can return a single-element broadcast list on
        # FATE standalone, even though the caller is a single Party proxy.
        while isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise RuntimeError(
                    f"FedGODE expected one {label} payload, received {len(value)} values"
                )
            value = value[0]
        return value

    def _send_fedgode_state(party, tag, state):
        payload = _pack_fedgode_state(state)
        chunks = [
            payload[offset: offset + _FEDGODE_STATE_CHUNK_BYTES]
            for offset in range(0, len(payload), _FEDGODE_STATE_CHUNK_BYTES)
        ] or [b""]
        party.put(f"{tag}__chunk_count", len(chunks))
        for index, chunk in enumerate(chunks):
            party.put(f"{tag}__chunk_{index}", chunk)
        return len(payload), len(chunks)

    def _receive_fedgode_state(party, tag):
        chunk_count = int(_unwrap_fedgode_payload(party.get(f"{tag}__chunk_count"), "chunk-count"))
        if chunk_count <= 0:
            raise RuntimeError(f"FedGODE received invalid chunk count {chunk_count}")
        payload = b"".join(
            _unwrap_fedgode_payload(party.get(f"{tag}__chunk_{index}"), f"chunk {index}")
            for index in range(chunk_count)
        )
        return _unpack_fedgode_state(payload)

    def _receive_fedgode_states_from_hosts(tag):
        counts = ctx.hosts.get(f"{tag}__chunk_count")
        counts = counts if isinstance(counts, list) else [counts]
        counts = [int(_unwrap_fedgode_payload(count, "host chunk-count")) for count in counts]
        if not counts or any(count <= 0 for count in counts):
            raise RuntimeError(f"FedGODE received invalid host chunk counts: {counts}")
        if len(set(counts)) != 1:
            raise RuntimeError(f"FedGODE host state chunk counts differ: {counts}")
        payloads = [bytearray() for _ in counts]
        for index in range(counts[0]):
            chunks = ctx.hosts.get(f"{tag}__chunk_{index}")
            chunks = chunks if isinstance(chunks, list) else [chunks]
            if len(chunks) != len(payloads):
                raise RuntimeError("FedGODE received an unexpected number of host state chunks")
            for payload, chunk in zip(payloads, chunks):
                payload.extend(_unwrap_fedgode_payload(chunk, f"host chunk {index}"))
        return [_unpack_fedgode_state(bytes(payload)) for payload in payloads]

    def _broadcast_fedgode_state(tag, state):
        payload = _pack_fedgode_state(state)
        chunks = [
            payload[offset: offset + _FEDGODE_STATE_CHUNK_BYTES]
            for offset in range(0, len(payload), _FEDGODE_STATE_CHUNK_BYTES)
        ] or [b""]
        ctx.guest.put(f"{tag}__chunk_count", len(chunks))
        # ``hosts.put`` broadcasts a scalar to every host.  Passing a list
        # sends that entire list to every host (rather than distributing it),
        # which makes each receiver see 31 chunk counts in the 32-client run.
        ctx.hosts.put(f"{tag}__chunk_count", len(chunks))
        for index, chunk in enumerate(chunks):
            ctx.guest.put(f"{tag}__chunk_{index}", chunk)
            ctx.hosts.put(f"{tag}__chunk_{index}", chunk)

    # Each client owns a graph with a different local node count.  Graph
    # adjacency buffers are local topology, not federated model parameters;
    # exclude them by both name and shape so they never enter DP/HE aggregation.
    local_node_count = len(args.nodes_per[ctx.rank]) if not ctx.is_on_arbiter else None

    def _fedgode_shareable_state(name, value):
        lowered = name.lower()
        if 'a_hat' in lowered or 'adj' in lowered:
            return False
        if (
            local_node_count is not None
            and value.dim() >= 2
            and value.shape[0] == local_node_count
            and value.shape[1] == local_node_count
        ):
            return False
        return True

    # A positive command-line value is a fixed clipping threshold.  Only the
    # auto-C mode (``--dp_clip_norm 0``) needs the first-round norm exchange.
    # Leaving this as ``None`` for a fixed C made the arbiter wait forever for
    # ``dp_delta_norm_*`` messages that clients correctly did not send.
    dp_clip_norm = (
        float(args.dp_clip_norm)
        if args.protection == 'dp' and float(getattr(args, 'dp_clip_norm', 0.0)) > 0
        else None
    )
    he_public_key = None
    he_private_key = None
    he_upload_bytes = 0
    he_download_bytes = 0
    he_backend = str(getattr(args, 'he_backend', 'auto')).lower()
    he_scheme = str(getattr(args, 'he_scheme', 'ckks')).lower()
    if args.protection == 'he':
        if he_backend != 'he_sa':
            raise ValueError(
                'FedGODE HE uses the HE-SA backend. Set --he_backend he_sa '
                '(or leave auto, which is normalized to he_sa).'
            )
        if he_scheme == 'ckks':
            key_tag = '__fedgode_he_sa_ckks_context'
            if ctx.is_on_arbiter:
                he_private_key = ckks_generate_context(
                    args.he_ckks_poly_modulus_degree, args.he_ckks_scale_bits,
                )
                public_key_payload = ckks_export_public_context(he_private_key)
                ctx.guest.put(key_tag, public_key_payload)
                ctx.hosts.put(key_tag, public_key_payload)
                print(
                    f"[HESA] arbiter generated packed CKKS context "
                    f"poly_degree={args.he_ckks_poly_modulus_degree} "
                    f"slots={args.he_ckks_poly_modulus_degree // 2}",
                    flush=True,
                )
            else:
                he_public_key = ckks_import_context(ctx.arbiter.get(key_tag))
                print(f"[HESA] rank={ctx.rank} received packed CKKS public context", flush=True)
        else:
            key_tag = '__fedgode_he_sa_public_key'
            if ctx.is_on_arbiter:
                he_public_key, he_private_key = generate_keypair(int(args.he_key_bits))
                public_key_payload = export_public_key(he_public_key)
                ctx.guest.put(key_tag, public_key_payload)
                ctx.hosts.put(key_tag, public_key_payload)
                print(f"[HESA] arbiter generated Paillier key bits={args.he_key_bits}", flush=True)
            else:
                he_public_key = import_public_key(ctx.arbiter.get(key_tag))
                print(f"[HESA] rank={ctx.rank} received Paillier public key", flush=True)

    # Plain FedAvg averages complete local states in its first round, which
    # implicitly aligns independently-created client models.  Delta based
    # DP/HE aggregation must make that alignment explicit; otherwise every
    # client adds the averaged update back to a *different* initial model.
    # The exchanged initial state is data-independent model initialisation,
    # deliberately sent as a public protocol message (not a DP upload).
    # All three modes must share a common model before round one.  Plain
    # FedAvg still aggregates complete local states; only DP/HE use deltas.
    # Without this Plain-side sync, a sigma=0 comparison would be confounded
    # by independently seeded client initialisations in the first local epoch.
    needs_initial_state_sync = args.protection in ('plain', 'dp', 'he')
    uses_delta_aggregation = args.protection in ('dp', 'he')
    fedgode_global_state = None
    if needs_initial_state_sync:
        init_tag = 'fedgode_public_init_state'
        if ctx.is_on_arbiter:
            init_guest = _receive_fedgode_state(ctx.guest, init_tag)
            init_hosts = _receive_fedgode_states_from_hosts(init_tag)
            init_states = [init_guest] + init_hosts
            fedgode_global_state = {}
            for key, value in init_guest.items():
                candidates = [state.get(key) for state in init_states]
                if (
                    all(torch.is_tensor(item) for item in candidates)
                    and all(item.shape == value.shape and item.dtype == value.dtype for item in candidates)
                ):
                    fedgode_global_state[key] = value.detach().cpu().clone()
            if not fedgode_global_state:
                raise RuntimeError('FedGODE DP/HE found no common shareable state at public initialisation.')
            _broadcast_fedgode_state(init_tag, fedgode_global_state)
            print(f"[FedGODEStateSync] arbiter initialised canonical shared state keys={len(fedgode_global_state)}", flush=True)
        else:
            local_init_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
                if _fedgode_shareable_state(key, value)
            }
            # Do not use protected_arbiter_put here: this is public random
            # initialisation, rather than a data-derived client update.
            _send_fedgode_state(ctx.arbiter, init_tag, local_init_state)
            fedgode_global_state = _receive_fedgode_state(ctx.arbiter, init_tag)
            model.load_state_dict(fedgode_global_state, strict=False)
            print(f"[FedGODEStateSync] rank={ctx.rank} loaded canonical shared state keys={len(fedgode_global_state)}", flush=True)

    # ================= 核心联邦循环 =================
    try:
        for epoch in range(args.epochs):
            if not ctx.is_on_arbiter:
                # ---------------- A. 本地训练 ----------------
                round_start_w = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                    if _fedgode_shareable_state(k, v)
                }
                if uses_delta_aggregation:
                    start_delta = {
                        key: value - fedgode_global_state[key]
                        for key, value in round_start_w.items()
                        if key in fedgode_global_state and value.is_floating_point()
                    }
                    start_mismatch = l2_norm(start_delta).item() if start_delta else 0.0
                    print(
                        f"[FedGODEStateSync] rank={ctx.rank} epoch={epoch + 1} "
                        f"start_global_l2_diff={start_mismatch:.8e}",
                        flush=True,
                    )
                model.train()
                epoch_loss = 0.0
                epoch_reg = 0.0
                train_start = time.time()
                for i, (x, y) in enumerate(train_loader):
                    if i >= STEPS_TRAIN: break
                    x, y = x.to(args.device), y.to(args.device)
                    optimizer.zero_grad()
                    
                    pred = model(x)
                    capture_revised_quantized_prediction(ctx, args, f"fedgode_prediction_{epoch}_{i}", prediction=pred, model_state_dict=model.state_dict())
                    if y.dim() == 4 and y.shape[-1] == 1: y = y.squeeze(-1)
                    if pred.dim() == 4 and pred.shape[-1] == 1: pred = pred.squeeze(-1)
                    if pred.shape != y.shape: pred = pred.reshape_as(y)
                    
                    reg_loss = torch.zeros((), device=args.device)
                    loss = loss_func(pred, y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    epoch_reg += reg_loss.item()
                    
                total_train_time += (time.time() - train_start)
                
                # 【约束 2】训练日志打印归一化数据
                print(f"Client {ctx.rank} Epoch {epoch}: Train Loss (Norm) = {epoch_loss/max(1, STEPS_TRAIN):.4f}")
                
                # ---------------- B. 标准 FedAvg 聚合 ----------------
               
                local_state = {}
                for k, v in model.state_dict().items():
                    if not _fedgode_shareable_state(k, v):
                        continue
                    local_state[k] = v.cpu().clone()
                local_delta = {k: local_state[k] - round_start_w[k] for k in local_state}
                print(f"Rank {ctx.rank}: delta_l2_norm epoch={epoch + 1} value={l2_norm(local_delta).item():.8f}", flush=True)
                # HE-SA and DP protect model deltas. Integer bookkeeping buffers
                # (e.g., BatchNorm num_batches_tracked) are retained locally.
                local_w = local_delta if args.protection == 'dp' else local_state
                if args.protection == 'he':
                    local_w = {k: value for k, value in local_delta.items() if value.is_floating_point()}
                if args.protection == 'dp' and float(args.dp_clip_norm) <= 0:
                    ctx.arbiter.put(f"dp_delta_norm_{epoch}", float(l2_norm(local_delta).item()))
                    calibrated_clip = ctx.arbiter.get(f"dp_clip_norm_{epoch}")
                    if isinstance(calibrated_clip, (list, tuple)):
                        calibrated_clip = calibrated_clip[0]
                    args.dp_clip_norm = float(calibrated_clip)
                    print(f"[DPCalibration] rank={ctx.rank} epoch={epoch + 1} clip_norm={args.dp_clip_norm:.8f}", flush=True)
                print(
                    f"Rank {ctx.rank}: upload_l2_norm epoch={epoch + 1} "
                    f"value={l2_norm(local_w).item():.8f}",
                    flush=True,
                )
                if args.protection == 'he':
                    he_started = time.perf_counter()
                    he_total_elements = sum(value.numel() for value in local_w.values())
                    he_completed_elements = 0
                    print(
                        f"[HESA] rank={ctx.rank} epoch={epoch + 1} "
                        f"begin_encrypt scheme={he_scheme} elements={he_total_elements}",
                        flush=True,
                    )
                    def _he_encrypt_progress(completed: int):
                        nonlocal he_completed_elements
                        he_completed_elements += completed
                        print(
                            f"[HESA] rank={ctx.rank} epoch={epoch + 1} "
                            f"encrypt_progress={he_completed_elements}/{he_total_elements}",
                            flush=True,
                        )
                    if he_scheme == 'ckks':
                        encrypted_w = ckks_encrypt_tree(
                            local_w,
                            he_public_key,
                            _he_encrypt_progress,
                            args.he_ckks_poly_modulus_degree // 2,
                        )
                        upload_bytes = ckks_ciphertext_bytes(encrypted_w)
                    else:
                        encrypted_w = encrypt_tree(local_w, he_public_key, _he_encrypt_progress)
                        upload_bytes = ciphertext_bytes(encrypted_w)
                    he_upload_bytes += upload_bytes
                    total_train_time += time.perf_counter() - he_started
                    print(
                        f"[HESA] rank={ctx.rank} epoch={epoch + 1} encrypted_upload_bytes={upload_bytes} "
                        f"encrypt_s={time.perf_counter() - he_started:.6f}",
                        flush=True,
                    )
                    ctx.arbiter.put(f"w_{epoch}", encrypted_w)
                    ctx.arbiter.put(f"he_upload_bytes_{epoch}", upload_bytes)
                elif args.protection == 'plain':
                    payload_bytes, chunk_count = _send_fedgode_state(ctx.arbiter, f"w_{epoch}", local_w)
                    if epoch == 0:
                        print(
                            f"[FedGODETransport] rank={ctx.rank} using chunked byte-state transport "
                            f"payload_bytes={payload_bytes} chunks={chunk_count}",
                            flush=True,
                        )
                else:
                    protected_arbiter_put(ctx, args, f"w_{epoch}", local_w)
                
                if args.protection == 'plain':
                    global_w = _receive_fedgode_state(ctx.arbiter, f"gw_{epoch}")
                else:
                    global_w_data = ctx.arbiter.get(f"gw_{epoch}")
                    global_w = global_w_data[ctx.rank-1] if isinstance(global_w_data, list) else global_w_data
                if uses_delta_aggregation:
                    # The arbiter returns the complete canonical shared
                    # state, never a delta that clients add to private starts.
                    fedgode_global_state = {k: v.cpu() for k, v in global_w.items()}
                elif needs_initial_state_sync:
                    # Plain FedAvg already returns a complete global state.
                    fedgode_global_state = {k: v.cpu() for k, v in global_w.items()}
                if args.protection == 'he':
                    he_download_bytes += sum(value.numel() * value.element_size() for value in global_w.values())
                    he_arbiter_seconds = ctx.arbiter.get(f"he_arbiter_seconds_{epoch}")
                    # FATE may return a one-element list for a broadcast even
                    # on the client side; normalize it before timing/accounting.
                    while isinstance(he_arbiter_seconds, (list, tuple)):
                        if not he_arbiter_seconds:
                            raise RuntimeError("FedGODE HE received an empty Arbiter timing payload")
                        he_arbiter_seconds = he_arbiter_seconds[0]
                    total_train_time += float(he_arbiter_seconds)
                    print(
                        f"[HESA] rank={ctx.rank} epoch={epoch + 1} "
                        f"arbiter_aggregate_decrypt_s={float(he_arbiter_seconds):.6f}",
                        flush=True,
                    )
                model.load_state_dict(global_w, strict=False)
                
                # ---------------- C. 验证与早停裁判 ----------------
                model.eval()
                val_start = time.time()
                val_mae, val_rmse = 0.0, 0.0
                val_elements = 0
                
                with torch.no_grad():
                    for i, (x, y) in enumerate(val_loader):
                        if i >= STEPS_VAL: break
                        x, y = x.to(args.device), y.to(args.device)
                        pred = model(x)
                        
                        if y.dim() == 4 and y.shape[-1] == 1: y = y.squeeze(-1)
                        if pred.dim() == 4 and pred.shape[-1] == 1: pred = pred.squeeze(-1)
                        if pred.shape != y.shape: pred = pred.reshape_as(y)
                        
                        # 【约束 2】验证过程不进行 inverse_transform，直接计算归一化误差
                        y_cpu, pred_cpu = y.cpu().numpy(), pred.cpu().numpy()
                        val_mae += np.sum(np.abs(y_cpu - pred_cpu))
                        val_rmse += np.sum((y_cpu - pred_cpu)**2)
                        val_elements += y.numel()
                        
                norm_mae = val_mae / val_elements
                norm_rmse = math.sqrt(val_rmse / val_elements)
                total_val_time += (time.time() - val_start)
                
                # 【约束 2】验证日志打印归一化数据
                print(f"   ---> [验证结果(归一化)] Client {ctx.rank} Epoch {epoch} | Norm MAE: {norm_mae:.4f} | Norm RMSE: {norm_rmse:.4f}")
                
                if norm_mae < best_norm_mae:
                    best_norm_mae = norm_mae
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                
                # 更新实际运行轮数
                actual_epochs = epoch + 1
                
                # 调用早停器
                should_stop = stopper.check_and_sync(norm_mae)
                # The arbiter does not participate in ExplicitEarlyStopper.
                # Send the decision before a client skips this round's upload.
                ctx.arbiter.put(f"fedgode_stop_{epoch}", bool(should_stop))
                if should_stop:
                    print(f"Rank {ctx.rank}: 🛑 收到全局早停信号，在第 {actual_epochs} 轮跳出！")
                    raise EarlyStopSignal("触发早停")
                    
            else: 
                # ================= Arbiter 服务端逻辑 =================
                if args.protection == 'dp' and dp_clip_norm is None:
                    norm_guest = float(ctx.guest.get(f"dp_delta_norm_{epoch}"))
                    norm_hosts = ctx.hosts.get(f"dp_delta_norm_{epoch}")
                    norm_hosts = norm_hosts if isinstance(norm_hosts, list) else [norm_hosts]
                    dp_clip_norm = float(np.quantile([norm_guest] + [float(v) for v in norm_hosts], 0.9))
                    print(f"[DPCalibration] arbiter epoch={epoch + 1} clip_norm={dp_clip_norm:.8f}", flush=True)
                    ctx.guest.put(f"dp_clip_norm_{epoch}", dp_clip_norm)
                    ctx.hosts.put(f"dp_clip_norm_{epoch}", [dp_clip_norm] * len(norm_hosts))
                if args.protection == 'plain':
                    w_guest = _receive_fedgode_state(ctx.guest, f"w_{epoch}")
                    w_hosts = _receive_fedgode_states_from_hosts(f"w_{epoch}")
                else:
                    w_guest = ctx.guest.get(f"w_{epoch}")
                    w_hosts = ctx.hosts.get(f"w_{epoch}")
                    w_hosts = w_hosts if isinstance(w_hosts, list) else [w_hosts]
                all_w = [w_guest] + w_hosts
                if args.protection == 'he':
                    he_started = time.perf_counter()
                    if he_scheme == 'ckks':
                        encrypted_sum = ckks_homomorphic_sum_tree(all_w, he_private_key)
                        summed_delta = ckks_decrypt_tree(encrypted_sum, he_private_key)
                    else:
                        encrypted_sum = homomorphic_sum_tree(all_w)
                        summed_delta = decrypt_tree(encrypted_sum, he_private_key)
                    capture_he_sa_aggregate(ctx, args, f"fedgode_aggregate_{epoch}", summed_delta)
                    gw = {key: value / len(all_w) for key, value in summed_delta.items()}
                    upload_guest = int(ctx.guest.get(f"he_upload_bytes_{epoch}"))
                    upload_hosts = ctx.hosts.get(f"he_upload_bytes_{epoch}")
                    upload_hosts = upload_hosts if isinstance(upload_hosts, list) else [upload_hosts]
                    print(
                        f"[HESA] arbiter epoch={epoch + 1} aggregate_decrypt_s={time.perf_counter() - he_started:.6f} "
                        f"encrypted_upload_bytes={upload_guest + sum(int(value) for value in upload_hosts)}",
                        flush=True,
                    )
                    he_arbiter_seconds = time.perf_counter() - he_started
                else:
                    gw = {}
                    for k in all_w[0].keys():
                        shapes = [w[k].shape for w in all_w if k in w]
                        
                        if len(shapes) == len(all_w) and all(s == shapes[0] for s in shapes):
                            gw[k] = sum(w[k] for w in all_w) / len(all_w)
                    he_arbiter_seconds = None

                if uses_delta_aggregation:
                    # Keep the only authoritative shared model at the
                    # arbiter.  This makes sigma=0 DP algebraically match
                    # FedAvg for every shared floating-point parameter.
                    next_global_state = {
                        key: value.detach().cpu().clone()
                        for key, value in fedgode_global_state.items()
                    }
                    for key, delta in gw.items():
                        if key in next_global_state and next_global_state[key].is_floating_point():
                            next_global_state[key] = next_global_state[key] + delta.cpu()
                    fedgode_global_state = next_global_state
                    gw = fedgode_global_state
                    
                if args.protection == 'plain':
                    _broadcast_fedgode_state(f"gw_{epoch}", gw)
                else:
                    ctx.guest.put(f"gw_{epoch}", gw)
                    ctx.hosts.put(f"gw_{epoch}", [gw] * len(w_hosts))
                if args.protection == 'he':
                    ctx.guest.put(f"he_arbiter_seconds_{epoch}", he_arbiter_seconds)
                    ctx.hosts.put(f"he_arbiter_seconds_{epoch}", [he_arbiter_seconds] * len(w_hosts))
                # Clients evaluate only after receiving gw.  Consume their
                # decision here, after this round's aggregation, so an early
                # stop prevents the *next* round without blocking this one.
                stop_guest = bool(ctx.guest.get(f"fedgode_stop_{epoch}"))
                stop_hosts = ctx.hosts.get(f"fedgode_stop_{epoch}")
                stop_hosts = stop_hosts if isinstance(stop_hosts, list) else [stop_hosts]
                if stop_guest or any(bool(value) for value in stop_hosts):
                    print(f"[FedGODE] arbiter received early-stop at epoch={epoch + 1}; exiting normally.", flush=True)
                    break
                
    except EarlyStopSignal:
        if not ctx.is_on_arbiter:
            print(f"Rank {ctx.rank}: ⚔️ 早停触发，实际运行 {actual_epochs} 轮，进入测试。")

    # ================= 最终测试与数据落盘 =================
    if not ctx.is_on_arbiter:
        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)
        
        print(f"Rank {ctx.rank}: 🚀 启动最终物理尺度 Test 评估...")
        model.eval()
        test_start = time.time()
        test_mae, test_rmse, test_mape = 0.0, 0.0, 0.0
        test_elements, valid_mape_count = 0, 0
        
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                if i >= STEPS_TEST: break
                x, y = x.to(args.device), y.to(args.device)
                pred = model(x)
                
                if y.dim() == 4 and y.shape[-1] == 1: y = y.squeeze(-1)
                if pred.dim() == 4 and pred.shape[-1] == 1: pred = pred.squeeze(-1)
                if pred.shape != y.shape: pred = pred.reshape_as(y)
                
                # 【约束 3】仅在最终 Test Set 评估时，执行反归一化 (inverse_transform)
                y_real = scaler.inverse_transform(y).cpu().numpy()
                pred_real = scaler.inverse_transform(pred).cpu().numpy()
                
                test_mae += np.sum(np.abs(y_real - pred_real))
                test_rmse += np.sum((y_real - pred_real)**2)
                test_elements += y_real.size
                
                mask = y_real > 0.5
                if np.sum(mask) > 0:
                    test_mape += np.sum(np.abs(y_real[mask] - pred_real[mask]) / y_real[mask])
                    valid_mape_count += np.sum(mask)
                    
        acc_mae = test_mae / test_elements
        acc_rmse = math.sqrt(test_rmse / test_elements)
        acc_mse = acc_rmse**2
        acc_mape = (test_mape / valid_mape_count * 100) if valid_mape_count > 0 else 0.0
        eff_test_time = time.time() - test_start
        
        # 【约束 1 & 4】统一通信量口径：按实际轮数 actual_epochs 计算 (上传 + 下发 乘以 2)
        comm_params = sum(p.numel() for p in model.parameters())
        if args.protection == 'he':
            eff_comm_mb = round((he_upload_bytes + he_download_bytes) / (1024 * 1024), 4)
        else:
            eff_comm_mb = round((comm_params * 4 * 2 * actual_epochs) / (1024 * 1024), 4)
        
        # 【约束 1】计算并记录 FLOPs
        eff_flops = 0.0
        try:
            from thop import profile
            dummy_x, _ = next(iter(val_loader))
            flops, _ = profile(model, inputs=(dummy_x.to(args.device),), verbose=False)
            eff_flops = round(flops / 1e9, 4)
        except Exception as e:
            print(f"Rank {ctx.rank}: FLOPs 计算失败: {e}")
            
        # 【约束 3】写入 CSV 的全部是物理反归一化后的数据
        log_experiment_results(
            model_name=args.model,
            dataset_client=f"{args.dataset_name}_client{ctx.rank}",
            feature_type=args.feature_type,
            best_epoch=best_epoch + 1,
            acc_mae=round(acc_mae, 4),
            acc_mse=round(acc_mse, 4),
            acc_rmse=round(acc_rmse, 4),
            acc_mape=round(acc_mape, 4),
            eff_train_time=round(total_train_time, 2),
            eff_val_time=round(total_val_time / max(actual_epochs, 1), 4),
            eff_test_time=round(eff_test_time, 4),
            eff_comm_size_mb=eff_comm_mb,
            eff_train_round=actual_epochs,
            eff_flops=eff_flops,
            dp_noise=getattr(args, 'dp_noise', 0.0),
            protection=getattr(args, 'protection', 'plain'),
            dp_sigma=getattr(args, 'dp_sigma', ''),
            dp_clip_norm=getattr(args, 'dp_clip_norm', '')
        )
        
        print(f"🎉 Client {ctx.rank} FedGODE 测试完成，真实物理指标与严谨开销已成功落盘！")
        
        # Do not terminate the FATE launcher here.  Killing the parent process
        # makes a single experiment appear to finish, but aborts a shell script
        # before it can start the next dataset.  The arbiter loop has already
        # completed, so every rank can return naturally.
        if ctx.is_on_guest:
            print("[FedGODE] guest result saved; returning normally so the next dataset can start.", flush=True)

def _sfl_param_state(model):
    param_names = {name for name, _ in model.named_parameters()}
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if name in param_names
    }


def _sfl_shared_param_keys(state_dicts):
    if not state_dicts:
        return []
    shared = []
    first = state_dicts[0]
    for name, tensor in first.items():
        if all(name in state and state[name].shape == tensor.shape for state in state_dicts[1:]):
            shared.append(name)
    return shared


def _sfl_flatten_state(state_dict, keys, device):
    return torch.cat([state_dict[name].detach().to(device).reshape(-1) for name in keys], dim=0)


def _sfl_client_weights(device):
    node_counts = [len(nodes) for nodes in getattr(args, "nodes_per", []) if len(nodes) > 0]
    if not node_counts:
        node_counts = [1] * max(int(getattr(args, "num_clients", 1)), 1)
    weights = torch.tensor(node_counts, dtype=torch.float32, device=device)
    weights = weights / weights.sum().clamp_min(1.0)
    return weights


def _sfl_structure_aggregate(state_dicts, device):
    num_clients = len(state_dicts)
    if num_clients == 0:
        raise ValueError("SFL aggregation requires at least one client state.")
    keys = _sfl_shared_param_keys(state_dicts)
    if not keys:
        raise ValueError("SFL aggregation found no shared trainable parameters across clients.")

    stacked = torch.stack([_sfl_flatten_state(state, keys, device) for state in state_dicts], dim=0)
    if num_clients == 1:
        adjacency = torch.ones((1, 1), device=device)
        personalized = stacked.clone()
    else:
        normalized = F.normalize(stacked, p=2, dim=1)
        sim = torch.matmul(normalized, normalized.T)
        sim = 0.5 * (sim + sim.T)
        sim.fill_diagonal_(1.0)

        gamma = max(float(getattr(args, "sfl_gamma", 0.01) or 0.01), 1e-6)
        topk = int(getattr(args, "sfl_topk", 0) or 0)
        if topk <= 0:
            topk = min(max(2, int(math.ceil(math.sqrt(num_clients)))), num_clients)

        logits = sim / gamma
        if topk < num_clients:
            _, indices = torch.topk(sim, k=topk, dim=1)
            mask = torch.zeros_like(sim, dtype=torch.bool)
            mask.scatter_(1, indices, True)
            mask = torch.logical_or(mask, mask.T)
            logits = logits.masked_fill(~mask, float("-inf"))
        adjacency = torch.softmax(logits, dim=1)

        personalized = stacked.clone()
        for _ in range(max(int(getattr(args, "sfl_m_steps", 1) or 1), 1)):
            personalized = torch.matmul(adjacency, personalized)

    readout_weights = _sfl_client_weights(device)
    if readout_weights.numel() != personalized.shape[0]:
        readout_weights = torch.ones((personalized.shape[0],), device=device) / max(personalized.shape[0], 1)
    global_vec = torch.matmul(readout_weights.unsqueeze(0), personalized).squeeze(0)

    shapes = {name: state_dicts[0][name].shape for name in keys}
    numels = {name: state_dicts[0][name].numel() for name in keys}

    personalized_states = []
    for client_idx in range(num_clients):
        ptr = 0
        state = {}
        for name in keys:
            numel = numels[name]
            state[name] = personalized[client_idx, ptr:ptr + numel].view(shapes[name]).detach().cpu()
            ptr += numel
        personalized_states.append(state)

    ptr = 0
    global_state = {}
    for name in keys:
        numel = numels[name]
        global_state[name] = global_vec[ptr:ptr + numel].view(shapes[name]).detach().cpu()
        ptr += numel

    return global_state, personalized_states, adjacency.detach().cpu()


def _sfl_parameter_mse(model, reference_state):
    total_sq = None
    total_numel = 0
    for name, param in model.named_parameters():
        ref = reference_state.get(name)
        if ref is None or ref.shape != param.shape:
            continue
        diff = param - ref.detach().to(param.device)
        sq = diff.pow(2).sum()
        total_sq = sq if total_sq is None else total_sq + sq
        total_numel += diff.numel()
    if total_sq is None or total_numel <= 0:
        return torch.zeros((), device=next(model.parameters()).device)
    return total_sq / float(total_numel)


def _evaluate_sfl_loader(model, loader, scaler, device):
    model.eval()
    total_abs = 0.0
    total_sq = 0.0
    total_elements = 0
    total_mape = 0.0
    total_mape_elements = 0

    with torch.no_grad():
        for x_val, y_val in loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            pred_val = model(x_val)
            if isinstance(pred_val, tuple):
                pred_val = pred_val[0]
            if y_val.dim() == 4 and y_val.shape[-1] == 1:
                y_val = y_val.squeeze(-1)
            if pred_val.dim() == 4 and pred_val.shape[-1] == 1:
                pred_val = pred_val.squeeze(-1)
            if pred_val.shape != y_val.shape:
                if pred_val.dim() == 3 and y_val.dim() == 3 and pred_val.shape[1] == y_val.shape[2]:
                    pred_val = pred_val.transpose(1, 2).contiguous()
                else:
                    pred_val = pred_val.reshape_as(y_val)

            total_abs += torch.abs(pred_val - y_val).sum().item()
            total_sq += torch.square(pred_val - y_val).sum().item()
            total_elements += y_val.numel()

            y_real = scaler.inverse_transform(y_val).detach().cpu().numpy()
            pred_real = scaler.inverse_transform(pred_val).detach().cpu().numpy()
            diff_real = pred_real - y_real
            mask = y_real > 0.5
            if np.any(mask):
                total_mape += (np.abs(diff_real[mask]) / y_real[mask]).sum() * 100.0
                total_mape_elements += int(mask.sum())

    mse = total_sq / max(total_elements, 1)
    return {
        "normalized_mae": total_abs / max(total_elements, 1),
        "normalized_rmse": math.sqrt(mse),
        "client_mae": (total_abs / max(total_elements, 1)) * scaler.metrics_coef,
        "client_mse": mse * (scaler.metrics_coef ** 2),
        "client_rmse": math.sqrt(mse) * scaler.metrics_coef,
        "client_mape": (total_mape / total_mape_elements) if total_mape_elements > 0 else 0.0,
        "elements": int(total_elements),
        "abs_error_sum": float(total_abs * scaler.metrics_coef),
        "sq_error_sum": float(total_sq * (scaler.metrics_coef ** 2)),
        "mape_error_sum": float(total_mape),
        "mape_elements": int(total_mape_elements),
    }


def train_sfl(ctx):

    # SFL learns its client relation graph from each client's trainable
    # parameter state ``v``.  Under DP, the complete client upload is clipped
    # and noised before the Arbiter constructs that graph.  C is calibrated
    # once from the first-round client upload norms and then reused.
    sfl_dp_clip = (
        float(args.dp_clip_norm)
        if str(getattr(args, "protection", "plain")).lower() == "dp"
        and float(getattr(args, "dp_clip_norm", 0.0)) > 0
        else None
    )

    if ctx.is_on_arbiter:
        # ================= Server Logic (图结构推断与聚合中心) =================
        print("[SFL Server] Starting SFL aggregation with sparse structure learning...")

        for epoch in range(args.epochs):
            if (
                str(getattr(args, "protection", "plain")).lower() == "dp"
                and sfl_dp_clip is None
            ):
                norm_guest = float(ctx.guest.get(f"sfl_dp_model_norm_{epoch}"))
                norm_hosts = ctx.hosts.get(f"sfl_dp_model_norm_{epoch}")
                norm_hosts = norm_hosts if isinstance(norm_hosts, list) else [norm_hosts]
                sfl_dp_clip = float(np.quantile(
                    [norm_guest] + [float(value) for value in norm_hosts], 0.9
                ))
                print(
                    f"[DPCalibration] SFL arbiter epoch={epoch + 1} "
                    f"model_state_clip={sfl_dp_clip:.8f}", flush=True,
                )
                ctx.guest.put(f"sfl_dp_model_clip_{epoch}", sfl_dp_clip)
                host_clips = [sfl_dp_clip] * len(norm_hosts)
                ctx.hosts.put(
                    f"sfl_dp_model_clip_{epoch}",
                    host_clips if len(host_clips) > 1 else host_clips[0],
                )
            v_guest = unprotect_he_ttp_payload(args, ctx.guest.get(f"v_{epoch}"))
            v_hosts = ctx.hosts.get(f"v_{epoch}")
            if not isinstance(v_hosts, list): v_hosts = [v_hosts]
            v_hosts = [unprotect_he_ttp_payload(args, payload) for payload in v_hosts]
            all_v = [v_guest] + v_hosts
            global_w, all_u, adjacency = _sfl_structure_aggregate(all_v, args.device)
            if epoch % 10 == 0:
                print(f"[SFL Server] Epoch {epoch} learned adjacency:")
                print(adjacency.numpy().round(3))
            ctx.guest.put(f"w_{epoch}", global_w)
            ctx.guest.put(f"u_{epoch}", all_u[0])
            ctx.hosts.put(f"w_{epoch}", global_w)
            ctx.hosts.put(f"u_{epoch}", all_u[1:])
            continue
            
            # --- 1. 参数扁平化 (Flattening) ---
            valid_keys = []
            for k in all_v[0].keys():
                shape_0 = all_v[0][k].shape
                if all((k in all_v[j]) and (all_v[j][k].shape == shape_0) for j in range(num_clients)):
                    valid_keys.append(k)
            
            shapes = [all_v[0][k].shape for k in valid_keys]
            numels = [all_v[0][k].numel() for k in valid_keys]
            
            V_tensors = []
            for i in range(num_clients):
                vec = torch.cat([all_v[i][k].flatten().to(args.device) for k in valid_keys])
                V_tensors.append(vec)
            V = torch.stack(V_tensors)  # Shape: (num_clients, Total_Params)

            # --- 2. 纯正的结构自学习 (Structure Learning via Cosine Similarity) ---
            # 直接计算客户端模型参数的余弦相似度，无需反向传播
            V_norm = F.normalize(V, p=2, dim=1)
            sim_matrix = torch.matmul(V_norm, V_norm.T)
            
            # 使用 ReLU 过滤负相关，保留正相关的客户端关系，然后进行 Softmax 归一化
            A_logits = torch.relu(sim_matrix)
            A_mat = F.softmax(A_logits, dim=1)

            if epoch % 10 == 0:
                print(f"[SFL Server] Epoch {epoch} Learned Graph Topology:")
                print(A_mat.detach().cpu().numpy().round(3))

            # --- 3. 多跳图聚合 (Multi-step GCN) ---
            U = V.clone()
            for _ in range(m_steps):
                U = torch.matmul(A_mat, U)

            # 全局模型 W 为聚合后的均值
            W_vec = U.mean(dim=0)

            # --- 4. 还原为字典格式下发 ---
            all_u = []
            for i in range(num_clients):
                u_dict = {}
                ptr = 0
                for key, shape, numel in zip(valid_keys, shapes, numels):
                    u_dict[key] = U[i][ptr:ptr+numel].view(shape).cpu()
                    ptr += numel
                all_u.append(u_dict)

            global_w = {}
            ptr = 0
            for key, shape, numel in zip(valid_keys, shapes, numels):
                global_w[key] = W_vec[ptr:ptr+numel].view(shape).cpu()
                ptr += numel
                
            ctx.guest.put(f"w_{epoch}", global_w)
            ctx.guest.put(f"u_{epoch}", all_u[0]) 
            ctx.hosts.put(f"w_{epoch}", global_w)
            ctx.hosts.put(f"u_{epoch}", all_u[1:]) 
            
        print("[SFL Server] SFL training and graph learning done.")

    else:
        # ================= Client Logic (本地个性化训练中心) =================
        print(f"[SFL Client {ctx.rank}] Starting SFL training...")
        train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = get_setting(ctx)
        sfl_lambda = float(getattr(args, 'sfl_lambda', 0.1) or 0.1)
        
        for dset in [train_set, val_set, test_set]:
            if hasattr(dset, 'data') and isinstance(dset.data, np.ndarray):
                dset.data = torch.from_numpy(dset.data).to(args.device)
                
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

        best_norm_mae = float('inf')
        best_epoch = -1
        best_model_wts = None
        
        total_train_time, total_val_time = 0.0, 0.0
        actual_epochs = 0
        stopper = ExplicitEarlyStopper(ctx, args, patience=50, min_delta=1e-4)

        try:
            for epoch in range(args.epochs):
                lambda_t = sfl_lambda if epoch > 0 else 0.0
                if epoch > 0:
                    global_w_cpu = record_he_ttp_downlink(args, ctx.arbiter.get(f"w_{epoch-1}"))
                    u_data_cpu = record_he_ttp_downlink(args, ctx.arbiter.get(f"u_{epoch-1}"))
                    
                    personalized_u_cpu = u_data_cpu[ctx.rank - 1] if isinstance(u_data_cpu, list) else u_data_cpu
                    global_w = {k: v.to(args.device) for k, v in global_w_cpu.items()}
                    personalized_u = {k: v.to(args.device) for k, v in personalized_u_cpu.items()}
                else:
                    global_w, personalized_u = None, None

                # --- 本地双重正则化训练 ---
                model.train()
                epoch_loss = 0.0
                epoch_reg = 0.0
                train_start = time.time()
                
                for x, y in train_loader:
                    x, y = x.to(args.device), y.to(args.device)
                    optimizer.zero_grad()
                    outputs = model(x)
                    ypred = outputs[0] if isinstance(outputs, tuple) else outputs
                    capture_revised_quantized_prediction(ctx, args, f"sfl_prediction_{epoch}", prediction=ypred, model_state_dict=model.state_dict())
                    
                    if y.dim() == 4 and y.shape[-1] == 1: y = y.squeeze(-1)
                    if ypred.dim() == 4 and ypred.shape[-1] == 1: ypred = ypred.squeeze(-1)
                    if ypred.shape != y.shape:
                        ypred = ypred.transpose(1, 2) if ypred.dim() == 3 and y.dim() == 3 and ypred.shape[1] == y.shape[2] else ypred.reshape_as(y)
                    
                    base_loss = loss_func(ypred, y)
                    reg_loss = torch.zeros((), device=args.device)
                    
                    # 【核心重构：使用 MSE_loss 替换 sum，解决尺度爆炸问题】
                    if lambda_t > 0.0 and global_w is not None and personalized_u is not None:
                        reg_loss = lambda_t * (
                            _sfl_parameter_mse(model, global_w) +
                            _sfl_parameter_mse(model, personalized_u)
                        )
                    
                    loss = base_loss + reg_loss
                    loss.backward()
                    
                    # 加上梯度裁剪作为安全带
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    epoch_reg += reg_loss.item()
                    
                total_train_time += (time.time() - train_start)
                print(
                    f"[SFL Client {ctx.rank}] Epoch {epoch} Local Train Loss: {epoch_loss/len(train_loader):.4f} "
                    f"| Reg: {epoch_reg/len(train_loader):.4f} | lambda={lambda_t:.4f}"
                )
                
                # --- 验证早停（严谨的归一化指标） ---
                model.eval()
                val_start = time.time()
                val_mae, val_rmse = 0.0, 0.0
                total_val_elements = 0
                
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        x_val, y_val = x_val.to(args.device), y_val.to(args.device)
                        outputs = model(x_val)
                        pred_val = outputs[0] if isinstance(outputs, tuple) else outputs
                        
                        if y_val.dim() == 4 and y_val.shape[-1] == 1: y_val = y_val.squeeze(-1)
                        if pred_val.dim() == 4 and pred_val.shape[-1] == 1: pred_val = pred_val.squeeze(-1)
                        if pred_val.shape != y_val.shape:
                            pred_val = pred_val.transpose(1, 2) if pred_val.dim() == 3 and y_val.dim() == 3 and pred_val.shape[1] == y_val.shape[2] else pred_val.reshape_as(y_val)
                        
                        y_cpu, pred_cpu = y_val.cpu().numpy(), pred_val.cpu().numpy()
                        val_mae += np.sum(np.abs(y_cpu - pred_cpu))
                        val_rmse += np.sum((y_cpu - pred_cpu) ** 2)
                        total_val_elements += y_val.numel()

                norm_val_mae = val_mae / total_val_elements
                norm_val_rmse = math.sqrt(val_rmse / total_val_elements)
                total_val_time += (time.time() - val_start)
                
                print(f"   ---> [SFL 验证结果] Client {ctx.rank} Epoch {epoch} | Norm MAE: {norm_val_mae:.4f} | Norm RMSE: {norm_val_rmse:.4f}")

                if norm_val_mae < best_norm_mae:
                    best_norm_mae = norm_val_mae
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                
                if stopper.check_and_sync(norm_val_mae):
                    print(f"Rank {ctx.rank}: 🛑 收到全局早停信号，抛出中断异常！")
                    raise EarlyStopSignal("触发早停机制")

                local_v = _sfl_param_state(model)
                if (
                    str(getattr(args, "protection", "plain")).lower() == "dp"
                    and sfl_dp_clip is None
                ):
                    ctx.arbiter.put(
                        f"sfl_dp_model_norm_{epoch}",
                        float(l2_norm(local_v).item()),
                    )
                    calibrated_clip = ctx.arbiter.get(f"sfl_dp_model_clip_{epoch}")
                    if isinstance(calibrated_clip, list):
                        calibrated_clip = (
                            calibrated_clip[ctx.rank - 1]
                            if len(calibrated_clip) > 1 else calibrated_clip[0]
                        )
                    sfl_dp_clip = float(calibrated_clip)
                    args.dp_clip_norm = sfl_dp_clip
                    print(
                        f"[DPCalibration] SFL rank={ctx.rank} epoch={epoch + 1} "
                        f"model_state_clip={sfl_dp_clip:.8f}", flush=True,
                    )
                if str(getattr(args, "protection", "plain")).lower() == "dp":
                    protected_arbiter_put(
                        ctx, args, f"v_{epoch}", local_v,
                        clip_norm=sfl_dp_clip,
                    )
                elif str(getattr(args, "protection", "plain")).lower() == "he":
                    protected_arbiter_put(ctx, args, f"v_{epoch}", local_v)
                else:
                    ctx.arbiter.put(f"v_{epoch}", local_v)
                actual_epochs = epoch + 1
                
        except EarlyStopSignal:
            actual_epochs = epoch + 1
            print(f"Rank {ctx.rank}: 🎉 成功穿透黑盒！准备进行最终 Test(反归一化) 并写入 CSV...")

        # --- 最终评估与 CSV 写入 (反归一化) ---
        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)
            
        test_start = time.time()
        final_metrics = _evaluate_sfl_loader(model, test_loader, scaler, args.device)
        eff_test_time = round(time.time() - test_start, 4)
        
        sfl_comm_params = sum(p.numel() for p in model.parameters()) * 3
        eff_flops = 0.0
        try:
            from thop import profile
            dummy_x, _ = next(iter(val_loader))
            flops, _ = profile(model, inputs=(dummy_x.to(args.device),), verbose=False)
            eff_flops = round(flops / 1e9, 4)
        except Exception: pass

        log_experiment_results(
            model_name="SFL", 
            dataset_client=f"{args.dataset_name}_client{ctx.rank}", 
            feature_type=args.feature_type,
            best_epoch=best_epoch + 1,
            acc_mae=round(final_metrics["client_mae"], 4), 
            acc_mse=round(final_metrics["client_mse"], 4), 
            acc_rmse=round(final_metrics["client_rmse"], 4), 
            acc_mape=round(final_metrics["client_mape"], 4),
            eff_train_time=round(total_train_time, 2), 
            eff_val_time=round(total_val_time / actual_epochs, 4) if actual_epochs > 0 else 0.0,
            eff_test_time=eff_test_time, 
            eff_comm_size_mb=round((sfl_comm_params * 4 * actual_epochs) / (1024 * 1024), 4), 
            eff_train_round=actual_epochs, 
            eff_flops=eff_flops,
            acc_elements=final_metrics["elements"],
            acc_abs_error_sum=round(final_metrics["abs_error_sum"], 6),
            acc_sq_error_sum=round(final_metrics["sq_error_sum"], 6),
            acc_mape_error_sum=round(final_metrics["mape_error_sum"], 6),
            acc_mape_elements=final_metrics["mape_elements"],
            dp_noise=getattr(args, 'dp_noise', 0.0)
        )
        print(f"Client {ctx.rank} SFL metrics recorded with both client/sample MAE.", flush=True)
        if ctx.is_on_guest and not bool(getattr(args, "cross_device_multiplex", False)):
            import os
            import signal
            print("Guest 节点：SFL 数据已安全保存，准备清理后台 Arbiter 释放进程...", flush=True)
            time.sleep(2)
            os.kill(os.getppid(), signal.SIGTERM)
        return

def extract_ctx_data(ctx, data_list):
    """
    FATE 通信列表解包工具：
    处理 Server 发送给多个 Host 时的 List 索引解包问题
    """
    if isinstance(data_list, list):
        if len(data_list) == 1:
            return data_list[0]
        else:
            my_idx = ctx.rank - 1 # Host 从 rank 1 开始
            if 0 <= my_idx < len(data_list):
                return data_list[my_idx]
            return data_list[0] # 兜底
    return data_list

def train_fuels(ctx):
    import copy
    import math
    import time
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    fuels_is_he = str(getattr(args, "protection", "plain")).lower() == "he"

    def compute_jsd_distance(p, q, eps=1e-8):
        """
        更稳的 JSD 计算
        p, q: [B, dr] 概率分布
        """
        p = p.clamp_min(eps)
        q = q.clamp_min(eps)
        m = 0.5 * (p + q)

        kl_pm = (p * (p.log() - m.log())).sum(dim=-1).mean()
        kl_qm = (q * (q.log() - m.log())).sum(dim=-1).mean()
        return 0.5 * (kl_pm + kl_qm)

    def build_local_prototype(model, proto_loader, device, args):
        """
        稳健版 local prototype 构造：
        1) 只聚合满 batch，保证 [B, dr] 中 B 一致
        2) 不再使用 torch.stack(all_r_n)
        3) 若没有任何满 batch，则 fallback 到一个补齐 batch
        """
        model.eval()

        proto_sum = None
        proto_count = 0
        target_bs = args.batch_size

        def _pad_to_full_batch(x, target_bs):
            cur_bs = x.shape[0]
            if cur_bs == target_bs:
                return x
            if cur_bs > target_bs:
                return x[:target_bs]

            pad_num = target_bs - cur_bs
            # 重复最后一个样本补齐
            repeat_shape = [pad_num] + [1] * (x.dim() - 1)
            pad = x[-1:].repeat(*repeat_shape)
            return torch.cat([x, pad], dim=0)

        with torch.no_grad():
            for batch in proto_loader:
                if isinstance(batch, (list, tuple)):
                    x_proto = batch[0]
                else:
                    x_proto = batch

                # 只保留满 batch，避免 [64, dr] 和 [31, dr] 不能相加
                if x_proto.shape[0] != target_bs:
                    continue

                x_proto = x_proto.to(device)
                r_proto = model.encode(x_proto).detach()  # [B, dr]

                if proto_sum is None:
                    proto_sum = torch.zeros_like(r_proto)

                proto_sum += r_proto
                proto_count += 1

        # 如果一个满 batch 都没有，fallback
        if proto_count == 0:
            print(f"[FUELS Warning] No full batch found for prototype on rank {ctx.rank}. Using padded fallback batch.")

            dataset = proto_loader.dataset
            if len(dataset) == 0:
                raise RuntimeError("build_local_prototype failed: proto_loader.dataset is empty")

            fallback_loader = DataLoader(
                dataset,
                batch_size=min(target_bs, len(dataset)),
                shuffle=False,
                drop_last=False
            )

            with torch.no_grad():
                batch = next(iter(fallback_loader))
                if isinstance(batch, (list, tuple)):
                    x_proto = batch[0]
                else:
                    x_proto = batch

                x_proto = _pad_to_full_batch(x_proto, target_bs).to(device)
                r_proto = model.encode(x_proto).detach()  # [B, dr]

                proto_sum = r_proto
                proto_count = 1

        R_n = proto_sum / proto_count

        if hasattr(args, 'dp_noise') and args.dp_noise > 0:
            laplace_noise = torch.distributions.Laplace(
                torch.tensor(0.0, device=R_n.device),
                torch.tensor(args.dp_noise, device=R_n.device)
            ).sample(R_n.shape)
            R_n = R_n + laplace_noise

        return R_n

    if ctx.is_on_arbiter:
        print(f"[FUELS Server] 正在启动...")
        PR_dict = {}
        NR_dict = {}
        fuels_dp_clip = float(args.dp_clip_norm) if args.protection == "dp" and float(getattr(args, "dp_clip_norm", 0.0)) > 0 else None
    else:
        print(f"[FUELS Client {ctx.rank}] 正在启动...")

        train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = get_setting(ctx)

        # 训练流：保持 benchmark 统一协议
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True
        )

        # prototype 流：顺序读取 + 只保留满 batch
        proto_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True
        )

        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False
        )

        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False
        )

        steps = len(train_loader)
        ctx.arbiter.put("steps", steps)

        best_norm_mae = float('inf')
        best_epoch = -1
        best_model_wts = None
        total_train_time, total_val_time = 0.0, 0.0
        fuels_dp_clip = float(args.dp_clip_norm) if args.protection == "dp" and float(getattr(args, "dp_clip_norm", 0.0)) > 0 else None

    # ================= 步数同步 =================
    if ctx.is_on_arbiter:
        s_guest = ctx.guest.get("steps")
        s_hosts = ctx.hosts.get("steps")
        if not isinstance(s_hosts, list):
            s_hosts = [s_hosts]
        STEPS = min([s_guest] + s_hosts)
        print(f"[FUELS Server] 已同步步数: {STEPS} steps/epoch")
    else:
        STEPS = steps
        stopper = ExplicitEarlyStopper(ctx, args, patience=50, min_delta=1e-4)

    try:
        for epoch in range(args.epochs):

            if not ctx.is_on_arbiter:
                # =========================
                # Client side
                # =========================
                model.train()
                epoch_loss, epoch_intra, epoch_inter = 0.0, 0.0, 0.0
                epoch_start_time = time.time()

                PR_n, NR_n = None, None
                if epoch > 0:
                    pr_data = record_he_ttp_downlink(args, ctx.arbiter.get(f"pr_{epoch}"))
                    nr_data = record_he_ttp_downlink(args, ctx.arbiter.get(f"nr_{epoch}"))
                    PR_n = extract_ctx_data(ctx, pr_data).to(args.device)
                    NR_n = extract_ctx_data(ctx, nr_data).to(args.device)

                # [1] 本地训练
                for i, batch in enumerate(train_loader):
                    if i >= STEPS:
                        break

                    x, y = unpack_spatiotemporal_batch(batch)
                    x = x.to(args.device)
                    y = y.to(args.device)

                    # Revised-protocol ablation only: retain one public
                    # fixed-point prediction before its local optimizer step.
                    # The native FUELS prototype upload remains HE protected.
                    if i == 0:
                        trace_state = copy.deepcopy(model.state_dict())
                        with torch.no_grad():
                            trace_prediction = model(x)
                        capture_revised_quantized_prediction(
                            ctx,
                            args,
                            f"fuels_prediction_{epoch}",
                            prediction=trace_prediction,
                            model_state_dict=trace_state,
                        )

                    optimizer.zero_grad()

                    x_prime = model.make_augmented_view(x)

                    r_n = model.encode(x)
                    r_n_prime = model.encode(x_prime)
                    pred = model.decode(r_n)

                    pred, y = align_prediction_and_target(pred, y)

                    loss_pred = loss_func(pred, y)
                    loss_intra = model.compute_intra_loss(
                        r_n, r_n_prime, tau=args.fuels_tau
                    )

                    loss_inter = torch.tensor(0.0, device=args.device)
                    if PR_n is not None and NR_n is not None:
                        loss_inter = model.compute_inter_loss(
                            r_n, PR_n, NR_n, tau=args.fuels_tau
                        )

                    loss = (
                        loss_pred
                        + (getattr(args, "fuels_intra_weight", 1.0) * loss_intra)
                        + (args.fuels_rho * loss_inter)
                    )
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss_pred.item()
                    epoch_intra += loss_intra.item()
                    epoch_inter += loss_inter.item()

                # [2] 单独生成 prototype
                R_n = build_local_prototype(
                    model=model,
                    proto_loader=proto_loader,
                    device=args.device,
                    args=args
                )
                if fuels_is_he:
                    protected_arbiter_put(ctx, args, f"R_{epoch}", R_n)
                elif args.protection == "dp":
                    ctx.arbiter.put(f"fuels_dp_proto_norm_{epoch}", float(l2_norm(R_n).item()) if fuels_dp_clip is None else None)
                    if fuels_dp_clip is None:
                        fuels_dp_clip = ctx.arbiter.get(f"fuels_dp_proto_clip_{epoch}")
                        if isinstance(fuels_dp_clip, (list, tuple)): fuels_dp_clip = fuels_dp_clip[0]
                        fuels_dp_clip = float(fuels_dp_clip)
                    _, observed_R_n, dp_info = protected_arbiter_put(
                        ctx, args, f"R_{epoch}", R_n, clip_norm=fuels_dp_clip,
                        return_protected_payload=True,
                    )
                    capture_dp_reconstruction_trace(
                        ctx, args, f"R_{epoch}", model_state=model.state_dict(),
                        observed_leak={"prototype": observed_R_n}, dp_info=dp_info,
                    )
                else:
                    ctx.arbiter.put(f"R_{epoch}", R_n)

                epoch_train_time = time.time() - epoch_start_time
                total_train_time += epoch_train_time

                print(
                    f"Client {ctx.rank} Epoch {epoch} 完毕 | "
                    f"Pred Loss: {epoch_loss / max(STEPS, 1):.4f} | "
                    f"Intra: {epoch_intra / max(STEPS, 1):.4f} | "
                    f"Inter: {epoch_inter / max(STEPS, 1):.4f}"
                )

                # [3] 验证
                model.eval()
                val_start_time = time.time()
                val_mae, val_rmse, val_elements = 0.0, 0.0, 0

                with torch.no_grad():
                    for batch in val_loader:
                        x_val, y_val = unpack_spatiotemporal_batch(batch)
                        x_val = x_val.to(args.device)
                        y_val = y_val.to(args.device)

                        r_n_val = model.encode(x_val)
                        pred_val = model.decode(r_n_val)
                        pred_val, y_val = align_prediction_and_target(pred_val, y_val)

                        y_cpu = y_val.cpu().numpy()
                        pred_cpu = pred_val.cpu().numpy()

                        val_mae += np.sum(np.abs(y_cpu - pred_cpu))
                        val_rmse += np.sum((y_cpu - pred_cpu) ** 2)
                        val_elements += y_val.numel()

                norm_val_mae = val_mae / max(val_elements, 1)
                norm_val_rmse = math.sqrt(val_rmse / max(val_elements, 1))

                epoch_val_time = time.time() - val_start_time
                total_val_time += epoch_val_time

                print(
                    f"   ---> [验证集结果] Client {ctx.rank} | "
                    f"Norm MAE: {norm_val_mae:.4f} | Norm RMSE: {norm_val_rmse:.4f}"
                )

                if norm_val_mae < best_norm_mae:
                    best_norm_mae = norm_val_mae
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())

                should_stop = stopper.check_and_sync(norm_val_mae)
                if should_stop:
                    print(f"Rank {ctx.rank}: 收到全局早停信号，停止训练。")
                    raise EarlyStopSignal("FUELS early stopping triggered")

            else:
                # =========================
                # Server side
                # =========================
                if epoch > 0:
                    ctx.guest.put(f"pr_{epoch}", PR_dict[0])
                    ctx.guest.put(f"nr_{epoch}", NR_dict[0])

                    host_prs = [PR_dict[i] for i in range(1, args.num_clients)]
                    host_nrs = [NR_dict[i] for i in range(1, args.num_clients)]

                    if len(host_prs) == 1:
                        ctx.hosts.put(f"pr_{epoch}", host_prs[0])
                        ctx.hosts.put(f"nr_{epoch}", host_nrs[0])
                    else:
                        ctx.hosts.put(f"pr_{epoch}", host_prs)
                        ctx.hosts.put(f"nr_{epoch}", host_nrs)

                if args.protection == "dp" and fuels_dp_clip is None:
                    proto_norm_guest = ctx.guest.get(f"fuels_dp_proto_norm_{epoch}")
                    proto_norm_hosts = ctx.hosts.get(f"fuels_dp_proto_norm_{epoch}")
                    proto_norm_hosts = proto_norm_hosts if isinstance(proto_norm_hosts, list) else [proto_norm_hosts]
                    fuels_dp_clip = float(np.quantile([float(v) for v in [proto_norm_guest] + proto_norm_hosts], 0.9))
                    ctx.guest.put(f"fuels_dp_proto_clip_{epoch}", fuels_dp_clip)
                    ctx.hosts.put(f"fuels_dp_proto_clip_{epoch}", [fuels_dp_clip] * len(proto_norm_hosts))
                    print(f"[DPCalibration] FUELS target=prototype clip_norm={fuels_dp_clip:.8f}", flush=True)
                r_guest = unprotect_he_ttp_payload(args, ctx.guest.get(f"R_{epoch}"))
                r_hosts = ctx.hosts.get(f"R_{epoch}")
                r_hosts = r_hosts if isinstance(r_hosts, list) else [r_hosts]
                r_hosts = [unprotect_he_ttp_payload(args, item) for item in r_hosts]

                all_R_n = [r_guest] + r_hosts
                N = len(all_R_n)

                probs = [F.softmax(r, dim=-1) for r in all_R_n]

                jsd_matrix = torch.zeros((N, N), device=probs[0].device)
                for i in range(N):
                    for j in range(i + 1, N):
                        jsd_val = compute_jsd_distance(probs[i], probs[j])
                        jsd_matrix[i, j] = jsd_val
                        jsd_matrix[j, i] = jsd_val

                active_jsds = jsd_matrix[jsd_matrix > 0]
                beta = (
                    torch.quantile(active_jsds, args.fuels_beta_percentile / 100.0)
                    if active_jsds.numel() > 0
                    else torch.tensor(0.0, device=jsd_matrix.device)
                )

                for i in range(N):
                    pos_protos, neg_protos = [], []

                    for j in range(N):
                        if i == j:
                            pos_protos.append(all_R_n[j])
                            continue

                        if jsd_matrix[i, j] <= beta:
                            pos_protos.append(all_R_n[j])
                        else:
                            neg_protos.append(all_R_n[j])

                    PR_dict[i] = torch.stack(pos_protos, dim=0).mean(dim=0)

                    if len(neg_protos) > 0:
                        NR_dict[i] = torch.stack(neg_protos, dim=0).mean(dim=0)
                    else:
                        NR_dict[i] = torch.zeros_like(all_R_n[i])

                print(f"[FUELS Server] Epoch {epoch} | beta={beta.item():.6f}")

    except EarlyStopSignal:
        print(f"Rank {ctx.rank}: 训练结束，进入测试阶段。")

    # =========================
    # 测试与落盘
    # =========================
    if not ctx.is_on_arbiter:
        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)
            print(f"Rank {ctx.rank}: 已回滚至第 {best_epoch + 1} 轮最优权重")

        model.eval()
        test_start_time = time.time()

        total_abs_error_sum = 0.0
        total_sq_error_sum = 0.0
        total_mape_error_sum = 0.0
        valid_mape_count = 0  # 关键修复：MAPE 只除有效样本数

        with torch.no_grad():
            total_elements_test = 0
            for batch in test_loader:
                x_test, y_test = unpack_spatiotemporal_batch(batch)
                x_test = x_test.to(args.device)
                y_test = y_test.to(args.device)

                r_n_test = model.encode(x_test)
                pred_test = model.decode(r_n_test)
                pred_test, y_test = align_prediction_and_target(pred_test, y_test)

                y_real = scaler.inverse_transform(y_test).cpu().numpy()
                pred_real = scaler.inverse_transform(pred_test).cpu().numpy()
                diff = pred_real - y_real

                total_abs_error_sum += float(np.sum(np.abs(diff)))
                total_sq_error_sum += float(np.sum(diff ** 2))
                total_elements_test += int(y_real.size)

                mask = y_real > 0.5
                if np.sum(mask) > 0:
                    total_mape_error_sum += float(np.sum(np.abs(diff[mask]) / y_real[mask]))
                    valid_mape_count += np.sum(mask)

        acc_mae = total_abs_error_sum / max(total_elements_test, 1)
        acc_mse = total_sq_error_sum / max(total_elements_test, 1)
        acc_rmse = math.sqrt(acc_mse)
        acc_mape = (total_mape_error_sum / valid_mape_count) if valid_mape_count > 0 else 0.0

        eff_test_time = time.time() - test_start_time
        eff_train_time = total_train_time
        actual_epochs_ran = epoch + 1
        eff_val_time = total_val_time / max(actual_epochs_ran, 1)

        # 通信量：每轮 1 个上传 prototype + 2 个下发 prototype
        prototype_params_per_round = args.batch_size * getattr(args, 'fuels_dr', 128)
        comm_params_per_round = 3 * prototype_params_per_round
        eff_comm_size_mb = (comm_params_per_round * 4 * actual_epochs_ran) / (1024 * 1024)

        eff_flops = 0.0
        try:
            from thop import profile
            dummy_x, _ = unpack_spatiotemporal_batch(next(iter(DataLoader(val_set, batch_size=args.batch_size))))
            flops, _ = profile(model, inputs=(dummy_x.to(args.device),), verbose=False)
            eff_flops = round(flops / 1e9, 4)
        except Exception:
            pass

        dataset_client_name = f"{args.dataset_name}_client{ctx.rank}"

        log_experiment_results(
            model_name="FUELS",
            dataset_client=dataset_client_name,
            feature_type=args.feature_type,
            best_epoch=best_epoch + 1,
            acc_mae=round(acc_mae, 4),
            acc_mse=round(acc_mse, 4),
            acc_rmse=round(acc_rmse, 4),
            acc_mape=round(acc_mape, 4),
            eff_train_time=round(eff_train_time, 2),
            eff_val_time=round(eff_val_time, 4),
            eff_test_time=round(eff_test_time, 4),
            eff_comm_size_mb=round(eff_comm_size_mb, 4),
            eff_train_round=actual_epochs_ran,
            eff_flops=eff_flops,
            acc_elements=total_elements_test,
            acc_abs_error_sum=round(total_abs_error_sum, 4),
            acc_sq_error_sum=round(total_sq_error_sum, 4),
            acc_mape_error_sum=round(total_mape_error_sum, 4),
            acc_mape_elements=int(valid_mape_count),
            dp_noise=getattr(args, 'dp_noise', 0.0)
        )

        print(f"🎉 Client {ctx.rank} 实验指标落盘完成。")

        if ctx.is_on_guest:
            import os
            import signal
            print("Guest 节点：准备退出并清理后台进程...")
            time.sleep(2)
            os.kill(os.getppid(), signal.SIGTERM)
    
def train_fedstn(ctx):
    def _as_list(value):
        return value if isinstance(value, list) else [value]

    def _select_rank_payload(value):
        if isinstance(value, list):
            if ctx.rank > 0 and len(value) > 1:
                return value[ctx.rank - 1]
            return value[0]
        return value

    def _aggregate_fedstn_hidden(all_hs):
        pooled_hs = [h.mean(dim=1) for h in all_hs]
        aggregated_features = []
        for i in range(len(all_hs)):
            scores = []
            for j in range(len(all_hs)):
                score_ij = (pooled_hs[i] * pooled_hs[j]).sum(dim=-1, keepdim=True) / math.sqrt(args.hidden_dim)
                scores.append(score_ij)

            attention_weights = torch.nn.functional.softmax(torch.cat(scores, dim=1), dim=1)
            context_i = sum([attention_weights[:, j:j + 1] * pooled_hs[j] for j in range(len(all_hs))])
            N_i = all_hs[i].shape[1]
            expanded_context = context_i.unsqueeze(1).repeat(1, N_i, 1)
            aggregated_features.append((all_hs[i] + expanded_context).detach().contiguous())
        return aggregated_features

    def _server_aggregate_fedstn_step(tag):
        hs_guest = unprotect_he_ttp_payload(args, ctx.guest.get(f"hs_{tag}"))
        hs_hosts = [
            unprotect_he_ttp_payload(args, item)
            for item in _as_list(ctx.hosts.get(f"hs_{tag}"))
        ]
        all_hs = [hs_guest] + hs_hosts
        aggregated_features = _aggregate_fedstn_hidden(all_hs)
        ctx.guest.put(f"agg_hs_{tag}", aggregated_features[0])
        ctx.hosts.put(
            f"agg_hs_{tag}",
            aggregated_features[1:] if len(aggregated_features) > 2 else aggregated_features[1]
        )

    # FedSTN exchanges per-batch GRU hidden states, rather than a FedAvg
    # model update.  Therefore DP protects this client -> Arbiter ``hs``
    # payload explicitly.  A single C is calibrated from the first training
    # batch and then reused for train/validation/test hidden-state uploads.
    fedstn_dp_clip = (
        float(args.dp_clip_norm)
        if str(getattr(args, "protection", "plain")).lower() == "dp"
        and float(getattr(args, "dp_clip_norm", 0.0)) > 0
        else None
    )

    def _put_fedstn_hidden(tag, hidden_state):
        if str(getattr(args, "protection", "plain")).lower() == "he":
            return protected_arbiter_put(ctx, args, f"hs_{tag}", hidden_state)
        if str(getattr(args, "protection", "plain")).lower() == "dp":
            if fedstn_dp_clip is None or fedstn_dp_clip <= 0:
                raise RuntimeError(
                    "FedSTN DP hidden-state clip norm was not calibrated before upload."
                )
            return protected_arbiter_put(
                ctx, args, f"hs_{tag}", hidden_state,
                clip_norm=fedstn_dp_clip,
            )
        return ctx.arbiter.put(f"hs_{tag}", hidden_state)

    def _fedstn_checkpoint_dir():
        ckpt_dir = getattr(args, "fedstn_checkpoint_dir", "") or os.path.join(
            file_dir,
            "checkpoints",
            "FedSTN",
            str(args.dataset_name),
            str(args.feature_type),
            f"seed{args.seed}",
        )
        os.makedirs(ckpt_dir, exist_ok=True)
        return ckpt_dir

    def _fedstn_checkpoint_path(rank):
        return os.path.join(_fedstn_checkpoint_dir(), f"client{rank}_best.pt")

    def _save_fedstn_checkpoint(model, epoch, global_val_mae, local_val_mae):
        ckpt_path = _fedstn_checkpoint_path(ctx.rank)
        tmp_path = ckpt_path + ".tmp"
        payload = {
            "model_state": {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
            "epoch": int(epoch),
            "global_val_mae": float(global_val_mae),
            "local_val_mae": float(local_val_mae),
            "dataset_name": args.dataset_name,
            "feature_type": args.feature_type,
            "seed": int(args.seed),
            "rank": int(ctx.rank),
        }
        torch.save(payload, tmp_path)
        os.replace(tmp_path, ckpt_path)
        print(f"Rank {ctx.rank}: [FedSTN checkpoint] saved {ckpt_path} at epoch {epoch + 1}")

    def _try_resume_fedstn_checkpoint(model):
        if not getattr(args, "fedstn_resume", False):
            return None
        ckpt_path = _fedstn_checkpoint_path(ctx.rank)
        if not os.path.exists(ckpt_path):
            print(f"Rank {ctx.rank}: [FedSTN checkpoint] resume requested but not found: {ckpt_path}")
            return None
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        state = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state, strict=False)
        print(
            f"Rank {ctx.rank}: [FedSTN checkpoint] resumed {ckpt_path} "
            f"(epoch={checkpoint.get('epoch', 'unknown')}, global_val_mae={checkpoint.get('global_val_mae', 'unknown')})"
        )
        return checkpoint

    if ctx.is_on_arbiter:
        print("[FedSTN Server] 启动 FedGAT 服务端注意力聚合中心...")
        STEPS_PER_EPOCH = 0
    else:
        print(f"[FedSTN Client {ctx.rank}] 启动...")
        train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = get_setting(ctx)
        resume_checkpoint = _try_resume_fedstn_checkpoint(model)
        
        if hasattr(train_set, 'data') and isinstance(train_set.data, np.ndarray):
            train_set.data = torch.from_numpy(train_set.data).to(args.device)
        if hasattr(val_set, 'data') and isinstance(val_set.data, np.ndarray):
            val_set.data = torch.from_numpy(val_set.data).to(args.device)
        if hasattr(test_set, 'data') and isinstance(test_set.data, np.ndarray):
            test_set.data = torch.from_numpy(test_set.data).to(args.device)
            
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
       
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        
        STEPS_PER_EPOCH = len(train_loader)
        ctx.arbiter.put("steps", STEPS_PER_EPOCH)
        ctx.arbiter.put("steps_val", len(val_loader))
        ctx.arbiter.put("steps_test", len(test_loader))
        
        best_mae, best_rmse, best_mape, best_epoch = float('inf'), float('inf'), float('inf'), -1
        if resume_checkpoint is not None:
            best_epoch = int(resume_checkpoint.get("epoch", -1))
            best_mae = float(resume_checkpoint.get("global_val_mae", float('inf')))
            best_model_wts = {k: v.cpu().contiguous().clone() for k, v in model.state_dict().items()}
        start_time = time.time()
        total_val_time = 0.0

    # 步数同步
    if ctx.is_on_arbiter:
        s_guest = ctx.guest.get("steps")
        s_hosts = ctx.hosts.get("steps")
        if not isinstance(s_hosts, list): s_hosts = [s_hosts]
        STEPS_PER_EPOCH = min([s_guest] + s_hosts)
        s_val_guest = ctx.guest.get("steps_val")
        s_val_hosts = _as_list(ctx.hosts.get("steps_val"))
        STEPS_VAL = min([s_val_guest] + s_val_hosts)
        s_test_guest = ctx.guest.get("steps_test")
        s_test_hosts = _as_list(ctx.hosts.get("steps_test"))
        STEPS_TEST = min([s_test_guest] + s_test_hosts)
        step_sync = {
            "train": int(STEPS_PER_EPOCH),
            "val": int(STEPS_VAL),
            "test": int(STEPS_TEST),
        }
        ctx.guest.put("fedstn_step_sync", step_sync)
        host_step_payload = [copy.deepcopy(step_sync) for _ in range(args.num_clients - 1)]
        ctx.hosts.put(
            "fedstn_step_sync",
            host_step_payload if len(host_step_payload) > 1 else host_step_payload[0]
        )
        fedstn_global_best = float('inf')
        fedstn_best_epoch = -1
        fedstn_patience_count = 0
    else:
        step_sync = ctx.arbiter.get("fedstn_step_sync")
        step_sync = _select_rank_payload(step_sync)
        STEPS_PER_EPOCH = int(step_sync["train"])
        STEPS_VAL = int(step_sync["val"])
        STEPS_TEST = int(step_sync["test"])

    if not ctx.is_on_arbiter:
        best_model_wts = locals().get("best_model_wts", None)
        actual_epochs = 0 

    try:
        for epoch in range(args.epochs):
            if not ctx.is_on_arbiter: # ======== Client 训练逻辑 ========
                model.train()
                total_loss = 0
                
                for i, batch in enumerate(train_loader):
                    if i >= STEPS_PER_EPOCH: break
                    
                    tag = f"e{epoch}_s{i}"
                    optimizer.zero_grad()
                    
                    if len(batch) == 5:
                        x_c, x_p, x_t, x_ext, y = batch
                        x_c, x_ext, y = x_c.to(args.device), x_ext.to(args.device), y.to(args.device)
                    else:
                        x_c, y = batch
                        x_c, y = x_c.to(args.device), y.to(args.device)
                        x_ext = None

                    h_s_i, rlcn_out, scn_out = model.forward_phase1(x_c, x_ext)
                    
                    # 【核心修复 1】：Client 发送 GRU 输出特征前，强制内存变连续！
                    hidden_upload = h_s_i.detach().cpu().contiguous()
                    if (
                        str(getattr(args, "protection", "plain")).lower() == "dp"
                        and fedstn_dp_clip is None
                    ):
                        ctx.arbiter.put(
                            f"fedstn_dp_hs_norm_{tag}",
                            float(l2_norm(hidden_upload).item()),
                        )
                        fedstn_dp_clip = float(_select_rank_payload(
                            ctx.arbiter.get(f"fedstn_dp_hs_clip_{tag}")
                        ))
                        args.dp_clip_norm = fedstn_dp_clip
                        print(
                            f"[DPCalibration] FedSTN rank={ctx.rank} tag={tag} "
                            f"hidden_state_clip={fedstn_dp_clip:.8f}", flush=True,
                        )
                    _put_fedstn_hidden(tag, hidden_upload)
                    
                    agg_hs_data = ctx.arbiter.get(f"agg_hs_{tag}")
                    if isinstance(agg_hs_data, list): 
                        agg_hs_data = agg_hs_data[ctx.rank - 1] if len(agg_hs_data) > 1 else agg_hs_data[0]
                    agg_hs_data = record_he_ttp_downlink(args, agg_hs_data, tag=tag)
                    
                    aggregated_h_s = agg_hs_data.to(args.device).requires_grad_()
                    pred = model.forward_phase2(aggregated_h_s, rlcn_out, scn_out)
                    capture_revised_quantized_prediction(ctx, args, f"fedstn_prediction_{tag}", prediction=pred, model_state_dict=model.state_dict())
                    
                    if y.dim() == 4 and y.shape[-1] == 1: y = y.squeeze(-1)
                    if pred.dim() == 4 and pred.shape[-1] == 1: pred = pred.squeeze(-1)
                    
                    # =================【核心修复：智能维度对齐】=================
                    if pred.shape != y.shape:
                        # 如果发现时间和节点维度错位 [Batch, N, T, F] vs [Batch, T, N, F]
                        if pred.dim() == 4 and y.dim() == 4 and pred.shape[1] == y.shape[2] and pred.shape[2] == y.shape[1]:
                            pred = pred.transpose(1, 2).contiguous() 
                        # 如果是单通道 3D 情况 [Batch, N, T] vs [Batch, T, N]
                        elif pred.dim() == 3 and y.dim() == 3 and pred.shape[1] == y.shape[2] and pred.shape[2] == y.shape[1]:
                            pred = pred.transpose(1, 2).contiguous()
                        else:
                            pred = pred.reshape_as(y) # 终极无脑保底
                    # ========================================================
                    
                    loss = loss_func(pred, y)
                    loss.backward()  # 上一次说的，去掉 retain_graph=True

                    torch.autograd.backward(h_s_i, aggregated_h_s.grad)
                    optimizer.step()
                    total_loss += loss.item()

                train_loss_avg = total_loss / max(STEPS_PER_EPOCH, 1)
                
                # ================= 评估逻辑 =================
                model.eval()
                val_start_t = time.time()
                val_loss, val_mae, val_rmse, val_mape = 0.0, 0.0, 0.0, 0.0
                total_elements, valid_mape_count = 0, 0
                
                with torch.no_grad():
                    for i, batch in enumerate(val_loader):
                        if i >= STEPS_VAL:
                            break
                        tag = f"val_e{epoch}_s{i}"
                        # 1. 动态解包 5 元组
                        if len(batch) == 5:
                            x_c, x_p, x_t, x_ext, y = batch
                            x_c, x_ext, y = x_c.to(args.device), x_ext.to(args.device), y.to(args.device)
                        else:
                            x_c, y = batch
                            x_c, y = x_c.to(args.device), y.to(args.device)
                            x_ext = None
                        
                        # 2. 前向传播
                        h_s, r_out, s_out = model.forward_phase1(x_c, x_ext)
                        _put_fedstn_hidden(tag, h_s.detach().cpu().contiguous())
                        agg_hs_data = _select_rank_payload(ctx.arbiter.get(f"agg_hs_{tag}"))
                        aggregated_h_s = record_he_ttp_downlink(args, agg_hs_data, tag=tag).to(args.device)
                        pred_val = model.forward_phase2(aggregated_h_s, r_out, s_out)
                        
                        # 3. 智能维度对齐
                        if y.dim() == 4 and y.shape[-1] == 1: y = y.squeeze(-1)
                        if pred_val.dim() == 4 and pred_val.shape[-1] == 1: pred_val = pred_val.squeeze(-1)
                        if pred_val.shape != y.shape:
                            if pred_val.dim() == 4 and y.dim() == 4 and pred_val.shape[1] == y.shape[2]:
                                pred_val = pred_val.transpose(1, 2).contiguous()
                            elif pred_val.dim() == 3 and y.dim() == 3 and pred_val.shape[1] == y.shape[2]:
                                pred_val = pred_val.transpose(1, 2).contiguous()
                            else:
                                pred_val = pred_val.reshape_as(y)
                                
                        loss = loss_func(pred_val, y)
                        val_loss += loss.item()
                        
                        # 直接使用归一化数据计算 MAE 和 RMSE (极速模式)
                        y_norm = y.cpu().numpy()
                        pred_norm = pred_val.cpu().numpy()

                        val_mae += np.sum(np.abs(y_norm - pred_norm))
                        val_rmse += np.sum((y_norm - pred_norm) ** 2)
                        total_elements += y_norm.size

                # 5. 汇总结算
                test_loss = val_loss / max(len(val_loader), 1)
                final_mae = val_mae / total_elements
                final_rmse = math.sqrt(val_rmse / total_elements)
                final_mape = 0.0 # 验证阶段不看 MAPE
                
                total_val_time += (time.time() - val_start_t)
                actual_epochs = epoch + 1 
                final_mae_float = float(final_mae)

                print(f"👉 [Client {ctx.rank}] Epoch {epoch} 总结 | Train Loss: {train_loss_avg:.4f} || Val MAE: {final_mae_float:.4f} | Val RMSE: {final_rmse:.4f} | Val MAPE: {final_mape:.4f}")

                ctx.arbiter.put(
                    f"fedstn_val_stats_{epoch}",
                    {"mae": final_mae_float, "elements": int(total_elements)}
                )
                sync_info = ctx.arbiter.get(f"fedstn_val_sync_{epoch}")
                sync_info = _select_rank_payload(sync_info)
                should_stop = bool(sync_info["should_stop"])
                best_epoch = int(sync_info["best_epoch"])

                if sync_info["is_best"]:
                    best_mae = float(sync_info["global_val_mae"])
                    best_rmse, best_mape = final_rmse, final_mape
                    best_model_wts = {k: v.cpu().contiguous().clone() for k, v in model.state_dict().items()}
                    _save_fedstn_checkpoint(model, epoch, best_mae, final_mae_float)

                if should_stop:
                    print(f"Rank {ctx.rank}: 🛑 收到全局早停信号，抛出中断异常！")
                    raise EarlyStopSignal("触发早停机制，切断框架死循环")
                
            else: # ======== Server 逻辑 (FedGAT 聚合) ========
                for step in range(STEPS_PER_EPOCH):
                    tag = f"e{epoch}_s{step}"
                    if (
                        str(getattr(args, "protection", "plain")).lower() == "dp"
                        and fedstn_dp_clip is None
                    ):
                        norm_guest = float(ctx.guest.get(f"fedstn_dp_hs_norm_{tag}"))
                        norm_hosts = _as_list(ctx.hosts.get(f"fedstn_dp_hs_norm_{tag}"))
                        fedstn_dp_clip = float(np.quantile(
                            [norm_guest] + [float(value) for value in norm_hosts], 0.9
                        ))
                        print(
                            f"[DPCalibration] FedSTN arbiter tag={tag} "
                            f"hidden_state_clip={fedstn_dp_clip:.8f}", flush=True,
                        )
                        ctx.guest.put(f"fedstn_dp_hs_clip_{tag}", fedstn_dp_clip)
                        host_clips = [fedstn_dp_clip] * len(norm_hosts)
                        ctx.hosts.put(
                            f"fedstn_dp_hs_clip_{tag}",
                            host_clips if len(host_clips) > 1 else host_clips[0],
                        )
                    hs_guest = unprotect_he_ttp_payload(args, ctx.guest.get(f"hs_{tag}"))
                    hs_hosts = ctx.hosts.get(f"hs_{tag}")
                    if not isinstance(hs_hosts, list): hs_hosts = [hs_hosts]
                    hs_hosts = [unprotect_he_ttp_payload(args, item) for item in hs_hosts]
                    all_hs = [hs_guest] + hs_hosts 
                    num_clients = len(all_hs)
                    
                    pooled_hs = [h.mean(dim=1) for h in all_hs] 
                    aggregated_features = []
                    for i in range(num_clients):
                        scores = []
                        for j in range(num_clients):
                            score_ij = (pooled_hs[i] * pooled_hs[j]).sum(dim=-1, keepdim=True) / math.sqrt(args.hidden_dim)
                            scores.append(score_ij)
                        
                        attention_weights = torch.nn.functional.softmax(torch.cat(scores, dim=1), dim=1) 
                        context_i = sum([attention_weights[:, j:j+1] * pooled_hs[j] for j in range(num_clients)]) 
                        N_i = all_hs[i].shape[1]
                        expanded_context = context_i.unsqueeze(1).repeat(1, N_i, 1)
                        
                        final_hs_i = all_hs[i] + expanded_context
                        # 【核心修复 3】：Server 准备下发的全局特征，强制压实！
                        aggregated_features.append(final_hs_i.detach().contiguous())
                    
                    ctx.guest.put(f"agg_hs_{tag}", aggregated_features[0])
                    ctx.hosts.put(f"agg_hs_{tag}", aggregated_features[1:] if len(aggregated_features)>2 else aggregated_features[1])

                for step in range(STEPS_VAL):
                    tag = f"val_e{epoch}_s{step}"
                    _server_aggregate_fedstn_step(tag)

                val_guest = ctx.guest.get(f"fedstn_val_stats_{epoch}")
                val_hosts = _as_list(ctx.hosts.get(f"fedstn_val_stats_{epoch}"))
                val_stats = [val_guest] + val_hosts
                weighted_abs = sum(float(item["mae"]) * int(item["elements"]) for item in val_stats)
                weighted_count = sum(int(item["elements"]) for item in val_stats)
                global_val_mae = weighted_abs / max(weighted_count, 1)

                if global_val_mae < fedstn_global_best - 1e-4:
                    fedstn_global_best = global_val_mae
                    fedstn_best_epoch = epoch
                    fedstn_patience_count = 0
                    is_best = True
                else:
                    fedstn_patience_count += 1
                    is_best = False

                should_stop = fedstn_patience_count >= 50
                print(
                    f"--- [FedSTN Server] Epoch {epoch + 1} | Global Val MAE: {global_val_mae:.4f} "
                    f"| Best: {fedstn_global_best:.4f} | Patience: {fedstn_patience_count}/50 ---"
                )
                sync_payload = {
                    "global_val_mae": float(global_val_mae),
                    "is_best": bool(is_best),
                    "should_stop": bool(should_stop),
                    "best_epoch": int(fedstn_best_epoch),
                }
                ctx.guest.put(f"fedstn_val_sync_{epoch}", sync_payload)
                host_payload = [copy.deepcopy(sync_payload) for _ in range(args.num_clients - 1)]
                ctx.hosts.put(
                    f"fedstn_val_sync_{epoch}",
                    host_payload if len(host_payload) > 1 else host_payload[0]
                )
                if should_stop:
                    break

    except EarlyStopSignal:
        if not ctx.is_on_arbiter:
            print(f"Rank {ctx.rank}: 🎉 成功穿透黑盒跳出训练循环 (实际运行 {actual_epochs} 轮)！")

    if ctx.is_on_arbiter:
        for step in range(STEPS_TEST):
            tag = f"test_s{step}"
            _server_aggregate_fedstn_step(tag)
        return

    # =================【训练结束，计算开销与记录】=================
    if not ctx.is_on_arbiter:
        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)
            # 保底内存整理
            for param in model.parameters():
                if not param.is_contiguous(): param.data = param.data.contiguous()
        
        # =================【新增：全量 Test 阶段 (真实验收)】=================
        print(f"Rank {ctx.rank}: 🚀 开始最终测试集评估 (反归一化)...")
        model.eval()
        synchronize_cuda_for_timing(args.device)
        test_start_t = time.perf_counter()
        test_mae, test_rmse, test_mape = 0.0, 0.0, 0.0
        test_elements, valid_test_mape_count = 0, 0
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader): # 注意这里跑的是 test_loader
                if i >= STEPS_TEST:
                    break
                tag = f"test_s{i}"
                if len(batch) == 5:
                    x_c, x_p, x_t, x_ext, y = batch
                    x_c, x_ext, y = x_c.to(args.device), x_ext.to(args.device), y.to(args.device)
                else:
                    x_c, y = batch
                    x_c, y = x_c.to(args.device), y.to(args.device)
                    x_ext = None
                
                h_s, r_out, s_out = model.forward_phase1(x_c, x_ext)
                _put_fedstn_hidden(tag, h_s.detach().cpu().contiguous())
                agg_hs_data = _select_rank_payload(ctx.arbiter.get(f"agg_hs_{tag}"))
                aggregated_h_s = record_he_ttp_downlink(args, agg_hs_data, tag=tag).to(args.device)
                pred_test = model.forward_phase2(aggregated_h_s, r_out, s_out)
                
                if y.dim() == 4 and y.shape[-1] == 1: y = y.squeeze(-1)
                if pred_test.dim() == 4 and pred_test.shape[-1] == 1: pred_test = pred_test.squeeze(-1)
                if pred_test.shape != y.shape:
                    if pred_test.dim() == 4 and y.dim() == 4 and pred_test.shape[1] == y.shape[2]:
                        pred_test = pred_test.transpose(1, 2).contiguous()
                    elif pred_test.dim() == 3 and y.dim() == 3 and pred_test.shape[1] == y.shape[2]:
                        pred_test = pred_test.transpose(1, 2).contiguous()
                    else:
                        pred_test = pred_test.reshape_as(y)

                # 👉 核心：仅在最后测试时，进行耗时的反归一化！
                y_real = scaler.inverse_transform(y).cpu().numpy()
                pred_real = scaler.inverse_transform(pred_test).cpu().numpy()
                
                test_mae += np.sum(np.abs(y_real - pred_real))
                test_rmse += np.sum((y_real - pred_real) ** 2)
                test_elements += y_real.size
                
                mask = y_real > 0.5
                if np.sum(mask) > 0:
                    test_mape += np.sum(np.abs(y_real[mask] - pred_real[mask]) / y_real[mask])
                    valid_test_mape_count += np.sum(mask)

        # 结算真实物理指标
        acc_mae = round(test_mae / test_elements, 4)
        acc_rmse = round(math.sqrt(test_rmse / test_elements), 4)
        acc_mse = round(acc_rmse ** 2, 4)
        acc_mape = round((test_mape / valid_test_mape_count) if valid_test_mape_count > 0 else 0.0, 4)
        synchronize_cuda_for_timing(args.device)
        eff_test_time = round(time.perf_counter() - test_start_t, 4)
        print(
            f"[TestTimingAudit] model=FedSTN rank={ctx.rank} "
            f"batches={min(len(test_loader), STEPS_TEST)} elements={test_elements} "
            f"seconds={eff_test_time:.6f}",
            flush=True,
        )
        # =============================================================

        # 减去测试花的时间，精准统计训练时长
        total_train_time = time.time() - start_time - eff_test_time 
        dataset_client_name = f"{args.dataset_name}_client{ctx.rank}"
        
        N_local = len(args.nodes_per[ctx.rank])
        params_per_tensor = args.batch_size * N_local * args.hidden_dim
        params_per_epoch = 3 * STEPS_PER_EPOCH * params_per_tensor
        eff_comm_size_mb = (params_per_epoch * 4 * actual_epochs) / (1024 * 1024)
        
        eff_train_time = round(total_train_time, 2)
        eff_val_time = round(total_val_time / max(actual_epochs, 1), 4)
        eff_comm = round(eff_comm_size_mb, 4)
        # ================= [新增：动态计算 FLOPs] =================
        eff_flops = 0.0
        try:
            from thop import profile
            
            # 1. 临时构造一个包裹器，将 Phase1 和 Phase2 缝合为标准 forward
            class FedSTN_Wrapper(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def forward(self, x_c, x_ext):
                    # 前向1
                    h_s, r_out, s_out = self.m.forward_phase1(x_c, x_ext)
                    # 模拟本地 Context (替代 Server 聚合)
                    local_context = h_s.mean(dim=1, keepdim=True).repeat(1, h_s.shape[1], 1)
                    pseudo_agg_hs = h_s + local_context
                    # 前向2
                    return self.m.forward_phase2(pseudo_agg_hs, r_out, s_out)

            wrapper = FedSTN_Wrapper(model).to(args.device)
            
            # 2. 抓取一个 Batch 的真实数据进行摸底测试
            batch = next(iter(val_loader))
            if len(batch) == 5:
                x_c, _, _, x_ext, _ = batch
                x_c, x_ext = x_c.to(args.device), x_ext.to(args.device)
                flops, _ = profile(wrapper, inputs=(x_c, x_ext), verbose=False)
            else:
                x_c, _ = batch
                x_c = x_c.to(args.device)
                flops, _ = profile(wrapper, inputs=(x_c, None), verbose=False)
                
            # 3. 换算成 G FLOPs (十亿次浮点运算)，保留4位小数
            eff_flops = round(flops / 1e9, 4)
            print(f"Rank {ctx.rank}: 成功计算 FLOPs: {eff_flops} G")
            
        except Exception as e:
            print(f"Rank {ctx.rank}: FLOPs 计算失败，已回退为 0.0。原因: {e}")
            eff_flops = 0.0
        # ========================================================    

        final_csv_string = f"{args.model},{dataset_client_name},{args.feature_type},{best_epoch + 1},{acc_mae},{acc_mse},{acc_rmse},{acc_mape},{eff_train_time},{eff_val_time},{eff_test_time},{eff_comm},{actual_epochs},{eff_flops}"
        print(f"\n📊 [最终物理指标] Model,Dataset_Client,Feature,Best_Epoch,Acc_MAE,Acc_MSE,Acc_RMSE,Acc_MAPE,Eff_TrainTime(s),Eff_ValTime(s),Eff_TestTime(s),Eff_CommSize(MB),Eff_TrainRound,Eff_FLOPs(G)")
        print(f"✅ {final_csv_string}\n")
        
        log_experiment_results(
            model_name=args.model, 
            dataset_client=dataset_client_name, 
            feature_type=args.feature_type,
            best_epoch=best_epoch + 1, 
            acc_mae=acc_mae, 
            acc_mse=acc_mse, 
            acc_rmse=acc_rmse, 
            acc_mape=acc_mape,
            eff_train_time=eff_train_time,
            eff_val_time=eff_val_time,
            eff_test_time=eff_test_time,
            eff_comm_size_mb=eff_comm,
            eff_train_round=actual_epochs,
            eff_flops=eff_flops,
            dp_noise=getattr(args, 'dp_noise', 0.0)     
        )
        
        if ctx.is_on_guest:
            print("Guest 节点：数据已安全保存，准备清理后台 Arbiter 释放进程...")
            time.sleep(2)
            os.kill(os.getppid(), signal.SIGTERM)

def _run_single_context(ctx: Context):
    _normalize_model_alias()
    # Populate args.nodes_per on arbiter too, because a few FATE trainers use
    # the partition before they call get_setting().
    _initialize_partition(ctx)
    # Covers every tensor-bearing client->arbiter message emitted by the
    # custom trainers below (updates, gradients, activations and prototypes).
    # Plain/HE paths are unchanged; HE is handled only by a real backend.
    # FedGODE has an explicit Paillier HE-SA path in train_fedgode().  Resolve
    # `auto` before runtime installation so the legacy HE-TTP proxy wrapper is
    # never selected for this custom training loop.
    if getattr(args, 'privacy_result_dir', ''):
        args.privacy_result_dir = str(Path(args.privacy_result_dir).expanduser().resolve())
        print(f"[PrivacyOutput] rank={ctx.rank} result_dir={args.privacy_result_dir}", flush=True)
    if (
        args.model in ('FedGODE', 'FCFedGCN', 'FCGCN', 'FedGRU', 'FedTSE', 'FedGTP', 'TDLR_SEDLR')
        or getattr(args, 'trainer_mode', '') == 'ufcl'
    ) and args.protection == 'he' and args.he_backend == 'auto':
        args.he_backend = 'he_sa'
    # FCFedGCN owns every tensor upload in its custom trainer through
    # ``protected_arbiter_put``.  Installing the generic DP Party.put wrapper
    # as well can wrap the same FATE proxy twice (and on some proxy versions
    # leaves the Arbiter waiting after all clients appear to have uploaded).
    # Keep one explicit DP boundary for this trainer; its calls still perform
    # clipping/noising/audit logging in privacy.runtime_protection.
    if (
        args.model == 'FCFedGCN'
        or getattr(args, 'trainer_mode', '') == 'ufcl'
    ) and args.protection == 'dp':
        print(
            f"[PrivacyRuntime] rank={ctx.rank} "
            f"{'FCFedGCN' if args.model == 'FCFedGCN' else 'UFCL'} owns its DP update path; "
            "generic Party.put wrapper is disabled to prevent applying DP twice.",
            flush=True,
        )
    else:
        install_runtime_protection(ctx, args)

    if getattr(args, 'trainer_mode', '') == 'ufcl':
        print(f"Running UFCL Framework on Rank {ctx.rank} with model {args.model}")
        # A zero-noise, fixed-C UFCL run is an equivalence diagnostic.  It must
        # execute the exact native FATE UFCL protocol used by Plain, rather than
        # the explicit DP/HE aggregation implementation below.  Otherwise a
        # difference at sigma=0 could be caused by two training loops instead
        # of privacy protection.
        ufcl_dp_protocol_audit = bool(getattr(args, 'ufcl_dp_protocol_audit', False))
        ufcl_native_delta_dp_audit = bool(getattr(args, 'ufcl_native_delta_dp_audit', False))
        ufcl_native_dp_equivalence = (
            args.protection == 'dp'
            and float(getattr(args, 'dp_sigma', 0.0)) == 0.0
            and float(getattr(args, 'dp_clip_norm', 0.0)) > 0.0
            and not ufcl_dp_protocol_audit
            and not ufcl_native_delta_dp_audit
        )
        # Native FATE delta-DP is the formal UFCL DP backend.  It preserves
        # the Plain UFCL trainer/teacher/replay lifecycle and changes only the
        # payload entering the original FATE model aggregation boundary.
        # The legacy explicit-loop audit remains available only when requested
        # explicitly for diagnosis.
        ufcl_native_delta_dp = (
            args.protection == 'dp'
            and not ufcl_native_dp_equivalence
            and not ufcl_dp_protocol_audit
        )
        if ufcl_native_delta_dp:
            from privacy.ufcl_native_delta_dp import install_ufcl_native_delta_dp
            install_ufcl_native_delta_dp(ctx, args)
            print(
                "[UFCL-NativeDP] using native FATE trainer with update-level delta DP"
                + (" (sigma=0 inactive-clip audit)." if ufcl_native_delta_dp_audit else "."),
                flush=True,
            )
        if ufcl_native_dp_equivalence:
            print(
                "[UFCL-DP Equivalence] sigma=0 with fixed C: using native Plain "
                "UFCL protocol; this run validates protocol equivalence only.",
                flush=True,
            )
        elif ufcl_dp_protocol_audit:
            print(
                "[UFCL-DP Protocol Audit] forcing the explicit delta-aggregation "
                "trainer with sigma=0; this checks DP-loop equivalence rather "
                "than the native Plain protocol.",
                flush=True,
            )
        if (
            (args.protection in ('dp', 'he') or bool(getattr(args, 'efficiency_audit_plain', False)))
            and not ufcl_native_dp_equivalence
            and not ufcl_native_delta_dp
        ):
            if ctx.is_on_arbiter:
                train_ufcl_dp_task(ctx, args, get_setting)
            else:
                (
                    best_epoch, acc_mae, acc_rmse, acc_mape,
                    eff_train_time, eff_val_time, eff_test_time,
                    comm_params, actual_epochs, eff_flops,
                    acc_elements, acc_abs_error_sum, acc_sq_error_sum,
                    acc_mape_error_sum, acc_mape_elements,
                ) = train_ufcl_dp_task(ctx, args, get_setting)
                log_experiment_results(
                    model_name="UFCL", dataset_client=f"{args.dataset_name}_client{ctx.rank}",
                    feature_type=args.feature_type, best_epoch=best_epoch + 1,
                    acc_mae=round(acc_mae, 4), acc_mse=round(acc_rmse ** 2, 4),
                    acc_rmse=round(acc_rmse, 4), acc_mape=round(acc_mape, 4),
                    eff_train_time=round(eff_train_time, 2), eff_val_time=round(eff_val_time, 4),
                    eff_test_time=round(eff_test_time, 4),
                    eff_comm_size_mb=round(
                        comm_params / (1024 * 1024)
                        if args.protection == 'he'
                        else (comm_params * 4 * 2 * actual_epochs) / (1024 * 1024),
                        4,
                    ),
                    eff_train_round=actual_epochs, eff_flops=eff_flops,
                    acc_elements=acc_elements, acc_abs_error_sum=round(acc_abs_error_sum, 4),
                    acc_sq_error_sum=round(acc_sq_error_sum, 4),
                    acc_mape_error_sum=round(acc_mape_error_sum, 4),
                    acc_mape_elements=acc_mape_elements,
                    dp_sigma=getattr(args, 'dp_sigma', ''), dp_clip_norm=getattr(args, 'dp_clip_norm', ''),
                )
                print(f"[UFCL-{args.protection.upper()}] client {ctx.rank} result saved.", flush=True)
            return
        completed = False
        try:
            if ctx.is_on_arbiter:
                train(ctx)
            else:
                setting = get_setting(ctx)
                train(ctx, *setting)
            completed = True
        finally:
            if completed and getattr(args, "force_exit_after_run", False):
                print(f"Rank {ctx.rank}: UFCL finished; force exiting to bypass Python semaphore cleanup hang.", flush=True)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(0)
        return
    
    if getattr(args, 'trainer_mode', '') == 'tistgnn_ic' and args.model == 'ISTGNN':
        print(f"Running T-ISTGNN(i-c) on Rank {ctx.rank}")
        from lib.tistgnn_ic_trainer import train_tistgnn_ic

        if ctx.is_on_arbiter:
            train_tistgnn_ic(ctx, args, setting=None)
        else:
            setting = get_setting(ctx)
            final_metrics = train_tistgnn_ic(ctx, args, setting=setting)

            if final_metrics is not None:
                log_experiment_results(
                    model_name=args.model,
                    dataset_client=f"{args.dataset_name}_client{ctx.rank}",
                    feature_type=args.feature_type,
                    best_epoch=final_metrics["best_epoch"],      # 这里现在记录的是性能最优轮次 [cite: 1]
            
                    # 准确度（严格来自测试集且已反归一化）
                    acc_mae=round(final_metrics["mae"], 4),
                    acc_mse=round(final_metrics["mse"], 4),
                    acc_rmse=round(final_metrics["rmse"], 4),
                    acc_mape=round(final_metrics["mape"], 4),
            
                    # 效率指标
                    eff_train_time=round(final_metrics["train_t"], 2),
                    eff_val_time=round(final_metrics["val_t"], 4), # 现在不再是 0.0 了
                    eff_test_time=round(final_metrics["test_t"], 4),
                    eff_comm_size_mb=round(final_metrics["comm_mb"], 4),
                    eff_train_round=final_metrics["actual_rounds"], # 实际跑的总轮次
                    eff_flops=final_metrics["flops"],
                    dp_noise=getattr(args, 'dp_noise', 0.0)
                )
                print(f"🎉 Client {ctx.rank} T-ISTGNN 实验完成！各项严格指标已成功写入 CSV 📊")
        return
    
    
    if args.model == 'REFOL':
        print(f"Running Authentic Distributed REFOL Mode on Rank {ctx.rank}")
        if bool(getattr(args, "refol_legacy_rank_init", False)):
            # The archived four-client runner did not reseed individual FATE
            # ranks here.  Preserve its process-level initialization exactly
            # for historical reproduction only.
            print(
                f"[REFOL HistoricalInit] rank={ctx.rank} preserving FATE process RNG state.",
                flush=True,
            )
        else:
            # Each FATE rank is a separate process.  Reset its RNG before
            # get_setting() creates the local GRU (and before the Arbiter
            # creates AttGCN) for deterministic modern experiments.
            init_seed(args.seed)
            print(
                f"[REFOL-Reproducibility] rank={ctx.rank} reset RNG seed={args.seed} "
                "before REFOL model construction.",
                flush=True,
            )
        if ctx.is_on_arbiter:
            train_refol_distributed(ctx, args, setting=None)
        else:
            # 调用你已经写好的 get_setting，加载当前 Client 专属的局部数据
            setting = get_setting(ctx)
            train_refol_distributed(ctx, args, setting=setting)
            
        return

    if args.model == 'FGNNEH':
        print(f"Running FGNNEH Mode on Rank {ctx.rank}")
        from lib.fgnneh_trainer import train_fgnneh_task
        
        if ctx.is_on_arbiter:
            train_fgnneh_task(ctx, args, get_setting)
        else:
            # 接收 10 个返回值
            (best_epoch, acc_mae, acc_rmse, acc_mape, 
             eff_train_time, eff_val_time, eff_test_time, 
             actual_epochs, eff_flops, total_comm_params) = train_fgnneh_task(ctx, args, get_setting)
            
            dataset_client_name = f"{args.dataset_name}_client{ctx.rank}"
            
            # 按实际运行轮数和参数量转换为 MB
            eff_comm_size_mb = round((total_comm_params * 4) / (1024 * 1024), 4)
            acc_mse = round(acc_rmse ** 2, 4)
            
            log_experiment_results(
                model_name=args.model, 
                dataset_client=dataset_client_name, 
                feature_type=args.feature_type,
                best_epoch=best_epoch + 1,  # 习惯上 epoch 从 1 开始展示
                
                # 测试集物理尺度准确度指标
                acc_mae=round(acc_mae, 4), 
                acc_mse=acc_mse, 
                acc_rmse=round(acc_rmse, 4), 
                acc_mape=round(acc_mape, 4),
                
                # 效率指标 (实际耗时、通信量、早停轮数、FLOPs)
                eff_train_time=round(eff_train_time, 2),
                eff_val_time=round(eff_val_time, 4),
                eff_test_time=round(eff_test_time, 4),
                eff_comm_size_mb=eff_comm_size_mb,
                eff_train_round=actual_epochs,
                eff_flops=eff_flops,
                
                dp_noise=getattr(args, 'dp_noise', 0.0)     
            )
            print(f"🎉 Client {ctx.rank} FGNNEH 实验完成！各项严格指标已成功写入 CSV 📊")
            
            # ================= 【核心：同归于尽按钮】 =================
            if ctx.is_on_guest and not bool(getattr(args, "privacy_trace_only", False)):
                # ``run`` already uses the module-level ``time`` import.
                # Do not bind ``time`` locally: that shadows it for every
                # later branch in this large dispatcher (including Fed4TP).
                import time as _fgnneh_time
                print("Guest 节点：FGNNEH 数据已安全保存，准备清理后台 Arbiter 释放进程...")
                time.sleep(2) # 停顿2秒，确保其他 Host 的 CSV 也彻底写完
                os.kill(os.getppid(), signal.SIGTERM) # 强制发送终止信号

        return

    if args.model == 'CNFGNN':
        print(f"Running CNFGNN Mode (AT + FedAvg) on Rank {ctx.rank}")
        from lib.cnfgnn_trainer import train_cnfgnn_task # 从新模块中引入
        
        if ctx.is_on_arbiter:
            train_cnfgnn_task(ctx, args)
        else:
            # 优雅接住 10 个返回值，统一在外部处理 IO
            (best_epoch, acc_mae, acc_rmse, acc_mape, 
             eff_train_time, eff_val_time, eff_test_time, 
             eff_comm_size_mb, actual_epochs, eff_flops) = train_cnfgnn_task(ctx, args)
            
            dataset_client_name = f"{args.dataset_name}_client{ctx.rank}"
            acc_mse = round(acc_rmse ** 2, 4)
            
            log_experiment_results(
                model_name=args.model, 
                dataset_client=dataset_client_name, 
                feature_type=args.feature_type,
                best_epoch=best_epoch + 1,
                
                # 准确度指标 (全部是真实物理尺度 Test 结果)
                acc_mae=round(acc_mae, 4), 
                acc_mse=acc_mse, 
                acc_rmse=round(acc_rmse, 4), 
                acc_mape=round(acc_mape, 4),
                
                # 效率指标 (实际耗时、通信量、早停轮数、FLOPs)
                eff_train_time=round(eff_train_time, 2),
                eff_val_time=round(eff_val_time, 4),
                eff_test_time=round(eff_test_time, 4),
                eff_comm_size_mb=eff_comm_size_mb,
                eff_train_round=actual_epochs,
                eff_flops=eff_flops,
                
                dp_noise=getattr(args, 'dp_noise', 0.0)     
            )
            print(f"🎉 Client {ctx.rank} CNFGNN 实验完成！各项严格指标已成功写入 CSV 📊")
            
            # 同归于尽清理机制
            if ctx.is_on_guest and not bool(getattr(args, "cross_device_multiplex", False)):
                time.sleep(2)
                os.kill(os.getppid(), signal.SIGTERM)
        return
        
    if args.model == 'FedMetro':
        print(f"Running FedMetro Mode on Rank {ctx.rank}")
        train_fedmetro_fixed(ctx)
        return
    
    if args.model == 'STFAM':
        print(f"Running STFAM Mode on Rank {ctx.rank}")
        if ctx.is_on_arbiter:
            train_stfam_task(ctx)
        else:
            (best_epoch, acc_mae, acc_rmse, acc_mape, 
             eff_train_time, eff_val_time, eff_test_time, 
             total_comm_bytes, actual_epochs, eff_flops) = train_stfam_task(ctx)

            dataset_client_name = f"{args.dataset_name}_client{ctx.rank}"
            acc_mse = round(acc_rmse ** 2, 4)
            # STFAM reports the actual conditional protocol payloads.  In
            # particular, a ``SKIP`` uploader has no extractor-state bytes.
            eff_comm_size_mb = round(total_comm_bytes / (1024 * 1024), 4)

            log_experiment_results(
                model_name=args.model, 
                dataset_client=dataset_client_name, 
                feature_type=args.feature_type,
                best_epoch=best_epoch + 1,  
                
                acc_mae=round(acc_mae, 4), 
                acc_mse=acc_mse, 
                acc_rmse=round(acc_rmse, 4), 
                acc_mape=round(acc_mape, 4),
                
                eff_train_time=round(eff_train_time, 2),
                eff_val_time=round(eff_val_time / max(actual_epochs, 1), 4),
                eff_test_time=round(eff_test_time, 4),
                eff_comm_size_mb=eff_comm_size_mb,
                eff_train_round=actual_epochs,
                eff_flops=round(eff_flops, 4),
                
                dp_noise=getattr(args, 'dp_noise', 0.0)     
            )
            print(f"🎉 Client {ctx.rank} STFAM 实验完成！真实物理测试指标已成功落盘 📊")

            if ctx.is_on_guest:
                # Keep ``time`` global within ``run``; see FGNNEH cleanup.
                import time as _stfam_time
                print("Guest 节点：STFAM 数据已安全保存，准备清理后台 Arbiter 释放进程...")
                time.sleep(2) 
                os.kill(os.getppid(), signal.SIGTERM)
        return

    if args.model == 'FedSTG':
        print(f"Running FedSTG Mode on Rank {ctx.rank}")
        train_fedstg(ctx)
        return

    if args.trainer_mode == 'sfl':
        print(f"Running SFL Mode on Rank {ctx.rank}")
        train_sfl(ctx)
        return

    if args.model == 'FUELS':
        print(f"Running FUELS Dual-Contrastive Learning Mode on Rank {ctx.rank}")
        train_fuels(ctx)
        return  

    if args.model == 'FedSTN':
        print(f"Running Authentic FedSTN Mode on Rank {ctx.rank}")
        train_fedstn(ctx)
        return

    if args.model == 'FedGTP':
        print(f"Running Authentic FedGTP Mode on Rank {ctx.rank}")
        train_fedgtp(ctx)
        return

    if args.model == 'ASTGAT':
        print(f"Running FedAGAT Masked Global-ID Mode on Rank {ctx.rank}")
        from lib.fedagat_masked_trainer import train_fedagat_masked_global_id
        train_fedagat_masked_global_id(ctx, args, get_setting, log_experiment_results)
        return

    if args.model == 'STGCN': 
        print(f"Running STAGCN-EC Edge Computing Mode on Rank {ctx.rank}")
        
        if ctx.is_on_arbiter:
            train_stagcn_ec_task(ctx, args, get_setting)
        else:
            (best_epoch, acc_mae, acc_rmse, acc_mape, eff_train_time,
             eff_val_time, eff_test_time, initialization_comm_bytes, eff_flops,
             actual_epochs) = train_stagcn_ec_task(ctx, args, get_setting)
            
            dataset_client_name = f"{args.dataset_name}_client{ctx.rank}"
            acc_mse = round(acc_rmse ** 2, 4) 
            # STAGCN-EC has a one-time neighbour-initialization exchange,
            # not a model synchronization in every local epoch.
            eff_comm_size_mb = round(initialization_comm_bytes / (1024 * 1024), 4)
            
            log_experiment_results(
                model_name="STAGCN-EC", 
                dataset_client=dataset_client_name, 
                feature_type=args.feature_type,
                best_epoch=best_epoch, 
                
                # 1. 准确度指标
                acc_mae=acc_mae, 
                acc_mse=acc_mse, 
                acc_rmse=acc_rmse, 
                acc_mape=acc_mape,
                
                # 2. 效率指标
                eff_train_time=eff_train_time, 
                eff_val_time=eff_val_time, 
                eff_test_time=eff_test_time, 
                eff_comm_size_mb=eff_comm_size_mb, 
                eff_train_round=actual_epochs,
                
                # 【修改】：把 0.0 替换成真实的 eff_flops
                eff_flops=eff_flops,
                
                dp_noise=getattr(args, 'dp_noise', 0.0)     
            )
            print(f"🎉 Client {ctx.rank} STAGCN-EC 实验完成！严谨的 Test 指标与拆分耗时已写入 CSV 📊")
        return

    if args.model == 'FCFedGCN':
        print(f"Running Authentic FC-FedGCN Mode on Rank {ctx.rank}")
        if ctx.is_on_arbiter:
            train_fcfedgcn_task(ctx, args, get_setting)
        else:
            # 1. 接收返回值（包含真实的 actual_epochs、eff_flops 和样本级误差和）
            (best_epoch, acc_mae, acc_rmse, acc_mape, 
             eff_train_time, eff_val_time, eff_test_time, 
             comm_params, actual_epochs, eff_flops,
             acc_elements, acc_abs_error_sum, acc_sq_error_sum,
             acc_mape_error_sum, acc_mape_elements) = train_fcfedgcn_task(ctx, args, get_setting)
            
            dataset_client_name = f"{args.dataset_name}_client{ctx.rank}"
            
            # 2. 【核心要求1】：按早停时的实际运行轮数(actual_epochs)计算精确通信量
            # FC-FedGCN 每一轮: Client 上传权重 (1) + Server 下发权重 (1) = 2 次通信
            if args.protection == 'he':
                eff_comm_size_mb = round(comm_params / (1024 * 1024), 4)
            else:
                eff_comm_size_mb = round((comm_params * 4 * 2 * actual_epochs) / (1024 * 1024), 4)
            acc_mse = round(acc_rmse ** 2, 4)
            
            log_experiment_results(
                model_name=args.model, 
                dataset_client=dataset_client_name, 
                feature_type=args.feature_type,
                best_epoch=best_epoch + 1,  # 习惯上 epoch 从 1 开始展示
                
                # 准确度指标 (全部是真实物理尺度 Test 结果)
                acc_mae=round(acc_mae, 4), 
                acc_mse=acc_mse, 
                acc_rmse=round(acc_rmse, 4), 
                acc_mape=round(acc_mape, 4),
                
                # 效率指标 (实际耗时、通信量、早停轮数、FLOPs)
                eff_train_time=round(eff_train_time, 2),
                eff_val_time=round(eff_val_time, 4),
                eff_test_time=round(eff_test_time, 4),
                eff_comm_size_mb=eff_comm_size_mb,
                eff_train_round=actual_epochs,
                eff_flops=eff_flops,
                acc_elements=acc_elements,
                acc_abs_error_sum=acc_abs_error_sum,
                acc_sq_error_sum=acc_sq_error_sum,
                acc_mape_error_sum=acc_mape_error_sum,
                acc_mape_elements=acc_mape_elements,
                
                dp_noise=getattr(args, 'dp_noise', 0.0)     
            )
            print(f"🎉 Client {ctx.rank} FCFedGCN 实验完成！各项严格指标已成功写入 CSV 📊")
            
            # 同归于尽清理机制
            if ctx.is_on_guest:
                print("[FCFedGCN] guest result saved; returning normally.", flush=True)
        return

    if (
        args.model in ('FCGCN', 'FedGRU')
        and (args.protection in ('dp', 'he') or bool(getattr(args, 'efficiency_audit_plain', False)))
        and getattr(args, 'trainer_mode', '') != 'fed4tp'
    ):
        print(f"Running explicit {args.model} {args.protection.upper()} Mode on Rank {ctx.rank}")
        if ctx.is_on_arbiter:
            train_fedavg_dp_task(ctx, args, get_setting)
        else:
            (
                best_epoch, acc_mae, acc_rmse, acc_mape,
                eff_train_time, eff_val_time, eff_test_time,
                comm_params, actual_epochs, eff_flops,
                acc_elements, acc_abs_error_sum, acc_sq_error_sum,
                acc_mape_error_sum, acc_mape_elements,
            ) = train_fedavg_dp_task(ctx, args, get_setting)
            acc_mse = round(acc_rmse ** 2, 4)
            eff_comm_size_mb = round(
                (comm_params / (1024 * 1024))
                if args.protection == 'he'
                else (comm_params * 4 * 2 * actual_epochs) / (1024 * 1024),
                4,
            )
            log_experiment_results(
                model_name=args.model,
                dataset_client=f"{args.dataset_name}_client{ctx.rank}",
                feature_type=args.feature_type,
                best_epoch=best_epoch + 1,
                acc_mae=round(acc_mae, 4),
                acc_mse=acc_mse,
                acc_rmse=round(acc_rmse, 4),
                acc_mape=round(acc_mape, 4),
                eff_train_time=round(eff_train_time, 2),
                eff_val_time=round(eff_val_time, 4),
                eff_test_time=round(eff_test_time, 4),
                eff_comm_size_mb=eff_comm_size_mb,
                eff_train_round=actual_epochs,
                eff_flops=eff_flops,
                acc_elements=acc_elements,
                acc_abs_error_sum=round(acc_abs_error_sum, 4),
                acc_sq_error_sum=round(acc_sq_error_sum, 4),
                acc_mape_error_sum=round(acc_mape_error_sum, 4),
                acc_mape_elements=acc_mape_elements,
            )
            print(f"[{args.model}-{args.protection.upper()}] client {ctx.rank} result saved.", flush=True)
        return

    if args.model == 'FedTPS':
        print(f"Running Authentic FedTPS Mode on Rank {ctx.rank}")
        from lib.fedtps_trainer import train_fedtps_task
        
        if ctx.is_on_arbiter:
            train_fedtps_task(ctx, args, get_setting)
        else:
            (best_epoch, acc_mae, acc_rmse, acc_mape, 
             eff_train_time, eff_val_time, eff_test_time, 
             comm_params, actual_epochs, eff_flops) = train_fedtps_task(ctx, args, get_setting)
            
            dataset_client_name = f"{args.dataset_name}_client{ctx.rank}"
            
            # 计算 MB：参数量 * 4字节 * 2(上传+下发) * 实际轮数
            eff_comm_size_mb = round((comm_params * 4 * 2 * actual_epochs) / (1024 * 1024), 4)
            acc_mse = round(acc_rmse ** 2, 4)
            
            log_experiment_results(
                model_name=args.model, 
                dataset_client=dataset_client_name, 
                feature_type=args.feature_type,
                best_epoch=best_epoch + 1, 
               
                acc_mae=round(acc_mae, 4), 
                acc_mse=acc_mse, 
                acc_rmse=round(acc_rmse, 4), 
                acc_mape=round(acc_mape, 4),
                
                eff_train_time=round(eff_train_time, 2),
                eff_val_time=round(eff_val_time, 4),
                eff_test_time=round(eff_test_time, 4),
                eff_comm_size_mb=eff_comm_size_mb,
                eff_train_round=actual_epochs,
                eff_flops=eff_flops,
                
                dp_noise=getattr(args, 'dp_noise', 0.0)     
            )
            print(f"🎉 Client {ctx.rank} FedTPS 实验完成！")
            
            if ctx.is_on_guest:
                print("[FedTPS] guest result saved; returning normally.", flush=True)
        return

    if args.model == 'pFedCTP':
        print(f"Running Authentic pFedCTP Mode on Rank {ctx.rank}")
        from lib.pfedctp_trainer import train_pfedctp

        if ctx.is_on_arbiter:
            train_pfedctp(ctx, args, get_setting)
        else:
            (
                best_epoch,
                acc_mae,
                acc_rmse,
                acc_mape,
                eff_train_time,
                eff_val_time,        # <--- 稳稳接住新增的验证耗时
                eff_test_time,
                comm_params_per_round,
                actual_stage1_epochs,
                total_actual_epochs,
                eff_flops,
                run_finetune,
                target_rank,
            ) = train_pfedctp(ctx, args, get_setting)

            if run_finetune:
                dataset_client_name = f"{args.dataset_name}_client{ctx.rank}"

                # 通信量严格按照 Stage 1 真实提交流轮数计算
                eff_comm_size_mb = round((comm_params_per_round * 4 * 2 * actual_stage1_epochs) / (1024 * 1024), 4)
                acc_mse = round(acc_rmse ** 2, 4) if math.isfinite(acc_rmse) else 0.0

                log_experiment_results(
                    model_name=args.model,
                    dataset_client=dataset_client_name,
                    feature_type=args.feature_type,
                    best_epoch=(best_epoch + 1) if best_epoch >= 0 else -1,

                    # 准确度 (全为 Test Set 物理指标)
                    acc_mae=round(acc_mae, 4),
                    acc_mse=acc_mse,
                    acc_rmse=round(acc_rmse, 4),
                    acc_mape=round(acc_mape, 4),

                    # 效率 (时间不再是 0.0 了)
                    eff_train_time=round(eff_train_time, 2),
                    eff_val_time=round(eff_val_time, 4),
                    eff_test_time=round(eff_test_time, 4),
                    eff_comm_size_mb=eff_comm_size_mb,
                    eff_train_round=total_actual_epochs,
                    eff_flops=eff_flops,

                    dp_noise=getattr(args, 'dp_noise', 0.0),
                )
                print(f"Client {ctx.rank} pFedCTP 实验完成！真实物理指标已成功写入 CSV")
            else:
                print(f"Client {ctx.rank} 不是目标客户端(target_rank={target_rank})，跳过 pFedCTP 结果落盘。")
            
            # 防死锁同步逻辑
            if ctx.is_on_guest:
                if not run_finetune:
                    print("Guest 节点：正在等待 Target Host 跑完微调并完成 CSV 落盘...")
                    _ = ctx.hosts.get("pfedctp_finished") # 阻塞等待

                print("Guest 节点：全员确认落袋为安，准备清理后台死锁的 Arbiter...")
                time.sleep(2)
                os.kill(os.getppid(), signal.SIGTERM)
            else:
                if run_finetune:
                    ctx.guest.put("pfedctp_finished", 1)
        return

    if args.model == 'TwoMGTCN':
        print(f"Running Authentic 2MGTCN Mode (FPASS + Domain Adapt) on Rank {ctx.rank}")
        if ctx.is_on_arbiter:
            train_twomgtcn_task(ctx, args, get_setting)
        else:
            # 1. 接收 9 个返回值（拿到真实的 actual_epochs）
            best_epoch, acc_mae, acc_rmse, acc_mape, eff_train_time, eff_val_time, eff_test_time, comm_params, actual_epochs = train_twomgtcn_task(ctx, args, get_setting)
            
            dataset_client_name = f"{args.dataset_name}_client{ctx.rank}"
            
            # 2. 【严格保证】：按实际运行轮数计算通信量 (每轮包含上传权重+下载聚合，故乘2)
            eff_comm_size_mb = round((comm_params * 4 * 2 * actual_epochs) / (1024 * 1024), 4)
            
            # 3. 【动态计算 FLOPs】：抓取一个真实 batch 计算算力开销
            eff_flops = 0.0
            try:
                from thop import profile
                train_set, val_set, test_set, model, optimizer, loss_func, _, _, _, scaler, _ = get_setting(ctx)
                val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
                batch = next(iter(val_loader))
                x_c, x_p, x_t, x_ext, _ = batch
                # 传入流量特征和外部环境特征计算
                flops, _ = profile(model, inputs=(x_c.to(args.device), x_ext.to(args.device)), verbose=False)
                eff_flops = round(flops / 1e9, 4)
            except Exception as e:
                print(f"Rank {ctx.rank}: FLOPs 计算失败，已回退为 0.0。原因: {e}")
            
            # 4. 【严谨落盘】：严格对齐 14 个字段，准确率指标均来自最终的 test_loader
            log_experiment_results(
                model_name=args.model, 
                dataset_client=dataset_client_name, 
                feature_type=args.feature_type,
                best_epoch=best_epoch, 
                acc_mae=round(acc_mae, 4), 
                acc_mse=round(acc_rmse**2, 4), 
                acc_rmse=round(acc_rmse, 4), 
                acc_mape=round(acc_mape, 4),
                eff_train_time=round(eff_train_time, 2), 
                eff_val_time=round(eff_val_time, 4), 
                eff_test_time=round(eff_test_time, 4), 
                eff_comm_size_mb=eff_comm_size_mb, 
                eff_train_round=actual_epochs,     # 绝对真实的训练轮数
                eff_flops=eff_flops,               # 真实算力开销
                dp_noise=getattr(args, 'dp_noise', 0.0)
            )
            print(f"🎉 Client {ctx.rank} 2MGTCN 实验完成！各项严格指标已成功写入 CSV 📊")
            
             
            if ctx.is_on_guest:
                time.sleep(2) 
                os.kill(os.getppid(), signal.SIGTERM)
        return

    if args.model == 'FedOSTC':
        print(f"Running node-client Cross-Device FedOSTC Mode on logical rank {ctx.rank}")
        if ctx.is_on_arbiter:
            train_fedostc_task(ctx, args, None)
        else:
            train_fedostc_task(ctx, args, get_setting)
        return

    if args.model == 'FedGODE':
        print(f"Running Authentic FedGODE Mode on Rank {ctx.rank}")
        train_fedgode(ctx)
        return

    if getattr(args, 'trainer_mode', '') == 'fed4tp':
        print(f"Running Authentic Fed4TP Mode on Rank {ctx.rank}")
        if ctx.is_on_arbiter:
            train_fed4tp_task(ctx, args, setting=None)
        else:
            setting = get_setting(ctx)
            best_epoch, acc_mae, acc_rmse, acc_mape, eff_train_time, eff_val_time, eff_test_time, comm_params, actual_epochs, eff_flops = train_fed4tp_task(ctx, args, setting)
            
            dataset_client_name = f"{args.dataset_name}_client{ctx.rank}"
            
            acc_mse = round(acc_rmse ** 2, 4)
            eff_comm_size_mb = round((comm_params * 4 * 2 * actual_epochs) / (1024 * 1024), 4)
            
            log_experiment_results(
                model_name="Fed4TP_" + args.model, 
                dataset_client=dataset_client_name, 
                feature_type=args.feature_type,
                best_epoch=best_epoch, 
                acc_mae=acc_mae, 
                acc_mse=acc_mse, 
                acc_rmse=acc_rmse, 
                acc_mape=acc_mape,
                eff_train_time=eff_train_time,
                eff_val_time=eff_val_time,
                eff_test_time=eff_test_time,
                eff_comm_size_mb=eff_comm_size_mb,
                eff_train_round=actual_epochs,
                eff_flops=eff_flops,
                dp_noise=getattr(args, 'dp_noise', 0.0)     
            )
            print(f"🎉 Client {ctx.rank} Fed4TP 实验完成！各项严格指标已成功写入 CSV 📊")
            
            if ctx.is_on_guest:
                # The Fed4TP arbiter consumes the synchronized stop flags and
                # returns normally.  Killing the launcher's parent process
                # here races its result collection and can turn a completed
                # experiment into a failed shell command.
                print("[Fed4TP] guest results saved; returning to the FATE launcher normally.")
        return

    if args.model == 'FedmSSA':
        if train_fedmssa_task is None:
            raise RuntimeError(
                "FedmSSA implementation is unavailable: "
                "lib/fedmssa_trainer.py is missing or does not define train_fedmssa_task."
            )
        print(f"Running Authentic Federated FedmSSA Mode on Rank {ctx.rank}")
        if ctx.is_on_arbiter:
            train_fedmssa_task(ctx, args, setting=None, get_setting_fn=None)
        else:
            setting = get_setting(ctx)
            (best_epoch, acc_mae, acc_rmse, acc_mape,
             acc_elements, acc_abs_error_sum, acc_sq_error_sum, acc_mape_error_sum, acc_mape_elements,
             eff_train_time, eff_val_time, eff_test_time,
             eff_comm_size_mb, actual_epochs, eff_flops) = train_fedmssa_task(
                ctx, args, setting=setting, get_setting_fn=get_setting
            )

            dataset_client_name = f"{args.dataset_name}_client{ctx.rank}"
            acc_mse = round(acc_rmse ** 2, 4)

            log_experiment_results(
                model_name="FedmSSA",
                dataset_client=dataset_client_name,
                feature_type=args.feature_type,
                best_epoch=best_epoch + 1,
                acc_mae=round(acc_mae, 4),
                acc_mse=acc_mse,
                acc_rmse=round(acc_rmse, 4),
                acc_mape=round(acc_mape, 4),
                eff_train_time=round(eff_train_time, 2),
                eff_val_time=round(eff_val_time, 4),
                eff_test_time=round(eff_test_time, 4),
                eff_comm_size_mb=eff_comm_size_mb,
                eff_train_round=actual_epochs,
                eff_flops=eff_flops,
                acc_elements=acc_elements,
                acc_abs_error_sum=round(acc_abs_error_sum, 4),
                acc_sq_error_sum=round(acc_sq_error_sum, 4),
                acc_mape_error_sum=round(acc_mape_error_sum, 4),
                acc_mape_elements=acc_mape_elements,
                dp_noise=getattr(args, 'dp_noise', 0.0)
            )
            print(f"Client {ctx.rank} FedmSSA test metrics saved.", flush=True)

            # Returning normally is required for DP batch scripts to advance
            # to the next dataset/model after FATE has cleaned up all ranks.
            if ctx.is_on_guest:
                print(f"[{args.model}] guest result saved; returning normally.", flush=True)
        return

    if args.model == 'FedTSE':
        print(f"Running Authentic FedTSE Mode (AsynWeight Baseline) on Rank {ctx.rank}")
        from lib.fedtse_trainer import train_fedtse_task
        
        if ctx.is_on_arbiter:
            train_fedtse_task(ctx, args, setting=None)
        else:
            setting = get_setting(ctx)
            
            # 【要求 3 满足】这里接住的 acc_mae, acc_rmse, acc_mape 百分之百是 Test 上的物理结果
            (best_epoch, acc_mae, acc_rmse, acc_mape,
             acc_elements, acc_abs_error_sum, acc_sq_error_sum, acc_mape_error_sum, acc_mape_elements,
             eff_train_time, eff_val_time, eff_test_time,
             eff_comm_size_mb, actual_epochs, eff_flops) = train_fedtse_task(ctx, args, setting)
            
            dataset_client_name = f"{args.dataset_name}_client{ctx.rank}"
            acc_mse = round(acc_rmse ** 2, 4)
            
            log_experiment_results(
                model_name="FedTSE", 
                dataset_client=dataset_client_name, 
                feature_type=args.feature_type,
                best_epoch=best_epoch + 1,
                
                # 【要求 3 满足】写入 CSV 的完全是 Test 物理结果
                acc_mae=round(acc_mae, 4), 
                acc_mse=acc_mse, 
                acc_rmse=round(acc_rmse, 4), 
                acc_mape=round(acc_mape, 4),
                
                # 【要求 1 满足】写入的是基于 early stopping 截断的真实轮数和通信量
                eff_train_time=round(eff_train_time, 2),
                eff_val_time=round(eff_val_time, 4),
                eff_test_time=round(eff_test_time, 4),
                eff_comm_size_mb=eff_comm_size_mb,
                eff_train_round=actual_epochs,
                eff_flops=eff_flops,
                acc_elements=acc_elements,
                acc_abs_error_sum=round(acc_abs_error_sum, 4),
                acc_sq_error_sum=round(acc_sq_error_sum, 4),
                acc_mape_error_sum=round(acc_mape_error_sum, 4),
                acc_mape_elements=acc_mape_elements,
                
                dp_noise=getattr(args, 'dp_noise', 0.0)     
            )
            print(f"🎉 Client {ctx.rank} FedTSE (AsynWeight) 测试完成！严格的 Benchmark 指标已落盘 📊")
            
            # Return normally: killing the launcher here would abort a
            # multi-dataset DP shell script before its next dataset starts.
            if ctx.is_on_guest:
                print("[FedTSE] guest result saved; returning normally.", flush=True)
        return

    if args.model in ('TDLR', 'SEDLR', 'TDLR_SEDLR'):
        print(f"Running Streaming Federated {args.model} Mode on Rank {ctx.rank}")
        if ctx.is_on_arbiter:
            train_tdlr_sedlr_task(ctx, args, setting=None)
        else:
            setting = get_setting(ctx)
            (best_epoch, acc_mae, acc_rmse, acc_mape,
             acc_elements, acc_abs_error_sum, acc_sq_error_sum, acc_mape_error_sum, acc_mape_elements,
             eff_train_time, eff_val_time, eff_test_time,
             eff_comm_size_mb, actual_rounds, eff_flops) = train_tdlr_sedlr_task(
                ctx, args, setting=setting
            )

            dataset_client_name = f"{args.dataset_name}_client{ctx.rank}"
            acc_mse = round(acc_rmse ** 2, 4)

            log_experiment_results(
                model_name=args.model,
                dataset_client=dataset_client_name,
                feature_type=args.feature_type,
                best_epoch=best_epoch + 1,
                acc_mae=round(acc_mae, 4),
                acc_mse=acc_mse,
                acc_rmse=round(acc_rmse, 4),
                acc_mape=round(acc_mape, 4),
                eff_train_time=round(eff_train_time, 2),
                eff_val_time=round(eff_val_time, 4),
                eff_test_time=round(eff_test_time, 4),
                eff_comm_size_mb=eff_comm_size_mb,
                eff_train_round=actual_rounds,
                eff_flops=eff_flops,
                acc_elements=acc_elements,
                acc_abs_error_sum=round(acc_abs_error_sum, 4),
                acc_sq_error_sum=round(acc_sq_error_sum, 4),
                acc_mape_error_sum=round(acc_mape_error_sum, 4),
                acc_mape_elements=acc_mape_elements,
                dp_noise=getattr(args, 'dp_noise', 0.0)
            )
            print(f"Client {ctx.rank} {args.model} test metrics saved.", flush=True)

            if ctx.is_on_guest:
                print(f"[{args.model}] guest result saved; returning normally.", flush=True)
        return

    # ==== 分支5: 标准 FedAvg 模式 (其他模型) ====
    print(f"Running Standard FedAvg Mode for {args.model}")
    if ctx.is_on_arbiter:
        train(ctx)
    else:
        setting = get_setting(ctx)
        train(ctx, *setting)


def run(ctx: Context):
    """FATE entry point with node-client multiplexing for cross-device methods."""
    _configure_cross_device_node_clients()
    if not bool(getattr(args, "cross_device_multiplex", False)):
        return _run_single_context(ctx)

    logical_clients = int(args.logical_num_clients)
    num_workers = int(args.num_workers)
    assignments = _cross_device_worker_assignments(logical_clients, num_workers)

    if ctx.is_on_arbiter:
        print(
            f"[CrossDeviceMux][Arbiter] exposing {logical_clients} logical clients "
            f"over {num_workers} physical workers.",
            flush=True,
        )
        virtual_ctx = _CrossDeviceServerContext(
            ctx, logical_clients, num_workers, assignments
        )
        return _run_single_context(virtual_ctx)

    worker_id = int(ctx.rank)
    if worker_id < 0 or worker_id >= num_workers:
        raise RuntimeError(
            f"Physical worker rank {worker_id} is outside configured P={num_workers}."
        )
    logical_client_ids = assignments[worker_id]
    print(
        f"[CrossDeviceMux][Worker {worker_id}] logical_clients={logical_client_ids} "
        f"count={len(logical_client_ids)}",
        flush=True,
    )
    mux = _CrossDeviceWorkerMux(
        ctx, worker_id, logical_client_ids, assignments
    )

    # Every logical client retains independent model/optimizer/data-loader
    # state.  Threads are used because FATE's party proxy belongs to this
    # physical process; the mux above is the only object that touches it.
    failures = []
    with ThreadPoolExecutor(
        max_workers=len(logical_client_ids),
        thread_name_prefix=f"logical-client-w{worker_id}",
    ) as executor:
        futures = {
            executor.submit(
                _run_single_context,
                _CrossDeviceClientContext(ctx, mux, logical_client_id),
            ): logical_client_id
            for logical_client_id in logical_client_ids
        }
        for future in as_completed(futures):
            logical_client_id = futures[future]
            try:
                future.result()
            except BaseException as exc:
                failures.append((logical_client_id, exc))

    if failures:
        logical_client_id, exc = failures[0]
        raise RuntimeError(
            f"Cross-device logical client {logical_client_id} failed on worker {worker_id}."
        ) from exc
    return None

if __name__ == "__main__":
    _configure_cross_device_node_clients()
    init_seed(args.seed)
    _log_cuda_environment()
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    current_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(current_dir)

    args.log_dir = os.path.join(
        _federated_log_root(project_root, args.t_out),
        "fate_main",
        f"in{args.t_in}_out{args.t_out}",
    )

    # args.data_list = eval(args.data_list)
    # args.data_list = args.data_list.split('_')
    # args.num_clients = len(args.data_list) - 1

    grid_datasets = ['TaxiBJ', 'TaxiNYC', 'BikeNYC']
    if any(grid_name in args.dataset_name for grid_name in grid_datasets):
        print(f"Detected grid dataset {args.dataset_name}; using num_clients = {args.num_clients}")

    parties = ['guest:10000']
    launch_workers = int(getattr(args, "num_workers", args.num_clients))
    for i in range(1, launch_workers):
        parties.append(f'host:{i + 10000}')
    parties.append(f'arbiter:{launch_workers + 10000}')
    args.parties = parties
    args.log_level = "INFO"
    sys.argv.append('--parties')
    sys.argv.extend(parties)
    sys.argv.append('--log_level')
    sys.argv.append("INFO")

    items = ['PeMS04', 'PeMS08', 'HK', 'FT']
    target_size = 0.05



    launch(run)
    # for t_out in [3, 12]:
    #     args.t_out = t_out
    #     for target_city in items:
    #         source_cities = [x for x in items if x != target_city]
    #         source_cities.sort()
    #         source_cities.append(target_city)
    #         args.data_list = source_cities
    #         launch(run)
    #
    # args.t_out = 3
    #
    # for target_city in items:
    #     source_cities = [x for x in items if x != target_city]
    #     print(source_cities, target_city)
    #     for target_size in [0.05, 0.1, 0.2, 0.4]:
    #         args.target_train_ratio = target_size
    #         launch(run)
