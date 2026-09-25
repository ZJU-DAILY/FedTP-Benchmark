import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

DATA_DIR = "/home/zzh/ymy/Fed/data"

_GRID_BASE_CACHE = {}
_GRID_ADJ_CACHE = {}
_GRID_DISK_CACHE_VERSION = "v1"


class UnifiedTrafficDataset(Dataset):
    def __init__(
        self,
        city,
        split="train",
        len_c=6,
        use_period=True,
        use_trend=True,
        t_out=3,
        view_type="grid",
        selected_nodes=None,
        model_name="default",
    ):
        self.city = city
        self.split = split
        self.len_c = len_c
        self.use_period = use_period
        self.use_trend = use_trend
        self.t_out = t_out
        self.view_type = view_type
        self.selected_nodes = selected_nodes
        self.model_name = model_name

        self.target_dir = os.path.join(DATA_DIR, city)
        meta_path = os.path.join(self.target_dir, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.H, self.W = self.meta["H"], self.meta["W"]
        self.N = self.meta["num_nodes"]
        self.ext_names = self.meta.get("ext_names", [])
        self.spd = self.meta["slots_per_day"]

        self._load_and_reshape_data()
        self._split_data()
        self._normalize_data()

        if self.view_type == "graph":
            self.adj = self._build_adj()
        else:
            self.adj = None

        self.valid_indices = self._generate_valid_indices()
        print(f"[{city} - {split}] Dataset 初始化完成! 有效样本数: {len(self.valid_indices)}")

    def _base_cache_key(self):
        selected_nodes_key = tuple(self.selected_nodes) if self.selected_nodes is not None else None
        return self.city, selected_nodes_key

    def _adj_cache_key(self):
        selected_nodes_key = tuple(self.selected_nodes) if self.selected_nodes is not None else None
        return self.city, self.model_name, selected_nodes_key

    def _disk_cache_dir(self):
        return os.path.join(self.target_dir, f"unified_cache_{_GRID_DISK_CACHE_VERSION}")

    def _disk_cache_lock_dir(self):
        return os.path.join(self._disk_cache_dir(), ".build_lock")

    def _disk_cache_manifest_path(self):
        return os.path.join(self._disk_cache_dir(), "manifest.json")

    def _disk_cache_paths(self):
        cache_dir = self._disk_cache_dir()
        return {
            "flow_data": os.path.join(cache_dir, "flow_data.npy"),
            "ext_data": os.path.join(cache_dir, "ext_data.npy"),
            "is_observed": os.path.join(cache_dir, "is_observed.npy"),
            "mask_hw": os.path.join(cache_dir, "mask_hw.npy"),
            "manifest": self._disk_cache_manifest_path(),
        }

    def _disk_cache_ready(self):
        paths = self._disk_cache_paths()
        return all(os.path.exists(path) for path in paths.values())

    def _build_disk_cache_from_parquet(self):
        os.makedirs(self._disk_cache_dir(), exist_ok=True)

        parquet_path = os.path.join(self.target_dir, "records.parquet")
        cols_to_read = ["timestamp", "flow_0", "flow_1", "is_observed"] + self.ext_names
        df = pd.read_parquet(parquet_path, columns=cols_to_read)

        t_total = len(df) // (self.H * self.W)
        df_time = df.iloc[:: self.H * self.W]

        is_observed = df_time["is_observed"].values.astype(np.int8)
        if self.ext_names:
            ext_data = df_time[self.ext_names].values.astype(np.float32)
        else:
            ext_data = np.empty((t_total, 0), dtype=np.float32)

        flow_0 = df["flow_0"].values.astype(np.float32)
        flow_1 = df["flow_1"].values.astype(np.float32)
        flow_stacked = np.stack([flow_0, flow_1], axis=-1)
        flow_data = flow_stacked.reshape(t_total, self.H, self.W, 2)

        obs_flow = flow_data[is_observed == 1]
        mask_hw = (~np.isnan(obs_flow).all(axis=(0, 3))).astype(np.int8)
        flow_data = np.nan_to_num(flow_data, nan=0.0)

        paths = self._disk_cache_paths()
        np.save(paths["flow_data"], flow_data)
        np.save(paths["ext_data"], ext_data)
        np.save(paths["is_observed"], is_observed)
        np.save(paths["mask_hw"], mask_hw)

        manifest = {
            "city": self.city,
            "H": self.H,
            "W": self.W,
            "N": self.N,
            "t_total": int(t_total),
            "ext_names": list(self.ext_names),
        }
        with open(paths["manifest"], "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=True, indent=2)

    def _ensure_disk_cache(self):
        if self._disk_cache_ready():
            return

        os.makedirs(self._disk_cache_dir(), exist_ok=True)
        lock_dir = self._disk_cache_lock_dir()
        acquired = False
        while not self._disk_cache_ready():
            try:
                os.mkdir(lock_dir)
                acquired = True
                break
            except FileExistsError:
                time.sleep(0.2)

        if acquired:
            try:
                if not self._disk_cache_ready():
                    self._build_disk_cache_from_parquet()
            finally:
                try:
                    os.rmdir(lock_dir)
                except FileNotFoundError:
                    pass

    def _load_base_arrays_from_disk_cache(self):
        self._ensure_disk_cache()
        paths = self._disk_cache_paths()
        with open(paths["manifest"], "r", encoding="utf-8") as f:
            manifest = json.load(f)

        flow_data = np.load(paths["flow_data"], mmap_mode="r")
        ext_data = np.load(paths["ext_data"], mmap_mode="r")
        is_observed = np.load(paths["is_observed"], mmap_mode="r")
        mask_hw = np.load(paths["mask_hw"], mmap_mode="r")

        return {
            "T_total": int(manifest["t_total"]),
            "flow_data_full": flow_data,
            "ext_data": ext_data,
            "is_observed": is_observed,
            "mask_hw": mask_hw,
        }

    def _load_and_reshape_data(self):
        cache_key = self._base_cache_key()
        cached = _GRID_BASE_CACHE.get(cache_key)
        if cached is None:
            base_arrays = self._load_base_arrays_from_disk_cache()
            t_total = base_arrays["T_total"]
            is_observed = base_arrays["is_observed"]
            ext_data = base_arrays["ext_data"]
            mask_hw = base_arrays["mask_hw"]
            flow_data_full = base_arrays["flow_data_full"]

            if self.selected_nodes is not None:
                flow_data = flow_data_full.reshape(t_total, self.H * self.W, 2)
                flow_data = flow_data[:, self.selected_nodes, :]
                num_nodes = len(self.selected_nodes)
            else:
                flow_data = flow_data_full
                num_nodes = self.N

            train_end = int(t_total * 0.7)
            train_flow = flow_data[:train_end]
            train_obs = is_observed[:train_end]
            valid_train_flow = train_flow[train_obs == 1]
            axis_to_reduce = tuple(range(valid_train_flow.ndim - 1))
            flow_min = valid_train_flow.min(axis=axis_to_reduce, keepdims=True)
            flow_max = valid_train_flow.max(axis=axis_to_reduce, keepdims=True)

            cached = {
                "T_total": t_total,
                "is_observed": is_observed,
                "ext_data": ext_data,
                "flow_data": flow_data,
                "mask_hw": mask_hw,
                "N": num_nodes,
                "flow_min": flow_min,
                "flow_max": flow_max,
            }
            _GRID_BASE_CACHE[cache_key] = cached

        self.T_total = cached["T_total"]
        self.is_observed = cached["is_observed"]
        self.ext_data = cached["ext_data"]
        self.flow_data = cached["flow_data"]
        self.mask_hw = cached["mask_hw"]
        self.N = cached["N"]
        self.flow_min = cached["flow_min"]
        self.flow_max = cached["flow_max"]

    def _split_data(self):
        train_end = int(self.T_total * 0.7)
        val_end = int(self.T_total * 0.8)

        if self.split == "train":
            self.start_idx, self.end_idx = 0, train_end
        elif self.split == "val":
            self.start_idx, self.end_idx = train_end, val_end
        else:
            self.start_idx, self.end_idx = val_end, self.T_total

        self.flow_split = self.flow_data[self.start_idx : self.end_idx]
        self.ext_split = self.ext_data[self.start_idx : self.end_idx]
        self.obs_split = self.is_observed[self.start_idx : self.end_idx]
        self.T_split = len(self.flow_split)

    def _normalize_data(self):
        denominator = self.flow_max - self.flow_min
        denominator[denominator == 0] = 1.0
        self.flow_split = (self.flow_split - self.flow_min) / denominator

    def _generate_valid_indices(self):
        valid_indices = []

        max_offset = self.len_c
        if self.use_period:
            max_offset = max(max_offset, self.spd)
        if self.use_trend:
            max_offset = max(max_offset, 7 * self.spd)

        for i in range(max_offset, self.T_split - self.t_out + 1):
            if not np.all(self.obs_split[i : i + self.t_out] == 1):
                continue

            if self.len_c > 0 and not np.all(self.obs_split[i - self.len_c : i] == 1):
                continue

            if self.use_period:
                p_start = i - self.spd
                if not np.all(self.obs_split[p_start : p_start + self.t_out] == 1):
                    continue

            if self.use_trend:
                t_start = i - 7 * self.spd
                if not np.all(self.obs_split[t_start : t_start + self.t_out] == 1):
                    continue

            valid_indices.append(i)
        return valid_indices

    def _build_adj(self):
        cache_key = self._adj_cache_key()
        cached_adj = _GRID_ADJ_CACHE.get(cache_key)
        if cached_adj is not None:
            return cached_adj

        if self.model_name == "TwoMGTCN":
            adj_path = os.path.join(self.target_dir, f"{self.city}_2mgtcn_adj.npy")
            if not os.path.exists(adj_path):
                raise FileNotFoundError(
                    f"[!] TwoMGTCN 需要语义图，但在 {adj_path} 未找到！请先运行离线脚本。"
                )
            print(f"[*] 命中 TwoMGTCN! 成功加载高级语义图矩阵: {adj_path}")
            adj = np.load(adj_path)
        else:
            mask_n = self.mask_hw.flatten()
            full_n = self.H * self.W
            adj = np.zeros((full_n, full_n), dtype=np.float32)
            for r in range(self.H):
                for c in range(self.W):
                    idx = r * self.W + c
                    if mask_n[idx] == 0:
                        continue
                    adj[idx, idx] = 1.0

                    neighbors = []
                    if r > 0:
                        neighbors.append((r - 1) * self.W + c)
                    if r < self.H - 1:
                        neighbors.append((r + 1) * self.W + c)
                    if c > 0:
                        neighbors.append(r * self.W + (c - 1))
                    if c < self.W - 1:
                        neighbors.append(r * self.W + (c + 1))

                    for n_idx in neighbors:
                        if mask_n[n_idx] == 1:
                            adj[idx, n_idx] = 1.0

        if self.selected_nodes is not None:
            adj = adj[self.selected_nodes][:, self.selected_nodes]

        _GRID_ADJ_CACHE[cache_key] = adj
        return adj

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        y_flow = self.flow_split[i : i + self.t_out]

        empty_shape = (0, self.N, 2) if self.selected_nodes is not None else (0, self.H, self.W, 2)

        x_c = self.flow_split[i - self.len_c : i] if self.len_c > 0 else np.zeros(empty_shape)
        x_p = (
            self.flow_split[i - self.spd : i - self.spd + self.t_out]
            if self.use_period
            else np.zeros(empty_shape)
        )
        x_t = (
            self.flow_split[i - 7 * self.spd : i - 7 * self.spd + self.t_out]
            if self.use_trend
            else np.zeros(empty_shape)
        )
        x_ext = self.ext_split[i : i + self.t_out]

        y_flow = torch.FloatTensor(y_flow)
        x_c = torch.FloatTensor(x_c)
        x_p = torch.FloatTensor(x_p)
        x_t = torch.FloatTensor(x_t)
        x_ext = torch.FloatTensor(x_ext)

        if self.view_type == "grid":
            y_flow = y_flow.permute(0, 3, 1, 2)
            if self.len_c > 0:
                x_c = x_c.permute(0, 3, 1, 2)
            if self.use_period:
                x_p = x_p.permute(0, 3, 1, 2)
            if self.use_trend:
                x_t = x_t.permute(0, 3, 1, 2)
        elif self.view_type == "graph":
            y_flow = y_flow.reshape(self.t_out, self.N, 2)
            if self.len_c > 0:
                x_c = x_c.reshape(self.len_c, self.N, 2)
            if self.use_period:
                x_p = x_p.reshape(self.t_out, self.N, 2)
            if self.use_trend:
                x_t = x_t.reshape(self.t_out, self.N, 2)

        return x_c, x_p, x_t, x_ext, y_flow


if __name__ == "__main__":
    print("测试 1: 包含所有周期特征的 Grid 视图")
    dataset = UnifiedTrafficDataset(
        city="TaxiNYC",
        split="train",
        len_c=6,
        use_period=True,
        use_trend=True,
        view_type="grid",
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    x_c, x_p, x_t, x_ext, y_flow = next(iter(loader))
    print(f"近期特征 (X_c) 形状: {x_c.shape}")
    print(f"昨天特征 (X_p) 形状: {x_p.shape}")
    print(f"上周特征 (X_t) 形状: {x_t.shape}")
    print(f"外部特征 (Ext) 形状: {x_ext.shape}")
    print(f"目标预测 (Y) 形状: {y_flow.shape}\n")
