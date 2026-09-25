import os

import torch
from torch.utils.data import TensorDataset

def slice_data_for_twt(dataset, time_window_num):
    """
    将 FATE 的 TensorDataset 按时间顺序切分为 time_window_num 个子数据集
    用于 Fed4TP 的 TWT 训练
    """
    # 1. 检查数据类型
    if not isinstance(dataset, TensorDataset):
        print(f"[Fed4TP Warning] Dataset type is {type(dataset)}, not TensorDataset. TWT slicing skipped.")
        # Privacy trace-only mode deliberately wraps a single deterministic
        # record and therefore has no .tensors field to slice.  Fed4TP's
        # server still executes every configured time window, so returning a
        # one-element list makes clients crash at window 1.  Reuse that one
        # fixed record for each logical window; this preserves the trace-only
        # experiment's fixed input while keeping the protocol round schedule
        # identical on every party.
        if os.environ.get("PRIVACY_TRACE_ONLY", "0") == "1":
            windows = max(1, int(time_window_num))
            print(f"[Fed4TP PrivacyTrace] repeating deterministic dataset across {windows} TWT windows.")
            return [dataset] * windows
        return [dataset]

    # 2. 获取原始 Tensor (X, Y)
    # dataset.tensors[0] 是 X (Feature), [1] 是 Y (Label)
    X_full, Y_full = dataset.tensors
    total_samples = X_full.shape[0]
    
    # 3. 计算每个窗口的大小
    # Fed4TP 原文是均匀切分
    window_size = total_samples // time_window_num
    
    sliced_datasets = []
    print(f"[Fed4TP] Slicing dataset (Total: {total_samples}) into {time_window_num} windows...")
    
    for i in range(time_window_num):
        start = i * window_size
        # 最后一个窗口包含剩余所有数据，防止丢数据
        end = (i + 1) * window_size if i < time_window_num - 1 else total_samples
        
        # 切片
        X_win = X_full[start:end]
        Y_win = Y_full[start:end]
        
        # 重新封装为 TensorDataset
        sub_dataset = TensorDataset(X_win, Y_win)
        sliced_datasets.append(sub_dataset)
        
        print(f"  -> Window {i}: {len(sub_dataset)} samples")
        
    return sliced_datasets
