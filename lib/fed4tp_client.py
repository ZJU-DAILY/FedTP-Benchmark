import copy
import pywt
import torch
import numpy as np
from torch.utils.data import TensorDataset
from fate.ml.nn.homo.fedavg import FedAVGClient

class Fed4TPClient(FedAVGClient):
    def __init__(self, ctx, model, train_set_list, val_set, optimizer, loss_fn, scheduler, training_args, fed_args, compute_metrics, args):
        # 这里的 train_set_list 是我们在 fate_main.py 里切好的列表
        self.twt_data_list = train_set_list
        first_window = self.twt_data_list[0] if self.twt_data_list else None
        
        # 调用父类初始化
        super().__init__(
            ctx=ctx,
            model=model,
            train_set=first_window,  # 传入第一个窗口作为初始数据
            val_set=val_set,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            training_args=training_args,
            fed_args=fed_args,
            compute_metrics=compute_metrics
        )
        
        # 【关键修复 1】手动保存 training_args，解决 'no attribute' 报错
        self.training_args = training_args
        
        # 【关键修复 2】将全局 args 改名为 fed4tp_config，避免覆盖父类可能存在的 self.args
        self.fed4tp_config = args
        
        self.current_window_idx = 0
        self.rounds_counter = 0 
        self.is_unreliable = False
        
        print(f"[Fed4TPClient] Initialized on Rank {ctx.rank}. TWT Windows: {len(self.twt_data_list)}")

    def _perform_local_denoising(self):
        """
        GLD 模块: 本地去噪 (Local Denoising)
        """
        # 注意：这里改用了 self.fed4tp_config
        if not self.fed4tp_config.fed4tp_gd_use or not self.is_unreliable:
            return

        print(f"[Fed4TP GLD] Client detected as UNRELIABLE. Performing Local Denoising on Window {self.current_window_idx}...")
        
        # 1. 获取当前数据
        current_dataset = self.train_dataset 
        if not isinstance(current_dataset, TensorDataset):
            return

        X_tensor, Y_tensor = current_dataset.tensors
        data_np = X_tensor.cpu().numpy()
        
        # 2. 执行小波去噪
        try:
            # 简化版：对最后一维特征进行去噪 (db4 小波, level=3)
            coeffs = pywt.wavedec(data_np, 'db4', level=3, axis=1) 
            coeffs_filtered = [pywt.threshold(c, 0.5 * np.max(np.abs(c)), mode='soft') for c in coeffs]
            data_denoised = pywt.waverec(coeffs_filtered, 'db4', axis=1)
            
            # 形状修正
            if data_denoised.shape != data_np.shape:
                data_denoised = data_denoised[:, :data_np.shape[1], :]

            # 3. 更新数据集
            X_new = torch.from_numpy(data_denoised).float().to(self.ctx.device)
            self.train_dataset = TensorDataset(X_new, Y_tensor)
            
            # 更新 Trainer 内部引用
            if hasattr(self, 'trainer'):
                self.trainer.train_dataset = self.train_dataset
                
            print("[Fed4TP GLD] Denoising complete.")
            
        except Exception as e:
            print(f"[Fed4TP GLD Error] Denoising failed: {e}")

    def _check_and_switch_window(self):
        """
        TWT 模块: 时间窗口切换逻辑
        """
        if len(self.twt_data_list) <= 1:
            return

        # 这里的 self.training_args 现在可以正常访问了
        total_epochs = self.training_args.num_train_epochs
        epochs_per_window = max(1, total_epochs // len(self.twt_data_list))
        
        # 计算目标窗口
        target_window = int(self.rounds_counter / epochs_per_window)
        
        # 边界限制
        if target_window >= len(self.twt_data_list):
            target_window = len(self.twt_data_list) - 1
            
        # 如果需要切换
        if target_window != self.current_window_idx:
            print(f"[Fed4TP TWT] Switching Window: {self.current_window_idx} -> {target_window}")
            self.current_window_idx = target_window
            
            # 1. 获取新数据
            new_dataset = self.twt_data_list[target_window]
            
            # 2. Near Window 补全逻辑
            if len(new_dataset) == 0:
                print(f"[Fed4TP Warning] Window {target_window} is empty! Searching for neighbors...")
                found = False
                for offset in range(1, len(self.twt_data_list)):
                    # Check Left
                    left = target_window - offset
                    if left >= 0 and len(self.twt_data_list[left]) > 0:
                        new_dataset = self.twt_data_list[left]
                        print(f" -> Replaced with previous window {left}")
                        found = True
                        break
                    # Check Right
                    right = target_window + offset
                    if right < len(self.twt_data_list) and len(self.twt_data_list[right]) > 0:
                        new_dataset = self.twt_data_list[right]
                        print(f" -> Replaced with next window {right}")
                        found = True
                        break
                
                if not found:
                    print(" -> Fatal: All windows empty! Keeping current dataset.")
                    return

            # 3. 热替换 Trainer 的数据
            self.train_dataset = new_dataset
            if hasattr(self, 'trainer'):
                self.trainer.train_dataset = new_dataset
                # 强制 Trainer 重新生成 DataLoader
                if hasattr(self.trainer, 'get_train_dataloader'):
                     self.trainer._train_dataloader = None 

    def train(self):
        """
        重写 Train 方法，注入 Fed4TP 逻辑
        """
        self._check_and_switch_window()
        self._perform_local_denoising()
        
        train_output = super().train()
        
        self.rounds_counter += 1
        return train_output