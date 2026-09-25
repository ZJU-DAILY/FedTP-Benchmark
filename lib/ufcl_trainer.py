# 文件路径: lib/ufcl_trainer.py
import torch
from torch import nn
import torch.nn.functional as F
import copy
from transformers import Trainer
from lib.ufcl_core import ReplayBuffer, spatio_temporal_mixup

class UFCLTrainer(Trainer):
    """
    集成 UFCL 策略的自定义 Trainer (V5 完整版：包含合成数据与知识蒸馏)
    """
    def __init__(
        self,
        model,
        args,
        train_dataset,
        eval_dataset,
        compute_metrics,
        optimizers,
        ufcl_max_nodes=None,
        use_replay_mixup=True,
        use_synthetic_replay=True,
        use_teacher_kd=True,
        noise_std=0.05,
        kd_weight=1.0,
        **kwargs,
    ):
        super().__init__(model=model, args=args, train_dataset=train_dataset, 
                         eval_dataset=eval_dataset, compute_metrics=compute_metrics, 
                         optimizers=optimizers, **kwargs)
        
        self.replay_buffer = ReplayBuffer(max_size=2000)
        self.ufcl_max_nodes = ufcl_max_nodes
        self.use_replay_mixup = bool(use_replay_mixup)
        self.use_synthetic_replay = bool(use_synthetic_replay)
        self.use_teacher_kd = bool(use_teacher_kd)
        self.noise_std = float(noise_std)
        self.kd_weight = float(kd_weight)
        self.needs_teacher = self.use_synthetic_replay or self.use_teacher_kd
        self.last_ufcl_stats = {}
        
        # 【新增】教师模型（保存上一轮全局模型的快照）
        self.teacher_model = None

    def compute_loss(self, model, inputs, return_outputs=False):
        # =======================================================
        # 0. 初始化/更新 Teacher 模型 (仅在每轮训练刚开始时执行一次)
        # =======================================================
        if model.training and self.needs_teacher and self.teacher_model is None:
            # 复制当前刚从 Server 下载下来的全局模型作为 Teacher
            self.teacher_model = copy.deepcopy(model)
            self.teacher_model.eval() # 冻结 Teacher 的参数
            # print(">>> [UFCLTrainer] Teacher Model 已快照，开启知识蒸馏与数据合成")

        # 1. 解析数据
        if isinstance(inputs, dict):
            real_x = inputs.get("x") or inputs.get("input_ids") or inputs.get("inputs")
            real_y = inputs.get("labels") or inputs.get("y") or inputs.get("targets")
        elif isinstance(inputs, (tuple, list)):
            real_x = inputs[0]
            real_y = inputs[1]
        else:
            raise ValueError(f"UFCLTrainer 收到不支持的输入格式: {type(inputs)}")

        device = real_x.device
        target_nodes = self.ufcl_max_nodes
        if target_nodes is None:
             if hasattr(model, 'num_nodes'): target_nodes = model.num_nodes
             elif hasattr(model, 'module'): target_nodes = getattr(model.module, 'num_nodes', None)

        real_num_nodes = 0
        node_dim_index = -1 

        # 2. 动态维度补齐 (Padding)
        if target_nodes is not None:
            min_diff = float('inf')
            for dim in range(1, real_x.dim()): 
                size = real_x.size(dim)
                diff = target_nodes - size
                if 0 <= diff < min_diff and diff < 200:
                    min_diff = diff
                    node_dim_index = dim
                    real_num_nodes = size
            
            if node_dim_index != -1 and min_diff > 0:
                ndim = real_x.dim()
                pad_arg = [0] * (ndim * 2)
                pad_idx = 2 * (ndim - 1 - node_dim_index) + 1
                pad_arg[pad_idx] = min_diff
                real_x = F.pad(real_x, tuple(pad_arg))
                
                if real_y.dim() == real_x.dim():
                    real_y = F.pad(real_y, tuple(pad_arg))
                elif real_y.dim() == real_x.dim() - 1: 
                    y_ndim = real_y.dim()
                    if node_dim_index < y_ndim:
                        y_pad_arg = [0] * (y_ndim * 2)
                        y_pad_idx = 2 * (y_ndim - 1 - node_dim_index) + 1
                        y_pad_arg[y_pad_idx] = min_diff
                        real_y = F.pad(real_y, tuple(y_pad_arg))
        
        real_x = real_x.to(device)
        real_y = real_y.to(device)

        # 3. UFCL Mixup & 【合成数据生成】
        if model.training:
            # 3.1 Mixup 历史合成数据
            buffer_before = len(self.replay_buffer.buffer_data)
            buf_x, buf_y = self.replay_buffer.sample(real_x.size(0)) if self.use_replay_mixup else (None, None)
            if self.use_replay_mixup and buf_x is not None:
                train_x, train_y, mix_lambda = spatio_temporal_mixup(
                    real_x, real_y, buf_x, buf_y, return_lambda=True
                )
            else:
                train_x, train_y = real_x, real_y
                mix_lambda = None
            
            if self.use_synthetic_replay:
                # =======================================================
                # 3.2 【核心新增】合成当前数据的伪造版本 (Synthetic Data)
                # =======================================================
                with torch.no_grad():
                    # A. 扰动特征空间生成假数据
                    noise = torch.randn_like(real_x) * self.noise_std
                    syn_x = real_x + noise
                    
                    # B. 用 Teacher 模型给假数据打上全局视角的“伪标签”
                    teacher_outputs = self.teacher_model(syn_x)
                    syn_y = teacher_outputs[0] if isinstance(teacher_outputs, (tuple, list)) else teacher_outputs
                    
                # C. 存入 Buffer 的是 (syn_x, syn_y)，真实数据绝不入库
                self.replay_buffer.push(syn_x.detach(), syn_y.detach())
        else:
            buffer_before = len(self.replay_buffer.buffer_data)
            mix_lambda = None
            train_x, train_y = real_x, real_y

        # 4. Student 模型前向传播
        outputs = model(train_x)
        pred = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        # =======================================================
        # 5. Masked 任务损失 + 【知识蒸馏损失 L_distill】
        # =======================================================
        if node_dim_index != -1 and real_num_nodes > 0:
            mask = torch.zeros_like(pred, device=device)
            slices = [slice(None)] * pred.dim()
            slices[node_dim_index] = slice(0, real_num_nodes)
            mask[tuple(slices)] = 1
            
            # 计算主任务损失
            loss_fct = nn.MSELoss(reduction='none')
            task_loss = (loss_fct(pred, train_y) * mask).sum() / (mask.sum() + 1e-9)
            
            # 计算知识蒸馏损失 (用 Student 匹配 Teacher 对当前真实数据的理解)
            if model.training and self.use_teacher_kd and self.teacher_model is not None:
                with torch.no_grad():
                    t_pred_outputs = self.teacher_model(real_x)
                    t_pred = t_pred_outputs[0] if isinstance(t_pred_outputs, (tuple, list)) else t_pred_outputs
                
                kd_loss = (loss_fct(pred, t_pred) * mask).sum() / (mask.sum() + 1e-9)
                # 论文中的 lambda 权重，这里暂时设为 1.0 (可以根据表现调整)
                loss = task_loss + self.kd_weight * kd_loss
            else:
                kd_loss = torch.zeros((), device=device)
                loss = task_loss
        else:
            loss_fct = nn.MSELoss()
            task_loss = loss_fct(pred, train_y)
            
            if model.training and self.use_teacher_kd and self.teacher_model is not None:
                with torch.no_grad():
                    t_pred_outputs = self.teacher_model(real_x)
                    t_pred = t_pred_outputs[0] if isinstance(t_pred_outputs, (tuple, list)) else t_pred_outputs
                kd_loss = loss_fct(pred, t_pred)
                loss = task_loss + self.kd_weight * kd_loss
            else:
                kd_loss = torch.zeros((), device=device)
                loss = task_loss

        self.last_ufcl_stats = {
            "buffer_before": int(buffer_before),
            "buffer_after": int(len(self.replay_buffer.buffer_data)),
            "used_replay": bool(buffer_before > 0),
            "mix_lambda": mix_lambda,
            "task_loss": float(task_loss.detach().cpu()),
            "kd_loss": float(kd_loss.detach().cpu()),
            "kd_weight": self.kd_weight,
            "total_loss": float(loss.detach().cpu()),
            "use_replay_mixup": self.use_replay_mixup,
            "use_synthetic_replay": self.use_synthetic_replay,
            "use_teacher_kd": self.use_teacher_kd,
        }
        return (loss, outputs) if return_outputs else loss
