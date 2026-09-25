# 文件路径: lib/ufcl_core.py
import torch
import torch.nn as nn
import numpy as np

class ReplayBuffer:
    """
    UFCL 记忆缓存区：基于 RMIR (Ranking-based Maximally Interfered Retrieval) 采样策略 [cite: 254]
    """
    def __init__(self, max_size=2000):
        self.max_size = max_size
        self.buffer_data = []  # 存储 X
        self.buffer_label = [] # 存储 Y

    def push(self, data, label):
        """
        存入数据 (保持 FIFO 策略更新 Buffer)
        """
        data = data.detach().cpu()
        label = label.detach().cpu()
        
        batch_len = data.size(0)
        current_len = len(self.buffer_data)
        
        if current_len + batch_len > self.max_size:
            remove_cnt = (current_len + batch_len) - self.max_size
            if remove_cnt < current_len:
                self.buffer_data = self.buffer_data[remove_cnt:]
                self.buffer_label = self.buffer_label[remove_cnt:]
            else:
                self.buffer_data = []
                self.buffer_label = []

        for i in range(batch_len):
            self.buffer_data.append(data[i])
            self.buffer_label.append(label[i])

    def sample(self, batch_size):
        """
        Light-weight replay sampling API used by UFCLTrainer.
        RMIR sampling is still available through sample_rmir when the caller
        provides current batch/model context.
        """
        current_len = len(self.buffer_data)
        if current_len < 1:
            return None, None

        real_sample_size = min(int(batch_size), current_len)
        if current_len <= real_sample_size:
            sampled_indices = list(range(current_len))
        else:
            sampled_indices = np.random.choice(current_len, size=real_sample_size, replace=False)

        sampled_data = torch.stack([self.buffer_data[int(i)] for i in sampled_indices])
        sampled_label = torch.stack([self.buffer_label[int(i)] for i in sampled_indices])
        return sampled_data, sampled_label

    def sample_rmir(self, current_x, current_y, model, batch_size, lr=0.001, N_mult=2):
        """
        执行 RMIR 采样算法 
        Args:
            current_x, current_y: 当前训练批次的数据
            model: 当前训练的模型 (用于计算前瞻更新和干扰)
            batch_size: 最终需要采样的数量 S
            lr: 虚拟更新的学习率
            N_mult: 候选集 N 的倍数 (N = batch_size * N_mult)
        """
        current_len = len(self.buffer_data)
        if current_len < 1:
            return None, None
            
        device = current_x.device
        real_sample_size = min(batch_size, current_len)
        N_size = min(real_sample_size * N_mult, current_len)

        # 如果 Buffer 数据太少，直接全部返回
        if current_len <= real_sample_size:
            sampled_data = torch.stack(self.buffer_data)
            sampled_label = torch.stack(self.buffer_label)
            return sampled_data, sampled_label

        # ---------------------------------------------------------
        # 步骤 0: 预抽取一个较大的候选池，避免在全量 Buffer 上算梯度导致内存爆炸
        # ---------------------------------------------------------
        pool_size = min(N_size * 4, current_len)
        pool_indices = np.random.choice(current_len, size=pool_size, replace=False)
        pool_x = torch.stack([self.buffer_data[i] for i in pool_indices]).to(device)
        pool_y = torch.stack([self.buffer_label[i] for i in pool_indices]).to(device)

        criterion = nn.L1Loss(reduction='none') # MAE 作为采样损失标准 [cite: 260]

        # ---------------------------------------------------------
        # 步骤 1: 计算原始参数下，候选池的 Loss
        # ---------------------------------------------------------
        model.eval()
        with torch.no_grad():
            orig_pred = model(pool_x)
            if isinstance(orig_pred, (list, tuple)): orig_pred = orig_pred[0]
            loss_orig = criterion(orig_pred, pool_y).view(pool_size, -1).mean(dim=1)

        # ---------------------------------------------------------
        # 步骤 2: 虚拟参数更新 (Foreseen Parameters Update) 
        # ---------------------------------------------------------
        model.train()
        orig_weights = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

        # 计算当前 Batch 的梯度
        model.zero_grad()
        curr_pred = model(current_x)
        if isinstance(curr_pred, (list, tuple)): curr_pred = curr_pred[0]
        curr_loss = criterion(curr_pred, current_y).mean()
        curr_loss.backward()

        # 对模型施加虚拟更新: theta^v = theta - alpha * grad
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    p.sub_(lr * p.grad)

        # ---------------------------------------------------------
        # 步骤 3: 计算虚拟参数下的候选池 Loss 并得出干扰度 (Interference)
        # ---------------------------------------------------------
        model.eval()
        with torch.no_grad():
            new_pred = model(pool_x)
            if isinstance(new_pred, (list, tuple)): new_pred = new_pred[0]
            loss_new = criterion(new_pred, pool_y).view(pool_size, -1).mean(dim=1)

        # 恢复模型原始权重
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    p.copy_(orig_weights[n])

        # 干扰度 = 更新后的Loss - 原始Loss (越大说明当前更新越会引发遗忘)
        interference = loss_new - loss_orig
        
        # 选出受干扰最大的 N_size 个样本的索引
        top_n_local_indices = torch.topk(interference, N_size).indices
        N_x = pool_x[top_n_local_indices]
        N_indices = [pool_indices[i.item()] for i in top_n_local_indices]

        # ---------------------------------------------------------
        # 步骤 4: 皮尔逊相关性筛选 (Pearson Coefficient) 
        # ---------------------------------------------------------
        # 把 current_x 按 Batch 取均值，展平为 1D 向量
        flat_curr_x = current_x.view(current_x.size(0), -1).mean(dim=0)
        curr_mean = flat_curr_x.mean()
        curr_std = flat_curr_x.std() + 1e-8
        curr_norm = (flat_curr_x - curr_mean) / curr_std

        pearson_scores = []
        flat_N_x = N_x.view(N_size, -1)
        
        for i in range(N_size):
            nx = flat_N_x[i]
            nx_mean = nx.mean()
            nx_std = nx.std() + 1e-8
            nx_norm = (nx - nx_mean) / nx_std
            pearson_corr = (curr_norm * nx_norm).mean().item()
            pearson_scores.append(pearson_corr)

        # 选出皮尔逊系数最高的 S 个样本 (即 real_sample_size)
        top_s_local_indices = np.argsort(pearson_scores)[-real_sample_size:]
        final_indices = [N_indices[i] for i in top_s_local_indices]

        # ---------------------------------------------------------
        # 步骤 5: 拼装最终结果并返回
        # ---------------------------------------------------------
        sampled_data = torch.stack([self.buffer_data[i] for i in final_indices])
        sampled_label = torch.stack([self.buffer_label[i] for i in final_indices])

        return sampled_data, sampled_label

def spatio_temporal_mixup(real_x, real_y, buf_x, buf_y, alpha=0.2, return_lambda=False):
    """
    [cite_start]UFCL Mixup 混合机制 [cite: 263, 268]
    公式: x_mix = lambda * x_real + (1 - lambda) * x_buf
    """
    if buf_x is None:
        if return_lambda:
            return real_x, real_y, None
        return real_x, real_y
        
    device = real_x.device
    
    # 确保 Buffer 数据移动到了正确的 Device (GPU)
    buf_x = buf_x.to(device)
    buf_y = buf_y.to(device)
    
    # --- 维度对齐核心逻辑 ---
    curr_batch = real_x.size(0)
    buf_batch = buf_x.size(0)
    
    # 情况1: Buffer 数据比当前 Batch 少 -> 重复 Buffer 数据来填满
    if buf_batch < curr_batch:
        repeat_times = (curr_batch // buf_batch) + 1
        # 构建重复维度: [repeat_times, 1, 1, ...]
        repeat_dims = [1] * buf_x.dim()
        repeat_dims[0] = repeat_times
        
        buf_x = buf_x.repeat(*repeat_dims)[:curr_batch]
        buf_y = buf_y.repeat(*repeat_dims)[:curr_batch]
        
    # 情况2: Buffer 数据比当前 Batch 多 (罕见，但需处理) -> 截断
    elif buf_batch > curr_batch:
        buf_x = buf_x[:curr_batch]
        buf_y = buf_y[:curr_batch]
        
    # --- 混合计算 ---
    # 生成混合系数 lambda，服从 Beta 分布
    lam = np.random.beta(alpha, alpha)
    
    # 执行混合
    mixed_x = lam * real_x + (1 - lam) * buf_x
    mixed_y = lam * real_y + (1 - lam) * buf_y
    
    if return_lambda:
        return mixed_x, mixed_y, float(lam)
    return mixed_x, mixed_y
