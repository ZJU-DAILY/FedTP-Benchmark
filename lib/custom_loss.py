import torch
import torch.nn as nn
import math

class FedAGATLoss(nn.Module):
    """
    FedAGAT 专属复合损失函数 [cite: 353, 355]
    包含：基础预测误差 (MAE/MSE) + 基于动态权重的图正则化损失
    """
    def __init__(self, base_loss_func, max_epochs=30):
        super().__init__()
        self.base_loss_func = base_loss_func
        self.max_epochs = max_epochs
        self.batch_count = 0 
        self.batches_per_epoch = 1  # 稍后在主程序中动态注入

    def forward(self, outputs, targets):
        # 优雅地处理模型输出：解包元组
        if isinstance(outputs, tuple) and len(outputs) == 2:
            preds, dyn_adj = outputs
        else:
            preds, dyn_adj = outputs, None
            
        # 1. 基础预测损失
        mae_loss = self.base_loss_func(preds, targets)
        
        # 2. 图正则化损失 L_G [cite: 353, 355]
        if dyn_adj is not None:
            # 计算当前 epoch 进度
            current_epoch = self.batch_count / max(1, self.batches_per_epoch)
            
            # 严格按照文献公式计算动态权重 u(epoch) [cite: 357, 358, 359]
            max_val = 0.1
            mult = -5.0
            u_epoch = max_val * math.exp(mult * (1. - current_epoch / self.max_epochs)**2)
            
            # 计算图正则化 L_G (Frobenius 范数)
            L_G = torch.norm(dyn_adj, p='fro')
            
            loss = mae_loss + u_epoch * L_G
            self.batch_count += 1
        else:
            loss = mae_loss
            
        return loss