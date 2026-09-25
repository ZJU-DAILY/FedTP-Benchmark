import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import numpy as np
import math
import matplotlib.pyplot as plt

import sys

sys.path.append("../")

from lib.load_dataset import load_dataset


class NetworkGenerator(nn.Module):
    def __init__(self, max_nodes):
        super().__init__()
        # 对应文献中参数化矩阵，用于学习 a_ij1 和 a_ij2
        self.param_matrix = nn.Parameter(torch.randn(max_nodes, max_nodes, 2))

    def forward(self, actual_nodes, tau=0.5, node_ids=None):
        if node_ids is None:
            logits = self.param_matrix[:actual_nodes, :actual_nodes, :]
        else:
            node_ids = node_ids.to(self.param_matrix.device)
            logits = self.param_matrix.index_select(0, node_ids).index_select(1, node_ids)
        # 严格遵守文献 Eq(2) 的 Gumbel-Softmax 采样 
        # F.gumbel_softmax 内部自动添加独立同分布的 Gumbel 噪声并除以温度参数 tau
        probs = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
        # 返回 a_ij1 对应的概率作为邻接矩阵权重
        return probs[..., 0]

class GTHA(nn.Module):
    """加入 Talking-Heads 机制的图多头注意力层"""
    def __init__(self, in_dim, heads=6):
        super().__init__()
        self.heads = heads
        self.d_k = in_dim // heads

        self.W = nn.Linear(in_dim, self.d_k * heads)
        self.attn = nn.Linear(2 * self.d_k, 1)
        
        # Talking-heads 投影矩阵 (跨头部信息交互)
        self.proj_heads = nn.Linear(heads, heads, bias=False)

    def forward(self, h, adj):
        batch, nodes, feat = h.size()
        h_proj = self.W(h).view(batch, nodes, self.heads, self.d_k)

        # 构造拼接特征
        h_i = h_proj.unsqueeze(2).expand(-1, -1, nodes, -1, -1)
        h_j = h_proj.unsqueeze(1).expand(-1, nodes, -1, -1, -1)
        
        # 注意力 logits，形状: (batch, nodes, nodes, heads)
        e = self.attn(torch.cat([h_i, h_j], dim=-1)).squeeze(-1)
        
        # 【核心修改】Talking-heads 机制：在 Softmax 前进行跨 head 线性投影
        e = self.proj_heads(e)
        e = F.leaky_relu(e)

        # 应用邻接矩阵掩码
        e = e.masked_fill(adj.unsqueeze(0).unsqueeze(-1) == 0, -1e9)
        attn = F.softmax(e, dim=2) 

        # 聚合特征
        out = torch.einsum('bnmh,bmhd->bnhd', attn, h_proj)
        out = out.reshape(batch, nodes, self.heads * self.d_k)
        return out

class GTCN(nn.Module):
    """门控时序卷积层"""

    def __init__(self, in_dim, out_dim, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, (1, kernel_size), padding=(0, 1))
        self.conv2 = nn.Conv2d(in_dim, out_dim, (1, kernel_size), padding=(0, 1))
        self.gate = nn.Conv2d(in_dim, out_dim, (1, kernel_size), padding=(0, 1))

    def forward(self, x):
        # print(x.shape)
        x = x.transpose(1, 2)
        gate = torch.sigmoid(self.gate(x))
        return gate * self.conv1(x) + (1 - gate) * self.conv2(x)

class ASTGAT(nn.Module):
    def __init__(self, num_nodes, in_dim, pred_len, adj, emb_dim=64, max_nodes=None, node_ids=None):
        super().__init__()
        self.max_nodes = max_nodes if max_nodes is not None else num_nodes
        self.actual_nodes = num_nodes
        
        self.generator = NetworkGenerator(self.max_nodes)
        self.register_buffer("adj", adj.float())
        if node_ids is None:
            node_ids = torch.arange(num_nodes, dtype=torch.long)
        else:
            node_ids = torch.as_tensor(node_ids, dtype=torch.long)
        if node_ids.numel() != num_nodes:
            raise ValueError(f"node_ids length {node_ids.numel()} does not match num_nodes {num_nodes}")
        self.register_buffer("node_ids", node_ids)
        
        # 空间嵌入
        self.spatial_emb = nn.Embedding(self.max_nodes, emb_dim)

        self.gtcn1 = GTCN(in_dim, emb_dim)
        
        # 明确定义 heads 数量
        heads = 6
        self.gtha1 = GTHA(emb_dim, heads=heads)

        # 【修复核心】：精确计算 GTHA 拼接后的真实输出维度
        # 避免 64 // 6 导致的维度丢失问题 (例如 64 输入 -> 60 输出)
        gtha_out_dim = (emb_dim // heads) * heads 

        self.end_conv = nn.Sequential(
            nn.Conv2d(gtha_out_dim, 512, 1), # 使用计算出的真实维度
            nn.ReLU(),
            nn.Conv2d(512, pred_len, 1)
        )

    def forward(self, x):
        dyn_adj = self.generator(self.actual_nodes, node_ids=self.node_ids)

        # 时空特征提取 (简化处理对齐维度)
        x_t = self.gtcn1(x).squeeze(-1).transpose(1, 2)
        
        # 加入空间嵌入
        s_emb = self.spatial_emb(self.node_ids.to(x.device))
        x_t = x_t + s_emb.unsqueeze(0)
        
        combined_adj = dyn_adj * self.adj.to(dyn_adj.device)
        x_out = self.gtha1(x_t, combined_adj)  
        
        pred = self.end_conv(x_out.unsqueeze(-1).transpose(1, 2)).transpose(1, 2)
        
        # 训练模式下：返回 (预测值, 动态图)，供 custom_loss 计算图正则化 L_G
        if self.training:
            return pred, dyn_adj
        # 评估/测试模式下：只返回预测值，防止 FATE 内部的 metrics 函数拿到元组后崩溃
        else:
            return pred
