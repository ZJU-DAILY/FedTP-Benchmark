import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# 引用你刚刚修改好的 Manual Euler 版 ODEGCN
from model.odegcn import ODEG

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.bn_input = nn.BatchNorm1d(dim_hidden)
        self.swish = Swish()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.bn_input(x)
        x = self.swish(x)
        x = self.dropout(x)
        x = self.layer_hidden(x)
        return x

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size // 2
            self.conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), dilation=(1, dilation_size),
                                  padding=(0, padding))
            self.bn = nn.BatchNorm2d(out_channels)
            self.activation = nn.LeakyReLU(0.1)

            layers += [nn.Sequential(self.conv, self.bn, self.activation)]

        self.network = nn.Sequential(*layers)
        self.downsample = nn.Conv2d(num_inputs, num_channels[-1], (1, 1)) if num_inputs != num_channels[-1] else None
        if self.downsample:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y = x.permute(0, 3, 1, 2)
        y = self.network(y) + self.downsample(y) if self.downsample else y
        y = y.permute(0, 2, 3, 1)
        return y

class GCN(nn.Module):
    def __init__(self, A_hat, in_channels, out_channels, ):
        super(GCN, self).__init__()
        self.A_hat = A_hat
        # 原代码：self.theta = nn.Parameter(torch.cuda.FloatTensor(in_channels, out_channels))
        
        self.theta = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        
        self.reset()

class STGCNBlock(nn.Module):
    # 增加 t_in 参数，去除写死的 12
    def __init__(self, in_channels, out_channels, num_nodes, A_hat, t_in=12):
        super(STGCNBlock, self).__init__()
        self.register_buffer('A_hat', A_hat)
        self.temporal1 = TemporalConvNet(num_inputs=in_channels, num_channels=out_channels)
        self.odeg = ODEG(out_channels[-1], t_in, self.A_hat, time=1) # 动态传入时间步长
        self.temporal2 = TemporalConvNet(num_inputs=out_channels[-1], num_channels=out_channels)
        # 使用 Channel 维度归一化，避免 Metis 切分导致节点数不一致报错
        self.batch_norm = nn.BatchNorm2d(out_channels[-1])

    def forward(self, X):
        t = self.temporal1(X)
        t = self.odeg(t)
        t = self.temporal2(F.relu(t)) 
        
        t = t.permute(0, 3, 1, 2)
        t = self.batch_norm(t)
        t = t.permute(0, 2, 3, 1)

        return t

class ODEGCN(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, A_sp_hat, A_se_hat):
        super(ODEGCN, self).__init__()
        
        # 传递动态 num_timesteps_input (即 t_in)
        self.sp_blocks = nn.ModuleList([
            nn.Sequential(
                STGCNBlock(in_channels=num_features, out_channels=[64, 32, 64], num_nodes=num_nodes, A_hat=A_sp_hat, t_in=num_timesteps_input),
                STGCNBlock(in_channels=64, out_channels=[64, 32, 64], num_nodes=num_nodes, A_hat=A_sp_hat, t_in=num_timesteps_input)
            ) for _ in range(3)
        ])
        
        self.se_blocks = nn.ModuleList([
            nn.Sequential(
                STGCNBlock(in_channels=num_features, out_channels=[64, 32, 64], num_nodes=num_nodes, A_hat=A_se_hat, t_in=num_timesteps_input),
                STGCNBlock(in_channels=64, out_channels=[64, 32, 64], num_nodes=num_nodes, A_hat=A_se_hat, t_in=num_timesteps_input)
            ) for _ in range(3)
        ])

        # 因为双路特征最后会 concat，所以第一层的输入维度是 num_timesteps_input * 64 * 2
        self.pred = nn.Sequential(
            nn.Linear(num_timesteps_input * 64 * 2, num_timesteps_output * 32),
            nn.ReLU(),
            nn.Linear(num_timesteps_output * 32, num_timesteps_output)
        )

        self.batch_counter = 0

    def _max_pool_blocks(self, blocks, x):
        pooled = None
        for blk in blocks:
            out = blk(x)
            pooled = out if pooled is None else torch.maximum(pooled, out)
        return pooled

    def forward(self, x):
        if x.shape[1] == 12 and x.shape[2] != 12:
             x = x.permute(0, 2, 1, 3)
        if x.dim() == 3:
             x = x.unsqueeze(-1)

        # 空间分支和语义分支分别独立提特征并池化
        sp_pooled = self._max_pool_blocks(self.sp_blocks, x)

        se_pooled = self._max_pool_blocks(self.se_blocks, x)

        # 沿特征维度拼接 (batch, N, T, 64) -> (batch, N, T, 128)
        fused = torch.cat([sp_pooled, se_pooled], dim=-1)
        fused = fused.reshape((fused.shape[0], fused.shape[1], -1))

        prediction = self.pred(fused)
        prediction = prediction.unsqueeze(-1)

        return prediction
