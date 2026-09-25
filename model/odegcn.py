import torch
from torch import nn
import torch.nn.functional as F

# from torchdiffeq import odeint 

class ODEFunc(nn.Module):
    def __init__(self, feature_dim, temporal_dim, adj):
        super(ODEFunc, self).__init__()
        self.register_buffer('adj', adj)
        self.x0 = None
        self.alpha = nn.Parameter(torch.tensor(0.8))
        self.beta = 0.6
        self.w = nn.Parameter(torch.eye(feature_dim))
        self.d = nn.Parameter(torch.zeros(feature_dim) + 1)
        self.w2 = nn.Parameter(torch.eye(temporal_dim))
        self.d2 = nn.Parameter(torch.zeros(temporal_dim) + 1)

    def forward(self, t, x):
        
        alpha = torch.sigmoid(self.alpha)
        
        # 1. 维度调整: (B, N, T, F) -> (B, T, F, N)
        x_perm = x.permute(0, 2, 3, 1) 
        
        # 2. 图卷积
        xa_perm = torch.matmul(x_perm, self.adj.t())
        
        # 3. 还原维度
        xa = xa_perm.permute(0, 3, 1, 2)

        d = torch.clamp(self.d, min=0, max=1)
        w = torch.mm(self.w * d, torch.t(self.w))
        w = (1 + self.beta) * w - self.beta * torch.mm(torch.mm(w, torch.t(w)), w)
        xw = torch.einsum('ijkl, lm->ijkm', x, w)

        d2 = torch.clamp(self.d2, min=0, max=1)
        w2 = torch.mm(self.w2 * d2, torch.t(self.w2))
        w2 = (1 + self.beta) * w2 - self.beta * torch.mm(torch.mm(w2, torch.t(w2)), w2)
        xw2 = torch.einsum('ijkl, km->ijml', x, w2)

        f = alpha / 2 * xa - x + xw - x + xw2 - x + self.x0
        return f

class ODEblock(nn.Module):
    def __init__(self, odefunc, t=torch.tensor([0,1])):
        super(ODEblock, self).__init__()
        self.t = t
        self.odefunc = odefunc

    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()

    def forward(self, x):
       
        # 1. 获取时间步长 dt (t通常是[0, 1]，所以 dt=1)
        t_start = self.t[0].type_as(x)
        t_end = self.t[-1].type_as(x)
        dt = t_end - t_start
        
        # 2. 计算导数 dx/dt
        derivative = self.odefunc(t_start, x)
        
        # 3. 欧拉更新: x_new = x + derivative * dt
        z = x + derivative * dt
        
        return z

class ODEG(nn.Module):
    def __init__(self, feature_dim, temporal_dim, adj, time):
        super(ODEG, self).__init__()
        # time=1
        self.odeblock = ODEblock(ODEFunc(feature_dim, temporal_dim, adj), t=torch.tensor([0, time]))

    def forward(self, x):
        self.odeblock.set_x0(x)
        z = self.odeblock(x)
        return F.relu(z)