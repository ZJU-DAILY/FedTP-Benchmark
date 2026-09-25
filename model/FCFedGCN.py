import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FCGCN_Layer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(FCGCN_Layer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = self.linear(x) 
        if support.dim() == 3: 
            # 修复了原来 'nn,bnf->bnf' 的灾难级 Bug，恢复真正的图卷积
            output = torch.einsum('ij,bjf->bif', adj, support)
        else: 
            output = torch.mm(adj, support)
            
        if self.bias is not None:
            return output + self.bias
        return output

class FC_FedGCN_Traffic(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim, hidden_dim, fca_dim, adj_matrix=None):
        super(FC_FedGCN_Traffic, self).__init__()
        self.num_nodes = num_nodes
        self.fca_dim = fca_dim
        
        # 【完全贴合论文 Eq.9】：拼接特征维度
        combined_dim = in_dim + fca_dim
        self.gc1 = FCGCN_Layer(combined_dim, hidden_dim, bias=True)
        self.gc2 = FCGCN_Layer(hidden_dim, out_dim, bias=True)
        
        self.dropout = 0.5
        
        if adj_matrix is not None:
            self.register_buffer('adj', torch.tensor(adj_matrix, dtype=torch.float32))
        else:
            self.register_buffer('adj', torch.eye(num_nodes))
            
        self.register_buffer('fca_features', torch.zeros(num_nodes, fca_dim))

    def set_fca_features(self, fca_features):
        with torch.no_grad():
            self.fca_features.copy_(fca_features)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(-1) 
            
        batch_size = x.shape[0]
        fca_batch = self.fca_features.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 【完全贴合论文 Eq.9】：严格拼接 X* = X ⊕ f
        x_enhanced = torch.cat([x, fca_batch], dim=-1)
        
        # 经过共享的一套 GCN
        h = F.relu(self.gc1(x_enhanced, self.adj))
        h = F.dropout(h, self.dropout, training=self.training)
        
        out = self.gc2(h, self.adj)
        return out.unsqueeze(-1)