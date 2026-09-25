import torch
import torch.nn as nn
import numpy as np
from .CNFGNN_base import GraphNet, GRUSeq2Seq

class CNFGNN(nn.Module):
    # ⚠️ 注意这里：增加了 edge_weight=None 参数
    def __init__(self, num_nodes, in_dim, out_dim, hidden_dim, edge_index, edge_weight=None, dropout=0.0): 
        super(CNFGNN, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.LongTensor(edge_index)
        self.register_buffer('edge_index', edge_index)
        
        # ================= 新增：处理并注册 edge_weight =================
        if edge_weight is None:
            # 兜底：如果没有传入，默认给全 1，避免 MLP 读到 0
            edge_weight = torch.ones((edge_index.shape[1], 1), dtype=torch.float32)
        elif not isinstance(edge_weight, torch.Tensor):
            edge_weight = torch.FloatTensor(edge_weight)
            
        # 强制规范化形状为 [num_edges, 1]
        if edge_weight.dim() == 1:
            edge_weight = edge_weight.view(-1, 1)
            
        self.register_buffer('edge_weight', edge_weight)
        # ==============================================================
        
        # 1. 客户端模型
        self.client_model = GRUSeq2Seq(
            input_size=1, 
            hidden_size=hidden_dim,
            output_size=1, 
            dropout=dropout
        )
        
        # 2. 服务器模型
        self.server_model = GraphNet(
            node_input_size=hidden_dim, 
            edge_input_size=1,          
            global_input_size=hidden_dim,
            hidden_size=256,            
            updated_node_size=128,
            updated_edge_size=128,
            updated_global_size=128,
            node_output_size=hidden_dim, 
            gn_layer_num=2,
            activation='ReLU',
            dropout=dropout
        )
        
        self.predictor = nn.Linear(1, out_dim)

    # --- Client Encoder --- (保持不变)
    def forward_client_encoder(self, x):
        if x.ndim == 4 and x.shape[1] == self.num_nodes and x.shape[2] != self.num_nodes:
            x = x.permute(0, 2, 1, 3)
            
        B, T_in, N, F = x.shape
        x_reshaped = x.permute(1, 0, 2, 3).reshape(T_in, B*N, F)
        _, h_encode = self.client_model.encoder(x_reshaped) 
        
        return h_encode, x_reshaped

    # --- Server GNN --- (修改这部分)
    def forward_server_gnn(self, h_encode_global, batch_size, total_nodes):
        gnn_input = h_encode_global 
        
        # 扩展 edge_index
        edge_index_batch = self.expand_edge_index(self.edge_index, batch_size, total_nodes)
        
        # ================= 新增：扩展 edge_weight =================
        edge_weight_batch = self.edge_weight.repeat(batch_size, 1) # [num_edges * batch_size, 1]
        # ==========================================================
        
        device = h_encode_global.device
        batch_idx = self.get_batch_idx(batch_size, total_nodes, device)
        
        # ---> 修复：传入 edge_weight_batch 替换掉原来的 None
        h_spatial = self.server_model(gnn_input, edge_index_batch, edge_weight_batch, batch_idx)
        return h_spatial

    # --- 后面的代码完全保持不变 ---
    def forward_client_decoder(self, x_reshaped, y, h_encode, h_spatial):
        B_N = x_reshaped.shape[1]
        F = x_reshaped.shape[2]
        B = B_N // self.num_nodes 
        N = self.num_nodes

        y_reshaped = None
        if y is not None:
             if y.ndim == 4 and y.shape[1] == self.num_nodes and y.shape[2] != self.num_nodes:
                 y = y.permute(0, 2, 1, 3)
             y_reshaped = y.permute(1, 0, 2, 3).reshape(self.out_dim, B*N, F)
        
        out_gru = self.client_model(x_reshaped, y_reshaped, h_encode, h_spatial, horizon=self.out_dim)
        
        out = out_gru.reshape(self.out_dim, B, N, 1).permute(1, 2, 0, 3)
        return out

    def forward(self, data, labels=None):
        pass 

    def expand_edge_index(self, edge_index, batch_size, num_nodes):
        num_edges = edge_index.shape[1]
        big_edge_index = edge_index.repeat(1, batch_size)
        offset = torch.arange(batch_size, device=edge_index.device) * num_nodes
        offset = offset.repeat_interleave(num_edges)
        big_edge_index = big_edge_index + offset
        return big_edge_index

    def get_batch_idx(self, batch_size, num_nodes, device):
        batch_idx = torch.arange(batch_size, device=device).repeat_interleave(num_nodes)
        return batch_idx