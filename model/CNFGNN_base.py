import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn import MetaLayer

# --- 1. GNN Components (From GraphNets.py) ---

class MLP_GN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, 
        hidden_layer_num=2, activation='ReLU', dropout=0.0):
        super().__init__()
        self.net = []
        last_layer_size = input_size
        for _ in range(hidden_layer_num):
            self.net.append(nn.Linear(last_layer_size, hidden_size))
            self.net.append(getattr(nn, activation)())
            self.net.append(nn.Dropout(p=dropout))
            last_layer_size = hidden_size
        self.net.append(nn.Linear(last_layer_size, output_size))
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)

class EdgeModel(nn.Module):
    def __init__(self, node_input_size, edge_input_size, global_input_size, 
        hidden_size, edge_output_size, activation, dropout):
        super(EdgeModel, self).__init__()
        edge_mlp_input_size = 2 * node_input_size + edge_input_size + global_input_size
        self.edge_mlp = MLP_GN(edge_mlp_input_size, hidden_size, edge_output_size, 2, activation, dropout)

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr], -1)
        if u is not None:
            out = torch.cat([out, u[batch]], -1)
        return self.edge_mlp(out)

class NodeModel(nn.Module):
    def __init__(self, node_input_size, edge_input_size, global_input_size,
        hidden_size, node_output_size, activation, dropout):
        super(NodeModel, self).__init__()
        node_mlp_input_size = node_input_size + edge_input_size + global_input_size
        self.node_mlp = MLP_GN(node_mlp_input_size, hidden_size, node_output_size, 2, activation, dropout)

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        received_msg = scatter_add(edge_attr, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, received_msg], dim=-1)
        if u is not None:
            out = torch.cat([out, u[batch]], dim=-1)
        return self.node_mlp(out)

class GlobalModel(nn.Module):
    def __init__(self, node_input_size, edge_input_size, global_input_size,
        hidden_size, global_output_size, activation, dropout):
        super(GlobalModel, self).__init__()
        global_mlp_input_size = node_input_size + edge_input_size + global_input_size
        self.global_mlp = MLP_GN(global_mlp_input_size, hidden_size, global_output_size, 2, activation, dropout)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # 【修复点】 这里必须解包 edge_index 来获取 col
        row, col = edge_index 
        
        agg_node = scatter_add(x, batch, dim=0)
        agg_edge = scatter_add(scatter_add(edge_attr, col, dim=0, dim_size=x.size(0)), batch, dim=0)
        out = torch.cat([agg_node, agg_edge, u], dim=-1)
        return self.global_mlp(out)

class GraphNet(nn.Module):
    def __init__(self, node_input_size, edge_input_size, global_input_size, 
        hidden_size, updated_node_size, updated_edge_size, updated_global_size,
        node_output_size, gn_layer_num, activation='ReLU', dropout=0.0):
        super().__init__()
        self.global_input_size = global_input_size
        self.net = []
        last_node_input_size = node_input_size
        last_edge_input_size = edge_input_size
        last_global_input_size = global_input_size
        for _ in range(gn_layer_num):
            edge_model = EdgeModel(last_node_input_size, last_edge_input_size, last_global_input_size, hidden_size, updated_edge_size, activation, dropout)
            last_edge_input_size += updated_edge_size
            node_model = NodeModel(last_node_input_size, updated_edge_size, last_global_input_size, hidden_size, updated_node_size, activation, dropout)
            last_node_input_size += updated_node_size
            global_model = GlobalModel(updated_node_size, updated_edge_size, last_global_input_size, hidden_size, updated_global_size, activation, dropout)
            last_global_input_size += updated_global_size
            self.net.append(MetaLayer(edge_model, node_model, global_model))
        self.net = nn.ModuleList(self.net)
        self.node_out_net = nn.Linear(last_node_input_size, node_output_size)
    
    def forward(self, x, edge_index, edge_attr, batch):
        if edge_attr is None:
            num_edges = edge_index.size(1)
            edge_attr = x.new_zeros((num_edges, 1))
        
        max_batch = int(batch.max().item()) if batch.numel() > 0 else 0
        u = x.new_zeros(max_batch + 1, self.global_input_size)
        
        for net in self.net:
            updated_x, updated_edge_attr, updated_u = net(x, edge_index, edge_attr, u, batch)
            x = torch.cat([updated_x, x], dim=-1)
            edge_attr = torch.cat([updated_edge_attr, edge_attr], dim=-1)
            u = torch.cat([updated_u, u], dim=-1)
        node_out = self.node_out_net(x)
        return node_out

# --- 2. GRU Components (Modified for Auto-regressive Logic) ---

class GRUSeq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.0,
        gru_num_layers=1):
        super().__init__()
        self.encoder = nn.GRU(input_size, hidden_size, num_layers=gru_num_layers, dropout=dropout)
        
        # Decoder 初始状态 = Encoder State + Graph State (Paper Sec 3.2.1)
        self.decoder = nn.GRU(input_size, 2 * hidden_size, num_layers=gru_num_layers, dropout=dropout)
        self.out_net = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x, y, h_encode, graph_encoding, horizon=12):
        """
        x: (T_in, BN, F)
        y: (T_out, BN, F) or None
        h_encode: (L, BN, H)
        graph_encoding: (BN, H)
        horizon: int, needed for inference when y is None
        """
        # 1. 准备 Decoder 的隐状态 (Combine Temporal & Spatial)
        L, BN, H = h_encode.shape
        # 将 graph_encoding (BN, H) 扩展并对齐到 (L, BN, H)
        graph_encoding_expanded = graph_encoding.unsqueeze(0).expand(L, -1, -1)
        
        # 拼接: [L, BN, 2H]
        decoder_hidden = torch.cat([h_encode, graph_encoding_expanded], dim=-1) 
        
        # 2. 准备第一个输入 (Encoder 的最后一步)
        last_x = x[-1:, :, :] # [1, BN, F]

        if y is not None:
             # --- 训练模式 (Teacher Forcing) ---
             # 使用真实标签作为输入
             decoder_input = torch.cat([last_x, y[:-1, :, :]], dim=0) # [T_out, BN, F]
             
             out_hidden, _ = self.decoder(decoder_input, decoder_hidden)
             out = self.out_net(out_hidden) # [T_out, BN, F]
             return out
        else:
             # --- 推理模式 (Auto-regressive) ---
             # 强制循环 horizon 次
             outputs = []
             current_input = last_x
             current_hidden = decoder_hidden
             
             for _ in range(horizon):
                 # 单步 GRU
                 out_step, current_hidden = self.decoder(current_input, current_hidden)
                 # 映射到输出空间
                 pred = self.out_net(out_step) # [1, BN, F]
                 outputs.append(pred)
                 # 下一步的输入就是当前的预测值
                 current_input = pred
            
             return torch.cat(outputs, dim=0) # [horizon, BN, F]