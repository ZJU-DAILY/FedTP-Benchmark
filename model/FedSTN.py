import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

class MFDense(nn.Module):
    def __init__(self, input_dim, output_dim, k=16):
        super(MFDense, self).__init__()
        self.L = nn.Linear(input_dim, k, bias=False)
        self.U = nn.Linear(k, output_dim, bias=True)

    def forward(self, x):
        return self.U(F.relu(self.L(x)))

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class FedSTN(nn.Module):
    # 1. 初始化增加 ext_dim
    def __init__(self, num_nodes, input_dim, hidden_dim, out_dim, edge_index, output_features=1, ext_dim=21):
        super(FedSTN, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.output_features = output_features

        # 取消 register_buffer，改为普通属性，避免 Server 强制拉取导致维度崩溃
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.LongTensor(edge_index)
        self.edge_index = edge_index
        
        # RLCN Module
        self.lstm_rlcn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.mfdense = MFDense(hidden_dim, hidden_dim)
        self.conv_lstm = ConvLSTMCell(input_dim=hidden_dim, hidden_dim=hidden_dim, kernel_size=(1,1), bias=True)
        self.fc_rlcn = nn.Linear(hidden_dim, out_dim * output_features)

        # AMFN Module
        self.gat = GATConv(in_channels=input_dim, out_channels=hidden_dim, heads=2, concat=False)
        # 【关键修复】：GRU 的输入维度变为 hidden_dim + ext_dim (融合气象特征)
        self.gru_amfn = nn.GRU(input_size=hidden_dim + ext_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc_amfn = nn.Linear(hidden_dim, out_dim * output_features)

        # SCN Module
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.fc_scn = nn.Linear(hidden_dim, out_dim * output_features)
        
        self.fusion_weight = nn.Parameter(torch.ones(3, 1))

    # 快速生成“超级大图”的边索引 (保留你原有的方法)
    def _get_batched_edge_index(self, batch_size):
        if (
            hasattr(self, 'cached_batch_size')
            and self.cached_batch_size == batch_size
            and hasattr(self, 'batched_edge_index')
            and self.batched_edge_index.device == self.edge_index.device
        ):
            return self.batched_edge_index
            
        num_edges = self.edge_index.size(1)
        batched_edge_index = self.edge_index.repeat(1, batch_size)
        
        offset = torch.arange(batch_size, device=self.edge_index.device) * self.num_nodes
        offset = offset.view(-1, 1).repeat(1, num_edges).view(-1)
        
        self.cached_batch_size = batch_size
        self.batched_edge_index = batched_edge_index + offset
        return self.batched_edge_index

    def _external_sequence(self, x_ext, batch_size, time_steps, num_nodes, device, dtype):
        ext_dim = self.gru_amfn.input_size - self.hidden_dim
        if x_ext is None:
            ext_seq = torch.zeros(batch_size, time_steps, ext_dim, device=device, dtype=dtype)
        else:
            x_ext = x_ext.to(device=device, dtype=dtype)
            if x_ext.dim() == 2:
                x_ext = x_ext.unsqueeze(1)
            if x_ext.dim() != 3:
                x_ext = x_ext.reshape(batch_size, -1, x_ext.shape[-1])

            if x_ext.shape[-1] < ext_dim:
                pad = x_ext.new_zeros(*x_ext.shape[:-1], ext_dim - x_ext.shape[-1])
                x_ext = torch.cat([x_ext, pad], dim=-1)
            elif x_ext.shape[-1] > ext_dim:
                x_ext = x_ext[..., :ext_dim]

            if x_ext.shape[1] == time_steps:
                ext_seq = x_ext
            else:
                ext_seq = x_ext.mean(dim=1, keepdim=True).expand(-1, time_steps, -1)

        return ext_seq.unsqueeze(1).expand(-1, num_nodes, -1, -1).contiguous()

    # 【完全替换的 forward_phase1】：接收 x_ext 并接入缺失的网络层
    def forward_phase1(self, x, x_ext=None):
        self.edge_index = self.edge_index.to(x.device)
        
        if x.shape[1] == self.num_nodes and x.shape[2] != self.num_nodes:
            x = x.permute(0, 2, 1, 3) 
        
        batch_size, T, N, F_dim = x.shape
        
        # --- 1. RLCN (局部长效特征) ---
        x_reshaped = x.contiguous().view(batch_size * N, T, F_dim)
        lstm_out, _ = self.lstm_rlcn(x_reshaped)
        rlcn_feat = lstm_out[:, -1, :]
        rlcn_feat = self.mfdense(rlcn_feat) 
        
        # 接入 ConvLSTM
        rlcn_feat_4d = rlcn_feat.view(batch_size, N, self.hidden_dim).permute(0, 2, 1).unsqueeze(-1)
        h_cur = torch.zeros(batch_size, self.hidden_dim, N, 1, device=x.device)
        c_cur = torch.zeros(batch_size, self.hidden_dim, N, 1, device=x.device)
        h_next, c_next = self.conv_lstm(rlcn_feat_4d, (h_cur, c_cur))
        
        rlcn_feat_final = h_next.squeeze(-1).permute(0, 2, 1).contiguous().view(batch_size * N, self.hidden_dim)
        rlcn_out = self.fc_rlcn(rlcn_feat_final).view(batch_size, N, -1) 

        # =================【向量化提速核心区】=================
        x_last = x[:, -1, :, :] 
        x_last_flat = x_last.contiguous().view(batch_size * N, F_dim)
        batched_edge_index = self._get_batched_edge_index(batch_size).to(x.device)

        # --- 2. SCN (局部语义特征) ---
        g_out = F.relu(self.gcn1(x_last_flat, batched_edge_index))
        g_out = F.relu(self.gcn2(g_out, batched_edge_index))
        scn_out = self.fc_scn(g_out).view(batch_size, N, -1) 

        # --- 3. AMFN short-term hidden state ---
        # Paper: S_AMFN=[X_{t-tau_s},...,X_t].  Encode every short-term
        # input step and upload the final temporal hidden state h_s_i.
        gat_steps = []
        for step in range(T):
            x_step = x[:, step, :, :].contiguous().view(batch_size * N, F_dim)
            gat_step = self.gat(x_step, batched_edge_index).view(batch_size, N, self.hidden_dim)
            gat_steps.append(gat_step)
        gat_seq = torch.stack(gat_steps, dim=2)

        ext_seq = self._external_sequence(
            x_ext=x_ext,
            batch_size=batch_size,
            time_steps=T,
            num_nodes=N,
            device=x.device,
            dtype=x.dtype,
        )
        amfn_seq = torch.cat([gat_seq, ext_seq], dim=-1)
        amfn_seq = amfn_seq.contiguous().view(batch_size * N, T, -1)
        _, amfn_hidden = self.gru_amfn(amfn_seq)
        h_s_i = amfn_hidden[-1].view(batch_size, N, self.hidden_dim)
        
        return h_s_i, rlcn_out, scn_out
    # 【修复点 3】: 拆分前向传播 - 阶段 2 (接收 Server 融合的全局 AMFN 特征，进行最终预测)
    def forward_phase2(self, aggregated_h_s, rlcn_out, scn_out):
        batch_size = rlcn_out.shape[0]
        
        # 接收全局聚合的 h_s，输入本地预测器
        amfn_out = self.fc_amfn(aggregated_h_s)

        # Fusion 融合
        w = F.softmax(self.fusion_weight, dim=0)
        
        # 【核心修复】：去掉 torch.tanh()！让模型可以输出任何范围的实数，去完美匹配 StandardScaler
        final_out = (
            w[0] * rlcn_out + 
            w[1] * amfn_out + 
            w[2] * scn_out
        ) 
        
        return final_out.view(batch_size, self.num_nodes, -1, self.output_features)
