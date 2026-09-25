import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from pytorch_wavelets import DWT1DForward, DWT1DInverse

# ==========================================
# 优化版组件: 避免重复计算图支持矩阵
# ==========================================

class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(2 * cheb_k * dim_in, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)

    def forward(self, x, support_set):
        """
        [性能优化]: support_set 直接由顶层预计算并传入，
        避免每层、每个时间步重复执行代价高昂的 torch.eye 和矩阵乘法。
        """
        x_g = []
        for support in support_set:
            x_g.append(torch.einsum("nm,bmc->bnc", support, x))
        
        x_g = torch.cat(x_g, dim=-1)
        x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias
        return x_gconv

class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k)
        self.update = AGCN(dim_in + self.hidden_dim, dim_out, cheb_k)

    def forward(self, x, state, support_set):
        # 移除了 state.to(x.device)，它本来就应该在同一个设备上，省去检测开销
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, support_set))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, support_set))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size, device):
        # [性能优化]: 直接在目标设备分配显存
        return torch.zeros(batch_size, self.node_num, self.hidden_dim, device=device)

class ADCRNN_Encoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(ADCRNN_Encoder, self).__init__()
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, x, init_state, support_set):
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, support_set)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size, device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size, device))
        return init_states

class ADCRNN_Decoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(ADCRNN_Decoder, self).__init__()
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, xt, init_state, support_set):
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.dcrnn_cells[i](current_inputs, init_state[i], support_set)
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden


# ==========================================
#  主模型: DCRNN_TP (FedTPS)
# ==========================================

class DCRNN_TP(nn.Module):
    def __init__(self, num_nodes, input_dim=1, output_dim=1, horizon=12, rnn_units=64, num_layers=2, cheb_k=2,
                 ycov_dim=1, pattern_num=20, pattern_dim=64, cl_decay_steps=2000, use_curriculum_learning=True, wave="coif1"):
        super(DCRNN_TP, self).__init__()
        
        self.num_nodes = int(num_nodes)
        self.input_dim = int(input_dim)
        self.rnn_units = int(rnn_units)
        self.output_dim = int(output_dim)
        self.horizon = int(horizon)
        self.num_layers = int(num_layers)
        self.cheb_k = int(cheb_k)
        self.ycov_dim = int(ycov_dim)
        self.cl_decay_steps = int(cl_decay_steps)
        self.use_curriculum_learning = use_curriculum_learning
        self.wave = wave

        self.register_buffer('batches_seen', torch.LongTensor([0]))

        self.node_embed = 20
        self.We1 = nn.Parameter(torch.randn(self.num_nodes, self.node_embed))
        self.We2 = nn.Parameter(torch.randn(self.num_nodes, self.node_embed))
        nn.init.xavier_normal_(self.We1)
        nn.init.xavier_normal_(self.We2)

        # Patterns
        self.pattern_num = pattern_num
        self.pattern_dim = pattern_dim
        self.Patterns = nn.Parameter(torch.randn(self.pattern_num, self.pattern_dim))
        self.Wq = nn.Parameter(torch.randn(self.rnn_units, self.pattern_dim))
        nn.init.xavier_normal_(self.Patterns)
        nn.init.xavier_normal_(self.Wq)

        # Encoder & Decoder
        self.encoder = ADCRNN_Encoder(self.num_nodes, self.input_dim, self.rnn_units, self.cheb_k, self.num_layers)
        self.encoder_D = ADCRNN_Encoder(self.num_nodes, self.input_dim, self.rnn_units, self.cheb_k, self.num_layers)
        self.decoder_dim = self.rnn_units + self.pattern_dim
        self.decoder = ADCRNN_Decoder(self.num_nodes, self.output_dim + self.ycov_dim, self.decoder_dim, self.cheb_k, self.num_layers)
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim))

        # [性能优化]: 将 DWT 初始化移至 __init__，复用模块，极其节省开销！
        self.dwt = DWT1DForward(wave=self.wave, J=1, mode="symmetric")
        self.idwt = DWT1DInverse(wave=self.wave, mode="symmetric")

    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def forward(self, x, y_cov=None, labels=None, **kwargs):
        # 1. 维度清洗
        if x.dim() == 5: 
            x = x.squeeze(-1) 
        
        if x.shape[1] == self.num_nodes or (x.shape[1] != self.horizon and x.shape[2] == self.horizon):
            x = x.permute(0, 2, 1, 3)

        batch_size, seq_len, num_nodes, _ = x.shape
        
        # [性能优化]: 直接在 GPU 原生创建张量，禁止 cpu -> gpu 同步
        if y_cov is None:
            y_cov = torch.zeros(batch_size, self.horizon, num_nodes, self.ycov_dim, device=x.device)

        if self.training:
            self.batches_seen += 1
        current_batches = self.batches_seen.item()

        # ======================================================================
        # [性能优化]: 预计算切比雪夫多项式，每个 Batch 只计算 1 次，而非 48 次！
        # ======================================================================
        g1 = F.softmax(F.relu(torch.mm(self.We1, self.We2.T)), dim=-1)
        g2 = F.softmax(F.relu(torch.mm(self.We2, self.We1.T)), dim=-1)
        
        support_set = []
        for support in [g1, g2]:
            support_ks = [torch.eye(support.shape[0], device=x.device), support]
            for k in range(2, self.cheb_k):
                support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2])
            support_set.extend(support_ks)

        # ==================== 小波变换 ====================
        if x.shape[-1] == 1:
            x_input = x.squeeze(-1).permute(0, 2, 1) # -> (B, N, T)
        else:
            x_input = x[..., 0].permute(0, 2, 1)

        # 直接调用 __init__ 中声明的模块
        xl, xh = self.dwt(x_input) 
        xh_zeros = [torch.zeros_like(h, device=x.device) for h in xh]
        x_l = self.idwt((xl, xh_zeros)) # -> (B, N, T)
        x_l = x_l.permute(0, 2, 1).unsqueeze(-1)

        # ==================== 编码器流 ====================
        init_state = self.encoder.init_hidden(x.shape[0], x.device)
        h_en, _ = self.encoder(x, init_state, support_set)
        h_t = h_en[:, -1, :, :] 

        init_state_D = self.encoder_D.init_hidden(x.shape[0], x.device)
        h_en_D, _ = self.encoder_D(x_l, init_state_D, support_set)
        h_t_D = h_en_D[:, -1, :, :]

        # ==================== 注意力匹配 ====================
        query = torch.matmul(h_t_D, self.Wq)
        att_score = torch.softmax(torch.matmul(query, self.Patterns.t()), dim=-1)
        h_att = torch.matmul(att_score, self.Patterns)
        h_t = torch.cat([h_t, h_att], dim=-1)

        # ==================== 解码器流 ====================
        ht_list = [h_t] * self.num_layers
        go = torch.zeros((x.shape[0], self.num_nodes, self.output_dim), device=x.device)
        out = []
        
        for t in range(self.horizon):
            decoder_input = torch.cat([go, y_cov[:, t, ...]], dim=-1)
            h_de, ht_list = self.decoder(decoder_input, ht_list, support_set)
            go = self.proj(h_de)
            out.append(go)
            
            if self.training and self.use_curriculum_learning and labels is not None:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(current_batches):
                    if labels.shape[1] == self.horizon:
                        go = labels[:, t, ...]
                    elif labels.shape[2] == self.horizon: 
                        go = labels[:, :, t, ...]
        
        output = torch.stack(out, dim=1)
        output = output.permute(0, 2, 1, 3)
        return output