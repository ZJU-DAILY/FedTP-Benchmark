import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. AVWGCN (图卷积层 - 联邦版)
# ==========================================
class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, embed_dim, poly_k):
        super(AVWGCN, self).__init__()
        self.poly_k = poly_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

        nn.init.xavier_normal_(self.weights_pool)
        nn.init.zeros_(self.bias_pool)

    def fast_cartesian_prod(self, A, B):
        # A: [N, d1], B: [N, d2] -> [N, d1*d2]
        return torch.einsum("ni,nj->nij", A, B).flatten(start_dim=1)

    def transform(self, k, E):
        transformed = torch.ones(E.shape[0], 1, device=E.device, dtype=E.dtype)
        cur_pow = E
        while k > 0:
            if k % 2 == 1:
                transformed = self.fast_cartesian_prod(transformed, cur_pow)
            cur_pow = self.fast_cartesian_prod(cur_pow, cur_pow)
            k //= 2
        return transformed

    def forward(self, x, node_embeddings, poly_coefficients, model_ref):
        E = node_embeddings
        H = x

        if model_ref.cached_transformed_E is None:
            raise RuntimeError("FedGTP_Model.cached_transformed_E 为空，请检查 forward 流程。")

        transformed_E = model_ref.cached_transformed_E
        EH = [torch.einsum("dn,bnc->bdc", e.transpose(0, 1), H) for e in transformed_E]
        if getattr(model_ref, "privacy_capture_eh", False):
            if getattr(model_ref, "privacy_eh_records", None) is None:
                model_ref.privacy_eh_records = []
            model_ref.privacy_eh_records.append(tuple(EH))

        tag = f"{model_ref.batch_tag}_c{model_ref.comm_step}"
        model_ref.comm_step += 1
        model_ref.last_comm_tag = tag

        if model_ref.debug_comm and model_ref.should_log_comm(tag):
            print(f"[FedGTP Model] -> comm_hook tag={tag}", flush=model_ref.debug_flush)

        EH_cpu = [eh.detach().to(torch.float16).cpu().contiguous() for eh in EH]
        comm_hook = getattr(model_ref, "comm_hook", None)
        if comm_hook is None:
            comm_hook = lambda eh_list, _tag: [t.clone() for t in eh_list]
        sum_EH_cpu = comm_hook(EH_cpu, tag)

        if model_ref.debug_comm and model_ref.should_log_comm(tag):
            print(f"[FedGTP Model] <- comm_hook done tag={tag}", flush=model_ref.debug_flush)

        sum_EH = [seh.to(E.device, dtype=H.dtype) for seh in sum_EH_cpu]

        Z_list = [torch.einsum("nd,bdc->bnc", transformed_E[i], sum_EH[i]) for i in range(self.poly_k + 1)]
        Z_stack = torch.stack(Z_list)
        Z_poly = torch.einsum("ak,kbnc->abnc", poly_coefficients, Z_stack)[0]
        Z = H + Z_poly

        weights = torch.einsum("nd,dio->nio", E, self.weights_pool)
        bias = torch.matmul(E, self.bias_pool)
        x_gconv = torch.einsum("bni,nio->bno", Z, weights) + bias

        return x_gconv


# ==========================================
# 2. AGCRNCell & AVWDCRNN (时序模块)
# ==========================================
class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, embed_dim, poly_k):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in + self.hidden_dim, 2 * dim_out, embed_dim, poly_k)
        self.update = AVWGCN(dim_in + self.hidden_dim, dim_out, embed_dim, poly_k)

    def forward(self, x, state, node_embeddings, poly_coefficients, model_ref):
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)

        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings, poly_coefficients, model_ref))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)

        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings, poly_coefficients, model_ref))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, embed_dim, num_layers, poly_k):
        super(AVWDCRNN, self).__init__()
        self.node_num = node_num
        self.num_layers = num_layers

        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, embed_dim, poly_k))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, embed_dim, poly_k))

    def forward(self, x, init_state, node_embeddings, poly_coefficients, model_ref):
        seq_length = x.shape[1]
        current_inputs = x

        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](
                    current_inputs[:, t, :, :],
                    state,
                    node_embeddings,
                    poly_coefficients,
                    model_ref
                )
                inner_states.append(state)
            current_inputs = torch.stack(inner_states, dim=1)

        return current_inputs

    def init_hidden(self, batch_size):
        return [self.dcrnn_cells[i].init_hidden_state(batch_size) for i in range(self.num_layers)]


# ==========================================
# 3. FedGTP 主模型
# ==========================================
class FedGTP_Model(nn.Module):
    def __init__(self, num_nodes, max_nodes, in_dim, out_dim, feature_dim, hidden_dim, emb_dim=2, poly_k=2):
        super(FedGTP_Model, self).__init__()
        self.num_nodes = num_nodes
        self.max_nodes = max_nodes
        self.horizon = out_dim
        self.output_dim = 1
        self.poly_k = poly_k

        # 当前平台版就是 1 层
        self.num_layers = 1

        # PartialFedAvg 保护本地 embedding
        self.node_embeddings = nn.Parameter(torch.randn(self.max_nodes, emb_dim), requires_grad=True)
        self.poly_coefficients = nn.Parameter(torch.randn(1, poly_k + 1), requires_grad=True)

        self.encoder = AVWDCRNN(self.num_nodes, feature_dim, hidden_dim, emb_dim, self.num_layers, poly_k)
        self.end_conv = nn.Conv2d(1, self.horizon * self.output_dim, kernel_size=(1, hidden_dim), bias=True)

        # 联邦通信状态
        self.comm_hook = None
        self.batch_tag = ""
        self.comm_step = 0
        self.last_comm_tag = None
        self.cached_transformed_E = None
        if getattr(self, "comm_hook", None) is None:
            self.comm_hook = lambda eh_list, _tag: [t.clone() for t in eh_list]

        # 调试状态
        self.debug_comm = False
        self.debug_comm_full = False
        self.debug_flush = True

    def set_debug(self, enabled=False, full=False, flush=True):
        self.debug_comm = bool(enabled)
        self.debug_comm_full = bool(full)
        self.debug_flush = bool(flush)

    def should_log_comm(self, tag):
        if self.debug_comm_full:
            return True
        try:
            idx = int(str(tag).rsplit("_c", 1)[1])
        except Exception:
            return True
        total_comm = self.num_layers * 12 * 2  # 当前平台固定 t_in=12 时的日志边界控制
        return idx in (0, total_comm - 1)

    def forward(self, source, comm_hook=None, batch_tag=""):
        if comm_hook is None:
            comm_hook = lambda eh, tag: [t.clone() for t in eh]
        self.comm_hook = comm_hook
        self.batch_tag = batch_tag
        self.comm_step = 0
        self.last_comm_tag = None

        # 兼容 [B, N, T, C] -> [B, T, N, C]
        if source.shape[1] == self.num_nodes:
            source = source.permute(0, 2, 1, 3)

        real_embeddings = self.node_embeddings[:self.num_nodes, :]
        init_state = [state.to(source.device) for state in self.encoder.init_hidden(source.shape[0])]

        # 每个 batch 只缓存一次多项式展开
        self.cached_transformed_E = [
            self.encoder.dcrnn_cells[0].gate.transform(k, real_embeddings)
            for k in range(self.poly_k + 1)
        ]

        output = self.encoder(
            source,
            init_state,
            real_embeddings,
            self.poly_coefficients,
            self
        )

        output = output[:, -1:, :, :]
        output = self.end_conv(output)
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_nodes)
        output = output.permute(0, 1, 3, 2)
        return output

    def load_shared_params(self, shared_dict):
        """仅加载联邦共享参数，严格保护本地 node_embeddings。"""
        local_state = self.state_dict()
        for k, v in shared_dict.items():
            if "node_embeddings" not in k and k in local_state:
                local_state[k].copy_(v)
