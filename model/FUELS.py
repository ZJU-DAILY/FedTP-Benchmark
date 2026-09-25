import torch
import torch.nn as nn
import torch.nn.functional as F


class FUELS_Model(nn.Module):
    """
    FUELS final draft under fixed benchmark protocol.

    说明：
    - 不改外部 benchmark 统一数据协议
    - 在模型内部尽量逼近论文的双分支 encoder 语义
    - 由于外部只提供连续 T_in 窗口，这里的 periodicity 是“伪周期/扩张历史”
      而不是真正基于 period p 的原论文 pv_n
    """

    def __init__(
        self,
        num_nodes,
        in_dim,
        out_dim,
        hidden_dim,
        dr,
        batch_size,
        seq_len,
        pred_len,
        device,
        fuels_c=3,
        fuels_q=3,
        aug_noise_std=0.01,
        aug_mask_ratio=0.10,
        aug_shift_prob=0.50,
        aug_shift_pad_mode="edge",
    ):
        super(FUELS_Model, self).__init__()

        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.dr = dr
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.device = device

        self.fuels_c = fuels_c
        self.fuels_q = fuels_q

        self.aug_noise_std = aug_noise_std
        self.aug_mask_ratio = aug_mask_ratio
        self.aug_shift_prob = aug_shift_prob
        self.aug_shift_pad_mode = aug_shift_pad_mode

        self.gru_input_dim = num_nodes * in_dim
        self.gru_output_dim = num_nodes * out_dim

        # 双分支 encoder
        self.gru_c = nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=dr // 2,
            batch_first=True
        )
        self.gru_p = nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=dr // 2,
            batch_first=True
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(dr, dr // 2),
            nn.ReLU(),
            nn.Linear(dr // 2, self.gru_output_dim * pred_len)
        )

        # 动态过滤矩阵
        self.W_n = nn.Parameter(torch.zeros(batch_size, batch_size))
        nn.init.xavier_uniform_(self.W_n)

    # =========================================================
    # shape utils
    # =========================================================
    def _reshape_input(self, x):
        """
        支持:
            [B, N, T, C] -> [B, T, N*C]
            [B, T, N, C] -> [B, T, N*C]
            [B, T, F]    -> [B, T, F]
        """
        if x.dim() == 4:
            if x.shape[2] == self.seq_len:      # [B, N, T, C]
                x = x.permute(0, 2, 1, 3).contiguous()
            elif x.shape[1] == self.seq_len:    # [B, T, N, C]
                pass
            else:
                raise ValueError(f"Unexpected 4D input shape: {x.shape}")

            B, T, N, C = x.shape
            x = x.view(B, T, N * C)

        elif x.dim() == 3:
            pass
        else:
            raise ValueError(f"Unexpected input dim: {x.dim()}, shape={x.shape}")

        return x

    def build_periodic_indices(self, seq_len=None, c=None, q=None):
        """
        在连续窗口输入上构造:
        - closeness: 最后 c 个点
        - pseudo-periodicity: 在更早的历史段中均匀选 q 个点

        注意：
        这不是论文严格定义的 pv_n = (v_{k-qp}, ..., v_{k-p})
        因为你的公共数据管线没有给到真实 period-p 历史。
        """
        seq_len = seq_len if seq_len is not None else self.seq_len
        c = c if c is not None else self.fuels_c
        q = q if q is not None else self.fuels_q

        if c <= 0 or q <= 0:
            raise ValueError(f"Invalid fuels_c={c}, fuels_q={q}")

        if seq_len < c + q:
            raise ValueError(
                f"seq_len={seq_len} is too short for fuels_c={c}, fuels_q={q}"
            )

        cv_idx = list(range(seq_len - c, seq_len))

        earlier_len = seq_len - c
        if q == 1:
            pv_idx = [max(0, earlier_len - 1)]
        else:
            pv_idx = torch.linspace(
                0, earlier_len - 1, steps=q
            ).round().long().tolist()

        return cv_idx, pv_idx

    def _split_views(self, x, cv_idx=None, pv_idx=None):
        x = self._reshape_input(x)  # [B, T, F]

        if cv_idx is None or pv_idx is None:
            cv_idx, pv_idx = self.build_periodic_indices(seq_len=x.shape[1])

        cv_n = x[:, cv_idx, :]   # [B, c, F]
        pv_n = x[:, pv_idx, :]   # [B, q, F]
        return cv_n, pv_n

    # =========================================================
    # encoder / decoder
    # =========================================================
    def encode(self, x, cv_idx=None, pv_idx=None):
        """
        r_n = concat( GRU_c(cv_n), GRU_p(pv_n) )
        """
        cv_n, pv_n = self._split_views(x, cv_idx=cv_idx, pv_idx=pv_idx)

        _, hc = self.gru_c(cv_n)
        _, hp = self.gru_p(pv_n)

        hc = hc.squeeze(0)  # [B, dr/2]
        hp = hp.squeeze(0)  # [B, dr/2]

        r_n = torch.cat([hc, hp], dim=-1)  # [B, dr]
        return r_n

    def decode(self, r_n):
        B = r_n.shape[0]
        out = self.decoder(r_n)
        out = out.view(B, self.num_nodes, self.pred_len, self.out_dim)
        return out

    def forward(self, x, cv_idx=None, pv_idx=None):
        r_n = self.encode(x, cv_idx=cv_idx, pv_idx=pv_idx)
        return self.decode(r_n)

    # =========================================================
    # augmentation
    # =========================================================
    def temporal_shift_no_roll(self, x, shift=1, pad_mode="edge"):
        """
        非循环时间平移，避免 torch.roll 的时序穿越问题
        pad_mode:
            - edge: 用最早时间步填充
            - zero: 用 0 填充
        """
        if shift <= 0:
            return x

        if x.dim() == 4:
            if x.shape[2] == self.seq_len:   # [B, N, T, C]
                if pad_mode == "edge":
                    pad = x[:, :, 0:1, :].repeat(1, 1, shift, 1)
                else:
                    pad = torch.zeros_like(x[:, :, 0:shift, :])
                return torch.cat([pad, x[:, :, :-shift, :]], dim=2)

            elif x.shape[1] == self.seq_len: # [B, T, N, C]
                if pad_mode == "edge":
                    pad = x[:, 0:1, :, :].repeat(1, shift, 1, 1)
                else:
                    pad = torch.zeros_like(x[:, 0:shift, :, :])
                return torch.cat([pad, x[:, :-shift, :, :]], dim=1)

        elif x.dim() == 3:                   # [B, T, F]
            if pad_mode == "edge":
                pad = x[:, 0:1, :].repeat(1, shift, 1)
            else:
                pad = torch.zeros_like(x[:, 0:shift, :])
            return torch.cat([pad, x[:, :-shift, :]], dim=1)

        raise ValueError(f"Unexpected shape: {x.shape}")

    def make_augmented_view(
        self,
        x,
        noise_std=None,
        mask_ratio=None,
        shift_prob=None,
        shift_pad_mode=None
    ):
        """
        时序增强:
        1) 小幅噪声
        2) 时间掩码
        3) 非循环 shift
        """
        noise_std = self.aug_noise_std if noise_std is None else noise_std
        mask_ratio = self.aug_mask_ratio if mask_ratio is None else mask_ratio
        shift_prob = self.aug_shift_prob if shift_prob is None else shift_prob
        shift_pad_mode = self.aug_shift_pad_mode if shift_pad_mode is None else shift_pad_mode

        x_aug = x.clone()

        # (1) 小幅噪声
        if noise_std > 0:
            x_aug = x_aug + torch.randn_like(x_aug) * noise_std

        # (2) 时间掩码
        if mask_ratio > 0:
            if x_aug.dim() == 4:
                if x_aug.shape[2] == self.seq_len:  # [B, N, T, C]
                    B, N, T, C = x_aug.shape
                    num_mask = max(1, int(T * mask_ratio))
                    for b in range(B):
                        mask_idx = torch.randperm(T, device=x_aug.device)[:num_mask]
                        x_aug[b, :, mask_idx, :] = 0

                elif x_aug.shape[1] == self.seq_len:  # [B, T, N, C]
                    B, T, N, C = x_aug.shape
                    num_mask = max(1, int(T * mask_ratio))
                    for b in range(B):
                        mask_idx = torch.randperm(T, device=x_aug.device)[:num_mask]
                        x_aug[b, mask_idx, :, :] = 0

            elif x_aug.dim() == 3:  # [B, T, F]
                B, T, F = x_aug.shape
                num_mask = max(1, int(T * mask_ratio))
                for b in range(B):
                    mask_idx = torch.randperm(T, device=x_aug.device)[:num_mask]
                    x_aug[b, mask_idx, :] = 0

        # (3) 非循环 shift
        if shift_prob > 0 and torch.rand(1).item() < shift_prob:
            x_aug = self.temporal_shift_no_roll(
                x_aug,
                shift=1,
                pad_mode=shift_pad_mode
            )

        return x_aug

    # =========================================================
    # losses
    # =========================================================
    def compute_intra_loss(self, r_n, r_n_prime, tau=0.02, eps=1e-8):
        """
        工程稳定版 intra loss:
        - 保留可学习过滤矩阵 W_n
        - 使用 soft gating，而不是完全依赖硬截断
        """
        B = r_n.shape[0]
        W = self.W_n[:B, :B]

        r_n = F.normalize(r_n, dim=-1)
        r_n_prime = F.normalize(r_n_prime, dim=-1)

        sim = torch.matmul(r_n, r_n_prime.t()) / tau      # [B, B]
        SM = torch.exp(sim).clamp_min(eps)

        gate = torch.sigmoid(W)                           # soft gating

        pos = torch.diagonal(SM)

        neg_mask = ~torch.eye(B, dtype=torch.bool, device=SM.device)
        neg_weighted = SM * gate
        neg_sum = (neg_weighted * neg_mask.float()).sum(dim=1)

        loss = -torch.log(pos / (pos + neg_sum + eps))
        return loss.mean()

    def compute_inter_loss(self, r_n, PR_n, NR_n, tau=0.02, eps=1e-8):
        """
        Inter-client contrastive loss
        """
        B = r_n.shape[0]

        if PR_n.dim() == 1:
            PR_n = PR_n.unsqueeze(0).expand(B, -1)
        if NR_n.dim() == 1:
            NR_n = NR_n.unsqueeze(0).expand(B, -1)

        if PR_n.shape[0] != B:
            PR_n = PR_n[:B]
        if NR_n.shape[0] != B:
            NR_n = NR_n[:B]

        r_n = F.normalize(r_n, dim=-1)
        PR_n = F.normalize(PR_n, dim=-1)
        NR_n = F.normalize(NR_n, dim=-1)

        sim_pos = F.cosine_similarity(r_n, PR_n, dim=-1) / tau
        sim_neg = F.cosine_similarity(r_n, NR_n, dim=-1) / tau

        pos = torch.exp(sim_pos).clamp_min(eps)
        neg = torch.exp(sim_neg).clamp_min(eps)

        loss = -torch.log(pos / (pos + neg + eps))
        return loss.mean()