from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.STFAM import Global_1D_CNN_Autoencoder, Global_2D_CNN_Autoencoder, STFAM_Client_Model
from privacy.baseline_adapters.common import (
    align_prediction_and_target,
    args_from_batch,
    fixed_quantized_prediction,
    dense_adj_from_batch,
    gradient_tuple,
    load_state_dict_if_available,
    loss_fn,
)
from privacy.privacy_data import TOTAL_NODES_MAP


class STFAMPrivacyAdapter:
    name = "STFAM"
    default_attack = "model_update"
    requires_dummy_y = True
    # The uploaded local-extractor update is produced from a mini-batch loss.
    batch_aggregate_leak = True
    # STFAM trains its local feature extractor with Adam. This smoothed
    # epsilon is used only for differentiating the one-step attack surrogate;
    # it keeps the second-order derivative finite at inactive parameters.
    attack_local_adam_epsilon = 1e-4
    # The reconstruction target is the conditional local extractor update,
    # which is named `params` by lib/stfam_strategy.py.  The setup-only TSVD
    # vector is not a sample-level reconstruction target.
    dp_payload_group = "params"

    def build_model(self, args: Any, batch: Any) -> nn.Module:
        num_nodes = int(batch.real_x.shape[1])
        embed_dim = int(getattr(args, "stfam_embed_dim", 64))
        model = STFAM_Client_Model(
            t_in=args.t_in,
            num_local_nodes=num_nodes,
            embed_dim=embed_dim,
            in_channels=args.input_dim,
            pred_steps=args.t_out,
            out_channels=args.output_dim,
        ).to(args.device)

        adj_prob = dense_adj_from_batch(batch, num_nodes, args.device)
        model.register_buffer("privacy_stfam_adj_prob", adj_prob)
        model.register_buffer("privacy_stfam_U", self._build_u_matrix(batch, adj_prob, args))
        v_d, p_embed = self._build_global_embeddings(batch, args, embed_dim)
        model.register_buffer("privacy_stfam_V_D", v_d)
        model.register_buffer("privacy_stfam_P_embed", p_embed)

        load_state_dict_if_available(model, args)
        return model

    def _training_windows(self, batch: Any) -> torch.Tensor | None:
        dataset = getattr(batch, "dataset", None)
        tensors = getattr(dataset, "tensors", None)
        if tensors and torch.is_tensor(tensors[0]):
            x_all = tensors[0].detach()
            if x_all.dim() == 4 and x_all.shape[1] == batch.real_x.shape[1]:
                return x_all

        if dataset is None or not hasattr(dataset, "__len__"):
            return None

        xs = []
        for idx in range(len(dataset)):
            sample = dataset[idx]
            if not isinstance(sample, (tuple, list)) or len(sample) < 1:
                return None
            x_c = sample[0]
            if not torch.is_tensor(x_c):
                x_c = torch.as_tensor(x_c)
            if x_c.dim() != 3:
                return None
            # UnifiedTrafficDataset graph view returns [T,N,C]; STFAM uses [N,T,C].
            args = getattr(batch, "args", None)
            t_in = getattr(args, "t_in", None)
            if t_in is not None and x_c.shape[0] == t_in:
                x_c = x_c.permute(1, 0, 2).contiguous()
            xs.append(x_c.detach().cpu())

        if not xs:
            return None
        return torch.stack(xs, dim=0)

    def _build_u_matrix(self, batch: Any, adj_prob: torch.Tensor, args: Any) -> torch.Tensor:
        x_ref = self._training_windows(batch)
        if x_ref is None:
            x_ref = batch.real_x.detach().cpu()
        x_ref = x_ref.to(device=args.device, dtype=torch.float32)
        # x_ref: [Samples, N, T, C] -> total_flow: [C, N]
        total_flow = x_ref.sum(dim=(0, 2)).permute(1, 0)
        return total_flow.unsqueeze(-1) * adj_prob.unsqueeze(0)

    def _build_global_embeddings(
        self,
        batch: Any,
        args: Any,
        embed_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from lib.stfam_loader import build_global_data

        dataset_name = getattr(args, "dataset_name", "")
        num_global_nodes = TOTAL_NODES_MAP.get(dataset_name)
        if num_global_nodes is None:
            return (
                torch.zeros(1, embed_dim, device=args.device),
                torch.zeros(1, embed_dim, device=args.device),
            )

        project_root = str(getattr(args, "project_root", "")) or "."
        d_tensor, p_tensor = build_global_data(dataset_name, num_global_nodes, project_root)

        g2d = Global_2D_CNN_Autoencoder(1, embed_dim).to(args.device)
        g1d = Global_1D_CNN_Autoencoder(1, embed_dim).to(args.device)
        optimizer = torch.optim.Adam(
            list(g2d.parameters()) + list(g1d.parameters()),
            lr=float(getattr(args, "stfam_global_lr", 1e-3)),
        )
        mse_loss = nn.MSELoss()
        d_input = d_tensor.to(args.device).unsqueeze(0)
        p_input = p_tensor.to(args.device).view(1, 1, -1)

        pretrain_epochs = max(0, int(getattr(args, "stfam_global_pretrain_epochs", 200)))
        g2d.train()
        g1d.train()
        for _ in range(pretrain_epochs):
            optimizer.zero_grad(set_to_none=True)
            _, d_recon = g2d(d_input)
            _, p_recon = g1d(p_input)
            loss = mse_loss(d_recon, d_input) + mse_loss(p_recon, p_input)
            loss.backward()
            optimizer.step()

        selected_nodes = torch.as_tensor(batch.selected_nodes, dtype=torch.long, device=args.device)
        sub_d = d_tensor[0].to(args.device)[selected_nodes][:, selected_nodes].unsqueeze(0).unsqueeze(0)
        sub_p = p_tensor.to(args.device)[selected_nodes, :].view(1, 1, -1)

        g2d.eval()
        g1d.eval()
        with torch.no_grad():
            v_d, _ = g2d(sub_d)
            p_embed, _ = g1d(sub_p)
        return v_d.detach(), p_embed.detach()

    def _make_stfam_inputs(
        self,
        model: nn.Module,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 4:
            raise ValueError(f"STFAM expects x=[B,N,T,C], got shape={tuple(x.shape)}")

        adj_prob = model.privacy_stfam_adj_prob.to(device=x.device, dtype=x.dtype)
        u_base = model.privacy_stfam_U.to(device=x.device, dtype=x.dtype)
        v_d = model.privacy_stfam_V_D.to(device=x.device, dtype=x.dtype)
        p_embed = model.privacy_stfam_P_embed.to(device=x.device, dtype=x.dtype)

        # Match lib.stfam_loader.STFAM_Client_Dataset.__getitem__:
        # x_i [N,T,C] -> flow_i [T,N,C]
        # agg_i = A_prob @ flow_i -> [T,N,C]
        # Tr_i = cat(flow_i, agg_i, dim=-1).reshape(T, N*C*2)
        flow = x.permute(0, 2, 1, 3).contiguous()
        agg = torch.einsum("ij,btjc->btic", adj_prob, flow)
        tr = torch.cat([flow, agg], dim=-1).reshape(x.shape[0], x.shape[2], -1)

        u = u_base.unsqueeze(0).expand(x.shape[0], -1, -1, -1).contiguous()
        return tr, u, v_d, p_embed

    def compute_leak(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        batch: Any,
        create_graph: bool,
        leak_type: str,
    ):
        args = args_from_batch(batch)
        tr, u, v_d, p_embed = self._make_stfam_inputs(model, x)

        was_training = model.training
        model.train()
        try:
            with torch.backends.cudnn.flags(enabled=False):
                pred, tr_recon, u_recon = model(tr, u, v_d, p_embed)
        finally:
            model.train(was_training)

        pred, target = align_prediction_and_target(pred, y)
        if leak_type == "activation" and bool(getattr(args, "quantized_prediction_sidechannel", False)):
            return fixed_quantized_prediction(pred, args)
        task_loss = loss_fn(args)(pred, target)
        recon_loss = 0.1 * F.mse_loss(tr_recon, tr) + 0.1 * F.mse_loss(u_recon, u)
        total_loss = task_loss + recon_loss

        if leak_type == "gradient":
            return gradient_tuple(model=model, loss=total_loss, create_graph=create_graph)

        if leak_type != "model_update":
            raise ValueError("STFAM supports gradient diagnostics and model_update reconstruction.")

        named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        local_named_params = [(name, param) for name, param in named_params if "local_" in name]
        if not local_named_params:
            local_named_params = named_params

        params = [param for _, param in local_named_params]
        grads = torch.autograd.grad(
            total_loss,
            params,
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=True,
        )

        lr = float(getattr(args, "local_update_lr", getattr(args, "lr", 1e-3)))
        wd_arg = getattr(args, "local_update_wd", None)
        weight_decay = float(getattr(args, "wd", 0.0) if wd_arg is None else wd_arg)
        eps = self.attack_local_adam_epsilon
        updates = []
        for param, grad in zip(params, grads):
            grad = torch.zeros_like(param) if grad is None else grad
            if weight_decay != 0.0:
                grad = grad + weight_decay * param
            # The actual STFAM client uses Adam. At the first local step,
            # bias correction yields m_hat=g and v_hat=g^2. Smooth the
            # denominator only for the higher-order reconstruction derivative.
            updates.append(-lr * grad / torch.sqrt(grad.square() + eps ** 2))
        return tuple(updates)
