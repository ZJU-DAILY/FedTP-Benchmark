import torch
import torch.nn as nn


class FedSTGClientManager:
    """
    FedSTG client-side state manager.
    """

    def __init__(self, ctx, model, optimizer, loss_fn, alpha, device):
        self.ctx = ctx
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.alpha = alpha
        self.device = device

    def compute_total_loss(self, pred, y, L_k):
        """
        Compute prediction loss and adaptive regularization loss.
        """
        if y.dim() == 4 and y.shape[-1] == 1:
            y = y.squeeze(-1)
        if pred.dim() == 4 and pred.shape[-1] == 1:
            pred = pred.squeeze(-1)

        loss_p = self.loss_fn(pred, y)
        reg_loss = self.alpha * L_k
        total_loss = loss_p + reg_loss
        return total_loss, loss_p, reg_loss

    def extract_gru_parameters(self):
        """
        Only share GRU parameters for personalized aggregation.
        """
        gru_params = {}
        for name, param in self.model.named_parameters():
            if "gru" in name:
                gru_params[name] = param.data.cpu().clone()
        return gru_params

    def load_global_gru_parameters(self, global_gru_params):
        """
        Load aggregated GRU parameters back into the local model.
        """
        local_state_dict = self.model.state_dict()
        for name, param_tensor in global_gru_params.items():
            if name in local_state_dict:
                local_state_dict[name].copy_(param_tensor.to(self.device))
        self.model.load_state_dict(local_state_dict)

    def step_epoch_start(self):
        """
        Cache previous TP-Bank parameters for the next regularization term.
        """
        self.model.update_B_prev()
