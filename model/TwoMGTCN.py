import torch
import torch.nn as nn
from fate.arch.protocol.mpc.functions.dropout import dropout
from torch_geometric.nn import GCNConv
import torch.optim as optim
import time
import numpy as np
from torch.nn.utils import weight_norm
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
import math
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import sys

sys.path.append("../")

from lib.load_dataset import load_dataset

# edge_index = torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]], dtype=torch.long)
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

from torch_geometric.data import Data


class GCNModule(nn.Module):
    def __init__(self, A, in_feats, num_layers=3, hidden_size=64, output_size=64):
        super().__init__()
        # self.adj = torch.tensor(adj_matrix, dtype=torch.float32)
        # self.gcn1 = GCNConv(in_feats, hidden_size)
        # self.gcn2 = GCNConv(hidden_size, output_size)
        self.A = A
        self.A_sparse = None
        layers = [nn.Linear(in_feats, hidden_size)]
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))

        self.model = nn.ModuleList(layers)

        # self.conv_weight1 = nn.Linear(in_feats, hidden_size)
        # self.conv_weight2 = nn.Linear(hidden_size, hidden_size)
        # self.conv_weight3 = nn.Linear(hidden_size, hidden_size)
        # self.conv_weight4 = nn.Linear(hidden_size, output_size)

        # self.dropout = dropout
        # self.fc1 = nn.Linear(in_feats, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, hidden_size)

    def set_sparse_topk(self, topk=32):
        if topk is None or topk <= 0:
            self.A_sparse = None
            return

        A = self.A
        if A.is_sparse:
            self.A_sparse = A.coalesce()
            return

        topk = min(int(topk), A.shape[1])
        values, indices = torch.topk(A, k=topk, dim=1)
        row_sums = values.sum(dim=1, keepdim=True).clamp_min(1e-12)
        values = values / row_sums
        rows = torch.arange(A.shape[0], device=A.device).unsqueeze(1).expand_as(indices)
        sparse_indices = torch.stack([rows.reshape(-1), indices.reshape(-1)], dim=0)
        self.A_sparse = torch.sparse_coo_tensor(
            sparse_indices,
            values.reshape(-1),
            size=A.shape,
            device=A.device,
            dtype=A.dtype,
        ).coalesce()

    def _graph_mm(self, x):
        if self.A_sparse is None:
            return self.A @ x
        B, N, F_dim = x.shape
        x_flat = x.permute(1, 0, 2).reshape(N, B * F_dim)
        out = torch.sparse.mm(self.A_sparse, x_flat)
        return out.reshape(N, B, F_dim).permute(1, 0, 2).contiguous()

    def forward(self, x):
        # x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # x, edge_index, edge_weight = data["x"], data["edge_index"][0], data["edge_weight"][0]

        for layer in self.model:
            x = layer(self._graph_mm(x))
            x = F.relu(x)

        # h_gcn = self.conv_weight1(A_norm @ x)
        # h_gcn = F.relu(h_gcn)
        # h_gcn = self.conv_weight2(A_norm @ h_gcn)
        # h_gcn = F.relu(h_gcn)
        # h_gcn = self.conv_weight3(A_norm @ h_gcn)
        # h_gcn = F.relu(h_gcn)
        # h_gcn = self.conv_weight4(A_norm @ h_gcn)
        # h_gcn = F.relu(h_gcn)
        # x = self.gcn1(x, edge_index, edge_weight)
        # x = F.relu(x)
        # x = F.dropout(x, self.dropout)
        # x = self.gcn2(x, edge_index, edge_weight)

        return x


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModule(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCNModule, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TwoMGTCN(nn.Module):
    def __init__(self, A, T_in=12, T_out=3, hidden_size=64, num_layers=3, nb_flow=2, ext_dim=21):
        super().__init__()
        self.nb_flow = nb_flow
        self.ext_dim = ext_dim
        self.T_in = T_in
        self.T_out = T_out  # 保存 T_out
        self.node_emb_dim = int(os.environ.get("TWOMGTCN_NODE_EMB_DIM", "0"))
        self.node_embedding = None
        if self.node_emb_dim > 0:
            self.node_embedding = nn.Embedding(A.shape[0], self.node_emb_dim)
        
        # 👉 1. 多模态特征融合层 (Flow: 12*2=24维, Ext: 3*21=63维. 融合后为 87 维)
        self.feature_fusion = nn.Linear((nb_flow * T_in) + (ext_dim * T_out) + self.node_emb_dim, hidden_size)
        
        # 👉 2. GCN 与 TCN 保持不变
        self.gcn = GCNModule(A, in_feats=hidden_size, num_layers=num_layers, hidden_size=hidden_size, output_size=hidden_size)
        self.tcn = TCNModule(hidden_size, tuple(hidden_size for _ in range(num_layers)), 3)
        self.fast_temporal = None
        
        # 👉 3. 最终预测层
        self.fc = nn.Linear(hidden_size * 2, T_out * nb_flow)

    def enable_fast_temporal(self):
        hidden_size = self.fc.in_features // 2
        self.fast_temporal = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        ).to(self.fc.weight.device)

    def forward(self, x_flow, x_ext=None):
        """
        x_flow: [Batch, T_in, N, 2]  (时间长度为 12)
        x_ext:  [Batch, T_out, 21]   (时间长度为 3)
        """
       
        x_flow = x_flow.permute(0, 2, 1, 3).contiguous()
        B, N, L, F_flow = x_flow.shape
        
       
        x_flow_flat = x_flow.view(B, N, -1)
        if self.ext_dim <= 0:
            x_ext_flat = x_flow.new_zeros(B, 0)
        else:
            if x_ext is None:
                x_ext = x_flow.new_zeros(B, self.T_out, self.ext_dim)
            x_ext_flat = x_ext.reshape(B, -1)
        x_ext_expanded = x_ext_flat.unsqueeze(1).expand(B, N, -1)
        if self.node_embedding is not None:
            node_ids = torch.arange(N, device=x_flow.device)
            node_emb = self.node_embedding(node_ids).unsqueeze(0).expand(B, N, -1)
            x_fused = torch.cat([x_flow_flat, x_ext_expanded, node_emb], dim=-1)
        else:
            x_fused = torch.cat([x_flow_flat, x_ext_expanded], dim=-1)
        fused_feat = self.feature_fusion(x_fused)
        spatial_feat = self.gcn(fused_feat)
        if self.fast_temporal is not None:
            temporal_feat = self.fast_temporal(fused_feat)
        else:
            x_temporal = fused_feat.view(B * N, 1, -1)
            temporal_feat = self.tcn(x_temporal.transpose(1, 2)).transpose(1, 2)[:, -1, :].view(B, N, -1)
        combined = torch.cat([spatial_feat, temporal_feat], dim=-1)
        out = self.fc(combined).reshape(B, N, self.T_out, self.nb_flow)
        out = out.permute(0, 2, 1, 3).contiguous() # 转回 [B, T_out, N, F]
        
        return out, fused_feat

def print_model_parameters(model):
    # if not only_num:
    for name, param in model.named_parameters():
        print('{} {} {}'.format(name, param.shape, param.requires_grad))
    total_num = sum([param.nelement() for param in model.parameters()])
    return total_num


if __name__ == '__main__':
    # A[A >= 0.5] = 1
    # A[A < 0.5] = 0
    #
    # edge_index = []
    # edge_weight = []
    # for i in range(grid_size * grid_size):
    #     for j in range(grid_size * grid_size):
    #         if A[i, j] == 1:
    #             edge_index.append((i, j))
    #             edge_weight.append(A[i, j])
    device = 'cuda:0'
    # edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
    # edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    trainSet, testSet, A_norm, scaler = load_dataset(dataset_name="groningen", feature_type="flow", normalizer='std',
                                                        T_in=12,
                                                        T_out=3, train_ratio=0.8, val_ratio=0.2, device=device)

    train_loader = torch.utils.data.DataLoader(trainSet, batch_size=512, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testSet, batch_size=512, shuffle=True)

    # trainSet = FedSTNDataset(torch.FloatTensor(X_train), A, torch.FloatTensor(Y_train))
    # testSet = FedSTNDataset(torch.FloatTensor(X_test), A, torch.FloatTensor(Y_test))
    # train_loader = torch.utils.data.DataLoader(trainSet, batch_size=64, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(testSet, batch_size=64, shuffle=False)
    T_in, T_out, n_flow = 12, 3, 1

    model = TwoMGTCN(A_norm, T_in, T_out, 256, 3, n_flow)
    model.to(device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.kaiming_uniform_(p)
        else:
            nn.init.uniform_(p)

    total_num = print_model_parameters(model)
    print('Total number of parameters: {}'.format(total_num))

    optimizer = torch.optim.NAdam(params=model.parameters(), lr=0.005)

    train_losses = []
    val_losses = []

    start_time = time.time()
    for ep in range(10000):
        epoch_loss = []
        model.train()
        for i, (data, y) in enumerate(train_loader):
            # if self.gpu_available:
            #     xc = xc.to(self.gpu)
            #     xp = xp.to(self.gpu)
            #     xt = xt.to(self.gpu)
            #     ext = ext.to(self.gpu)
            #     y = y.to(self.gpu)
            # data = data.to(device)
            y = y.to(device)
            ypred = model(data)
            loss = ((ypred - y) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            # if i % 50 == 0:
            #     print("Train [%.2fs] ep %d it %d, loss %.4f" % (time.time() - start_time, ep, i, loss.item()))
        print("Train [%.2fs] ep %d, loss %.4f" % (
        time.time() - start_time, ep, np.sqrt(np.mean(np.array(epoch_loss))) * scaler.metrics_coef))

        train_losses.append(np.sqrt(np.mean(np.array(epoch_loss))) * scaler.metrics_coef)

        epoch_loss = []
        model.eval()
        for i, (data, y) in enumerate(test_loader):
            # if self.gpu_available:
            #     xc = xc.to(self.gpu)
            #     xp = xp.to(self.gpu)
            #     xt = xt.to(self.gpu)
            #     ext = ext.to(self.gpu)
            #     y = y.to(self.gpu)
            # data = data.to(device)
            y = y.to(device)
            ypred = model(data)
            loss = ((ypred - y) ** 2).mean()
            epoch_loss.append(loss.item())
            # if i % 50 == 0:
            #     print("Test [%.2fs] ep %d it %d, loss %.4f" % (time.time() - start_time, ep, i, loss.item()))
        print("Test [%.2fs] ep %d, loss %.4f" % (
        time.time() - start_time, ep, np.sqrt(np.mean(np.array(epoch_loss))) * scaler.metrics_coef))
        val_losses.append(np.sqrt(np.mean(np.array(epoch_loss))) * scaler.metrics_coef)

    plt.plot(train_losses[1000:], label='train')
    plt.plot(val_losses[1000:], label='val')
    plt.legend()
    plt.show()

    def MAPE_torch1(output, label):
        mask = ~torch.isnan(label)
        mask = mask.float()
        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = 2.0 * (torch.abs(output - label) / (torch.abs(output) + torch.abs(label)))
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)

    def MAPE_torch2(output, label):
        loss = 2.0 * (torch.abs(output - label) / (torch.abs(output) + torch.abs(label)))
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)

        # print("upload test")
    model.eval()
    epoch_loss = []
    metrics_d = {'mae': 0,
                 'rmse': 0,
                 'mse': 0,
                 'mape': 0}

    mape_loss = []
    mape_loss1 = 0
    mape_loss2 = 0

    for i, (data, y) in enumerate(test_loader):
        ypred = model(data)
        loss = ((ypred - y) ** 2).mean()
        epoch_loss.append(loss.item())
        if i % 50 == 0:
            print("Test [%.2fs] it %d, loss %.4f" % (time.time() - start_time, i, loss.item()))
        mape_loss.append(((ypred - y) / y).abs().mean().item())
        y_cpu = y.cpu().detach().numpy()
        y_pred_cpu = ypred.cpu().detach().numpy()
        metrics_d['mae'] += np.sum(np.abs(y_cpu - y_pred_cpu)).item() / T_out / A_norm.shape[0]
        metrics_d['rmse'] += np.sum((y_cpu - y_pred_cpu) ** 2).item() / T_out / A_norm.shape[0]
        metrics_d['mse'] += np.sum((y_cpu - y_pred_cpu) ** 2).item() / T_out / A_norm.shape[0]
        metrics_d['mape'] += np.sum(np.abs((y_cpu - y_pred_cpu) / y_cpu) * 100).item() / T_out / A_norm.shape[0]

        mape_loss1 += MAPE_torch1(ypred, y).item()
        mape_loss2 += MAPE_torch2(ypred, y).item()

    mape_loss1 /= len(test_loader)
    mape_loss2 /= len(test_loader)
    print("MAPE_torch1", mape_loss1)
    print("MAPE_torch2", mape_loss2)

    target_eval_loss = np.mean(np.array(epoch_loss)).item()
    print(target_eval_loss)
    print(np.mean(np.array(mape_loss)).item())
    print(metrics_d)
    print(len(testSet), scaler.metrics_coef)
    metrics_d['mae'] = metrics_d['mae'] / len(testSet) * scaler.metrics_coef
    metrics_d['rmse'] = math.sqrt(metrics_d['rmse'] / len(testSet)) * scaler.metrics_coef
    metrics_d['mse'] = metrics_d['mse'] / len(testSet) * scaler.metrics_coef * scaler.metrics_coef
    metrics_d['mape'] = metrics_d['mape'] / len(testSet)

    print(metrics_d)

