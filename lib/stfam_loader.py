import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch_geometric.utils as pyg_utils

def build_global_data(dataset_name, num_global_nodes, project_root):
    print(f"[{dataset_name}] 正在构建 STFAM 全局数据 (D, P)...")
    D_matrix = np.zeros((num_global_nodes, num_global_nodes), dtype=np.float32)
    dist_path = os.path.join(project_root, f'data/{dataset_name}/distance.csv')
    if os.path.exists(dist_path):
        dist_df = pd.read_csv(dist_path)
        for _, row in dist_df.iterrows():
            src, dst, cost = int(row['from']), int(row['to']), float(row['cost'])
            if src < num_global_nodes and dst < num_global_nodes:
                D_matrix[src, dst] = cost
                D_matrix[dst, src] = cost 
    else:
        D_matrix = np.eye(num_global_nodes, dtype=np.float32)
    D_tensor = torch.tensor(D_matrix, dtype=torch.float32).unsqueeze(0) 
    
    poi_path = os.path.join(project_root, f'data/{dataset_name}/poi.csv')
    if os.path.exists(poi_path):
        P_matrix = pd.read_csv(poi_path).values.astype(np.float32)
    else:
        degree = (D_matrix > 0).sum(axis=1, keepdims=True)
        P_matrix = np.repeat(degree, 16, axis=1).astype(np.float32)
    P_tensor = torch.tensor(P_matrix, dtype=torch.float32)
    return D_tensor, P_tensor

def unified_dataset_to_stfam_xy(dataset, t_in, t_out):
    """
    专门为 Grid 底座写的转接头：将 5 元组转为 STFAM 认识的 X, Y 张量
    """
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    x_list, y_list = [], []
    for batch in loader:
        if len(batch) == 5:
            x_c, _, _, _, y = batch
        else:
            x_c, y = batch
        
        # Grid graph view 默认输出 [B, T, N, C]，而 STFAM 需要 [B, N, T, C]
        if x_c.dim() == 4 and x_c.shape[1] == t_in:
            x_c = x_c.transpose(1, 2)
        if y.dim() == 4 and y.shape[1] == t_out:
            y = y.transpose(1, 2)
            
        x_list.append(x_c)
        y_list.append(y)
        
    return torch.cat(x_list, dim=0), torch.cat(y_list, dim=0)

class STFAM_Client_Dataset(Dataset):
    """
    内存极度优化的自定义 Dataset，按需实时生成转移矩阵
    """
    def __init__(self, x_tra, y_tra, edge_index, max_local_nodes, device):
        if isinstance(x_tra, np.ndarray): x_tra = torch.from_numpy(x_tra).float()
        if isinstance(y_tra, np.ndarray): y_tra = torch.from_numpy(y_tra).float()
        
        # 获取输入数据当前的设备（通常是大内存 CPU），强制辅助矩阵对齐到该设备
        data_device = x_tra.device
        
        B, N_local, T_in, C = x_tra.shape
        pad_len = max_local_nodes - N_local
        
        if pad_len > 0:
            self.x = F.pad(x_tra, (0, 0, 0, 0, 0, pad_len))
            self.y = F.pad(y_tra, (0, 0, 0, 0, 0, pad_len))
        else:
            self.x = x_tra
            self.y = y_tra
            
        # 预计算图连通概率矩阵，将其强制拉到与 x 相同的设备上（解决 CPU/CUDA 冲突）
        A_dense = pyg_utils.to_dense_adj(edge_index.to(data_device), max_num_nodes=max_local_nodes)[0]
        A_sum = A_dense.sum(dim=1, keepdim=True)
        A_sum[A_sum == 0] = 1.0
        self.A_prob = A_dense / A_sum 
        
        # 代数降维：此时 x 和 A_prob 必在同一设备，完美运算
        # [B, N, T, C] -> sum(0, 2) -> [N, C] -> permute -> [C, N]
        total_flow = self.x.sum(dim=(0, 2)).permute(1, 0) 
        self.U = total_flow.unsqueeze(-1) * self.A_prob.unsqueeze(0)
        dense_tr_dim = max_local_nodes * max_local_nodes * C
        compact_tr_dim = max_local_nodes * C * 2
        print(
            f"[STFAM compact federated] samples={len(self.x)} nodes={max_local_nodes} "
            f"Tr_dim={compact_tr_dim} instead_of_dense_Tr_dim={dense_tr_dim}",
            flush=True,
        )
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        x_i = self.x[idx] 
        y_i = self.y[idx] 
        
        flow_i = x_i.permute(1, 0, 2).contiguous()
        # 实时生成单个样本的转移矩阵 (同设备运算)
        agg_i = torch.einsum("ij,tjc->tic", self.A_prob, flow_i)
        Tr_i = torch.cat([flow_i, agg_i], dim=-1).reshape(flow_i.size(0), -1)
        
        return Tr_i, self.U, y_i

def load_stfam_client_dataset(x_tra, y_tra, edge_index, max_local_nodes, device):
    """
    直接返回内存安全、设备对齐的 Custom Dataset
    """
    return STFAM_Client_Dataset(x_tra, y_tra, edge_index, max_local_nodes, device)
