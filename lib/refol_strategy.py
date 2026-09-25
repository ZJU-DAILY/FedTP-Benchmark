import torch
import torch.nn as nn
import numpy as np
import xlsxwriter
from torch.utils.data import TensorDataset
from torch_geometric.data import DataLoader
from copy import deepcopy
from collections import defaultdict
import scipy.stats

# =============================================================================
# 模块引用区域
# 既然我们在 lib 文件夹下，根据 Python 运行根目录在项目根目录的习惯：
# 1. 神经网络从 model 文件夹引用
# 2. 数据工具从同级 lib 文件夹引用
# =============================================================================
from model.refol_nets import GRU, AttGCN
from lib.refol_loader import load_dataset, unscaled_metrics

# =============================================================================
# Part 1: REFOL Client Strategy (对应原 client_oa.py)
# =============================================================================
class REFOL_Client(object):

    def __init__(self, client_id, client_dataset, feature_scaler,
                 input_size, output_size, args):
        self.client_id = client_id
        self.client_dataset = client_dataset
        self.feature_scaler = feature_scaler
        self.input_size = input_size
        self.output_size = output_size
        self.args = args
        self.lr = self.args.lr
        self.batch_size = self.args.batch_size
        # 这里的 DataLoader 是 PyTorch Geometric 的，适合处理图数据
        self.dataloader = DataLoader(self.client_dataset, batch_size=self.batch_size)
        
        # 初始化模型 (调用 refol_nets.py 中的 GRU)
        self.model = GRU(input_size, self.args.hidden_size, output_size, self.args.dropout, self.args.num_layers)

        self.state_dict = None
        self.h_client_dataset = None
        self.selected = False # 标记本轮是否参与训练

        use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')


    def local_execute(self, state_dict_to_load):
        # 如果本轮被选中 (selected=True)，则进行训练
        # 否则 (selected=False)，仅进行预测评估，不更新模型
        self.dataloader = DataLoader(self.client_dataset, batch_size=self.batch_size)
        
        if self.selected:
            # === Case 1: 参与训练 ===
            # 加载全局模型参数
            if state_dict_to_load is not None:
                self.model.load_state_dict(state_dict_to_load)
            self.model.to(self.device)
            self.model.train()
            
            with torch.enable_grad():
                for epoch_i in range(self.args.epoch):
                    num_samples = 0
                    epoch_log = defaultdict(lambda: 0.0)
                    for batch in self.dataloader:
                        x, y, x_attr, y_attr = batch
                        x = x.to(self.device) if (x is not None) else None
                        y = y.to(self.device) if (y is not None) else None
                        x_attr = x_attr.to(self.device) if (x_attr is not None) else None
                        y_attr = y_attr.to(self.device) if (y_attr is not None) else None
                        
                        data = dict(x=x, x_attr=x_attr, y=y, y_attr=y_attr)
                        y_pred = self.model(data)
                        loss = nn.MSELoss()(y_pred, y)
                        
                        loss.backward()
                        # 手动更新梯度 (SGD)
                        for param in self.model.parameters():
                            param.data = param.data - self.lr * param.grad.data
                            param.grad.data.zero_()
                            
                        num_samples += x.shape[0]
                        metrics = unscaled_metrics(y_pred, y, self.feature_scaler)
                        epoch_log['loss'] += loss.detach() * x.shape[0]
                        for k in metrics:
                            epoch_log[k] += metrics[k] * x.shape[0]
                            
                    for k in epoch_log:
                        epoch_log[k] /= num_samples
                        epoch_log[k] = epoch_log[k].cpu()
            
            # 更新历史数据集，用于下一次检测概念漂移
            self.h_client_dataset = deepcopy(self.client_dataset)
        else:
            # === Case 2: 不参与训练 (复用旧模型) ===
            self.model.to(self.device)
            self.model.eval()
            with torch.no_grad():
                num_samples = 0
                epoch_log = defaultdict(lambda: 0.0)
                for batch in self.dataloader:
                    x, y, x_attr, y_attr = batch
                    x = x.to(self.device) if (x is not None) else None
                    y = y.to(self.device) if (y is not None) else None
                    x_attr = x_attr.to(self.device) if (x_attr is not None) else None
                    y_attr = y_attr.to(self.device) if (y_attr is not None) else None
                    
                    data = dict(x=x, x_attr=x_attr, y=y, y_attr=y_attr)
                    y_pred = self.model(data)
                    loss = nn.MSELoss()(y_pred, y)

                    num_samples += x.shape[0]
                    metrics = unscaled_metrics(y_pred, y, self.feature_scaler)
                    epoch_log['loss'] += loss.detach() * x.shape[0]
                    for k in metrics:
                        epoch_log[k] += metrics[k] * x.shape[0]
                for k in epoch_log:
                    epoch_log[k] /= num_samples
                    epoch_log[k] = epoch_log[k].cpu()

        # 重置标志位
        self.selected = False
        self.state_dict = deepcopy(self.model.to(self.device).state_dict())

        epoch_log['num_samples'] = num_samples
        epoch_log = dict(**epoch_log)
        
        self.local_result = {
            'state_dict': self.state_dict, 
            'log': epoch_log
        }

    def eval_dataset(self):
        """
        核心逻辑：计算 KL 散度，判断是否发生概念漂移
        """
        if self.h_client_dataset is None:
            self.selected = True # 第一次必选
        else:
            # 获取当前数据分布
            self.dataloader = DataLoader(self.client_dataset, batch_size=self.batch_size)
            data_now = []
            for batch in self.dataloader:
                x, _, _, _ = batch
                data_now.extend(x.flatten().tolist())
            data_now = np.array(data_now)

            # 获取历史数据分布
            self.h_dataloader = DataLoader(self.h_client_dataset, batch_size=self.batch_size)
            data_h = []
            for batch in self.h_dataloader:
                x, _, _, _ = batch
                data_h.extend(x.flatten().tolist())
            data_h = np.array(data_h)
            
            # 反归一化后计算 KL 散度
            data_h = self.feature_scaler.inverse_transform(data_h)
            data_now = self.feature_scaler.inverse_transform(data_now)
            
            KL = scipy.stats.entropy(data_now, data_h)
            
            # 判断阈值
            if KL > self.args.kl_threshold:
                self.selected = True
            else:
                self.selected = False


# =============================================================================
# Part 2: REFOL Server Strategy (对应原 refol.py)
# =============================================================================
class REFOL(object): # 在 fate_main.py 里会被调用
    def __init__(self, config):
        self.config = config
        # 参数自适应映射 (防止 FATE 框架参数名不一致)
        if not hasattr(self.config, 'num_clients'):
            self.config.num_clients = getattr(self.config, 'num_nodes', 10)
        if not hasattr(self.config, 'agg_model'):
            self.config.agg_model = getattr(self.config, 'model', 'REFOL')
        if not hasattr(self.config, 'pred_steps'):
            self.config.pred_steps = getattr(self.config, 'pred_len', 1)

    def boot(self):
        print(f'Booting {self.config.agg_model} fl-server...')
        self.num_clients = self.config.num_clients
        print(f'Total clients: {self.num_clients}')
        
        # 加载数据 (调用 refol_loader.py)
        # 注意：这里 adj_mx_name 硬编码为 pems.pkl，根据实际情况可能需要调整
        data, selected_node = load_dataset(name=self.config.dataset
                                           , adj_mx_name='pems.pkl' 
                                           , num_clients=self.num_clients
                                           , pred_len=self.config.pred_steps
                                           )

        self.data = data
        input_size = self.data['x'].shape[-1] + self.data['x_attr'].shape[-1]
        output_size = self.data['y'].shape[-1]
        self.max_epoch = data['x'].shape[0]
        self.train_per_num_samples = 1

        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # 初始化客户端
        clients = []
        for client_i in range(self.num_clients):
            # 初始切片数据
            client_dataset = TensorDataset(
                data['x'][:self.train_per_num_samples, :, client_i:client_i + 1, :],
                data['y'][:self.train_per_num_samples, :, client_i:client_i + 1, :],
                data['x_attr'][:self.train_per_num_samples, :, client_i:client_i + 1, :],
                data['y_attr'][:self.train_per_num_samples, :, client_i:client_i + 1, :]
            )

            client_tmp = REFOL_Client(client_id=client_i,
                                client_dataset=client_dataset,
                                feature_scaler=self.data['feature_scaler'],
                                input_size=input_size,
                                output_size=output_size,
                                args=self.config)
            clients.append(client_tmp)

        self.clients = clients
        
        # 初始化服务端聚合模型 (图卷积网络)
        self.gcn = AttGCN()

        self.server_datasets = TensorDataset(
            self.data['x'], self.data['y'],
            self.data['x_attr'], self.data['y_attr'])

        use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.global_model = None

    def run(self):
        # 计算最大轮数
        rounds = self.max_epoch - 1 - self.train_per_num_samples
        # 注意：这里移除了 rounds = 10 的硬编码限制，让它跑完全程
        
        # 创建 Excel 记录
        filename = f'{self.config.dataset}_{self.config.pred_steps}.xlsx'
        workbook = xlsxwriter.Workbook(filename)
        sheet_round = workbook.add_worksheet('round')
        sheet_round.write(0, 0, 'round_num')
        sheet_round.write(0, 1, f'{self.config.agg_model}_rmse')
        sheet_round.write(0, 2, f'{self.config.agg_model}_mae')

        # 开始在线学习循环
        for rround in range(1, rounds + 1):
            print(f'**** Round {rround}/{rounds} ****')
            
            # 执行一轮训练
            train_log = self.train_round(rround)
            
            # 获取结果
            train_loss = train_log['log']['rmse'].item()
            train_mae = train_log['log']['mae'].item()

            # 写入日志
            sheet_round.write(rround, 0, rround)
            sheet_round.write(rround, 1, train_loss)
            sheet_round.write(rround, 2, train_mae)
            
            # 终端打印
            print(f'  Result: RMSE={train_loss:.4f}, MAE={train_mae:.4f}')

        workbook.close()
        print(f"Training finished. Logs saved to {filename}")

    def train_round(self, rround):
        # 1. 更新所有客户端的数据 (模拟在线流式数据)
        self.update_train_data(rround, self.clients)
        
        # 2. 统计本轮哪些客户端决定参与训练 (KL散度 > 阈值)
        agg_id_list = []
        for client in self.clients:
            if client.selected:
                agg_id_list.append(client.client_id)
        print('  Selected clients:', agg_id_list)

        local_logs = []
        agg_state_dict = []
        
        # 3. 客户端并行执行 (模拟)
        for idx, client in enumerate(self.clients):
            if client.selected:
                # 参与者：下载全局模型 -> 本地训练 -> 上传新参数
                client.local_execute(state_dict_to_load=deepcopy(self.global_model))
                agg_state_dict.append(deepcopy(client.local_result['state_dict']))
                local_logs.append(client.local_result['log'])
            else:
                # 不参与者：不下载模型 -> 使用旧模型预测 -> 不上传
                client.local_execute(state_dict_to_load=None)
                local_logs.append(client.local_result['log'])

        # 4. 服务端聚合 (GCN 聚合)
        agg_local_train_results = self.aggregate_local_train_results(local_logs, agg_state_dict, agg_id_list, rround)
        agg_log = agg_local_train_results['log']
        
        return {
            'loss': torch.tensor(0).float(),
            'progress_bar': agg_log,
            'log': agg_log
        }

    def update_train_data(self, rround, sample_clients):
        """
        滑动窗口，更新每个 Client 的 dataset，并触发 KL 散度检测
        """
        for client in sample_clients:
            client_i = client.client_id
            # 模拟时间推移，切片向后移动一步
            client.client_dataset = TensorDataset(
                self.data['x'][rround - 1:self.train_per_num_samples + rround - 1, :, client_i:client_i + 1, :],
                self.data['y'][rround - 1:self.train_per_num_samples + rround - 1, :, client_i:client_i + 1, :],
                self.data['x_attr'][rround - 1:self.train_per_num_samples + rround - 1, :, client_i:client_i + 1, :],
                self.data['y_attr'][rround - 1:self.train_per_num_samples + rround - 1, :, client_i:client_i + 1, :]
            )
            # 更新数据后，立即评估是否发生概念漂移
            client.eval_dataset()

    def aggregate_local_train_results(self, local_logs, local_states, agg_id_list, round):
        # 只有当有客户端参与训练时，才进行聚合
        if len(local_states) > 0:
            self.aggregate_local_train_state_dicts(local_states, agg_id_list, round)
        
        return {
            'log': self.aggregate_local_logs(local_logs)
        }

    # 聚合客户端本地模型 (核心创新点：基于图卷积的聚合)
    def aggregate_local_train_state_dicts(self, local_states, agg_id_list, round):
        # 1. 动态构建图结构
        edge_index = np.array(deepcopy(self.data['edge_index']))
        sample_id = np.array(agg_id_list)
        
        # 只保留参与训练的节点之间的边
        mask = np.isin(edge_index, sample_id)
        mask1 = np.isin(np.sum(mask, axis=0), 2)
        edge_index = edge_index[:, mask1]
        
        # 重新映射节点 ID 到 0~len(sample_id)
        table = np.zeros(sample_id.max() + 1, np.int64)
        table[sample_id] = np.arange(sample_id.size)
        edge_index = torch.from_numpy(table[edge_index])
        
        # 添加虚拟节点 (Virtual Node) 连接所有参与者，用于聚合
        tmp = np.full((sample_id.shape), len(sample_id))
        tmp = np.stack((table[sample_id], tmp))
        # 添加自环
        tmp = np.hstack((tmp, [[len(sample_id)],[len(sample_id)]]))
        edge_index = torch.from_numpy(np.concatenate((edge_index, tmp), axis=1))

        # 2. 准备特征 (模型参数作为 GCN 的输入特征)
        tmp_model = self.global_model
        if tmp_model is None:
            # 第一轮如果没有全局模型，用第一个客户端的作为初始值
            tmp_model = deepcopy(local_states[0])
        local_states.append(tmp_model) # 虚拟节点的特征初始化为上一轮全局模型

        local_results = []
        for i, local_train_result in enumerate(local_states):
            # 将所有参数展平成一维向量
            for name in local_train_result:
                local_results += local_train_result[name].flatten().tolist()
        
        # 转换为 Tensor: [Num_Clients + 1, Total_Params]
        local_results = torch.Tensor(local_results).view((len(sample_id) + 1, -1))

        # 3. 执行图卷积
        self.gcn.to(self.device)
        # 输入：[参数矩阵, 动态边] -> 输出：[聚合后的参数矩阵]
        local_results = self.gcn(
            x=local_results.to(self.device)
            , edge_index=edge_index.to(self.device)
        )
        
        # 4. 恢复参数形状
        global_model_flat = local_results[-1] # 取虚拟节点的输出作为新全局模型
        agg_state_dict = {}
        len_start = 0
        # 这里的 local_train_result 是循环里的最后一个，即 tmp_model，结构是一样的
        for name in local_train_result:
            length = len(local_train_result[name].flatten().tolist())
            agg_state_dict[name] = global_model_flat[len_start:len_start + length].reshape_as(local_train_result[name])
            len_start += length
            
        self.global_model = agg_state_dict

    def aggregate_local_logs(self, local_logs):
        # 简单的加权平均计算日志 (Loss, MAE, RMSE)
        agg_log = deepcopy(local_logs[0])
        for k in agg_log:
            agg_log[k] = 0
            for local_log_idx, local_log in enumerate(local_logs):
                if k == 'num_samples':
                    agg_log[k] += local_log[k]
                else:
                    agg_log[k] += local_log[k] * local_log['num_samples']
        for k in agg_log:
            if k != 'num_samples':
                agg_log[k] /= agg_log['num_samples']
        return agg_log