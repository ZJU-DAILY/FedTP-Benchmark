import torch
import random
import numpy as np
import math
from transformers import TrainerCallback
import os


def synchronize_cuda_for_timing(device):
    """Synchronize queued CUDA work so wall-clock evaluation timings are comparable."""
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def unpack_spatiotemporal_batch(batch):
    """
    Normalize loader outputs to a common (x, y) pair.
    """
    if not isinstance(batch, (list, tuple)):
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    if len(batch) == 2:
        x, y = batch
    elif len(batch) >= 5:
        x, y = batch[0], batch[4]
        if isinstance(y, torch.Tensor) and y.dim() == 4:
            y = y.transpose(1, 2).contiguous()
    else:
        raise ValueError(f"Unsupported batch structure with length {len(batch)}")

    return x, y


def align_prediction_and_target(pred, y):
    """
    Align prediction layout to target layout.
    """
    if isinstance(pred, tuple):
        pred = pred[0]

    if y.dim() == 4 and y.shape[-1] == 1:
        y = y.squeeze(-1)
    if pred.dim() == 4 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)

    if pred.shape != y.shape:
        if pred.dim() == 3 and y.dim() == 3 and pred.shape[1] == y.shape[2]:
            pred = pred.transpose(1, 2).contiguous()
        elif pred.dim() == 4 and y.dim() == 4 and pred.shape[1] == y.shape[2] and pred.shape[2] == y.shape[1]:
            pred = pred.transpose(1, 2).contiguous()
        else:
            pred = pred.reshape_as(y)

    return pred, y


def init_seed(seed):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def init_device(opt):
    if torch.cuda.is_available():
        opt.cuda = True
        torch.cuda.set_device(int(opt.device[5]))
    else:
        opt.cuda = False
        opt.device = 'cpu'
    return opt

def init_optim(model, opt):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),lr=opt.lr_init)

def init_lr_scheduler(optim, opt):
    '''
    Initialize the learning rate scheduler
    '''
    #return torch.optim.lr_scheduler.StepLR(optimizer=optim,gamma=opt.lr_scheduler_rate,step_size=opt.lr_scheduler_step)
    return torch.optim.lr_scheduler.MultiStepLR(optimizer=optim, milestones=opt.lr_decay_steps,
                                                gamma = opt.lr_scheduler_rate)

def print_model_parameters(model, logger, only_num = True):
    logger.info('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            logger.info('{} {} {}'.format(name, param.shape, param.requires_grad))
    total_num = sum([param.nelement() for param in model.parameters()])
    logger.info('Total params num: {}'.format(total_num))
    logger.info('*****************Finish Parameter****************')

def get_memory_usage(device):
    allocated_memory = torch.cuda.memory_allocated(device) / (1024*1024.)
    cached_memory = torch.cuda.memory_cached(device) / (1024*1024.)
    print('Allocated Memory: {:.2f} MB, Cached Memory: {:.2f} MB'.format(allocated_memory, cached_memory))
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    return allocated_memory, cached_memory

def init_parameters(model):
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
        else:
            torch.nn.init.uniform_(p)
    return model

def MAE_torch(output, label):
    return torch.mean(torch.abs(output - label))

def RMSE_torch(output, label):
    return torch.sqrt(torch.mean((output - label)**2))

def MAPE_torch(output, label):
    mask = ~torch.isnan(label)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = 2.0 * (torch.abs(output - label) / (torch.abs(output) + torch.abs(label)))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)
    # return torch.mean(torch.abs(output - label)/label)


def All_Metrics(pred, true, mask1, mask2):
    mae  = MAE_torch(pred, true)
    rmse = RMSE_torch(pred, true)
    mape = MAPE_torch(pred, true)
    rrse = 0.0
    corr = 0.0
    return mae, rmse, mape, rrse, corr

# ==========================================
#        Added for FedGODE Integration
# ==========================================

def get_normalized_adj(A):
    """
    Returns a tensor, the degree normalized adjacency matrix.
    Used specifically by the FedGODE (ODEGCN) model.
    """
    import numpy as np
    import torch
    
    # 确保输入是 numpy 数组，防止传入 Tensor 报错
    if isinstance(A, torch.Tensor):
        A = A.cpu().detach().numpy()
        
    alpha = 0.8
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    A_reg = alpha / 2 * (np.eye(A.shape[0]) + A_wave)
    return torch.from_numpy(A_reg.astype(np.float32))



def evaluate_client_model(model, val_loader, loss_func, scaler, device, forward_fn=None):
    """
    Generic client-side evaluation helper.
    """
    model.eval()
    test_loss, test_mae, test_rmse, test_mape = 0.0, 0.0, 0.0, 0.0
    total_elements = 0
    total_mask_elements = 0

    with torch.no_grad():
        for batch in val_loader:
            x_val, y_val = unpack_spatiotemporal_batch(batch)
            x_val, y_val = x_val.to(device), y_val.to(device)

            if forward_fn is not None:
                pred_val = forward_fn(model, x_val)
            else:
                pred_val = model(x_val)

            pred_val, y_val = align_prediction_and_target(pred_val, y_val)

            loss = loss_func(pred_val, y_val)
            test_loss += loss.item()
            total_elements += y_val.numel()

            y_cpu = y_val.cpu().numpy()
            pred_cpu = pred_val.cpu().numpy()
            test_mae += np.sum(np.abs(y_cpu - pred_cpu))
            test_rmse += np.sum((y_cpu - pred_cpu) ** 2)

            y_real = scaler.inverse_transform(y_val).cpu().numpy()
            pred_real = scaler.inverse_transform(pred_val).cpu().numpy()
            mask = y_real > 0.5
            valid_elements = np.sum(mask)
            if valid_elements > 0:
                test_mape += np.sum(np.abs(y_real[mask] - pred_real[mask]) / y_real[mask])
                total_mask_elements += valid_elements

    test_loss /= max(len(val_loader), 1)
    final_mae = (test_mae / max(total_elements, 1)) * scaler.metrics_coef
    final_rmse = math.sqrt(test_rmse / max(total_elements, 1)) * scaler.metrics_coef
    final_mape = (test_mape / total_mask_elements) if total_mask_elements > 0 else 0.0

    return test_loss, final_mae, final_rmse, final_mape


def extract_ctx_data(ctx, data_list):
    """
    FATE 通信列表解包工具：
    处理 Server 发送给多个 Host 时的 List 索引解包问题
    """
    if isinstance(data_list, list):
        if len(data_list) == 1:
            return data_list[0]
        else:
            my_idx = ctx.rank - 1 # Host 从 rank 1 开始
            if 0 <= my_idx < len(data_list):
                return data_list[my_idx]
            return data_list[0] # 兜底
    return data_list



class GlobalEarlyStopping:
    def __init__(self, patience=50, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        is_best = False
        if self.best_score is None:
            self.best_score = score
            is_best = True
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'[Server] 全局早停计数: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            is_best = True
        return is_best, self.early_stop

def federated_early_stopping(ctx, epoch, local_metric=None, global_es=None):
    """
    统一的联邦早停同步钩子。
    Client 端传入 local_metric，返回 (is_best, should_stop)
    Server 端传入 global_es，返回 (is_best, should_stop)
    """
    if not ctx.is_on_arbiter:
        # Client: 发送本地验证指标，接收全局裁决
        ctx.arbiter.put(f"es_metric_{epoch}", local_metric)
        signal_data = ctx.arbiter.get(f"es_signal_{epoch}")
        return signal_data[0] if isinstance(signal_data, list) else signal_data
    else:
        # Server: 收集指标 -> 求平均 -> 裁决 -> 广播
        val_guest = ctx.guest.get(f"es_metric_{epoch}")
        val_hosts = ctx.hosts.get(f"es_metric_{epoch}")
        if not isinstance(val_hosts, list): val_hosts = [val_hosts]
        
        avg_metric = sum([val_guest] + val_hosts) / len([val_guest] + val_hosts)
        is_best, should_stop = global_es(avg_metric)
        
        signal = (is_best, should_stop)
        ctx.guest.put(f"es_signal_{epoch}", signal)
        ctx.hosts.put(f"es_signal_{epoch}", [signal] * len(val_hosts))
        return is_best, should_stop



class FATEGlobalEarlyStoppingCallback(TrainerCallback):
    def __init__(self, ctx, patience=50, min_delta=1e-3, log_file="logs/early_stop_losses.csv"):
        """
        :param min_delta: 全局平均 Loss 必须至少下降 min_delta，才算作有效改善。
        :param log_file: 记录各客户端和全局 Loss 的文件路径。
        """
        self.ctx = ctx
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.log_file = log_file

        # 仅 Guest 负责初始化日志文件和写入表头
        if self.ctx.is_on_guest:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            if not os.path.exists(self.log_file):
                with open(self.log_file, "w") as f:
                    # 表头支持动态数量的 Host
                    f.write("Epoch,Guest_Loss,Host_Losses_List,Global_Avg_Loss,Best_Loss,Patience_Counter\n")

    def on_epoch_end(self, args, state, control, **kwargs):
        """在每个 Epoch 结束时触发"""
        print(f"====== 🚨 探针测试：Rank {self.ctx.rank} 成功进入了 on_evaluate 函数！======")
        if self.ctx.is_on_arbiter:
            return 

        tag = f"early_stop_{state.epoch}"

        # 1. 提取当前 Epoch 的验证集 Loss (如果没找到 loss 则兜底为无穷大)
        eval_loss = float('inf')
        for log in reversed(state.log_history):
            if "eval_loss" in log:
                eval_loss = log["eval_loss"]
                break
            elif "eval_mae" in log:  
                eval_loss = log["eval_mae"]
                break

        # ================= Guest 裁判 & 记录逻辑 =================
        if self.ctx.is_on_guest:
            # 接收所有 Host 的 Loss
            host_losses_data = self.ctx.hosts.get(tag)
            if not isinstance(host_losses_data, list):
                host_losses_data = [host_losses_data]
            
            # 计算全局平均 Loss
            all_losses = [eval_loss] + host_losses_data
            avg_loss = np.mean(all_losses)

            # 【修复点】：早停判定引入 min_delta
            # 只有当下降幅度超过 min_delta 时，才认为是有效改善
            if avg_loss < (self.best_loss - self.min_delta):
                self.best_loss = avg_loss
                self.counter = 0  # 真正有效的改善，重置耐心
            else:
                self.counter += 1 # 否则耐心消耗

            should_stop = (self.counter >= self.patience)
            
            # 将判决结果发送给所有 Host
            self.ctx.hosts.put(f"stop_{tag}", [should_stop] * len(host_losses_data))
            
            # 【新增点】：将本轮的所有详细 Loss 追加写入 CSV
            with open(self.log_file, "a") as f:
                # 把 Host 的 Loss 拼成一个用分号隔开的字符串，例如 "12.5;13.2;11.8"
                host_str = ";".join([f"{l:.4f}" for l in host_losses_data])
                f.write(f"{state.epoch},{eval_loss:.4f},{host_str},{avg_loss:.4f},{self.best_loss:.4f},{self.counter}\n")

            print(f"--- [Guest] Epoch {state.epoch} 全局平均 Loss: {avg_loss:.4f} | 最佳: {self.best_loss:.4f} | 耐心值: {self.counter}/{self.patience} ---")

        # ================= Host 参赛者逻辑 =================
        else:
            # 上报自己的 Loss 给 Guest
            self.ctx.guest.put(tag, eval_loss)
            
            # 接收 Guest 的停止指令
            should_stop_data = self.ctx.guest.get(f"stop_{tag}")
            should_stop = should_stop_data[0] if isinstance(should_stop_data, list) else should_stop_data

        # ================= 统一执行早停 =================
        if should_stop:
            print(f"Rank {self.ctx.rank}: 收到全局早停信号，训练安全终止于 Epoch {state.epoch}!")
            control.should_training_stop = True

class EarlyStopSignal(BaseException):
    """自定义异常，用于强制跳出 FATE 黑盒循环。
    继承 BaseException 是为了防止被框架内部的 try-except Exception 给意外吞掉。"""
    pass

class EarlyStopDecision(tuple):
    """Tuple-compatible early-stop signal whose truth value is should_stop."""

    __slots__ = ()

    def __new__(cls, is_best=False, should_stop=False):
        return super().__new__(cls, (bool(is_best), bool(should_stop)))

    @property
    def is_best(self):
        return self[0]

    @property
    def should_stop(self):
        return self[1]

    def __bool__(self):
        return self.should_stop


class ExplicitEarlyStopper:
    # 【修改 1】：__init__ 增加 args 参数
    def __init__(self, ctx, args, patience=50, min_delta=1e-4): 
        self.ctx = ctx
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.epoch_counter = 1
        
        # 【新增】：拼接出专属的 Task_ID，例如 "FedGRU_PeMS04_flow"
        self.task_id = f"{args.model}_{args.dataset_name}_{args.feature_type}"
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        self.log_file = os.path.join(project_root, "logs", "early_stop_losses.csv")
        
        if self.ctx.is_on_guest:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            if not os.path.exists(self.log_file):
                with open(self.log_file, "w") as f:
                    # 【修改 2】：表头最前面加上 Task_ID
                    f.write("Task_ID,Epoch,Guest_Loss,Host_Losses_List,Global_Avg_Loss,Best_Loss,Patience_Counter\n")

    def check_and_sync(self, eval_loss):
        eval_loss = float(eval_loss)
        
        if self.ctx.is_on_arbiter:
            return EarlyStopDecision(False, False)
            
        tag = f"early_stop_{self.epoch_counter}"
        should_stop = False
        
        if self.ctx.is_on_guest:
            host_losses_data = self.ctx.hosts.get(tag)
            if not isinstance(host_losses_data, list):
                host_losses_data = [host_losses_data]
            
            all_losses = [eval_loss] + host_losses_data
            avg_loss = float(np.mean(all_losses))
            
            is_best = avg_loss < (self.best_loss - self.min_delta)
            if is_best:
                self.best_loss = avg_loss
                self.counter = 0
            else:
                self.counter += 1
                
            should_stop = bool(self.counter >= self.patience)
            signal = (is_best, should_stop)
            self.ctx.hosts.put(f"stop_{tag}", [signal] * len(host_losses_data))
            
            try:
                with open(self.log_file, "a") as f:
                    host_str = ";".join([f"{float(l):.4f}" for l in host_losses_data])
                    # 【修改 3】：写入数据时，把 self.task_id 写在最前面
                    f.write(f"{self.task_id},{self.epoch_counter},{eval_loss:.4f},{host_str},{avg_loss:.4f},{self.best_loss:.4f},{self.counter}\n")
            except Exception as e:
                print(f"写入日志失败: {e}")
                
            print(f"--- [Guest 裁判] {self.task_id} | Epoch {self.epoch_counter} | 平均 Loss: {avg_loss:.4f} | 最佳: {self.best_loss:.4f} | 耐心值: {self.counter}/{self.patience} ---")

        else:
            self.ctx.guest.put(tag, eval_loss)
            signal_data = self.ctx.guest.get(f"stop_{tag}")
            signal = signal_data[0] if isinstance(signal_data, list) else signal_data
            if isinstance(signal, tuple):
                is_best, should_stop = signal
            else:
                is_best, should_stop = False, bool(signal)
            
        self.epoch_counter += 1
        return EarlyStopDecision(is_best, should_stop)
