import numpy as np
from fate.arch import Context

class Fed4TPServer:
    def __init__(self, ctx: Context, args):
        self.ctx = ctx
        self.args = args
        self.num_clients = args.num_clients
        
        # GLD 参数
        self.gd_use = getattr(args, 'fed4tp_gd_use', False)
        self.rho = getattr(args, 'rho', 0.1)
        self.history_len = getattr(args, 'last_gradients_length', 10)
        
        # 维护历史梯度范数: {client_id: [norm1, norm2, ...]}
        self.historical_norms = {i: [] for i in range(self.num_clients)}

    def global_detection(self, epoch):
        """
        GLD 模块：收集各客户端梯度范数，执行异常检测，下发去噪标志
        """
        if not self.gd_use:
            return

        print(f"[Fed4TP Server] Epoch {epoch} - Executing Global Detection (GLD)...")

        # 1. 接收各个客户端本地计算的梯度范数 (L2 Norm)
        norm_guest = self.ctx.guest.get(f"grad_norm_{epoch}")
        norm_hosts = self.ctx.hosts.get(f"grad_norm_{epoch}")
        
        # FATE 通信解包
        if not isinstance(norm_hosts, list):
            norm_hosts = [norm_hosts]
            
        all_norms = [norm_guest] + norm_hosts
        unreliable_flags = []

        # 2. 核心检测逻辑 (文献中的 Global Detection)
        for client_id, current_norm in enumerate(all_norms):
            history = self.historical_norms[client_id]
            
            if len(history) < 2:
                # 历史记录太少，无法计算稳定的移动平均，默认可靠
                is_unreliable = False
            else:
                # 计算过去 N 轮的移动平均范数 (Moving Average)
                ma_norm = np.mean(history[-self.history_len:])
                
                # 触发条件：当前梯度范数 > 移动平均 * (1 + rho)
                threshold = ma_norm * (1 + self.rho)
                if current_norm > threshold:
                    is_unreliable = True
                    print(f"  -> 🚨 [GLD Alert] Client {client_id} 梯度范数骤增 ({current_norm:.4f} > Threshold {threshold:.4f}). 标记为 Unreliable!")
                else:
                    is_unreliable = False
                    
            unreliable_flags.append(is_unreliable)
            
            # 更新历史记录，并保持列表长度不超过设定的窗口大小
            self.historical_norms[client_id].append(current_norm)
            if len(self.historical_norms[client_id]) > self.history_len:
                self.historical_norms[client_id].pop(0)

        # 3. 将判定结果 (is_unreliable) 下发回各个客户端
        self.ctx.guest.put(f"unreliable_flag_{epoch}", unreliable_flags[0])
        
        if len(unreliable_flags) > 1:
            host_flags = unreliable_flags[1:]
            self.ctx.hosts.put(f"unreliable_flag_{epoch}", host_flags if len(host_flags) > 1 else host_flags[0])
            
        print(f"[Fed4TP Server] Epoch {epoch} - GLD Flags Distributed: {unreliable_flags}")

    def aggregate_weights(self, epoch):
        """
        除了 GLD，服务端还需要做正常的模型聚合 (FedAVG)
        这里预留好接口，后续可以在这里执行参数平均
        """
        # (这部分逻辑我们会用显式循环来写，或者复用你的 sfl_server 逻辑)
        pass