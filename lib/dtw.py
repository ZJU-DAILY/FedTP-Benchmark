import numpy as np
import torch

try:
    from fastdtw import fastdtw
    # [修改] 不需要 euclidean 了，删掉它，防止报错
    HAS_FASTDTW = True
except ImportError:
    HAS_FASTDTW = False

def compute_dtw_matrix(train_set, num_nodes, limit_samples=100, epsilon=0.6, fast_mode=False):
    """
    计算基于 DTW 的语义邻接矩阵 (A_se)
    
    Args:
        train_set: 训练数据集
        num_nodes: 节点数
        limit_samples: 用于计算 DTW 的最大样本数
        epsilon: 阈值，用于构建邻接矩阵
        fast_mode: 如果为 True，使用更少的样本和更快的计算方式
    """
    if not HAS_FASTDTW:
        print(" [!] 警告: 未安装 fastdtw 库，无法计算真实语义矩阵。")
        print("     请运行 `pip install fastdtw`。正在返回单位矩阵...")
        return np.eye(num_nodes)

    # 快速模式：减少样本数和计算量
    if fast_mode:
        limit_samples = min(limit_samples, 50)  # 快速模式最多用 50 个样本
        print(f" [DTW] [快速模式] 正在为 {num_nodes} 个节点计算语义矩阵 (使用前 {limit_samples} 个样本)...")
    else:
        print(f" [DTW] 正在为 {num_nodes} 个节点计算语义矩阵 (使用前 {limit_samples} 个样本)...")
    
    # 1. 提取数据
    try:
        # 尝试直接提取 Tensor (速度快)
        all_x = train_set.tensors[0][:limit_samples] 
    except:
        # 如果失败则循环提取 (兼容性好)
        current_limit = min(len(train_set), limit_samples)
        all_x = torch.stack([train_set[i][0] for i in range(current_limit)])
        
    data = all_x.cpu().numpy()
    
    # 2. 数据重塑：(Batch, Node, Time, Feat) -> (Node, Long_Series)
    if data.ndim == 4: 
        # (Batch, Node, Time, Feat) -> (Node, Batch, Time, Feat)
        data = np.transpose(data, (1, 0, 2, 3))
        # 展平为 (Node, Total_Steps)
        data = data.reshape(num_nodes, -1)
    elif data.ndim == 3: 
        if data.shape[1] == num_nodes:
             data = np.transpose(data, (1, 0, 2)).reshape(num_nodes, -1)
        else:
             data = np.transpose(data, (2, 0, 1)).reshape(num_nodes, -1)
        
    # 3. 计算 DTW 距离矩阵
    dist_matrix = np.zeros((num_nodes, num_nodes))
    
    total_pairs = num_nodes * (num_nodes - 1) // 2
    computed_pairs = 0
    
    print(f" [DTW] 开始计算 {total_pairs} 个节点对的 DTW 距离...")
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            series_i = data[i, :]
            series_j = data[j, :]
            
            # [关键修改] 去掉 dist=euclidean
            # fastdtw 默认使用 2-norm，对于标量就是绝对值，完全正确且更快
            distance, _ = fastdtw(series_i, series_j)
            
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance
            
            # 每计算 10% 的节点对就打印一次进度
            computed_pairs += 1
            if computed_pairs % max(1, total_pairs // 10) == 0:
                progress = 100 * computed_pairs / total_pairs
                print(f" [DTW] 进度: {progress:.1f}% ({computed_pairs}/{total_pairs})")
            
    # 4. 归一化与构建邻接矩阵
    std = np.std(dist_matrix[dist_matrix != 0])
    if std == 0: std = 1
    
    # 高斯核转换
    adj = np.exp(- (dist_matrix / std) ** 2)
    
    # 阈值截断
    adj[adj < epsilon] = 0
    adj[adj >= epsilon] = 1 
    
    print(" [DTW] 语义矩阵计算完成。")
    return adj