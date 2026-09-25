import h5py
import numpy as np
import torch
import os
import sys

def create_grid_graph(height, width):
    """
    核心函数：生成网格图的邻接矩阵 (edge_index)
    节点编号逻辑：行优先，即 (0,0)是0, (0,1)是1 ... (r, c) 是 r*width+c
    """
    print(f"正在构建 {height}x{width} 的网格图结构...")
    edge_index = []
    # 遍历所有网格点
    for r in range(height):
        for c in range(width):
            # 当前节点 ID (0 ~ H*W-1)
            curr_node = r * width + c
            
            # 定义上下左右四个方向 (row_offset, col_offset)
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                # 检查邻居是否在网格范围内（防止越界）
                if 0 <= nr < height and 0 <= nc < width:
                    neighbor_node = nr * width + nc
                    # 添加一条边：当前节点 -> 邻居节点
                    edge_index.append([curr_node, neighbor_node])
    
    # 转为 PyTorch Geometric 需要的 LongTensor 格式 (2, Num_Edges)
    # 并转置，使其形状变为 [2, E]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

def process_h5_dataset(file_path, save_path):
    print(f"\n🚀 开始处理文件: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ 错误：找不到文件 {file_path}")
        print(f"   当前工作目录是: {os.getcwd()}")
        return

    # 1. 读取 .h5 数据
    try:
        with h5py.File(file_path, 'r') as f:
            # 打印 keys 方便调试
            print(f"   H5 Keys: {list(f.keys())}") 
            
            # 自动寻找数据 Key (通常是 'data' 或 'tensor')
            if 'data' in f.keys():
                raw_data = f['data'][:]  
            else:
                key = list(f.keys())[0]
                raw_data = f[key][:]
                print(f"   ⚠️ 未找到 'data' key，使用第一个 key: '{key}'")

        print(f"   原始数据形状: {raw_data.shape}") 
        # 预期形状: (Time, Feature=2, Height=32, Width=32)
        
        # 2. 获取维度信息
        T, F, H, W = raw_data.shape
        print(f"   检测到: 时间步={T}, 特征数={F}, 网格大小={H}x{W}")
        
        # 3. 生成图结构 (Edge Index)
        edge_index = create_grid_graph(H, W)
        print(f"   图结构构建完毕: {edge_index.shape[1]} 条边")
        
        # 4. 数据变形 (Flatten)
        # 目标: 将网格拉平为节点
        # 原始: (T, F, H, W)
        # 步骤 1: reshape -> (T, F, H*W)  (注意这里是 H*W，对应行优先的节点编号)
        # 步骤 2: transpose -> (T, H*W, F) (FedSTN 需要 Time, Node, Feature)
        data_flatten = raw_data.reshape(T, F, H * W).transpose(0, 2, 1)
        
        print(f"   转换后数据形状: {data_flatten.shape} (Time, Nodes, Features)")
        
        # 5. 保存处理后的数据
        # 自动创建输出目录（如果不存在）
        output_dir = os.path.dirname(save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        np.savez(save_path, 
                 data=data_flatten,       # 流量特征
                 edge_index=edge_index.numpy()) # 图结构
        
        print(f"✅ 保存成功: {save_path}")
        print(f"   (包含数组: 'data', 'edge_index')")

    except Exception as e:
        print(f"❌ 处理过程中发生错误: {e}")

if __name__ == '__main__':
    # ================= 路径配置区域 =================
    
    # 假设你的目录结构是：
    # 项目根目录/
    # ├── data/
    # │   └── TaxiBJ/
    # │       └── BJ15_M32x32_T30_InOut.h5  <-- 你下载的源文件放这里
    # ├── preprocessing/
    # │   └── preprocess_grid.py            <-- 本脚本在这里
    # ├── lib/
    # └── fate_run.sh
    
    # 1. 源文件路径 (使用相对路径，指向上一级目录的 data)
    # 请根据你实际下载的文件名修改 'BJ15_M32x32_T30_InOut.h5'
    source_file = os.path.join('..', 'data', 'TaxiBJ', 'BJ15_M32x32_T30_InOut.h5')
    
    # 2. 输出文件路径 (处理好的图数据)
    output_file = os.path.join('..', 'data','TaxiBJ', 'TaxiBJ15_Graph.npz')

    # ===============================================
    
    # 打印一下绝对路径，方便你自己核对
    print(f"当前脚本位置: {os.path.abspath(__file__)}")
    print(f"计划读取源文件: {os.path.abspath(source_file)}")
    print(f"计划输出到: {os.path.abspath(output_file)}")
    
    process_h5_dataset(source_file, output_file)