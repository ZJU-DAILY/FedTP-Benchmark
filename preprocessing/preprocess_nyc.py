import pandas as pd
import numpy as np
import torch
import os
import csv

def create_grid_graph(height, width):
    """ 生成 HxW 网格的邻接矩阵 """
    print(f"   正在构建 {height}x{width} 的网格图结构...")
    edge_index = []
    for r in range(height):
        for c in range(width):
            curr_node = r * width + c
            # 上下左右
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    neighbor_node = nr * width + nc
                    edge_index.append([curr_node, neighbor_node])
    
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

def process_libcity_dataset(data_dir, dataset_name, save_path):
    print(f"🚀 开始处理 LibCity 数据集: {dataset_name}")
    
    grid_file = os.path.join(data_dir, f'{dataset_name}.grid')
    
    if not os.path.exists(grid_file):
        print(f"❌ 错误：找不到文件 {grid_file}")
        return

    # 1. 读取 .grid 文件 (它本质上是 CSV)
    # 根据 config.json，包含 row_id, column_id, pickup, dropoff
    # LibCity 格式通常第一列是 id，第二列是 type，后面是数据
    print(f"   正在读取 {dataset_name}.grid (这可能需要几秒钟)...")
    
    try:
        # 尝试读取，LibCity 文件通常包含表头，如 dyna_id,type,time,row_id,column_id,pickup,dropoff
        df = pd.read_csv(grid_file)
        
        # 标准化列名 (转小写)
        df.columns = [c.lower() for c in df.columns]
        print(f"   列名: {list(df.columns)}")
        
        # 2. 确定关键列
        # 必须包含: time, row_id, column_id, inflow(pickup), outflow(dropoff)
        time_col = 'time'
        row_col = 'row_id'
        col_col = 'column_id'
        
        # 不同的数据集叫法可能不同 (pickup/inflow)
        in_col = 'pickup' if 'pickup' in df.columns else 'inflow'
        out_col = 'dropoff' if 'dropoff' in df.columns else 'outflow'
        
        if in_col not in df.columns:
            # 容错：有些数据可能叫 population 等，取最后两列数值列
            print("   ⚠️ 未找到 pickup/inflow 列，尝试推断...")
            in_col = df.columns[-2]
            out_col = df.columns[-1]

        # 3. 推断网格大小
        H = df[row_col].max() + 1
        W = df[col_col].max() + 1
        print(f"   推断网格尺寸: Height={H}, Width={W} (总节点={H*W})")
        
        # 4. 转换数据形状
        # 目标: (Time, Nodes, Features=2)
        # 先按 time, row, col 排序确保顺序正确
        df = df.sort_values(by=[time_col, row_col, col_col])
        
        # 提取流量数据
        flow_data = df[[in_col, out_col]].values
        
        # 计算时间步数量
        num_samples = len(df)
        T = num_samples // (H * W)
        
        if num_samples % (H * W) != 0:
            print(f"   ⚠️ 警告：数据行数 ({num_samples}) 不能被 H*W ({H*W}) 整除，可能存在数据缺失！")
            # 截断多余数据
            flow_data = flow_data[:T * H * W]
            
        print(f"   检测到时间步数: {T}")
        
        # Reshape: (Time, H*W, 2)
        # 因为我们已经排好序了 (time 慢变, row 中变, col 快变)，直接 reshape 即可
        data_flatten = flow_data.reshape(T, H * W, 2)
        
        # 5. 生成图结构
        edge_index = create_grid_graph(H, W)
        
        # 6. 保存
        output_dir = os.path.dirname(save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        np.savez(save_path, 
                 data=data_flatten, 
                 edge_index=edge_index.numpy())
        
        print(f"✅ 处理完成！保存至: {save_path}")
        print(f"   数据形状: {data_flatten.shape}")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # 配置路径
    # 假设你的 TaxiNYC 文件夹在 ../data/TaxiNYC
    data_dir = os.path.join('..', 'data', 'TaxiNYC')
    dataset_name = 'NYCTaxi' # 文件名前缀 (NYCTaxi.grid)
    
    output_file = os.path.join('..', 'data', 'TaxiNYC_Graph.npz')
    
    process_libcity_dataset(data_dir, dataset_name, output_file)