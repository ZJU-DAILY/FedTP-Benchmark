import torch
import networkx as nx
import numpy as np

def noderank_partition(edge_index, num_nodes, num_clients, beta=0.9):
    """
    STAGCN-EC 原文献的 NodeRank 图切分算法 [cite: 184, 198-202, 266-288]
    """
    # 1. 构建 NetworkX 全局图
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()
    
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_index.T)

    # 2. 计算介数中心性 (Betweenness Centrality) [cite: 203, 204]
    cb = nx.betweenness_centrality(G, normalized=True)
    
    # 3. 构建转移矩阵 M (Eq. 5) [cite: 268, 269]
    M = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        neighbors = list(G.neighbors(i))
        if len(neighbors) > 0:
            sum_cb = sum([cb[j] for j in neighbors])
            if sum_cb > 0:
                for j in neighbors:
                    M[i, j] = cb[j] / sum_cb
            else:
                for j in neighbors:
                    M[i, j] = 1.0 / len(neighbors)
                    
    # 4. 计算阻尼系数 alpha (Eq. 6) [cite: 272-276]
    # 简化：如果没有真实物理距离，使用最短路径长度近似距离 d_ij
    alpha = np.zeros(num_nodes)
    shortest_paths = dict(nx.shortest_path_length(G))
    for j in range(num_nodes):
        sum_inv_d = 0
        for i in range(num_nodes):
            if i != j and i in shortest_paths and j in shortest_paths[i]:
                d_ij = shortest_paths[i][j]
                sum_inv_d += 1.0 / d_ij
        alpha[j] = beta * (1.0 / sum_inv_d) if sum_inv_d > 0 else 0
        
    # 5. 迭代计算 NodeRank (Eq. 7) [cite: 278-282]
    R = np.ones(num_nodes) / num_nodes
    alpha_matrix = np.diag(alpha)
    I_minus_alpha = np.diag(1 - alpha)
    E_matrix = np.ones((num_nodes, num_nodes)) / num_nodes
    
    for _ in range(100): # 迭代求解稳态分布
        R_new = np.dot(alpha_matrix, np.dot(M, R)) + np.dot(I_minus_alpha, np.dot(E_matrix, R))
        if np.linalg.norm(R_new - R) < 1e-6:
            break
        R = R_new

    # 6. 选取 Top M 个核心节点作为子图中心 (Algorithm 1, Step 2) [cite: 226, 289]
    top_m_indices = np.argsort(R)[-num_clients:][::-1]
    
    # 7. BFS 将剩余节点分配给最近的中心节点
    partitions = {i: [] for i in range(num_clients)}
    node_to_client = {}
    
    for client_id, center_node in enumerate(top_m_indices):
        partitions[client_id].append(center_node)
        node_to_client[center_node] = client_id
        
    for node in range(num_nodes):
        if node not in node_to_client:
            # 找离它最近的中心节点
            min_dist = float('inf')
            closest_client = 0
            for client_id, center_node in enumerate(top_m_indices):
                if node in shortest_paths and center_node in shortest_paths[node]:
                    dist = shortest_paths[node][center_node]
                    if dist < min_dist:
                        min_dist = dist
                        closest_client = client_id
            partitions[closest_client].append(node)
            node_to_client[node] = closest_client
            
    # 修改后：将 NumPy 标量强制转为 Python 原生 int，安全通过 FATE 序列化
    nodes_per = [sorted([int(node) for node in partitions[i]]) for i in range(num_clients)]
    
    # 顺手把 node_to_client 里的 key 和 value 也转成原生 int
    safe_node_to_client = {int(k): int(v) for k, v in node_to_client.items()}
    
    return nodes_per, safe_node_to_client