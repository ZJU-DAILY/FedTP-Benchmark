import torch
import networkx as nx

def get_equiconcept_matrix(edge_index, num_nodes, device, max_fca_dim=100):
    print(f"🔄 [FCA] 提取最大团并对齐全局维度至 {max_fca_dim}...")
    
    # 转换为无向图提取 Clique
    edges = edge_index.t().cpu().numpy().tolist()
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)
    
    equi_concepts = list(nx.find_cliques(G))
    
    # 创建固定维度的 0 矩阵进行 Padding
    f_matrix = torch.zeros((num_nodes, max_fca_dim), device=device)
    
    for col_idx, clique_nodes in enumerate(equi_concepts):
        if col_idx >= max_fca_dim:
            break # 超过最大维度直接截断
        f_matrix[clique_nodes, col_idx] = 1.0
        
    return f_matrix