from typing import Literal
import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import euclidean_distances

from lib.normalization import MinMax01Scaler, MinMax11Scaler, StandardScaler
from lib.get_adjacent_matrix import get_normalized_matrix
from data.dataset import UnifiedTrafficDataset 

def _build_compact_edge_index_from_distance(dist_df, num_nodes, dataset_name):
    """
    将 distance.csv 中的 from/to 统一转成 0..N-1 的紧凑节点编号。

    适配三类情况：
    1. PeMS04 / PeMS08：from/to 已经是 0..N-1，直接使用；
    2. PeMS03：from/to 是原始 sensor id，需要映射到 0..N-1；
    3. PeMSD7：通常也是 0..N-1，直接使用。
    """
    required_cols = {"from", "to"}
    if not required_cols.issubset(set(dist_df.columns)):
        raise ValueError(
            f"[Graph Parser] {dataset_name} distance.csv 缺少 from/to 列，"
            f"当前列为: {list(dist_df.columns)}"
        )

    src_raw = dist_df["from"].astype(int).tolist()
    dst_raw = dist_df["to"].astype(int).tolist()

    unique_ids = sorted(set(src_raw) | set(dst_raw))

    # 情况一：已经是 0..N-1 编号，例如 PeMS04 / PeMS08 / PeMSD7
    if unique_ids == list(range(num_nodes)):
        edge_index = torch.tensor([src_raw, dst_raw], dtype=torch.long)
        print(
            f"[Graph Parser] {dataset_name} 使用紧凑节点编号 0..{num_nodes - 1} | "
            f"edges={edge_index.shape[1]}"
        )
        return edge_index

    # 情况二：原始 sensor id，例如 PeMS03
    if len(unique_ids) != num_nodes:
        raise ValueError(
            f"[Graph Parser] {dataset_name} 无法建立节点映射："
            f"distance.csv 唯一节点数={len(unique_ids)}，"
            f"但数据节点数={num_nodes}。"
            f"请检查 distance.csv 是否与 npz 数据对应，或补充 sensor_ids.txt。"
        )

    raw_to_compact = {raw_id: idx for idx, raw_id in enumerate(unique_ids)}

    src = [raw_to_compact[x] for x in src_raw]
    dst = [raw_to_compact[x] for x in dst_raw]

    edge_index = torch.tensor([src, dst], dtype=torch.long)

    print(
        f"[Graph Parser] {dataset_name} 检测到原始 sensor id，"
        f"已映射为紧凑编号 0..{num_nodes - 1} | "
        f"raw_id_range=({min(unique_ids)}, {max(unique_ids)}) | "
        f"edges={edge_index.shape[1]}"
    )

    return edge_index


def read_st_dataset_file(dataset, feature_type, selected_nodes=None):
    type2index = {"flow": 0, "occ": 1, "speed": 2}
    feature_index = type2index[feature_type]

    # 动态获取项目根目录路径
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path))

    edge_index = None

    if dataset == 'PeMS04':
        data_path = os.path.join(project_root, 'data/PeMS04/pems04.npz')
        dist_path = os.path.join(project_root, 'data/PeMS04/distance.csv')

        data = np.load(data_path)
        data = data['data']
        data = data[:, :, feature_index]
        dist = pd.read_csv(dist_path)
        edge_index = _build_compact_edge_index_from_distance(
            dist_df=dist,
            num_nodes=data.shape[1],
            dataset_name=dataset
        )

    elif dataset == 'PeMS08':
        data_path = os.path.join(project_root, 'data/PeMS08/pems08.npz')
        dist_path = os.path.join(project_root, 'data/PeMS08/distance.csv')

        data = np.load(data_path)
        data = data['data']
        data = data[:, :, feature_index]
        dist = pd.read_csv(dist_path)
        edge_index = _build_compact_edge_index_from_distance(
            dist_df=dist,
            num_nodes=data.shape[1],
            dataset_name=dataset
        )

    elif dataset == 'PeMS03':
        data_path = os.path.join(project_root, 'data/PeMS03/pems03.npz')
        dist_path = os.path.join(project_root, 'data/PeMS03/distance.csv')

        data = np.load(data_path)
        data = data['data']
        data = data[:, :, feature_index]
        dist = pd.read_csv(dist_path)
        edge_index = _build_compact_edge_index_from_distance(
            dist_df=dist,
            num_nodes=data.shape[1],
            dataset_name=dataset
        )

    elif dataset == 'PeMSD7':
        data_path = os.path.join(project_root, 'data/PeMSD7/pemsd7.npz')
        dist_path = os.path.join(project_root, 'data/PeMSD7/distance.csv')

        data = np.load(data_path)
        data = data['data']
        data = data[:, :, feature_index]
        dist = pd.read_csv(dist_path)
        
        # 【核心修复：Top-K 图稀疏化】
        if 'cost' in dist.columns and dist['cost'].max() <= 1.0:
            K = 8  # 交通领域经典的近邻数
            # 1. 为每个节点保留权重最大的 K 个邻居
            topk_dist = dist.sort_values(['from', 'cost'], ascending=[True, False]).groupby('from').head(K)
            
            # 2. 强制对称化 (A = A + A^T)，保证 FCA 能提取无向最大团
            rev_dist = topk_dist.copy()
            rev_dist['from'], rev_dist['to'] = topk_dist['to'], topk_dist['from']
            dist = pd.concat([topk_dist, rev_dist]).drop_duplicates(subset=['from', 'to'])
            print(f"👉 [Dataset] PeMSD7 已执行 Top-{K} 稀疏化与对称化！剩余边数: {len(dist)}")

        edge_index = torch.tensor([dist['from'].tolist(), dist['to'].tolist()], dtype=torch.long)

    # 同时支持 TaxiBJ / TaxiNYC 等图数据
    elif 'Taxi' in dataset:
        data_path = os.path.join(project_root, f'data/{dataset}.npz')

        if not os.path.exists(data_path):
            prefix = dataset.split('_')[0]
            data_path = os.path.join(project_root, f'data/{prefix}/{dataset}.npz')

        if os.path.exists(data_path):
            print(f"Loading Graph data from: {data_path}")
            raw = np.load(data_path)
            data = raw['data']  # (Time, Nodes, Features)
            edge_index = raw['edge_index']  # (2, Edges)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
        else:
            raise FileNotFoundError(f"Cannot find dataset file: {data_path}")

    elif dataset == "HK":
        data_path = os.path.join(project_root, 'data/HK/hk.npy')
        dist_path = os.path.join(project_root, 'data/HK/distance.csv')
        data = np.load(data_path)
        data = data[:, :, feature_index]
        dist = pd.read_csv(dist_path)
        edge_index = torch.tensor([dist['from'].tolist(), dist['to'].tolist()], dtype=torch.long)

    elif dataset == "FT":
        data_path = os.path.join(project_root, 'data/FT-AED/nashville.npy')
        dist_path = os.path.join(project_root, 'data/FT-AED/distance.csv')
        data = np.load(data_path)
        data = data[:, :, feature_index]
        dist = pd.read_csv(dist_path)
        edge_index = torch.tensor([dist['from'].tolist(), dist['to'].tolist()], dtype=torch.long)

    elif dataset == "essen":
        data = np.load(os.path.join(project_root, 'data/utd19/essen_1h.npy'))
        data = data[:, :, feature_index]

    elif dataset == "groningen":
        data = np.load(os.path.join(project_root, 'data/utd19/groningen_1h.npy'))
        data = data[:, :, feature_index]

    elif dataset == "hamburg":
        data = np.load(os.path.join(project_root, 'data/utd19/hamburg_1h.npy'))
        data = data[:, :, feature_index]

    elif dataset == "marseille":
        data = np.load(os.path.join(project_root, 'data/utd19/marseille_1h.npy'))
        data = data[:, :, feature_index]

    elif dataset == "paris":
        data = np.load(os.path.join(project_root, 'data/utd19/paris_1h.npy'))
        data = data[:, :, feature_index]

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if selected_nodes is not None:
        # 1. 切分数据 (Global Data -> Local Data)
        data = data[:, selected_nodes]

        # 2. 筛边 + 全局 ID 映射到本地 ID
        if edge_index is not None:
            sel_nodes_tensor = torch.tensor(selected_nodes, dtype=torch.long)
            mask0 = torch.isin(edge_index[0], sel_nodes_tensor)
            mask1 = torch.isin(edge_index[1], sel_nodes_tensor)
            mask = mask0 & mask1
            edge_index = edge_index[:, mask]

            # 建立映射字典: {Global_ID: Local_Index}
            node_map = {global_id: local_id for local_id, global_id in enumerate(selected_nodes)}

            src_np = edge_index[0].numpy()
            dst_np = edge_index[1].numpy()

            new_src = [node_map[x] for x in src_np]
            new_dst = [node_map[x] for x in dst_np]

            edge_index = torch.tensor([new_src, new_dst], dtype=torch.long)

            print(
                f"[Local Graph] {dataset} | selected_nodes={len(selected_nodes)} | "
                f"local_edges={edge_index.shape[1]}"
                )

            if edge_index.shape[1] == 0:
                print(
                    f"[严重警告] {dataset} 当前客户端筛边后 local_edges=0。"
                    f"这通常说明 distance.csv 节点编号与 selected_nodes 编号体系不一致，"
                    f"或者 METIS 划分与当前图文件不匹配。"
                )

    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)

    print(
        'Load %s Dataset shaped: ' % dataset,
        data.shape, data.max(), data.min(), data.mean(), np.std(data), np.median(data)
    )
    return data, edge_index


def _sanitize_std(std, eps=1e-8):
    """
    避免 std 为 0 导致除零。
    """
    if np.isscalar(std):
        return 1.0 if abs(std) < eps else std
    std = np.array(std, copy=True)
    std[np.abs(std) < eps] = 1.0
    return std


def _sanitize_minmax(minimum, maximum, eps=1e-8):
    """
    避免 max == min 导致除零。
    """
    if np.isscalar(minimum) and np.isscalar(maximum):
        if abs(maximum - minimum) < eps:
            maximum = minimum + 1.0
        return minimum, maximum

    minimum = np.array(minimum, copy=True)
    maximum = np.array(maximum, copy=True)
    same_mask = np.abs(maximum - minimum) < eps
    maximum[same_mask] = minimum[same_mask] + 1.0
    return minimum, maximum


def _fit_axes_for_scope(norm_scope, column_wise=False):
    scope = (norm_scope or "global").lower()
    if column_wise and scope == "global":
        return 0
    if scope == "global":
        return None
    if scope in {"node", "per_node", "per-node"}:
        # train_data is (samples, nodes, time, features). Keep a separate
        # scaler for each node/feature and fit over samples + time.
        return (0, 2)
    if scope in {"column", "column_wise", "column-wise"} or column_wise:
        return 0
    raise ValueError(
        f"Unsupported norm_scope: {norm_scope}. "
        "Use one of: global, node, column."
    )


def _scope_name(norm_scope, column_wise=False):
    if column_wise and (norm_scope is None or norm_scope == "global"):
        return "column"
    return (norm_scope or "global").lower()


def build_scaler_from_train(train_data, normalizer, column_wise=False, norm_scope="global"):
    """
    只用训练集拟合 scaler，避免数据泄漏。
    """
    fit_axes = _fit_axes_for_scope(norm_scope, column_wise=column_wise)
    scope = _scope_name(norm_scope, column_wise=column_wise)

    if normalizer == 'max01':
        if fit_axes is None:
            minimum = train_data.min()
            maximum = train_data.max()
        else:
            minimum = train_data.min(axis=fit_axes, keepdims=True)
            maximum = train_data.max(axis=fit_axes, keepdims=True)
        minimum, maximum = _sanitize_minmax(minimum, maximum)
        scaler = MinMax01Scaler(minimum, maximum)
        print(f'Fit MinMax01 scaler on training split only | scope={scope}')

    elif normalizer == 'max11':
        if fit_axes is None:
            minimum = train_data.min()
            maximum = train_data.max()
        else:
            minimum = train_data.min(axis=fit_axes, keepdims=True)
            maximum = train_data.max(axis=fit_axes, keepdims=True)
        minimum, maximum = _sanitize_minmax(minimum, maximum)
        scaler = MinMax11Scaler(minimum, maximum)
        print(f'Fit MinMax11 scaler on training split only | scope={scope}')

    elif normalizer == 'std':
        if fit_axes is None:
            mean = train_data.mean()
            std = train_data.std()
        else:
            mean = train_data.mean(axis=fit_axes, keepdims=True)
            std = train_data.std(axis=fit_axes, keepdims=True)
        std = _sanitize_std(std)
        scaler = StandardScaler(mean, std)
        print(f'Fit Standard scaler on training split only | scope={scope}')

    else:
        raise ValueError(f"Unsupported normalizer: {normalizer}")

    return scaler


def apply_scaler(data, scaler):
    return scaler.transform(data)


def normalize_dataset(data, normalizer, column_wise=False, norm_scope="global"):
    """
    保留这个接口，兼容其他可能的调用。
    注意：这个函数本身不会切分 train/val/test；
    在 load_dataset() 中，真正使用的是“先切分、再仅用训练集拟合 scaler”的流程。
    """
    scaler = build_scaler_from_train(
        data,
        normalizer,
        column_wise=column_wise,
        norm_scope=norm_scope,
    )
    data = apply_scaler(data, scaler)
    return data, scaler


def apply_scaler_to_raw_series(data, scaler):
    """Scale raw series shaped (time, nodes, features) with a window scaler."""
    if data.ndim != 3:
        return apply_scaler(data, scaler)
    data_window_layout = np.expand_dims(data, axis=0).transpose(0, 2, 1, 3)
    scaled = apply_scaler(data_window_layout, scaler)
    return scaled[0].transpose(1, 0, 2)


def Add_Window_Horizon(data, window=3, horizon=1, single=False):
    """
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    """
    length = len(data)
    end_index = length - horizon - window + 1
    X = []
    Y = []
    index = 0

    if single:
        while index < end_index:
            X.append(data[index:index + window])
            Y.append(data[index + window + horizon - 1:index + window + horizon])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index:index + window])
            Y.append(data[index + window:index + window + horizon])
            index = index + 1

    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def split_data_by_time(X, Y, train_ratio, val_ratio, test_ratio):
    """
    严格按时间顺序切分：
    Train -> Val -> Test
    """
    data_len = X.shape[0]

    train_end = int(data_len * train_ratio)
    val_end = int(data_len * (train_ratio + val_ratio))

    x_tra = X[:train_end]
    y_tra = Y[:train_end]

    x_val = X[train_end:val_end]
    y_val = Y[train_end:val_end]

    x_test = X[val_end:]
    y_test = Y[val_end:]

    return x_tra, y_tra, x_val, y_val, x_test, y_test


def getTensorDataset(X, Y, device):
    X, Y = torch.FloatTensor(X).to(device), torch.FloatTensor(Y).to(device)
    data = torch.utils.data.TensorDataset(X, Y)
    return data


def load_dataset(
    dataset_name: str,
    feature_type: Literal["flow", "occ", "speed"] = "flow",
    normalizer='std',
    T_in=12,
    T_out=3,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    return_edge_index=False,
    device='cpu',
    selected_nodes=None,
    norm_scope="global",
    scaler_fit_scope="selected"
):
    scaler_fit_scope = (scaler_fit_scope or "selected").lower()
    use_full_graph_scaler = selected_nodes is not None and scaler_fit_scope in {
        "full", "global", "global_graph", "all",
    }
    if use_full_graph_scaler and (norm_scope or "global").lower() != "global":
        raise ValueError(
            "scaler_fit_scope='full' currently requires norm_scope='global' "
            "because node/column-wise scalers have full-graph shapes."
        )

    full_graph_scaler = None
    if use_full_graph_scaler:
        full_data, _ = read_st_dataset_file(dataset_name, feature_type, selected_nodes=None)
        full_X, full_Y = Add_Window_Horizon(full_data, T_in, T_out)
        full_X = full_X.transpose(0, 2, 1, 3)
        full_Y = full_Y.transpose(0, 2, 1, 3)
        full_x_tra_raw, full_y_tra_raw, _, _, _, _ = split_data_by_time(
            full_X, full_Y, train_ratio, val_ratio, test_ratio
        )
        full_train_fit_data = np.concatenate([full_x_tra_raw, full_y_tra_raw], axis=2)
        full_graph_scaler = build_scaler_from_train(
            full_train_fit_data,
            normalizer,
            column_wise=False,
            norm_scope=norm_scope,
        )
        print(
            f"Fit {normalizer} scaler on FULL graph training split; "
            f"apply to selected_nodes={len(selected_nodes)}"
        )
    # 1. 加载原始时空数据: (T, N, 1)
    data, edge_index = read_st_dataset_file(dataset_name, feature_type, selected_nodes)

    # 2. 先在原始数据上构造滑动窗口
    X, Y = Add_Window_Horizon(data, T_in, T_out)
    X = X.transpose(0, 2, 1, 3)  # -> (Samples, N, T_in, 1)
    Y = Y.transpose(0, 2, 1, 3)  # -> (Samples, N, T_out, 1)

    # 3. 严格按时间切分原始窗口
    x_tra_raw, y_tra_raw, x_val_raw, y_val_raw, x_test_raw, y_test_raw = split_data_by_time(
        X, Y, train_ratio, val_ratio, test_ratio
    )

    # 4. 只用训练集拟合 scaler，避免数据泄漏
    #    用训练集的输入和标签共同拟合，保证训练阶段涉及到的数值范围都被覆盖
    train_fit_data = np.concatenate([x_tra_raw, y_tra_raw], axis=2)
    scaler = full_graph_scaler or build_scaler_from_train(
        train_fit_data,
        normalizer,
        column_wise=False,
        norm_scope=norm_scope,
    )

    # 5. 用同一个 scaler 变换 train / val / test
    x_tra = apply_scaler(x_tra_raw, scaler)
    y_tra = apply_scaler(y_tra_raw, scaler)
    x_val = apply_scaler(x_val_raw, scaler)
    y_val = apply_scaler(y_val_raw, scaler)
    x_test = apply_scaler(x_test_raw, scaler)
    y_test = apply_scaler(y_test_raw, scaler)

    # 6. 同样把整段时序用训练集 scaler 变换，供 A_norm 使用
    data_scaled = apply_scaler_to_raw_series(data, scaler)

    print(f'Train Data: {x_tra.shape}, {y_tra.shape}')
    print(f'Val Data:   {x_val.shape}, {y_val.shape}')
    print(f'Test Data:  {x_test.shape}, {y_test.shape}')

    # 7. 封装为 TensorDataset
    train_data = getTensorDataset(x_tra, y_tra, device)
    val_data = getTensorDataset(x_val, y_val, device)
    test_data = getTensorDataset(x_test, y_test, device)

    # 8. 返回
    if not return_edge_index:
        A_norm = get_normalized_matrix(data_scaled.transpose(1, 0, 2))
        A_norm = torch.FloatTensor(A_norm).to(device)
        return train_data, val_data, test_data, A_norm, scaler
    else:
        return train_data, val_data, test_data, edge_index.to(device), scaler


def spectral_community_detection(adj_mx, n_clusters):
    """
    FedAGAT 中用到的谱聚类社区发现。
    这里保留原函数，和本次归一化修改无冲突。
    """
    distances = euclidean_distances(adj_mx)
    max_dist = distances.max()

    affinity_mx = adj_mx.copy()
    affinity_mx[affinity_mx == 0] = max_dist

    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42
    )
    clusters = model.fit_predict(affinity_mx)

    local_subnetworks = []
    for i in range(n_clusters):
        idx = np.where(clusters == i)[0].tolist()
        idx.sort()
        local_subnetworks.append(idx)

    return local_subnetworks


def load_grid_dataset_for_fedstn(dataset_name, t_in, t_out, device, selected_nodes=None, model_name="default"):
    print(f"Loading Grid Dataset [{dataset_name}] with Graph View for model [{model_name}]...")
    # 传入 model_name
    train_set = UnifiedTrafficDataset(city=dataset_name, split="train", len_c=t_in, t_out=t_out, view_type="graph", selected_nodes=selected_nodes, model_name=model_name)
    val_set = UnifiedTrafficDataset(city=dataset_name, split="val", len_c=t_in, t_out=t_out, view_type="graph", selected_nodes=selected_nodes, model_name=model_name)
    test_set = UnifiedTrafficDataset(city=dataset_name, split="test", len_c=t_in, t_out=t_out, view_type="graph", selected_nodes=selected_nodes, model_name=model_name)
    
    # 2. 从 train_set 中提取图的 4-邻接矩阵，并转为 edge_index
    adj_matrix = torch.tensor(train_set.adj, dtype=torch.float32)
    edge_index = adj_matrix.nonzero(as_tuple=False).t().contiguous()
    
    # 3. 构造适配器 Scaler
    scaler = GridScalerAdapter(train_set.flow_min, train_set.flow_max)
    
    return train_set, val_set, test_set, edge_index, scaler



class GridScalerAdapter:
    """包装 UnifiedTrafficDataset 中的 min/max，伪装成 fate_main 认识的 scaler"""
    def __init__(self, flow_min, flow_max):
        # 你的 dataset 算出来的是 numpy 数组，转为 tensor
        self.min = torch.tensor(flow_min, dtype=torch.float32)
        self.max = torch.tensor(flow_max, dtype=torch.float32)
        self.metrics_coef = 1.0 # 占位，适配代码
        
    def inverse_transform(self, x):
        den = self.max - self.min
        den[den == 0] = 1.0
        if isinstance(x, torch.Tensor):
            # 将 min/max 移动到与 x 相同的设备上
            den = den.to(x.device)
            dmin = self.min.to(x.device)
            return (x * den) + dmin
        else:
            return (x * den.numpy()) + self.min.numpy()


def build_fedstg_static_adj(dataset_name, num_nodes, sigma=0.0, kappa=0.0, project_root='.'):
    """
    为 FedSTG 专门构建的全局静态图 (阈值高斯核加权)
    严格遵循 Eq(7): 仅当 distance <= kappa 时保留边，并保证对称性。
    """

    dist_path = os.path.join(project_root, 'data', dataset_name, 'distance.csv')
    adj = np.zeros((num_nodes, num_nodes))

    if not os.path.exists(dist_path):
        print(f"[FedSTG] 警告: 未找到 {dist_path}，使用单位阵退化处理。")
        return torch.eye(num_nodes)

    df = pd.read_csv(dist_path)
    dist_col = 'cost' if 'cost' in df.columns else 'weight'
    valid_dists = df[dist_col].values

    # 自适应参数计算 (工程启发式)
    if sigma <= 0.0:
        sigma = np.std(valid_dists) if len(valid_dists) > 0 else 10.0
        print(f"[FedSTG] 启用自适应 Sigma (距离标准差): {sigma:.4f}")
    
    if kappa <= 0.0:
        kappa = float('inf') # 不设硬截断，仅靠高斯核自然衰减
        print(f"[FedSTG] 未设置硬性距离阈值 Kappa，仅依靠高斯核自然衰减。")

    edges_added = 0
    # 严格遵循公式构图
    for _, row in df.iterrows():
        u, v, dist = int(row['from']), int(row['to']), float(row[dist_col])
        
        if u < num_nodes and v < num_nodes and u != v:
            if dist <= kappa:
                weight = np.exp(- (dist**2) / (sigma**2))
                # 强制对称写入
                adj[u, v] = weight
                adj[v, u] = weight
                edges_added += 1

    print(f"[FedSTG] 静态图构建完成 | 全局节点: {num_nodes} | 无向边数: {edges_added//2}")
    return torch.FloatTensor(adj)
