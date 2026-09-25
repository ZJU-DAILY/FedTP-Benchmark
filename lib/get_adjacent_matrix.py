import numpy as np
import pandas as pd

def compute_gra_similarity(matrix):
    # 计算灰色关联相似性
    n = matrix.shape[0]
    gra_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # GRA计算逻辑（简化示例）
            diff = np.abs(matrix[i] - matrix[j])
            min_diff = np.min(diff)
            max_diff = np.max(diff)
            gra = (min_diff + 0.5 * max_diff) / (diff + 0.5 * max_diff)
            gra_matrix[i,j] = np.mean(gra)
    return gra_matrix


def DTWDistance(s1, s2):
    DTW = {}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    return np.sqrt(DTW[len(s1) - 1, len(s2) - 1])


def get_dwt_matrix(x: np.ndarray, max_len=20):
    # X.shape [N, L, C]
    N, L, C = x.shape
    if L > max_len:
        x = x[:, :max_len, :]
    S = []
    for c in range(C):
        x_input = x[..., c] # [N, L]
        sim_matrix = np.zeros((N, N))
        for i in range(N):
            s1 = x_input[i]
            for j in range(N):
                s2 = x_input[j]
                dist = DTWDistance(s1, s2)
                sim_matrix[i, j] = dist
        S.append(sim_matrix)

    S = np.stack(S, 0).mean(0)

    return S

def get_gra_matrix(x: np.ndarray, rou=0.5):
    # no climate feature data
    # X.shape [N, L, C]
    N, L, C = x.shape
    S = []
    for c in range(C):
        F = x[..., c] # [N, L]
        mu = np.mean(F, axis=1).reshape(-1, 1) # [N, 1]
        sigma = np.std(F, axis=1).reshape(-1, 1) # [N, 1]
        F = (F - mu) / sigma

        # No extra feature data
        G = F
        gra_matrix = np.zeros((N, N))
        for i in range(N):
            delta = np.abs(G[i] - G)
            c = np.min(delta)
            d = np.max(delta)
            sim = (c + rou * d) / (delta + rou * d)
            gra_matrix[i] = sim.mean(1)

        S.append(gra_matrix)

    S = np.stack(S, 0).mean(0)
    return S


def normalize_sim_matrix(sim_matrix: np.ndarray):
    """ similarity should be between 0-1 """
    # sim_matrix.shape [N, N]
    min_val = sim_matrix.min()
    max_val = sim_matrix.max()
    sim_matrix = (sim_matrix - min_val) / (max_val - min_val)
    return sim_matrix

def get_adjacent_matrix(x: np.ndarray):
    # X.shape [N, L, C]
    N, L, C = x.shape
    S1 = get_dwt_matrix(x)
    S1 = normalize_sim_matrix(S1)
    S2 = get_gra_matrix(x)
    S2 = normalize_sim_matrix(S2)
    k1, k2 = 1, -1
    A = np.exp(-k1 * S1 - k2 * S2)
    return A

def norm_adj(A):
    D = np.array(np.sum(A, axis=1)).reshape((-1,))

    D[D <= 1e-5] = 1e-5  # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_norm = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    A_norm[A_norm <= 1e-5] = 0
    return A_norm


def get_normalized_matrix(X: np.ndarray):
    assert type(X) == np.ndarray and X.ndim == 3
    A = get_adjacent_matrix(X)
    A_norm = norm_adj(A)
    return A_norm


if __name__ == '__main__':
    X = np.random.random((200, 100, 2))
    print(X.shape)
    print(X.min(), X.max(), X.mean(), X.std())
    A = get_adjacent_matrix(X)
    print(A.shape)
    print(A.diagonal())
    print(A.min(), A.max(), A.mean(), A.std())

    A_norm = get_normalized_matrix(X)
    print(A_norm.shape)
    print(A_norm.diagonal())
    print(A_norm.min(), A_norm.max(), A_norm.mean(), A_norm.std())

    X = X * 100
    print(X.shape)
    print(X.min(), X.max(), X.mean(), X.std())
    A = get_adjacent_matrix(X)
    print(A.shape)
    print(A.diagonal())
    print(A.min(), A.max(), A.mean(), A.std())

    A_norm = get_normalized_matrix(X)
    print(A_norm.shape)
    print(A_norm.diagonal())
    print(A_norm.min(), A_norm.max(), A_norm.mean(), A_norm.std())
