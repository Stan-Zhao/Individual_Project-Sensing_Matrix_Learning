import numpy as np

def compute_mu_bound(m, n):
    """
    计算论文 Theorem 2 中定义的理论相干性下界 mu
    m: 测量维度 (d in paper)
    n: 字典原子数 (N in paper)
    """
    if m >= n: 
        return 0
    # 对应论文公式 mu = sqrt((N-d) / (d*(N-1)))
    return np.sqrt((n - m) / (m * (n - 1)))


def get_schnass_sensing_dictionary_pocs(Phi, max_iter=20, tol=1e-6):
    """
    使用 POCS (交替投影) 算法计算论文中的 Sensing Dictionary Psi
    
    参数:
        Phi: 原始字典/测量矩阵 (形状 m x n)
        max_iter: POCS 最大迭代次数
        tol: 收敛容差
    返回:
        Psi: 优化后的感知字典 (形状 m x n)
    """
    m, n = Phi.shape
    mu = compute_mu_bound(m, n)
    
    # 预计算 Phi 的伪逆和投影算子
    Phi_pinv = np.linalg.pinv(Phi)
    P_G = Phi_pinv @ Phi  # 这实际上就是 Phi^dagger * Phi
    
    # 初始化 Gram 矩阵 G = Phi^T * Phi
    G = Phi.T @ Phi
    
    for _ in range(max_iter):
        G_prev = G.copy()
        
        # --- Step 1: Project onto set H (理想的 Gram 矩阵结构) ---
        H = G.copy()
        # 对角线元素设为 1
        np.fill_diagonal(H, 1.0)
        # 非对角线元素截断到 [-mu, mu] 之间
        mask = ~np.eye(n, dtype=bool)
        H[mask] = np.clip(H[mask], -mu, mu)
        
        # --- Step 2: Project onto set G (可实现的 Gram 矩阵) ---
        G = H @ P_G
        
        # 检查收敛
        if np.linalg.norm(G - G_prev, 'fro') < tol:
            break
            
    # 计算最终的 Sensing Dictionary Psi
    # 根据论文: Psi^* = H * Phi^dagger => Psi = (Phi_pinv)^T * H^T
    Psi_T = H @ Phi_pinv
    Psi = Psi_T.T
    
    # 对 Psi 的列进行 L2 归一化
    Psi = Psi / (np.linalg.norm(Psi, axis=0, keepdims=True) + 1e-10)
    
    return Psi


def omp_with_sensing_dict(Phi, Psi, y, k):
    """
    改进的 OMP 算法：使用 Psi 进行匹配，使用 Phi 进行重建
    
    参数:
        Phi: 原始字典/测量矩阵 (形状 m x n)
        Psi: 感知字典 (形状 m x n)，由 get_sensing_dictionary_pocs 计算得到
        y: 测量向量 (形状 m,)
        k: 稀疏度
    返回:
        x_hat: 恢复的稀疏信号 (形状 n,)
    """
    m, n = Phi.shape
    residual = y.copy()
    support = []
    
    x_hat = np.zeros(n)
    
    for _ in range(k):
        # 1. Sensing step (使用 Psi): 找与残差内积最大的原子
        # 对应论文公式: i = arg max |<psi_j, r>|
        correlations = np.abs(Psi.T @ residual)
        
        # 排除已经选过的原子
        correlations[support] = 0  
        
        best_idx = np.argmax(correlations)
        support.append(best_idx)
        
        # 2. Reconstruction step (使用 Phi): 最小二乘投影
        # 对应论文公式: a = Phi_I * Phi_I^dagger * y
        Phi_S = Phi[:, support]
        
        # 计算当前支撑集下的稀疏系数
        x_S = np.linalg.pinv(Phi_S) @ y
        
        # 更新残差 r = y - a
        residual = y - Phi_S @ x_S
        
    # 填充最终的信号
    if len(support) > 0:
        Phi_S = Phi[:, support]
        x_S = np.linalg.pinv(Phi_S) @ y
        x_hat[support] = x_S
        
    return x_hat