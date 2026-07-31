import numpy as np

def design_pinv_psi_fast(Phi, X):
    """
    [改进版] 计算最优传感矩阵 Psi。
    利用 Woodbury 恒等式将求逆复杂度从 O(M^3) 降低到 O(K^3)。
    
    公式变换: 
    原公式: Psi = (A A^T + eps*I)^-1 A X^T
    新公式: Psi = A (A^T A + eps*I)^-1 X^T
    
    其中 A = Phi * X
    """
    # 1. 维度处理
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        
    N, K = X.shape
    M, _ = Phi.shape

    # 2. 计算 A = Phi * X (M x K)
    # 这一步将高维信号降维到测量域
    A = np.dot(Phi, X)
    
    # 3. 计算核矩阵 G = A^T * A (K x K)
    # [核心优势] 这里的 G 非常小，例如 15x15，而原方法是 256x256
    G = np.dot(A.T, A)
    
    # 4. 正则化参数 (自适应)
    # 避免矩阵奇异
    trace_val = np.trace(G)
    if trace_val > 0:
        epsilon = 1e-6 * trace_val / K
    else:
        epsilon = 1e-6
        
    # 5. 求逆 (K x K)
    # 速度极快
    try:
        Q = np.linalg.inv(G + epsilon * np.eye(K))
    except np.linalg.LinAlgError:
        Q = np.linalg.pinv(G + epsilon * np.eye(K))
        
    # 6. 组合最终的 Psi (M x N)
    # Psi = A * Q * X^T
    # 注意乘法顺序以减少计算量：先算 (X @ Q)，结果是 (N, K)
    # 再算 A @ ...
    # 但根据你的 OMP，你需要的是 Psi，使得 Psi.T @ r 近似 x
    # 原代码 Psi = (AA^T)^-1 A X^T
    # 变换后 Psi = A (A^T A)^-1 X^T = A Q X^T
    
    # 我们先计算中间项，避免生成巨大的中间矩阵
    # part1 = A @ Q  -> (M, K)
    part1 = np.dot(A, Q)
    
    # Psi = part1 @ X.T -> (M, N)
    Psi = np.dot(part1, X.T)
    
    return Psi
