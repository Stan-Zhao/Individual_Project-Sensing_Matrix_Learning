import numpy as np



def design_pinv_psi_fast(Phi, X,tau=1e-6):
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
    # trace_val = np.trace(G)
    # if trace_val > 0:
    #     epsilon = tau * trace_val / K
    # else:
    #     epsilon = tau
        
    # 5. 求逆 (K x K)
    # 速度极快
    try:
        Q = np.linalg.inv(G + tau * np.eye(K))
    except np.linalg.LinAlgError:
        Q = np.linalg.pinv(G + tau * np.eye(K))
        
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

    #     # =========================
    # # ✅ 6. 列归一化（关键🔥）
    # # =========================
    # norms = np.linalg.norm(Psi, axis=0, keepdims=True) + 1e-12
    # Psi = Psi / norms
    
    return Psi

def pro_omp_solve(Phi, Psi, y, sparsity):
    M, N = Phi.shape
    residual = y.copy()
    support = []

    # 预先分配空间，避免循环中重复分配
    x_est = np.zeros(N)
    Psi_T = Psi.T

    for _ in range(sparsity):
        # 1. 计算相关性
        correlations = np.abs(Psi_T @ residual)
        
        # 【关键修正】: 将已经在 support 中的索引的相关性置为 0
        # 这样 argmax 就会自动去寻找“还没被选中的”里面最大的那个
        if support:
            correlations[support] = 0.0
            
        # 2. 选择最佳原子
        idx = np.argmax(correlations)
        
        # 双重保险：如果 mask 之后最大值还是 0 (说明可能所有都被选完了或者数值问题)，则退出
        if correlations[idx] < 1e-10: 
            break
            
        # 3. 更新支撑集
        support.append(idx)

        # 4. 投影与残差更新 (Least Squares)
        # 注意：这里必须用 Phi (物理测量矩阵) 来做投影，而不是 Psi
        Phi_sub = Phi[:, support]
        x_ls, _, _, _ = np.linalg.lstsq(Phi_sub, y, rcond=None)
        
        # 更新残差：residual = y - Phi_S * x_S
        residual = y - Phi_sub @ x_ls

    # 构建最终结果
    for i, s in enumerate(support):
        x_est[s] = x_ls[i]
        
    return x_est

def pinv_omp(Phi, y, X, k,tau):
    # 计算 Psi 矩阵
    Psi = design_pinv_psi_fast(Phi, X,tau)
    
    # 使用改进的 OMP 算法进行稀疏恢复
    x_hat = pro_omp_solve(Phi, Psi, y, k)
    
    return x_hat