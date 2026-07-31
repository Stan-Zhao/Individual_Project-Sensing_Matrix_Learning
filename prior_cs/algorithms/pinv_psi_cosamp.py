import numpy as np
from numpy.linalg import norm, lstsq, pinv

def design_pinv_psi(Phi, X):
    """
    计算 Pinv 先验矩阵
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # 增加正则化项防止奇异矩阵 (Robust Pinv)
    # Rxx = X @ X.T
    # Psi = (Phi Rxx Phi^T + lambda I)^-1 (Phi Rxx)
    
    n_samples = X.shape[1]
    # 计算相关矩阵 Rxx (简化版)
    # 直接用你的逻辑: Psi = (Phi X X^T Phi^T)^+ Phi X X^T
    
    A = Phi @ X
    AA_T = A @ A.T
    AX_T = A @ X.T
    
    # 增加微小的对角加载，极大提高数值稳定性
    epsilon = 1e-6 * np.trace(AA_T) / AA_T.shape[0]
    AA_T_reg = AA_T + epsilon * np.eye(AA_T.shape[0])
    
    AA_T_inv = pinv(AA_T_reg)
    Psi = AA_T_inv @ AX_T
    
    return Psi

def pro_cosamp_solve(Phi, Psi, y, sparsity, max_iter=20, tol=1e-6):
    """
    修正后的 Pinv-CoSaMP
    """
    m, n = Phi.shape
    x_hat = np.zeros(n)
    residual = y.copy()
    support = np.array([], dtype=int)
    
    # [关键] 动态调整 sparsity
    # 如果 M < 3*K，CoSaMP 必死无疑。
    # 这里做一个简单的保护：如果 3K > M，强行降低内部计算的 K，防止报错，
    # 但这会导致恢复质量下降（欠拟合）。
    internal_k = sparsity
    if 3 * sparsity > m:
        # print(f"Warning: M={m} is too small for CoSaMP with K={sparsity}. Reducing internal K to {m//3}.")
        internal_k = max(1, m // 3 - 1) 

    for it in range(max_iter):
        
        # 1. Proxy step (利用先验矩阵 Psi 进行匹配)
        # Psi.T @ residual 本质上是估计 residual 对应的信号 x_res
        proxy = np.abs(Psi.T @ residual)

        # [修正] 删除了 proxy[support] = 0.0
        # CoSaMP 需要看到全局的相关性

        # 2. Identification (选 2K)
        # 注意：这里选的是 internal_k (为了防止矩阵欠定)
        omega = np.argsort(proxy)[-2*internal_k:]

        # 3. Merge
        T = np.union1d(support, omega)
        Psi_T = Psi.T
        
        # [双重保险] 再次确保 T 的大小不超过 M (防止 lstsq 欠定)
        if len(T) >= m:
            # 如果候选集太大，只保留 proxy 值最大的 m-1 个
            # 这是一个并不完美的补丁，但能防止程序崩溃
            # 更好的做法是重新计算 T 中元素的 proxy 值并排序
            current_proxy = np.abs(Psi_T @ residual) # 或者用 Phi.T @ y 近似
            T = T[np.argsort(current_proxy[T])][-(m-1):]

        # 4. Estimation (Least Squares)
        Phi_T = Phi[:, T]
        # 使用 lstsq 求解系数
        b_T, _, _, _ = lstsq(Phi_T, y, rcond=None)

        # 5. Prune (保留 K)
        # 注意：最终输出还是保留用户请求的 sparsity
        # 但中间过程受限于 internal_k
        keep_k = min(sparsity, len(b_T))
        idx = np.argsort(np.abs(b_T))[-keep_k:]
        support = T[idx]
        
        # 6. Update
        x_hat = np.zeros(n)
        x_hat[support] = b_T[idx]
        
        new_residual = y - Phi @ x_hat
        
        # 7. Stopping
        if norm(new_residual) < tol or norm(new_residual - residual) < 1e-6:
            break
            
        residual = new_residual

    return x_hat