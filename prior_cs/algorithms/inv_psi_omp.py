import numpy as np

def design_inv_psi(Phi, X_all):
    XXT = X_all @ X_all.T
    A = Phi @ XXT @ Phi.T
    A_inv = np.linalg.inv(A + 0 * np.eye(A.shape[0]))
    Psi = A_inv @ Phi @ XXT
    return Psi

def pro_omp_solve(Phi, Psi, y, sparsity):
    M, N = Phi.shape
    residual = y.copy()
    support = []

    # 预先分配空间，避免循环中重复分配
    x_est = np.zeros(N)

    for _ in range(sparsity):
        # 1. 计算相关性
        correlations = np.abs(Psi.T @ residual)
        
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

def inv_omp(Phi, X_all, y, sparsity):   
    Psi = design_inv_psi(Phi, X_all)
    x_recovered = pro_omp_solve(Phi, Psi, y, sparsity)
    return x_recovered