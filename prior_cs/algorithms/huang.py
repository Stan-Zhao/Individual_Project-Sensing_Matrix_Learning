# design_dictionary_huang.py
import numpy as np

def update_weights_huang(x_estimated, epsilon=1e-6):
    """
    根据估计的信号更新权重
    参考文献: Anmin Huang et al., "A re-weighted algorithm..."
    逻辑: 信号越强，对应的原子越重要，给予特定权重处理
    """
    # 类似于 Reweighted L1，权重通常是幅度的倒数或相关函数
    # 在 Huang 的论文中，利用 posteriori knowledge revising weights
    weights = 1.0 / (np.abs(x_estimated) + epsilon)
    return np.diag(weights)

def design_sensing_dictionary(Phi_init, Y, max_iter=10):
    """
    Data Dependent Sensing Dictionary Design
    参数:
        Phi_init: 初始感知矩阵
        Y: 观测数据 (m, L) (MMV 情况) 或 (m, 1)
    """
    Phi = Phi_init.copy()
    
    for k in range(max_iter):
        # 1. 稀疏编码 (Sparse Coding)
        # 使用当前 Phi 恢复 X (可用 OMP 或 Lasso)
        # 这里简化假设 X 已知或通过伪逆估计
        X_est = np.linalg.pinv(Phi) @ Y 
        
        # 2. 更新权重 (Revising Weights)
        # 对每一列信号（如果是 MMV）或单信号计算权重
        # Huang 的核心是利用观测数据的后验
        W = update_weights_huang(np.mean(X_est, axis=1))
        
        # 3. 更新感知字典 (Update Dictionary)
        # 目标是优化 Phi 使得加权后的相干性最小
        # 这是一个简化的梯度下降或 SVD 步骤
        # Phi_new = min || Y - Phi * X ||_W ...
        
        # 在 Huang 的论文中，通常涉及对 (X X^T) 的加权分解
        # 此处展示核心思想：Phi 向着能更好表示 Y 的方向更新
        term = Y @ X_est.T @ np.linalg.inv(X_est @ X_est.T + 1e-6 * np.eye(X_est.shape[0]))
        Phi = term
        
        # 归一化列
        norms = np.linalg.norm(Phi, axis=0)
        Phi = Phi / (norms + 1e-10)
        
    return Phi