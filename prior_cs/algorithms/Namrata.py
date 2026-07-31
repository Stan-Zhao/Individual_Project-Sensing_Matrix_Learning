# modified_cs.py
import numpy as np
import cvxpy as cp

def solve_modified_cs(y, A, T):
    """
    Modified-CS 算法核心逻辑
    参考文献: Namrata Vaswani and Wei Lu, "Modified-CS: Modifying Compressive Sensing..."
    
    参数:
        y: 观测向量 (m,)
        A: 感知矩阵 (m, n)
        T: 已知支撑集的索引列表 (list or set)
    返回:
        x_rec: 重构的稀疏信号
    """
    m, n = A.shape
    
    # 定义优化变量
    x = cp.Variable(n)
    
    # 1. 构建目标函数: min ||x_{T^c}||_1
    # 创建掩码，只有不在 T 中的索引为 1 (需要最小化)，在 T 中的索引为 0 (不惩罚)
    mask = np.ones(n)
    if T is not None and len(T) > 0:
        mask[list(T)] = 0
    
    # cvxpy 支持 element-wise multiply
    # 目标：最小化 T 补集上的 L1 范数
    objective = cp.Minimize(cp.norm(cp.multiply(mask, x), 1))
    
    # 2. 约束条件: y = Ax
    constraints = [A @ x == y]
    
    # 3. 求解
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    return x.value

# # --- 测试示例 ---
# if __name__ == "__main__":
#     # 模拟数据
#     N = 100
#     M = 50
#     K = 10
#     A = np.random.randn(M, N)
#     x_true = np.zeros(N)
    
#     # 假设真实支撑集
#     true_support = np.random.choice(N, K, replace=False)
#     x_true[true_support] = np.random.randn(K)
#     y = A @ x_true
    
#     # 假设先验 T 包含了 70% 的真实支撑集 (模拟部分已知)
#     known_k = int(0.7 * K)
#     T_prior = true_support[:known_k]
    
#     # 运行 Modified-CS
#     x_rec = solve_modified_cs(y, A, T_prior)
#     print(f"重构误差: {np.linalg.norm(x_rec - x_true):.4f}")