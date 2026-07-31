import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ==================== 导入你的算法 ====================
sys.path.append('/Users/stanzhao/Desktop/prior_cs')
from experiments.S_curve.Proposed import pinv_omp, pro_omp_solve
from experiments.S_curve.schnass import get_schnass_sensing_dictionary_pocs

# ==================== 1. 全局配置与测试点 ====================
NEW_TRIALS = 0       # 本次额外运行的 Trial 次数 (0 则直接读数据画图)
PHI_REPEATS = 3      # 每个测试点下 Phi 的重复次数
DB_FILE = "exp2_sparsity_structured_results.npy"

# 模型维度配置
N, M = 256, 64
# [核心自变量]: 稀疏度测试范围 (可方便修改)
X_TEST_POINTS = [5, 10, 15, 20, 25, 30, 35,40, 45, 50, 55, 60, 65, 70, 75]
# [对比方法]: 
METHODS = ["Baseline", "Schnass", "Proposed"]
# 选定一个性能较好的 tau 供 Proposed 方法使用
TAU_OPT = 1

# ==================== 2. 辅助函数 ====================
def generate_structured_power_law(N, K, alpha=1.0):
    """
    生成幂律衰减 + Top-k 截断的结构化信号。
    位置不随机，永远集中在最前面的 K 个元素。
    """
    x = np.zeros(N)
    vals = (np.arange(1, K + 1) ** (-alpha))
    vals *= np.sign(np.random.randn(K)) # 随机符号
    x[:K] = vals
    return x

def generate_train_data(N, samples, k_max, alpha=1.0):
    """为先验算法生成训练集"""
    X = np.zeros((N, samples))
    for i in range(samples):
        k = np.random.randint(5, k_max + 1)
        X[:, i] = generate_structured_power_law(N, k, alpha)
    return X

def calc_metrics(x_true, x_est):
    """计算 SRR 和 NRE"""
    supp_true = set(np.where(np.abs(x_true) > 1e-5)[0])
    supp_est = set(np.where(np.abs(x_est) > 1e-5)[0])
    srr = len(supp_true.intersection(supp_est)) / len(supp_true) if len(supp_true) > 0 else 1.0
    nre = np.linalg.norm(x_true - x_est) / (np.linalg.norm(x_true) + 1e-10)
    return srr, nre

# ==================== 3. 主逻辑 ====================
# 提前学习的训练集
X_train = generate_train_data(N, 500, max(X_TEST_POINTS) + 5, alpha=1.0)

# 尝试加载历史数据
if os.path.exists(DB_FILE):
    db = np.load(DB_FILE, allow_pickle=True).item()
    print(f"成功加载历史数据库: {DB_FILE}")
else:
    db = {m: {'srr_sum': np.zeros(len(X_TEST_POINTS)), 'nre_sum': np.zeros(len(X_TEST_POINTS)), 'count': 0} for m in METHODS}
    print("未找到历史数据，创建新数据库。")

if NEW_TRIALS > 0:
    for trial in range(NEW_TRIALS):
        print(f"--- Running Trial {trial+1}/{NEW_TRIALS} ---")
        temp_srr = {m: np.zeros(len(X_TEST_POINTS)) for m in METHODS}
        temp_nre = {m: np.zeros(len(X_TEST_POINTS)) for m in METHODS}
        
        for i, k in enumerate(X_TEST_POINTS):
            # 严格使用结构化数据作为本次测试真值
            x_true = generate_structured_power_law(N, k, alpha=1.0)
            
            for _ in range(PHI_REPEATS):
                Phi = np.random.randn(M, N)
                Phi /= np.linalg.norm(Phi, axis=0)
                y = Phi @ x_true
                
                # 1. Gaussian (Standard OMP)
                x_g = pro_omp_solve(Phi, Phi, y, k)
                
                # 2. Schnass (POCS Sensing Dictionary)
                Psi_s = get_schnass_sensing_dictionary_pocs(Phi, max_iter=20)
                x_s = pro_omp_solve(Phi, Psi_s, y, k)
                
                # 3. Proposed (Pinv OMP with Prior)
                x_p = pinv_omp(Phi, y, X_train, k, TAU_OPT)
                
                # 统计计算
                for m_name, x_est in zip(METHODS, [x_g, x_s, x_p]):
                    srr, nre = calc_metrics(x_true, x_est)
                    temp_srr[m_name][i] += srr / PHI_REPEATS
                    temp_nre[m_name][i] += nre / PHI_REPEATS
                    
        # 将本次 Trial 均值累加到总库
        for m in METHODS:
            db[m]['srr_sum'] += temp_srr[m]
            db[m]['nre_sum'] += temp_nre[m]
            db[m]['count'] += 1
            
    np.save(DB_FILE, db)
    print(f"测试完成，数据已累加保存至 {DB_FILE}。")

# ==================== 4. 画图 (严格按照用户标准) ====================
plt.rcParams.update({
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'font.size': 14
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

markers = ['o', 's', '^', 'D', 'v', '*']
linestyles = ['-', '--', '-.', ':', '-', '--']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for idx, m in enumerate(METHODS):
    if db[m]['count'] > 0:
        avg_srr = db[m]['srr_sum'] / db[m]['count']
        avg_nre = db[m]['nre_sum'] / db[m]['count']

        label_name = m

        ax1.plot(X_TEST_POINTS, avg_srr,
                 marker=markers[idx % len(markers)], linestyle=linestyles[idx % len(linestyles)],
                 color=colors[idx % len(colors)], label=label_name,
                 linewidth=2, markersize=7, alpha=0.7)

        ax2.plot(X_TEST_POINTS, avg_nre,
                 marker=markers[idx % len(markers)], linestyle=linestyles[idx % len(linestyles)],
                 color=colors[idx % len(colors)], label=label_name,
                 linewidth=2, markersize=7, alpha=0.7)

# 图 1 细节配置
ax1.set_xlabel("Sparsity")
ax1.set_ylabel("Support Recovery Rate (SRR)")
ax1.set_xlim(min(X_TEST_POINTS), max(X_TEST_POINTS))
ax1.set_xticks(X_TEST_POINTS)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='lower left')

# 图 2 细节配置
ax2.set_xlabel("Sparsity")
ax2.set_ylabel("Normalized Recon Error (NRE)")
ax2.set_xlim(min(X_TEST_POINTS), max(X_TEST_POINTS))
ax2.set_xticks(X_TEST_POINTS)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper left')

plt.tight_layout()
plt.show()