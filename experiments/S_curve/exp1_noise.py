import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ==================== 导入你的算法 ====================
sys.path.append('/Users/stanzhao/Desktop/prior_cs')
from experiments.S_curve.Proposed import pinv_omp

# ==================== 1. 全局配置与测试点 ====================
NEW_TRIALS = 0      # 本次额外运行的 Trial 次数 (0 则直接读数据画图)
PHI_REPEATS = 3      # 每个测试点下 Phi 的重复次数
DB_FILE = "exp1_tau_structured_noisy_results.npy"  # 换了个新名字，避免和无噪数据混淆

# 模型维度配置
N, M = 256, 64
# [核心自变量]: 稀疏度测试范围 (可方便修改)
X_TEST_POINTS = [40,45, 50, 55, 60, 65, 70, 75]
# [对比变量]: 不同的 tau 值 (对应样式列表长度，最多 6 个)
TAU_LIST = [100, 1, 1e-2, 1e-4]

# [新增参数]: 测量噪声的信噪比 (dB)
SNR_DB = 30 

# ==================== 2. 辅助函数 ====================
def format_tau(tau):
    """将浮点数转换为 LaTeX 上标格式，如 1e-4 转换为 10^{-4}"""
    if tau == 0: return "0"
    exp = int(np.floor(np.log10(tau)))
    return f"10^{{{exp}}}"

def generate_structured_power_law(N, K, alpha=1.0):
    """
    生成幂律衰减 + Top-k 截断的结构化信号。
    位置不随机，永远集中在最前面的 K 个元素。
    """
    x = np.zeros(N)
    # 幂律衰减: x[i] = (i+1)^(-alpha)
    vals = (np.arange(1, K + 1) ** (-alpha))
    # 加上随机符号，增加数据多样性但不破坏幅度结构
    vals *= np.sign(np.random.randn(K))
    x[:K] = vals
    return x

def generate_train_data(N, samples, k_max, alpha=1.0):
    """为先验算法生成训练集 (结构同上)"""
    X = np.zeros((N, samples))
    for i in range(samples):
        # 训练集包含多种可能的稀疏度
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
# 提前学习的训练集 (最大稀疏度基于测试点最大值)
X_train = generate_train_data(N, 500, max(X_TEST_POINTS) + 5, alpha=1.0)

# 尝试加载历史数据
if os.path.exists(DB_FILE):
    db = np.load(DB_FILE, allow_pickle=True).item()
    print(f"成功加载历史数据库: {DB_FILE}")
else:
    db = {format_tau(tau): {'srr_sum': np.zeros(len(X_TEST_POINTS)), 'nre_sum': np.zeros(len(X_TEST_POINTS)), 'count': 0} for tau in TAU_LIST}
    print("未找到历史数据，创建新数据库。")

if NEW_TRIALS > 0:
    for trial in range(NEW_TRIALS):
        print(f"--- Running Trial {trial+1}/{NEW_TRIALS} ---")
        temp_srr = {format_tau(tau): np.zeros(len(X_TEST_POINTS)) for tau in TAU_LIST}
        temp_nre = {format_tau(tau): np.zeros(len(X_TEST_POINTS)) for tau in TAU_LIST}
        
        for i, k in enumerate(X_TEST_POINTS):
            # 严格使用结构化数据作为本次测试真值
            x_true = generate_structured_power_law(N, k, alpha=1.0)
            
            for _ in range(PHI_REPEATS):
                Phi = np.random.randn(M, N)
                Phi /= np.linalg.norm(Phi, axis=0)
                
                # 计算纯净的观测值
                y_clean = Phi @ x_true
                
                # --- 【核心新增：加入高斯白噪声】 ---
                P_signal = np.var(y_clean)
                P_noise = P_signal / (10 ** (SNR_DB / 10.0))
                noise = np.random.randn(M) * np.sqrt(P_noise)
                y_noisy = y_clean + noise
                # -----------------------------------
                
                for tau in TAU_LIST:
                    key = format_tau(tau)
                    # 运行你的算法 (注意这里传入的是带噪的 y_noisy)
                    x_est = pinv_omp(Phi, y_noisy, X_train, k, tau)
                    srr, nre = calc_metrics(x_true, x_est)
                    temp_srr[key][i] += srr / PHI_REPEATS
                    temp_nre[key][i] += nre / PHI_REPEATS
                    
        # 将本次 Trial 均值累加到总库
        for tau in TAU_LIST:
            key = format_tau(tau)
            db[key]['srr_sum'] += temp_srr[key]
            db[key]['nre_sum'] += temp_nre[key]
            db[key]['count'] += 1
            
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

for idx, tau in enumerate(TAU_LIST):
    key = format_tau(tau)
    if db[key]['count'] > 0:
        avg_srr = db[key]['srr_sum'] / db[key]['count']
        avg_nre = db[key]['nre_sum'] / db[key]['count']

        label_name = f"$\\tau={key}$"

        ax1.plot(X_TEST_POINTS, avg_srr,
                 marker=markers[idx % len(markers)], linestyle=linestyles[idx % len(linestyles)],
                 color=colors[idx % len(colors)], label=label_name,
                 linewidth=2, markersize=7, alpha=0.7)

        ax2.plot(X_TEST_POINTS, avg_nre,
                 marker=markers[idx % len(markers)], linestyle=linestyles[idx % len(linestyles)],
                 color=colors[idx % len(colors)], label=label_name,
                 linewidth=2, markersize=7, alpha=0.7)

# 图 1 细节配置 (标题加上 SNR 信息)
ax1.set_xlabel("Sparsity")
ax1.set_ylabel("Support Recovery Rate (SRR)")
ax1.set_xlim(min(X_TEST_POINTS), max(X_TEST_POINTS))
ax1.set_xticks(X_TEST_POINTS)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='lower left')

# 图 2 细节配置 (标题加上 SNR 信息)
ax2.set_xlabel("Sparsity")
ax2.set_ylabel("Normalized Recon Error (NRE)")
ax2.set_xlim(min(X_TEST_POINTS), max(X_TEST_POINTS))
ax2.set_xticks(X_TEST_POINTS)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper left')

plt.tight_layout()
plt.show()