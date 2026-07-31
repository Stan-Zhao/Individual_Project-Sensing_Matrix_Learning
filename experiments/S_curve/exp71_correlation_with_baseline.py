import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ==================== 导入你的算法 ====================
sys.path.append('/Users/stanzhao/Desktop/prior_cs')
# 【新增】导入 pro_omp_solve 用于运行 Baseline
from experiments.S_curve.Proposed import pinv_omp, pro_omp_solve

# ==================== 1. 全局配置与测试点 ====================
NEW_TRIALS = 0       # 【修改】设为5，因为你需要跑一次生成 Baseline 的数据
PHI_REPEATS = 3      # 每个测试点下 Phi 的重复次数
DB_FILE = "exp7_correlation_with_baseline.npy" # 【修改】换了新名字，避免污染之前的旧数据

# 模型维度配置
N = 256
TAU_OPT = 1.0        # 固定 tau = 1
L_LENGTH = 100       # 固定的先验学习长度

# [极限拓展] 选取跨度极大的 M 和 K 组合
M_LIST = [16, 32, 64, 100, 128, 192]
K_LIST = [5, 20, 45, 60, 75, 100]

# ==================== 2. 辅助函数 ====================
def generate_structured_power_law(N, K, alpha=1.0):
    """生成幂律衰减 + Top-k 截断的结构化信号"""
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
# 提前学习的训练集 (长度固定为 100)
X_train = generate_train_data(N, L_LENGTH, max(K_LIST) + 5, alpha=1.0)

# 尝试加载历史数据
if os.path.exists(DB_FILE):
    db = np.load(DB_FILE, allow_pickle=True).item()
    print(f"成功加载历史数据库: {DB_FILE}")
else:
    db = {}
    for m in M_LIST:
        for k in K_LIST:
            # 【新增】加入 baseline 的数据记录键值
            db[f"M={m}_K={k}"] = {'srr_sum': 0.0, 'nre_sum': 0.0, 'baseline_srr_sum': 0.0, 'baseline_nre_sum': 0.0, 'count': 0}
    print("未找到历史数据，创建新数据库。")

if NEW_TRIALS > 0:
    for trial in range(NEW_TRIALS):
        print(f"--- Running Trial {trial+1}/{NEW_TRIALS} ---")
        
        for m in M_LIST:
            for k in K_LIST:
                key = f"M={m}_K={k}"
                x_true = generate_structured_power_law(N, k, alpha=1.0)
                
                temp_srr, temp_nre = 0.0, 0.0
                temp_base_srr, temp_base_nre = 0.0, 0.0 # 【新增】
                
                for _ in range(PHI_REPEATS):
                    Phi = np.random.randn(m, N)
                    Phi /= np.linalg.norm(Phi, axis=0)
                    y = Phi @ x_true
                    
                    # 1. Proposed Method
                    x_est = pinv_omp(Phi, y, X_train, k, TAU_OPT)
                    srr, nre = calc_metrics(x_true, x_est)
                    temp_srr += srr / PHI_REPEATS
                    temp_nre += nre / PHI_REPEATS
                    
                    # 2. Baseline Method (Phi = Psi) 【新增】
                    x_base = pro_omp_solve(Phi, Phi, y, k)
                    srr_b, nre_b = calc_metrics(x_true, x_base)
                    temp_base_srr += srr_b / PHI_REPEATS
                    temp_base_nre += nre_b / PHI_REPEATS
                    
                # 累加到总库
                db[key]['srr_sum'] += temp_srr
                db[key]['nre_sum'] += temp_nre
                db[key]['baseline_srr_sum'] += temp_base_srr
                db[key]['baseline_nre_sum'] += temp_base_nre
                db[key]['count'] += 1
            
    np.save(DB_FILE, db)
    print(f"测试完成，数据已累加保存至 {DB_FILE}。")

# ==================== 4. 画图 (散点图与相关性) ====================
plt.rcParams.update({
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'font.size': 14
})

fig, ax = plt.subplots(figsize=(11, 6))

srr_vals, nre_vals = [], []
srr_base_vals, nre_base_vals = [], []

for m in M_LIST:
    for k in K_LIST:
        key = f"M={m}_K={k}"
        if db[key]['count'] > 0:
            # Proposed 指标
            avg_srr = db[key]['srr_sum'] / db[key]['count']
            avg_nre = db[key]['nre_sum'] / db[key]['count']
            
            # Baseline 指标 【新增】
            avg_base_srr = db[key]['baseline_srr_sum'] / db[key]['count']
            avg_base_nre = db[key]['baseline_nre_sum'] / db[key]['count']
            
            # 【截断逻辑】将 NRE 小于 1e-1 的值强制截断为 1e-1
            if avg_nre < 1e-1: avg_nre = 1e-1
            if avg_base_nre < 1e-1: avg_base_nre = 1e-1
            
            srr_vals.append(avg_srr)
            nre_vals.append(avg_nre)
            
            srr_base_vals.append(avg_base_srr)
            nre_base_vals.append(avg_base_nre)

# 绘制散点图
# 1. Baseline 散点图 (灰色)
ax.scatter(srr_base_vals, nre_base_vals, c='gray', s=120, alpha=0.6, edgecolors='black', linewidth=1.5, zorder=2, label="Baseline",marker='s')

# 2. Proposed 散点图 (深蓝色)
ax.scatter(srr_vals, nre_vals, c='#1f77b4', s=120, alpha=0.75, edgecolors='black', linewidth=1.5, zorder=3, label="Proposed")


ax.set_xlabel("Support Recovery Rate (SRR)")
ax.set_ylabel("Normalized Reconstruction Error (NRE)")

# 保持你取消了 log 的设定
# ax.set_yscale('log')
ax.grid(True, linestyle=':', alpha=0.6, zorder=0)

# 动态调整 X 轴范围，找到 Baseline 和 Proposed 中最小的 SRR
min_srr = min(min(srr_vals), min(srr_base_vals))
ax.set_xlim(min_srr - 0.08, 1.08)

# 【新增】添加图例
ax.legend(loc='upper right', framealpha=0.9)

# 紧凑布局
plt.tight_layout()
plt.show()