import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# ==================== 导入你的算法 ====================
sys.path.append('/Users/stanzhao/Desktop/prior_cs')
from experiments.S_curve.Proposed import pinv_omp, pro_omp_solve
from experiments.S_curve.schnass import get_schnass_sensing_dictionary_pocs

# ==================== 1. 全局配置 ====================
NEW_TRIALS = 0
DB_FILE = "exp6_time_metrics_structured_results.npy"

N, M, K = 256, 32, 20
METHODS = ["Baseline", "Schnass", "Proposed"]
TAU_OPT = 1 

# ==================== 2. 辅助函数 ====================
def generate_structured_power_law(N, K, alpha=1.0):
    x = np.zeros(N)
    vals = (np.arange(1, K + 1) ** (-alpha))
    vals *= np.sign(np.random.randn(K))
    x[:K] = vals
    return x

def generate_train_data(N, samples, k_max, alpha=1.0):
    X = np.zeros((N, samples))
    for i in range(samples):
        k = np.random.randint(5, k_max + 1)
        X[:, i] = generate_structured_power_law(N, k, alpha)
    return X

def calc_metrics(x_true, x_est):
    supp_true = set(np.where(np.abs(x_true) > 1e-5)[0])
    supp_est = set(np.where(np.abs(x_est) > 1e-5)[0])
    srr = len(supp_true.intersection(supp_est)) / len(supp_true) if len(supp_true) > 0 else 1.0
    nre = np.linalg.norm(x_true - x_est) / (np.linalg.norm(x_true) + 1e-10)
    return srr, nre

def format_val(val):
    """两位小数，0显示为0"""
    return "0" if abs(val) < 1e-6 else f"{val:.2f}"

# ==================== 3. 数据加载 ====================
X_train = generate_train_data(N, 80, K + 10, alpha=1.0)

if os.path.exists(DB_FILE):
    db = np.load(DB_FILE, allow_pickle=True).item()
    print(f"成功加载历史数据库: {DB_FILE}")
else:
    db = {m: {'time_sum': 0.0, 'srr_sum': 0.0, 'nre_sum': 0.0, 'count': 0} for m in METHODS}
    print("未找到历史数据，创建新数据库。")

# ==================== 4. 绘图 ====================
plt.rcParams.update({
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'font.size': 14
})

fig, ax1 = plt.subplots(figsize=(10, 6))

# --- 数据 ---
times = [db[m]['time_sum'] / db[m]['count'] * 1000 for m in METHODS]  # 转毫秒
srrs  = [db[m]['srr_sum'] / db[m]['count'] for m in METHODS]
nres  = [db[m]['nre_sum'] / db[m]['count'] for m in METHODS]

# --- 左轴：时间 ---
colors_bar = ['#7f7f7f', '#2ca02c', '#d62728']
bars = ax1.bar(METHODS, times, color=colors_bar, alpha=0.6,
               width=0.5, edgecolor='black', linewidth=1)

ax1.set_ylabel("Execution Time (ms)")
ax1.grid(axis='y', linestyle=':', alpha=0.6)
ax1.set_ylim(0, max(times) * 1.25)

# 时间标注（毫秒）
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2,
             yval + (max(times)*0.02),
             f"{yval:.2f} ms",
             ha='center', va='bottom',
             fontweight='bold')

# --- 右轴 ---
ax2 = ax1.twinx()
x_pos = np.arange(len(METHODS))

line1, = ax2.plot(x_pos, srrs,
                  color='#ff7f0e', marker='o',
                  markersize=8, linewidth=2.5,
                  label='SRR')

line2, = ax2.plot(x_pos, nres,
                  color='#1f77b4', marker='s',
                  markersize=8, linewidth=2.5,
                  linestyle='--', label='NRE')

ax2.set_ylabel("Metrics Value")
ax2.set_ylim(0, 1.15)

# --- SRR 标注（上方） ---
for i, val in enumerate(srrs):
    ax2.text(x_pos[i]-0.015,
             val + 0.023,
             format_val(val),
             color='#ff7f0e',
             ha='center',
             va='bottom',
             fontweight='bold')

# --- NRE 标注（改为上方） ---
for i, val in enumerate(nres):
    ax2.text(x_pos[i]-0.015,
             val + 0.023,
             format_val(val),
             color='#1f77b4',
             ha='center',
             va='bottom',
             fontweight='bold')

# 图例 → 左上角
ax2.legend(loc='upper left', framealpha=0.9)

plt.tight_layout()
plt.show()