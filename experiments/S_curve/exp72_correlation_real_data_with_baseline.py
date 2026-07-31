import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct

# ==================== 导入你的算法 ====================
sys.path.append('/Users/stanzhao/Desktop/prior_cs')
from experiments.S_curve.Proposed import design_pinv_psi_fast, pro_omp_solve

# ==================== 1. 全局配置与测试点 ====================
NEW_TRIALS = 0       # 【修改】设为 3，运行一次以生成包含 Baseline 的新数据
PHI_REPEATS = 3      # 每个样本下 Phi 的重复次数
DB_FILE = "exp7_correlation_real_data_with_baseline.npy" # 【修改】换了新名字

TAU_OPT = 1.0        # 固定 tau = 1
L_LENGTH = 100       # 固定的先验学习长度

# ==================== 2. 辅助函数 ====================
def keep_top_k(x, k):
    """
    对真实的 DCT 信号进行 Top-K 硬阈值处理，
    保留绝对值最大的 K 个元素，其余置零，作为严格 K-稀疏的 Ground Truth。
    """
    x_trunc = np.zeros_like(x)
    idx = np.argsort(np.abs(x))[-k:]  # 找到绝对值最大的 K 个索引
    x_trunc[idx] = x[idx]
    return x_trunc

def calc_metrics(x_true, x_est):
    """计算 SRR 和 NRE"""
    supp_true = set(np.where(np.abs(x_true) > 1e-5)[0])
    supp_est = set(np.where(np.abs(x_est) > 1e-5)[0])
    srr = len(supp_true.intersection(supp_est)) / len(supp_true) if len(supp_true) > 0 else 1.0
    nre = np.linalg.norm(x_true - x_est) / (np.linalg.norm(x_true) + 1e-10)
    return srr, nre

# ==================== 3. 数据准备 ====================
print("加载并处理真实数据集...")
try:
    X_spatial_train = np.load("../../prior_cs/data/mniset/X_train.npy") 
    # 如果是 (样本数, 28, 28) 格式，将其拉平为 (784, 样本数)
    if X_spatial_train.ndim == 3:
        X_spatial_train = X_spatial_train.reshape(X_spatial_train.shape[0], -1).T
except FileNotFoundError:
    print("未找到真实的 X_train.npy 数据文件，生成随机平滑数据用于演示...")
    X_spatial_train = np.cumsum(np.random.randn(784, 1500), axis=0)

# 对空间域图像进行 1D-DCT 变换，得到频域稀疏表示
X_dct_full = dct(X_spatial_train, axis=0, norm='ortho')

N = X_dct_full.shape[0]  # 物理维度，如 784

# 选取前 L_LENGTH 个样本作为先验学习库 (保留真实 DCT 分布，不强制截断)
X_train_prior = X_dct_full[:, :L_LENGTH]

# 选取后面的样本作为测试集
X_test_pool = X_dct_full[:, L_LENGTH:L_LENGTH + 50]

# [动态生成测试参数] 
M_LIST = [int(0.1*N), int(0.2*N), int(0.35*N),int(0.4*N), int(0.5*N), int(0.7*N), int(0.8*N)]  
K_LIST = [int(0.02*N), int(0.05*N), int(0.07*N),int(0.1*N), int(0.15*N), int(0.2*N), int(0.25*N)] 

# ==================== 4. 主逻辑 ====================
if os.path.exists(DB_FILE):
    db = np.load(DB_FILE, allow_pickle=True).item()
    print(f"成功加载历史数据库: {DB_FILE}")
else:
    db = {}
    for m in M_LIST:
        for k in K_LIST:
            # 【新增】加入 baseline 的记录键值
            db[f"M={m}_K={k}"] = {'srr_sum': 0.0, 'nre_sum': 0.0, 'baseline_srr_sum': 0.0, 'baseline_nre_sum': 0.0, 'count': 0}
    print("未找到历史数据，创建新数据库。")

if NEW_TRIALS > 0:
    for trial in range(NEW_TRIALS):
        print(f"--- Running Trial {trial+1}/{NEW_TRIALS} ---")
        
        # 每次 trial 随机挑一个测试图像
        test_img_raw = X_test_pool[:, np.random.randint(X_test_pool.shape[1])]
        
        for m in M_LIST:
            for k in K_LIST:
                key = f"M={m}_K={k}"
                
                # 对真实的 DCT 信号进行 Top-K 硬阈值处理，生成 Ground Truth
                x_true = keep_top_k(test_img_raw, k)
                
                temp_srr, temp_nre = 0.0, 0.0
                temp_base_srr, temp_base_nre = 0.0, 0.0  # 【新增】
                
                for _ in range(PHI_REPEATS):
                    Phi = np.random.randn(m, N)
                    Phi /= np.linalg.norm(Phi, axis=0)
                    y = Phi @ x_true
                    
                    # 1. Proposed Method
                    Psi = design_pinv_psi_fast(Phi, X_train_prior, tau=TAU_OPT)
                    x_est = pro_omp_solve(Phi, Psi, y, sparsity=k)
                    srr, nre = calc_metrics(x_true, x_est)
                    temp_srr += srr / PHI_REPEATS
                    temp_nre += nre / PHI_REPEATS
                    
                    # 2. Baseline Method (Phi = Psi) 【新增】
                    x_base = pro_omp_solve(Phi, Phi, y, sparsity=k)
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

# ==================== 5. 画图 ====================
plt.rcParams.update({
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'font.size': 14
})

fig, ax = plt.subplots(figsize=(11, 6))

srr_vals, nre_vals = [], []
srr_base_vals, nre_base_vals = [], []
labels = []

for m in M_LIST:
    for k in K_LIST:
        key = f"M={m}_K={k}"
        if db[key]['count'] > 0:
            # Proposed
            avg_srr = db[key]['srr_sum'] / db[key]['count']
            avg_nre = db[key]['nre_sum'] / db[key]['count']
            
            # Baseline 【新增】
            avg_base_srr = db[key]['baseline_srr_sum'] / db[key]['count']
            avg_base_nre = db[key]['baseline_nre_sum'] / db[key]['count']
            
            # 【截断】低于 0.1 的误差全部统一截断为 0.1 方便线性展示
            if avg_nre < 1e-1: avg_nre = 1e-1
            if avg_base_nre < 1e-1: avg_base_nre = 1e-1
            
            srr_vals.append(avg_srr)
            nre_vals.append(avg_nre)
            srr_base_vals.append(avg_base_srr)
            nre_base_vals.append(avg_base_nre)
            
            # labels.append(f"M={m/N:.2g}N\nK={k/N:.2g}N")

# 绘制散点图
# 1. Baseline 散点图 (灰色)
ax.scatter(srr_base_vals, nre_base_vals, c='gray', s=120, alpha=0.6, edgecolors='black', linewidth=1.5, zorder=2, label="Baseline",marker='s')

# 2. Proposed 散点图 (深蓝色)
ax.scatter(srr_vals, nre_vals, c='#1f77b4', s=120, alpha=0.75, edgecolors='black', linewidth=1.5, zorder=3, label="Proposed")

# 细节配置
ax.set_xlabel("Support Recovery Rate (SRR)")
ax.set_ylabel("Normalized Reconstruction Error (NRE)")

ax.grid(True, linestyle=':', alpha=0.6, zorder=0)

# X 轴和 Y 轴范围动态适应
min_srr = min(min(srr_vals), min(srr_base_vals))
ax.set_xlim(min_srr - 0.08, 1.08)
ax.set_ylim(0, 5)  

# 【新增】图例
ax.legend(loc='upper right', framealpha=0.9)

plt.tight_layout()
plt.show()