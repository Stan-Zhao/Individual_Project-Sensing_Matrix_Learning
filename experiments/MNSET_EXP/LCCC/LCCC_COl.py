import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import pinv
from scipy.fftpack import dct
import os
import warnings

# --- 引用你的自定义库 ---
from prior_cs.algorithms.psi_fast import design_pinv_psi_fast

# 忽略除了我们正在修复的 SyntaxWarning 之外的其他警告
warnings.filterwarnings('ignore')

# ==========================================
# 0. 基础工具
# ==========================================

def dct_matrix(N):
    D = np.zeros((N, N))
    for k in range(N):
        e = np.zeros(N)
        e[k] = 1.0
        D[:, k] = dct(e, norm='ortho')
    return D

def apply_dct_transform(X):
    N = X.shape[0]
    D = dct_matrix(N)
    return D @ X, D

# ==========================================
# 1. POCS (Paper Method)
# ==========================================
def compute_mu_bound(M, N):
    if M >= N: return 0
    return np.sqrt((N - M) / (M * (N - 1)))

def get_psi_paper_pocs(Phi, max_iter=50, tol=1e-6):
    M, N = Phi.shape
    mu = compute_mu_bound(M, N)
    Phi_pinv = pinv(Phi)
    P_G = Phi_pinv @ Phi
    G = Phi.T @ Phi

    for _ in range(max_iter):
        G_prev = G.copy()
        H = G.copy()
        np.fill_diagonal(H, 1.0)
        mask = ~np.eye(N, dtype=bool)
        H[mask] = np.clip(H[mask], -mu, mu)
        G = H @ P_G
        if np.linalg.norm(G - G_prev, 'fro') < tol: break

    Psi_T = H @ Phi_pinv
    Psi = Psi_T.T
    Psi = Psi / (np.linalg.norm(Psi, axis=0, keepdims=True) + 1e-10)
    return Psi

# ==========================================
# 2. 数据加载
# ==========================================
def load_or_generate_data():
    path = "../../prior_cs/data/mniset/X_train.npy"
    if os.path.exists(path):
        print(f"[Info] Loading data from {path}")
        X = np.load(path)
        if X.shape[0] > X.shape[1]: X = X.T
        X = X / (np.max(np.abs(X)) + 1e-8)
        return X
    else:
        print("[Warning] Generating synthetic data...")
        N, L = 256, 2000
        idx = np.arange(N)
        dist = np.abs(idx[:, None] - idx[None, :])
        Cov = 0.8 ** dist
        rng = np.random.default_rng(42)
        X = rng.multivariate_normal(np.zeros(N), Cov, L).T
        X[np.abs(X) < 0.1] = 0
        return X

# ==========================================
# 3. 核心实验：直方图统计
# ==========================================

def run_histogram_analysis():
    # --- 设置 ---
    CR = 0.2          # 极低压缩率
    K = 30            # 固定稀疏度
    NUM_SAMPLES = 200 # [关键] 增加采样数，相当于重复实验 200 次并叠加数据
    
    # 1. 准备数据
    X_spatial = load_or_generate_data()
    N, L_total = X_spatial.shape
    X_full, D_dct = apply_dct_transform(X_spatial)
    
    M = int(N * CR)
    L_train = int(L_total * 0.8)
    X_train = X_full[:, :L_train]
    X_test = X_full[:, L_train:]
    
    print(f"Generating matrices for CR={CR}, M={M}, N={N}...")

    # 2. 生成矩阵 (只生成一次，用于统计)
    Phi = np.random.randn(M, N)
    Phi = Phi / np.linalg.norm(Phi, axis=0, keepdims=True)

    # 计算三种 Psi
    print("Computing Psi matrices...")
    Psi_base = Phi.copy()
    Psi_paper = get_psi_paper_pocs(Phi, max_iter=50)
    Psi_prop = design_pinv_psi_fast(Phi, X_train) 
    Psi_prop = Psi_prop / (np.linalg.norm(Psi_prop, axis=0, keepdims=True) + 1e-10)

    # 计算 Gram 矩阵的绝对值
    AbsG_base = np.abs(Psi_base.T @ Phi)
    AbsG_paper = np.abs(Psi_paper.T @ Phi)
    AbsG_prop = np.abs(Psi_prop.T @ Phi)

    # 3. 收集互相关数据
    vals_base = []
    vals_paper = []
    vals_prop = []

    print(f"Collecting correlation data from {NUM_SAMPLES} random samples (Repeated Trials)...")
    
    # 随机选择测试样本
    test_indices = np.random.choice(X_test.shape[1], NUM_SAMPLES, replace=False)
    
    for idx in test_indices:
        x = X_test[:, idx].copy()
        x[0] = 0 # 去除 DC
        
        # 确定支撑集 Gamma
        idx_sorted = np.argsort(np.abs(x))
        Gamma = idx_sorted[-K:] # Support indices
        
        # 确定非支撑集 Gamma_c
        mask = np.ones(N, dtype=bool)
        mask[Gamma] = False
        Gamma_c = np.where(mask)[0] # Non-support indices
        
        # 提取互相关子矩阵的值
        # 我们统计的是：非支撑集原子 对 支撑集原子 的干扰
        vals_base.extend(AbsG_base[Gamma_c][:, Gamma].flatten())
        vals_paper.extend(AbsG_paper[Gamma_c][:, Gamma].flatten())
        vals_prop.extend(AbsG_prop[Gamma_c][:, Gamma].flatten())

    # 转为 numpy 数组
    vals_base = np.array(vals_base)
    vals_paper = np.array(vals_paper)
    vals_prop = np.array(vals_prop)

    print(f"Total data points collected per method: {len(vals_base)}")

    # ==========================================
    # 4. 绘图 (直方图)
    # ==========================================
    
    # 【新增】全局字体加粗、加大设置
    plt.rcParams.update({
        'font.size': 14,              # 全局基础字体大小
        'font.weight': 'bold',        # 全局字体加粗
        'axes.labelweight': 'bold',   # 坐标轴标签加粗
        'axes.titleweight': 'bold',   # 子图标题加粗
        'axes.labelsize': 14,         # 坐标轴标签大小
        'axes.titlesize': 14,         # 子图标题大小
        'xtick.labelsize': 14,        # X轴刻度数字大小
        'ytick.labelsize': 14,        # Y轴刻度数字大小
        'legend.fontsize': 14         # 图例字体大小
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True, sharex=True)
    
    bins = 100
    x_range = (0, 0.6) # 稍微调整显示范围，关注 0~0.6 即可
    
    # (a) Baseline
    axes[0].hist(vals_base, bins=bins, range=x_range, density=True, color='gray', alpha=0.7, label='Baseline')
    axes[0].set_title(f'Baseline\nMean: {np.mean(vals_base):.4f}, Max: {np.max(vals_base):.4f}')
    axes[0].set_ylabel('Probability Density')
    axes[0].set_xlabel(r'Coherence Value') 
    axes[0].grid(True, linestyle=':', alpha=0.5)

    # (b) Paper (POCS)
    axes[1].hist(vals_paper, bins=bins, range=x_range, density=True, color='blue', alpha=0.7, label='Paper (POCS)')
    axes[1].set_title(f'Schnass Method\nMean: {np.mean(vals_paper):.4f}, Max: {np.max(vals_paper):.4f}')
    axes[1].set_xlabel(r'Coherence Value') 
    axes[1].grid(True, linestyle=':', alpha=0.5)
    
    # 添加 mu 理论界线
    mu = compute_mu_bound(M, N)
    axes[1].axvline(mu, color='red', linestyle='--', linewidth=2.0, label=rf'Theory $\mu$ ({mu:.2f})') # 加粗了虚线

    # (c) Proposed
    axes[2].hist(vals_prop, bins=bins, range=x_range, density=True, color='red', alpha=0.7, label='Proposed')
    axes[2].set_title(f'Proposed Method\nMean: {np.mean(vals_prop):.4f}, Max: {np.max(vals_prop):.4f}')
    axes[2].set_xlabel(r'Coherence Value') 
    axes[2].grid(True, linestyle=':', alpha=0.5)

    
    plt.tight_layout()
    plt.show()
    
    print("[Done] Histogram generated.")

if __name__ == "__main__":
    run_histogram_analysis()