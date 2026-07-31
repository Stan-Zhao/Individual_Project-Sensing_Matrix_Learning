import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import pinv
from scipy.fftpack import dct
import os
import warnings
from mpl_toolkits.axes_grid1 import ImageGrid

# --- 引用你的自定义库 ---
from prior_cs.algorithms.psi_fast import design_pinv_psi_fast

# 忽略除了我们正在修复的 SyntaxWarning 之外的其他警告
warnings.filterwarnings('ignore')
np.random.seed(42)

# ==========================================
# 0. 基础工具 & 指标计算
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

def get_off_diag_mean(G):
    """计算非对角线元素的均值（评估平均干扰强度）"""
    N = G.shape[0]
    mask = ~np.eye(N, dtype=bool)
    return np.mean(G[mask])

# ==========================================
# 1. 数据加载
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
# 2. 核心实验：矩阵生成与量化计算
# ==========================================
def run_gram_matrix_analysis():
    # --- 设置 ---
    CR = 0.2          
    X_spatial = load_or_generate_data()
    N, L_total = X_spatial.shape
    X_full, D_dct = apply_dct_transform(X_spatial)
    
    M = int(N * CR)
    L_train = int(L_total * 0.8)
    X_train = X_full[:, :L_train]
    
    print(f"Generating matrices for CR={CR}, M={M}, N={N}...")

    # 1. 生成物理测量矩阵 Phi
    Phi = np.random.randn(M, N)
    Phi = Phi / np.linalg.norm(Phi, axis=0, keepdims=True)

    # 2. 计算三种 Psi
    print("Computing Psi matrices...")
    Psi_base = Phi.copy()
    
    Psi_paper = get_psi_paper_pocs(Phi, max_iter=50)
    
    Psi_prop = design_pinv_psi_fast(Phi, X_train) 
    Psi_prop = Psi_prop / (np.linalg.norm(Psi_prop, axis=0, keepdims=True) + 1e-10)

    # 3. 计算 Gram 矩阵的绝对值 (|Psi^T * Phi|)
    print("Computing Pseudo-Gram matrices and metrics...")
    AbsG_base = np.abs(Psi_base.T @ Phi)
    AbsG_paper = np.abs(Psi_paper.T @ Phi)
    AbsG_prop = np.abs(Psi_prop.T @ Phi)

    # 4. 计算量化指标（平均干扰强度）
    mean_base = get_off_diag_mean(AbsG_base)
    mean_paper = get_off_diag_mean(AbsG_paper)
    mean_prop = get_off_diag_mean(AbsG_prop)

    print("-" * 40)
    print(f"Avg Interference - Baseline: {mean_base:.4f}")
    print(f"Avg Interference - Schnass:  {mean_paper:.4f}")
    print(f"Avg Interference - Proposed: {mean_prop:.4f}")
    print("-" * 40)

    # ==========================================
    # 3. 绘图 (热力图)
    # ==========================================
    print("Plotting matrices...")
    
    # 全局学术字体加粗设置
    plt.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold',
        'axes.titleweight': 'bold',
        'axes.titlesize': 15
    })

    fig = plt.figure(figsize=(16, 6))

    # 使用 ImageGrid 完美排版
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(1, 3),
                     axes_pad=0.4,           # 给多行标题留出空间
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="5%",
                     cbar_pad=0.15)

    # N=256，CR=0.2，截断显示范围以获得最佳对比度
    VMAX = 0.25 

    # (a) Baseline
    im = grid[0].imshow(AbsG_base, cmap='inferno', vmin=0, vmax=VMAX)
    grid[0].set_title(r"Baseline: $|\Phi^T \Phi|$" + f"\nAvg Interference: {mean_base:.4f}")

    # (b) Paper (POCS)
    grid[1].imshow(AbsG_paper, cmap='inferno', vmin=0, vmax=VMAX)
    grid[1].set_title(r"Schnass: $|\Psi_{schnass}^T \Phi|$" + f"\nAvg Interference: {mean_paper:.4f}")

    # (c) Proposed
    grid[2].imshow(AbsG_prop, cmap='inferno', vmin=0, vmax=VMAX)
    grid[2].set_title(r"Proposed: $|\Psi_{prop}^T \Phi|$" + f"\nAvg Interference: {mean_prop:.4f}")

    # 隐藏坐标轴刻度
    for ax in grid:
        ax.set_xticks([])
        ax.set_yticks([])

    # 添加共享的 Colorbar
    cbar = grid.cbar_axes[0].colorbar(im)
    cbar.ax.set_title("Coherence", fontsize=12, pad=12)

    save_path = "sensing_matrices_1d_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"[Done] Image saved to {save_path}")

if __name__ == "__main__":
    run_gram_matrix_analysis()