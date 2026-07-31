import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct
from scipy.linalg import pinv, svd, norm
from tqdm import tqdm
import os
import warnings

# --- 引入库 ---
from prior_cs.algorithms.psi_fast import design_pinv_psi_fast
from prior_cs.utils.normalize import normalize_columns

warnings.filterwarnings('ignore')

# ==========================================
# 0. 基础配置
# ==========================================
# 定义要绘制的方法 (保留3种)
methods_config = {
    "Gaussian":               {"color": "gray",   "title": "Gaussian (Random Matrix)"},
    "Huang":                  {"color": "cyan",   "title": "Huang (Optimized Phi)"},
    "Proposed+Spatial+Decay": {"color": "red",    "title": "Proposed + Spatial + Decay"}
}

def dct2(block):
    return dct(dct(block.T, norm="ortho").T, norm="ortho")

def get_spatial_neighbors(grid_buffer, r, c, Ny, Nx):
    neighbors = []
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
    center_vec = grid_buffer[r][c]
    for dr, dc in offsets:
        nr, nc = r + dr, c + dc
        if 0 <= nr < Ny and 0 <= nc < Nx:
            neighbors.append(grid_buffer[nr][nc].reshape(-1, 1))
        else:
            neighbors.append(center_vec.reshape(-1, 1))
    return np.hstack(neighbors)

def load_video_data():
    try:
        path = "VIDEO/video_patches.npy"
        frames_array = np.load(path)
        # 归一化
        frames_array = (frames_array - frames_array.min()) / (frames_array.max() - frames_array.min())
        return frames_array
    except:
        print("[Warning] Using synthetic data")
        return np.random.rand(100, 6, 6, 32, 32).astype(np.float32)

# ==========================================
# 1. 算法类 (Huang Only)
# ==========================================
def design_huang_phi(Phi_init, x_prior_avg):
    m, n = Phi_init.shape
    Cov = np.outer(x_prior_avg, x_prior_avg) + 1e-6 * np.eye(n)
    U, _, _ = svd(Cov)
    Phi_new = U[:, :m].T
    return normalize_columns(Phi_new)

# ==========================================
# 2. 核心计算：提取互相关性
# ==========================================
def extract_cross_coherence(D, support_gamma):
    """
    计算非支撑集原子与支撑集原子之间的相关性
    """
    N = D.shape[1]
    # 归一化
    D = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-10)
    # 计算 Gram 矩阵 G = |D^T @ D|
    G = np.abs(D.T @ D)
    
    # 确定非支撑集
    all_indices = np.arange(N)
    support_set = set(support_gamma)
    non_support_indices = np.array([i for i in all_indices if i not in support_set])
    
    if len(non_support_indices) == 0:
        return np.array([])
    
    # 提取子矩阵 (Rows: Non-Support, Cols: Support)
    sub_matrix = G[np.ix_(non_support_indices, support_gamma)]
    
    return sub_matrix.flatten()

# ==========================================
# 3. 主程序
# ==========================================
def run_interference_3subplots():
    # 参数设置
    CR = 0.1
    K = 30            
    HISTORY_LEN = 10 
    DECAY_FACTOR = 0.8
    NUM_SAMPLES = 200 # 采样 Patch 数量
    
    frames_array = load_video_data()
    num_frames, Ny, Nx, P, _ = frames_array.shape
    n = P ** 2
    m = int(n * CR)
    
    TARGET_FRAME_IDX = HISTORY_LEN 
    if TARGET_FRAME_IDX >= num_frames: TARGET_FRAME_IDX = num_frames - 1
    
    print(f"Interference Histogram (3 Methods): CR={CR}, K={K}")
    
    # --- 1. 静态矩阵准备 ---
    Phi_base = np.random.randn(m, n)
    Phi_base = normalize_columns(Phi_base)
    
    # 帧级先验
    history_frames = frames_array[TARGET_FRAME_IDX-HISTORY_LEN : TARGET_FRAME_IDX]
    history_coeffs = []
    for f in range(len(history_frames)):
        for r in range(Ny):
            for c in range(Nx):
                history_coeffs.append(dct2(history_frames[f, r, c]).flatten())
    x_prior_avg = np.mean(np.abs(np.array(history_coeffs)), axis=0)
    
    # 静态字典
    D_gauss = Phi_base
    D_huang = design_huang_phi(Phi_base, x_prior_avg)
    
    # 收集数据容器
    vals = {k: [] for k in methods_config.keys()}
    
    # --- 2. 采样循环 ---
    print(f"Collecting data from {NUM_SAMPLES} patches...")
    rng = np.random.default_rng(42)
    rand_r = rng.integers(0, Ny, NUM_SAMPLES)
    rand_c = rng.integers(0, Nx, NUM_SAMPLES)
    
    for i in tqdm(range(NUM_SAMPLES)):
        r, c = rand_r[i], rand_c[i]
        
        # 真实信号与支撑集
        true_patch = frames_array[TARGET_FRAME_IDX, r, c]
        x_curr = dct2(true_patch).flatten()
        
        # 获取 Top-K 支撑集 Gamma
        gamma = np.argsort(np.abs(x_curr))[::-1][:K]
        
        # 1. 静态方法
        vals["Gaussian"].extend(extract_cross_coherence(D_gauss, gamma))
        vals["Huang"].extend(extract_cross_coherence(D_huang, gamma))
        
        # 2. 动态方法准备
        y = Phi_base @ x_curr
        x_coarse = (pinv(Phi_base) @ y).reshape(-1, 1)
        
        # 3. Proposed+Spatial+Decay
        priors_decay = []
        grid_buffer = [[None for _ in range(Nx)] for _ in range(Ny)]
        for rr in range(Ny):
            for cc in range(Nx):
                grid_buffer[rr][cc] = dct2(frames_array[TARGET_FRAME_IDX-1, rr, cc]).flatten()
        x_neighbors = get_spatial_neighbors(grid_buffer, r, c, Ny, Nx)
        priors_decay.append(x_neighbors)
        
        for h in range(HISTORY_LEN):
            f_idx = TARGET_FRAME_IDX - 1 - h
            if f_idx >= 0:
                vec = dct2(frames_array[f_idx, r, c]).flatten()
                w = DECAY_FACTOR ** h
                priors_decay.append(vec.reshape(-1, 1) * np.sqrt(w))
        priors_decay.append(x_coarse)
        
        X_prior_decay = np.hstack(priors_decay)
        Psi_decay = design_pinv_psi_fast(Phi_base, X_prior_decay)
        vals["Proposed+Spatial+Decay"].extend(extract_cross_coherence(Psi_decay, gamma))

    # --- 3. 绘图 (1x3 Subplots) ---
    print("Plotting histogram...")
    # 这里使用 1 行 3 列的布局
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    axes = axes.flatten()
    
    bins = 100
    x_range = (0, 0.8) # 关注 0-0.8 区间
    
    for i, (method_name, config) in enumerate(methods_config.items()):
        ax = axes[i]
        data = vals[method_name]
        
        # 绘制直方图
        ax.hist(data, bins=bins, range=x_range, density=True, 
                color=config["color"], alpha=0.7, label=method_name)
        
        # 计算统计量
        mean_val = np.mean(data)
        
        # 设置标题和样式
        ax.set_title(f"{config['title']}\nMean Coherence: {mean_val:.4f}", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # 轴标签
        ax.set_xlabel(r'Coherence Value $|G_{ij}| (i \in \Gamma^c, j \in \Gamma)$', fontsize=11)
        if i == 0:
            ax.set_ylabel('Probability Density', fontsize=11)
            
        # 添加一条均值线 (虚线)
        ax.axvline(mean_val, color='black', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.3f}')
        ax.legend(loc='upper right', fontsize=9)

    plt.suptitle(f'Interference Correlation Histogram (Separated)\nCR={CR}, K={K}, Prior={HISTORY_LEN} Frames', fontsize=15, y=0.98)
    plt.tight_layout()
    plt.savefig("interference_histogram_3methods.png", dpi=300)
    plt.show()
    print("[Done] Histogram saved.")

if __name__ == "__main__":
    run_interference_3subplots()