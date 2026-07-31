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
style_map = {
    "Gaussian":               {"color": "gray",   "ls": "--", "marker": "o"},
    "Modified-CS":            {"color": "black",  "ls": "-.", "marker": "s"}, 
    "Bo Li":                  {"color": "green",  "ls": "-.", "marker": "d"},
    "Huang":                  {"color": "cyan",   "ls": "-.", "marker": "x"},
    "Proposed":               {"color": "blue",   "ls": "-",  "marker": "*"},
    "Proposed+Spatial+Decay": {"color": "red",    "ls": "-",  "marker": "^"}
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
        frames_array = (frames_array - frames_array.min()) / (frames_array.max() - frames_array.min())
        return frames_array
    except:
        print("[Warning] Using synthetic data")
        return np.random.rand(100, 6, 6, 32, 32).astype(np.float32)

# ==========================================
# 1. 算法类
# ==========================================
class BoLiMatrixDesign:
    def __init__(self, n, m, tau=0.5):
        self.n, self.m, self.tau = n, m, tau
    def optimize(self, x_prior_avg):
        x_abs = np.abs(x_prior_avg).flatten()
        max_val = np.max(np.sqrt(x_abs)) + 1e-10
        w_diag = self.tau + (1 - self.tau) * (np.sqrt(x_abs) / max_val)
        idx = np.argsort(w_diag)[::-1]
        P_opt = np.zeros((self.m, self.n))
        for i in range(self.m):
            P_opt[i, idx[i]] = 1.0
        return P_opt

def design_huang_phi(Phi_init, x_prior_avg):
    m, n = Phi_init.shape
    Cov = np.outer(x_prior_avg, x_prior_avg) + 1e-6 * np.eye(n)
    U, _, _ = svd(Cov)
    Phi_new = U[:, :m].T
    return normalize_columns(Phi_new)

# ==========================================
# 2. Babel 函数计算 (核心修改)
# ==========================================
def calculate_babel_function(D, max_k):
    """
    计算 Babel 函数 mu_1(K)
    Definition: mu_1(k) = max_{i} sum_{j \in TopK} |<d_i, d_j>|
    即：字典 D 的列向量之间的最大累积相干性
    """
    # 1. 归一化列 (Self-Normalization)
    D = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-10)
    
    # 2. 计算 Gram 矩阵 G = D^T * D (自相关)
    G = np.abs(D.T @ D)
    np.fill_diagonal(G, 0) # 对角线置零
    
    # 3. 排序
    G_sorted = -np.sort(-G, axis=0) # 降序排列
    
    # 4. 累积求和
    cum_coh = np.cumsum(G_sorted, axis=0)
    
    # 5. 取最坏情况 (Max over all columns)
    babel = np.max(cum_coh, axis=1)
    
    # 6. 返回前 max_k 个
    # 注意：babel[0] 对应 K=1 (最大的1个非对角元素)
    return babel[:max_k]

# ==========================================
# 3. 主程序
# ==========================================
def run_babel_analysis():
    # 参数设置
    CR = 0.2
    MAX_K = 5        # K 取 1 到 10
    HISTORY_LEN = 10  # 10帧先验
    DECAY_FACTOR = 0.8
    NUM_SAMPLES = 50  # 动态方法采样数
    
    frames_array = load_video_data()
    num_frames, Ny, Nx, P, _ = frames_array.shape
    n = P ** 2
    m = int(n * CR)
    
    TARGET_FRAME_IDX = HISTORY_LEN 
    if TARGET_FRAME_IDX >= num_frames: TARGET_FRAME_IDX = num_frames - 1
    
    print(f"Babel Analysis: CR={CR}, Max K={MAX_K}")
    print(f"Target Frame: {TARGET_FRAME_IDX}, History: {HISTORY_LEN} frames")
    
    # --- 1. 静态矩阵 ---
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
    
    # 计算静态字典
    D_gauss = Phi_base
    
    boli_solver = BoLiMatrixDesign(n, m)
    D_boli = boli_solver.optimize(x_prior_avg)
    
    D_huang = design_huang_phi(Phi_base, x_prior_avg)
    
    # 计算静态 Babel (K=1..10)
    babel_gauss = calculate_babel_function(D_gauss, MAX_K)
    babel_boli = calculate_babel_function(D_boli, MAX_K)
    babel_huang = calculate_babel_function(D_huang, MAX_K)
    
    # --- 2. 动态字典 Babel (Proposed) ---
    print("Calculating Babel for Proposed methods...")
    babel_prop_sum = np.zeros(MAX_K)
    babel_decay_sum = np.zeros(MAX_K)
    
    rng = np.random.default_rng(42)
    rand_r = rng.integers(0, Ny, NUM_SAMPLES)
    rand_c = rng.integers(0, Nx, NUM_SAMPLES)
    
    for i in tqdm(range(NUM_SAMPLES)):
        r, c = rand_r[i], rand_c[i]
        
        true_patch = frames_array[TARGET_FRAME_IDX, r, c]
        x_curr = dct2(true_patch).flatten()
        y = Phi_base @ x_curr
        x_coarse = (pinv(Phi_base) @ y).reshape(-1, 1)
        
        prev_patch = frames_array[TARGET_FRAME_IDX-1, r, c]
        x_prev = dct2(prev_patch).flatten().reshape(-1, 1)
        
        # (A) Proposed
        X_prior_prop = np.hstack([x_prev, x_coarse])
        Psi_prop = design_pinv_psi_fast(Phi_base, X_prior_prop)
        # [关键] 使用 Psi^T * Psi
        babel_prop_sum += calculate_babel_function(Psi_prop, MAX_K)
        
        # (B) Proposed+Spatial+Decay
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
        # [关键] 使用 Psi^T * Psi
        babel_decay_sum += calculate_babel_function(Psi_decay, MAX_K)
        
    babel_prop = babel_prop_sum / NUM_SAMPLES
    babel_decay = babel_decay_sum / NUM_SAMPLES
    
    # --- 3. 绘图 ---
    plt.figure(figsize=(9, 6))
    k_axis = np.arange(1, MAX_K + 1)
    
    # Gaussian & Modified-CS (物理矩阵相同，重合)
    plt.plot(k_axis, babel_gauss, **style_map["Gaussian"], label="Gaussian / Modified-CS", markevery=1)
    
    plt.plot(k_axis, babel_boli, **style_map["Bo Li"], label="Bo Li", markevery=1)
    plt.plot(k_axis, babel_huang, **style_map["Huang"], label="Huang", markevery=1)
    plt.plot(k_axis, babel_prop, **style_map["Proposed"], label="Proposed", markevery=1)
    plt.plot(k_axis, babel_decay, **style_map["Proposed+Spatial+Decay"], label="Proposed+Spatial+Decay", markevery=1)
    
    plt.title(f'Babel Function (Cumulative Coherence)\nCR={CR}, Prior={HISTORY_LEN} Frames', fontsize=14)
    plt.xlabel(r'Sparsity Level $K$', fontsize=12)
    plt.ylabel(r'Cumulative Coherence $\mu_1(K)$', fontsize=12)
    plt.xticks(k_axis) # 强制显示整数刻度
    plt.xlim(1, MAX_K)
    plt.ylim(0, 5)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig("babel_function_K1_10.png", dpi=300)
    plt.show()
    print("[Done] Babel Function plot saved.")

if __name__ == "__main__":
    run_babel_analysis()