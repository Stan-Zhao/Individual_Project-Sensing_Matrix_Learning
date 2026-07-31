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
# 2. 加权 Babel 函数计算 (核心修改)
# ==========================================
def calculate_weighted_babel(D, x_true, max_k):
    """
    计算加权 Babel 函数 (Weighted Cumulative Coherence)
    考虑实际信号幅度的干扰：Interference = Sum(|G_ij| * |x_j|)
    """
    # 1. 归一化列 (D的列必须单位化，否则相关性计算不对)
    D = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-10)
    
    # 2. 计算 Gram 矩阵 G = |D^T * D|
    G = np.abs(D.T @ D)
    np.fill_diagonal(G, 0) # 排除自相关
    
    # 3. 获取真实信号幅度并排序
    x_abs = np.abs(x_true).flatten()
    # 找到幅度最大的索引 (Top K components)
    sorted_indices = np.argsort(x_abs)[::-1]
    
    results = []
    
    # 4. 遍历 K 计算加权干扰
    for k in range(1, max_k + 1):
        # 取前 K 个最大的支撑集
        support_idx = sorted_indices[:k]
        # 对应的权重 (信号幅度)
        weights = x_abs[support_idx]
        
        # 计算这 K 个原子对所有其他原子产生的总干扰
        # 向量化计算: G[:, support] 是一个 (N, K) 的矩阵
        # 乘以权重 weights (K,) -> 得到每个原子受到的总干扰 (N,)
        total_interference = G[:, support_idx] @ weights
        
        # Babel 定义为最坏情况 (Max interference experienced by any atom)
        # 实际上通常关心非支撑集受到的干扰，但标准定义是 max over all i
        max_interf = np.max(total_interference)
        
        results.append(max_interf)
        
    return np.array(results)

# ==========================================
# 3. 主程序
# ==========================================
def run_weighted_babel_analysis():
    # 参数设置
    CR = 0.1
    MAX_K = 10        # K 取 1 到 10
    HISTORY_LEN = 10  
    DECAY_FACTOR = 0.8
    NUM_SAMPLES = 50  # 采样数
    
    frames_array = load_video_data()
    num_frames, Ny, Nx, P, _ = frames_array.shape
    n = P ** 2
    m = int(n * CR)
    
    TARGET_FRAME_IDX = HISTORY_LEN 
    if TARGET_FRAME_IDX >= num_frames: TARGET_FRAME_IDX = num_frames - 1
    
    print(f"Weighted Babel Analysis: CR={CR}, Max K={MAX_K}")
    
    # --- 1. 静态矩阵生成 ---
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
    
    boli_solver = BoLiMatrixDesign(n, m)
    D_boli = boli_solver.optimize(x_prior_avg)
    
    D_huang = design_huang_phi(Phi_base, x_prior_avg)
    
    # 结果累加器
    accum_results = {
        "Gaussian": np.zeros(MAX_K),
        "Bo Li": np.zeros(MAX_K),
        "Huang": np.zeros(MAX_K),
        "Proposed": np.zeros(MAX_K),
        "Proposed+Spatial+Decay": np.zeros(MAX_K)
    }
    
    # --- 2. 循环采样计算 (所有方法都在循环内计算加权值) ---
    print(f"Calculating Weighted Babel over {NUM_SAMPLES} samples...")
    
    rng = np.random.default_rng(42)
    rand_r = rng.integers(0, Ny, NUM_SAMPLES)
    rand_c = rng.integers(0, Nx, NUM_SAMPLES)
    
    for i in tqdm(range(NUM_SAMPLES)):
        r, c = rand_r[i], rand_c[i]
        
        # 1. 获取当前 Patch 的真实系数 x
        true_patch = frames_array[TARGET_FRAME_IDX, r, c]
        x_curr = dct2(true_patch).flatten()
        
        # 2. 静态方法的加权 Babel (因为 x 变了，所以要在这里算)
        accum_results["Gaussian"] += calculate_weighted_babel(D_gauss, x_curr, MAX_K)
        accum_results["Bo Li"] += calculate_weighted_babel(D_boli, x_curr, MAX_K)
        accum_results["Huang"] += calculate_weighted_babel(D_huang, x_curr, MAX_K)
        
        # 3. 动态方法准备
        y = Phi_base @ x_curr
        x_coarse = (pinv(Phi_base) @ y).reshape(-1, 1)
        prev_patch = frames_array[TARGET_FRAME_IDX-1, r, c]
        x_prev = dct2(prev_patch).flatten().reshape(-1, 1)
        
        # (A) Proposed
        X_prior_prop = np.hstack([x_prev, x_coarse])
        Psi_prop = design_pinv_psi_fast(Phi_base, X_prior_prop)
        accum_results["Proposed"] += calculate_weighted_babel(Psi_prop, x_curr, MAX_K)
        
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
        accum_results["Proposed+Spatial+Decay"] += calculate_weighted_babel(Psi_decay, x_curr, MAX_K)
        
    # 取平均
    for k in accum_results:
        accum_results[k] /= NUM_SAMPLES
    
    # --- 3. 绘图 ---
    plt.figure(figsize=(9, 6))
    k_axis = np.arange(1, MAX_K + 1)
    
    # 绘制曲线
    plt.plot(k_axis, accum_results["Gaussian"], **style_map["Gaussian"], label="Gaussian / Modified-CS", markevery=1)
    plt.plot(k_axis, accum_results["Bo Li"], **style_map["Bo Li"], label="Bo Li", markevery=1)
    plt.plot(k_axis, accum_results["Huang"], **style_map["Huang"], label="Huang", markevery=1)
    plt.plot(k_axis, accum_results["Proposed"], **style_map["Proposed"], label="Proposed", markevery=1)
    plt.plot(k_axis, accum_results["Proposed+Spatial+Decay"], **style_map["Proposed+Spatial+Decay"], label="Proposed+Spatial+Decay", markevery=1)
    
    plt.title(f'Weighted Babel Function (Signal-Aware Interference)\nCR={CR}, Prior={HISTORY_LEN} Frames', fontsize=14)
    plt.xlabel(r'Sparsity Level $K$', fontsize=12)
    
    plt.ylabel(r'Weighted Cumulative Coherence $\mu_{w}(K)$', fontsize=12)
    plt.xticks(k_axis)
    plt.xlim(1, MAX_K)
    plt.ylim(bottom=0)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig("weighted_babel_function.png", dpi=300)
    plt.show()
    print("[Done] Weighted Babel plot saved.")

if __name__ == "__main__":
    run_weighted_babel_analysis()