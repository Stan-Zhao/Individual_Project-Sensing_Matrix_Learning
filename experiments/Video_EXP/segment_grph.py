import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from scipy.linalg import svd, norm, pinv
from tqdm import tqdm
import warnings

# ==========================================
# 0. 基础设置
# ==========================================
from prior_cs.utils.normalize import normalize_columns
from prior_cs.algorithms.omp import omp
from prior_cs.algorithms.pinv_psi_omp import pro_omp_solve
from prior_cs.algorithms.psi_fast import design_pinv_psi_fast

np.random.seed(42)
warnings.filterwarnings('ignore')

# ==========================================
# 1. 算法辅助函数
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

def modified_omp(Phi, y, T_support, k=30):
    m, n = Phi.shape
    residual = y.copy()
    support = []
    if T_support is not None and len(T_support) > 0:
        valid_support = [idx for idx in T_support if idx < n]
        support = list(set(valid_support))
        if len(support) > 0:
            Phi_S = Phi[:, support]
            x_S = np.linalg.pinv(Phi_S) @ y
            residual = y - Phi_S @ x_S
    for _ in range(k - len(support)):
        if np.linalg.norm(residual) < 1e-6: break
        projections = np.abs(Phi.T @ residual)
        projections[support] = 0
        best_idx = np.argmax(projections)
        support.append(best_idx)
        Phi_S = Phi[:, support]
        x_S = np.linalg.pinv(Phi_S) @ y
        residual = y - Phi_S @ x_S
    x_rec = np.zeros(n)
    if len(support) > 0:
        Phi_S = Phi[:, support]
        x_S = np.linalg.pinv(Phi_S) @ y
        x_rec[support] = x_S
    return x_rec

def get_support(x, k):
    return np.argsort(np.abs(x))[::-1][:k]

def dct2(block):
    return dct(dct(block.T, norm="ortho").T, norm="ortho")

def idct2(coeff):
    return idct(idct(coeff.T, norm="ortho").T, norm="ortho")

def psnr(x_true, x_rec):
    mse = np.mean((x_true - x_rec) ** 2)
    if mse < 1e-12: return 100.0
    return 10 * np.log10(1.0 / mse)

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

# ==========================================
# 2. 数据加载
# ==========================================
try:
    frames_array = np.load("VIDEO/video_patches.npy")
except:
    print("[Info] Generating synthetic data...")
    frames_array = np.random.rand(200, 6, 6, 32, 32).astype(np.float32)

if frames_array.shape[0] < 185:
    padding = np.tile(frames_array[-1:], (185 - frames_array.shape[0] + 1, 1, 1, 1, 1))
    frames_array = np.concatenate([frames_array, padding], axis=0)

frames_array = (frames_array - frames_array.min()) / (frames_array.max() - frames_array.min())
num_frames, Ny, Nx, P, _ = frames_array.shape
n = P ** 2

# 计算全图分辨率
H_full = Ny * P
W_full = Nx * P

# ==========================================
# 3. 实验配置
# ==========================================
compression_ratio = 0.2
m = int(compression_ratio * n)
sparsity_k = 64
HISTORY_LEN = 5     
DECAY_FACTOR = 0.8  
WARMUP_LEN = 10

methods = [
    "Baseline", 
    "Modified-CS", 
    "Bo Li", 
    "Schnass", 
    "Proposed", 
    "Proposed+Spatial+Decay"
]

segments = [(0, 1)]
segment_results = []

# ==========================================
# 4. 分段运行函数 (改为全图 PSNR 计算)
# ==========================================
def run_segment_simulation(start_idx, end_idx):
    
    actual_start = max(0, start_idx - WARMUP_LEN)
    print(f"Processing Segment {start_idx}-{end_idx} (Warming up from {actual_start})...")
    
    prev_rec_dct_grid = {met: [[np.zeros(n) for _ in range(Nx)] for _ in range(Ny)] for met in methods}
    history_buffer_decay = [[[] for _ in range(Nx)] for _ in range(Ny)]
    
    Phi_random = np.random.randn(m, n)
    Phi_random = normalize_columns(Phi_random)
    
    prior_coeffs = []
    for t in range(actual_start, start_idx + 1):
        for r in range(Ny):
            for c in range(Nx):
                prior_coeffs.append(dct2(frames_array[t, r, c]).flatten())
    
    if len(prior_coeffs) > 0:
        x_prior_avg = np.mean(np.abs(np.array(prior_coeffs)), axis=0)
    else:
        x_prior_avg = np.zeros(n)

    if start_idx == 0:
        Phi_boli = Phi_random
        Phi_huang = Phi_random
    else:
        boli_solver = BoLiMatrixDesign(n, m, tau=0.5)
        Phi_boli = boli_solver.optimize(x_prior_avg)
        Phi_huang = design_huang_phi(Phi_random, x_prior_avg)

    # 存储最终结果的字典
    psnr_data = {met: [] for met in methods} 

    for t in tqdm(range(actual_start, end_idx), leave=False):
        
        # 1. 在每一帧开始时，初始化各算法的全图画布
        full_true_frame = np.zeros((H_full, W_full))
        full_rec_frames = {met: np.zeros((H_full, W_full)) for met in methods}
        
        for r in range(Ny):
            for c in range(Nx):
                true_patch = frames_array[t, r, c]
                x_true = dct2(true_patch).flatten()
                
                # 计算当前块在画布上的绝对坐标
                row_start, row_end = r * P, (r + 1) * P
                col_start, col_end = c * P, (c + 1) * P
                
                # 填入 Ground Truth 画布
                full_true_frame[row_start:row_end, col_start:col_end] = true_patch
                
                # --- A. Gaussian ---
                y_gauss = Phi_random @ x_true
                coef_gauss = omp(Phi_random, y_gauss, k=sparsity_k)
                
                # --- B. Modified-CS ---
                x_prev_mod = prev_rec_dct_grid["Modified-CS"][r][c]
                T_support = get_support(x_prev_mod, k=sparsity_k)
                coef_mod = modified_omp(Phi_random, y_gauss, T_support, k=sparsity_k)
                prev_rec_dct_grid["Modified-CS"][r][c] = coef_mod

                # --- C. Bo Li ---
                y_boli = Phi_boli @ x_true
                coef_boli = omp(Phi_boli, y_boli, k=sparsity_k)
                
                # --- D. Huang ---
                y_huang = Phi_huang @ x_true
                coef_huang = omp(Phi_huang, y_huang, k=sparsity_k)

                # Common Coarse
                coef_coarse = coef_gauss 
                x_coarse = coef_coarse.reshape(-1, 1)
                
                # --- E. Proposed ---
                x_prev_prop = prev_rec_dct_grid["Proposed"][r][c].reshape(-1, 1)
                X_prior = np.hstack([x_prev_prop, x_coarse])
                Psi_prop = design_pinv_psi_fast(Phi_random, X_prior)
                coef_prop = pro_omp_solve(Phi_random, Psi_prop, y_gauss, sparsity=sparsity_k)
                prev_rec_dct_grid["Proposed"][r][c] = coef_prop

                # --- F. Proposed+Spatial+Decay ---
                buf = history_buffer_decay[r][c]
                priors_decay = []
                x_neighbors = get_spatial_neighbors(prev_rec_dct_grid["Proposed+Spatial+Decay"], r, c, Ny, Nx)
                priors_decay.append(x_neighbors)
                if len(buf) > 0:
                    for i, vec in enumerate(buf):
                        w = DECAY_FACTOR ** i
                        priors_decay.append(vec.reshape(-1, 1) * np.sqrt(w))
                else:
                    priors_decay.append(np.zeros((n, 1)))
                priors_decay.append(x_coarse)
                
                X_prior_decay = np.hstack(priors_decay)
                Psi_prop_d = design_pinv_psi_fast(Phi_random, X_prior_decay)
                coef_prop_d = pro_omp_solve(Phi_random, Psi_prop_d, y_gauss, sparsity=sparsity_k)
                
                prev_rec_dct_grid["Proposed+Spatial+Decay"][r][c] = coef_prop_d
                history_buffer_decay[r][c].insert(0, coef_prop_d)
                if len(history_buffer_decay[r][c]) > HISTORY_LEN: history_buffer_decay[r][c].pop()

                # 2. 如果在评估范围内，将空间域重构块填入画布
                if t >= start_idx:
                    full_rec_frames["Baseline"][row_start:row_end, col_start:col_end] = idct2(coef_gauss.reshape(P, P))
                    full_rec_frames["Modified-CS"][row_start:row_end, col_start:col_end] = idct2(coef_mod.reshape(P, P))
                    full_rec_frames["Bo Li"][row_start:row_end, col_start:col_end] = idct2(coef_boli.reshape(P, P))
                    full_rec_frames["Schnass"][row_start:row_end, col_start:col_end] = idct2(coef_huang.reshape(P, P))
                    full_rec_frames["Proposed"][row_start:row_end, col_start:col_end] = idct2(coef_prop.reshape(P, P))
                    full_rec_frames["Proposed+Spatial+Decay"][row_start:row_end, col_start:col_end] = idct2(coef_prop_d.reshape(P, P))

        # 3. 帧循环结束，在全图级别统一计算 PSNR
        if t >= start_idx:
            for met in methods:
                psnr_data[met].append(psnr(full_true_frame, full_rec_frames[met]))
                
    return psnr_data

# ==========================================
# 5. 执行实验
# ==========================================
for (s, e) in segments:
    res = run_segment_simulation(s, e)
    segment_results.append(res)

# ==========================================
# 6. 分段绘图
# ==========================================
style_map = {
    "Baseline":               {"color": "gray",   "ls": "--", "marker": "o"},
    "Modified-CS":            {"color": "black",  "ls": "-.", "marker": "s"},
    "Bo Li":                  {"color": "green",  "ls": "-.", "marker": "d"},
    "Schnass":                  {"color": "cyan",   "ls": "-.", "marker": "x"},
    "Proposed":               {"color": "blue",   "ls": "-",  "marker": "*"},
    "Proposed+Spatial+Decay": {"color": "red",    "ls": "-",  "marker": "^"}
}

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(18, 6))
plt.subplots_adjust(wspace=0.05) 

axes = [ax1, ax2, ax3]

for i, (start, end) in enumerate(segments):
    ax = axes[i]
    res = segment_results[i]
    x_indices = np.arange(start, end)
    
    for met in methods:
        data = res[met]
        s = style_map[met]
        lbl = met if i == 0 else "_nolegend_"
        
        ax.plot(x_indices, data, 
                color=s["color"], linestyle=s["ls"], marker=s["marker"], 
                label=lbl, linewidth=2, markersize=6, alpha=0.8)
    
    ax.set_xlim(start, end - 1)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.text(0.5, 0.95, f"Frames {start}-{end}", transform=ax.transAxes, 
            ha='center', va='top', fontsize=14, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    if i == 0:
        ax.set_ylabel("Full-Frame PSNR (dB)", fontsize=14)
    
    d = .015 
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    if i > 0: 
        ax.spines['left'].set_visible(False)
        ax.tick_params(labelleft=False, left=False)
        ax.plot((-d, +d), (-d, +d), **kwargs)
        ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    if i < 2: 
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelright=False, right=False)
        ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

fig.suptitle(f"Full-Frame PSNR Comparison at Key Stages (CR={compression_ratio})", fontsize=16, y=0.98)
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, 0.02), fontsize=14)
plt.subplots_adjust(bottom=0.15)

plt.show()