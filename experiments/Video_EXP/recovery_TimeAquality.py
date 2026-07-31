import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from scipy.linalg import svd, norm, pinv
from tqdm import tqdm
import warnings

# ==========================================
# 0. Basic Setup
# ==========================================
from prior_cs.utils.normalize import normalize_columns
from prior_cs.algorithms.omp import omp
from prior_cs.algorithms.pinv_psi_omp import pro_omp_solve
from prior_cs.algorithms.psi_fast import design_pinv_psi_fast

np.random.seed(42)
DB_FILE = "experiment_database_clean.npy"  # Changed filename to avoid conflict with old data
warnings.filterwarnings('ignore')

# ==========================================
# 1. Algorithm Classes & Functions
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
# 2. Data Loading
# ==========================================
try:
    frames_array = np.load("VIDEO/video_patches.npy")
except:
    print("[Info] Generating synthetic data...")
    frames_array = np.random.rand(200, 6, 6, 32, 32).astype(np.float32)
    for t in range(1, 200):
        # Simulate scene change every 50 frames to explain drops
        if t % 50 == 0:
            frames_array[t] = np.random.rand(*frames_array[0].shape)
        else:
            frames_array[t] = frames_array[t-1] * 0.95 + np.random.normal(0, 0.02, frames_array[0].shape)

# Slicing
START_FRAME = 0
END_FRAME = 180 
if frames_array.shape[0] < END_FRAME:
    frames_array = frames_array[-20:]
else:
    frames_array = frames_array[START_FRAME:END_FRAME]

frames_array = (frames_array - frames_array.min()) / (frames_array.max() - frames_array.min())
num_frames, Ny, Nx, P, _ = frames_array.shape
n = P ** 2

# 计算全图的分辨率
H_full = Ny * P
W_full = Nx * P

# ==========================================
# 3. Experiment Parameters
# ==========================================
compression_ratio = 0.1
m = int(compression_ratio * n)
sparsity_k = 20
HISTORY_LEN = 9     
DECAY_FACTOR = 0.8  

methods = [
    "Gaussian", 
    "Bo Li", 
    "Proposed", 
    "Proposed+Spatial+Decay"
]

experiment_key = f"CR{compression_ratio:.2f}_K{sparsity_k}_Frames{num_frames}_Clean"
print(f"------------------------------------------------")
print(f"Experiment Key: {experiment_key}")
print(f"Methods: {methods}")
print(f"------------------------------------------------")

# ==========================================
# 4. Load Database
# ==========================================
if os.path.exists(DB_FILE):
    try:
        db = np.load(DB_FILE, allow_pickle=True).item()
        print(f"[Info] Database loaded. Contains {len(db)} experiments.")
    except:
        db = {}
        print("[Info] Corrupted DB. Creating new.")
else:
    db = {}
    print("[Info] New DB created.")

if experiment_key not in db:
    db[experiment_key] = {}

for method in methods:
    if method not in db[experiment_key]:
        db[experiment_key][method] = {
            "psnr_sum": np.zeros(num_frames),
            "time_sum": 0.0,
            "trials": 0
        }

# ==========================================
# 5. Main Loop
# ==========================================
NEW_TRIALS = 1 
print(f"Running {NEW_TRIALS} new trials...")

for experiment in range(NEW_TRIALS):
    current_trial_psnr = {m: np.zeros(num_frames) for m in methods}
    current_trial_time = {m: 0.0 for m in methods}
    
    Phi_random = np.random.randn(m, n)
    Phi_random = normalize_columns(Phi_random)
    
    prev_frame_coeffs = {method: [] for method in ["Bo Li"]}
    prev_rec_dct_grid = {method: [[np.zeros(n) for _ in range(Nx)] for _ in range(Ny)] for method in methods}
    history_buffer_decay = [[[] for _ in range(Nx)] for _ in range(Ny)]
    
    boli_solver = BoLiMatrixDesign(n, m, tau=0.5)

    for t in tqdm(range(num_frames), desc=f"Trial {experiment+1}"):
        
        # Frame-level Prior (for Bo Li)
        prior_avg = {method: np.zeros(n) for method in ["Bo Li"]}
        if t > 0:
            stacked = np.array(prev_frame_coeffs["Bo Li"]) 
            prior_avg["Bo Li"] = np.mean(np.abs(stacked), axis=0)
        
        # Matrix Optimization (Bo Li)
        t0 = time.time()
        if t == 0: Phi_boli = Phi_random
        else:      Phi_boli = boli_solver.optimize(prior_avg["Bo Li"])
        current_trial_time["Bo Li"] += (time.time() - t0)
        
        prev_frame_coeffs = {method: [] for method in ["Bo Li"]}
        
        # 【修改点 1】: 准备当前帧的全图画布
        full_true_frame = np.zeros((H_full, W_full))
        full_rec_frames = {m: np.zeros((H_full, W_full)) for m in methods}

        # Patch Loop
        for r in range(Ny):
            for c in range(Nx):
                true_patch = frames_array[t, r, c]
                x_true = dct2(true_patch).flatten()
                
                # 计算当前 patch 在全图中的像素范围
                row_start, row_end = r * P, (r + 1) * P
                col_start, col_end = c * P, (c + 1) * P
                
                # 拼入 Ground Truth 画布
                full_true_frame[row_start:row_end, col_start:col_end] = true_patch
                
                # 1. Gaussian (Baseline)
                t0 = time.time()
                y_gauss = Phi_random @ x_true
                coef_gauss = omp(Phi_random, y_gauss, k=sparsity_k)
                rec_gauss = idct2(coef_gauss.reshape(P, P))
                full_rec_frames["Gaussian"][row_start:row_end, col_start:col_end] = rec_gauss # 【修改点 2】
                current_trial_time["Gaussian"] += (time.time() - t0)
                
                # 2. Bo Li
                t0 = time.time()
                y_boli = Phi_boli @ x_true
                coef_boli = omp(Phi_boli, y_boli, k=sparsity_k)
                rec_boli = idct2(coef_boli.reshape(P, P))
                full_rec_frames["Bo Li"][row_start:row_end, col_start:col_end] = rec_boli # 【修改点 2】
                prev_frame_coeffs["Bo Li"].append(coef_boli)
                current_trial_time["Bo Li"] += (time.time() - t0)
                
                # Common Coarse
                y_prop = Phi_random @ x_true
                coef_coarse = omp(Phi_random, y_prop, k=sparsity_k)
                x_coarse = coef_coarse.reshape(-1, 1)

                # 3. Proposed (Temporal Only)
                t0 = time.time()
                x_prev_prop = prev_rec_dct_grid["Proposed"][r][c].reshape(-1, 1)
                X_prior = np.hstack([x_prev_prop, x_coarse])
                Psi_prop = design_pinv_psi_fast(Phi_random, X_prior)
                coef_prop = pro_omp_solve(Phi_random, Psi_prop, y_prop, sparsity=sparsity_k)
                rec_prop = idct2(coef_prop.reshape(P, P))
                full_rec_frames["Proposed"][row_start:row_end, col_start:col_end] = rec_prop # 【修改点 2】
                prev_rec_dct_grid["Proposed"][r][c] = coef_prop
                current_trial_time["Proposed"] += (time.time() - t0)

                # 4. Proposed+Spatial+Decay
                t0 = time.time()
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
                coef_prop_d = pro_omp_solve(Phi_random, Psi_prop_d, y_prop, sparsity=sparsity_k)
                rec_prop_d = idct2(coef_prop_d.reshape(P, P))
                full_rec_frames["Proposed+Spatial+Decay"][row_start:row_end, col_start:col_end] = rec_prop_d # 【修改点 2】
                
                prev_rec_dct_grid["Proposed+Spatial+Decay"][r][c] = coef_prop_d
                history_buffer_decay[r][c].insert(0, coef_prop_d)
                if len(history_buffer_decay[r][c]) > HISTORY_LEN: history_buffer_decay[r][c].pop()
                current_trial_time["Proposed+Spatial+Decay"] += (time.time() - t0)

        # 【修改点 3】: 在整个帧完成所有块的重构后，计算整图的 PSNR
        for m in methods:
            current_trial_psnr[m][t] = psnr(full_true_frame, full_rec_frames[m])
            
    # Accumulate results for database
    for m in methods:
        db[experiment_key][m]["psnr_sum"] += current_trial_psnr[m]
        db[experiment_key][m]["time_sum"] += current_trial_time[m]
        db[experiment_key][m]["trials"] += 1
    
    np.save(DB_FILE, db)
    print(f"Trial {experiment+1} saved.")

# ==========================================
# 6. Plotting
# ==========================================
print("\n[Analysis] Loading accumulated results...")
final_stats = db[experiment_key]

avg_psnr_curves = {}
avg_times = {}
trial_counts = {}

for m in methods:
    trials = final_stats[m]["trials"]
    if trials > 0:
        avg_psnr_curves[m] = final_stats[m]["psnr_sum"] / trials
        avg_times[m] = final_stats[m]["time_sum"] / trials
        trial_counts[m] = trials
    else:
        avg_psnr_curves[m] = np.zeros(num_frames)
        avg_times[m] = 0.0
        trial_counts[m] = 0

print(f"\n=== Summary (Key: {experiment_key}) ===")
print(f"{'Method':<25} | {'Trials':<6} | {'Time (s)':<10} | {'Mean PSNR (dB)':<14}")
print("-" * 65)
for m in methods:
    mean_p = np.mean(avg_psnr_curves[m])
    print(f"{m:<25} | {trial_counts[m]:<6} | {avg_times[m]:<10.4f} | {mean_p:<14.2f}")

# 【核心修改】全局字体加粗与放大配置
plt.rcParams.update({
    'font.size': 11,              
    'font.weight': 'bold',        
    'axes.labelweight': 'bold',   
    'axes.titleweight': 'bold',   
    'axes.labelsize': 12,         
    'axes.titlesize': 13,         
    'xtick.labelsize': 11,        
    'ytick.labelsize': 11,        
    'legend.fontsize': 10         
})

plt.figure(figsize=(16, 6))

# Subplot 1: PSNR Curve
plt.subplot(1, 2, 1)
style_map = {
    "Gaussian":               {"color": "gray",   "ls": "--"},
    "Bo Li":                  {"color": "green",  "ls": "--"},
    "Proposed":               {"color": "blue",   "ls": "-"},
    "Proposed+Spatial+Decay": {"color": "red",    "ls": "-"}
}

x_axis = np.arange(START_FRAME, START_FRAME + num_frames)

for m in methods:
    if trial_counts[m] > 0:
        s = style_map[m]
        plt.plot(x_axis, avg_psnr_curves[m], 
                 color=s["color"], linestyle=s["ls"], 
                 label=f"{m}", linewidth=2, markersize=5, alpha=0.8)

plt.title(f"Average Full-Frame PSNR (CR={compression_ratio})")
plt.xlabel("Frame Index")
plt.ylabel("PSNR (dB)")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

# Subplot 2: Execution Time Bar Chart
plt.subplot(1, 2, 2)
times_vals = [avg_times[m] for m in methods]
colors = [style_map[m]["color"] for m in methods]
bars = plt.bar(methods, times_vals, color=colors, alpha=0.8, width=0.7)

plt.title("Execution Time per Trial")
plt.ylabel("Time (s)")
plt.grid(axis='y', linestyle=':', alpha=0.6)

# Add text on top of bars with bold formatting
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + (yval * 0.01), 
             f"{yval:.1f}s", ha='center', va='bottom', 
             fontweight='bold', fontsize=10)

plt.xticks(rotation=15, ha='center')

plt.tight_layout(w_pad=2.0)
plt.savefig("psnr_time_comparison_bold.png", dpi=300, bbox_inches='tight')
plt.show()