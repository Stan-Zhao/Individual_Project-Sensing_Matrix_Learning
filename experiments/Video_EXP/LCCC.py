import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import pinv
from scipy.fftpack import dctn
import os
import warnings
import random

# --- Import your custom library ---
from prior_cs.algorithms.psi_fast import design_pinv_psi_fast

warnings.filterwarnings('ignore')

# ==========================================
# 0. Data & Transform Utilities
# ==========================================

def perform_2d_dct_flatten(patch):
    """
    输入: 空间域 patch, shape (P, P)
    输出: DCT域扁平向量, shape (N,)
    """
    patch_dct = dctn(patch, norm='ortho')
    return patch_dct.flatten()

def load_or_generate_video_data(save_dir='VIDEO', T=10, Ny=6, Nx=6, P=8):
    """
    加载或生成模拟视频序列 (带维度检查)
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    file_path = os.path.join(save_dir, "video_patches.npy")
    generate_flag = True
    
    if os.path.exists(file_path):
        try:
            frames_array = np.load(file_path)
            # 维度检查: 最后一个维度必须等于 P
            if frames_array.shape[-1] == P:
                print(f"[Info] Loading video data from {file_path} (P={P})...")
                generate_flag = False
            else:
                print(f"[Warning] Dimension mismatch (Saved P={frames_array.shape[-1]} != Requested P={P}). Regenerating...")
        except:
            print("[Warning] File corrupted. Regenerating...")
            
    if generate_flag:
        print(f"[Info] Generating synthetic video sequence！！！！！(AR-1 process, P={P})...")
        frames_array = np.zeros((T, Ny, Nx, P, P), dtype=np.float32)
        # 第一帧随机
        frames_array[0] = np.random.rand(Ny, Nx, P, P)
        
        # 后续帧：强时间相关
        for t in range(1, T):
            noise = np.random.normal(0, 0.05, frames_array[0].shape)
            frames_array[t] = frames_array[t-1] * 0.95 + noise
        
        # 归一化
        frames_array = (frames_array - frames_array.min()) / (frames_array.max() - frames_array.min() + 1e-12)
        np.save(file_path, frames_array)
        
    return frames_array

def get_all_past_priors(frames_array, t, r, c):
    """
    [修改点] 获取所有历史帧 (0 到 t-1) 同一位置的 Patch
    返回: DCT 域矩阵, shape (N, t)
    """
    if t == 0:
        return None
    
    # 取出 0 到 t-1 帧在 (r,c) 位置的所有 patches
    # shape: (t, P, P)
    past_patches_spatial = frames_array[:t, r, c] 
    
    # 转换为 DCT 域并堆叠为列向量
    # Result shape: (N, t)
    N = past_patches_spatial.shape[1] * past_patches_spatial.shape[2]
    X_prior = np.zeros((N, t))
    
    for i in range(t):
        X_prior[:, i] = perform_2d_dct_flatten(past_patches_spatial[i])
        
    return X_prior

# ==========================================
# 1. POCS Algorithm (Paper Method)
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
# 2. Main Experiment
# ==========================================

def run_experiment():
    # --- Settings ---
    P_size = 32
    N = P_size * P_size
    
    NUM_TRIALS = 50  
    CR_list = [0.1, 0.2, 0.3] 
    K_values = np.arange(2, 17, 2) 
    
    # 1. 加载数据
    frames_array = load_or_generate_video_data(T=15, Ny=10, Nx=10, P=P_size) # T稍微大一点，体现积累优势
    T, Ny, Nx, _, _ = frames_array.shape
    
    final_results = {}

    for CR in CR_list:
        M = int(N * CR)
        if M < 1: M = 1
        
        print(f"\n=== Processing CR: {CR} (M={M}, N={N}) ===")
        
        res_weighted = np.zeros((NUM_TRIALS, 3, len(K_values)))

        for trial_idx in range(NUM_TRIALS):
            # 随机选择位置。确保 t 足够大，以便有历史数据。
            # 比如从第 5 帧开始测，这样至少有 5 个历史帧作为先验
            t = random.randint(5, T - 1) 
            r = random.randint(0, Ny - 1)
            c = random.randint(0, Nx - 1)
            
            # --- A. 测试数据 (当前帧 t) ---
            true_patch_spatial = frames_array[t, r, c]
            x_true = perform_2d_dct_flatten(true_patch_spatial)
            
            # --- B. 先验数据 (0 到 t-1 帧) ---
            # X_prior shape: (N, t)
            X_prior = get_all_past_priors(frames_array, t, r, c)
            
            # 1. Measurement Matrix
            Phi = np.random.randn(M, N)
            Phi = Phi / np.linalg.norm(Phi, axis=0, keepdims=True)

            # 2. Design Sensing Matrices
            # (A) Baseline
            Psi_base = Phi.copy()
            
            # (B) Paper (POCS)
            Psi_paper = get_psi_paper_pocs(Phi, max_iter=30)
            
            # (C) Proposed (History Adaptive)
            # design_pinv_psi_fast 会利用 X_prior 计算协方差 R = X X^T
            # 这里 X_prior 列数越多，R 越稳健
            Psi_prop = design_pinv_psi_fast(Phi, X_prior) 
            Psi_prop = Psi_prop / (np.linalg.norm(Psi_prop, axis=0, keepdims=True) + 1e-10)

            # 3. Abs Gram Matrices
            AbsG_base = np.abs(Psi_base.T @ Phi)
            AbsG_paper = np.abs(Psi_paper.T @ Phi)
            AbsG_prop = np.abs(Psi_prop.T @ Phi)

            # 4. Evaluation (Weighted Interference)
            # 去除 DC
            x_eval = x_true.copy()
            x_eval[0] = 0 
            
            if np.all(x_eval == 0): continue
            
            idx_sorted = np.argsort(np.abs(x_eval))
            
            for k_idx, K in enumerate(K_values):
                Gamma = idx_sorted[-K:]
                x_gamma = np.abs(x_eval[Gamma])
                
                mask = np.ones(N, dtype=bool)
                mask[Gamma] = False
                Gamma_c = np.where(mask)[0]
                
                # Metric
                res_weighted[trial_idx, 0, k_idx] = np.max(AbsG_base[Gamma_c][:, Gamma] @ x_gamma)
                res_weighted[trial_idx, 1, k_idx] = np.max(AbsG_paper[Gamma_c][:, Gamma] @ x_gamma)
                res_weighted[trial_idx, 2, k_idx] = np.max(AbsG_prop[Gamma_c][:, Gamma] @ x_gamma)

        final_results[CR] = np.mean(res_weighted, axis=0)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    styles = [
        {'c': 'gray', 'ls': '--', 'm': 'o', 'lbl': 'Baseline'},
        {'c': 'blue', 'ls': '-', 'm': 's', 'lbl': 'Paper (POCS)'},
        {'c': 'red', 'ls': '-', 'm': '^', 'lbl': 'Proposed (All-History)'}
    ]

    for idx, CR in enumerate(CR_list):
        ax = axes[idx]
        res = final_results[CR]
        
        for m_id in range(3):
            ax.plot(K_values, res[m_id], 
                    color=styles[m_id]['c'], linestyle=styles[m_id]['ls'], 
                    marker=styles[m_id]['m'], label=styles[m_id]['lbl'], 
                    linewidth=2, alpha=0.8)
        
        ax.set_title(f'CR = {CR}\nWeighted Interference (Full History Prior)', fontsize=13)
        ax.set_xlabel('Sparsity K')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if idx == 0: 
            ax.set_ylabel(r'Max Weighted Interference')
            ax.legend(fontsize=10)

    plt.suptitle('Video Patch Experiment: Using ALL Previous Frames as Prior', fontsize=16)
    plt.tight_layout()
    
    save_file = "weighted_interference_video_history.png"
    plt.savefig(save_file, dpi=300)
    plt.show()

    print(f"[Done] Saved to {save_file}")
    print("Observation: With more history (more columns in X_prior), the subspace estimation")
    print("becomes more stable, potentially leading to even lower interference than single-frame prior.")

if __name__ == "__main__":
    run_experiment()