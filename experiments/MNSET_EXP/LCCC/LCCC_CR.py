import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import pinv
from scipy.fftpack import dct
import os
import warnings

# --- Import your custom library ---
from prior_cs.algorithms.psi_fast import design_pinv_psi_fast
# from prior_cs.algorithms.pinv_psi_omp import design_pinv_psi 

warnings.filterwarnings('ignore')

# ==========================================
# 0. DCT Utilities
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
# 2. Data Loading
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
# 3. Main Experiment
# ==========================================

def run_experiment():
    # --- Experiment Settings ---
    NUM_TRIALS = 30           # Monte Carlo trials per CR
    CR_list = [0.1, 0.2, 0.3] # List of Compression Ratios to test
    K_values = np.arange(5, 41, 5) # Sparsity levels

    # --- Data Prep ---
    X_spatial = load_or_generate_data()
    N, L_total = X_spatial.shape
    X_full, D_dct = apply_dct_transform(X_spatial)
    
    print("[Info] Experiments conducted in DCT domain.")
    
    L_train = int(L_total * 0.8)
    X_train = X_full[:, :L_train]
    X_test = X_full[:, L_train:]

    # Dictionary to store mean results for plotting
    # Structure: results[cr] = {'lccc': ..., 'energy': ...}
    final_results = {}

    for CR in CR_list:
        M = int(N * CR)
        print(f"\n=== Processing Compression Ratio: {CR} (M={M}, N={N}) ===")
        
        # Temp storage for this CR
        res_lccc = np.zeros((NUM_TRIALS, 3, len(K_values)))    # Metric 1: Peak L-inf
        res_energy = np.zeros((NUM_TRIALS, 3, len(K_values)))  # Metric 2: Total Energy L2

        for t in range(NUM_TRIALS):
            if (t + 1) % 10 == 0: print(f"  Trial {t+1}/{NUM_TRIALS}...")

            # 1. Measurement Matrix
            Phi = np.random.randn(M, N)
            Phi = Phi / np.linalg.norm(Phi, axis=0, keepdims=True)

            # 2. Sensing Matrices
            # (A) Baseline
            Psi_base = Phi.copy()
            # (B) Paper (POCS)
            Psi_paper = get_psi_paper_pocs(Phi, max_iter=30)
            # (C) Proposed
            Psi_prop = design_pinv_psi_fast(Phi, X_train) 
            Psi_prop = Psi_prop / (np.linalg.norm(Psi_prop, axis=0, keepdims=True) + 1e-10)

            # 3. Gram Matrices
            AbsG_base = np.abs(Psi_base.T @ Phi)
            AbsG_paper = np.abs(Psi_paper.T @ Phi)
            AbsG_prop = np.abs(Psi_prop.T @ Phi)

            # 4. Evaluation on Test Batch
            sample_indices = np.random.choice(X_test.shape[1], 15, replace=False) # Faster eval
            X_batch = X_test[:, sample_indices]

            for k_idx, K in enumerate(K_values):
                mu_vals = np.zeros((3, len(sample_indices)))
                eng_vals = np.zeros((3, len(sample_indices)))

                for i in range(len(sample_indices)):
                    x_raw = X_batch[:, i]
                    x = x_raw.copy()
                    x[0] = 0 # Remove DC

                    if np.all(x == 0): continue

                    # Hard Thresholding
                    idx_sorted = np.argsort(np.abs(x))
                    Gamma = idx_sorted[-K:]
                    
                    # Normalized Coeffs
                    x_gamma = np.abs(x[Gamma])
                    if np.linalg.norm(x_gamma) > 0:
                        x_gamma = x_gamma / np.linalg.norm(x_gamma)
                    
                    mask = np.ones(N, dtype=bool)
                    mask[Gamma] = False
                    Gamma_c = np.where(mask)[0]

                    # Interference Vectors
                    w_vec_base = AbsG_base[Gamma_c][:, Gamma] @ x_gamma
                    w_vec_paper = AbsG_paper[Gamma_c][:, Gamma] @ x_gamma
                    w_vec_prop = AbsG_prop[Gamma_c][:, Gamma] @ x_gamma

                    # Metric 1: Peak (Geometric Worst Case)
                    mu_vals[0, i] = np.max(w_vec_base)
                    mu_vals[1, i] = np.max(w_vec_paper)
                    mu_vals[2, i] = np.max(w_vec_prop)

                    # Metric 2: Energy (Effective Noise) -> THIS PROVES YOUR POINT
                    eng_vals[0, i] = np.linalg.norm(w_vec_base)
                    eng_vals[1, i] = np.linalg.norm(w_vec_paper)
                    eng_vals[2, i] = np.linalg.norm(w_vec_prop)

                res_lccc[t, :, k_idx] = np.mean(mu_vals, axis=1)
                res_energy[t, :, k_idx] = np.mean(eng_vals, axis=1)
        
        # Save averaged results for this CR
        final_results[CR] = {
            'lccc': np.mean(res_lccc, axis=0),
            'energy': np.mean(res_energy, axis=0)
        }

    # --- Plotting Grid ---
    # Create a figure with 2 rows (Metrics) and 3 columns (CRs)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    
    # Define styles
    styles = [
        {'c': 'gray', 'ls': '--', 'm': 'o', 'lbl': 'Baseline'},
        {'c': 'blue', 'ls': '-', 'm': 's', 'lbl': 'Paper (POCS)'},
        {'c': 'red', 'ls': '-', 'm': '^', 'lbl': 'Proposed (Ours)'}
    ]

    for idx, CR in enumerate(CR_list):
        res = final_results[CR]
        
        # --- Row 1: Peak Interference (Geometric) ---
        ax_top = axes[0, idx]
        for m_id in range(3):
            ax_top.plot(K_values, res['lccc'][m_id], 
                        color=styles[m_id]['c'], linestyle=styles[m_id]['ls'], 
                        marker=styles[m_id]['m'], label=styles[m_id]['lbl'], 
                        linewidth=2, alpha=0.8)
        
        ax_top.set_title(f'CR = {CR}\n(a) Peak Interference (L-inf)', fontsize=13)
        ax_top.grid(True, linestyle=':', alpha=0.6)
        if idx == 0: ax_top.set_ylabel('Peak Value')

        # --- Row 2: Effective Interference Energy (L2) ---
        ax_bot = axes[1, idx]
        for m_id in range(3):
            ax_bot.plot(K_values, res['energy'][m_id], 
                        color=styles[m_id]['c'], linestyle=styles[m_id]['ls'], 
                        marker=styles[m_id]['m'], label=styles[m_id]['lbl'], 
                        linewidth=2, alpha=0.8)
            
        ax_bot.set_title(f'(b) Effective Interference Energy (L2)', fontsize=13)
        ax_bot.grid(True, linestyle=':', alpha=0.6)
        ax_bot.set_xlabel('Sparsity K')
        if idx == 0: ax_bot.set_ylabel('Total Energy')
        
        # Legend only in the first column to save space
        if idx == 0:
            ax_top.legend(fontsize=10)
            ax_bot.legend(fontsize=10)

    plt.suptitle('Comparison across Compression Ratios (CR): Peak vs. Energy', fontsize=16, y=0.98)
    plt.tight_layout()
    
    save_file = "interference_CR_comparison.png"
    plt.savefig(save_file, dpi=300)
    plt.show()

    print(f"[Done] Multi-CR experiment finished. Saved to {save_file}")

if __name__ == "__main__":
    run_experiment()