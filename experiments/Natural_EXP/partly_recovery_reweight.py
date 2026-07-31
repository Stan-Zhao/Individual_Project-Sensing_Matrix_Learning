import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from skimage.util import view_as_blocks
from skimage.color import rgb2gray

# Import necessary modules from your library
from prior_cs.utils.normalize import normalize_columns
from prior_cs.algorithms.pinv_psi_omp import (
    design_pinv_psi,
    pro_omp_solve
)

np.random.seed(41)

# =====================================================
# 0. Basic Utility Functions
# =====================================================
def dct2(block):
    return dct(dct(block.T, norm="ortho").T, norm="ortho")

def idct2(coeff):
    return idct(idct(coeff.T, norm="ortho").T, norm="ortho")

def psnr(x_true, x_rec):
    mse = np.mean((x_true - x_rec) ** 2)
    if mse < 1e-12:
        return 100.0
    return 10 * np.log10(1.0 / mse)

# =====================================================
# 1. Reweighted OMP Implementation (Strong Baseline)
# =====================================================
def get_frequency_weights(n, boost_factor=2.0):
    """
    Creates a weight vector that boosts low frequencies (top-left 3x3).
    """
    w_mat = int(np.sqrt(n)) # 8 for 8x8 patches
    weights = np.ones(n)
    
    # Map 1D index back to 2D to identify the top-left corner
    indices = np.arange(n).reshape(w_mat, w_mat)
    
    # Boost the 3x3 low-frequency block
    for r in range(3):
        for c in range(3):
            idx = indices[r, c]
            weights[idx] = boost_factor
            
    return weights

def reweighted_omp(Phi, y, weights, k=10, tol=1e-6):
    """
    OMP where atom selection is biased by 'weights'.
    """
    m, n = Phi.shape
    r = y.copy()
    omega = []  # Selected indices
    x_hat = np.zeros(n)
    
    for _ in range(k):
        # 1. Calculate correlations
        corrs = Phi.T @ r
        
        # 2. Weighted Selection: Scale correlation by frequency weights
        # This forces the algorithm to check low-freq atoms first
        weighted_corrs = np.abs(corrs) * weights
        
        # Mask already selected indices
        weighted_corrs[omega] = -1.0
        
        best_idx = np.argmax(weighted_corrs)
        
        if weighted_corrs[best_idx] < tol:
            break
            
        omega.append(best_idx)
        
        # 3. Least Squares (Standard projection, unweighted)
        Phi_S = Phi[:, omega]
        # Use pinv for stability
        x_S = np.linalg.pinv(Phi_S) @ y
        
        # 4. Update Residual
        r = y - Phi_S @ x_S
        
    x_hat[omega] = x_S
    return x_hat

# =====================================================
# 2. Prior Construction Functions (Proposed)
# =====================================================
def get_neighborhood_prior_patches(grid, cy, cx, radius=1):
    Ny, Nx = grid.shape[:2]
    patches = []

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            py, px = cy + dy, cx + dx
            if 0 <= py < Ny and 0 <= px < Nx:
                patches.append(dct2(grid[py, px]).flatten())

    return np.array(patches).T   # (64, L)


def apply_frequency_mask(X, keep_dc=True, keep_lf=True):
    n, L = X.shape
    w = int(np.sqrt(n))
    mask = np.zeros(n)

    if keep_dc:
        mask[0] = 1.0

    if keep_lf:
        idx = np.arange(n).reshape(w, w)
        for i in range(3):
            for j in range(3):
                mask[idx[i, j]] = 1.0

    return X * mask[:, None]

# =====================================================
# 3. Random Image Loader
# =====================================================
def load_random_image(pic_dir):
    files = [f for f in os.listdir(pic_dir)
             if f.lower().endswith(('.png', '.jpg', '.bmp'))]
    assert len(files) > 0, "pic folder is empty"

    fname = random.choice(files)
    img = plt.imread(os.path.join(pic_dir, fname)).astype(np.float32)

    if img.ndim == 3:
        img = rgb2gray(img)

    img /= img.max() + 1e-12
    return img, fname

# =====================================================
# 4. Experiment Parameters
# =====================================================
NUM_TRIALS = 500

radius_test  = 3
radius_prior = 1

patch_size = 8
compression_ratio = 0.3
n = patch_size ** 2
m = int(compression_ratio * n)
sparsity_k = 10

grid_dim = 2 * radius_test + 1

# Setup Weights for Reweighted OMP
# Boosting low frequencies by 3x makes it a very strong baseline
weight_factor = 3.0
freq_weights = get_frequency_weights(n, boost_factor=weight_factor)

print("Monte-Carlo Comparison Setup:")
print(f"Trials: {NUM_TRIALS}")
print(f"Grid: {grid_dim}x{grid_dim}")
print(f"Baseline: Reweighted OMP (LF Boost x{weight_factor})")
print(f"Proposed: Pinv-OMP (Spatial Prior)")

# =====================================================
# 5. Statistical Accumulator
# =====================================================
acc_gain = np.zeros((grid_dim, grid_dim))

# =====================================================
# 6. Monte-Carlo Main Loop
# =====================================================
for t in range(NUM_TRIALS):

    print(f"Trial {t+1}/{NUM_TRIALS}")

    # ---- (A) Load Image ----
    img, fname = load_random_image("pic")
    H, W = img.shape
    Ny, Nx = H // patch_size, W // patch_size
    img = img[:Ny*patch_size, :Nx*patch_size]

    patches = view_as_blocks(img, (patch_size, patch_size))
    patch_grid = patches.reshape(Ny, Nx, patch_size, patch_size)

    # ---- (B) Random Center Patch ----
    margin = max(radius_test, radius_prior) + 1
    cy = np.random.randint(margin, Ny - margin)
    cx = np.random.randint(margin, Nx - margin)

    # ---- (C) Random Measurement Matrix ----
    Phi = np.random.randn(m, n)
    Phi = normalize_columns(Phi)

    # ---- (D) Construct Prior Psi (For Proposed) ----
    X_train = get_neighborhood_prior_patches(
        patch_grid, cy, cx, radius=radius_prior
    )
    X_train = apply_frequency_mask(X_train)
    Psi = design_pinv_psi(Phi, X_train)

    # ---- (E) Iterate Neighborhood ----
    for dy in range(-radius_test, radius_test + 1):
        for dx in range(-radius_test, radius_test + 1):

            if dy == 0 and dx == 0:
                continue

            r = dy + radius_test
            c = dx + radius_test

            py, px = cy + dy, cx + dx
            true_patch = patch_grid[py, px]
            true_dct = dct2(true_patch).flatten()
            y = Phi @ true_dct

            # --- Method A: Reweighted OMP (Strong Baseline) ---
            # Uses 'freq_weights' to prioritize low frequencies generically
            coef_base = reweighted_omp(Phi, y, freq_weights, k=sparsity_k)
            rec_base = idct2(coef_base.reshape(8, 8))
            p_base = psnr(true_patch, rec_base)

            # --- Method B: Proposed Pinv-OMP ---
            # Uses 'Psi' to prioritize specific values from neighbors
            coef_p = pro_omp_solve(Phi, Psi, y, sparsity=sparsity_k)
            rec_p = idct2(coef_p.reshape(8, 8))
            p_p = psnr(true_patch, rec_p)

            # Accumulate Gain (Proposed - Baseline)
            acc_gain[r, c] += (p_p - p_base)

# =====================================================
# 7. Visualization
# =====================================================
avg_gain = acc_gain / NUM_TRIALS
avg_gain[radius_test, radius_test] = 0

plt.figure(figsize=(8, 7))
# Adjust color scale: negative values might occur far away now that baseline is stronger
plt.imshow(avg_gain, cmap="RdYlGn", vmin=0.5, vmax=1)
plt.colorbar(label="PSNR Gain (Proposed - Reweighted OMP)")

for r in range(grid_dim):
    for c in range(grid_dim):
        dy, dx = r-radius_test, c-radius_test
        if dy == 0 and dx == 0:
            plt.text(c, r, "Prior", ha="center", va="center", fontweight="bold")
        else:
            val = avg_gain[r,c]
            # Use white text for dark green/red, black for light colors
            color = "white" if abs(val) > 1.0 else "black"
            plt.text(c, r, f"{val:+.1f}",
                     ha="center", va="center", color=color, fontweight="bold", fontsize=9)

ticks = np.arange(grid_dim)
labels = [str(i-radius_test) for i in ticks]
plt.xticks(ticks, labels)
plt.yticks(ticks, labels)

plt.title(f"Gain over Reweighted OMP ({NUM_TRIALS} trials)\nIsolating Spatial Info from Frequency Bias")
plt.xlabel("Δx (patch)")
plt.ylabel("Δy (patch)")
plt.tight_layout()
plt.show()