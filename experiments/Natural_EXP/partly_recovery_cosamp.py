import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from skimage.util import view_as_blocks
from skimage.color import rgb2gray

from prior_cs.algorithms.omp import omp
from prior_cs.utils.normalize import normalize_columns
from prior_cs.algorithms.pinv_psi_omp import pro_omp_solve
from prior_cs.algorithms.pinv_psi_cosamp import pro_cosamp_solve, design_pinv_psi

np.random.seed(41)

# =====================================================
# 0. 基础工具函数
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
# 1. 先验构造
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
# 2. 随机图像
# =====================================================
def load_random_image(pic_dir):
    files = [f for f in os.listdir(pic_dir)
             if f.lower().endswith(('.png', '.jpg', '.bmp'))]
    fname = random.choice(files)

    img = plt.imread(os.path.join(pic_dir, fname)).astype(np.float32)
    if img.ndim == 3:
        img = rgb2gray(img)

    img /= img.max() + 1e-12
    return img, fname

# =====================================================
# 3. 参数
# =====================================================
NUM_TRIALS = 500

radius_test  = 3
radius_prior = 1

patch_size = 8
n = patch_size ** 2
m = int(0.3 * n)
k = 4

grid_dim = 2 * radius_test + 1

# =====================================================
# 4. 统计量
# =====================================================
acc_gain_omp   = np.zeros((grid_dim, grid_dim))
acc_gain_cosamp = np.zeros((grid_dim, grid_dim))

# =====================================================
# 5. Monte-Carlo
# =====================================================
for t in range(NUM_TRIALS):

    print(f"Trial {t+1}/{NUM_TRIALS}")

    img, _ = load_random_image("pic")
    H, W = img.shape
    Ny, Nx = H // patch_size, W // patch_size
    img = img[:Ny*patch_size, :Nx*patch_size]

    patch_grid = view_as_blocks(img, (patch_size, patch_size)) \
                    .reshape(Ny, Nx, patch_size, patch_size)

    margin = max(radius_test, radius_prior) + 1
    cy = np.random.randint(margin, Ny - margin)
    cx = np.random.randint(margin, Nx - margin)

    Phi = normalize_columns(np.random.randn(m, n))

    # ---- Prior Psi ----
    X_train = get_neighborhood_prior_patches(
        patch_grid, cy, cx, radius=radius_prior
    )
    X_train = apply_frequency_mask(X_train)
    Psi = design_pinv_psi(Phi, X_train)

    # ---- Test neighborhood ----
    for dy in range(-radius_test, radius_test + 1):
        for dx in range(-radius_test, radius_test + 1):

            if dy == 0 and dx == 0:
                continue

            r = dy + radius_test
            c = dx + radius_test

            py, px = cy + dy, cx + dx
            patch = patch_grid[py, px]
            x_true = dct2(patch).flatten()
            y = Phi @ x_true

            # ===== Baseline OMP =====
            coef_omp = omp(Phi, y, k=k)
            rec_omp = idct2(coef_omp.reshape(8, 8))
            p_omp = psnr(patch, rec_omp)

            # ===== Pro-OMP =====
            coef_pomp = pro_omp_solve(Phi, Psi, y, sparsity=k)
            rec_pomp = idct2(coef_pomp.reshape(8, 8))
            p_pomp = psnr(patch, rec_pomp)

            # ===== Pro-CoSaMP =====
            coef_pcosamp = pro_cosamp_solve(
                Phi, Psi, y, sparsity=k, max_iter=20
            )
            rec_pcosamp = idct2(coef_pcosamp.reshape(8, 8))
            p_pcosamp = psnr(patch, rec_pcosamp)

            acc_gain_omp[r, c] += (p_pomp - p_omp)
            acc_gain_cosamp[r, c] += (p_pcosamp - p_omp)

# =====================================================
# 6. 可视化（带数值标注）
# =====================================================
gain_omp = acc_gain_omp / NUM_TRIALS
gain_cosamp = acc_gain_cosamp / NUM_TRIALS

gain_omp[radius_test, radius_test] = 0
gain_cosamp[radius_test, radius_test] = 0

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

titles = ["Pro-OMP gain over OMP", "Pro-CoSaMP gain over OMP"]
data_list = [gain_omp, gain_cosamp]

for ax, data, title in zip(axes, data_list, titles):

    im = ax.imshow(data, cmap="RdYlGn", vmin=-1, vmax=1.5)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    # --- 标注数值 ---
    for i in range(grid_dim):
        for j in range(grid_dim):

            if i == radius_test and j == radius_test:
                ax.text(j, i, "0",
                        ha="center", va="center",
                        color="black", fontweight="bold")
                continue

            val = data[i, j]
            color = "black" if abs(val) < 0.5 else "white"

            ax.text(j, i, f"{val:+.2f}",
                    ha="center", va="center",
                    fontsize=10,
                    color=color,
                    fontweight="bold")

    # 坐标轴标注为 patch 偏移
    ticks = np.arange(grid_dim)
    labels = [str(i - radius_test) for i in ticks]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Δx (patch)")
    ax.set_ylabel("Δy (patch)")

plt.tight_layout()
plt.show()
