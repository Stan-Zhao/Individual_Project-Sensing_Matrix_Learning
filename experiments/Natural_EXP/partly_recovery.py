import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from skimage.util import view_as_blocks
from skimage.color import rgb2gray

from prior_cs.algorithms.omp import omp
from prior_cs.utils.normalize import normalize_columns
from prior_cs.algorithms.pinv_psi_omp import (
    design_pinv_psi,
    pro_omp_solve
)

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
# 1. 先验构造函数
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
# 2. 随机图像读取
# =====================================================
def load_random_image(pic_dir):
    files = [f for f in os.listdir(pic_dir)
             if f.lower().endswith(('.png', '.jpg', '.bmp'))]
    assert len(files) > 0, "pic 文件夹为空"

    fname = random.choice(files)
    img = plt.imread(os.path.join(pic_dir, fname)).astype(np.float32)

    if img.ndim == 3:
        img = rgb2gray(img)

    img /= img.max() + 1e-12
    return img, fname

# =====================================================
# 3. 实验参数
# =====================================================
NUM_TRIALS = 10

radius_test  = 2
radius_prior = 1

patch_size = 8
compression_ratio = 0.3
n = patch_size ** 2
m = int(compression_ratio * n)
sparsity_k = 10

grid_dim = 2 * radius_test + 1

print("Monte-Carlo setup:")
print(f"Trials: {NUM_TRIALS}")
print(f"Grid: {grid_dim}x{grid_dim}")
print(f"m={m}, k={sparsity_k}")

# =====================================================
# 4. 统计累加器
# =====================================================
acc_gain = np.zeros((grid_dim, grid_dim))

# =====================================================
# 5. Monte-Carlo 主循环
# =====================================================
for t in range(NUM_TRIALS):

    print(f"Trial {t+1}/{NUM_TRIALS}")

    # ---- (A) 随机选图像 ----
    img, fname = load_random_image("pic")
    H, W = img.shape

    Ny, Nx = H // patch_size, W // patch_size
    img = img[:Ny*patch_size, :Nx*patch_size]

    patches = view_as_blocks(img, (patch_size, patch_size))
    patch_grid = patches.reshape(Ny, Nx, patch_size, patch_size)

    # ---- (B) 随机中心 patch ----
    margin = max(radius_test, radius_prior) + 1
    cy = np.random.randint(margin, Ny - margin)
    cx = np.random.randint(margin, Nx - margin)

    # ---- (C) 随机测量矩阵 ----
    Phi = np.random.randn(m, n)
    Phi = normalize_columns(Phi)

    # ---- (D) 构造先验 Psi ----
    X_train = get_neighborhood_prior_patches(
        patch_grid, cy, cx, radius=radius_prior
    )
    X_train = apply_frequency_mask(X_train)
    Psi = design_pinv_psi(Phi, X_train)

    # ---- (E) 遍历邻域 ----
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

            # Classic OMP
            coef_c = omp(Phi, y, k=sparsity_k)
            rec_c = idct2(coef_c.reshape(8, 8))
            p_c = psnr(true_patch, rec_c)

            # Proposed
            coef_p = pro_omp_solve(Phi, Psi, y, sparsity=sparsity_k)
            rec_p = idct2(coef_p.reshape(8, 8))
            p_p = psnr(true_patch, rec_p)

            acc_gain[r, c] += (p_p - p_c)

# =====================================================
# 6. 统计 & 可视化
# =====================================================
avg_gain = acc_gain / NUM_TRIALS
avg_gain[radius_test, radius_test] = 0

plt.figure(figsize=(8, 7))
plt.imshow(avg_gain, cmap="RdYlGn", vmin=-1, vmax=3)
plt.colorbar(label="Average PSNR Gain (dB)")

for r in range(grid_dim):
    for c in range(grid_dim):
        dy, dx = r-radius_test, c-radius_test
        if dy == 0 and dx == 0:
            plt.text(c, r, "Prior", ha="center", va="center", fontweight="bold")
        else:
            plt.text(c, r, f"{avg_gain[r,c]:+.1f}",
                     ha="center", va="center", fontweight="bold")

ticks = np.arange(grid_dim)
labels = [str(i-radius_test) for i in ticks]
plt.xticks(ticks, labels)
plt.yticks(ticks, labels)

plt.title(f"Monte-Carlo over Random Images ({NUM_TRIALS} trials)")
plt.xlabel("Δx (patch)")
plt.ylabel("Δy (patch)")
plt.tight_layout()
plt.show()
