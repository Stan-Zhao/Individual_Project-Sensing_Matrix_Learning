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
# 基础工具函数
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
# 先验构造函数
# =====================================================
def get_neighborhood_prior_patches(grid, cy, cx, radius=1):
    Ny, Nx = grid.shape[:2]
    patches = []

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            py, px = cy + dy, cx + dx
            if 0 <= py < Ny and 0 <= px < Nx:
                patches.append(dct2(grid[py, px]).flatten())

    return np.array(patches).T

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
# 随机读取图片
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
# 实验参数
# =====================================================
NUM_TRIALS = 10      # Monte-Carlo 次数
patch_size = 8
compression_ratio = 0.3
sparsity_k = 10
radius_prior = 1     # 先验邻域
radius_test  = 300   # 全局范围 (~整张图)
n = patch_size ** 2
m = int(compression_ratio * n)

print(f"Monte-Carlo: trials={NUM_TRIALS}, patch_size={patch_size}, m={m}, k={sparsity_k}")

# =====================================================
# 累加器
# =====================================================
acc_gain = {}  # 按距离聚合

# =====================================================
# Monte-Carlo 主循环
# =====================================================
for t in range(NUM_TRIALS):
    print(f"Trial {t+1}/{NUM_TRIALS}")

    img, fname = load_random_image("pic")
    H, W = img.shape
    Ny, Nx = H // patch_size, W // patch_size
    img = img[:Ny*patch_size, :Nx*patch_size]
    patches = view_as_blocks(img, (patch_size, patch_size))
    patch_grid = patches.reshape(Ny, Nx, patch_size, patch_size)

    # 随机中心 patch
    margin = radius_prior + 1
    cy = np.random.randint(margin, Ny - margin)
    cx = np.random.randint(margin, Nx - margin)

    # 随机测量矩阵
    Phi = np.random.randn(m, n)
    Phi = normalize_columns(Phi)

    # 构造先验 Psi
    X_train = get_neighborhood_prior_patches(patch_grid, cy, cx, radius=radius_prior)
    X_train = apply_frequency_mask(X_train)
    Psi = design_pinv_psi(Phi, X_train)

    # 遍历整张图
    for py in range(Ny):
        for px in range(Nx):
            if py == cy and px == cx:
                continue  # 跳过中心

            true_patch = patch_grid[py, px]
            true_dct = dct2(true_patch).flatten()
            y = Phi @ true_dct

            coef_c = omp(Phi, y, k=sparsity_k)
            rec_c = idct2(coef_c.reshape(patch_size, patch_size))
            p_c = psnr(true_patch, rec_c)

            coef_p = pro_omp_solve(Phi, Psi, y, sparsity=sparsity_k)
            rec_p = idct2(coef_p.reshape(patch_size, patch_size))
            p_p = psnr(true_patch, rec_p)

            # Chebyshev 距离
            d = max(abs(py - cy), abs(px - cx))
            acc_gain.setdefault(d, []).append(p_p - p_c)

# =====================================================
# 按距离统计
# =====================================================
distances = sorted(acc_gain.keys())
mean_gain = [np.mean(acc_gain[d]) for d in distances]
std_gain  = [np.std(acc_gain[d])  for d in distances]

# =====================================================
# 全局折线图
# =====================================================
plt.figure(figsize=(8, 5))
plt.errorbar(
    distances,
    mean_gain,
    yerr=std_gain,
    fmt='o-',
    capsize=4,
    linewidth=2
)
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Patch Distance (Chebyshev)")
plt.ylabel("Average PSNR Gain (dB)")
plt.title(f"Global Influence of Prior (Full Image)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# =====================================================
# 打印摘要
# =====================================================
print("-"*40)
for d, mg in zip(distances, mean_gain):
    print(f"Distance {d}: mean gain = {mg:.3f} dB")
print("-"*40)
