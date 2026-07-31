import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from tqdm import tqdm

# ==========================================
# 0. 引入库
# ==========================================
from prior_cs.utils.normalize import normalize_columns
from prior_cs.algorithms.omp import omp
from prior_cs.algorithms.pinv_psi_omp import pro_omp_solve
from prior_cs.algorithms.psi_fast import design_pinv_psi_fast

np.random.seed(42)

# ==========================================
# 1. 工具函数
# ==========================================
def dct2(block):
    return dct(dct(block.T, norm="ortho").T, norm="ortho")

def idct2(coeff):
    return idct(idct(coeff.T, norm="ortho").T, norm="ortho")

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

def stitch_patches(patches_grid, Ny, Nx, P):
    """将 patch 网格拼接成完整图像"""
    full_h = Ny * P
    full_w = Nx * P

    img = np.zeros((full_h, full_w))

    for r in range(Ny):
        for c in range(Nx):
            img[r*P:(r+1)*P, c*P:(c+1)*P] = patches_grid[r][c]

    return img

def psnr_metric(img1, img2):
    mse = np.mean((img1 - img2) ** 2)

    if mse < 1e-12:
        return 100.0

    return 10 * np.log10(1.0 / mse)

# ==========================================
# 2. 数据加载
# ==========================================
try:
    frames_array = np.load("VIDEO/video_patches.npy")
    print(f"[Info] Loaded data shape: {frames_array.shape}")

except:
    print("[Info] Generating synthetic data...")
    frames_array = np.random.rand(100, 6, 6, 32, 32)

# 全局归一化
frames_array = (
    frames_array - frames_array.min()
) / (
    frames_array.max() - frames_array.min()
)

num_frames, Ny, Nx, P, _ = frames_array.shape
n = P ** 2

# ==========================================
# 3. 实验配置
# ==========================================
TARGET_FRAME = 50
PRE_COMPUTE_FRAMES = 10
START_FRAME = max(0, TARGET_FRAME - PRE_COMPUTE_FRAMES)

compression_ratio = 0.2
m = int(compression_ratio * n)

sparsity_k = 64

HISTORY_LEN = 5
DECAY_FACTOR = 0.8

print(f"Target Frame: {TARGET_FRAME}")
print(f"Starting Simulation from Frame: {START_FRAME}")
print(f"Compression Ratio: {compression_ratio}")
print(f"Measurement Dimension: {m}")
print(f"Sparsity K: {sparsity_k}")

# ==========================================
# 4. 初始化 sensing matrix
# ==========================================
Phi = np.random.randn(m, n)
Phi = normalize_columns(Phi)

# Proposed 状态缓存
prev_rec_dct_grid = [
    [np.zeros(n) for _ in range(Nx)]
    for _ in range(Ny)
]

history_buffer = [
    [[] for _ in range(Nx)]
    for _ in range(Ny)
]

# 保存目标帧结果
rec_patches_proposed = None
rec_patches_gaussian = None

# ==========================================
# 5. 开始恢复
# ==========================================
print("Running simulation...")

for t in tqdm(range(START_FRAME, TARGET_FRAME + 1)):

    current_frame_patches_prop = [
        [None for _ in range(Nx)]
        for _ in range(Ny)
    ]

    current_frame_patches_gauss = [
        [None for _ in range(Nx)]
        for _ in range(Ny)
    ]

    for r in range(Ny):
        for c in range(Nx):

            # ----------------------------------
            # Ground Truth
            # ----------------------------------
            true_patch = frames_array[t, r, c]

            x_true = dct2(true_patch).flatten()

            # 测量
            y = Phi @ x_true

            # ==================================
            # Method 1: Proposed
            # ==================================

            # (A) Coarse OMP
            coef_coarse = omp(Phi, y, k=sparsity_k)

            x_coarse = coef_coarse.reshape(-1, 1)

            # ----------------------------------
            # (B) 构建 Prior
            # ----------------------------------
            priors = []

            # 空间邻域
            x_neighbors = get_spatial_neighbors(
                prev_rec_dct_grid,
                r, c,
                Ny, Nx
            )

            priors.append(x_neighbors)

            # 时间历史
            buf = history_buffer[r][c]

            if len(buf) > 0:

                for i, vec in enumerate(buf):

                    w = DECAY_FACTOR ** i

                    priors.append(
                        vec.reshape(-1, 1) * np.sqrt(w)
                    )

            else:
                priors.append(np.zeros((n, 1)))

            # coarse prior
            priors.append(x_coarse)

            # 合并 prior
            X_prior = np.hstack(priors)

            # ----------------------------------
            # (C) 设计 Psi
            # ----------------------------------
            Psi = design_pinv_psi_fast(
                Phi,
                X_prior
            )

            # ----------------------------------
            # (D) Proposed OMP
            # ----------------------------------
            coef_est = pro_omp_solve(
                Phi,
                Psi,
                y,
                sparsity=sparsity_k
            )

            # ----------------------------------
            # 更新缓存
            # ----------------------------------
            prev_rec_dct_grid[r][c] = coef_est

            history_buffer[r][c].insert(0, coef_est)

            if len(history_buffer[r][c]) > HISTORY_LEN:
                history_buffer[r][c].pop()

            # ==================================
            # 保存目标帧
            # ==================================
            if t == TARGET_FRAME:

                # Proposed
                rec_patch_prop = idct2(
                    coef_est.reshape(P, P)
                )

                current_frame_patches_prop[r][c] = rec_patch_prop

                # Gaussian baseline
                coef_gauss = omp(
                    Phi,
                    y,
                    k=sparsity_k
                )

                rec_patch_gauss = idct2(
                    coef_gauss.reshape(P, P)
                )

                current_frame_patches_gauss[r][c] = rec_patch_gauss

    # 保存最终帧
    if t == TARGET_FRAME:

        rec_patches_proposed = current_frame_patches_prop

        rec_patches_gaussian = current_frame_patches_gauss

# ==========================================
# 6. 图像拼接
# ==========================================
print("Stitching images...")

orig_patches = frames_array[TARGET_FRAME]

img_orig = stitch_patches(
    orig_patches,
    Ny, Nx, P
)

img_prop = stitch_patches(
    rec_patches_proposed,
    Ny, Nx, P
)

img_gauss = stitch_patches(
    rec_patches_gaussian,
    Ny, Nx, P
)

# 截断显示范围
img_prop_disp = np.clip(img_prop, 0, 1)
img_gauss_disp = np.clip(img_gauss, 0, 1)

# ==========================================
# 7. PSNR
# ==========================================
psnr_prop = psnr_metric(
    img_orig,
    img_prop_disp
)

psnr_gauss = psnr_metric(
    img_orig,
    img_gauss_disp
)

print(f"\nBaseline PSNR : {psnr_gauss:.2f} dB")
print(f"Proposed PSNR : {psnr_prop:.2f} dB")

# ==========================================
# 8. Error / Gain Map
# ==========================================
err_gauss = np.abs(
    img_orig - img_gauss_disp
)

err_prop = np.abs(
    img_orig - img_prop_disp
)

# Gain > 0:
# Proposed better
gain_map = err_gauss - err_prop

# ==========================================
# 9. Gain / Loss Statistics
# ==========================================
max_gain = np.max(gain_map)
max_loss = np.min(gain_map)

mean_gain = np.mean(gain_map)
std_gain = np.std(gain_map)

num_gain_pixels = np.sum(gain_map > 0)
num_loss_pixels = np.sum(gain_map < 0)
num_equal_pixels = np.sum(gain_map == 0)

total_pixels = gain_map.size

gain_ratio = 100 * num_gain_pixels / total_pixels
loss_ratio = 100 * num_loss_pixels / total_pixels

# 最大 gain/loss 位置
max_gain_pos = np.unravel_index(
    np.argmax(gain_map),
    gain_map.shape
)

max_loss_pos = np.unravel_index(
    np.argmin(gain_map),
    gain_map.shape
)

print("\n========== Gain / Loss Statistics ==========")

print(f"Maximum Gain        : {max_gain:.6f}")
print(f"Maximum Loss        : {max_loss:.6f}")

print(f"Mean Gain           : {mean_gain:.6f}")
print(f"Std Gain            : {std_gain:.6f}")

print(f"Gain Pixels         : {num_gain_pixels} ({gain_ratio:.2f}%)")
print(f"Loss Pixels         : {num_loss_pixels} ({loss_ratio:.2f}%)")
print(f"Equal Pixels        : {num_equal_pixels}")

print(f"\nMax Gain Position   : {max_gain_pos}")
print(f"Max Loss Position   : {max_loss_pos}")

# ==========================================
# 10. 微小收益增强显示
# ==========================================
clip_val = np.percentile(
    np.abs(gain_map),
    98
)

if clip_val < 1e-5:
    clip_val = 1e-5

# ==========================================
# 11. 可视化
# ==========================================
from mpl_toolkits.axes_grid1 import ImageGrid

plt.rcParams.update({
    'font.size': 14,
    'font.weight': 'bold',
    'axes.titleweight': 'bold',
    'axes.titlesize': 14
})

fig = plt.figure(figsize=(16, 5))

grid = ImageGrid(
    fig,
    111,
    nrows_ncols=(1, 4),
    axes_pad=0.2,
    cbar_location="right",
    cbar_mode="single",
    cbar_size="5%",
    cbar_pad=0.15
)

# ------------------------------------------
# Original
# ------------------------------------------
grid[0].imshow(
    img_orig,
    cmap='gray',
    vmin=0,
    vmax=1
)

grid[0].set_title(
    f"Original Frame {TARGET_FRAME}"
)

grid[0].axis('off')

# ------------------------------------------
# Baseline
# ------------------------------------------
grid[1].imshow(
    img_gauss_disp,
    cmap='gray',
    vmin=0,
    vmax=1
)

grid[1].set_title(
    f"Baseline\nPSNR: {psnr_gauss:.2f} dB"
)

grid[1].axis('off')

# ------------------------------------------
# Proposed
# ------------------------------------------
grid[2].imshow(
    img_prop_disp,
    cmap='gray',
    vmin=0,
    vmax=1
)

grid[2].set_title(
    f"Proposed\nPSNR: {psnr_prop:.2f} dB"
)

grid[2].axis('off')

# ------------------------------------------
# Improvement Map
# ------------------------------------------
im4 = grid[3].imshow(
    gain_map,
    cmap='coolwarm',
    vmin=-clip_val,
    vmax=clip_val
)

grid[3].set_title(
    "Improvement Map\n(Red: Gain, Blue: Loss)"
)

grid[3].axis('off')

# ------------------------------------------
# Colorbar
# ------------------------------------------
cbar = grid.cbar_axes[0].colorbar(im4)

cbar.ax.set_title(
    "Diff",
    fontsize=10,
    pad=10
)

cbar.set_ticks([
    -clip_val,
    0,
    clip_val
])

cbar.set_ticklabels([
    f"-{clip_val:.2f}",
    "0",
    f"+{clip_val:.2f}"
])

# ==========================================
# 12. 保存图片
# ==========================================
save_path = f"frame_{TARGET_FRAME}_improvement_map_boosted.png"

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print(f"\n[Done] Image saved to {save_path}")