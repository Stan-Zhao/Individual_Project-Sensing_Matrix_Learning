import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct
from skimage.util import view_as_blocks

# =============================
# 配置参数
# =============================
IMG_PATH = '/Users/stanzhao/Desktop/prior_cs/experiments/Natural_EXP/pic/MRI.png'
PATCH_SIZE = 32        # 修改为 32x32
NUM_TRIALS = 1000      # 重复实验次数 (随机采样的 Patch 数量)
MAX_SHIFT = 4          # 邻域偏移范围

# =============================
# 工具函数
# =============================
def dct2(block):
    return dct(dct(block.T, norm="ortho").T, norm="ortho")

def monte_carlo_corr_map(patch_grid, extractor, num_trials=1000, max_shift=4):
    """
    随机采样版相关性计算：
    1. 随机选择 num_trials 个有效位置的 Patch。
    2. 提取这些 Patch 及其邻域 Patch 的特征。
    3. 计算两组特征向量之间的皮尔逊相关系数。
    """
    Ny, Nx = patch_grid.shape[:2]
    
    # 1. 确定有效的中心点范围 (为了保证邻居不越界)
    valid_y_range = np.arange(max_shift, Ny - max_shift)
    valid_x_range = np.arange(max_shift, Nx - max_shift)
    
    # 2. 随机采样中心点坐标 (重复实验 num_trials 次)
    # 使用随机种子保证可复现性，实际使用可去掉
    rng = np.random.default_rng(42) 
    rand_y = rng.choice(valid_y_range, num_trials)
    rand_x = rng.choice(valid_x_range, num_trials)
    
    # 3. 预先提取所有中心 Patch 的特征
    # center_patches shape: (num_trials, P, P)
    center_patches = patch_grid[rand_y, rand_x]
    center_feats = np.array([extractor(p) for p in center_patches])
    center_vec = center_feats.flatten() # 拉平用于计算相关系数
    
    # 4. 遍历邻域偏移量计算相关性
    size = 2 * max_shift + 1
    corr_map = np.zeros((size, size))
    
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            if dy == 0 and dx == 0:
                corr_map[dy + max_shift, dx + max_shift] = 1.0
                continue
            
            # 获取对应的邻居 Patch
            neighbor_y = rand_y + dy
            neighbor_x = rand_x + dx
            neighbor_patches = patch_grid[neighbor_y, neighbor_x]
            
            # 提取邻居特征
            neighbor_feats = np.array([extractor(p) for p in neighbor_patches])
            neighbor_vec = neighbor_feats.flatten()
            
            # 计算相关系数 (Pearson Correlation)
            if center_vec.size > 0 and neighbor_vec.size > 0:
                # np.corrcoef 返回矩阵 [[1, r], [r, 1]]，取 [0,1]
                corr = np.corrcoef(center_vec, neighbor_vec)[0, 1]
                corr_map[dy + max_shift, dx + max_shift] = corr
                
    return corr_map

# =============================
# 1. 读取并预处理图像
# =============================
try:
    img = plt.imread(IMG_PATH)
except FileNotFoundError:
    print(f"Error: 找不到文件 {IMG_PATH}，请生成随机噪声图像演示。")
    img = np.random.rand(512, 512)

if img.ndim == 3:
    img = img.mean(axis=2)   # RGB -> Gray

# 归一化
img = (img - img.min()) / (img.max() - img.min() + 1e-8)

H, W = img.shape
# 裁剪图像以适配 Patch 大小
Hc, Wc = H // PATCH_SIZE * PATCH_SIZE, W // PATCH_SIZE * PATCH_SIZE
img = img[:Hc, :Wc]

# 切块
patches = view_as_blocks(img, (PATCH_SIZE, PATCH_SIZE))
Ny, Nx = patches.shape[:2]
patch_grid = patches.reshape(Ny, Nx, PATCH_SIZE, PATCH_SIZE)

print(f"Image Shape: {img.shape}")
print(f"Patch Size: {PATCH_SIZE}x{PATCH_SIZE}")
print(f"Grid Layout: {Ny}x{Nx}")
print(f"Random Trials: {NUM_TRIALS}")

# =============================
# 2. 定义特征提取器 (适配 32x32)
# =============================

# (1) 像素域 (全量素)
pixel_extractor = lambda p: p.flatten()

# (2) DC 分量 (左上角 1x1)
dc_extractor = lambda p: np.array([dct2(p)[0, 0]])

# (3) 低频 DCT 
# Patch变大为32，我们将低频范围扩大到左上角 8x8 (保持比例，捕捉结构)
lf_extractor = lambda p: dct2(p)[:8, :8].flatten()

# (4) 高频 DCT 
# 提取右下角 8x8 (捕捉高频纹理/噪声)
hf_extractor = lambda p: dct2(p)[-8:, -8:].flatten()

# =============================
# 3. 计算相关性 (蒙特卡洛模拟)
# =============================
print("Calculating correlations...")

corr_pixel = monte_carlo_corr_map(patch_grid, pixel_extractor, NUM_TRIALS, MAX_SHIFT)
corr_dc    = monte_carlo_corr_map(patch_grid, dc_extractor,    NUM_TRIALS, MAX_SHIFT)
corr_lf    = monte_carlo_corr_map(patch_grid, lf_extractor,    NUM_TRIALS, MAX_SHIFT)
corr_hf    = monte_carlo_corr_map(patch_grid, hf_extractor,    NUM_TRIALS, MAX_SHIFT)

# =============================
# 4. 可视化
# =============================
plt.figure(figsize=(16, 4))

# 设置统一的字体和样式
plt.rcParams.update({'font.size': 10})

# --- 图 1: Pixels ---
plt.subplot(1, 4, 1)
plt.imshow(corr_pixel, cmap="coolwarm", vmin=0, vmax=1)
plt.colorbar(fraction=0.046, pad=0.04)
plt.title(f"1. Pixel Space\n(Patch {PATCH_SIZE}x{PATCH_SIZE})")
plt.xlabel("Δx")
plt.ylabel("Δy")

# --- 图 2: DC Only ---
plt.subplot(1, 4, 2)
plt.imshow(corr_dc, cmap="coolwarm", vmin=0, vmax=1)
plt.colorbar(fraction=0.046, pad=0.04)
plt.title("2. DC Component\n(Global Brightness)")
plt.xlabel("Δx")
plt.ylabel("Δy")

# --- 图 3: Low-Freq DCT ---
plt.subplot(1, 4, 3)
plt.imshow(corr_lf, cmap="coolwarm", vmin=0.2, vmax=1) 
plt.colorbar(fraction=0.046, pad=0.04)
plt.title("3. Low-Freq (8x8)\n(Structure)")
plt.xlabel("Δx")
plt.ylabel("Δy")

# --- 图 4: High-Freq DCT ---
plt.subplot(1, 4, 4)
# 高频通常相关性很低，范围设小一点便于观察微弱相关性
plt.imshow(corr_hf, cmap="coolwarm", vmin=-0.2, vmax=0.5)
plt.colorbar(fraction=0.046, pad=0.04)
plt.title("4. High-Freq (8x8)\n(Texture/Noise)")
plt.xlabel("Δx")
plt.ylabel("Δy")

plt.suptitle(f"Spatial Correlation Analysis (Random {NUM_TRIALS} Patches Averaged)", fontsize=14, y=1.05)
plt.tight_layout()
plt.show()