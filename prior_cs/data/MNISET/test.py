import numpy as np
import matplotlib.pyplot as plt
from scipy.datasets import face
from scipy.fftpack import dct
from skimage.util import view_as_blocks

# -----------------------------
# 1. 读取并预处理图像
# -----------------------------
img = face(gray=True).astype(np.float32)
img /= 255.0  # 归一化

H, W = img.shape
print("Image shape:", img.shape)

# 裁剪成 8 的整数倍
Hc, Wc = H // 8 * 8, W // 8 * 8
img = img[:Hc, :Wc]

# -----------------------------
# 2. 划分 8x8 patches
# -----------------------------
patches = view_as_blocks(img, block_shape=(8, 8))
# patches shape: (num_y, num_x, 8, 8)
Ny, Nx = patches.shape[:2]
patches = patches.reshape(-1, 8, 8)
print("Number of patches:", patches.shape[0])

# -----------------------------
# 3. 证明 patch 内像素强相关
# -----------------------------
# 把每个 patch 拉平成 64 维
X = patches.reshape(len(patches), -1)

# 像素协方差矩阵
cov_pixels = np.cov(X, rowvar=False)

plt.figure(figsize=(5, 4))
plt.imshow(cov_pixels, cmap="hot")
plt.colorbar()
plt.title("Covariance between pixels inside 8x8 patches")
plt.tight_layout()
plt.show()

# -----------------------------
# 4. 相邻 patch 之间的相关性
# -----------------------------
# 取水平方向相邻 patch
left = []
right = []

for y in range(Ny):
    for x in range(Nx - 1):
        p1 = patches[y * Nx + x].flatten()
        p2 = patches[y * Nx + x + 1].flatten()
        left.append(p1)
        right.append(p2)

left = np.array(left)
right = np.array(right)

# 计算平均相关系数
corrs = [np.corrcoef(left[i], right[i])[0, 1] for i in range(len(left))]
print("Mean correlation between adjacent patches:", np.mean(corrs))

# -----------------------------
# 5. 对所有 patch 做 DCT
# -----------------------------
def dct2(block):
    return dct(dct(block.T, norm="ortho").T, norm="ortho")

dct_patches = np.array([dct2(p) for p in patches])

# 计算 DCT 系数的平均能量
energy = np.mean(dct_patches ** 2, axis=0)

plt.figure(figsize=(4, 4))
plt.imshow(np.log(energy + 1e-6), cmap="inferno")
plt.colorbar()
plt.title("Mean DCT energy (log scale)")
plt.tight_layout()
plt.show()

# -----------------------------
# 6. 稀疏性定量证明
# -----------------------------
total_energy = np.sum(energy)
low_freq_energy = np.sum(energy[:4, :4])
print("Energy in top-left 4x4:", low_freq_energy / total_energy)

# -----------------------------
# 7. 对照实验：随机噪声
# -----------------------------
noise = np.random.randn(Hc, Wc).astype(np.float32)
noise_patches = view_as_blocks(noise, (8, 8)).reshape(-1, 8, 8)
noise_dct = np.array([dct2(p) for p in noise_patches])
noise_energy = np.mean(noise_dct ** 2, axis=0)

plt.figure(figsize=(4, 4))
plt.imshow(np.log(noise_energy + 1e-6), cmap="inferno")
plt.colorbar()
plt.title("Noise DCT energy (log scale)")
plt.tight_layout()
plt.show()
