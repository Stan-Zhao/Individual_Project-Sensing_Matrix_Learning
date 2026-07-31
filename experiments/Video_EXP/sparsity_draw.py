import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 0. 辅助函数：二维 DCT
# ==========================================
def dct2(block):
    """计算 2D DCT"""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

# ==========================================
# 1. 数据加载 
# ==========================================
try:
    frames_array = np.load("VIDEO/video_patches.npy")
    print(f"[Info] Loaded data shape: {frames_array.shape}")
except:
    print("[Info] Generating synthetic data for demonstration...")
    frames_array = np.zeros((10, 8, 8, 32, 32))
    x, y = np.meshgrid(np.linspace(-2, 2, 32), np.linspace(-2, 2, 32))
    base_pattern = np.exp(-(x**2 + y**2)) 
    
    for t in range(10):
        for i in range(8):
            for j in range(8):
                frames_array[t, i, j] = base_pattern * 0.8 + np.random.rand(32, 32) * 0.2
                
    frames_array = (frames_array - frames_array.min()) / (frames_array.max() - frames_array.min())

# ==========================================
# 2. 图像拼接与 DCT 变换
# ==========================================
frame_idx = 50
Ny, Nx, P, _ = frames_array.shape[1:]

full_image = np.zeros((Ny * P, Nx * P))
for r in range(Ny):
    for c in range(Nx):
        full_image[r*P:(r+1)*P, c*P:(c+1)*P] = frames_array[frame_idx, r, c]

image_dct = dct2(full_image)

# ==========================================
# 3. 完美复刻原图样式绘图 (字体加粗加大，解决重叠)
# ==========================================
# 全局设置刻度数字的大小和粗细
plt.rcParams.update({
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'font.weight': 'bold',
    'axes.labelweight': 'bold'
})

fig, axes = plt.subplots(1, 3, figsize=(16, 5)) 
# 【关键修复】增加 bottom 的预留空间，避免文字被切断
plt.subplots_adjust(bottom=0.25, wspace=0.25) 

# ------------------------------------------
# (a) Original Frame
# ------------------------------------------
ax1 = axes[0]
ax1.imshow(full_image, cmap='gray')
ax1.axis('off')
# 【关键修复】将 y 坐标从 -0.12 改为 -0.25 下移，增大字号并加粗
ax1.text(0.5, -0.25, "(a) Original MRI Frame", 
         transform=ax1.transAxes, ha='center', va='top', 
         fontsize=14, fontweight='bold')

# ------------------------------------------
# (b) Spatial Domain Histogram
# ------------------------------------------
ax2 = axes[1]
ax2.hist(full_image.flatten(), bins=100, color='blue', alpha=0.75, edgecolor='none')
ax2.set_yscale('log')
# 轴标签加大加粗
ax2.set_ylabel('Count (Log Scale)', fontsize=13, fontweight='bold')
ax2.set_xlabel('Pixel Intensity', fontsize=13, fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.4)
# 【关键修复】同步下移，加大加粗
ax2.text(0.5, -0.25, "(b) Spatial Domain Histogram", 
         transform=ax2.transAxes, ha='center', va='top', 
         fontsize=14, fontweight='bold')

# ------------------------------------------
# (c) DCT Coefficients Histogram
# ------------------------------------------
ax3 = axes[2]
dct_flat = image_dct.flatten()
p_low, p_high = np.percentile(dct_flat, [0.05, 99.95]) 
filtered_dct = dct_flat[(dct_flat >= p_low) & (dct_flat <= p_high)]

ax3.hist(filtered_dct, bins=100, color='red', alpha=0.75, edgecolor='none')
ax3.set_yscale('log') 
# 轴标签加大加粗
ax3.set_xlabel('Coefficient Value', fontsize=13, fontweight='bold')
ax3.grid(True, linestyle='--', alpha=0.4)
# 【关键修复】同步下移，加大加粗
ax3.text(0.5, -0.25, "(c) DCT Coefficients Histogram", 
         transform=ax3.transAxes, ha='center', va='top', 
         fontsize=14, fontweight='bold')

# 保存高质量图片
plt.savefig("sparsity_comparison_bold.png", dpi=300, bbox_inches='tight')
plt.show()