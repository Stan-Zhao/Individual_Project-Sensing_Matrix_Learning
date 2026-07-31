import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# ==================== 导入你的算法 ====================
sys.path.append('/Users/stanzhao/Desktop/prior_cs')
from experiments.S_curve.schnass import get_schnass_sensing_dictionary_pocs
from prior_cs.algorithms.psi_fast import design_pinv_psi_fast

# ==================== 1. 辅助函数 ====================
def normalize_columns(A):
    return A / (np.linalg.norm(A, axis=0) + 1e-10)

def generate_structured_power_law(N, K, alpha=1.0):
    x = np.zeros(N)
    vals = (np.arange(1, K + 1) ** (-alpha))
    vals *= np.sign(np.random.randn(K))
    x[:K] = vals
    return x

def generate_train_data(N, samples, k_max, alpha=1.0):
    X = np.zeros((N, samples))
    for i in range(samples):
        k = np.random.randint(5, k_max + 1)
        X[:, i] = generate_structured_power_law(N, k, alpha)
    return X

# ==================== 2. 生成矩阵 ====================
np.random.seed(42)

N, M = 256, 64
K_FIXED = 30
TAU_OPT = 1

# 1. 生成物理测量矩阵 Phi (Baseline)
Phi = np.random.randn(M, N)
Phi = normalize_columns(Phi)

# 2. 生成 Schnass 传感矩阵 Psi_s
print("Calculating Schnass dictionary...")
Psi_s = get_schnass_sensing_dictionary_pocs(Phi, max_iter=20)
Psi_s = normalize_columns(Psi_s) # 规范化以便于可视化对比

# 3. 生成 Proposed 传感矩阵 Psi_p
print("Calculating Proposed dictionary...")
X_train = generate_train_data(N, 500, K_FIXED + 10, alpha=1.0)
Psi_p = design_pinv_psi_fast(Phi, X_train)
Psi_p = normalize_columns(Psi_p) # 规范化

# ==================== 3. 计算伪格拉姆矩阵 (绝对值) ====================
# G = |Psi^T * Phi|，反映了原子间的互相干性 (Mutual Coherence)
G_baseline = np.abs(Phi.T @ Phi)
G_schnass  = np.abs(Psi_s.T @ Phi)
G_proposed = np.abs(Psi_p.T @ Phi)

# ==================== 4. 学术级可视化 ====================
print("Plotting...")
plt.rcParams.update({
    'font.size': 13,
    'font.weight': 'bold',
    'axes.titleweight': 'bold'
})

fig = plt.figure(figsize=(15, 5))

# 使用 ImageGrid 保证三张矩阵图绝对等大，且共享一个完美的 Colorbar
grid = ImageGrid(fig, 111,
                 nrows_ncols=(1, 3),
                 axes_pad=0.3,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="5%",
                 cbar_pad=0.15)

# 为了让对比更强烈，截断最大显示值为 0.3 (因为对角线是 1，不截断的话背景会全黑)
VMAX = 0.3  

# 1. Baseline
im = grid[0].imshow(G_baseline, cmap='inferno', vmin=0, vmax=VMAX)
grid[0].set_title(r"Baseline: $|\Phi^T \Phi|$")

# 2. Schnass
grid[1].imshow(G_schnass, cmap='inferno', vmin=0, vmax=VMAX)
grid[1].set_title(r"Schnass: $|\Psi_s^T \Phi|$")

# 3. Proposed
grid[2].imshow(G_proposed, cmap='inferno', vmin=0, vmax=VMAX)
grid[2].set_title(r"Proposed: $|\Psi_p^T \Phi|$")

# 统一去掉坐标轴刻度，因为它们只是抽象的索引
for ax in grid:
    ax.set_xticks([])
    ax.set_yticks([])

# 添加 Colorbar
cbar = grid.cbar_axes[0].colorbar(im)
cbar.ax.set_title("Coherence", fontsize=10, pad=10)

plt.savefig("sensing_matrices_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

print("[Done] 图像已保存为 sensing_matrices_comparison.png")