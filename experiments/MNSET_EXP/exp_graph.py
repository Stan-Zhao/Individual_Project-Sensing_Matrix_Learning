import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from scipy.linalg import pinv, norm

# 假设这些是你原本库里的引用，为了让代码在任何地方都能跑，
# 我会在下面手写一个简单的 omp_with_sensing 替代它们
from prior_cs.algorithms.omp import omp
from prior_cs.utils.normalize import normalize_columns
from prior_cs.algorithms.psi_fast import design_pinv_psi_fast
from prior_cs.algorithms.pinv_psi_omp import pro_omp_solve

np.random.seed(42)

# ===============================
# 0. 辅助函数定义 (新增部分)
# ===============================

def compute_mu_bound(M, N):
    """计算论文理论下界 mu"""
    if M >= N: return 0
    return np.sqrt((N - M) / (M * (N - 1)))

def get_psi_paper_pocs(Phi, max_iter=50, tol=1e-6):
    """
    [新增] 使用 POCS 算法计算论文中的 Sensing Matrix
    参考文献: Schnass & Vandergheynst, "Dictionary Preconditioning for Greedy Algorithms"
    """
    M, N = Phi.shape
    mu = compute_mu_bound(M, N)
    
    # 预计算伪逆和投影算子
    Phi_pinv = pinv(Phi)
    P_G = Phi_pinv @ Phi 
    
    # 初始化 G (Gram Matrix)
    G = Phi.T @ Phi
    
    for k in range(max_iter):
        G_prev = G.copy()
        
        # --- Step 1: Project onto H (Ideal Gram Matrix) ---
        H = G.copy()
        np.fill_diagonal(H, 1.0)
        # 截断非对角元素
        mask = ~np.eye(N, dtype=bool)
        H[mask] = np.clip(H[mask], -mu, mu)
        
        # --- Step 2: Project onto G (Realizable Gram Matrices) ---
        G = H @ P_G
        
        if np.linalg.norm(G - G_prev, 'fro') < tol:
            break
            
    # 计算 Psi: Psi^T = H @ Phi_pinv => Psi = (Phi_pinv)^T @ H^T
    # 注意：这里 H 应该是近似对称的
    Psi = (H @ Phi_pinv).T
    return normalize_columns(Psi)


# ===============================
# 1. 数据准备
# ===============================
# 注意：请确保路径正确，或者替换为你的数据加载逻辑
try:
    X_spatial_train = np.load("../../prior_cs/data/mniset/X_train.npy") 
    # 如果是 (N, 28, 28) 这种格式，需要拉平
    if X_spatial_train.ndim == 3:
        X_spatial_train = X_spatial_train.reshape(X_spatial_train.shape[0], -1).T
except FileNotFoundError:
    print("未找到数据文件，生成随机数据用于演示...")
    X_spatial_train = np.random.randn(784, 1000)

# 转到 DCT 域
X_dct_train = dct(X_spatial_train, axis=0, norm='ortho')

H, W = 28, 28
n = H * W
sample_idx = 0 # 测试样本索引

# 真正的信号
x_true_spatial = X_spatial_train[:, sample_idx]
x_true_sparse = X_dct_train[:, sample_idx]

# 训练集准备
X_train_clean = np.delete(X_dct_train, sample_idx, axis=1)

# 压缩测量
compression_ratio = 0.2
m = int(compression_ratio * n)
Phi = np.random.randn(m, n)
Phi = normalize_columns(Phi)

y = Phi @ x_true_sparse

# ===============================
# 2. 核心优化：去均值策略
# ===============================

# [Step 1] 计算均值
mu_dct = np.mean(X_train_clean, axis=1) 

# [Step 2] 去中心化测量值
y_centered = y - Phi @ mu_dct

# [Step 3] 恢复
sparsity = 30

# --- A. Classic OMP (Centered) ---
# 使用标准 OMP (Psi = Phi)
dx_hat_classic = pro_omp_solve(Phi, Psi=Phi, y=y_centered, sparsity=sparsity)
x_hat_classic = idct(dx_hat_classic + mu_dct, norm='ortho', axis=0)

# --- B. Paper's Method (POCS Sensing Matrix) ---
print("计算 Paper 的 Sensing Matrix (POCS)...")
Psi_paper = get_psi_paper_pocs(Phi)
# 使用 Psi_paper 进行原子选择
dx_hat_paper = pro_omp_solve(Phi, Psi=Psi_paper, y=y_centered, sparsity=sparsity)
x_hat_paper = idct(dx_hat_paper + mu_dct, norm='ortho', axis=0)

# --- C. Pinv OMP (Centered / Proposed) ---
# 这里模拟你的 pinv_omp 逻辑，本质是 MMSE Sensing Matrix
# Psi_proposed = (Phi R_x Phi^T + lambda I)^-1 Phi R_x
Psi_proposed = design_pinv_psi_fast(Phi, X_train_clean)
dx_hat_pinv = pro_omp_solve(Phi, Psi=Psi_proposed, y=y_centered, sparsity=sparsity)
x_hat_pinv = idct(dx_hat_pinv + mu_dct, norm='ortho', axis=0)

# ===============================
# 3. PSNR 评价
# ===============================
def psnr(x_true, x_rec):
    mse = np.mean((x_true - x_rec) ** 2)
    if mse == 0: return 100
    return 10 * np.log10(np.max(np.abs(x_true))**2 / mse)

psnr_classic = psnr(x_true_spatial, x_hat_classic)
psnr_paper = psnr(x_true_spatial, x_hat_paper)
psnr_pinv = psnr(x_true_spatial, x_hat_pinv)

print(f"PSNR Classic OMP: {psnr_classic:.2f} dB")
print(f"PSNR Paper OMP:   {psnr_paper:.2f} dB")
print(f"PSNR Pinv OMP:    {psnr_pinv:.2f} dB")

# ===============================
# 4. 可视化
# ===============================
# 【新增】全局字体加粗、加大设置
plt.rcParams.update({
    'font.size': 12,              # 全局基础字体大小
    'font.weight': 'bold',        # 全局字体加粗
    'axes.titleweight': 'bold',   # 子图标题加粗
    'axes.titlesize': 16          # 子图标题大小
})

plt.figure(figsize=(16, 5))

# 原图
plt.subplot(1, 4, 1)
plt.imshow(x_true_spatial.reshape(H, W), cmap='gray')
plt.title("Original")
plt.axis('off')

# Classic
plt.subplot(1, 4, 2)
plt.imshow(x_hat_classic.reshape(H, W), cmap='gray')
plt.title(f"Baseline\nPSNR={psnr_classic:.2f} dB")
plt.axis('off')

# Paper
plt.subplot(1, 4, 3)
plt.imshow(x_hat_paper.reshape(H, W), cmap='gray')
plt.title(f"Schnass Method\nPSNR={psnr_paper:.2f} dB")
plt.axis('off')

# Proposed (Pinv)
plt.subplot(1, 4, 4)
plt.imshow(x_hat_pinv.reshape(H, W), cmap='gray')
plt.title(f"Proposed Method\nPSNR={psnr_pinv:.2f} dB")
plt.axis('off')

plt.tight_layout()
# 建议加上 dpi=300 保证输出图片高清
plt.savefig("reconstruction_comparison.png", dpi=300, bbox_inches='tight') 
plt.show()