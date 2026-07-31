import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import os
import warnings

# --- 自定义库引用 ---
from prior_cs.algorithms.omp import omp
from prior_cs.utils.normalize import normalize_columns
from prior_cs.algorithms.pinv_psi_omp import pinv_omp

warnings.filterwarnings('ignore')

# ==========================================
# 0. 全局配置参数
# ==========================================
TOTAL_N = 2000
TEST_NUM = 100
H, W = 28, 28
n = H * W

# 实验 1 (Ratio) 参数
TRIALS_RATIO = 100
ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# 实验 2 (Sparsity K) 参数
TRIALS_K = 200
fixed_ratio_for_k = 0.2
m_fixed = 156
k_list = [5, 10, 30, 50, 70, 90, 110, 130, 150]

# ==========================================
# 1. 数据统一准备 (共享数据，避免重复加载)
# ==========================================
print("Loading and preparing data for both experiments...")
data_path = "../../prior_cs/data/mniset/X_train.npy"
try:
    X_full = np.load(data_path)
    X_full = X_full[:, :TOTAL_N]
except FileNotFoundError:
    print(f"[Warning] 未找到数据文件 {data_path}，生成随机数据...")
    X_full = np.random.randn(n, TOTAL_N)

# 转到 DCT 域
X_dct_full = dct(X_full, axis=0, norm='ortho')

# 划分测试集与训练集 (训练集用于提取先验)
X_test_dct = X_dct_full[:, -TEST_NUM:]
X_test_spatial = X_full[:, -TEST_NUM:]
X_train_dct = X_dct_full[:, :-TEST_NUM]

print(f"Data ready! Train Size: {X_train_dct.shape[1]}, Test Size: {X_test_dct.shape[1]}")
print("-" * 50)

# ==========================================
# 2. 实验 A: 采样率 (Sampling Ratio) 的影响
# ==========================================
psnr_classic_ratio = []
psnr_pinv_ratio = []

print(">>> Start Experiment A: Effect of Sampling Ratio")

for r in ratio_list:
    m = int(r * n)
    k_sparsity = 50 # 稀疏度设置
    if k_sparsity < 1: k_sparsity = 1
    
    curr_c = []
    curr_p = []
    
    for t in range(TRIALS_RATIO):
        Phi = np.random.randn(m, n)
        Phi = normalize_columns(Phi)
        
        idx = np.random.randint(0, TEST_NUM)
        x_true_sparse = X_test_dct[:, idx]
        x_true_spatial = X_test_spatial[:, idx]
        
        y = Phi @ x_true_sparse
        
        coef_c = omp(Phi, y, k=k_sparsity)
        coef_p = pinv_omp(Phi, y, X_train_dct, k=k_sparsity)
        
        rec_c = idct(coef_c, norm='ortho', axis=0)
        rec_p = idct(coef_p, norm='ortho', axis=0)
        
        mse_c = max(np.mean((x_true_spatial - rec_c)**2), 1e-10)
        mse_p = max(np.mean((x_true_spatial - rec_p)**2), 1e-10)

        max_val2 = np.max(x_true_spatial)**2
        curr_c.append(10 * np.log10(max_val2 / mse_c))
        curr_p.append(10 * np.log10(max_val2 / mse_p))
    
    psnr_classic_ratio.append(np.mean(curr_c))
    psnr_pinv_ratio.append(np.mean(curr_p))
    print(f"  Ratio: {r} (m={m}, k={k_sparsity}) | Classic: {psnr_classic_ratio[-1]:.2f}dB | Proposed: {psnr_pinv_ratio[-1]:.2f}dB")

print("-" * 50)

# ==========================================
# 3. 实验 B: 稀疏度参数 K 的影响
# ==========================================
psnr_classic_k = []
psnr_pinv_k = []
valid_k_list = [] # 用于记录实际运行的 K 值 (防止 K >= m)

print(f">>> Start Experiment B: Effect of Sparsity Parameter K (Fixed m={m_fixed})")

for k_val in k_list:
    if k_val >= m_fixed:
        print(f"  [Skip] k={k_val} as it approaches/exceeds m={m_fixed}")
        break
    
    valid_k_list.append(k_val)
    curr_c = []
    curr_p = []
    
    for t in range(TRIALS_K):
        Phi = np.random.randn(m_fixed, n)
        Phi = normalize_columns(Phi)
        
        idx = np.random.randint(0, TEST_NUM)
        x_true_sparse = X_test_dct[:, idx]
        x_true_spatial = X_test_spatial[:, idx]
        
        y = Phi @ x_true_sparse
        
        coef_c = omp(Phi, y, k=k_val)
        coef_p = pinv_omp(Phi, y, X_train_dct, k=k_val)
        
        rec_c = idct(coef_c, norm='ortho', axis=0)
        rec_p = idct(coef_p, norm='ortho', axis=0)
        
        mse_c = max(np.mean((x_true_spatial - rec_c)**2), 1e-10)
        mse_p = max(np.mean((x_true_spatial - rec_p)**2), 1e-10)

        max_val2 = np.max(x_true_spatial)**2
        curr_c.append(10 * np.log10(max_val2 / mse_c))
        curr_p.append(10 * np.log10(max_val2 / mse_p))
    
    psnr_classic_k.append(np.mean(curr_c))
    psnr_pinv_k.append(np.mean(curr_p))
    print(f"  Sparsity K: {k_val} | Classic: {psnr_classic_k[-1]:.2f}dB | Proposed: {psnr_pinv_k[-1]:.2f}dB")

print("-" * 50)

# ==========================================
# 4. 联合绘图 (全局字体加粗、并排双图)
# ==========================================
print("Plotting results...")

# 【核心设置】全局字体加粗、加大
plt.rcParams.update({
    'font.size': 14,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# --- 子图 1: 压缩率实验 ---
axes[0].plot(ratio_list, psnr_classic_ratio, color='gray', linestyle='--', marker='o', label='Baseline (Gaussian)')
axes[0].plot(ratio_list, psnr_pinv_ratio, color='red', linestyle='-', marker='^', label='Proposed Method')
axes[0].set_xlabel('Compression Ratio')
axes[0].set_ylabel('Average PSNR (dB)')
axes[0].set_title('Reconstruction Quality vs. Sampling Ratio')
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend()

# --- 子图 2: 稀疏度 K 实验 ---
axes[1].plot(valid_k_list, psnr_classic_k, color='gray', linestyle='--', marker='o', label='Baseline (Gaussian)')
axes[1].plot(valid_k_list, psnr_pinv_k, color='red', linestyle='-', marker='^', label='Proposed Method')
axes[1].set_xlabel('Sparsity Parameter (K)')
axes[1].set_ylabel('Average PSNR (dB)')
axes[1].set_title(f'Effect of Sparsity Parameter')
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend()

# 调整布局并展示
plt.tight_layout(w_pad=3.0) # 强制推开两个图的间距防止重叠
plt.savefig("ablation_ratio_and_k.png", dpi=300, bbox_inches='tight')
plt.show()

print("[Done] Experiments finished and plot saved to 'ablation_ratio_and_k.png'.")