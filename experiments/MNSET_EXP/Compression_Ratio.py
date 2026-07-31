import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

from prior_cs.algorithms.omp import omp
from prior_cs.utils.normalize import normalize_columns
from prior_cs.algorithms.pinv_psi_omp import pinv_omp

# ================= 配置参数 =================
TOTAL_N = 2000
TEST_NUM = 100
TRIALS = 100
H, W = 28, 28
n = H * W

# 变量：压缩率 (M/N)
ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# ================= 数据准备 =================
print("Loading data for Ratio Experiment...")
X_full = np.load("../../prior_cs/data/mniset/X_train.npy")
X_full = X_full[:, :TOTAL_N]

X_dct_full = dct(X_full, axis=0, norm='ortho')
X_test_dct = X_dct_full[:, -TEST_NUM:]
X_test_spatial = X_full[:, -TEST_NUM:]
X_train_dct = X_dct_full[:, :-TEST_NUM] # 使用所有剩余数据作为训练

# ================= 主循环 =================
psnr_classic_avg = []
psnr_pinv_avg = []

print(f"Start Experiment 2: Effect of Sampling Ratio (Train Size={X_train_dct.shape[1]})")

for r in ratio_list:
    m = int(r * n)
    k_sparsity = 50 # 稀疏度通常随观测数线性增加
    if k_sparsity < 1: k_sparsity = 1
    
    curr_c = []
    curr_p = []
    
    for t in range(TRIALS):
        # 1. 生成对应维度的测量矩阵
        Phi = np.random.randn(m, n)
        Phi = normalize_columns(Phi)
        
        # 2. 随机测试样本
        idx = np.random.randint(0, TEST_NUM)
        x_true_sparse = X_test_dct[:, idx]
        x_true_spatial = X_test_spatial[:, idx]
        
        # 3. 观测
        y = Phi @ x_true_sparse
        
        # 4. 恢复
        coef_c = omp(Phi, y, k=k_sparsity)
        coef_p = pinv_omp(Phi, y, X_train_dct, k=k_sparsity)
        
        rec_c = idct(coef_c, norm='ortho', axis=0)
        rec_p = idct(coef_p, norm='ortho', axis=0)
        
        # 5. PSNR
        mse_c = np.mean((x_true_spatial - rec_c)**2)
        mse_p = np.mean((x_true_spatial - rec_p)**2)
        
        # 防止 MSE 为 0
        mse_c = mse_c if mse_c > 1e-10 else 1e-10
        mse_p = mse_p if mse_p > 1e-10 else 1e-10

        curr_c.append(10 * np.log10(np.max(x_true_spatial)**2 / mse_c))
        curr_p.append(10 * np.log10(np.max(x_true_spatial)**2 / mse_p))
    
    psnr_classic_avg.append(np.mean(curr_c))
    psnr_pinv_avg.append(np.mean(curr_p))
    print(f"Ratio: {r} (m={m}, k={k_sparsity}) | Classic: {psnr_classic_avg[-1]:.2f}dB | Pinv: {psnr_pinv_avg[-1]:.2f}dB")

# ================= 绘图 =================
plt.figure(figsize=(8, 6))
plt.plot(ratio_list, psnr_classic_avg, 'b-o', label='Baseline (Gaussian)')
plt.plot(ratio_list, psnr_pinv_avg, 'r-s', label='Proposed Method')
plt.xlabel('Sampling Ratio (M/N)')
plt.ylabel('Average PSNR (dB)')
plt.title('Reconstruction Quality vs. Sampling Ratio')
plt.grid(True)
plt.legend()
plt.show()