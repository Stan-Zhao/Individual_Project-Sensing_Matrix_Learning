import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

from prior_cs.algorithms.omp import omp
from prior_cs.utils.normalize import normalize_columns
from prior_cs.algorithms.pinv_psi_omp import pinv_omp

# ================= 配置参数 =================
TOTAL_N = 2000
TEST_NUM = 100
TRIALS = 200
H, W = 28, 28
n = H * W
ratio = 0.2
m = int(ratio * n)

# 变量：稀疏度 K
# 理论最佳通常在 m/4 到 m/2 之间，我们扫一个范围
k_list = [5,10, 30, 50, 70, 90, 110, 130, 150]

# ================= 数据准备 =================
print("Loading data for Sparsity Experiment...")
X_full = np.load("../../prior_cs/data/mniset/X_train.npy")
X_full = X_full[:, :TOTAL_N]

X_dct_full = dct(X_full, axis=0, norm='ortho')
X_test_dct = X_dct_full[:, -TEST_NUM:]
X_test_spatial = X_full[:, -TEST_NUM:]
X_train_dct = X_dct_full[:, :-TEST_NUM]

# ================= 主循环 =================
psnr_classic_avg = []
psnr_pinv_avg = []

print(f"Start Experiment 3: Effect of Sparsity Parameter K (m={m})")

for k_val in k_list:
    if k_val >= m:
        print(f"Skipping k={k_val} as it approaches/exceeds m={m}")
        break

    curr_c = []
    curr_p = []
    
    for t in range(TRIALS):
        # 1. 生成 Phi
        Phi = np.random.randn(m, n)
        Phi = normalize_columns(Phi)
        
        # 2. 随机测试样本
        idx = np.random.randint(0, TEST_NUM)
        x_true_sparse = X_test_dct[:, idx]
        x_true_spatial = X_test_spatial[:, idx]
        
        # 3. 观测
        y = Phi @ x_true_sparse
        
        # 4. 恢复 (使用当前的 k_val)
        coef_c = omp(Phi, y, k=k_val)
        coef_p = pinv_omp(Phi, y, X_train_dct, k=k_val)
        
        rec_c = idct(coef_c, norm='ortho', axis=0)
        rec_p = idct(coef_p, norm='ortho', axis=0)
        
        # 5. PSNR
        mse_c = np.mean((x_true_spatial - rec_c)**2)
        mse_p = np.mean((x_true_spatial - rec_p)**2)
        
        # 简单数值保护
        mse_c = mse_c if mse_c > 1e-10 else 1e-10
        mse_p = mse_p if mse_p > 1e-10 else 1e-10

        curr_c.append(10 * np.log10(np.max(x_true_spatial)**2 / mse_c))
        curr_p.append(10 * np.log10(np.max(x_true_spatial)**2 / mse_p))
    
    psnr_classic_avg.append(np.mean(curr_c))
    psnr_pinv_avg.append(np.mean(curr_p))
    print(f"Sparsity K: {k_val} | Classic: {psnr_classic_avg[-1]:.2f}dB | Pinv: {psnr_pinv_avg[-1]:.2f}dB")

# ================= 绘图 =================
plt.figure(figsize=(8, 6))
plt.plot(k_list[:len(psnr_classic_avg)], psnr_classic_avg, 'b-o', label='Baseline (Gaussian)')
plt.plot(k_list[:len(psnr_pinv_avg)], psnr_pinv_avg, 'r-s', label='Proposed Method')
plt.xlabel('Sparsity Parameter (K)')
plt.ylabel('Average PSNR (dB)')
plt.title(f'Effect of Sparsity Parameter (m={m})')
plt.grid(True)
plt.legend()
plt.show()