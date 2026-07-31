import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from tqdm import tqdm

# 自定义模块
from prior_cs.algorithms.omp import omp
from prior_cs.utils.normalize import normalize_columns
from prior_cs.algorithms.pinv_psi_omp import pinv_omp

# ================= 实验配置 =================
digit_list = [1, 3, 5, 7, 9]

H, W = 28, 28
n = H * W

ratio = 0.2
m = int(ratio * n)
k_sparsity = 50
TRIALS = 300

# ================= 主循环：每个 digit 一次完整实验 =================
for digit in digit_list:

    print(f"\n==============================")
    print(f"   开始实验：Digit = {digit}")
    print(f"==============================")

    # ========== 1. 数据加载 ==========
    data_path = f"../../prior_cs/data/mniset/data/mniset_digits/X_digit_{digit}.npy"
    X_full_spatial = np.load(data_path)

    TOTAL_AVAILABLE = X_full_spatial.shape[1]
    print(f"数据量: {TOTAL_AVAILABLE}")

    # ========== 2. 划分训练 / 测试 ==========
    TEST_NUM = min(int(TOTAL_AVAILABLE * 0.1), 200)
    TRAIN_POOL_SIZE = TOTAL_AVAILABLE - TEST_NUM

    X_test_spatial = X_full_spatial[:, -TEST_NUM:]
    X_train_pool_dct = dct(X_full_spatial[:, :-TEST_NUM], axis=0, norm='ortho')
    X_test_dct = dct(X_test_spatial, axis=0, norm='ortho')

    print(f"训练池: {TRAIN_POOL_SIZE}, 测试集: {TEST_NUM}")

    # ========== 3. 训练规模列表 ==========
    train_size_list = np.geomspace(20, TRAIN_POOL_SIZE, num=10, dtype=int)
    train_size_list = np.unique(train_size_list)

    print("训练规模列表:", train_size_list)

    # ========== 4. 实验循环 ==========
    psnr_classic_avg = []
    psnr_pinv_avg = []
    psnr_pinv_std = []

    print(f">>> 固定压缩率={ratio}, m={m}")

    for train_size in tqdm(train_size_list):

        X_current_train = X_train_pool_dct[:, :train_size]

        temp_psnr_c = []
        temp_psnr_p = []

        for _ in range(TRIALS):

            # 随机测量矩阵
            Phi = np.random.randn(m, n)
            Phi = normalize_columns(Phi)

            # 随机测试样本
            idx = np.random.randint(0, TEST_NUM)
            x_true_spatial = X_test_spatial[:, idx]
            x_true_sparse = X_test_dct[:, idx]

            # 观测
            y = Phi @ x_true_sparse

            # Classic OMP
            coef_c = omp(Phi, y, k=k_sparsity)
            rec_c = idct(coef_c, norm='ortho', axis=0)

            # Pinv OMP
            coef_p = pinv_omp(Phi, y, X_current_train, k=k_sparsity)
            rec_p = idct(coef_p, norm='ortho', axis=0)

            # PSNR
            mse_c = np.mean((x_true_spatial - rec_c) ** 2) + 1e-10
            mse_p = np.mean((x_true_spatial - rec_p) ** 2) + 1e-10
            max_val = np.max(np.abs(x_true_spatial))

            temp_psnr_c.append(10 * np.log10(max_val**2 / mse_c))
            temp_psnr_p.append(10 * np.log10(max_val**2 / mse_p))

        psnr_classic_avg.append(np.mean(temp_psnr_c))
        psnr_pinv_avg.append(np.mean(temp_psnr_p))
        psnr_pinv_std.append(np.std(temp_psnr_p))

    # ========== 5. 绘图 ==========
    plt.figure(figsize=(10, 6))

    plt.plot(
        train_size_list,
        psnr_classic_avg,
        'b--o',
        label='Classic OMP',
        alpha=0.7
    )

    plt.errorbar(
        train_size_list,
        psnr_pinv_avg,
        yerr=psnr_pinv_std,
        fmt='r-s',
        capsize=5,
        label='Pinv OMP'
    )

    plt.xscale('log')
    plt.xlabel('Number of Training Samples (log scale)')
    plt.ylabel('Average PSNR (dB)')
    plt.title(f'Digit {digit}: Effect of Prior Knowledge Size')
    plt.grid(True, which="both", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
