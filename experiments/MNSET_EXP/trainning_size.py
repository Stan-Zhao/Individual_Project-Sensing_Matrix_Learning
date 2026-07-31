import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from tqdm import tqdm  # 强烈建议安装 tqdm: pip install tqdm

# 自定义模块
from prior_cs.algorithms.omp import omp
from prior_cs.utils.normalize import normalize_columns
from prior_cs.algorithms.pinv_psi_omp import pinv_omp

# ================= 1. 数据全自动加载 =================
print("正在加载全量数据...")
X_full_spatial = np.load("../../prior_cs/data/mniset/X_train.npy")

# 获取真实维度
TOTAL_AVAILABLE = X_full_spatial.shape[1]
H, W = 28, 28
n = H * W

print(f"检测到数据总量: {TOTAL_AVAILABLE} 张, 维度: {n}")

# ================= 2. 划分数据集 =================
# 策略：预留固定的 10% 或者 至少 200 张做测试，剩下的全做训练池
TEST_NUM = min(int(TOTAL_AVAILABLE * 0.1), 200) 
TRAIN_POOL_SIZE = TOTAL_AVAILABLE - TEST_NUM

X_test_spatial = X_full_spatial[:, -TEST_NUM:]
# 训练池数据转换到 DCT 域 (因为 Pinv-OMP 需要学习的是稀疏系数的协方差)
X_train_pool_dct = dct(X_full_spatial[:, :-TEST_NUM], axis=0, norm='ortho')
X_test_dct = dct(X_test_spatial, axis=0, norm='ortho')

print(f"训练池大小: {TRAIN_POOL_SIZE}, 测试集大小: {TEST_NUM}")

# ================= 3. 自适应生成实验参数 =================
# 生成 train_size_list
train_size_list = np.geomspace(20, TRAIN_POOL_SIZE, num=10, dtype=int)
# 去重并排序，防止最后几个数重复
train_size_list = np.unique(train_size_list)

print(f"生成的训练规模列表: {train_size_list}")

# 固定其他参数
ratio = 0.2
m = int(ratio * n)
k_sparsity = 50
TRIALS = 200  # 每个点重复实验次数

# ================= 4. 实验循环 =================
psnr_classic_avg = []
psnr_pinv_avg = []
psnr_pinv_std = []

print(f"\n>>> 开始实验：固定压缩率={ratio}, m={m} <<<")

for train_size in tqdm(train_size_list):
    # 从池中截取当前需要的样本数
    X_current_train = X_train_pool_dct[:, :train_size]
    
    temp_psnr_c = []
    temp_psnr_p = []
    
    for t in range(TRIALS):
        # 4.1 随机测量矩阵 (每次实验都变)
        Phi = np.random.randn(m, n)
        Phi = normalize_columns(Phi)
        
        # 4.2 随机选一个测试样本
        idx = np.random.randint(0, TEST_NUM)
        x_true_spatial = X_test_spatial[:, idx]
        x_true_sparse = X_test_dct[:, idx] # 用于生成 y
        
        # 4.3 观测
        y = Phi @ x_true_sparse
        
        # 4.4 恢复 - Classic OMP
        coef_c = omp(Phi, y, k=k_sparsity)
        rec_c = idct(coef_c, norm='ortho', axis=0)
        
        # 4.5 恢复 - Pinv OMP (使用当前的 X_current_train)
        coef_p = pinv_omp(Phi, y, X_current_train, k=k_sparsity)
        rec_p = idct(coef_p, norm='ortho', axis=0)
        
        # 4.6 计算 PSNR
        # 加上 1e-10 防止 MSE=0 报错
        mse_c = np.mean((x_true_spatial - rec_c)**2) + 1e-10
        mse_p = np.mean((x_true_spatial - rec_p)**2) + 1e-10
        
        max_val = np.max(np.abs(x_true_spatial))
        temp_psnr_c.append(10 * np.log10(max_val**2 / mse_c))
        temp_psnr_p.append(10 * np.log10(max_val**2 / mse_p))
    
    psnr_classic_avg.append(np.mean(temp_psnr_c))
    psnr_pinv_avg.append(np.mean(temp_psnr_p))
    psnr_pinv_std.append(np.std(temp_psnr_p))

# ================= 5. 绘图 =================
plt.figure(figsize=(10, 6))

# 画基准线
plt.plot(train_size_list, psnr_classic_avg, 'b--o', label='Classic OMP (Baseline)', alpha=0.7)

# 画带误差棒的 Pinv OMP
plt.errorbar(train_size_list, psnr_pinv_avg, yerr=psnr_pinv_std, 
             fmt='r-s', capsize=5, elinewidth=1, markeredgewidth=1, label='Pinv OMP (Proposed)')

plt.xscale('log') # 使用对数坐标轴，因为 train_size 是指数增长的
plt.xlabel('Number of Training Samples (Log Scale)')
plt.ylabel('Average PSNR (dB)')
plt.title(f'Effect of Prior Knowledge Size (Total Data={TOTAL_AVAILABLE})')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()