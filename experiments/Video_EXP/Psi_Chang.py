import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct
import os
from prior_cs.algorithms.psi_fast import design_pinv_psi_fast
from prior_cs.utils.normalize import normalize_columns

# ==========================================
# 0. 基础函数
# ==========================================
def dct2(block):
    return dct(dct(block.T, norm="ortho").T, norm="ortho")


def get_temporal_prior_stack(frames_array, t, r, c, window=5):
    """获取过去 window 帧的堆叠先验"""
    if t < 1: return None
    actual_window = min(t, window)
    cols = []
    for dt in range(1, actual_window + 1):
        idx = t - dt
        patch = frames_array[idx, r, c]
        coef = dct2(patch).flatten()
        cols.append(coef)
    return np.array(cols).T


# ==========================================
# 1. 数据加载
# ==========================================
try:
    # 尝试加载真实数据
    full_data = np.load("VIDEO/video_patches.npy")
    print(f"[Info] 成功加载本地数据: {full_data.shape}")
    frames_array = full_data
except FileNotFoundError:
    print("[Info] 未找到数据，生成模拟数据...")
    frames_array = np.random.rand(50, 6, 6, 32, 32).astype(np.float32)
    # 模拟平滑运动
    for t in range(1, 50):
        frames_array[t] = frames_array[t-1] * 0.98 + np.random.normal(0, 0.005, frames_array[0].shape)

num_frames, Ny, Nx, P, _ = frames_array.shape
n = P**2
m = int(0.25 * n) # Ratio 0.25

# ==========================================
# 2. 实验参数
# ==========================================
num_trials = 50  # 设置实验次数

# 选择图像中心的一个 Patch 进行追踪
r_test, c_test = Ny//2, Nx//2
window_size = 5  # 使用 5 帧的历史窗口

# 存储所有实验的结果
all_psi_diffs = []      # 存储相对变化率
all_cosine_sims = []    # 存储余弦相似度

# ==========================================
# 3. 多次实验
# ==========================================
print(f"\n[Analysing] 正在进行 {num_trials} 次实验，每次使用不同的 Phi 矩阵...")
print(f"窗口大小 (History Window): {window_size}")

for trial in range(num_trials):
    print(f"\n[Trial {trial + 1}] 正在执行实验...")

    psi_diffs = []      # 存储相对变化率
    cosine_sims = []    # 存储余弦相似度
    prev_Psi = None

    # 为当前实验生成一个新的 Phi 矩阵
    Phi = normalize_columns(np.random.randn(m, n))

    # 遍历前 50 帧 (或更少)
    limit_frames = min(50, num_frames)

    for t in range(1, limit_frames):
        # 1. 获取先验堆叠 (Stack)
        X_stack = get_temporal_prior_stack(frames_array, t, r_test, c_test, window=window_size)

        # 3. 构造 Psi
        Psi_curr = design_pinv_psi_fast(Phi, X_stack)

        # 4. 计算与上一时刻的差异
        if prev_Psi is not None:
            # Frobenius 范数差异
            diff_norm = np.linalg.norm(Psi_curr - prev_Psi, 'fro')
            prev_norm = np.linalg.norm(prev_Psi, 'fro')

            # 相对变化率 (Relative Change)
            rel_change = diff_norm / (prev_norm + 1e-10)

            # 余弦相似度 (Cosine Similarity) - 将矩阵展平为向量计算
            vec1 = prev_Psi.flatten()
            vec2 = Psi_curr.flatten()
            cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)

            psi_diffs.append(rel_change)
            cosine_sims.append(cos_sim)

        prev_Psi = Psi_curr

    # 记录该实验的结果
    all_psi_diffs.append(psi_diffs)
    all_cosine_sims.append(cosine_sims)

# ==========================================
# 4. 计算平均结果
# ==========================================
# 计算每个实验的平均相对变化率和平均相似度
avg_psi_diffs = np.mean(all_psi_diffs, axis=0)
avg_cosine_sims = np.mean(all_cosine_sims, axis=0)

# 计算所有实验的平均值
avg_change = np.mean(avg_psi_diffs)
avg_sim = np.mean(avg_cosine_sims)

print(f"\n[Result] 平均相对变化率: {avg_change:.4f} (越小越稳定)")
print(f"[Result] 平均相似度:     {avg_sim:.4f} (越高越相似)")

# ==========================================
# 5. 可视化
# ==========================================
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# 图 1: 相对变化率
ax[0].plot(range(2, limit_frames), avg_psi_diffs, 'r-o', markersize=4, linewidth=1.5)
ax[0].set_title(f"Relative Change of $\Psi$ Matrix\n(Avg: {avg_change:.2f})")
ax[0].set_xlabel("Frame Index")
ax[0].set_ylabel("||$\Psi_t$ - $\Psi_{t-1}$|| / ||$\Psi_{t-1}$||")
ax[0].grid(True, linestyle='--', alpha=0.5)

# 图 2: 相似度
ax[1].plot(range(2, limit_frames), avg_cosine_sims, 'b-s', markersize=4, linewidth=1.5)
ax[1].set_title(f"Cosine Similarity ($\Psi_t$ vs $\Psi_{{t-1}}$)\n(Avg: {avg_sim:.2f})")
ax[1].set_xlabel("Frame Index")
ax[1].set_ylabel("Similarity (0~1)")
ax[1].set_ylim(0.5, 1.05) # 通常相似度很高，缩放坐标轴
ax[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
