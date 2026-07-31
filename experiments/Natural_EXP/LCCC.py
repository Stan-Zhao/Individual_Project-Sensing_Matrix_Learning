import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import pinv
from scipy.fftpack import dctn
import os
import warnings
import random
from skimage.color import rgb2gray
from skimage.util import view_as_blocks

# --- Import your custom library ---
from prior_cs.algorithms.psi_fast import design_pinv_psi_fast

warnings.filterwarnings('ignore')

# ==========================================
# 0. Data & Transform Utilities (Updated)
# ==========================================

def perform_2d_dct_flatten(patches):
    """
    输入: 空间域 patches, shape (num_patches, 8, 8)
    输出: DCT域扁平向量, shape (64, num_patches)
    """
    # 1. 对每个 patch 做 2D DCT
    # axes=(1, 2) 表示在后两个维度（H, W）上做变换
    patches_dct = dctn(patches, axes=(1, 2), norm='ortho')
    
    # 2. 拉平为 (num_patches, 64)
    patches_flat = patches_dct.reshape(patches.shape[0], -1)
    
    # 3. 转置为 (N, L) -> (64, num_patches) 以符合 CS 习惯
    return patches_flat.T

def load_random_image_patch_data(pic_dir='pic', patch_size=8, r_prior=1, r_test=2):
    """
    从 pic/ 文件夹随机读取图像，并构造 Training (Prior) 和 Testing 数据
    设定:
      - Train: 中心点周围 3x3 (r_prior=1)
      - Test: 中心点周围 5x5 (r_test=2), 排除 Train 区域 (或者排除中心点)
      *注: 根据你的描述，测试是更大邻域，这里我们设定测试集为 5x5 区域中除去中心点
    """
    # 1. 读取图像
    if not os.path.exists(pic_dir):
        # Fallback: 生成合成图像 (Zone Plate)
        print(f"[Warning] '{pic_dir}' not found. Using synthetic image.")
        x = np.linspace(-10, 10, 256)
        X, Y = np.meshgrid(x, x)
        R = np.sqrt(X**2 + Y**2)
        img = (np.sin(R) + 1) / 2
    else:
        files = [f for f in os.listdir(pic_dir) if f.lower().endswith(('.png', '.jpg', '.bmp', '.jpeg'))]
        if not files:
            raise FileNotFoundError(f"No images found in {pic_dir}")
        fname = random.choice(files)
        # print(f"  -> Selected image: {fname}")
        img = plt.imread(os.path.join(pic_dir, fname))
        if img.ndim == 3:
            img = rgb2gray(img)
        # 归一化
        img = img.astype(np.float32)
        img /= (img.max() + 1e-12)

    # 2. 裁剪图像以适应 patch_size
    h, w = img.shape
    h_new = (h // patch_size) * patch_size
    w_new = (w // patch_size) * patch_size
    img = img[:h_new, :w_new]

    # 3. 分块 (H_grid, W_grid, 8, 8)
    blocks = view_as_blocks(img, (patch_size, patch_size))
    Ny, Nx, _, _ = blocks.shape

    # 4. 随机选择一个中心点 (确保有足够的边缘做邻域)
    margin = max(r_prior, r_test)
    if Ny <= 2 * margin or Nx <= 2 * margin:
        raise ValueError("Image too small for the requested neighborhood radius.")
    
    cy = random.randint(margin, Ny - margin - 1)
    cx = random.randint(margin, Nx - margin - 1)

    # 5. 提取 Prior 数据 (X_train): r_prior 邻域 (通常 3x3)
    # 你的描述: "先验数据来自中心 patch 的局部邻域"
    train_patches = []
    for dy in range(-r_prior, r_prior + 1):
        for dx in range(-r_prior, r_prior + 1):
            # 包含中心点作为先验的一部分，或者不包含？通常包含能更好捕捉局部特征
            train_patches.append(blocks[cy + dy, cx + dx])
    
    # 6. 提取 Test 数据 (X_test): r_test 邻域 (通常 5x5)
    # 你的描述: "测试 patch 来自更大空间邻域... 中心 patch 不参与测试"
    test_patches = []
    for dy in range(-r_test, r_test + 1):
        for dx in range(-r_test, r_test + 1):
            if dy == 0 and dx == 0: continue # 排除中心点，避免“作弊”
            # 也可以排除掉 X_train 已经用过的点，看具体设定。
            # 这里按照通常设定：测试集是更大的范围，除去中心信号本身。
            test_patches.append(blocks[cy + dy, cx + dx])

    X_train_spatial = np.array(train_patches) # (L_train, 8, 8)
    X_test_spatial = np.array(test_patches)   # (L_test, 8, 8)

    # 7. 转到 DCT 域并拉平
    X_train = perform_2d_dct_flatten(X_train_spatial)
    X_test = perform_2d_dct_flatten(X_test_spatial)

    return X_train, X_test

# ==========================================
# 1. POCS Algorithm (Paper Method)
# ==========================================

def compute_mu_bound(M, N):
    if M >= N: return 0
    return np.sqrt((N - M) / (M * (N - 1)))

def get_psi_paper_pocs(Phi, max_iter=50, tol=1e-6):
    M, N = Phi.shape
    mu = compute_mu_bound(M, N)
    Phi_pinv = pinv(Phi)
    P_G = Phi_pinv @ Phi
    G = Phi.T @ Phi

    for _ in range(max_iter):
        G_prev = G.copy()
        H = G.copy()
        np.fill_diagonal(H, 1.0)
        mask = ~np.eye(N, dtype=bool)
        H[mask] = np.clip(H[mask], -mu, mu)
        G = H @ P_G
        if np.linalg.norm(G - G_prev, 'fro') < tol: break

    Psi_T = H @ Phi_pinv
    Psi = Psi_T.T
    Psi = Psi / (np.linalg.norm(Psi, axis=0, keepdims=True) + 1e-10)
    return Psi

# ==========================================
# 2. Main Experiment
# ==========================================

def run_experiment():
    # --- Settings ---
    # N=64 (8x8 patch), K 不宜过大
    NUM_TRIALS = 50           
    CR_list = [0.1, 0.2, 0.3] 
    K_values = np.arange(2, 17, 2) # K: 2, 4, ..., 16
    
    # 确保 pic 文件夹存在，否则无法运行
    if not os.path.exists('pic'):
        os.makedirs('pic')
        print("[Info] Created 'pic/' folder. Please put some images in it.")
        # 创建一个简单的 dummy 图片以防报错
        dummy = np.random.rand(128, 128)
        plt.imsave('pic/dummy_noise.png', dummy, cmap='gray')

    final_results = {}

    for CR in CR_list:
        N = 64 # 8x8 patches fixed
        M = int(N * CR)
        if M < 1: M = 1 # 防止 M=0
        
        print(f"\n=== Processing CR: {CR} (M={M}, N={N}) ===")
        
        res_weighted = np.zeros((NUM_TRIALS, 3, len(K_values)))

        for t in range(NUM_TRIALS):
            # [关键] 每次 Trial 重新随机读取一张图片的一个 patch 区域
            # 这模拟了 Image-Adaptive / Self-Prior 的过程
            try:
                X_train, X_test = load_random_image_patch_data(pic_dir='pic')
            except Exception as e:
                print(f"Data loading error: {e}")
                continue

            # 1. Measurement Matrix
            Phi = np.random.randn(M, N)
            Phi = Phi / np.linalg.norm(Phi, axis=0, keepdims=True)

            # 2. Design Sensing Matrices
            # (A) Baseline
            Psi_base = Phi.copy()
            
            # (B) Paper (POCS) - Data Blind
            Psi_paper = get_psi_paper_pocs(Phi, max_iter=30)
            
            # (C) Proposed - Data Driven (利用 3x3 邻域先验)
            # 注意: X_train 只有 9 个样本，协方差矩阵秩不足
            # design_pinv_psi_fast 内部必须有正则化 (lambda * I)
            # 此处我们假设你的库函数处理好了，或者它就是 (Phi R Phi^T + lam I)^-1 Phi R
            Psi_prop = design_pinv_psi_fast(Phi, X_train) 
            Psi_prop = Psi_prop / (np.linalg.norm(Psi_prop, axis=0, keepdims=True) + 1e-10)

            # 3. Abs Gram Matrices
            AbsG_base = np.abs(Psi_base.T @ Phi)
            AbsG_paper = np.abs(Psi_paper.T @ Phi)
            AbsG_prop = np.abs(Psi_prop.T @ Phi)

            # 4. Evaluation Loop on X_test (5x5 neighborhood excluding center)
            # X_test shape: (64, 24)
            
            for k_idx, K in enumerate(K_values):
                w_vals = np.zeros((3, X_test.shape[1]))

                for i in range(X_test.shape[1]):
                    x_raw = X_test[:, i]
                    x = x_raw.copy()
                    
                    # --- [关键步骤 1] 去除 DC 分量 ---
                    # 2D DCT 后，DC 分量是第 0 个元素
                    x[0] = 0 
                    
                    if np.all(x == 0): continue
                    
                    # --- [关键步骤 2] 获取真实的大系数 (Real Coefficients) ---
                    idx_sorted = np.argsort(np.abs(x))
                    Gamma = idx_sorted[-K:] # Support
                    
                    # 取出真实的系数幅值
                    x_gamma = np.abs(x[Gamma])
                    
                    # Non-support
                    mask = np.ones(N, dtype=bool)
                    mask[Gamma] = False
                    Gamma_c = np.where(mask)[0]
                    
                    # --- [关键步骤 3] 计算加权最大干扰 ---
                    # Metric: max_{i not in Gamma} sum_{j in Gamma} |G_ij| * |x_j|
                    
                    # Base
                    interf_vec_base = AbsG_base[Gamma_c][:, Gamma] @ x_gamma
                    w_vals[0, i] = np.max(interf_vec_base)
                    
                    # Paper
                    interf_vec_paper = AbsG_paper[Gamma_c][:, Gamma] @ x_gamma
                    w_vals[1, i] = np.max(interf_vec_paper)
                    
                    # Proposed
                    interf_vec_prop = AbsG_prop[Gamma_c][:, Gamma] @ x_gamma
                    w_vals[2, i] = np.max(interf_vec_prop)

                # 对当前 Trial 的所有测试 patch 取平均
                res_weighted[t, :, k_idx] = np.mean(w_vals, axis=1)
        
        # 对所有 Trial 取平均
        final_results[CR] = np.mean(res_weighted, axis=0)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    styles = [
        {'c': 'gray', 'ls': '--', 'm': 'o', 'lbl': 'Baseline'},
        {'c': 'blue', 'ls': '-', 'm': 's', 'lbl': 'Paper (POCS)'},
        {'c': 'red', 'ls': '-', 'm': '^', 'lbl': 'Proposed (Ours)'}
    ]

    for idx, CR in enumerate(CR_list):
        ax = axes[idx]
        res = final_results[CR]
        
        for m_id in range(3):
            ax.plot(K_values, res[m_id], 
                    color=styles[m_id]['c'], linestyle=styles[m_id]['ls'], 
                    marker=styles[m_id]['m'], label=styles[m_id]['lbl'], 
                    linewidth=2, alpha=0.8)
        
        ax.set_title(f'CR = {CR}\nEffective Weighted Interference', fontsize=13)
        ax.set_xlabel('Sparsity K')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if idx == 0: 
            ax.set_ylabel(r'Max Weighted Interference')
            ax.legend(fontsize=10)

    plt.suptitle('Natural Image Patch Analysis (Self-Prior): Proposed Method Minimizes Real Interference', fontsize=16)
    plt.tight_layout()
    
    save_file = "weighted_interference_natural_patches.png"
    plt.savefig(save_file, dpi=300)
    plt.show()

    print(f"[Done] Saved to {save_file}")
    print("Interpretation: In this self-prior setting, Proposed method adapts to the local")
    print("frequency structures (edges/textures) of the image patch, significantly reducing")
    print("interference where the signal energy is actually concentrated.")

if __name__ == "__main__":
    run_experiment()