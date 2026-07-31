import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ==================== 导入你的算法 ====================
sys.path.append('/Users/stanzhao/Desktop/prior_cs')
from experiments.S_curve.Proposed import pinv_omp

# ==================== 1. 全局配置 ====================
NEW_TRIALS = 5
PHI_REPEATS = 3
DB_FILE = "exp88_tau_structured_noisy_results.npy"

N, M = 256, 64
X_TEST_POINTS = [40, 45, 50, 55, 60, 65, 70, 75]
TAU_LIST = [100, 1, 1e-2, 1e-4]

# ==================== 噪声模式 ====================
NOISE_MODE = "fixed_var"   # 可选: "snr" 或 "fixed_var"

SNR_DB = 30          # 当 NOISE_MODE = "snr" 时使用
FIXED_NOISE_VAR = 1e-2/500  # 当 NOISE_MODE = "fixed_var" 时使用

# ==================== 2. 工具函数 ====================
def format_tau(tau):
    if tau == 0:
        return "0"
    exp = int(np.floor(np.log10(tau)))
    return f"10^{{{exp}}}"

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

def calc_metrics(x_true, x_est):
    supp_true = set(np.where(np.abs(x_true) > 1e-5)[0])
    supp_est = set(np.where(np.abs(x_est) > 1e-5)[0])
    srr = len(supp_true & supp_est) / len(supp_true) if len(supp_true) > 0 else 1.0
    nre = np.linalg.norm(x_true - x_est) / (np.linalg.norm(x_true) + 1e-10)
    return srr, nre

# ==================== 噪声生成函数（核心改进） ====================
def add_noise(y_clean):
    if NOISE_MODE == "snr":
        P_signal = np.var(y_clean)
        P_noise = P_signal / (10 ** (SNR_DB / 10.0))
    elif NOISE_MODE == "fixed_var":
        P_noise = FIXED_NOISE_VAR
    else:
        raise ValueError("NOISE_MODE must be 'snr' or 'fixed_var'")

    noise = np.random.randn(*y_clean.shape) * np.sqrt(P_noise)
    return y_clean + noise, P_noise

# ==================== 3. 主流程 ====================
X_train = generate_train_data(N, 500, max(X_TEST_POINTS) + 5)

if os.path.exists(DB_FILE):
    db = np.load(DB_FILE, allow_pickle=True).item()
    print(f"加载数据库: {DB_FILE}")
else:
    db = {
        format_tau(tau): {
            'srr_sum': np.zeros(len(X_TEST_POINTS)),
            'nre_sum': np.zeros(len(X_TEST_POINTS)),
            'count': 0
        } for tau in TAU_LIST
    }

if NEW_TRIALS > 0:
    for trial in range(NEW_TRIALS):
        print(f"Trial {trial+1}/{NEW_TRIALS}")

        temp_srr = {format_tau(tau): np.zeros(len(X_TEST_POINTS)) for tau in TAU_LIST}
        temp_nre = {format_tau(tau): np.zeros(len(X_TEST_POINTS)) for tau in TAU_LIST}

        for i, k in enumerate(X_TEST_POINTS):
            x_true = generate_structured_power_law(N, k)

            for _ in range(PHI_REPEATS):
                Phi = np.random.randn(M, N)
                Phi /= np.linalg.norm(Phi, axis=0)

                y_clean = Phi @ x_true

                # ===== 加噪声 =====
                y_noisy, noise_var = add_noise(y_clean)

                for tau in TAU_LIST:
                    key = format_tau(tau)

                    x_est = pinv_omp(Phi, y_noisy, X_train, k, tau)
                    srr, nre = calc_metrics(x_true, x_est)

                    temp_srr[key][i] += srr / PHI_REPEATS
                    temp_nre[key][i] += nre / PHI_REPEATS

        for tau in TAU_LIST:
            key = format_tau(tau)
            db[key]['srr_sum'] += temp_srr[key]
            db[key]['nre_sum'] += temp_nre[key]
            db[key]['count'] += 1

    np.save(DB_FILE, db)
    print("数据已保存")

# ==================== 4. 画图 ====================
plt.rcParams.update({
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'font.size': 14
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

markers = ['o', 's', '^', 'D']
linestyles = ['-', '--', '-.', ':']

for idx, tau in enumerate(TAU_LIST):
    key = format_tau(tau)

    if db[key]['count'] > 0:
        avg_srr = db[key]['srr_sum'] / db[key]['count']
        avg_nre = db[key]['nre_sum'] / db[key]['count']

        label = f"$\\tau={key}$"

        ax1.plot(X_TEST_POINTS, avg_srr,
                 marker=markers[idx], linestyle=linestyles[idx],
                 linewidth=2, label=label)

        ax2.plot(X_TEST_POINTS, avg_nre,
                 marker=markers[idx], linestyle=linestyles[idx],
                 linewidth=2, label=label)

# ===== 标题根据模式自动变化 =====
if NOISE_MODE == "snr":
    title_suffix = f"(SNR = {SNR_DB} dB)"
else:
    title_suffix = f"(Noise Var = {FIXED_NOISE_VAR})"

ax1.set_title("SRR " + title_suffix)
ax2.set_title("NRE " + title_suffix)

ax1.set_xlabel("Sparsity")
ax1.set_ylabel("SRR")
ax1.grid(True)
ax1.legend()

ax2.set_xlabel("Sparsity")
ax2.set_ylabel("NRE")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()