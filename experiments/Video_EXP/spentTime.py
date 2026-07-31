import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from tqdm import tqdm
import time
from sklearn.linear_model import OrthogonalMatchingPursuit

from prior_cs.utils.normalize import normalize_columns
from prior_cs.algorithms.omp import omp
from prior_cs.algorithms.pinv_psi_omp import pro_omp_solve
from prior_cs.algorithms.pinv_psi_cosamp import pro_cosamp_solve
from prior_cs.algorithms.pinv_psi_omp import design_pinv_psi as design_pinv_psi_slow
from prior_cs.algorithms.psi_fast import design_pinv_psi_fast as design_pinv_psi_fast


# ==========================================
# 1. Utility Functions
# ==========================================
def dct2(block): return dct(dct(block.T, norm="ortho").T, norm="ortho")
def idct2(coeff): return idct(idct(coeff.T, norm="ortho").T, norm="ortho")

def psnr(x_true, x_rec):
    mse = np.mean((x_true - x_rec) ** 2)
    if mse < 1e-12: return 100.0
    return 10 * np.log10(1.0 / mse)

def get_temporal_prior(frames_array, t, r, c):
    if t == 0: return None
    return dct2(frames_array[t-1, r, c]).flatten().reshape(-1, 1)

# ==========================================
# 2. Data Loading
# ==========================================
np.random.seed(42)

try:
    full_data = np.load("VIDEO/video_patches.npy")
    print(f"Successfully loaded real data: {full_data.shape}")
    frames_array = full_data
    
except FileNotFoundError:
    print("File not found. Generating Synthetic Data (20 frames)...")
    frames_array = np.random.rand(20, 6, 6, 32, 32).astype(np.float32)
    for t in range(1, 20):
        frames_array[t] = frames_array[t-1] * 0.98 + np.random.normal(0, 0.01, frames_array[0].shape)

num_frames, Ny, Nx, P, _ = frames_array.shape
n = P ** 2
m = int(0.25 * n) # Ratio 0.25
sparsity_k = 30

print(f"Benchmark Params: M={m}, N={n}, K_sparse={sparsity_k}")
print(f"Total Patches to Process: {(num_frames-1) * Ny * Nx}")

# ==========================================
# 3. Benchmark Loop (over chunks)
# ==========================================
metrics = {
    "OMP":             {"time": [], "psnr": []},
    "Pinv (Slow Psi)": {"time": [], "psnr": []},
    "Pinv (Fast Psi)": {"time": [], "psnr": []}
}

# Define number of chunks to divide the data into
num_chunks = 10  # Modify this as needed

# Split the frames_array into chunks
chunk_size = num_frames // num_chunks
chunks = [frames_array[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]

# Run the benchmarking for each chunk
for chunk_idx, chunk in enumerate(chunks):
    print(f"\nProcessing Chunk {chunk_idx + 1}/{num_chunks}...")

    # Initialize metric storage for this chunk
    chunk_metrics = {
        "OMP":             {"time": [], "psnr": []},
        "Pinv (Slow Psi)": {"time": [], "psnr": []},
        "Pinv (Fast Psi)": {"time": [], "psnr": []}
    }

    # Warmup calls
    Phi = normalize_columns(np.random.randn(m, n))
    # Warmup for each chunk
    design_pinv_psi_slow(Phi, np.random.randn(n, 1))
    design_pinv_psi_fast(Phi, np.random.randn(n, 1))

    print("Starting Benchmark for this chunk...")

    for t in tqdm(range(1, chunk.shape[0]), desc=f"Benchmarking Chunk {chunk_idx + 1}"):
        for r in range(Ny):
            for c in range(Nx):
                # Data
                true_patch = chunk[t, r, c]
                x_true = dct2(true_patch).flatten()
                y = Phi @ x_true

                # Prior (Shared)
                X_prior = get_temporal_prior(chunk, t, r, c)

                # --- 1. Standard OMP ---
                t0 = time.perf_counter()
                coef_omp = omp(Phi, y, k=sparsity_k)
                t_cost = (time.perf_counter() - t0) * 1000

                rec_omp = idct2(coef_omp.reshape(P, P))
                chunk_metrics["OMP"]["time"].append(t_cost)
                chunk_metrics["OMP"]["psnr"].append(psnr(true_patch, rec_omp))

                # --- 2. Pinv (Slow) ---
                t0 = time.perf_counter()
                Psi_slow = design_pinv_psi_slow(Phi, X_prior)
                coef_slow = pro_omp_solve(Phi, Psi_slow, y, sparsity=sparsity_k)
                t_cost = (time.perf_counter() - t0) * 1000

                rec_slow = idct2(coef_slow.reshape(P, P))
                chunk_metrics["Pinv (Slow Psi)"]["time"].append(t_cost)
                chunk_metrics["Pinv (Slow Psi)"]["psnr"].append(psnr(true_patch, rec_slow))

                # --- 3. Pinv (Fast) ---
                t0 = time.perf_counter()
                Psi_fast = design_pinv_psi_fast(Phi, X_prior)
                coef_fast = pro_omp_solve(Phi, Psi_fast, y, sparsity=sparsity_k)
                t_cost = (time.perf_counter() - t0) * 1000

                rec_fast = idct2(coef_fast.reshape(P, P))
                chunk_metrics["Pinv (Fast Psi)"]["time"].append(t_cost)
                chunk_metrics["Pinv (Fast Psi)"]["psnr"].append(psnr(true_patch, rec_fast))

    # Accumulate chunk results into the main metrics
    for method in chunk_metrics:
        metrics[method]["time"].extend(chunk_metrics[method]["time"])
        metrics[method]["psnr"].extend(chunk_metrics[method]["psnr"])

# ==========================================
# 4. Statistics & Visualization
# ==========================================
methods = list(metrics.keys())
avg_times = [np.mean(metrics[m]["time"]) for m in methods]
avg_psnr = [np.mean(metrics[m]["psnr"]) for m in methods]

print("\n" + "="*60)
print(f"{'Method':<20} | {'Avg Time (ms)':<15} | {'Avg PSNR (dB)':<15}")
print("-" * 60)
for i, m in enumerate(methods):
    print(f"{m:<20} | {avg_times[i]:<15.2f} | {avg_psnr[i]:<14.2f}")
print("="*60)

# Calculate Speedup
speedup = avg_times[1] / avg_times[2]
print(f"\n[Conclusion] Optimization Speedup: {speedup:.2f}x faster!")
print(f"[Conclusion] PSNR Difference: {abs(avg_psnr[1] - avg_psnr[2]):.4f} dB (Should be ~0)")
import matplotlib.pyplot as plt
import seaborn as sns

# Set a publication-ready style
sns.set(style="whitegrid", palette="muted")

# Create the figure and axes
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

# Left: Time Comparison
colors = ['#4C72B0', '#55A868', '#C44E52']  # Muted blue, green, and red for better contrast
bars1 = ax[0].bar(methods, avg_times, color=colors, alpha=0.8)

# Set axis labels and title with larger fonts
ax[0].set_ylabel('Avg Time per Patch (ms)', fontsize=14)
ax[0].set_title(f'Execution Time Comparison', fontsize=16)
ax[0].grid(axis='y', linestyle='--', alpha=0.6)

# Annotate each bar with its value
for bar in bars1:
    height = bar.get_height()
    ax[0].text(bar.get_x() + bar.get_width() / 2., height, f'{height:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Right: PSNR Comparison
bars2 = ax[1].bar(methods, avg_psnr, color=colors, alpha=0.8)

# Set axis labels and title with larger fonts
ax[1].set_ylabel('Average PSNR (dB)', fontsize=14)
ax[1].set_title('Reconstruction Quality Comparison', fontsize=16)

# Set Y-limit to make differences visible and improve visibility
y_min = min(avg_psnr) * 0.9
y_max = max(avg_psnr) * 1.05
ax[1].set_ylim(y_min, y_max)
ax[1].grid(axis='y', linestyle='--', alpha=0.6)

# Annotate each bar with its value
for bar in bars2:
    height = bar.get_height()
    ax[1].text(bar.get_x() + bar.get_width() / 2., height, f'{height:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Tight layout to ensure there's no overlap
plt.tight_layout()

# Show the plot
plt.show()


# After the benchmarking loop

# Save the metrics as a .npz file
np.savez('TimeSpent.npz', metrics=metrics, avg_times=avg_times, avg_psnr=avg_psnr)

# Optionally, print a confirmation message
print("\n[Info] Benchmark results saved to 'TimeSpent.npz'")