import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 0. 全局字体加粗与大小设置
# ==========================================
plt.rcParams.update({
    'font.size': 12,              # 基础字体大小
    'font.weight': 'bold',        # 全局字体加粗
    'axes.labelweight': 'bold',   # 坐标轴标签加粗
    'axes.titleweight': 'bold',   # 子图标题加粗
    'axes.labelsize': 14,         # 坐标轴标签大小
    'axes.titlesize': 13,         # 子图标题大小
    'xtick.labelsize': 12,        # X轴刻度数字大小
    'ytick.labelsize': 12,        # Y轴刻度数字大小
    'legend.fontsize': 11,        # 图例字体大小
    'legend.title_fontsize': 12,  # 图例标题大小
})

# ==============================================================================
# ======================== 图 1: 消融实验 (Ablation Test) ========================
# ==============================================================================
def plot_figure_1():
    # --- 1. 提取数据点 ---
    # Subplot 1: CR
    cr_x = [0.1, 0.2, 0.3, 0.4, 0.5]
    cr_omp = [23.90, 27.60, 29.10, 29.50, 29.70]
    cr_prop_base = [24.95, 28.00, 28.60, 28.75, 28.90]
    cr_prop_decay = [25.10, 28.25, 28.85, 29.05, 29.15]

    # Subplot 2: K
    k_x = [5, 10, 20, 30, 40, 50, 60]
    k_omp = [24.80, 25.00, 24.00, 23.40, 23.20, 23.10, 23.05]
    k_prop_base = [25.00, 25.40, 25.00, 24.40, 24.20, 24.00, 23.85]
    k_prop_decay = [25.15, 25.75, 25.15, 24.45, 24.25, 24.05, 23.90]

    # Subplot 3: L
    l_x = [1, 2, 3, 4, 5, 6, 7, 9, 12, 16, 20]
    l_omp = [23.90, 23.94, 23.91, 24.00, 23.92, 23.85, 23.98, 23.89, 23.90, 23.95, 23.86]
    l_prop_base = [24.30, 24.50, 24.60, 24.70, 24.75, 24.80, 24.85, 24.88, 24.92, 24.95, 24.97]
    l_prop_decay = [24.60, 24.80, 24.90, 24.98, 25.02, 25.05, 25.07, 25.09, 25.10, 25.11, 25.12]

    # --- 2. 样式定义 ---
    style_omp = {'color': 'gray', 'ls': '--', 'marker': 'o', 'label': 'Classic OMP'}
    style_prop_base = {'color': 'blue', 'ls': '-', 'marker': 's', 'label': 'Proposed (Temporal Only)'}
    style_prop_decay = {'color': 'red', 'ls': '-', 'marker': '^', 'label': 'Proposed + Spatial + Decay'}

    # --- 3. 绘图 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    # Plot 1: CR
    axes[0].plot(cr_x, cr_omp, **style_omp)
    axes[0].plot(cr_x, cr_prop_base, **style_prop_base)
    axes[0].plot(cr_x, cr_prop_decay, **style_prop_decay)
    axes[0].set_title('Robustness against Compression Ratio\n(K=20, L=10)')
    axes[0].set_xlabel('Compression Ratio (CR)')
    axes[0].set_ylabel('Average Full-Frame PSNR (dB)')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='lower right')

    # Plot 2: K
    axes[1].plot(k_x, k_omp, **style_omp)
    axes[1].plot(k_x, k_prop_base, **style_prop_base)
    axes[1].plot(k_x, k_prop_decay, **style_prop_decay)
    axes[1].set_title('Robustness against Sparsity Level\n(CR=0.1, L=10)')
    axes[1].set_xlabel('Sparsity K (Number of Atoms)')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='upper right')

    # Plot 3: L
    axes[2].plot(l_x, l_omp, **style_omp)
    axes[2].plot(l_x, l_prop_base, **style_prop_base)
    axes[2].plot(l_x, l_prop_decay, **style_prop_decay)
    axes[2].set_title('Effect of History Prior Buffer Length\n(CR=0.1, K=20)')
    axes[2].set_xlabel('History Length (L)')
    axes[2].grid(True, linestyle=':', alpha=0.6)
    axes[2].legend(loc='center right')

    plt.tight_layout(w_pad=2.0)
    plt.savefig("recreated_ablation_bold.png", dpi=300, bbox_inches='tight')
    plt.show()

# ==============================================================================
# =================== 图 2: 关键帧对比 (Key Stages Broken Axis) ===================
# ==============================================================================
def plot_figure_2():
    # --- 1. 提取分段数据 ---
    segments = [(0, 10), (50, 60), (170, 180)]
    methods = ["Gaussian", "Modified-CS", "Bo Li", "Huang", "Proposed", "Proposed+Spatial+Decay"]
    
    # 按照原图视觉走势估算的精准数据字典
    data = {
        "Gaussian": [
            [29.0, 28.5, 27.5, 26.0, 25.8, 25.2, 24.8, 23.5, 24.2, 23.2], # 0-9
            [20.1, 20.1, 19.9, 20.5, 20.3, 19.9, 20.1, 20.4, 20.1, 20.2], # 50-59
            [20.2, 20.5, 20.2, 20.3, 21.2, 21.0, 22.0, 22.8, 22.3, 22.8]  # 170-179
        ],
        "Modified-CS": [
            [18.5, 17.8, 17.4, 17.3, 16.8, 16.2, 15.8, 15.2, 14.7, 14.2],
            [9.2,  9.2,  9.2,  9.2,  9.2,  9.2,  9.2,  9.2,  9.2,  9.2],
            [13.2, 13.4, 13.6, 13.8, 14.1, 14.2, 14.4, 14.7, 15.0, 15.4]
        ],
        "Bo Li": [
            [29.5, 28.5, 27.5, 26.0, 25.8, 25.2, 24.8, 23.5, 24.2, 23.2],
            [26.4, 26.2, 25.8, 26.2, 26.1, 25.6, 25.9, 26.0, 25.6, 26.0],
            [26.3, 26.5, 26.2, 26.8, 27.3, 27.1, 27.9, 28.3, 28.3, 28.8]
        ],
        "Huang": [
            [29.3, 28.5, 27.5, 26.2, 25.9, 25.3, 24.9, 23.6, 24.2, 23.4],
            [20.8, 20.1, 20.0, 20.0, 20.3, 20.0, 20.0, 20.3, 20.0, 20.0],
            [20.5, 20.6, 19.9, 20.8, 21.3, 21.5, 22.3, 22.7, 22.4, 23.0]
        ],
        "Proposed": [
            [29.5, 29.2, 27.8, 26.5, 26.1, 25.7, 25.4, 24.2, 24.5, 23.8],
            [20.9, 21.0, 20.5, 21.1, 21.0, 20.5, 20.7, 21.1, 20.8, 21.0],
            [20.8, 21.2, 20.8, 20.9, 21.8, 21.9, 22.6, 23.3, 22.8, 23.1]
        ],
        "Proposed+Spatial+Decay": [
            [30.1, 29.5, 28.2, 26.9, 26.6, 26.5, 26.0, 24.9, 24.9, 24.5],
            [21.2, 21.3, 20.9, 21.4, 21.4, 20.9, 21.1, 21.4, 21.0, 21.2],
            [21.3, 21.5, 21.4, 21.4, 22.4, 22.3, 23.0, 24.0, 23.4, 23.5]
        ]
    }

    style_map = {
        "Gaussian":               {"color": "gray",   "ls": "--", "marker": "o"},
        "Modified-CS":            {"color": "black",  "ls": "-.", "marker": "s"},
        "Bo Li":                  {"color": "forestgreen", "ls": "-.", "marker": "d"},
        "Huang":                  {"color": "cyan",   "ls": "-.", "marker": "x"},
        "Proposed":               {"color": "blue",   "ls": "-",  "marker": "*"},
        "Proposed+Spatial+Decay": {"color": "red",    "ls": "-",  "marker": "^"}
    }

    # --- 2. 创建带断轴的图表 ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(18, 6.5))
    plt.subplots_adjust(wspace=0.05, bottom=0.2) 
    axes = [ax1, ax2, ax3]

    for i, (start, end) in enumerate(segments):
        ax = axes[i]
        x_indices = np.arange(start, end)
        
        for met in methods:
            y_data = data[met][i]
            s = style_map[met]
            lbl = met if i == 0 else "_nolegend_"
            ax.plot(x_indices, y_data, color=s["color"], linestyle=s["ls"], 
                    marker=s["marker"], label=lbl, linewidth=2, markersize=6, alpha=0.9)
        
        ax.set_xlim(start, end - 1)
        ax.set_xticks(x_indices)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_xlabel("Frame Index")
        
        # 内部醒目文本框
        ax.text(0.5, 0.95, f"Frames {start}-{end}", transform=ax.transAxes, 
                ha='center', va='top', fontsize=14, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        if i == 0:
            ax.set_ylabel("Full-Frame PSNR (dB)")
        
        # 绘制断轴截断线 (//)
        d = .015 
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, linewidth=1.5)
        if i > 0: 
            ax.spines['left'].set_visible(False)
            ax.tick_params(labelleft=False, left=False)
            ax.plot((-d, +d), (-d, +d), **kwargs)
            ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        if i < 2: 
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelright=False, right=False)
            ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
            ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # --- 3. 添加全局标题和图例 ---
    fig.suptitle("Full-Frame PSNR Comparison at Key Stages (CR=0.1)", fontsize=18, fontweight='bold', y=0.95)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, 0.02))
    
    plt.savefig("recreated_key_stages_bold.png", dpi=300, bbox_inches='tight')
    plt.show()

# ==========================================
# 执行绘图函数
# ==========================================
if __name__ == "__main__":
    print("Generating Figure 1: Ablation Test...")
    plot_figure_1()
    
    print("Generating Figure 2: Key Stages Comparison...")
    plot_figure_2()
    print("Done!")