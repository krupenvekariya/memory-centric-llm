import numpy as np
import matplotlib.pyplot as plt
import os

PEAK_FLOPS_TFLOPS = 312
PEAK_BW_TBS = 2.0

def plot_roofline(operations, save_path="results/figures/roofline_model.png"):
    ai = np.logspace(-2, 4, 1000)
    perf_roof = np.minimum(PEAK_FLOPS_TFLOPS, PEAK_BW_TBS * ai)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(ai, perf_roof, 'b-', linewidth=2.5, label='A100 Roofline')
    ax.axvline(x=PEAK_FLOPS_TFLOPS/PEAK_BW_TBS, color='blue',
               linestyle='--', alpha=0.5, label='Ridge Point')

    colors = ['#e74c3c','#e67e22','#f39c12','#2ecc71','#9b59b6']
    markers = ['o','s','^','D','v']

    for (name, flops, bytes_accessed), color, marker in zip(operations, colors, markers):
        ai_val = flops / bytes_accessed
        achievable = min(PEAK_FLOPS_TFLOPS, PEAK_BW_TBS * ai_val)
        ax.scatter(ai_val, achievable, s=120, color=color,
                   marker=marker, zorder=5, label=name)
        ax.annotate(name, (ai_val, achievable),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)

    ax.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=12)
    ax.set_ylabel('Performance (TFLOP/s)', fontsize=12)
    ax.set_title('Roofline Model — A100 GPU', fontsize=13)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")
    plt.show()

if __name__ == "__main__":
    operations = [
        ("Naive Attn FP16", 2*512*512*64, 3*512*64*2),
        ("Attn INT8",       2*512*512*64, 3*512*64*1),
        ("Attn INT4",       2*512*512*64, 3*512*64*0.5),
        ("FlashAttention",  2*512*512*64, 512*64*2 + 512*64*2),
        ("FFN Layer",       2*512*4096*1024, 3*512*1024*2),
    ]
    plot_roofline(operations)