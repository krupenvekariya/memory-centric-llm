"""
Roofline Performance Model
===========================
Directly based on:
- Roofline Model (Williams et al., 2009):
  "Roofline: An Insightful Visual Performance Model for Multicore Architectures"
  https://dl.acm.org/doi/10.1145/1498765.1498785

  Formula:
  Attainable Performance = min(Peak FLOP/s, Peak BW * Arithmetic Intensity)
  Arithmetic Intensity (AI) = FLOP / Bytes Accessed

  Ridge Point = Peak FLOP/s / Peak BW
  - Operations LEFT of ridge point are MEMORY BOUND
  - Operations RIGHT of ridge point are COMPUTE BOUND

  LLM decode is memory bound (AI << Ridge Point)
  because each token generation reads entire KV cache
  but performs very few FLOPs.

Hardware specs used (NVIDIA A100 SXM):
  - Peak FP16 FLOP/s : 312 TFLOP/s
  - Peak HBM BW      : 2.0 TB/s
  - Ridge Point      : 312 / 2.0 = 156 FLOP/byte
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# NVIDIA A100 SXM specs
# Source: https://www.nvidia.com/en-us/data-center/a100/
PEAK_FLOPS_TFLOPS = 312
PEAK_BW_TBS = 2.0

def plot_roofline(operations, save_path="results/figures/roofline_model.png"):
    """
    Plot roofline model with LLM attention operations.
    Formula: Attainable Perf = min(Peak_FLOPS, Peak_BW * AI)
    Based on Williams et al., 2009.
    """
    # Arithmetic intensity range
    ai = np.logspace(-2, 4, 1000)

    # Roofline ceiling
    perf_roof = np.minimum(PEAK_FLOPS_TFLOPS, PEAK_BW_TBS * ai)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(ai, perf_roof, 'b-', linewidth=2.5, label='A100 Roofline')
    ax.axvline(x=PEAK_FLOPS_TFLOPS/PEAK_BW_TBS, color='blue',
               linestyle='--', alpha=0.5, label=f'Ridge Point ({PEAK_FLOPS_TFLOPS/PEAK_BW_TBS:.0f} FLOP/byte)')

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
    ax.set_title('Roofline Model — A100 GPU\n(Williams et al., 2009)', fontsize=13)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")
    plt.show()

if __name__ == "__main__":
    # Operations: (name, FLOPs, bytes_accessed)
    # FLOPs = 2 * seq * seq * heads * head_dim (attention matmul)
    # Bytes = KV cache reads (2 = K+V, dtype_bytes per element)
    operations = [
        # Naive FP16: reads K+V in FP16 (2 bytes each)
        ("Naive Attn FP16", 2*512*512*64, 3*512*64*2),
        # INT8: reads K+V in INT8 (1 byte each) — KVQuant inspired
        ("Attn INT8",       2*512*512*64, 3*512*64*1),
        # INT4: reads K+V in INT4 (0.5 bytes each) — KVQuant inspired
        ("Attn INT4",       2*512*512*64, 3*512*64*0.5),
        # FlashAttention: fused kernel reduces HBM reads
        ("FlashAttention",  2*512*512*64, 512*64*2 + 512*64*2),
        # FFN layer for comparison
        ("FFN Layer",       2*512*4096*1024, 3*512*1024*2),
    ]
    plot_roofline(operations)