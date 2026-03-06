"""
Energy Per Token Model
=======================
Directly based on:
- Roofline Model (Williams et al., 2009):
  "Roofline: An Insightful Visual Performance Model for Multicore Architectures"
  https://dl.acm.org/doi/10.1145/1498765.1498785

  Energy model formula:
  bottleneck_time = max(bytes_accessed / HBM_BW, flops / PEAK_FLOPS)
  energy_joules   = bottleneck_time * board_power_watts

- PagedAttention/vLLM (Kwon et al., 2023): https://arxiv.org/abs/2309.06180
  KV cache bytes formula used to compute bytes_accessed per token.

- KVQuant (Hooper et al., 2024): https://arxiv.org/abs/2401.18079
  dtype_bytes reduction (FP16->INT8->INT4) directly reduces
  bytes_accessed and therefore energy per token.

Hardware specs (NVIDIA A100 SXM):
  HBM Bandwidth : 2.0 TB/s
  Board Power   : 400 W
  Peak FLOP/s   : 312 TFLOP/s
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# NVIDIA A100 SXM specs
HBM_BW = 2e12
BOARD_POWER = 400
PEAK_FLOPS = 312e12

# GPT-2 medium architecture
N_LAYERS = 24
N_HEADS = 16
HEAD_DIM = 64

def energy_per_token(seq_len, dtype_bytes):
    """
    Estimate energy per token using roofline-based model.
    Formula from Williams et al., 2009.

    Args:
        seq_len    : sequence length (tokens)
        dtype_bytes: bytes per element (2=FP16, 1=INT8, 0.5=INT4)

    Returns:
        energy in microjoules (uJ)
    """
    # Bytes accessed = KV cache read per token
    # From PagedAttention (Kwon et al., 2023)
    bytes_accessed = 2 * seq_len * N_LAYERS * N_HEADS * HEAD_DIM * dtype_bytes

    # FLOPs = attention matmul
    flops = 2 * seq_len * seq_len * N_HEADS * HEAD_DIM

    # Bottleneck time (roofline model)
    mem_time = bytes_accessed / HBM_BW
    compute_time = flops / PEAK_FLOPS
    bottleneck_time = max(mem_time, compute_time)

    # Energy
    energy_j = bottleneck_time * BOARD_POWER
    return round(energy_j * 1e6, 4)

def run_energy_experiment():
    """
    Run energy model for all quant configs and sequence lengths.
    dtype_bytes reduction inspired by KVQuant (Hooper et al., 2024).
    """
    seq_lengths = [128, 256, 512]
    quant_configs = {"FP16": 2, "INT8": 1, "INT4": 0.5}
    rows = []

    for quant, db in quant_configs.items():
        for seq_len in seq_lengths:
            energy_uj = energy_per_token(seq_len, db)
            bottleneck = "memory"
            rows.append({
                "quant": quant,
                "seq_len": seq_len,
                "energy_uj": energy_uj,
                "bottleneck": bottleneck
            })
            print(f"{quant} seq={seq_len}: {energy_uj} uJ")

    df = pd.DataFrame(rows)
    os.makedirs("results/tables", exist_ok=True)
    df.to_csv("results/tables/energy_results.csv", index=False)
    print("Saved energy_results.csv")
    return df

def plot_energy(df):
    """Plot energy per token comparison across quantization levels."""
    fig, ax = plt.subplots(figsize=(10, 5))
    seq_lengths = df["seq_len"].unique()
    quants = df["quant"].unique()
    width = 0.25
    x = np.arange(len(seq_lengths))

    for i, quant in enumerate(quants):
        vals = df[df["quant"] == quant]["energy_uj"].values
        bars = ax.bar(x + i*width, vals, width, label=quant)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 50,
                    f"{val:.1f}", ha='center', fontsize=8)

    ax.set_title("Energy per Token (μJ)\nBased on Roofline Model (Williams et al., 2009)")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Energy (microjoules)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(seq_lengths)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/energy_per_token.png", dpi=150)
    print("Saved energy_per_token.png")
    plt.show()

if __name__ == "__main__":
    df = run_energy_experiment()
    plot_energy(df)