import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("results/figures", exist_ok=True)

def plot_quant_results():
    df = pd.read_csv("results/tables/quant_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for quant, grp in df.groupby("quant"):
        axes[0].plot(grp["seq_len"], grp["peak_mem_mb"], marker='o', label=quant)
    axes[0].set_title("Peak GPU Memory vs Sequence Length")
    axes[0].set_xlabel("Sequence Length (tokens)")
    axes[0].set_ylabel("Peak Memory (MB)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for quant, grp in df.groupby("quant"):
        axes[1].plot(grp["seq_len"], grp["throughput_tok_s"], marker='s', label=quant)
    axes[1].set_title("Throughput vs Sequence Length")
    axes[1].set_xlabel("Sequence Length (tokens)")
    axes[1].set_ylabel("Tokens / Second")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/figures/quant_memory_throughput.png", dpi=150)
    print("Saved quant_memory_throughput.png")

def plot_window_results():
    df = pd.read_csv("results/tables/window_results.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#3498db']
    bars = ax.bar(df["window"], df["mem_reduction_pct"], color=colors)
    ax.set_title("KV Cache Memory Reduction — Sliding Window")
    ax.set_xlabel("Window Size")
    ax.set_ylabel("Memory Reduction (%)")
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, df["mem_reduction_pct"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f"{val}%", ha='center', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("results/figures/window_memory_reduction.png", dpi=150)
    print("Saved window_memory_reduction.png")

def plot_perplexity():
    df = pd.read_csv("results/tables/perplexity_results.csv")
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ['#3498db','#e67e22','#e74c3c']
    bars = ax.bar(df["quant"], df["perplexity"], color=colors)
    ax.set_title("Perplexity by Quantization Level")
    ax.set_xlabel("Quantization")
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_ylim(0, 60)
    for bar, val in zip(bars, df["perplexity"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{val}", ha='center', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("results/figures/perplexity_comparison.png", dpi=150)
    print("Saved perplexity_comparison.png")

if __name__ == "__main__":
    plot_quant_results()
    plot_window_results()
    plot_perplexity()
    print("All figures saved!")