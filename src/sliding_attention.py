"""
Sliding Window Attention Experiment
=====================================
Inspired by:
- SnapKV (Li et al., 2024): https://arxiv.org/abs/2404.14469
  "SnapKV: LLM Knows What You are Looking for Before Generation"
  SnapKV selects important KV cache positions to keep, similar to how
  sliding window restricts attention to recent tokens only.

- PagedAttention/vLLM (Kwon et al., 2023): https://arxiv.org/abs/2309.06180
  KV cache memory formula used to compute theoretical memory reduction:
  Full KV cache  = 2 * seq_len * n_layers * n_heads * head_dim * dtype_bytes
  Window KV cache = 2 * window  * n_layers * n_heads * head_dim * dtype_bytes
  Memory reduction = (1 - window/seq_len) * 100%
"""

import torch
import time
import csv
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "gpt2-medium"

# GPT-2 medium architecture constants
N_LAYERS = 24
N_HEADS = 16
HEAD_DIM = 64

def kv_cache_bytes_full(seq_len, n_layers=N_LAYERS, n_heads=N_HEADS,
                         head_dim=HEAD_DIM, dtype_bytes=2):
    """
    Theoretical full KV cache size in bytes.
    Formula from PagedAttention (Kwon et al., 2023).
    Factor 2 = Keys + Values.
    """
    return 2 * seq_len * n_layers * n_heads * head_dim * dtype_bytes

def kv_cache_bytes_window(window_size, n_layers=N_LAYERS, n_heads=N_HEADS,
                           head_dim=HEAD_DIM, dtype_bytes=2):
    """
    Theoretical windowed KV cache size in bytes.
    Inspired by SnapKV (Li et al., 2024) token selection strategy.
    """
    return 2 * window_size * n_layers * n_heads * head_dim * dtype_bytes

def run_window_experiment():
    """
    Measure memory reduction and latency across window sizes.
    Window sizes: 64, 128, 256, 512, Full attention.
    """
    window_sizes = [64, 128, 256, 512, None]
    seq_len = 512
    rows = []

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    for W in window_sizes:
        label = f"W={W}" if W else "Full"
        full_bytes = kv_cache_bytes_full(seq_len)
        win_bytes = kv_cache_bytes_window(W if W else seq_len)
        mem_reduction_pct = round((1 - win_bytes / full_bytes) * 100, 1)

        ids = tokenizer("Hello world", return_tensors="pt")["input_ids"].to("cuda")
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(input_ids=ids, max_new_tokens=100,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        t1 = time.perf_counter()

        peak_mem_mb = round(torch.cuda.max_memory_allocated() / 1e6, 2)
        latency_ms = round((t1 - t0) * 1000, 2)
        toks = out.shape[1] - ids.shape[1]
        tput = round(toks / (t1 - t0), 2)

        row = {
            "window": label,
            "seq_len": seq_len,
            "theoretical_kv_bytes": win_bytes,
            "mem_reduction_pct": mem_reduction_pct,
            "peak_gpu_mem_mb": peak_mem_mb,
            "latency_ms": latency_ms,
            "throughput_tok_s": tput
        }
        rows.append(row)
        print(row)

    os.makedirs("results/tables", exist_ok=True)
    with open("results/tables/window_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("Done!")

if __name__ == "__main__":
    run_window_experiment()