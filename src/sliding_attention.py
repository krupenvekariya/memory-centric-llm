import torch
import time
import csv
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "gpt2-medium"

def kv_cache_bytes_full(seq_len, n_layers=24, n_heads=16, head_dim=64, dtype_bytes=2):
    return 2 * seq_len * n_layers * n_heads * head_dim * dtype_bytes

def kv_cache_bytes_window(window_size, n_layers=24, n_heads=16, head_dim=64, dtype_bytes=2):
    return 2 * window_size * n_layers * n_heads * head_dim * dtype_bytes

def run_window_experiment():
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