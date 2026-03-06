"""
Memory Profiler Utilities
==========================
Based on:
- PagedAttention/vLLM (Kwon et al., 2023): https://arxiv.org/abs/2309.06180
  "Efficient Memory Management for Large Language Model Serving with PagedAttention"
  Memory measurement approach inspired by their GPU memory profiling methodology.
  KV cache size formula:
  size = 2 * seq_len * n_layers * n_heads * head_dim * dtype_bytes

- Roofline Model (Williams et al., 2009):
  Peak memory bandwidth used as reference for bandwidth utilization.
  A100 HBM bandwidth: 2.0 TB/s
"""

import torch
import time

# NVIDIA A100 reference specs
A100_HBM_BW_TBS = 2.0
A100_PEAK_FLOPS_TFLOPS = 312

def reset_memory():
    """Reset GPU memory stats for clean measurement."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def get_peak_memory_mb():
    """Get peak GPU memory allocated in MB."""
    return round(torch.cuda.max_memory_allocated() / 1e6, 2)

def get_current_memory_mb():
    """Get current GPU memory allocated in MB."""
    return round(torch.cuda.memory_allocated() / 1e6, 2)

def theoretical_kv_cache_mb(seq_len, n_layers=24, n_heads=16,
                              head_dim=64, dtype_bytes=2):
    """
    Compute theoretical KV cache size in MB.
    Formula from PagedAttention (Kwon et al., 2023).
    Factor 2 = Keys + Values.

    Args:
        seq_len    : sequence length in tokens
        n_layers   : number of transformer layers
        n_heads    : number of attention heads
        head_dim   : dimension per head
        dtype_bytes: bytes per element (2=FP16, 1=INT8, 0.5=INT4)

    Returns:
        KV cache size in MB
    """
    bytes_total = 2 * seq_len * n_layers * n_heads * head_dim * dtype_bytes
    return round(bytes_total / 1e6, 4)

def measure_latency(model, input_ids, max_new_tokens=50, n_runs=3):
    """
    Measure average generation latency over n_runs.

    Args:
        model         : loaded HuggingFace model on CUDA
        input_ids     : tokenized input tensor
        max_new_tokens: tokens to generate
        n_runs        : number of runs to average

    Returns:
        average latency in milliseconds
    """
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return round(sum(times) / len(times), 2)

def profile_model(model, tokenizer, prompt="Hello world", max_new_tokens=50):
    """
    Full memory + latency profile of a model.

    Returns dict with:
        peak_mem_mb      : peak GPU memory in MB
        latency_ms       : average latency in ms
        throughput_tok_s : tokens per second
    """
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
    reset_memory()
    latency_ms = measure_latency(model, ids, max_new_tokens)
    peak_mem_mb = get_peak_memory_mb()
    throughput = round(max_new_tokens / (latency_ms / 1000), 2)
    return {
        "peak_mem_mb": peak_mem_mb,
        "latency_ms": latency_ms,
        "throughput_tok_s": throughput
    }

def bandwidth_utilization(bytes_accessed, latency_s):
    """
    Compute memory bandwidth utilization vs A100 peak.
    Based on roofline analysis (Williams et al., 2009).

    Returns:
        utilization as percentage of peak A100 HBM bandwidth
    """
    actual_bw_tbs = (bytes_accessed / latency_s) / 1e12
    utilization_pct = round((actual_bw_tbs / A100_HBM_BW_TBS) * 100, 2)
    return utilization_pct