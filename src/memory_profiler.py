import torch
import time

def reset_memory():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def get_peak_memory_mb():
    return round(torch.cuda.max_memory_allocated() / 1e6, 2)

def get_current_memory_mb():
    return round(torch.cuda.memory_allocated() / 1e6, 2)

def measure_latency(model, input_ids, max_new_tokens=50, n_runs=3):
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
    avg_latency = round(sum(times) / len(times), 2)
    return avg_latency

def profile_model(model, tokenizer, prompt="Hello world", max_new_tokens=50):
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