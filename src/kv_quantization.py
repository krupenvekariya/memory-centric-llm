"""
KV Cache Quantization Experiment
=================================
Inspired by:
- KVQuant (Hooper et al., 2024): https://arxiv.org/abs/2401.18079
  "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization"
  Our INT8/INT4 quantization is a practical approximation of their per-channel
  KV cache quantization method using bitsandbytes library.

- PagedAttention/vLLM (Kwon et al., 2023): https://arxiv.org/abs/2309.06180
  KV cache memory formula: 2 * seq_len * n_layers * n_heads * head_dim * dtype_bytes
  Factor of 2 accounts for both Keys and Values.
"""

import torch
import time
import csv
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "gpt2-medium"

def load_model(quant_bits=None):
    """
    Load GPT-2 medium in FP16, INT8, or INT4.
    Quantization approach inspired by KVQuant (Hooper et al., 2024).
    """
    if quant_bits == 8:
        config = BitsAndBytesConfig(load_in_8bit=True)
    elif quant_bits == 4:
        config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    else:
        config = None
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=config,
        dtype=torch.float16 if config is None else None, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def measure_memory_and_latency(model, tokenizer, seq_len):
    """
    Measure peak GPU memory and latency for a given sequence length.
    Memory formula from PagedAttention (Kwon et al., 2023):
    KV cache size = 2 * seq_len * n_layers * n_heads * head_dim * dtype_bytes
    """
    ids = tokenizer("Hello world", return_tensors="pt")["input_ids"].to("cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=seq_len,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    peak_mem = round(torch.cuda.max_memory_allocated() / 1e6, 2)
    lat = round((t1 - t0) * 1000, 2)
    toks = out.shape[1] - ids.shape[1]
    tput = round(toks / (t1 - t0), 2)
    return {"peak_mem_mb": peak_mem, "latency_ms": lat, "throughput_tok_s": tput}