import torch
import time
import csv
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "gpt2-medium"

def load_model(quant_bits=None):
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