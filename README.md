# Memory-Centric Architectures for Large Language Models
**CECS 530 — University Project | 2-Person Team**

## Overview
This project surveys, implements, and analyzes three memory-centric
optimizations for LLM inference:
1. KV Cache Quantization (INT8/INT4) — inspired by KVQuant (Hooper et al., 2024)
2. Sliding Window Attention — inspired by SnapKV (Li et al., 2024)
3. Roofline + Energy Modeling — based on Williams et al., 2009

## Research Papers
| Paper | Authors | Venue |
|-------|---------|-------|
| PagedAttention/vLLM | Kwon et al. | SOSP 2023 |
| KVQuant | Hooper et al. | 2024 |
| SnapKV | Li et al. | 2024 |
| FlashAttention-2 | Dao | ICLR 2024 |
| Roofline Model | Williams et al. | CACM 2009 |

## Hardware Used
- Google Colab T4 GPU
- Python 3.10, CUDA 11.8

## Setup
```bash
pip install -r requirements.txt
```

## Reproducing Results

| Experiment | Script | Output |
|---|---|---|
| KV Quantization | `python src/kv_quantization.py` | `results/tables/quant_results.csv` |
| Sliding Window | `python src/sliding_attention.py` | `results/tables/window_results.csv` |
| Energy Model | `python src/energy_model.py` | `results/tables/energy_results.csv` |
| All Charts | `python src/plot_results.py` | `results/figures/` |
| Roofline | `python src/roofline.py` | `results/figures/roofline_model.png` |

## Key Results (GPT-2 Medium, T4 GPU)

### KV Cache Quantization (seq_len=512)
| Method | Peak Memory | Throughput | Perplexity |
|--------|------------|------------|------------|
| FP16 (baseline) | 1533 MB | 45.88 tok/s | 47.10 |
| INT8 | 1396 MB | 11.50 tok/s | 47.04 |
| INT4 | 1260 MB | 27.52 tok/s | 49.98 |

### Sliding Window Attention (seq_len=512)
| Window | KV Memory Reduction | Throughput |
|--------|-------------------|------------|
| W=64 | 87.5% | 24.02 tok/s |
| W=128 | 75.0% | 28.80 tok/s |
| W=256 | 50.0% | 22.02 tok/s |
| Full | 0% | 27.73 tok/s |

### Energy per Token (seq_len=512)
| Method | Energy (μJ) | Reduction vs FP16 |
|--------|------------|-------------------|
| FP16 | 10066.33 | baseline |
| INT8 | 5033.16 | 50% |
| INT4 | 2516.58 | 75% |

## Repository Structure
```
memory-centric-llm/
├── src/
│   ├── kv_quantization.py
│   ├── sliding_attention.py
│   ├── roofline.py
│   ├── energy_model.py
│   ├── memory_profiler.py
│   └── plot_results.py
├── notebooks/
│   ├── 01_baseline_memory.ipynb
│   ├── 02_kv_quantization.ipynb
│   ├── 03_sliding_window.ipynb
│   ├── 04_roofline_model.ipynb
│   ├── 05_combined_results.ipynb
│   └── 06_energy_model.ipynb
├── results/
│   ├── tables/
│   │   ├── quant_results.csv
│   │   ├── perplexity_results.csv
│   │   ├── window_results.csv
│   │   └── energy_results.csv
│   └── figures/
│       ├── quant_memory_throughput.png
│       ├── window_memory_reduction.png
│       ├── perplexity_comparison.png
│       ├── roofline_model.png
│       ├── energy_per_token.png
│       └── combined_results.png
├── requirements.txt
└── paper_summaries.md
```

## Team
- Person A — Writing & Modeling Lead
- Person B — Implementation & Repo Lead