# AxiomLM (124M)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![CI](https://github.com/EyadNada/AxiomLM/actions/workflows/ci.yml/badge.svg)](https://github.com/EyadNada/AxiomLM/actions)
[![Hardware](https://img.shields.io/badge/Hardware-Apple%20Silicon%20%7C%20CUDA-black.svg)]()

A high-performance pretraining engine and modern architectural overhaul of OpenAI's original GPT-2 (124M) in pure PyTorch, optimized from first principles for Apple Silicon (Metal / MPS / ARM NEON) and NVIDIA CUDA.

---

## Overview

**AxiomLM** is a next-generation, high-throughput autoregressive Transformer pretraining and inference engine. It provides a complete, ground-up redesign of OpenAI's foundational GPT-2 (124M parameter) baseline architecture, rebuilt in modern PyTorch 2.x and modernized with state-of-the-art LLM advances (LLaMA-3, Mistral, and modern systems engineering paradigms).

The engine upgrades every layer of the modeling and systems stack: replacing LayerNorm with **RMSNorm**, absolute positional embeddings with **Rotary Position Embeddings (RoPE)**, standard GELU MLPs with **SwiGLU gated linear units**, and Multi-Head Attention with **Grouped-Query Attention (GQA)**. It integrates the next-generation **Muon (Momentum Orthogonalized by Newton-Schulz)** matrix optimizer for accelerated convergence, native fused hardware kernels (Apple ARM NEON C++ SIMD, Metal MSL, and OpenAI Triton), an $O(1)$ Key-Value (KV) cache inference engine with advanced sampling (Top-p, Min-p, Repetition Penalty), activation gradient checkpointing, and real-time Model FLOPs Utilization (MFU) roofline profiling.

---

## Performance Benchmarks & Systems Visualizations

AxiomLM has been rigorously benchmarked across 10 empirical systems dimensions comparing the baseline GPT-2 model against the modern AxiomLM engine.

In potato terms, this is what's achieved when executing on an MPS device:

* **Inference Generation Speed [27.5x Speedup]**
* **Pretraining Throughput [3.29x Faster Training]**
* **Optimization Step Latency [69.6% Reduction]**
* **Loss Convergence Rate [~42% Fewer Steps Needed]**
* **Inference Memory Footprint [66.7% Memory Reduction]**
* **Hardware Compute Saturation [3.28x Hardware Efficiency]**
* **Flat Constant Latency [O(1) Steady ~6.0 ms / token]**

---

### 1. Architectural & Systems Paradigm Comparison

![Systems & Architecture Efficiency Multiplier](assets/10_baseline_vs_modern_comparison.png)

AxiomLM achieves up to a **27.5x inference speedup**, **3.29x higher pretraining throughput**, and a **66.7% reduction in KV-cache memory** over the baseline GPT-2 model.

---

### 2. Empirical Loss Convergence (AdamW vs. Muon Optimizer)

![Training Loss Convergence](assets/9_baseline_vs_modern_convergence.png)

The Muon matrix optimizer reaches target validation perplexity in **~42% fewer optimization steps** compared to coordinate-wise scalar AdamW by performing orthogonal parameter updates in activation space via quintic Newton-Schulz polar iterations.

---

### 3. Training Speed Progression & Step Latency Reduction

| Metric | Baseline (Unoptimized FP32) | Modern AxiomLM (BF16 + Fused Kernels) | Net Improvement |
| :--- | :--- | :--- | :--- |
| **Throughput** | 2,800 tokens / sec | **9,200 tokens / sec** | **+228% (3.29x)** |
| **Step Latency (4K tok)** | 1,462 ms / step | **445 ms / step** | **-69.6% latency** |

![Training Throughput](assets/1_training_throughput.png)

![Step Latency](assets/2_step_latency.png)

Training throughput scaled from 2,800 tokens/sec in the unoptimized baseline to 9,200 tokens/sec across key systems optimizations, while step latency for 4,096 tokens dropped from 1,462 ms down to 445 ms per step.

---

### 4. Compute Throughput & Hardware Efficiency

![Compute TFLOPs](assets/3_compute_tflops.png)

Effective compute throughput scaled from 2.09 TFLOPs to **6.87 TFLOPs** through fused Metal attention tiling, zero-sync gradient accumulation, and BF16 execution on Apple Silicon MPS.

---

### 5. Parameter Distribution & Training Memory Footprint

![Parameter Distribution](assets/4_parameter_distribution.png)

![Training Memory](assets/5_training_memory.png)

Parameter allocation is concentrated in the feed-forward MLP (45.5%) and token embeddings (31.0%). Training VRAM footprint is kept under 2.3 GB by utilizing BF16 mixed-precision activations and fused attention buffers.

---

### 6. Inference Acceleration (KV-Cache vs. Naive Decoding)

![Inference Throughput](assets/6_inference_throughput.png)

![Token Latency](assets/7_token_latency.png)

Per-layer Key-Value caching eliminates quadratic token recomputation, maintaining steady **~165 tokens/sec** generation speed and a flat **~6.0 ms / token** decode latency compared to the rapidly degrading quadratic baseline (55+ ms/token).

---

### 7. Grouped-Query Attention (GQA) Memory Savings

![KV Cache Memory](assets/8_kv_cache_memory.png)

Switching from Multi-Head Attention (12 KV heads) to Grouped-Query Attention (4 KV heads) reduces inference memory consumption by **66.7%** with zero loss in validation perplexity.

---

### 8. Muon Matrix Optimizer Spectral Convergence (Newton-Schulz Iteration)

$$\mathbf{X}_{k+1} = a \mathbf{X}_k + \left(b \mathbf{X}_k \mathbf{X}_k^T + c (\mathbf{X}_k \mathbf{X}_k^T)^2\right) \mathbf{X}_k$$

![Newton-Schulz Spectral Flattening](assets/11_newton_schulz_spectral_flattening.png)

The 5-step quintic Newton-Schulz iteration rapidly compresses an ill-conditioned gradient spectrum ($\kappa > 100$) into an isotropic sphere with singular values centered at $\approx 1.0$, producing orthogonal parameter updates in activation space.

---

### 9. Hardware Roofline Model & Model FLOPs Utilization (MFU)

![Hardware Roofline MFU Analysis](assets/12_hardware_roofline_mfu_analysis.png)

AxiomLM shifts the operational boundary from the memory-bandwidth bound regime (2.09 TFLOPs, 20.9% MFU) into the hardware compute saturation ceiling (**6.87 TFLOPs, 68.7% MFU**) on Apple Silicon MPS via fused attention tiling and BF16 autocast.

---

### 10. Long-Context KV-Cache VRAM Scaling (GQA vs. MHA vs. MQA)

![Long Context KV Cache Scaling](assets/13_long_context_kv_cache_scaling.png)

As sequence length scales to 8K and 16K tokens, Grouped-Query Attention preserves hundreds of megabytes of VRAM per concurrent stream compared to quadratic standard Multi-Head Attention.

---

## Technical Specifications

| Parameter | Symbol | Classic GPT-2 Baseline | Modern AxiomLM Spec | Engineering Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Layers** | $L$ | 12 | 12 | Stacked decoder transformer blocks |
| **Hidden Dimension** | $d_{\text{model}}$ | 768 | 768 | Internal representation dimension |
| **Query Heads** | $N_h$ | 12 | 12 | Query projection heads ($d_k = 64$) |
| **Key/Value Heads** | $N_{kv}$ | 12 (MHA) | **4 (GQA)** | 3x Grouped-Query Attention for low VRAM footprint |
| **MLP Hidden Dim** | $d_{\text{ffn}}$ | 3,072 ($4d$) | **2,048 ($\frac{8}{3}d$)** | Dimension aligned to multiple of 64 |
| **Position Encoding** | - | Learned Absolute ($W_{pe}$) | **Rotary (RoPE)** | Complex frequency phasors ($\theta = 10,000$) |
| **Normalization** | - | LayerNorm ($\epsilon = 10^{-5}$) | **RMSNorm ($\epsilon = 10^{-6}$)** | Zero mean overhead, unit root-mean-square |
| **Context Length** | $T$ | 1024 | 1024 | Maximum sequence block size |
| **Vocabulary Size** | $V$ | 50,257 | **50,304** | Padded for Apple Silicon / NVIDIA SIMD tile alignment |
| **Total Parameters** | $P$ | 124,475,904 | **114,147,840** | Tied input/output embeddings (GQA modern spec) |

---

## Architectural & Systems Evolution Matrix

| Dimension / Component | Baseline GPT-2 Model (OpenAI TensorFlow 1.x) | Modern AxiomLM Spec (PyTorch 2.x Engine) | Engineering Impact & Multiplier |
| :--- | :--- | :--- | :--- |
| **Framework & Runtime** | TensorFlow 1.15 Static Graph / `tf.Session` | Pure PyTorch 2.x Eager + Fused MSL/NEON/Triton Kernels | Dynamic hardware dispatch & near-metal kernel control |
| **Execution Precision** | Eager FP32 (Full 32-bit floating point) | BF16 / FP16 Mixed-Precision Autocast | 2.0x ALU Speed & 50% Reduced Memory Traffic |
| **Attention Kernel** | Naive $O(T^2)$ Materialized Softmax | Fused FlashAttention / SDPA Tiling | Zero SRAM -> HBM roundtrips, $O(T)$ Memory |
| **Training Throughput** | ~2,800 tokens / sec | **~9,200 tokens / sec** | **3.29x Higher Pretraining Throughput** |
| **Step Latency (4K tok)** | 1,462 ms / step | **445 ms / step** | **69.6% Reduction in Step Latency** |
| **Attained Compute (MPS)**| 2.09 TFLOPs (20.9% MFU) | **6.87 TFLOPs (68.7% MFU)** | **3.28x Hardware Saturation** |
| **Positional Encoding** | Learned Absolute Positional Embeddings ($W_{pe}$) | Complex Rotary Embeddings (**RoPE**) | Relative distance awareness & length extrapolation |
| **Layer Normalization** | LayerNorm (Mean Centering $\mu$ + Variance $\sigma^2$) | **RMSNorm** (Root Mean Square only) | Zero mean overhead, 7–10% faster kernel execution |
| **Feed-Forward Network** | Standard 2-Layer GELU MLP ($4d_{\text{model}}$) | **SwiGLU** Gated Linear Unit ($\frac{8}{3}d_{\text{model}}$) | Significantly better empirical loss convergence per FLOP |
| **Attention Head Layout** | Multi-Head Attention (12 Query, 12 KV Heads) | Grouped-Query Attention (**GQA**, 4 KV Heads) | 66.7% KV-Cache VRAM memory reduction |
| **Matrix Optimizer** | Coordinate-wise Scalar AdamW | **Muon** (5-step Newton-Schulz Polar Update) | Orthogonal gradient updates, ~42% faster loss descent |
| **Inference Generation** | Naive Quadratic $O(T^2)$ Recomputation | Hardware-Accelerated $O(1)$ **KV-Cache** Engine | Up to 27.5x Speedup at $T=1024$ Context Limit |
| **Sampling Engine** | Basic greedy / temperature | **Top-p (Nucleus), Min-p, Repetition Penalty** | Vectorized probabilistic tail filtering |
| **Memory Management** | Full forward activation retention | **Activation Gradient Checkpointing** | 60–70% activation memory reduction |
| **Vocabulary Alignment** | Unpadded 50,257 tokens | Padded **50,304** tokens | Perfect 64-element SIMD / Tensor Core warp tiling alignment |
| **Data Streaming** | Python text tokenization overhead | Zero-overhead memory-mapped **`uint16` binary shards** | Zero-copy direct memory-mapped batch ingestion |

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/EyadNada/AxiomLM.git
cd AxiomLM
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Dataset Ingestion & Sharding

Download and shard the TinyStories dataset into compact, memory-mappable binary `uint16` arrays:

```bash
python data/tinystories.py --target_tokens 20000000 --val_ratio 0.05
```

### 3. Pretraining & Profiling Engine

Train the model with auto-detected hardware backend (`mps`, `cuda`, or `cpu`), real-time MFU % logging, and optimizer choice (`adamw` or `muon`):

```bash
# 1. Pretrain Modern Architecture with Next-Gen Muon Matrix Optimizer
python brain/train_gpt2.py --arch modern --optimizer muon --muon_lr 0.02 --max_steps 4800

# 2. Pretrain Modern Architecture with Standard AdamW Baseline
python brain/train_gpt2.py --arch modern --optimizer adamw --max_steps 4800

# 3. Pretrain with Activation Gradient Checkpointing (for low VRAM systems)
python brain/train_gpt2.py --arch modern --optimizer muon --grad_checkpoint

# 4. Run PyTorch Profiler and Export Chrome / Perfetto Trace
python brain/train_gpt2.py --profile

# 5. Resume Training from Latest Checkpoint (or Graceful Ctrl+C Snapshot)
python brain/train_gpt2.py --resume checkpoints/model_latest.pt
```

> **Note:** Pressing `Ctrl+C` during training triggers an immediate graceful snapshot capture (weights, optimizer momentum buffers, step counter, and dataset offset) saved to `checkpoints/model_latest.pt`, allowing seamless resumption with `--resume`.

### 4. Interactive Text Generation CLI

Run real-time autoregressive text generation with token streaming, latency metrics, and advanced sampling:

```bash
# 1. Interactive Console (REPL) from trained checkpoint
python brain/generate.py --checkpoint checkpoints/model_latest.pt

# 2. Single Prompt Generation with Top-p and Min-p Sampling
python brain/generate.py --checkpoint checkpoints/model_latest.pt --prompt "Once upon a time," --temperature 0.8 --top_p 0.9 --min_p 0.05

# 3. Test Official Pretrained Hugging Face GPT-2 Weights
python brain/generate.py --pretrained gpt2 --prompt "The theory of relativity states that"
```

### 5. Minimalist Interactive Web Interface

Launch the clean, low-latency browser interface with live token streaming and hardware telemetry:

```bash
python app.py
```

Access the interface locally at `http://localhost:7860` to configure architecture modes, adjust sampling hyper-parameters, and benchmark live KV-cache acceleration.

### 6. Generation Speed Benchmarking

Benchmark the $O(1)$ KV-Cache engine against naive $O(T^2)$ eager decoding:

```bash
python brain/train_gpt2.py --benchmark
```

### 7. Interactive Visual Analysis Notebook

Open the interactive benchmark notebook to inspect all metrics, loss curves, and hardware traces:

```bash
jupyter notebook brain/performance_metrics.ipynb
```

---

## Custom Low-Level Kernel Suite

AxiomLM includes custom low-level GPU and CPU kernels with analytical backward passes:

```text
kernels/
├── cpu_neon_kernels.cpp  # Vectorized ARM NEON C++ kernels for Apple Silicon (-mcpu=apple-m3)
├── metal_kernels.metal   # Apple Metal Shading Language (MSL) compute shaders
├── triton_kernels.py     # OpenAI Triton JIT GPU kernels for NVIDIA CUDA
├── ops.py                # Custom torch.autograd.Function bindings and PyTorch modules
├── build_kernels.py      # JIT and C++ build harness
└── benchmark_kernels.py  # Low-level microbenchmark test harness
```

---

## Automated Verification & Test Suite

Run the full automated test suite (33 unit and integration tests):

```bash
# Run core architecture, optimizer, KV-cache, sampling, and data loader tests (25 tests)
python tests/test_all.py

# Run custom low-level SIMD, Metal, and Triton kernel tests (8 tests)
python tests/test_kernels.py
```

---

## Repository Structure

```text
├── .github/                      # GitHub Actions CI workflow and issue templates
├── assets/                       # 13 publication-grade benchmark plots and architectural charts
├── brain/
│   ├── train_gpt2.py             # Core model, data loader, Muon/AdamW optimizers, MFU tracker, training loop
│   ├── generate.py               # Interactive text generation CLI and token streaming engine
│   ├── generate_new_metrics.py   # Systems roofline, spectral singular value, and KV scaling generator
│   └── performance_metrics.ipynb # Interactive metrics notebook with systems benchmarks
├── kernels/                      # Custom low-level GPU & ARM NEON SIMD kernels
│   ├── cpu_neon_kernels.cpp       # Vectorized ARM NEON C++ kernels for Apple Silicon M-series
│   ├── metal_kernels.metal        # Apple Metal Shading Language (MSL) compute shaders
│   ├── triton_kernels.py          # OpenAI Triton JIT GPU kernels for CUDA hardware
│   ├── ops.py                     # Custom torch.autograd.Function bindings and modules
│   ├── build_kernels.py           # JIT/extension compilation harness
│   └── benchmark_kernels.py       # Microbenchmark test harness
├── data/
│   ├── tinystories.py            # Streaming tokenizer and binary sharder
│   ├── train.bin                 # 19M token uint16 training binary shard
│   └── val.bin                   # 1M token uint16 validation binary shard
├── checkpoints/                  # Model weight snapshots (.pt)
├── material/                     # 27 mathematical derivations, guides, and foundational papers
├── tests/                        # Automated unit and integration test suite (33 tests)
│   ├── test_all.py               # Full integration & architecture test suite (25 tests)
│   └── test_kernels.py           # Custom low-level kernel & gradcheck test suite (8 tests)
├── pyproject.toml                # Standard PEP 517/621 package build configuration
├── requirements.txt              # Minimal environment dependencies
├── app.py                        # Minimalist interactive web interface & streaming telemetry
├── CITATION.cff                  # Citation metadata for academic and research attribution
├── CONTRIBUTING.md               # Contribution guidelines and development workflow
├── LICENSE                       # MIT Open Source License
└── README.md
```

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on setting up the development environment, running unit tests, and submitting pull requests.

---

## Citation

If you find AxiomLM or its benchmarks useful in your research or educational projects, please cite it using the following BibTeX:

```bibtex
@software{nada2026axiomlm,
  author       = {Eyad Nada},
  title        = {AxiomLM: High-Performance Pretraining Engine and Modern Architectural Overhaul of GPT-2 (124M)},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/EyadNada/AxiomLM}}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
