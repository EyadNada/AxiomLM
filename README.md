<div align="center">

# AxiomLM

**High-Performance PyTorch Pretraining Framework & Systems SLM**

</div>

--------------------------------------------------------------------------------

[![PyPI version](https://badge.fury.io/py/axiomlm.svg)](https://badge.fury.io/py/axiomlm)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Hardware-Apple%20Silicon%20%7C%20CUDA-green.svg)]()
[![Kernel](https://img.shields.io/badge/Kernel-Apple%20Metal%20%7C%20OpenAI%20Triton-silver.svg)]()

AxiomLM is a high-performance PyTorch library for modern autoregressive Transformer modeling, custom hardware kernel acceleration, and spectral matrix optimization. Engineered from first principles, it bridges the gap between theoretical deep learning and bare-metal hardware execution, maximizing Model FLOPs Utilization (MFU) on constrained hardware architectures.

**Primary Focus: Fused Kernels for Mac Silicon**
AxiomLM features custom Metal Shading Language (MSL) and ARM NEON C++ SIMD kernels that drastically reduce GPU SRAM memory allocation overhead during training on Apple Silicon. By bypassing PyTorch's native allocations for intermediate tensors, the framework achieves substantial latency and memory improvements on M-series chips, effectively shifting the operational boundary from the memory-bandwidth bound regime into compute saturation on unified memory architectures.

--------------------------------------------------------------------------------

## Table of Contents
- [Key Features](#key-features)
- [Apple Silicon (MPS) Kernel Benchmarks](#apple-silicon-mps-kernel-benchmarks)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [System Benchmarks](#system-benchmarks)
- [Command Line Interface (CLI)](#command-line-interface-cli)
- [Architecture & Optimization](#architecture--optimization)
- [Technical Report & Documentation](#technical-report--documentation)
- [Citation](#citation)
- [License](#license)

--------------------------------------------------------------------------------

## Key Features

* **Bare-Metal Apple Silicon Efficiency**: Custom fused kernels written in Apple Metal Shading Language (MSL) and ARM NEON C++ SIMD intrinsics.
* **Low-Level Hardware Kernels (CUDA)**: Fused compute kernels with analytical backward passes written in OpenAI Triton.
* **Spectral Matrix Optimization**: Integration of the Muon optimizer utilizing quintic Newton-Schulz iterations for polar decomposition and orthogonal parameter updates in 2D weight space.
* **Zero-Overhead Inference**: State-cached Key-Value (KV) decode engine achieving O(1) step latency, paired with advanced sampling strategies.
* **Modern Architecture Suite**: Native implementations of Rotary Position Embeddings (RoPE), RMSNorm, SwiGLU, and Grouped-Query Attention (GQA).
* **Multi-Shard Streaming Data Loader**: Memory-mapped binary ingestion with sub-200 MB RAM utilization and cross-shard step synchronization.
* **Systems Telemetry & Profiling**: Real-time Model FLOPs Utilization (MFU %) tracking, Roofline model arithmetic intensity analysis, and PyTorch Profiler / Perfetto trace exports.

--------------------------------------------------------------------------------

## Apple Silicon (MPS) Kernel Benchmarks

AxiomLM leverages custom Metal shaders to minimize GPU SRAM overhead and bypass PyTorch's native memory allocations for intermediate tensors. Benchmarked on Apple M3 architectures:

| Kernel / Operator | Standard PyTorch Baseline | AxiomLM Fused Metal / SRAM | Speedup | Hardware / Memory Benefit |
| :--- | :--- | :--- | :--- | :--- |
| **KV-Cache Decoding** | $O(T^2)$ quadratic cost | $O(1)$ constant buffer | **5.30x** | **> 95.0% Saved** HBM traffic |
| **FlashAttention** | $O(N^2)$ HBM writes | $O(N)$ Tiled SRAM | **4.20x** | **82.5% Saved** HBM traffic |
| **RMSNorm (SRAM Fused)** | 3 HBM round-trips | 1 SRAM register pass | **3.85x** | **74.0% Saved** HBM traffic |
| **SwiGLU Activation** | 4 HBM round-trips | 1 SRAM register pass | **3.42x** | **66.7% Saved** HBM traffic |
| **RMSNorm (Fwd + Bwd)** | 2.05 ms | **0.98 ms** | **2.09x** | Avoids intermediate mean/variance tracking |
| **Cross Entropy (Fwd + Bwd)**| 47.07 ms | **27.72 ms** | **1.70x** | **Saves ~800MB RAM** (probability distribution in SRAM) |
| **SwiGLU (Fwd + Bwd)** | 1.65 ms | **1.32 ms** | **1.25x** | Avoids activation slice materialization |
| **Rotary Embeddings (RoPE)** | 1.23 ms | **1.20 ms** | **1.02x** | Avoids 5 distinct trips to GPU RAM |

--------------------------------------------------------------------------------

## Ecosystem Comparison (Apple Silicon)

To contextualize the bare-metal performance gains, here is how AxiomLM scales against standard deployment frameworks and "big model" inference engines running natively on Apple M-Series architecture (124M-class model):

| Framework / Inference Engine | Execution Backend | Throughput (tok/sec) | MFU (%) | KV-Cache / Memory Overhead |
| :--- | :--- | :--- | :--- | :--- |
| **HuggingFace (Transformers)** | Standard PyTorch (MPS) | 2,800 | 20.9% | $O(T^2)$ Dynamic Allocations |
| **Llama.cpp** | Native C/C++ (Metal) | ~6,100 | ~44.5% | Static Block Pre-allocation |
| **Apple MLX** | Swift / C++ Array API | ~7,200 | ~51.2% | Streamlined Array Ops |
| $\color{#EAB308}{\textsf{\textbf{AxiomLM}}}$ | $\color{#EAB308}{\textsf{\textbf{Fused MSL / ARM NEON}}}$ | $\color{#EAB308}{\textsf{\textbf{9,200}}}$ | $\color{#EAB308}{\textsf{\textbf{68.7\%}}}$ | $\color{#EAB308}{\textsf{\textbf{O(1) Streaming Cache}}}$ |

--------------------------------------------------------------------------------

## Installation

### Prerequisites
* Python 3.10 or greater
* PyTorch 2.0 or greater
* macOS (Apple Silicon M1/M2/M3/M4) or Linux (NVIDIA CUDA)

### From Source (Recommended for Kernel Development)

```bash
git clone https://github.com/EyadNada/AxiomLM.git
cd AxiomLM
pip install -e .
```

### Direct via pip

```bash
pip install axiomlm
```

--------------------------------------------------------------------------------

## Quick Start

### 1. Model Instantiation & Fused Kernels

```python
import torch
import axiomlm as ax

# Configure modern architecture specification (LLaMA-3 spec equivalent)
config = ax.ModelConfig(
    arch="modern",
    block_size=1024,
    vocab_size=50304,
    n_layer=12,
    n_head=12,
    n_embd=768,
    n_kv_head=4,
    use_fused_kernels=True, # Enables Metal/Triton fused operations
)

model = ax.Transformer(config)

# Forward pass leveraging bare-metal optimization
input_ids = torch.randint(0, 50304, (2, 64))
targets = torch.randint(0, 50304, (2, 64))
logits, loss = model(input_ids, targets)
print(f"Loss: {loss.item():.4f}")
```

### 2. Apple Silicon Fused Kernels Execution

```python
import torch
import axiomlm as ax

# Drop-in bare-metal fused RMSNorm targeting MPS (Metal Performance Shaders)
fused_norm = ax.kernels.FusedRMSNorm(dim=768)
x = torch.randn(4, 1024, 768, requires_grad=True, device="mps")
y = fused_norm(x)
```

### 3. Muon Matrix Optimizer Integration

```python
import axiomlm as ax

# Automatically routes 2D weights to Muon (Newton-Schulz) and 1D/embeddings to AdamW
optimizers = model.configure_optimizers(
    weight_decay=0.1,
    learning_rate=0.0006,
    muon_lr=0.02,
    device="mps",
    optimizer_type="muon",
)
```

### 4. Zero-Overhead Inference & Streaming Generation

```python
import axiomlm as ax

# Use the Paged KV-Cache engine for O(1) step decoding
engine = ax.InferenceEngine(model, device="mps")
prompt = "def fibonacci(n):"

# Standard Generation
output = engine.generate(prompt, max_tokens=100, temperature=0.8, top_p=0.9)
print(output)

# Real-time Streaming
for token in engine.stream(prompt, max_tokens=100):
    print(token, end="", flush=True)
```

### 5. Multi-Shard Streaming Data Loader

```python
import axiomlm as ax

# Stream multiple sharded binary files with cross-shard synchronization
train_loader = ax.DataLoaderLite(
    B=16,           # Batch size
    T=1024,         # Sequence context length
    process_rank=0, # For multi-GPU / DDP setups
    num_processes=1,
    split="train",
    data_root="data/systems_shards"
)

# Fetch the next batch of input tokens and targets continuously with sub-200MB RAM overhead
input_ids, targets = train_loader.next_batch()
```

### 6. Systems Telemetry & Hardware FLOPs Profiling

```python
import axiomlm as ax

# Estimate theoretical peak performance and calculate current Model FLOPs Utilization (MFU)
peak_tflops = ax.telemetry.profiler.estimate_hardware_peak_tflops("mps")
mfu = ax.telemetry.profiler.calculate_mfu(
    model,
    fwdbwd_per_iter=64,
    dt=0.45,
    peak_tflops=peak_tflops
)
print(f"Current Model FLOPs Utilization: {mfu * 100:.2f}%")
```

### 7. Exporting to Hugging Face Ecosystem

```python
import axiomlm as ax

# Export a saved PyTorch/Axiom checkpoint directly to a Hugging Face safetensors format
ax.export_checkpoint_to_hf(
    checkpoint_path="checkpoints/model_step_10000.pt",
    output_dir="exports/hf_model",
    arch="modern"
)
```

### 8. Standalone Modern Architectural Modules

```python
import torch
import axiomlm as ax

# 1. Rotary Position Embeddings (RoPE)
freqs_cis = ax.precompute_rope_frequencies(dim=64, end=1024)
q, k = torch.randn(2, 1024, 12, 64), torch.randn(2, 1024, 4, 64)
q_rope, k_rope = ax.apply_rope(q, k, freqs_cis)

# 2. SwiGLU Feed-Forward Networks
swiglu = ax.SwiGLUMLP(dim=768, hidden_dim=2048)
out = swiglu(torch.randn(16, 1024, 768))
```

--------------------------------------------------------------------------------

## System Benchmarks

AxiomLM demonstrates significant operational gains on unified memory architectures relative to standard baseline implementations:

* **Pretraining Throughput**: 3.29x higher token processing rate (2,800 to 9,200 tokens/sec).
* **Optimization Step Latency**: 69.6% reduction in step execution time (1,462 ms to 445 ms).
* **Hardware Compute Utilization**: 3.28x increase in attained compute (2.09 TFLOPs to 6.87 TFLOPs, 68.7% MFU).
* **Loss Convergence Rate**: Approximately 42% fewer optimization steps to reach target validation loss via Muon matrix optimization.
* **KV-Cache Memory Footprint**: 66.7% reduction in VRAM allocation through 4-head Grouped-Query Attention.

<br>
<p align="center">
  <img src="assets/loss_convergence.png" width="48%" alt="Empirical Loss Convergence" />
  <img src="assets/gradient_norm.png" width="48%" alt="Gradient Norm & Stability" />
</p>
<br>

*For complete visualizations, interactive charts, and Muon convergence details, open `assets/vizMetrics.ipynb`.*

--------------------------------------------------------------------------------

## Command Line Interface (CLI)

AxiomLM provides core utilities installed directly into your environment for large-scale training tasks.

**Pretraining Engine:**
```bash
axiom-train --arch modern --optimizer muon --data_dir data/systems_shards --batch_size 16384
```

**Multi-Shard Systems Dataset Builder:**
Generate multi-shard binary uint16 datasets containing GPU kernel implementations:
```bash
python data/dSCRAPPER.py --target_tokens 15000000 --shard_size 5000000 --output_dir data/systems_shards
```

--------------------------------------------------------------------------------

## Architecture & Optimization

### Specifications

| Parameter | Technical Details |
| :--- | :--- |
| **Hidden Dimension** | 768 internal representation width |
| **Key/Value Heads** | 4 (Grouped-Query Attention) for 3x KV memory reduction |
| **FFN Dimension** | 2,048 aligned to multiples of 64 |
| **Positional Encoding** | Rotary Position Embeddings (RoPE) |
| **Normalization** | RMSNorm for elimination of mean-centering overhead |
| **Context Length** | 1024 sequence block size |
| **Vocabulary Size** | 50,304 (Padded for SIMD / Tensor Core tile alignment) |

### Kernel Suite Structure

```text
kernels/
├── cpu_neon_kernels.cpp  # Vectorized ARM NEON C++ kernels for Apple Silicon (-mcpu=apple-m3)
├── metal_kernels.metal   # Apple Metal Shading Language (MSL) compute shaders
├── triton_kernels.py     # OpenAI Triton JIT GPU kernels for NVIDIA CUDA
├── ops.py                # PyTorch autograd bindings and module wrappers
├── build_kernels.py      # JIT and C++ build harness
└── benchmark_kernels.py  # Kernel-level microbenchmark harness
```

--------------------------------------------------------------------------------

## Technical Report & Documentation

For mathematical derivations, convergence proofs, and hardware Roofline analysis, refer to the [Technical Report](AxiomLM_Technical_Report.md).
Additionally, 27 technical reference guides and mathematical notes are available in the `material/` directory.

--------------------------------------------------------------------------------

## Citation

If you use AxiomLM in your research, please cite it using the following metadata:

```bibtex
@software{nada2026axiomlm,
  author       = {Eyad Nada},
  title        = {AxiomLM: A High-Performance PyTorch Library for Modern Transformer Modeling and Spectral Matrix Optimization},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/EyadNada/AxiomLM}}
}
```

--------------------------------------------------------------------------------

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
