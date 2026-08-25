# Axiom-LM (124M)

A ground-up architectural and systems-level redesign of the 124-million-parameter autoregressive Transformer, engineered from low-level GPU tensor operations and memory hierarchies up to global pretraining dynamics.

---

## Overview

### What is Axiom-LM?
**Axiom-LM** is an engineered-from-scratch decoder-only language model and high-throughput pretraining engine. Rather than relying on high-level library abstractions or third-party wrappers, Axiom-LM reconstructs the entire autoregressive modeling stack directly from tensor primitives:
- **Custom Model Architecture**: Explicit implementation of causal multi-head self-attention, pre-layer normalization residual streams, parameter initialization scaling, and weight-tied token embeddings.
- **Hardware-Aware Memory & Compute Hierarchy**: Native integration with low-level GPU mechanics, including on-chip SRAM tiled attention, Ampere/Hopper TensorFloat-32 (TF32) precision, and Bfloat16 (BF16) mixed precision.
- **Compiled Execution Graph**: Kernel fusion and graph-level optimizations via PyTorch 2.x TorchInductor (`torch.compile`) targeting minimal Python dispatch overhead and zero redundant global memory roundtrips.
- **Systems-Optimized Training Engine**: Gradient accumulation with micro-batching, decoupled AdamW optimizer state management (2D tensor decay vs. 1D bias exclusion), and cosine learning rate decay with linear warmup.

### Why Axiom-LM?
1. **Hardware-Level Tensor Co-Design**: Modern LLM performance is predominantly memory-bandwidth bound. Axiom-LM addresses this directly at the hardware boundary—minimizing High Bandwidth Memory (HBM) traffic through fused FlashAttention kernels, aligning tensor dimensions for Tensor Core warp utilization, and leveraging hardware-native numeric formats.
2. **First-Principles Systems Architecture**: By controlling every layer of the compute stack—from byte-pair encoding (BPE) streaming to gradient synchronization and backward graphs—Axiom-LM provides full transparency and deterministic control over compute throughput and memory allocation.
3. **Foundation for Advanced Model Research**: Serves as a modular, high-performance platform for evaluating architectural evolutions (Rotary Positional Embeddings, RMSNorm, SwiGLU activations, Grouped-Query Attention) and next-generation matrix optimizers (such as Muon).

---

## Architecture & Tensor Flow

```
                   Input Token IDs (B, T)
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   Token Embedding (wte)           Position Embedding (wpe)
       (50,257 × 768)                   (1,024 × 768)
            └────────────────┬────────────────┘
                             ▼
                 x = wte(idx) + wpe(pos)
                             │
            ┌────────────────▼────────────────┐
            │     Transformer Block (×12)     │
            │   ├── LayerNorm (Pre-LN)        │
            │   ├── Causal Self-Attention     │ ◄── Fused SDPA / SRAM Tiling
            │   ├── Residual Connection (+)   │
            │   ├── LayerNorm                 │
            │   ├── MLP (GELU, 4× Expansion)  │ ◄── 768 → 3072 → 768
            │   └── Residual Connection (+)   │
            └────────────────┬────────────────┘
                             ▼
                      Final LayerNorm
                             ▼
                     Linear Output Head  ◄── Memory Tied to wte
                             ▼
                    Logits (B, T, 50,257)
```

### Technical Specifications

| Parameter | Symbol | Dimension / Value | Engineering Context |
| :--- | :---: | :---: | :--- |
| **Layers** | $L$ | `12` | Stacked decoder blocks |
| **Hidden Dimension** | $d_{\text{model}}$ | `768` | Model state capacity |
| **Attention Heads** | $N_h$ | `12` | Subspace attention splits |
| **Head Dimension** | $d_k$ | `64` | $768 / 12 = 64$ (Standard Tensor Core GEMM alignment) |
| **Context Length** | $T$ | `1024` | Maximum sequence length |
| **Vocabulary Size** | $V$ | `50,257` | Byte-Pair Encoding (BPE) vocabulary |
| **Parameter Count** | $P$ | `124,439,808` | Total parameters with tied embeddings |

---

## Systems & Kernel Optimizations

### 1. Fused Attention & SRAM Tiling
- Replaces naive $O(T^2)$ attention matrices with `F.scaled_dot_product_attention`.
- Blocks queries, keys, and values into on-chip SRAM partitions ($19\text{ TB/s}$ bandwidth), computing intermediate softmax statistics online and avoiding large intermediate activations in global VRAM.

### 2. Mixed Precision Arithmetic (TF32 & BF16)
- **TF32 Matmul**: Uses `torch.set_float32_matmul_precision('high')` to truncate mantissas to 10 bits while retaining the 8-bit dynamic exponent range for accelerated matrix multiplication on Ampere+ architectures.
- **BF16 Autocast**: Halves activation and gradient memory footprints without the dynamic range degradation and loss-scaling instability of standard IEEE FP16.

### 3. Graph Compilation (`torch.compile`)
- Utilizes TorchDynamo to trace computation graphs and TorchInductor to emit optimized Triton C++/CUDA kernels.
- Fuses pointwise operations (LayerNorm, GELU, residual additions) to eliminate back-to-back kernel launches and global memory roundtrips.

### 4. Decoupled Optimizer Parameter Partitioning
- Parameter tensors are categorized by dimensional rank:
  - **Rank $\ge 2$ (Weights, Projections)**: Decoupled weight decay ($0.1$).
  - **Rank $< 2$ (Biases, Normalization Scales)**: Zero weight decay.
- Fused AdamW updates parameters directly in a single pass over memory.

### 5. Depth-Dependent Residual Initialization
- Projection layers (`c_proj`) are scaled at initialization by $\sigma = \frac{0.02}{\sqrt{2L}}$ ($L=12$), stabilizing activation variance growth through the residual stream during early training iterations.

---

## Repository Structure

```
├── brain/
│   ├── train_gpt2.py       # Core Axiom-LM model, data loader, and training engine
│   └── play.ipynb          # Interactive experimentation, weight validation, and sampling
├── material/               # Mathematical formulations, systems analyses, and technical guides
│   ├── adamw_optimizer_guide.md
│   ├── automatic_mixed_precision_amp_guide.md
│   ├── distributed_data_parallel_ddp_guide.md
│   ├── flash_attention_guide.md
│   ├── gpt3_training_hyperparameters_guide.md
│   ├── torch_compile_guide.md
│   └── ... (research papers & literature)
├── pyrightconfig.json      # Static type-checking configuration
├── requirements.txt        # Minimal environment dependencies
└── README.md
```

---

## Quickstart

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/EyadNada/GPT-2.0-124M.git
cd GPT-2.0-124M

# Initialize Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Execution

The engine detects available hardware targets (`cuda`, `mps`, or `cpu`) and configures memory and precision contexts automatically:

```bash
python brain/train_gpt2.py
```

### 3. Interactive Analysis

Use the Jupyter environment for layer-by-layer inspection, tensor output checks, and generation sampling:

```bash
jupyter notebook brain/play.ipynb
```

---

## Technical Documentation (`material/`)

Reference documentation covering the mathematics and low-level engineering principles:

| Document | Area | Scope |
| :--- | :--- | :--- |
| [DDP Guide](material/distributed_data_parallel_ddp_guide.md) | Distributed Computing | Ring All-Reduce, gradient bucketing, and multi-GPU synchronization |
| [FlashAttention Guide](material/flash_attention_guide.md) | GPU Microarchitecture | SRAM tiling, online softmax, and IO complexity |
| [AMP Guide](material/automatic_mixed_precision_amp_guide.md) | Numerical Precision | Dynamic ranges, IEEE FP32 vs TF32 vs BF16 |
| [Torch Compile Guide](material/torch_compile_guide.md) | Compiler & JIT | TorchDynamo, TorchInductor, and fused Triton kernels |
| [AdamW Guide](material/adamw_optimizer_guide.md) | Optimization Theory | Decoupled weight decay vs L2 regularization dynamics |
| [Hyperparameters Guide](material/gpt3_training_hyperparameters_guide.md) | Training Dynamics | Learning rate schedules, warmup steps, and token throughput |
| [Pretraining Datasets](material/datasets_webtext_gpt2_vs_gpt3_guide.md) | Data Engineering | Dataset pipelines (WebText, FineWeb-Edu, SlimPajama) |

---

## Development Roadmap

- [x] Pure PyTorch Axiom-LM (124M) core architecture
- [x] Hugging Face parameter matching and verification
- [x] Low-level system optimizations (TF32, BF16 Autocast, FlashAttention / SDPA)
- [x] TorchInductor graph compilation (`torch.compile`)
- [x] Gradient accumulation and cosine warmup learning rate schedule
- [ ] Memory-mapped sharded binary data pipeline (TinyStories / FineWeb-Edu)
- [ ] Holdout validation loss evaluation loop
- [ ] Checkpoint persistence and resume mechanics (`torch.save`)
- [ ] Key-Value (KV) cache inference engine
- [ ] Modern architecture enhancements (RoPE, RMSNorm, SwiGLU, GQA)
- [ ] Muon matrix optimizer integration

---

## License

This project is released under the MIT License.
