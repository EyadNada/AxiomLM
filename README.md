# Axiom-LM (124M)

A high-performance, Apple Silicon (Metal Performance Shaders / MPS) and Unified Memory Architecture (UMA) native pretraining engine and architectural redesign of the 124M parameter autoregressive Transformer.

---

## Overview

### What is Axiom-LM?
**Axiom-LM** is an engineered-from-scratch decoder-only language model and pretraining engine optimized specifically for Apple Silicon and cross-platform hardware. While the broader deep learning ecosystem is predominantly hardwired for NVIDIA CUDA, Axiom-LM redesigns the complete autoregressive modeling and execution stack to treat Apple's Unified Memory Architecture (UMA) and Metal Performance Shaders as first-class compute targets:
- **Custom Model Architecture**: Pure PyTorch implementation of causal multi-head self-attention, pre-layer normalization residual streams, variance-scaled weight initialization, and tied input/output embedding matrices.
- **Apple Silicon & MPS Native Execution**: Direct integration with PyTorch's Metal Performance Shaders backend, including native `bfloat16` autocasting, Metal-accelerated Scaled Dot-Product Attention (SDPA), deterministic `torch.mps` synchronization, and zero-CPU-GPU-sync micro-batch loss accumulation.
- **Unified Memory Optimization**: Zero-copy `uint16` memory-mapped (`np.memmap`) data streaming architectures designed to exploit Apple Silicon's shared CPU-GPU memory pool, avoiding memory bloat and PCIe transfer bottlenecks.
- **High-Throughput Training Dynamics**: Gradient accumulation with micro-batch sizing tuned for unified memory pressure (16,384 tokens/step), decoupled AdamW optimizer state management (2D weight decay vs. 1D bias exclusion), cosine learning rate decay with linear warmup, periodic holdout validation loss evaluation, live text generation sampling, and model checkpointing (`.pt`).

### Why Axiom-LM? (The Systems Rationale)
1. **Breaking CUDA Lock-In**: Modern LLM pretraining frameworks (Triton, FlashAttention-CUDA, CUTLASS, Megatron) are tightly coupled to NVIDIA hardware. Axiom-LM establishes a high-performance alternative engineered for Apple Silicon's unified memory bandwidth (up to 800+ GB/s on Max/Ultra configurations).
2. **Unified Memory Exploitation**: In traditional discrete GPU setups, data must traverse high-latency PCIe buses from host RAM to VRAM. Apple Silicon shares a single high-bandwidth physical address space between CPU and GPU cores. Axiom-LM leverages zero-copy memory-mapped token arrays directly inside this unified pool.
3. **First-Principles Transparency**: Complete control over every tensor operation—from Byte-Pair Encoding (BPE) token sharding to gradient backward graphs and optimizer updates—without opaque library wrappers.
4. **Foundation for Modern Architecture Research**: Serves as a modular, hackable platform for integrating modern Transformer components (RoPE, RMSNorm, SwiGLU, GQA) and next-generation optimizers (such as Muon).

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
            │   ├── Causal Self-Attention     │ ◄── Metal Accelerated SDPA
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
| **Head Dimension** | $d_k$ | `64` | $768 / 12 = 64$ (Standard GEMM tile alignment) |
| **Context Length** | $T$ | `1024` | Maximum sequence length |
| **Vocabulary Size** | $V$ | `50,257` | GPT-2 Byte-Pair Encoding (BPE) vocabulary |
| **Parameter Count** | $P$ | `124,439,808` | Total parameters with tied embeddings |

---

## Hardware-Level Systems Optimizations

### 1. Metal Performance Shaders (MPS) & SDPA
- Leverages PyTorch's native MPS backend for `F.scaled_dot_product_attention`, executing causal self-attention through Apple Silicon's hardware matrix engines.
- Eliminates materializing the quadratic $(B, N_h, T, T)$ attention score matrix in global memory, computing softmax normalizers incrementally.

### 2. Native Bfloat16 Precision Contexts
- Executes forward activations in `bfloat16` via `torch.autocast(device_type="mps", dtype=torch.bfloat16)` on supported Apple Silicon chips.
- Cuts memory bandwidth consumption by 50% while preserving the dynamic exponent range of FP32, preventing numerical instability without requiring artificial loss scaling.

### 3. Asynchronous On-Device Loss Accumulation (Zero CPU-GPU Sync)
- Loss values are accumulated directly as device tensors during gradient accumulation micro-steps, eliminating per-microstep `.item()` CPU flushes that throttle the MPS command buffer pipeline.

### 4. Zero-Copy Memory-Mapped Sharding (`DataLoaderLite`)
- Token arrays are stored as raw binary `uint16` shards (`data/train.bin`, `data/val.bin`) and loaded via zero-copy `np.memmap`.
- Slices are converted directly to PyTorch tensors with zero CPU allocation overhead.

### 5. Decoupled Optimizer Parameter Partitioning
- Parameter tensors are categorized by dimensional rank:
  - **Rank $\ge 2$ (Weight Matrices, Projections)**: Decoupled weight decay ($0.1$).
  - **Rank $< 2$ (Biases, Normalization Scales)**: Zero weight decay.

### 6. Depth-Scaled Residual Initialization
- Projection layers (`c_proj`) are scaled at initialization by $\sigma = \frac{0.02}{\sqrt{2L}}$ ($L=12$), stabilizing activation variance through the residual stream during training.

---

## Repository Structure

```
├── brain/
│   ├── train_gpt2.py       # Core Axiom-LM model, data loader, validation, sampling, and training engine
│   └── play.ipynb          # Interactive experimentation, weight validation, and sampling
├── data/
│   ├── tinystories.py      # High-speed streaming tokenizer & binary sharder
│   ├── train.bin           # 19M token uint16 training binary shard
│   └── val.bin             # 1M token uint16 holdout validation binary shard
├── checkpoints/            # Saved model weights, optimizer states, and training snapshots
├── material/               # Mathematical formulations, systems analyses, and technical guides
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

### 2. Dataset Preparation (TinyStories 20M)

Download, tokenize, and shard the dataset into binary `uint16` memory-mappable arrays:

```bash
python data/tinystories.py --target_tokens 20000000 --val_ratio 0.05
```

### 3. Training Execution

Launch training with automatic hardware detection (`mps`, `cuda`, or `cpu`), periodic validation loss tracking, live story sampling, and checkpointing:

```bash
python brain/train_gpt2.py
```

### 4. Interactive Analysis

Use the Jupyter environment for layer-by-layer inspection, tensor output checks, and generation sampling:

```bash
jupyter notebook brain/play.ipynb
```

---

## Technical Documentation (`material/`)

Reference documentation covering the mathematics and low-level engineering principles:

| Document | Area | Scope |
| :--- | :--- | :--- |
| [DDP Guide](material/distributed_data_parallel_ddp_guide.md) | Distributed Systems | Ring All-Reduce, gradient bucketing, and multi-device synchronization |
| [FlashAttention Guide](material/flash_attention_guide.md) | GPU Microarchitecture | SRAM tiling, online softmax, and IO complexity |
| [AMP Guide](material/automatic_mixed_precision_amp_guide.md) | Numerical Precision | Dynamic ranges, IEEE FP32 vs TF32 vs BF16 |
| [Torch Compile Guide](material/torch_compile_guide.md) | Compiler & JIT | TorchDynamo, TorchInductor, and fused kernel generation |
| [AdamW Guide](material/adamw_optimizer_guide.md) | Optimization Theory | Decoupled weight decay vs L2 regularization dynamics |
| [Hyperparameters Guide](material/gpt3_training_hyperparameters_guide.md) | Training Dynamics | Learning rate schedules, warmup steps, and token throughput |
| [Pretraining Datasets](material/datasets_webtext_gpt2_vs_gpt3_guide.md) | Data Engineering | Dataset pipelines (WebText, FineWeb-Edu, SlimPajama) |

---

## Development Roadmap

- [x] Pure PyTorch Axiom-LM (124M) core architecture
- [x] Hugging Face parameter matching and verification
- [x] Apple Silicon (MPS) native execution & BF16 autocast support
- [x] SDPA / Fused Attention integration
- [x] Gradient accumulation and cosine warmup learning rate schedule
- [x] Unified Memory sharded binary dataset pipeline (`uint16` memory-mapping)
- [x] Holdout validation loss evaluation loop
- [x] Periodic live generation sampling inside training loop
- [x] Checkpoint persistence and resume mechanics (`torch.save`)
- [ ] Key-Value (KV) cache inference engine
- [ ] Modern architecture enhancements (RoPE, RMSNorm, SwiGLU, GQA)
- [ ] Muon matrix optimizer integration
- [ ] MFU & PyTorch profiler roofline analysis

---

## License

This project is released under the MIT License.
