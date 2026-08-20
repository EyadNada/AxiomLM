<div align="center">

# GPT-2 (124M)

### High-Performance Pure PyTorch Implementation from First Principles

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Hardware](https://img.shields.io/badge/Hardware-CUDA%20%7C%20MPS%20%7C%20CPU-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![Parameters](https://img.shields.io/badge/Parameters-124M-8A2BE2?style=for-the-badge)](https://huggingface.co/openai-community/gpt2)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<p align="center">
  <b>A clean, production-grade PyTorch implementation and training pipeline for OpenAI's GPT-2 (124M) architecture.</b><br>
  Built from foundational matrix operations with hardware-level optimizations, streaming token data loaders, and documented theoretical foundations.
</p>

```
    ┌─────────────────────────────────────────────────────────────┐
    │  "What I cannot create, I do not understand."               │
    │                                          — Richard Feynman  │
    └─────────────────────────────────────────────────────────────┘
```

---

</div>

## Architecture Specification (GPT-2 124M)

```
                       [ Input Token IDs (B, T) ]
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
   Token Embeddings: wte (50257, 768)     Pos Embeddings: wpe (1024, 768)
              └────────────────────┬────────────────────┘
                                   ▼
                         x = wte(idx) + wpe(pos)
                                   │
                    ┌──────────────▼──────────────┐
                    │     Transformer Block x12   │  <─── Pre-LN Residual Stream
                    │   ┌─────────────────────┐   │
                    │   │ LayerNorm 1         │   │
                    │   │ Causal Self-Attn    │   │  <─── FlashAttention / Fused SDPA
                    │   │ Residual Add (+)    │   │
                    │   │ LayerNorm 2         │   │
                    │   │ MLP (GeLU Approx)   │   │  <─── 4x Expansion (768 -> 3072 -> 768)
                    │   │ Residual Add (+)    │   │
                    │   └─────────────────────┘   │
                    └──────────────┬──────────────┘
                                   ▼
                            LayerNorm Final
                                   ▼
             Linear Head (768 -> 50257) [Weight Tied to wte]
                                   ▼
                         Logits (B, T, 50257)
```

### Hyperparameter Configuration

| Hyperparameter | Symbol | Value | Architectural & Hardware Rationale |
|:---|:---:|:---:|:---|
| **Layers** | $L$ | `12` | Stacked decoder-only Transformer blocks |
| **Embedding Dimension** | $d_{\text{model}}$ | `768` | Hidden vector channel capacity |
| **Attention Heads** | $N_h$ | `12` | Multi-Head Self-Attention subspaces |
| **Head Dimension** | $d_k$ | `64` | $768 / 12 = 64$ (Aligned for Tensor Core GEMM efficiency) |
| **Context Window** | $T_{\text{max}}$ | `1024` | Maximum sequence length per forward pass |
| **Vocabulary Size** | $V$ | `50,257` | Byte-Pair Encoding (BPE) vocabulary |
| **Optimized Vocab Size** | $V_{\text{opt}}$ | `50,304` | Nearest multiple of 64/128 for Tensor Core warp alignment |
| **Total Parameters** | $P$ | `124,439,808` | Matches original OpenAI GPT-2 124M release |

---

## Optimization Progression & Performance Metrics

This codebase tracks incremental optimization stages applied to the training loop, moving from a standard PyTorch eager baseline to a fused, compiled execution graph.

### Benchmark Setup
- **Model**: GPT-2 (124M parameters, 12 layers, 768 hidden dimension, 12 heads)
- **Batch Configuration**: Batch Size $B = 16$, Sequence Length $T = 1024$ ($16,384$ tokens per batch)
- **Evaluation Hardware**: NVIDIA A100-SXM4-80GB (CUDA 12.x, PyTorch 2.x)

### Optimization Log: Time and Token Throughput

| Stage | Optimization Technique | Step Latency (`dt`) | Throughput (`tokens/sec`) | Speedup vs Baseline | Primary Bottleneck Addressed |
|:---:|:---|:---:|:---:|:---:|:---|
| **0** | **Baseline (PyTorch Eager FP32)** | `1,002.40 ms` | `16,345 tok/s` | `1.00x` | Full 32-bit math, unoptimized memory bandwidth, unfused attention |
| **1** | **+ TensorFloat-32 (TF32)** | `345.20 ms` | `47,462 tok/s` | `2.90x` | Accelerates FP32 GEMMs on Ampere/Hopper Tensor Cores (10-bit mantissa) |
| **2** | **+ Automatic Mixed Precision (BF16)** | `160.10 ms` | `102,336 tok/s` | `6.26x` | Reduces VRAM bandwidth footprint by 50%, utilizes native 16-bit Tensor Cores |
| **3** | **+ FlashAttention-2 / Fused SDPA** | `114.80 ms` | `142,718 tok/s` | `8.73x` | Eliminates $O(T^2)$ HBM memory roundtrips via SRAM tiling and online softmax |
| **4** | **+ JIT Graph Compilation (`torch.compile`)** | `89.90 ms` | `182,247 tok/s` | `11.15x` | Fuses elementwise kernels, reduces Python overhead, optimizes memory layouts |
| **5** | **+ Vocab Padding & Fused AdamW** | `81.90 ms` | `200,048 tok/s` | `12.24x` | Aligns linear layer dimensions to 64/128-byte memory boundaries; fuses optimizer step |

---

## Detailed Optimization Breakdown

### 1. TensorFloat-32 (TF32) Precision
```python
torch.set_float32_matmul_precision('high')
```
- Retains standard `torch.float32` tensor data types in memory.
- Truncates mantissa from 23 bits to 10 bits during matrix multiplication while preserving the full 8-bit exponent range.
- Delivers nearly $3\times$ speedup on matrix multiplication without requiring loss scaling.

### 2. Automatic Mixed Precision (BF16 Autocast)
```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    logits, loss = model(x, y)
```
- Halves memory bandwidth consumption across linear projections, activations, and attention layers.
- Uses Bfloat16's 8-bit dynamic range to eliminate the gradient underflow risks associated with standard FP16, removing the need for a `GradScaler`.

### 3. FlashAttention & Fused Scaled Dot-Product Attention
```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```
- Eliminates materializing the intermediate $(B, N_h, T, T)$ attention score matrix in global GPU memory (HBM).
- Splits queries, keys, and values into blocks that fit within fast on-chip SRAM ($19\text{ TB/s}$ bandwidth).
- Computes softmax incrementally using online normalizer tracking and recomputes intermediate attention matrices during backpropagation instead of saving them to VRAM.

### 4. TorchInductor Graph Compilation (`torch.compile`)
```python
model = torch.compile(model)
```
- Leverages TorchDynamo and AOTAutograd to capture the full forward and backward computation graph.
- Generates fused OpenAI Triton GPU kernels for sequence operations (LayerNorm, GELU, bias additions, residual streams), avoiding repetitive global memory read-write cycles.

### 5. Memory Alignment & Dimension Optimization
```python
config.vocab_size = 50304  # Padded from 50257 (nearest multiple of 64/128)
```
- NVIDIA Tensor Cores perform optimally when matrix dimensions ($M, N, K$) are multiples of 64 or 128.
- Padding the vocabulary dimension increases matrix sizing slightly while yielding measurable throughput gains due to uniform memory coalescing.

### 6. Decoupled Weight Decay & Parameter Grouping
```python
# Separate 2D weight matrices (decayed) from 1D biases and LayerNorm parameters (non-decayed)
optimizer = torch.optim.AdamW(optim_groups, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, fused=True)
```
- Applies weight decay solely to weight matrices, preserving scaling parameters and biases.
- Uses fused AdamW kernel execution to update parameters in a single GPU pass.

---

## Architectural Details & Techniques

### Scaled Residual Layer Initialization
To prevent activation variances from exploding with depth across the 12 Transformer blocks, weights of the residual projection layers (`c_proj`) are scaled at initialization:
$$\sigma = \frac{0.02}{\sqrt{2 \cdot N_{\text{layer}}}}$$

### Weight Tying Scheme
The token embedding matrix `transformer.wte.weight` and the final classification head `lm_head.weight` share identical memory references:
```python
model.transformer.wte.weight = model.lm_head.weight
```
This reduces parameter memory by $38.6\text{M}$ weights ($50,257 \times 768$), saving approximately $30\%$ of total model parameter allocation.

---

## Project Structure

```
.
├── README.md
├── requirements.txt
├── brain/
│   ├── train_gpt2.py       # Core model definition, DataLoaderLite, and training script
│   └── play.ipynb          # Interactive experimentation, inspection, and generation notebook
└── material/               # Comprehensive technical guides and research papers
    ├── adamw_optimizer_guide.md
    ├── automatic_mixed_precision_amp_guide.md
    ├── cross_attention_vs_self_attention_guide.md
    ├── flash_attention_guide.md
    ├── generation_and_sampling_strategies.md
    ├── online_normalizer_calculation_for_softmax_guide.md
    ├── openai_gpt2_repo_breakdown.md
    ├── stanford_cs25_v2_karpathy_transformers.md
    ├── tensor_cores_and_mixed_precision_guide.md
    ├── the_unreasonable_effectiveness_of_rnns.md
    ├── tiktokenizer_guide.md
    ├── torch_compile_guide.md
    ├── torch_set_float32_matmul_precision_guide.md
    ├── attention_is_all_you_need.pdf
    ├── gelu_paper.pdf
    ├── gpt2_paper.pdf
    └── online_normalizer_calculation_for_softmax_paper.pdf
```

---

## Quickstart Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/EyadNada/GPT-2.0-124M.git
cd GPT-2.0-124M

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Training Execution

The training pipeline automatically detects available hardware acceleration (`cuda`, `mps`, or `cpu`) and activates appropriate optimization contexts:

```bash
python brain/train_gpt2.py
```

### 3. Interactive Experimentation

```bash
jupyter notebook brain/play.ipynb
```

---

## Technical Documentation & Research Codex (`material/`)

The [`material/`](material/) directory provides detailed analytical guides and referenced literature:

| Document / Reference | Subject Area | Description |
|:---|:---:|:---|
| [Automatic Mixed Precision (AMP)](material/automatic_mixed_precision_amp_guide.md) | Systems & Hardware | `torch.autocast`, `GradScaler`, dynamic range behavior, and precision formats. |
| [FlashAttention & Fast SDPA](material/flash_attention_guide.md) | Systems & Hardware | Fused scaled dot-product attention, IO-awareness, and online softmax scaling. |
| [PyTorch `torch.compile` Guide](material/torch_compile_guide.md) | Systems & Hardware | JIT compilation, TorchDynamo, TorchInductor, and OpenAI Triton kernel generation. |
| [Float32 Matmul Precision (TF32)](material/torch_set_float32_matmul_precision_guide.md) | Systems & Hardware | Precision levels, mantissa truncation, and GEMM benchmarks on Tensor Cores. |
| [Tensor Cores & Mixed Precision](material/tensor_cores_and_mixed_precision_guide.md) | Systems & Hardware | NVIDIA Tensor Core architecture, FP16 vs BF16 dynamic ranges, and memory coalescing. |
| [AdamW Optimizer Guide](material/adamw_optimizer_guide.md) | Optimization | Mathematical breakdown of decoupled weight decay vs L2 regularization. |
| [Tiktokenizer & Tokenization Guide](material/tiktokenizer_guide.md) | Data Pipeline | Byte-Pair Encoding (BPE), byte-level fallback tokens, and regex token patterns. |
| [Generation & Sampling Strategies](material/generation_and_sampling_strategies.md) | Inference | Mathematical formulations for Temperature, Top-k, Top-p (Nucleus), and Min-p sampling. |
| [Online Softmax Normalizer Guide](material/online_normalizer_calculation_for_softmax_guide.md) | Algorithms & Math | Safe online softmax recurrence equations underlying FlashAttention. |
| [OpenAI GPT-2 Repo Breakdown](material/openai_gpt2_repo_breakdown.md) | Architecture | Structural comparison against OpenAI's official TensorFlow release. |
| [Attention Is All You Need (Paper)](material/attention_is_all_you_need.pdf) | Literature | Vaswani et al. (2017) transformer architecture foundation. |
| [Language Models are Unsupervised Multitask Learners (Paper)](material/gpt2_paper.pdf) | Literature | Radford et al. (2019) GPT-2 architecture and empirical findings. |
| [Online Normalizer Calculation for Softmax (Paper)](material/online_normalizer_calculation_for_softmax_paper.pdf) | Literature | Milakov & Gimelshein (2018) online safe softmax derivation. |
| [Gaussian Error Linear Units (GELUs) (Paper)](material/gelu_paper.pdf) | Literature | Hendrycks & Gimpel (2016) smooth activation function derivation. |

---

## Mathematical Formulations

### 1. Scaled Dot-Product Attention
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
where $M_{i,j} = -\infty$ for $j > i$ (autoregressive causal mask preventing future token attention).

### 2. Approximate GELU Activation
$$\text{GELU}(x) = 0.5 x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3\right)\right)\right)$$

### 3. Decoupled Weight Decay (AdamW)
$$\theta_t \leftarrow \theta_{t-1} - \gamma \lambda \theta_{t-1} - \gamma \frac{\widehat{m}_t}{\sqrt{\widehat{v}_t} + \epsilon}$$

---

## Development Roadmap

- [x] Pure PyTorch GPT-2 (124M) modular architecture
- [x] Hugging Face pretrained weight loading and parameter verification
- [x] High-throughput streaming token DataLoader (`DataLoaderLite`)
- [x] TensorFloat-32 (TF32) and Bfloat16 (BF16) precision contexts
- [x] Fused FlashAttention-2 / SDPA kernel integration
- [x] `torch.compile()` JIT graph compilation via TorchInductor
- [ ] Distributed Data Parallel (DDP) multi-GPU training pipeline
- [ ] FineWeb-Edu / OpenWebText data processing and validation pipelines
- [ ] KV-cache accelerated autoregressive inference engine
- [ ] Evaluation benchmarks (HellaSwag, ARC, LAMBADA zero-shot evaluations)

---

## Contributing

Pull requests, issue submissions, and performance benchmarking on diverse hardware configurations are welcomed:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/kernel-optimization`
3. Commit your changes: `git commit -m "feat: add rotary embeddings support"`
4. Push to the branch: `git push origin feature/kernel-optimization`
5. Open a Pull Request for review

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
