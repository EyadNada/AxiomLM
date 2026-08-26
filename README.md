# Axiom-LM (124M)

What's AxiomLM? Why AxiomLM? its a high-performance pretraining engine and modern architectural redesign of OpenAI's original GPT2 124M parameter autoregressive Transformer, optimized for Apple Silicon (Metal Performance Shaders / MPS) and NVIDIA CUDA.

---

## Technical Specifications

| Parameter | Symbol | Value | Notes |
| :--- | :---: | :---: | :--- |
| **Layers** | $L$ | 12 | Stacked decoder transformer blocks |
| **Hidden Dimension** | $d_{\text{model}}$ | 768 | Model embedding dimension |
| **Attention Heads** | $N_h$ | 12 | Query heads ($d_k = 64$) |
| **KV Heads (GQA)** | $N_{kv}$ | 4 | Grouped-Query Attention (Modern spec) |
| **Context Length** | $T$ | 1024 | Maximum sequence length |
| **Vocabulary Size** | $V$ | 50,257 | GPT-2 BPE token vocabulary |
| **Total Parameters** | $P$ | 124,439,808 | Tied input/output embeddings |

---

## Performance Benchmarks

### 1. Architectural & Systems Paradigm Comparison

| Dimension / Metric |  2019 Baseline (GPT-2 / Eager FP32) |  2026 Axiom-LM (Modern Spec) | Impact / Multiplier |
| :--- | :--- | :--- | :--- |
| **Precision Execution** | Eager FP32 (Full 32-bit floats) | BF16 Mixed-Precision Autocast | **$2.0\times$ ALU Speed & Memory Efficiency** |
| **Attention Kernel** | Naive $O(T^2)$ Materialized Softmax | Fused FlashAttention / SDPA Tiling | **Zero SRAM $\rightarrow$ HBM roundtrips** |
| **Training Throughput** | ~2,800 tokens/sec | **~9,200 tokens/sec** | **$3.29\times$ Faster Pretraining** |
| **Step Latency (4K tok)** | 1,462 ms / step | **445 ms / step** | **69.6% Reduction in Step Time** |
| **Hardware Compute (MPS)** | 2.09 TFLOPs | **6.87 TFLOPs** | **$3.28\times$ Hardware Utilization** |
| **Positional Encoding** | Learned Absolute Positional (WPE) | Complex Rotary Embeddings (**RoPE**) | **Relative distance preservation & length extrapolation** |
| **Layer Normalization** | LayerNorm (Mean Centering + Variance) | **RMSNorm** (Root Mean Square only) | **7-10% faster kernel speed, zero mean overhead** |
| **Feed-Forward Network** | Standard GELU ($4d_{\text{model}}$) | **SwiGLU** Gated Linear Unit ($\frac{8}{3}d$) | **Better empirical loss convergence per FLOP** |
| **Attention Head Layout** | Multi-Head Attention (12 KV Heads) | Grouped-Query Attention (**GQA**, 4 KV) | **66.7% KV-Cache VRAM footprint reduction** |
| **Matrix Optimizer** | Coordinate-wise Scalar AdamW | **Muon** (5-step Newton-Schulz Polar Update) | **Orthogonal gradient updates, ~42% faster descent** |
| **Inference Generation** | Naive Quadratic $O(T^2)$ Recomputation | Per-Layer $O(1)$ **KV-Cache** Engine | **Up to $27.5\times$ Speedup at $T=1024$** |
| **Per-Token Decode Latency**| Degrades up to 55+ ms / token | Steady **~6.0 ms / token** | **Constant $O(1)$ flat latency profile** |

![Systems & Architecture Efficiency Multiplier](assets/10_baseline_vs_modern_comparison.png)

Axiom-LM achieves up to a 27.5x inference speedup, 3.3x higher pretraining throughput, and a 66.7% reduction in KV-cache memory over the 2019 FP32 baseline.

---

### 2. Empirical Loss Convergence (AdamW vs. Muon Optimizer)

![Training Loss Convergence](assets/9_baseline_vs_modern_convergence.png)

The Muon matrix optimizer reaches target validation perplexity in ~42% fewer optimization steps compared to coordinate-wise AdamW.

---

### 3. Training Speed Progression

![Training Throughput](assets/1_training_throughput.png)

Training throughput increased from 2,800 tokens/sec in the unoptimized baseline to 9,200 tokens/sec across five key systems optimizations.

![Step Latency](assets/2_step_latency.png)

Step latency for 4,096 tokens dropped from 1,462 ms down to 445 ms per optimization step.

---

### 4. Compute Throughput & Hardware Efficiency

![Compute TFLOPs](assets/3_compute_tflops.png)

Effective compute throughput scaled from 2.09 TFLOPs to 6.87 TFLOPs through fused Metal attention and BF16 execution.

---

### 5. Model Architecture & Training Memory Breakdown

![Parameter Distribution](assets/4_parameter_distribution.png)

Parameter allocation is concentrated in the feed-forward MLP (45.5%) and token embeddings (31.0%).

![Training Memory](assets/5_training_memory.png)

Training VRAM footprint is kept under 2.3 GB by utilizing BF16 mixed-precision activations and fused attention buffers.

---

### 6. Inference Acceleration (KV-Cache vs. Naive Decoding)

![Inference Throughput](assets/6_inference_throughput.png)

Key-Value caching eliminates quadratic token recomputation, maintaining steady ~165 tokens/sec generation speed compared to the degrading baseline.

![Token Latency](assets/7_token_latency.png)

Per-token generation latency remains constant at ~6.0 ms per token rather than growing up to 55+ ms per token.

---

### 7. Grouped-Query Attention (GQA) Memory Savings

![KV Cache Memory](assets/8_kv_cache_memory.png)

Switching from Multi-Head Attention (12 KV heads) to Grouped-Query Attention (4 KV heads) reduces inference memory consumption by 66.7%.

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/EyadNada/GPT-2.0-124M.git
cd GPT-2.0-124M
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Data Preparation

Download and shard the TinyStories dataset into binary `uint16` arrays:

```bash
python data/tinystories.py --target_tokens 20000000 --val_ratio 0.05
```

### 3. Pretraining

Train the model with auto-detected hardware backend (`mps`, `cuda`, or `cpu`) and choice of optimizer (`adamw` or `muon`):

```bash
# Classic GPT-2 architecture with AdamW baseline
python brain/train_gpt2.py --arch classic --optimizer adamw --max_steps 4800

# Modern LLaMA-3 architecture with AdamW
python brain/train_gpt2.py --arch modern --optimizer adamw --max_steps 4800

# Modern LLaMA-3 architecture with Next-Gen Muon Matrix Optimizer
python brain/train_gpt2.py --arch modern --optimizer muon --muon_lr 0.02 --max_steps 4800
```

### 4. Interactive Analysis

Open the benchmark notebook to run all metrics and visual inspections:

```bash
jupyter notebook brain/performance_metrics.ipynb
```

---

## Repository Structure

```
├── assets/                 # Benchmark charts and evaluation graphs
├── brain/
│   ├── train_gpt2.py       # Core model, data loader, Muon/AdamW optimizers, and training loop
│   └── performance_metrics.ipynb # Interactive metrics notebook with all 8 graphs
├── data/
│   ├── tinystories.py      # Streaming tokenizer and binary sharder
│   ├── train.bin           # 19M token uint16 training binary shard
│   └── val.bin             # 1M token uint16 validation binary shard
├── checkpoints/            # Model weight snapshots (.pt)
├── material/               # Mathematical formulations and systems guides
├── requirements.txt        # Minimal environment dependencies
└── README.md
```

---

## License

MIT License.
