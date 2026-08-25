# Axiom-LM (124M)

A high-performance pretraining engine and modern architectural redesign of the 124M parameter autoregressive Transformer, optimized for Apple Silicon (Metal Performance Shaders / MPS) and NVIDIA CUDA.

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

### 1. Training Throughput Progression

| Optimization Stage | Tokens / Sec | Step Time (4k tok) | Speedup vs Baseline | Systems Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline** (Standard PyTorch FP32) | 2,800 | 1462 ms | 1.0x | Unoptimized loop with standard attention |
| **+ BF16 Mixed Precision** | 5,100 | 803 ms | 1.8x | 50% reduction in memory bandwidth |
| **+ Metal SDPA Attention** | 7,450 | 550 ms | 2.7x | Removes global attention score tensor allocation |
| **+ Zero-Sync Loss Accumulation** | 9,200 | 445 ms | 3.3x | Removes CPU-GPU synchronization stalls |
| **+ Memory-Mapped Dataset** | 9,200 | 445 ms | 3.3x | Zero-overhead direct disk-to-memory streaming |

### 2. Effective Compute & Memory Footprint

| Metric | Baseline (FP32) | Optimized (Axiom-LM) | Notes |
| :--- | :---: | :---: | :--- |
| **Compute Throughput** | 2.09 TFLOPs | **6.87 TFLOPs** | Measured on Apple Silicon ($6 \times P \times \text{tok/s}$) |
| **Model Weights Memory** | 497.7 MB | 248.8 MB | FP32 vs BF16 storage |
| **Training VRAM (Batch=4, T=1024)** | ~3.8 GB | **~1.8 GB** | With SDPA and mixed precision |
| **Data Ingestion Latency** | ~12.5 ms | **0.00 ms** | `uint16` memory-mapped binary shards |

### 3. Inference Generation Latency & Throughput

| Generated Sequence Length | Baseline (Naive $O(T^2)$) | Optimized (KV-Cache $O(1)$) | Speedup |
| :---: | :---: | :---: | :---: |
| **25 tokens** | 32.4 tok/s (30.8 ms/tok) | **168.5 tok/s (5.9 ms/tok)** | **5.2x faster** |
| **50 tokens** | 24.1 tok/s (41.5 ms/tok) | **166.8 tok/s (6.0 ms/tok)** | **6.9x faster** |
| **100 tokens** | 18.2 tok/s (54.9 ms/tok) | **165.1 tok/s (6.1 ms/tok)** | **9.1x faster** |

### 4. Classic GPT-2 vs. Modern Architecture (LLaMA-3 Spec)

| Component | Classic GPT-2 | Modern Architecture | Advantage |
| :--- | :--- | :--- | :--- |
| **Normalization** | LayerNorm | **RMSNorm** | 15-20% faster; eliminates mean-centering arithmetic |
| **Position Encoding** | Learned Absolute Table | **RoPE (Rotary)** | Relative token awareness; zero parameter overhead |
| **Feed-Forward** | GELU ($4 d$) | **SwiGLU** ($\frac{8}{3} d$) | Higher representational capacity per parameter |
| **Attention** | MHA (12 KV heads) | **GQA (4 KV heads)** | **66.7% reduction** in KV cache memory during inference |

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

Train the model with auto-detected hardware backend (`mps`, `cuda`, or `cpu`):

```bash
# Classic GPT-2 architecture
python brain/train_gpt2.py --arch classic --max_steps 4800

# Modern LLaMA-3 architecture (RoPE + RMSNorm + SwiGLU + GQA)
python brain/train_gpt2.py --arch modern --max_steps 4800
```

### 4. Interactive Metrics & Visualizations

Open the metrics notebook to view live benchmark charts, memory breakdowns, and sample outputs:

```bash
jupyter notebook brain/play.ipynb
```

---

## Repository Structure

```
├── brain/
│   ├── train_gpt2.py       # Core model, data loader, validation, and training loop
│   └── play.ipynb          # Benchmark charts, memory analysis, and generation tests
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
