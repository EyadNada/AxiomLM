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

### 1. Training Speed Progression

Training throughput increased from 2,800 tokens/sec in the unoptimized baseline to 9,200 tokens/sec across five key systems optimizations.

![Training Throughput](assets/1_training_throughput.png)

Step latency for 4,096 tokens dropped from 1,462 ms down to 445 ms per optimization step.

![Step Latency](assets/2_step_latency.png)

---

### 2. Compute Throughput & Hardware Efficiency

Effective compute throughput scaled from 2.09 TFLOPs to 6.87 TFLOPs through fused Metal attention and BF16 execution.

![Compute TFLOPs](assets/3_compute_tflops.png)

---

### 3. Model Architecture & Training Memory Breakdown

Parameter allocation is concentrated in the feed-forward MLP (45.5%) and token embeddings (31.0%).

![Parameter Distribution](assets/4_parameter_distribution.png)

Training VRAM footprint is kept under 2.3 GB by utilizing BF16 mixed-precision activations and fused attention buffers.

![Training Memory](assets/5_training_memory.png)

---

### 4. Inference Acceleration (KV-Cache vs. Naive Decoding)

Key-Value caching eliminates quadratic token recomputation, maintaining steady ~165 tokens/sec generation speed compared to the degrading baseline.

![Inference Throughput](assets/6_inference_throughput.png)

Per-token generation latency remains constant at ~6.0 ms per token rather than growing up to 55+ ms per token.

![Token Latency](assets/7_token_latency.png)

---

### 5. Grouped-Query Attention (GQA) Memory Savings

Switching from Multi-Head Attention (12 KV heads) to Grouped-Query Attention (4 KV heads) reduces inference memory consumption by 66.7%.

![KV Cache Memory](assets/8_kv_cache_memory.png)

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

### 4. Interactive Analysis

Open the benchmark notebook to run all metrics and visual inspections:

```bash
jupyter notebook brain/play.ipynb
```

---

## Repository Structure

```
├── assets/                 # Benchmark charts and evaluation graphs
├── brain/
│   ├── train_gpt2.py       # Core model, data loader, validation, and training loop
│   └── play.ipynb          # Interactive metrics notebook with all 8 graphs
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
