<div align="center">

#  GPT-2 (124M) FROM SCRATCH 
### *From Raw Matrix Math to Shakespearean Hallucinations*

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Hardware](https://img.shields.io/badge/Hardware-CUDA%20%7C%20MPS%20%7C%20CPU-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![Parameters](https://img.shields.io/badge/Parameters-124M-8A2BE2?style=for-the-badge)](https://huggingface.co/openai-community/gpt2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<p align="center">
  <b>A battle-tested, no-black-box, pure PyTorch reproduction of OpenAI's GPT-2 (124M) architecture.</b><br>
  Engineered from first principles with full Tensor Core hardware optimizations, custom BPE dataloaders, and deep-dive technical guides.
</p>

```
    ┌─────────────────────────────────────────────────────────────┐
    │  "What I cannot create, I do not understand."               │
    │                                          — Richard Feynman  │
    └─────────────────────────────────────────────────────────────┘
```

---

</div>

## 🧠 Model Architecture Specification (GPT-2 Small)

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
                    │   │ Causal Self-Attn    │   │  <─── FlashAttention / RoPE / KV
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

### 🔬 Hyperparameter Cheat Sheet

| Parameter | Symbol | Value | Hardware / Math Rationale |
|:---|:---:|:---:|:---|
| **Layers** | $L$ | `12` | Stacked decoder-only Transformer blocks |
| **Embedding Dimension** | $d_{\text{model}}$ | `768` | Hidden vector channel capacity |
| **Attention Heads** | $N_h$ | `12` | Multi-Head Self-Attention subspaces |
| **Head Dimension** | $d_k$ | `64` | $768 / 12 = 64$ (Perfect multiple of 8 for Tensor Cores!) |
| **Context Window** | $T_{\text{max}}$ | `1024` | Maximum sequence length per forward pass |
| **Vocab Size** | $V$ | `50,257` | Byte-Pair Encoding (BPE) vocabulary |
| **Vocab (Padded)** | $V_{\text{opt}}$ | `50,304` | Nearest multiple of 64 for **$\sim 10\%$ speedup on Tensor Cores** |
| **Total Parameters** | $P$ | `124,439,808` | Exactly matches original OpenAI release |

---

## 🚀 The Speed & Optimization Arsenal

We don't do slow. This codebase incorporates modern LLM training speedups:

1. **⚡ TensorFloat-32 & BF16 Mixed Precision**: 
   - Uses Ampere/Ada/Hopper Tensor Cores for single-cycle $D = A \times B + C$ hardware matrix arithmetic.
2. **🪢 Weight Tying Scheme**:
   - `lm_head.weight` is tied directly to `wte.weight`, saving **$38.6\text{M}$ redundant parameters** ($30\%$ of total weights).
3. **🎯 Scaled Residual Initialization**:
   - Standard initialization $\sigma = 0.02$, scaled by $\frac{1}{\sqrt{2 \times N_{\text{layer}}}}$ for projection layers to prevent activation explosion at depth.
4. **🏎️ Custom High-Throughput DataLoader (`DataLoaderLite`)**:
   - Zero-overhead stateful streaming iterator directly in token space via `tiktoken` (GPT-2 regex pattern).
5. **🧮 Decoupled Weight Decay (AdamW)**:
   - Separate treatment of 2D matrix weights (decayed) vs. 1D biases/LayerNorms (non-decayed).

---

## 🛠️ Quickstart (Zero to Generation in 60s)

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone https://github.com/EyadNada/GPT-2.0-124M.git
cd GPT-2.0-124M

# Initialize Virtual Environment
python3 -m venv venv
source venv/bin/activate

# Install requirements (PyTorch, tiktoken, etc.)
pip install -r requirements.txt
```

### 2. Run Pre-Training & Verification
```bash
# Run training directly from terminal
python brain/train_gpt2.py
```

### 3. Interactive Playground & Experimentation
```bash
# Launch interactive Jupyter notebook
jupyter notebook brain/play.ipynb
```

---

## 📚 The Research & Theory Codex (`material/`)

Every technique implemented in this codebase is backed by deep theoretical documentation in the [`material/`](material/) vault:

| Document / Paper | Topic | Description |
|:---|:---:|:---|
| 📑 [Automatic Mixed Precision (AMP)](material/automatic_mixed_precision_amp_guide.md) | **Hardware & Speed** | PyTorch AMP recipe: `torch.autocast`, `GradScaler`, gradient clipping, and checkpointing. |
| 📑 [FlashAttention & Fast SDPA](material/flash_attention_guide.md) | **Hardware & Speed** | Fused scaled dot-product attention, IO-awareness, online softmax, and $O(T)$ memory scaling. |
| 📑 [PyTorch `torch.compile` Guide](material/torch_compile_guide.md) | **Hardware & Speed** | JIT compilation, TorchDynamo, TorchInductor, Triton kernel fusion, and modes. |
| 📑 [Float32 Matmul Precision (TF32)](material/torch_set_float32_matmul_precision_guide.md) | **Hardware & Speed** | `torch.set_float32_matmul_precision`: TF32 acceleration, precision levels, and GEMM benchmarks. |
| 📑 [Tensor Cores & Mixed Precision](material/tensor_cores_and_mixed_precision_guide.md) | **Hardware & Speed** | NVIDIA Tensor Core architecture, FP16/BF16/TF32 formats, loss scaling & alignment. |
| 📑 [AdamW Optimizer Guide](material/adamw_optimizer_guide.md) | **Optimization** | Decoupled Weight Decay math vs. L2 regularization (Loshchilov & Hutter). |
| 📑 [Tiktokenizer & Tokenization Guide](material/tiktokenizer_guide.md) | **Data Pipeline** | Byte-Pair Encoding (BPE), byte-level fallbacks, and regex tokenizers. |
| 📑 [Generation & Sampling Strategies](material/generation_and_sampling_strategies.md) | **Inference** | Greedy search, Temperature, Top-$k$, Top-$p$ (Nucleus), and Min-$p$ sampling math. |
| 📑 [Online Softmax Normalizer Guide](material/online_normalizer_calculation_for_softmax_guide.md) | **Algorithms & Math** | Mathematical breakdown of online safe softmax recurrence and the foundation of FlashAttention. |
| 📑 [OpenAI GPT-2 Repo Breakdown](material/openai_gpt2_repo_breakdown.md) | **Architecture** | Line-by-line breakdown comparing this repository to OpenAI's original release. |
| 📑 [Unreasonable Effectiveness of RNNs](material/the_unreasonable_effectiveness_of_rnns.md) | **Foundations** | Andrej Karpathy's foundational manifesto on autoregressive language generation. |
| 📄 [Online Softmax Normalizer Paper](material/online_normalizer_calculation_for_softmax_paper.pdf) | **Paper (Milakov & Gimelshein)** | *Online normalizer calculation for softmax* (NVIDIA, 2018). |
| 📄 [Attention Is All You Need](material/attention_is_all_you_need.pdf) | **Paper (Vaswani et al.)** | The legendary Transformer paper introducing scaled dot-product attention. |
| 📄 [GPT-2 Original Paper](material/gpt2_paper.pdf) | **Paper (Radford et al.)** | *Language Models are Unsupervised Multitask Learners* (OpenAI, 2019). |
| 📄 [GELU Activation Paper](material/gelu_paper.pdf) | **Paper (Hendrycks et al.)** | *Gaussian Error Linear Units (GELUs)* mathematical motivation. |

---

## 🔬 Mathematical Highlights

### 1. Scaled Dot-Product Attention
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
*Where $M_{i,j} = -\infty$ for $j > i$ (autoregressive causal mask preventing future token peeking).*

### 2. Approximate GeLU (Gaussian Error Linear Unit)
$$\text{GELU}(x) = 0.5 x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3\right)\right)\right)$$

### 3. Decoupled Weight Decay Step
$$\theta_t \leftarrow \theta_{t-1} - \gamma \lambda \theta_{t-1} - \gamma \frac{\widehat{m}_t}{\sqrt{\widehat{v}_t} + \epsilon}$$

---

## 🗺️ Community Roadmap & Future Experiments

- [x] Pure PyTorch GPT-2 (124M) architecture implementation
- [x] Hugging Face pretrained weight importer (`from_pretrained("gpt2")`)
- [x] High-speed BPE `DataLoaderLite`
- [x] Tensor Core & Mixed Precision optimizations (TF32 / BF16)
- [ ] **FlashAttention-2 / 3 Integration** (`torch.nn.functional.scaled_dot_product_attention`)
- [ ] **`torch.compile()` Full Graph JIT Compilation** (Operator fusion & inductor kernel generation)
- [ ] **DDP (Distributed Data Parallel)** Multi-GPU cluster training
- [ ] **FineWeb-Edu & OpenWebText** large-scale pretraining pipeline
- [ ] **KV-Cache Accelerated Autoregressive Generation Engine**

---

## 🤝 Contributing & Community

Contributions, bug reports, and experimental architectural tweaks are warmly welcomed! 

1. **Fork the repository**
2. **Create your feature branch**: `git checkout -b feature/blazing-fast-kernels`
3. **Commit your changes**: `git commit -m "feat: implement rotary position embeddings (RoPE)"`
4. **Push to branch**: `git push origin feature/blazing-fast-kernels`
5. **Open a Pull Request** and join the discussions!

---

<div align="center">
  <sub>Built with ☕, math, and GPUs by the open-source community. Star ⭐ this repo if you learned something cool!</sub>
</div>
