# OpenAI GPT-2 Original Repository Breakdown & TensorFlow vs. PyTorch Guide

> [!IMPORTANT]
> **Key Architecture Note**:
> The original [OpenAI GPT-2 repository (`openai/gpt-2`)](https://github.com/openai/gpt-2) released in 2019 was written **EXPLICITLY in TensorFlow 1.x** (specifically using static computation graphs, `tf.variable_scope`, `tf.Session`, and custom 1D convolutions).
>
> In our project ([`train_gpt2.py`](../brain/train_gpt2.py)), we are recreating and training GPT-2 (124M) in **modern PyTorch 2.x** with eager execution, modern vectorized operations, dynamic GPU/MPS/CPU dispatch, and `tiktoken`.

This document breaks down every single file in the original OpenAI repository, explains the inline comments and design choices OpenAI made, and maps each TensorFlow concept directly to its modern PyTorch equivalent so you understand the whole system without needing to read the legacy TensorFlow codebase from scratch.

---

## 1. Repository Structure & File Index

The original OpenAI GPT-2 repository is organized as follows:

```text
openai/gpt-2/
├── src/
│   ├── model.py           # Core Transformer architecture (Attention, MLP, LayerNorm, Embeddings)
│   ├── encoder.py         # Byte-Pair Encoding (BPE) tokenizer & unicode byte-mapping
│   ├── sample.py          # Autoregressive generation loop, top-k filtering, and sampling
│   └── load_dataset.py    # Text dataset loader and fixed-length context chunker
├── download_model.py      # Checkpoint downloader from Google Cloud Storage
├── interactive_conditional_samples.py  # CLI script for prompt-based interactive text generation
├── generate_unconditional_samples.py   # CLI script for random unconditional text generation
├── train.py               # Fine-tuning & optimization script in TF 1.x
├── requirements.txt       # Dependencies (tensorflow==1.15.2, regex, requests, tqdm)
└── README.md              # OpenAI release notes and model cards
```

---

## 2. Exhaustive File-by-File Deep Dive

### 2.1 `src/model.py` — Core Transformer Architecture
*This is the heart of GPT-2. It defines the mathematical forward pass of the model in TensorFlow 1.x.*

#### Key Functions & Implementation Details:

1. **`gelu(x)` (Gaussian Error Linear Unit):**
   ```python
   # Original TensorFlow 1.x
   def gelu(x):
       return 0.5 * x * (1.0 + tf.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * tf.pow(x, 3))))
   ```
   - **Why it was written:** At the time (2019), neither TensorFlow nor PyTorch had native fast GELU activations built-in. OpenAI used the tanh approximation formula $0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$.
   - **PyTorch Modern Equivalent:** `nn.GELU(approximate="tanh")` or `F.gelu(x, approximate="tanh")`.

2. **`norm(x, scope, *, axis=-1, epsilon=1e-5)` (Layer Normalization):**
   - Applies LayerNorm with learnable parameters $g$ (gain/scale, initialized to 1) and $b$ (bias, initialized to 0).
   - **PyTorch Modern Equivalent:** `nn.LayerNorm(config.n_embd, eps=1e-5)`.

3. **`conv1d(x, scope, nf, *, w_init=...)` (OpenAI 1D Linear Projection):**
   ```python
   # In OpenAI TF model.py:
   w = tf.get_variable('w', [1, nx, nf], initializer=w_init)
   b = tf.get_variable('b', [nf], initializer=b_init)
   c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf])) + b, tf.concat([start, [nf]], 0))
   ```
   - **Why OpenAI called it `conv1d`:** OpenAI called this 1D Convolution because it applies a 1D linear transformation across token sequence slices. However, mathematically it is simply a Linear projection $y = xW + b$.
   - **CRITICAL WEIGHT TRANSPOSITION QUIRK:** In OpenAI's `conv1d`, the weight matrix $W$ is stored with shape `(nx, nf)` ($D_{\text{in}}, D_{\text{out}}$). In PyTorch's `nn.Linear`, the weight tensor is stored with shape `(out_features, in_features)` ($D_{\text{out}}, D_{\text{in}}$).
   - **Impact on PyTorch weight importing:** This is why when loading pretrained HuggingFace/OpenAI checkpoints (`wte`, `c_attn`, `c_proj`, `c_fc`), we must call `.t()` (transpose) on the 2D weights:
     ```python
     # in train_gpt2.py from_pretrained:
     if any(k.endswith(w) for w in transposed):
         sd_hf[k] = sd_hf[k].t()
     ```

4. **`attn(x, scope, n_state, *, past, hparams)` (Causal Self-Attention):**
   - Projects input $x$ into combined $[Q, K, V]$ simultaneously via a single `c_attn` linear projection of shape `(n_embd, 3 * n_embd)`.
   - Computes causal masked self-attention:
     $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$
     where $M$ is the lower-triangular causal mask with $-\infty$ in upper positions to prevent tokens from looking into the future.
   - Outputs through `c_proj` linear layer.
   - **PyTorch Equivalent:** [`CausalSelfAttention`](../brain/train_gpt2.py#L18-L51) in our `train_gpt2.py`.

5. **`mlp(x, scope, n_state, *, hparams)` (Feed-Forward Network):**
   - Expands the embedding dimension $4\times$ from $768 \to 3072$ via `c_fc`, applies `gelu`, then projects back from $3072 \to 768$ via `c_proj`.
   - **PyTorch Equivalent:** [`MLP`](../brain/train_gpt2.py#L55-L69) in our `train_gpt2.py`.

6. **`block(x, scope, *, past, hparams)` (Transformer Block):**
   - Implements the Pre-LayerNorm Transformer residual connection:
     $$x = x + \text{Attention}(\text{LayerNorm}_1(x))$$
     $$x = x + \text{MLP}(\text{LayerNorm}_2(x))$$
   - **PyTorch Equivalent:** [`Block`](../brain/train_gpt2.py#L73-L88) in our `train_gpt2.py`.

7. **`model(hparams, X, past=None, scope='model')` (Full GPT-2 Container):**
   - Looks up Token Embeddings: `wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd])`.
   - Looks up Learned Position Embeddings: `wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd])`.
   - Stacks $N$ transformer blocks (`n_layer = 12` for 124M).
   - Applies final `ln_f` LayerNorm.
   - Computes logits by multiplying with `wte` (**Weight Sharing / Weight Tying**).

---

### 2.2 `src/encoder.py` — BPE Tokenizer
*Implements the Byte-Level Byte-Pair Encoding (BPE) tokenizer used in GPT-2.*

#### Key Implementation Details:
- **Byte to Unicode Mapping (`bytes_to_unicode()`):**
  - GPT-2 operates directly on raw bytes (0 to 255) rather than characters.
  - To avoid control characters and whitespace characters breaking token visualization/regex, OpenAI created a bijective mapping between every byte value $0 \le b < 256$ and a printable Unicode character.
- **Regex Splitting Pattern (`bpe_pattern`):**
  ```python
  r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
  ```
  - Prevents contractions (e.g., `'s`, `'ll`) and punctuation from merging across word boundaries.
- **Merge Rules (`encoder.json` and `vocab.bpe`):**
  - Builds the $50,257$ token vocabulary by iteratively merging the most frequent byte pairs.
- **Modern PyTorch Replacement:** We use OpenAI's official high-performance Rust-backed library **`tiktoken.get_encoding('gpt2')`**, which executes the exact same algorithm $100\times$ faster.

---

### 2.3 `src/sample.py` — Autoregressive Generation & Sampling
*Defines how tokens are generated one-by-one from the model's logits.*

#### Key Functions:
1. **`top_k_logits(logits, k)`:**
   - Filters the top-$k$ highest-probability tokens and sets all other token logits to $-\infty$ so they receive $0$ probability after softmax.
2. **`sample_sequence(...)`:**
   - Runs a loop `step in range(length)`:
     1. Feeds context tokens into the model.
     2. Takes the logits of the very last token position.
     3. Scales by temperature: $\text{logits} = \text{logits} / T$.
     4. Applies top-$k$ filtering.
     5. Draws a sample from categorical distribution: $x \sim \text{softmax}(\text{logits})$.
     6. Appends $x$ to the sequence and repeats.
- **PyTorch Modern Equivalent:** Our generation loop at the bottom of [`train_gpt2.py`](../brain/train_gpt2.py#L259-L295) using `torch.topk` and `torch.multinomial`.

---

### 2.4 `src/load_dataset.py` — Dataset Processing Pipeline
*Loads raw plain text files and tokenizes them into continuous integer arrays.*

- Tokenizes raw UTF-8 text into a single flat array of 1D token IDs.
- Chunks the flat array into non-overlapping context windows of size `block_size` ($1024$ tokens).
- **PyTorch Modern Equivalent:** Our clean `DataLoaderLite` class in [`train_gpt2.py`](../brain/train_gpt2.py#L210-L235) that loads Shakespeare/text into a 1D tensor and slices batches $(B, T)$.

---

### 2.5 `download_model.py` — Checkpoint Downloader
*Downloads official OpenAI pretrained weights from Google Cloud storage (`https://openaipublic.blob.core.windows.net/gpt-2/models/`).*

- Supports model sizes:
  - `124M` (Small) — 12 layers, 12 heads, 768 embed dim.
  - `355M` (Medium) — 24 layers, 16 heads, 1024 embed dim.
  - `774M` (Large) — 36 layers, 20 heads, 1280 embed dim.
  - `1558M` (XL) — 48 layers, 25 heads, 1600 embed dim.
- Files downloaded for each model:
  - `checkpoint` / `model.ckpt.data-00000-of-00001` / `model.ckpt.index` / `model.ckpt.meta` (TensorFlow 1.x graph & weights)
  - `hparams.json` (Hyperparameters: `n_vocab`, `n_ctx`, `n_embd`, `n_head`, `n_layer`)
  - `encoder.json` + `vocab.bpe` (Tokenizer vocabulary)
- **Modern HuggingFace Equivalent:** We load these exact same weights directly using `transformers.GPT2LMHeadModel.from_pretrained('gpt2')`.

---

## 3. Side-by-Side Comparison: TensorFlow 1.x vs. Modern PyTorch

| Feature | Original OpenAI Repo (TensorFlow 1.x) | Our Project ([`train_gpt2.py`](../brain/train_gpt2.py) in PyTorch 2.x) |
|:---|:---|:---|
| **Execution Paradigm** | Static computation graph (`tf.Graph()`, `tf.Session()`, `sess.run()`) | Eager execution (`model(x)`, automatic gradients `loss.backward()`) |
| **Variable Scoping** | `with tf.variable_scope('transformer'):` | Object-oriented `nn.ModuleDict`, `nn.ModuleList`, `nn.Module` |
| **Linear Transformations** | Custom `conv1d(x, scope, nf)` with weight shape `(D_in, D_out)` | Standard `nn.Linear(D_in, D_out)` with weight shape `(D_out, D_in)` |
| **Activation Function** | Custom manual formula approximation of `gelu` | `nn.GELU(approximate="tanh")` / `F.gelu` |
| **Tokenizer** | Python-based `src/encoder.py` parsing `vocab.bpe` | Rust-accelerated `tiktoken.get_encoding('gpt2')` |
| **Weight Sharing** | `tf.matmul(h, wte, transpose_b=True)` | `cast(nn.Embedding, self.transformer['wte']).weight = self.lm_head.weight` |
| **Device Dispatch** | Explicit CUDA placements / TPU graphs | Universal `device = "cuda" if ... else ("mps" if ... else "cpu")` |
| **Optimizer** | `tf.train.AdamOptimizer` | `torch.optim.AdamW(model.parameters(), lr=3e-4)` |

---

## 4. Key Architectural Insights & Inline Comments Explained

### 1. Why Weight Tying (Weight Sharing)?
In `model.py` and `train_gpt2.py`:
$$\text{Parameters}(\text{Embedding}) = 50257 \times 768 \approx 38.6\text{M}$$
By tying `lm_head.weight = wte.weight`, the input embedding table and output projection matrix point to the exact same tensor in memory. This saves **38.6 million parameters** (almost 30% of the entire 124M model!).

### 2. Why is Attention Bias a Buffer and Not a Parameter?
The causal attention mask (lower triangular matrix of $1$s and $0$s) contains no learnable weights.
- In PyTorch, we register it with `self.register_buffer("bias", ...)` so that it moves with `.to(device)` but is **not** updated by the optimizer or included in `model.parameters()`.
- When loading checkpoints, we ignore any key ending in `.attn.bias`.

### 3. Pre-LayerNorm vs. Post-LayerNorm
Original 2017 Transformer (*"Attention Is All You Need"*) used **Post-LayerNorm** ($x = \text{LayerNorm}(x + \text{SubLayer}(x))$), which suffered from unstable gradient flow in deep networks.
GPT-2 was among the first to switch to **Pre-LayerNorm** ($x = x + \text{SubLayer}(\text{LayerNorm}(x))$), with an extra final LayerNorm (`ln_f`) before the classifier head. This ensures clean residual paths directly through the backbone.
