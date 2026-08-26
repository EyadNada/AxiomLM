# 🧭 Master LLM Pretraining, Modernization & 0.01% Research Roadmap

> **Local Private Document**: This file is ignored by git (`.gitignore`) so it stays completely private to your local machine and will never be pushed to GitHub.

---

## 📊 Current Project Scorecard & Realistic Completion: ~75%

| Domain / Pillar | Progress | What is Implemented | What Remains for "Top 0.01%" Tier |
| :--- | :---: | :--- | :--- |
| **1. Classic GPT-2 Architecture** | **95%** | Full model from scratch, Pre-LN, weight tying, Hugging Face weight loading verification. | Padded vocab to `50,304` for Tensor Core tile alignment. |
| **2. Low-Level System Optimizations** | **95%** | TF32, BF16 Autocast, SDPA / FlashAttention, zero-sync on-device loss accumulation, parameter splitting. | MPS/CUDA kernel profiling & MFU calculations. |
| **3. Pretraining Data Pipeline** | **95%** | Token sharder (`data/tinystories.py`), memory-mapped binary `train.bin` (19M) / `val.bin` (1M) with CLI args. | Multi-shard rotation for multi-billion token corpora. |
| **4. Training Loop & Validation** | **95%** | Step loop, lr schedule, grad accum, throughput timer (`tok/s`), holdout val loss loop, live story sampling, checkpointing (`.pt`). | DDP multi-node distributed scaling. |
| **5. Modern LLM Architecture (LLaMA-3 Spec)** | **95%** | Modular **RMSNorm**, **RoPE (Complex Rotary Embeddings)**, **SwiGLU**, and **GQA (Grouped-Query Attention)** with CLI switch (`--arch modern`). | Sliding Window Attention (SWA). |
| **6. Next-Gen Optimizer (Muon)** | **100%** | 5-step Newton-Schulz algorithm, dual parameter routing (2D matrix Muon + AdamW vectors/embeddings), CLI switch (`--optimizer muon`). | Multi-shard ablation logging across billions of tokens. |
| **7. Inference Engine (KV-Cache)** | **95%** | Per-layer Key-Value caching ($O(1)$ decoding), prefill/decode transitions, exact greedy parity, latency benchmark (`tokens/s`). | PagedAttention / vLLM block table memory management. |
| **8. Systems Profiling & MFU Roofline** | **10%** | Basic `dt` and `tok/s` measurement. | PyTorch Profiler (`trace.json`), MFU % calculation, memory bandwidth analysis. |
| **9. Custom Low-Level Kernel** | **5%** | Standard PyTorch ops. | Custom Triton / Metal kernel for RMSNorm or Fused Attention. |
| **10. Research Artifacts & Tutorial** | **35%** | 20+ theoretical research guides, verified architecture benchmarks, clean repo docs. | Published Technical Paper / Blog, interactive Hugging Face Space, Video Tutorial. |

---

## 🏆 The 0.01% Elite Research & Systems Tier (Lab-Admission Standard)

To transform this repository from a standard tutorial project into a **world-class AI research & systems engineering portfolio**:

1. **Empirical Scientific Ablation Study**:
   * Controlled comparison: Train two identical 124M models on 20M tokens of TinyStories:
     * Model A: Standard AdamW + LayerNorm + GELU
     * Model B: Muon Optimizer + RMSNorm + SwiGLU + RoPE
   * Publish loss curves, perplexity, and wall-clock convergence times proving Muon reaches target perplexity in **~40% fewer steps**.
2. **Model FLOPs Utilization (MFU) & Roofline Profiling**:
   * Compute hardware theoretical peak TFLOPs (Apple Silicon MPS / NVIDIA CUDA).
   * Measure actual MFU %: $\text{MFU} = \frac{6 \times P \times \text{tokens\_per\_sec}}{\text{Peak FLOPs}}$.
   * Export and analyze Chrome traces using `torch.profiler.profile()`.
3. **Custom Hardware Kernel**:
   * Implement a custom **RMSNorm** or **Tiled Attention** kernel using **OpenAI Triton** (CUDA) or **Metal Shading Language** (Apple Silicon).
4. **Interactive Demo & Public Weights**:
   * Deploy an interactive Hugging Face Space / Streamlit app with live story generation, temperature/top-p sliders, and KV-cache speed metrics.
   * Host trained `.pt` / `.safetensors` model weights publicly on Hugging Face Hub.
5. **Video Lecture / Tutorial Series**:
   * Record a first-principles deep dive explaining the math, architecture transitions (GPT-2 $\rightarrow$ LLaMA-3), and live code implementation.

---

## 📋 Comprehensive Master Checklist

- [x] **Phase 1: Lightweight Dataset Pipeline (`data/tinystories.py`)**
- [x] **Phase 2: Train/Val Sharded `DataLoaderLite` with Dynamic Splitting**
- [x] **Phase 3: Periodic Validation Loss & Holdout Evaluation Loop**
- [x] **Phase 4: Live Generation & Visual Sampling Inside Training Loop**
- [x] **Phase 5: Model Checkpointing (`torch.save` & Resuming States)**
- [x] **Phase 6: Key-Value (KV) Cache Accelerated Inference Engine**
- [x] **Phase 7: Modern Architecture Upgrades (RoPE + RMSNorm + SwiGLU + GQA)**
- [x] **Phase 8: Next-Gen Muon Matrix Optimizer Integration**
- [ ] **Phase 9: Systems Profiling, PyTorch Profiler & MFU Roofline Analysis**
- [ ] **Phase 10: Custom Low-Level Kernel (Triton / Metal MSL)**
- [ ] **Phase 11: Interactive Streamlit / Hugging Face Demo**
- [ ] **Phase 12: Technical Research Paper / Blog Post & Video Tutorial**

---

## 🛠️ Phase 1: Lightweight Dataset Pipeline (`data/tinystories.py`)

### Goal
Download a ~15M–30M token slice of **TinyStories**, tokenize with `tiktoken` (`gpt2` BPE), prefix each story with `<|endoftext|>` (`50256`), and save as binary `uint16` arrays for zero-overhead loading.

### Code Implementation Template (`data/tinystories.py`)
```python
import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

print("Loading TinyStories dataset from Hugging Face...")
dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

target_tokens = 20_000_000   # 20M tokens (~35 mins training on Mac)
val_token_budget = 1_000_000 # 1M tokens for holdout validation

all_tokens = []
print(f"Tokenizing up to {target_tokens:,} tokens...")
pbar = tqdm(total=target_tokens, unit="tokens")

for item in dataset:
    text = item["text"]
    tokens = [eot] + enc.encode_ordinary(text)
    all_tokens.extend(tokens)
    pbar.update(len(tokens))
    if len(all_tokens) >= target_tokens:
        break
pbar.close()

all_tokens_np = np.array(all_tokens[:target_tokens], dtype=np.uint16)

# Split 95% train / 5% val
val_tokens = all_tokens_np[:val_token_budget]
train_tokens = all_tokens_np[val_token_budget:]

val_path = os.path.join(DATA_DIR, "val.bin")
train_path = os.path.join(DATA_DIR, "train.bin")

val_tokens.tofile(val_path)
train_tokens.tofile(train_path)

print(f"Saved: {train_path} ({len(train_tokens):,} tokens)")
print(f"Saved: {val_path} ({len(val_tokens):,} tokens)")
```

---

## 🛠️ Phase 2: Train/Val Sharded `DataLoaderLite`

```python
class DataLoaderLite:
    def __init__(self, B, T, split="train", data_dir="data"):
        self.B = B
        self.T = T
        self.split = split
        filename = os.path.join(data_dir, f"{split}.bin")
        assert os.path.exists(filename), f"Binary dataset file {filename} not found. Run data/tinystories.py first."
        
        self.tokens = np.memmap(filename, dtype=np.uint16, mode='r')
        print(f"[{split}] Loaded {len(self.tokens):,} tokens ({len(self.tokens) // (B * T)} batches per epoch)")
        self.current_position = 0

    def reset(self):
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1].astype(np.int64)
        x = torch.from_numpy(buf[:-1].copy()).view(B, T)
        y = torch.from_numpy(buf[1:].copy()).view(B, T)
        
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y
```

---

## 🛠️ Phase 3 & 4: Validation Loop, Live Sampling & Checkpointing

```python
val_loader = DataLoaderLite(B=B, T=T, split="val")

def evaluate_val_loss(model, val_loader, eval_steps=20, device="cpu"):
    model.eval()
    val_loader.reset()
    val_loss_accum = 0.0
    with torch.no_grad():
        for _ in range(eval_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with autocast_ctx:
                logits, loss = model(x, y)
            loss = loss / eval_steps
            val_loss_accum += loss.detach().item()
    model.train()
    return val_loss_accum

# In training loop every 100 steps:
if step > 0 and step % 100 == 0:
    val_loss = evaluate_val_loss(model, val_loader, eval_steps=20, device=device)
    print(f"\n>>> [STEP {step}] Validation Loss: {val_loss:.4f} <<<")
    
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': model.config,
    }, f"checkpoints/model_step_{step:05d}.pt")

    # Sample generation
    model.eval()
    sample_tokens = enc.encode("Once upon a time,")
    x_gen = torch.tensor(sample_tokens, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        for _ in range(30):
            logits, _ = model(x_gen)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            next_tok = torch.gather(topk_indices, -1, ix)
            x_gen = torch.cat((x_gen, next_tok), dim=1)
    print("Sample: >", enc.decode(x_gen[0].tolist()), "\n")
    model.train()
```

---

## 🛠️ Phase 5 & 6: KV-Cache Accelerated Inference Engine

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, kv_cache=None):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if kv_cache is not None:
            k_past, v_past = kv_cache
            if k_past is not None:
                k = torch.cat([k_past, k], dim=2)
                v = torch.cat([v_past, v], dim=2)
            new_kv_cache = (k, v)
        else:
            new_kv_cache = None

        if kv_cache is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y, new_kv_cache
```

---

## 🛠️ Phase 7: Modern Architecture Upgrades (RoPE + RMSNorm + SwiGLU + GQA)

```python
# 1. RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

# 2. RoPE
def precompute_rope_frequencies(dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rope(x, freqs_cis):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:x.shape[2], :].unsqueeze(0).unsqueeze(0)
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return x_rotated.type_as(x)

# 3. SwiGLU MLP
class SwiGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(2 * (4 * config.n_embd) / 3)
        self.w_gate = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w_up = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.w_down.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
```

---

## 🛠️ Phase 8: Next-Gen Muon Matrix Optimizer Integration

$$\mathbf{X}_{k+1} = \frac{1}{2} \mathbf{X}_k (3\mathbf{I} - \mathbf{X}_k^T \mathbf{X}_k)$$

```python
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16() if G.dtype == torch.bfloat16 else G.float()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.type_as(G)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                u = zeropower_via_newtonschulz5(g, steps=ns_steps)
                p.data.add_(u, alpha=-lr)
```

---

## 🛠️ Phase 9: Systems Profiling & MFU Roofline Analysis

```python
# Measure MFU (Model FLOPs Utilization)
def calculate_mfu(model, batch_size, seq_len, dt, peak_tflops=10.0):
    """
    Computes MFU percentage: 6 * N_params * tokens / dt / peak_flops
    """
    N = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tokens_per_iter = batch_size * seq_len
    flops_per_token = 6 * N
    flops_achieved = (flops_per_token * tokens_per_iter) / dt
    mfu = flops_achieved / (peak_tflops * 1e12)
    return mfu * 100.0

# Chrome Trace Export
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler_trace'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step in range(5):
        # run step...
        prof.step()
```

---

## 🛠️ Phase 10: Custom Low-Level Kernel (Triton / Metal MSL)

```python
# Custom RMSNorm in Triton (for CUDA GPUs)
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_kernel(
    X_ptr, Y_ptr, W_ptr,
    stride_x_row, stride_y_row,
    N, eps, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X_ptr + row_idx * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / N
    rsqrt = 1.0 / tl.sqrt(variance + eps)
    y = x * rsqrt * w
    tl.store(Y_ptr + row_idx * stride_y_row + cols, y, mask=mask)
```

---

## 🛠️ Phase 11 & 12: Research Paper / Blog & Video Tutorial Blueprint

### Structure of Your Technical Report (`REPORT.md` / Blog Post)
1. **Abstract**: Problem statement, efficiency goals on constrained hardware (Apple Silicon / Single GPU).
2. **Architecture Evolution**: Mathematical derivations of RoPE, RMSNorm, SwiGLU, and GQA vs. GPT-2 2019 baseline.
3. **The Muon Matrix Optimizer**: Newton-Schulz convergence theory vs. AdamW coordinate-wise updates.
4. **Experimental Setup**: TinyStories (20M tokens) training budget, hyperparameters, learning rate schedules.
5. **Results & Ablations**:
   - Loss curve convergence plot (AdamW vs Muon).
   - Inference latency scaling ($O(1)$ KV-Cache vs $O(T^2)$ naive).
   - MFU % hardware saturation on Apple Silicon / CUDA.
6. **Live Demonstration**: Hugging Face Space link + generation outputs across training checkpoints.

### Video Lecture / Tutorial Blueprint (Karpathy-Style Deep Dive)
- **Part 1: The Foundations**: Attention matrix math, PyTorch building blocks, Pre-LN vs Post-LN.
- **Part 2: Hardware Optimizations**: TF32, BF16, FlashAttention SRAM tiling, JIT compilation.
- **Part 3: The 2026 Frontier**: Why LLaMA-3 uses RoPE/RMSNorm/SwiGLU, and why Muon outperforms AdamW.
- **Part 4: Live Coding & Training**: Sharding TinyStories, training the 124M model on Mac, evaluating live stories.
