# 🧭 Master LLM Pretraining & Modernization TO-DO Roadmap

> **Local Private Document**: This file is added to `.gitignore` so it stays completely private to your local machine and will never be pushed to GitHub.

---

## 📋 High-Level Progress Tracker

- [ ] **Phase 1: Lightweight Dataset Pipeline (`data/tinystories.py`)**
- [ ] **Phase 2: Train/Val Sharded `DataLoaderLite` with Dynamic Splitting**
- [ ] **Phase 3: Periodic Validation Loss & Holdout Evaluation Loop**
- [ ] **Phase 4: Live Generation & Visual Sampling Inside Training Loop**
- [ ] **Phase 5: Checkpointing (`torch.save` & Resuming States)**
- [ ] **Phase 6: Key-Value (KV) Cache Accelerated Inference Engine**
- [ ] **Phase 7: Modern Architecture Upgrades (RoPE + RMSNorm + SwiGLU + GQA)**
- [ ] **Phase 8: Next-Gen Muon Matrix Optimizer Integration**

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

# 1. Stream or load TinyStories
print("Loading TinyStories dataset from Hugging Face...")
dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

target_tokens = 20_000_000 # 20M tokens (~35 mins training on Mac)
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

### Goal
Upgrade `DataLoaderLite` in `train_gpt2.py` to seamlessly read binary token files with zero memory copies using numpy `memmap` or contiguous `fromfile`.

### Implementation Template
```python
class DataLoaderLite:
    def __init__(self, B, T, split="train", data_dir="data"):
        self.B = B
        self.T = T
        self.split = split
        filename = os.path.join(data_dir, f"{split}.bin")
        assert os.path.exists(filename), f"Binary dataset file {filename} not found. Run data/tinystories.py first."
        
        # Memory map or load uint16 tokens
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

### Goal
1. Every $N$ steps (e.g. every 100 steps), evaluate validation loss over 20 batches without gradients (`torch.no_grad()`).
2. Generate text samples from a prompt (*"Once upon a time,"*) to watch text quality improve in real-time.
3. Save checkpoint weights and optimizer state to disk.

### Implementation Snippet
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

# Inside the main training loop:
if step > 0 and step % 100 == 0:
    val_loss = evaluate_val_loss(model, val_loader, eval_steps=20, device=device)
    print(f"\n>>> [STEP {step}] Validation Loss: {val_loss:.4f} <<<")
    
    # Save checkpoint
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

### Goal
Implement Key-Value caching in `CausalSelfAttention` so that generating tokens takes $O(1)$ time per step instead of quadratic $O(T^2)$ re-computation.

### Attention Layer with KV-Cache
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
            # Training / prefill: fast fused flash attention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Single-token decoding step
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y, new_kv_cache
```

---

## 🛠️ Phase 7: Modern Architecture Upgrades (RoPE + RMSNorm + SwiGLU)

### 1. RMSNorm (Root Mean Square Normalization)
```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
```

### 2. Rotary Position Embeddings (RoPE)
```python
def precompute_rope_frequencies(dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # complex form (cos + i*sin)
    return freqs_cis

def apply_rope(x, freqs_cis):
    # x is (B, nh, T, hs)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:x.shape[2], :].unsqueeze(0).unsqueeze(0)
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return x_rotated.type_as(x)
```

### 3. SwiGLU Feed-Forward Network
```python
class SwiGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(2 * (4 * config.n_embd) / 3) # 8/3 expansion
        self.w_gate = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w_up = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.w_down.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
```

---

## 🛠️ Phase 8: Next-Gen Muon Optimizer Integration

### Mathematical Formulation
$$\mathbf{X}_{k+1} = \frac{1}{2} \mathbf{X}_k (3\mathbf{I} - \mathbf{X}_k^T \mathbf{X}_k)$$

### Optimizer Definition
```python
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of 2D matrix G.
    """
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

## 🎯 Quick Command Checklist

```bash
# Step 1: Generate dataset shards
python data/tinystories.py

# Step 2: Run training
python brain/train_gpt2.py

# Step 3: Run fast interactive generation
jupyter notebook brain/play.ipynb
```
