# GPT-2 (124M) Advanced Implementation Roadmap & Follow-up Guide

This guide details the theoretical foundation, architecture specifications, and step-by-step implementation code for the remaining three key milestones:
1. **Pretraining Data Pipeline (FineWeb-Edu 10B Sharding & Streaming DataLoader)**
2. **Validation Loop, HellaSwag Zero-Shot Benchmarking & Checkpointing**
3. **KV-Cache Accelerated Autoregressive Inference Engine**
*(Bonus: Vocabulary Size Tensor Core Warp Alignment to 50,304)*

---

## Table of Contents
- [0. Vocabulary Alignment to 50,304](#0-vocabulary-alignment-to-50304)
- [1. FineWeb-Edu Data Pipeline (`fineweb.py`)](#1-fineweb-edu-data-pipeline-finewebpy)
  - [1.1 Dataset Architecture & Shard Format](#11-dataset-architecture--shard-format)
  - [1.2 Tokenization & Shard Writer Script](#12-tokenization--shard-writer-script)
  - [1.3 Sharded Multi-Shard `DataLoaderLite`](#13-sharded-multi-shard-dataloaderlite)
- [2. Validation & Benchmark Pipeline](#2-validation--benchmark-pipeline)
  - [2.1 Validation Loss Evaluation Loop](#21-validation-loss-evaluation-loop)
  - [2.2 HellaSwag Zero-Shot Evaluation Engine](#22-hellaswag-zero-shot-evaluation-engine)
  - [2.3 Checkpointing & Periodic Generation](#23-checkpointing--periodic-generation)
- [3. Key-Value (KV) Cache Inference Engine](#3-key-value-kv-cache-inference-engine)
  - [3.1 Theoretical Complexity ($O(T)$ vs $O(1)$)](#31-theoretical-complexity-ot-vs-o1)
  - [3.2 Implementation with KV-Cache in Attention Blocks](#32-implementation-with-kv-cache-in-attention-blocks)
- [4. Implementation Checklist](#4-implementation-checklist)

---

## 0. Vocabulary Alignment to 50,304

### Rationale
In modern GPU architectures (NVIDIA Tensor Cores, Apple Silicon NE, AMD Matrix Cores), matrix multiplications ($M \times K \times N$) execute through hardware warps (e.g. 32 threads on NVIDIA, 64-128 SIMD lanes).
* Default GPT-2 vocabulary: `50,257` (prime factor components: $50,257 = 7 \times 7179$, not divisible by 16, 32, 64, or 128).
* Aligned vocabulary: `50,304` ($50,304 = 64 \times 786 = 128 \times 393 = 256 \times 196.5$).

### Modification in `train_gpt2.py`
```python
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # Padded from 50257 for Tensor Core tile alignment
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
```
> **Note:** The added 47 vocabulary tokens will simply remain untrained (their embeddings receive zero gradient if never present in the target tokens) with negligible parameter increase (~36k parameters).

---

## 1. FineWeb-Edu Data Pipeline (`fineweb.py`)

### 1.1 Dataset Architecture & Shard Format
To train GPT-2 (124M) on high-quality web data, we use the Hugging Face **FineWeb-Edu** (10BT sample).
* Each document is delimited by `<|endoftext|>` token (ID: `50256`).
* Shards are written as contiguous 1D arrays of `uint16` (since $50,304 < 65,535$, saving 50% storage over `int32` / `int64`).
* Shard size: $100\text{M}$ tokens ($200\text{MB}$ per shard in `uint16`). Shard 0 is held out for validation; Shards 1..99 are training shards.

### 1.2 Tokenization & Shard Writer Script
Save this script as `data/fineweb.py`:

```python
"""
FineWeb-Edu dataset sharder for GPT-2 pretraining.
Downloads and tokenizes FineWeb-Edu (10B tokens sample) into 100M token shards.
"""
import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# 1) Download FineWeb-Edu
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

# 2) Tokenizer setup
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # 50256

def tokenize(doc):
    # tokenizes a single document and prefixes with <|endoftext|>
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens, dtype=np.uint16)
    return tokens_np

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# 3) Multiprocessing tokenization
nprocs = max(1, os.cpu_count() // 2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    for tokens in pool.imap(tokenize, dataset, chunksize=16):
        if token_count + len(tokens) < shard_size:
            all_tokens_np[token_count:token_count + len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write remainder to current shard
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate next shard with leftovers
            all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])
```

### 1.3 Sharded Multi-Shard `DataLoaderLite`
Replace `DataLoaderLite` in `train_gpt2.py` with this dynamic memory-mapped shard loader:

```python
def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, split, data_root="data/edu_fineweb10B"):
        self.B = B
        self.T = T
        self.split = split
        assert split in {'train', 'val'}
        self.data_root = data_root

        # list all shard files for the requested split
        shards = os.listdir(data_root)
        shards = [s for s in shards if s.startswith(f"edufineweb_{split}")]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"No shards found for split {split} in {data_root}"
        
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T

        # if loading next batch reaches end of shard, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0
        return x, y
```

---

## 2. Validation & Benchmark Pipeline

### 2.1 Validation Loss Evaluation Loop
Evaluate holdout loss over $N = 20$ batches without computing gradients:

```python
val_loader = DataLoaderLite(B=B, T=T, split="val")

def evaluate_val_loss(model, val_loader, eval_steps=20, device="cuda"):
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
```

---

### 2.2 HellaSwag Zero-Shot Evaluation Engine

#### Task Overview
HellaSwag is a hard sentence-completion benchmark. For each context, the model evaluates 4 candidate continuations and predicts the continuation with the lowest cross-entropy loss (highest average token log-likelihood over the completion segment).

```
Context: "A woman is seen sitting on a couch. She..."
Ending 0: "...picks up a remote control and turns on the TV." (Target)
Ending 1: "...starts running a marathon."
Ending 2: "...transforms into an airplane."
Ending 3: "...swims across the Pacific Ocean."
```

#### Evaluation Function
```python
import json
import urllib.request

def render_example(example):
    """
    Renders context + 4 endings into token tensors and a mask indicating completion tokens.
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
        "mask": []
    }
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end) # leading space for BPE boundary
        tok_rows.append(ctx_tokens + end_tokens)
        # mask is 0 on the context prefix, 1 on the candidate continuation
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
    
    # Pad all 4 rows to the same length
    max_len = max(len(r) for r in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask

def evaluate_hellaswag(model, device="cuda"):
    model.eval()
    num_correct_norm = 0
    num_total = 0
    
    # Download val set if not present
    hellaswag_val_file = "material/hellaswag_val.jsonl"
    if not os.path.exists(hellaswag_val_file):
        url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
        urllib.request.urlretrieve(url, hellaswag_val_file)

    with open(hellaswag_val_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        example = json.loads(line)
        data, tokens, mask = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # Get logits
        with torch.no_grad():
            with autocast_ctx:
                logits, _ = model(tokens) # (4, T, vocab_size)
            
            # Shift logits and tokens for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_tokens = tokens[:, 1:].contiguous()
            shift_mask = mask[:, 1:].contiguous()

            # Cross entropy per token
            loss_flat = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_tokens.view(-1), reduction="none")
            loss_per_token = loss_flat.view(4, -1)
            
            # Average loss only over continuation tokens (where shift_mask == 1)
            masked_loss = (loss_per_token * shift_mask).sum(dim=1) / shift_mask.sum(dim=1)
            pred_norm = masked_loss.argmin().item()

        if pred_norm == data["label"]:
            num_correct_norm += 1
        num_total += 1

    acc_norm = num_correct_norm / num_total
    model.train()
    return acc_norm
```
> **Reference Baseline**: OpenAI GPT-2 (124M) achieves **~29.55%** zero-shot accuracy on HellaSwag. Random guess is **25.00%**.

---

### 2.3 Checkpointing & Periodic Generation

```python
# In training loop every 250 steps:
if step > 0 and step % 250 == 0:
    val_loss = evaluate_val_loss(model, val_loader, device=device)
    hella_acc = evaluate_hellaswag(model, device=device)
    print(f"[EVAL step {step}] val_loss: {val_loss:.4f} | hellaswag_acc: {hella_acc*100:.2f}%")
    
    # Save checkpoint
    checkpoint_path = f"log/model_{step:05d}.pt"
    os.makedirs("log", exist_ok=True)
    checkpoint = {
        'model': model.state_dict(),
        'config': model.config,
        'step': step,
        'val_loss': val_loss,
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"=> saved checkpoint to {checkpoint_path}")
```

---

## 3. Key-Value (KV) Cache Inference Engine

### 3.1 Theoretical Complexity ($O(T)$ vs $O(1)$)

In naive autoregressive decoding, generating token $t+1$ requires passing all tokens $1 \dots t$ through all 12 transformer layers, recomputing queries, keys, and values for all past tokens.
* **Naive generation cost**: $\sum_{t=1}^{T} O(t) = O(T^2)$ matrix operations.
* **KV-Cache generation cost**: Cache $(K_{\text{past}}, V_{\text{past}})$ for previous tokens. On step $t+1$, only compute $Q, K, V$ for the single new token and concatenate $K_{\text{new}}, V_{\text{new}}$ to cache.
* **Cost per token**: $O(1)$ forward compute per token, total $O(T)$ complexity.

```
Step t:
Q_new: (B, nh, 1, hs)
K_cache: [ K_past (B, nh, t-1, hs)  |  K_new (B, nh, 1, hs) ] -> (B, nh, t, hs)
V_cache: [ V_past (B, nh, t-1, hs)  |  V_new (B, nh, 1, hs) ] -> (B, nh, t, hs)
Attn = Softmax(Q_new @ K_cache.T / sqrt(hs)) @ V_cache  -> (B, nh, 1, hs)
```

### 3.2 Implementation with KV-Cache in Attention Blocks

Modify `CausalSelfAttention` to support optional KV caching:

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        setattr(self.c_proj, "NANOGPT_SCALE_INIT", 1)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, kv_cache=None):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if kv_cache is not None:
            # kv_cache is a tuple of (k_past, v_past)
            k_past, v_past = kv_cache
            if k_past is not None:
                k = torch.cat([k_past, k], dim=2) # concatenate along sequence dimension T
                v = torch.cat([v_past, v], dim=2)
            new_kv_cache = (k, v)
        else:
            new_kv_cache = None

        if kv_cache is None:
            # Full sequence training: use FlashAttention / Fused SDPA
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Inference step: query is (B, nh, 1, hs), key/value is (B, nh, Total_T, hs)
            # causal mask not needed since query can attend to all past keys
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, 1, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y, new_kv_cache
```

#### Fast Generation Loop with KV-Cache
```python
@torch.no_grad()
def generate_with_kv_cache(model, prompt_tokens, max_new_tokens=50, temperature=0.8, top_k=50):
    model.eval()
    B = prompt_tokens.size(0)
    
    # 1) Prefill phase: feed prompt to prime the KV cache
    kv_caches = [None] * model.config.n_layer
    curr_tokens = prompt_tokens
    generated = [prompt_tokens]

    # Forward prompt through model layer-by-layer
    pos = torch.arange(0, curr_tokens.size(1), dtype=torch.long, device=curr_tokens.device)
    x = model.transformer['wte'](curr_tokens) + model.transformer['wpe'](pos)
    
    new_kv_caches = []
    for i, block in enumerate(model.transformer['h']):
        # LN1 + Attn with cache
        residual = x
        x_norm = block.ln_1(x)
        attn_out, cache = block.attn(x_norm, kv_cache=kv_caches[i])
        x = residual + attn_out
        new_kv_caches.append(cache)
        # LN2 + MLP
        x = x + block.mlp(block.ln_2(x))
    kv_caches = new_kv_caches

    logits = model.lm_head(model.transformer['ln_f'](x[:, -1:, :])) # (B, 1, vocab_size)
    
    # Sample first token
    logits = logits[:, -1, :] / temperature
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = -float('Inf')
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
    generated.append(next_token)

    # 2) Decoding phase: token-by-token with KV cache (shape 1, 1)
    curr_token = next_token
    for step in range(max_new_tokens - 1):
        pos = torch.tensor([prompt_tokens.size(1) + step], dtype=torch.long, device=curr_token.device)
        x = model.transformer['wte'](curr_token) + model.transformer['wpe'](pos)
        
        new_kv_caches = []
        for i, block in enumerate(model.transformer['h']):
            residual = x
            x_norm = block.ln_1(x)
            attn_out, cache = block.attn(x_norm, kv_cache=kv_caches[i])
            x = residual + attn_out
            new_kv_caches.append(cache)
            x = x + block.mlp(block.ln_2(x))
        kv_caches = new_kv_caches

        logits = model.lm_head(model.transformer['ln_f'](x[:, -1:, :]))[:, -1, :] / temperature
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated.append(next_token)
        curr_token = next_token

    return torch.cat(generated, dim=1)
```

---

## 4. Implementation Checklist

| Phase | Milestone | File Target | Status |
|:---:|:---|:---|:---:|
| **0** | Align `vocab_size` to `50304` | [`brain/train_gpt2.py`](file:///Users/apple/Desktop/Projects/gpt-2(124M)/brain/train_gpt2.py) | Ready to edit |
| **1** | Write FineWeb-Edu 10B sharding script | `data/fineweb.py` | Ready to create |
| **1** | Multi-shard streaming `DataLoaderLite` | [`brain/train_gpt2.py`](file:///Users/apple/Desktop/Projects/gpt-2(124M)/brain/train_gpt2.py) | Ready to integrate |
| **2** | Periodic holdout validation loss loop | [`brain/train_gpt2.py`](file:///Users/apple/Desktop/Projects/gpt-2(124M)/brain/train_gpt2.py) | Ready to integrate |
| **2** | HellaSwag zero-shot evaluation engine | [`brain/train_gpt2.py`](file:///Users/apple/Desktop/Projects/gpt-2(124M)/brain/train_gpt2.py) | Ready to integrate |
| **2** | Checkpoint saving & loading (`torch.save`) | [`brain/train_gpt2.py`](file:///Users/apple/Desktop/Projects/gpt-2(124M)/brain/train_gpt2.py) | Ready to integrate |
| **3** | KV-Cache inference engine for generation | [`brain/train_gpt2.py`](file:///Users/apple/Desktop/Projects/gpt-2(124M)/brain/train_gpt2.py) | Ready to integrate |
