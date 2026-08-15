# Text Generation & Sampling Strategies in GPT-2

This guide covers the mechanics of autoregressive text generation, logit manipulation, and sampling parameters (such as `top_k`, `temperature`, and `top_p`) used in Hugging Face Transformers and reproduced in PyTorch for GPT-2 (124M).

---

## 1. How Autoregressive Text Generation Works

GPT-2 generates text **autoregressively**—predicting one token at a time and feeding the new token back into the model for the next step.

```
Input Tokens: [Hello, I'm a language model,]  (B=5, T=8)
      │
      ▼
GPT-2 Forward Pass ──► Logits: (B, T, vocab_size=50257)
      │
      ▼
Slice Last Token Logits: logits[:, -1, :] ──► (B, 50257)
      │
      ▼
Softmax / Sampling (Top-K / Temperature / Top-P) ──► Next Token `xcol` (B, 1)
      │
      ▼
Append `xcol` to Input: x = [x, xcol] ──► (B, T+1)
      │
      └─► Repeat until T == max_length
```

---

## 2. Key Logit Manipulation & Sampling Parameters

*(Reference: Hugging Face `GenerationConfig` / `transformers.generation`)*

### 1. `top_k` (defaults to `50` in Hugging Face)
- **What it does:** Filters the vocabulary to retain only the top $k$ highest-probability tokens. All other tokens have their probability set to 0.
- **Why use it:** Prevents the model from sampling rare, out-of-context, or bizarre tokens from the long tail of the probability distribution.
- **In our code:**
  ```python
  # probs is (B, vocab_size)
  topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # (B, 50)
  ix = torch.multinomial(topk_probs, 1)                    # (B, 1) sample relative to top-50 probs
  xcol = torch.gather(topk_indices, -1, ix)                # (B, 1) map back to vocab token IDs
  ```

---

### 2. `temperature` (defaults to `1.0`)
- **What it does:** Modulates next-token probability distribution by scaling logits before softmax:
  $$\text{probs} = \text{softmax}\left(\frac{\text{logits}}{T}\right)$$
- **Effects:**
  - **$T = 1.0$ (Default):** Standard model predictions without scaling.
  - **$T < 1.0$ (Cold, e.g., 0.2 - 0.7):** Sharper distribution; model is more confident, repetitive, and conservative. As $T \to 0$, it approaches greedy `argmax`.
  - **$T > 1.0$ (Hot, e.g., 1.2 - 1.8):** Flatter distribution; higher randomness, creativity, and diversity (with increased risk of gibberish).

---

### 3. `top_p` (Nucleus Sampling, defaults to `1.0`)
- **What it does:** Keeps the smallest set of most probable tokens whose cumulative probability $\ge \text{top\_p}$ (e.g., 0.90 or 0.95).
- **Difference from `top_k`:** `top_k` uses a fixed number of tokens (e.g., always 50), whereas `top_p` dynamically adjusts the candidate pool size depending on model confidence (narrow pool when confident, wider pool when uncertain).

---

### 4. Advanced Logit Truncation Parameters

| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| **`min_p`** | `float` | `None` | Minimum token probability scaled by the probability of the most likely token (typically `0.01` to `0.2`). |
| **`typical_p`** | `float` | `1.0` | Local typicality metric comparing conditional probability to expected information content. |
| **`epsilon_cutoff`** | `float` | `0.0` | Discards tokens with probability strictly below a fixed threshold $\epsilon$ (Truncation Sampling). |
| **`eta_cutoff`** | `float` | `0.0` | Locally dynamic cutoff based on entropy of the distribution. |
| **`repetition_penalty`** | `float` | `1.0` | Penalizes logits of previously generated tokens to prevent infinite loops. |

---

## 3. Complete Generation Loop in PyTorch

Here is the exact implementation used in `train_gpt2.py`:

```python
import torch
import torch.nn.functional as F
import tiktoken

num_return_sequences = 5
max_length = 30

model = GPT.from_pretrained("gpt2")
model.eval()
model.to("mps") # or "cuda" / "cpu"

# 1. Prefix tokens
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to('mps')

# 2. Set seed for reproducibility
torch.manual_seed(42)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(42)

# 3. Autoregressive generation loop
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        
        # Take logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        
        # Calculate probabilities with softmax
        probs = F.softmax(logits, dim=-1) # (B, vocab_size)
        
        # Top-K filtering (k=50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # (B, 50)
        
        # Sample one token index per sequence from the top-k distribution
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        
        # Gather the corresponding token ID from vocab
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        
        # Append sampled token to sequence
        x = torch.cat((x, xcol), dim=1) # (B, T+1)

# 4. Print generated output sequences
for i in range(num_return_sequences):
    decoded_tokens = x[i, :max_length].tolist()
    decoded_text = enc.decode(decoded_tokens)
    print(f"sample {i}: {decoded_text}")
```

---

## 4. Key PyTorch Functions Explained

- `torch.topk(probs, k, dim=-1)`: Returns the $k$ largest elements and their indices along the specified dimension.
- `torch.multinomial(input, num_samples)`: Samples indices from a multinomial probability distribution.
- `torch.gather(input, dim, index)`: Gathers values along an axis specified by `dim` using `index` positions.
- `torch.cat((x, xcol), dim=1)`: Concatenates the newly sampled token column `xcol` to the running sequence `x`.
