# Grouped-Query Attention (GQA) Guide

A comprehensive guide on **Grouped-Query Attention (GQA)** (Ainslie et al., 2023), how it unifies Multi-Head Attention (MHA) and Multi-Query Attention (MQA), and how it dramatically reduces the memory footprint of autoregressive inference.

---

## 1. The Inference Memory Bottleneck

During autoregressive text generation, the key and value states of all past tokens must be preserved in GPU memory (the **KV-Cache**).

For sequence length $T$, batch size $B$, layers $L$, and hidden size $d$:
$$\text{Memory}_{\text{KV}} = 2 \times B \times L \times T \times d \times (\text{bytes per element})$$

For large context windows or concurrent serving, the KV-Cache quickly **exceeds total GPU VRAM**, making LLM serving memory-bound rather than compute-bound.

---

## 2. Spectrum: MHA vs. MQA vs. GQA

```
Multi-Head Attention (MHA)       Grouped-Query Attention (GQA)      Multi-Query Attention (MQA)
    (GPT-2 / Original)                 (Axiom-LM / LLaMA-3)                (StarCoder / Falcon)

Q0 Q1 Q2 Q3 Q4 Q5 ... Q11        Q0 Q1 Q2  Q3 Q4 Q5 ... Q11         Q0 Q1 Q2 Q3 ... Q11
 │  │  │  │  │  │      │          └──┬──┘   └──┬──┘      │           └────────┬────────┘
K0 K1 K2 K3 K4 K5 ... K11           KV0       KV1       KV3                  KV0
V0 V1 V2 V3 V4 V5 ... V11

12 Query Heads                   12 Query Heads                     12 Query Heads
12 Key/Value Heads               4 Key/Value Heads                  1 Key/Value Head
(1 KV head per Q head)           (1 KV head per 3 Q heads)          (1 KV head shared by all Q)
```

### Key Differences:
1. **Multi-Head Attention (MHA):** $N_q = N_{kv} = 12$. Rich expressivity, but maximum memory usage.
2. **Multi-Query Attention (MQA):** $N_q = 12, N_{kv} = 1$. Lowest possible memory, but noticeable quality degradation on complex reasoning tasks.
3. **Grouped-Query Attention (GQA):** $N_q = 12, N_{kv} = 4$. The sweet spot! Saves **66.7% of memory** with virtually no loss in model capacity.

---

## 3. How GQA Attention is Computed

In GQA, each KV head is shared across a group of $G = N_q / N_{kv}$ query heads.
For $N_q = 12$ and $N_{kv} = 4$, the group size is $G = 3$.

During the attention computation, we expand the Key and Value tensors along the head dimension using repeat/broadcast operations so that each query head interacts with its assigned key/value group:

```python
# Keys shape: (B, N_kv, T, d_k) -> Repeat G times -> (B, N_q, T, d_k)
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    if n_rep == 1:
        return x
    B, N_kv, T, d_k = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, N_kv, n_rep, T, d_k)
        .reshape(B, N_kv * n_rep, T, d_k)
    )
```

---

## 4. PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int = 768, n_heads: int = 12, n_kv_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Expand K and V to match Q head count
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Fused SDPA
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)
```

---

## 5. Quantitative Impact in Axiom-LM

* **KV-Cache Size per Token**: Reduced from $12 \times 64 \times 2 = 1,536$ bytes per layer to $4 \times 64 \times 2 = 512$ bytes per layer.
* **Serving Capacity**: At 1024 context tokens, the KV cache requires only **$12.5\text{ MB}$ per sequence** instead of **$37.7\text{ MB}$ per sequence**.
