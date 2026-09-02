# FlashAttention & Fast Scaled Dot-Product Attention Guide

A technical reference and mathematical guide for **FlashAttention**, **FlashAttention-2**, and PyTorch's native **`torch.nn.functional.scaled_dot_product_attention` (SDPA)**, as utilized in modern LLMs and Andrej Karpathy's *"Let's reproduce GPT-2 (124M)"*.

---

## 1. Papers & References

* **FlashAttention (2022):** *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* — Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré ([arXiv:2205.14135](https://arxiv.org/abs/2205.14135))
* **FlashAttention-2 (2023):** *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* — Tri Dao ([arXiv:2307.08691](https://arxiv.org/abs/2307.08691))
* **PyTorch Official API:** [`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

---

## 2. The Bottleneck of Standard Attention

Standard Self-Attention computes:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

Where:
* $Q, K, V \in \mathbb{R}^{B \times N_h \times T \times d_k}$
* $S = \frac{Q K^T}{\sqrt{d_k}} \in \mathbb{R}^{B \times N_h \times T \times T}$
* $P = \text{softmax}(S + M) \in \mathbb{R}^{B \times N_h \times T \times T}$
* $O = P V \in \mathbb{R}^{B \times N_h \times T \times d_k}$

### The Hardware Reality: Compute-Bound vs. Memory-Bound (IO-Bound)

Modern GPUs (like NVIDIA A100/H100 or Apple Silicon M-series) have two main types of memory:
1. **Global High-Bandwidth Memory (HBM / VRAM):** Large ($16\text{GB}-80\text{GB}$), but relatively slow ($\approx 1.5 - 3.3\text{ TB/s}$).
2. **On-Chip SRAM (L1 Cache / Shared Memory):** Extremely small ($\approx 192\text{KB}$ per SM), but blazing fast ($\approx 19\text{ TB/s}$).

```
┌─────────────────────────────────────────────────────────────┐
│ GPU Global Memory (HBM / VRAM) ~ 1.5 - 3.3 TB/s (Slow IO)   │
└──────────────────────────────┬──────────────────────────────┘
                               │  Roundtrips (Write/Read)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ On-Chip SRAM (Shared Memory / L1 Cache) ~ 19 TB/s (Fast)    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Tensor Cores / Compute Units (Floating Point ALUs)          │
└─────────────────────────────────────────────────────────────┘
```

### Why Standard Attention is Slow ($O(T^2)$ Memory Traffic):

In standard PyTorch eager attention:
1. Load $Q, K$ from HBM $\to$ Compute $S = Q K^T$ $\to$ **Write $(T \times T)$ matrix $S$ back to HBM**.
2. Read $S$ from HBM $\to$ Apply Mask & Softmax $\to$ **Write $(T \times T)$ matrix $P$ back to HBM**.
3. Read $P, V$ from HBM $\to$ Compute $O = P V$ $\to$ **Write output $O$ to HBM**.

For sequence length $T=1024$ or $T=8192$, materializing the full $(T \times T)$ matrix creates **massive memory bandwidth stalls** and consumes $O(T^2)$ memory space in VRAM.

---

## 3. How FlashAttention Solves This

FlashAttention is an **exact** algorithm (no approximation, same mathematical result) that makes attention **IO-Aware**:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        FLASHATTENTION ALGORITHM                        │
│                                                                        │
│  1. Tiling: Split Q, K, V into small blocks that fit in SRAM.          │
│  2. Online Softmax: Compute softmax incrementally block-by-block.      │
│  3. Kernel Fusion: Compute entire attention in ONE fused GPU kernel.   │
│  4. Recomputation: Never store (T x T) attention matrix in HBM!        │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. Tiling & Kernel Fusion
- Loads blocks of $Q, K, V$ into on-chip **SRAM**.
- Computes matrix multiplications and softmax entirely in SRAM.
- Writes **only the final output $O$** back to HBM.
- **Memory footprint drops from $O(T^2)$ to $O(T)$!**

### 2. Online Softmax (Incremental Rescaling)
Standard softmax requires knowing the maximum value $m = \max(x)$ and denominator $\sum e^{x_i - m}$ across the **entire row** of length $T$.

Online softmax maintains running statistics $(m^{(j)}, \ell^{(j)})$ and updates the output accumulator $O^{(j)}$ incrementally as each block of $K, V$ is streamed into SRAM:

$$m_{\text{new}} = \max(m_{\text{prev}}, m_{\text{block}})$$
$$\ell_{\text{new}} = e^{m_{\text{prev}} - m_{\text{new}}} \ell_{\text{prev}} + e^{m_{\text{block}} - m_{\text{new}}} \ell_{\text{block}}$$
$$O_{\text{new}} = \text{diag}\left(e^{m_{\text{prev}} - m_{\text{new}}}\right) O_{\text{prev}} + e^{S_{\text{block}} - m_{\text{new}}} V_{\text{block}}$$

### 3. Recomputation in the Backward Pass
- In standard backpropagation, the $(T \times T)$ attention matrix $P$ must be stored during the forward pass to compute $\nabla Q, \nabla K, \nabla V$.
- FlashAttention **does not store $P$ at all**.
- During the backward pass, it quickly recomputes $S$ and $P$ on-the-fly from $Q, K, V$ inside SRAM.
- Because SRAM compute is faster than HBM read/write latency, **recomputation is actually faster than reading from memory!**

---

## 4. PyTorch API: `F.scaled_dot_product_attention`

PyTorch 2.0+ includes native FlashAttention integration via `torch.nn.functional.scaled_dot_product_attention` (SDPA).

### Signature

```python
torch.nn.functional.scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None
)
```

| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| **`query`** | `Tensor` | *required* | Query tensor of shape `(B, n_head, T_q, head_dim)`. |
| **`key`** | `Tensor` | *required* | Key tensor of shape `(B, n_head, T_k, head_dim)`. |
| **`value`** | `Tensor` | *required* | Value tensor of shape `(B, n_head, T_v, head_dim)`. |
| **`attn_mask`** | `Tensor` / `None` | `None` | Optional boolean or float attention mask. |
| **`dropout_p`** | `float` | `0.0` | Dropout probability. |
| **`is_causal`** | `bool` | `False` | When `True`, automatically applies autoregressive lower-triangular causal masking **without materializing an explicit mask tensor**. |
| **`scale`** | `float` / `None` | `None` | Scaling factor. Defaults to $\frac{1}{\sqrt{d_k}}$. |

---

## 5. Implementation: Before vs. After in GPT-2

###  Before: Standard Manual Attention (Slow, High VRAM)

```python
# (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)  <-- Heavy O(T^2) HBM allocation!
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
att = F.softmax(att, dim=-1)
y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
```

###  After: FlashAttention SDPA (Fast, Fused, $O(T)$ Memory)

```python
# Single fused C++/CUDA/Metal kernel call:
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

> [!NOTE]
> When using `is_causal=True`, the lower-triangular causal mask buffer `self.bias` is **completely eliminated from the module**.

---

## 6. Full `CausalSelfAttention` Module with FlashAttention

```python
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # QKV linear projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        setattr(self.c_proj, "NANOGPT_SCALE_INIT", 1)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        # 1. Project to QKV
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # 2. Reshape to (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 3. FlashAttention (Fused Scaled Dot-Product Attention)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # 4. Re-assemble heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
```

---

## 7. Performance & Memory Comparison

| Feature | Standard Manual Attention | FlashAttention (SDPA) |
|:---|:---:|:---:|
| **Memory Complexity** | $O(T^2)$ | **$O(T)$** |
| **HBM Read/Writes** | $O(T^2)$ | **$O(T \cdot d_k)$** |
| **Speedup (Training)** | $1.0\times$ (Baseline) | **$2\times - 4\times$ faster** |
| **Max Context Length ($T$)** | $\approx 2\text{k} - 4\text{k}$ (OOM on GPU) | **$32\text{k} - 128\text{k}+$** |
| **Numerical Precision** | Standard FP32/FP16 | Identical (Exact Attention) |

---

## 8. Backend Dispatchers & Hardware Support

PyTorch automatically selects the fastest available backend:

1. **FlashAttention-2:** Used on NVIDIA Ampere, Ada, Hopper (CUDA compute capability $\ge 8.0$, FP16/BF16).
2. **Memory-Efficient Attention (Cutlass):** Used on older NVIDIA GPUs (Volta, Turing - V100, T4, RTX 2080).
3. **Apple Silicon Metal (MPS):** PyTorch dispatches to Apple's native fused MPS graph attention kernel.
4. **C++ / Math Backend:** Vectorized CPU fallback with OpenMP.

To query or force specific backends:
```python
# Check which backend is active
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```
