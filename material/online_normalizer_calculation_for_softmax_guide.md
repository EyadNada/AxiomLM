# Online Normalizer Calculation for Softmax — Technical Guide & Paper Breakdown

A deep-dive technical reference on the seminal NVIDIA paper **"Online normalizer calculation for softmax"** (Maxim Milakov & Natalia Gimelshein, 2018), explaining the mathematics of online safe softmax and how it serves as the foundational core for **FlashAttention**.

---

## 1. Paper Metadata & Citation

* **Title:** *Online normalizer calculation for softmax*
* **Authors:** Maxim Milakov & Natalia Gimelshein (NVIDIA)
* **Publication:** [arXiv:1805.02867](https://arxiv.org/abs/1805.02867) (July 2018)
* **Local PDF:** [`material/online_normalizer_calculation_for_softmax_paper.pdf`](./online_normalizer_calculation_for_softmax_paper.pdf)

---

## 2. Executive Summary & Why It Matters

The **Softmax** function is ubiquitous in deep learning, especially across Transformer attention mechanisms and output classification heads:

$$\text{Softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{V} e^{x_j}}$$

### The Big Idea:
1. Standard "safe" softmax requires **3 passes** over memory (1st pass: find max, 2nd pass: compute sum of exponentials, 3rd pass: normalize and write output).
2. Milakov & Gimelshein invented **Online Softmax**, an algorithm that merges the maximum-finding and exponential-summation steps into a **single pass** using an online rescaling recurrence relation.
3. **FlashAttention Connection:** Tri Dao et al. (2022) directly adopted this online normalizer technique to invent **FlashAttention**, enabling block-by-block SRAM attention computation without ever materializing the $O(T^2)$ matrix in GPU global memory.

---

## 3. The 3 Approaches to Softmax

### 1. Naive Softmax (1 Pass — Numerically Unstable)
$$\text{Softmax}(x)_i = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$
* **Problem:** If $x_i > 88$ (for float32) or $x_i > 11$ (for float16/bfloat16), $e^{x_i}$ **overflows to $+\infty$**, resulting in `NaN` loss.

---

### 2. Standard Safe Softmax (3 Passes over Memory — Standard PyTorch)
To prevent overflow, we subtract the maximum element $m = \max_j x_j$:
$$\text{Softmax}(x)_i = \frac{e^{x_i - m}}{\sum_{j} e^{x_j - m}}$$

```python
# Pass 1: Find row maximum (Read N elements from VRAM)
m = max(x)

# Pass 2: Compute exponential sum / normalizer (Read N elements from VRAM)
d = sum(exp(x[i] - m) for i in range(N))

# Pass 3: Normalize and write output (Read N elements, Write N elements to VRAM)
for i in range(N):
    y[i] = exp(x[i] - m) / d
```
* **Total IO:** $3 \times \text{reads} + 1 \times \text{write} = 4N$ memory transfers.

---

### 3. Milakov & Gimelshein: Online Safe Softmax (2 Passes / Streaming)

Can we compute both the max $m$ and the normalizer sum $d$ simultaneously in a single streaming pass without knowing the global max upfront? **Yes!**

#### The Recurrence Relation:

Let $m_k$ be the running maximum and $d_k$ be the running normalizer sum after seeing the first $k$ elements:

1. **Initial state:**
   $$m_0 = -\infty, \quad d_0 = 0$$

2. **Update rule when processing element $x_k$ (or vector block $x^{(k)}$):**
   $$m_k = \max(m_{k-1}, x_k)$$
   $$d_k = d_{k-1} \cdot e^{m_{k-1} - m_k} + e^{x_k - m_k}$$

```
When a new larger maximum is encountered (m_k > m_{k-1}):
The previous accumulator d_{k-1} is simply rescaled by multiplying with e^(m_{k-1} - m_k) <= 1.0!
```

```python
# Online Pass (Streaming / 1 Pass):
m = -float('inf')
d = 0.0

for i in range(N):
    m_prev = m
    m = max(m_prev, x[i])
    # Rescale previous sum and add new exponential:
    d = d * math.exp(m_prev - m) + math.exp(x[i] - m)

# Final Normalization Pass (1 Pass):
for i in range(N):
    y[i] = math.exp(x[i] - m) / d
```
* **Total IO:** Reduced from $3$ memory passes to **$2$ passes** (or $1$ pass when fused with downstream matrix multiplication in SRAM).

---

## 4. The Bridge to FlashAttention

In self-attention:
$$O = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

In standard attention, we cannot compute $O$ until the entire row of $Q K^T$ is computed and normalized.

Using the **Milakov-Gimelshein Online Normalizer**, FlashAttention processes $K, V$ in blocks $K^{(j)}, V^{(j)}$:

```
┌────────────────────────────────────────────────────────────────────────┐
│ FlashAttention Block Update (Directly using Milakov & Gimelshein):     │
│                                                                        │
│  1. Compute block scores: S_j = Q * (K_j)^T                            │
│  2. Compute block max:    m_curr = max(rowmax(S_j))                    │
│  3. Update running max:   m_new = max(m_prev, m_curr)                  │
│  4. Rescale running sum:  l_new = l_prev * e^(m_prev - m_new) +        │
│                                   sum(e^(S_j - m_new))                 │
│  5. Rescale running out:  O_new = O_prev * e^(m_prev - m_new) +        │
│                                   e^(S_j - m_new) * V_j                │
└────────────────────────────────────────────────────────────────────────┘
```

Because of this recurrence, **FlashAttention never has to write the $(T \times T)$ attention matrix to VRAM**, achieving $O(T)$ memory complexity.

---

## 5. Performance Improvements Reported in the Paper

| Benchmark | Standard Safe Softmax | Online Softmax | Speedup |
|:---|:---:|:---:|:---:|
| **Softmax (NVIDIA GPU Kernel)** | Baseline | Reduced Memory Reads | **$1.3\times$ faster** |
| **Softmax + Top-K Fused Kernel** | Multi-kernel pipeline | Fully fused single-pass | **Up to $5.0\times$ faster** |
| **FlashAttention (Downstream)** | $O(T^2)$ memory bound | SRAM Online Normalizer | **$2\times - 4\times$ faster** |

---

## 6. Summary Checklist

1. **Safety:** Keeps full numerical stability against overflow ($e^{x - m}$).
2. **Efficiency:** Merges max-finding and sum-accumulation into a single pass.
3. **Rescaling Trick:** Whenever max changes, multiply accumulated sum by $e^{m_{\text{old}} - m_{\text{new}}}$.
4. **Foundation:** Directly enables online tiling for FlashAttention-1, FlashAttention-2, and FlashAttention-3.
