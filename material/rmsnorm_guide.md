# RMSNorm (Root Mean Square Normalization) Guide

A technical and systems-oriented guide to **Root Mean Square Normalization (RMSNorm)** (Zhang & Sennrich, 2019) and why every modern frontier LLM (LLaMA-3, Mistral, DeepSeek) replaced LayerNorm with RMSNorm.

---

## 1. The Anatomy of Standard LayerNorm (GPT-2)

In standard LayerNorm across dimension $d$:
1. Compute Mean:
   $$\mu = \frac{1}{d} \sum_{i=1}^d x_i$$
2. Compute Variance:
   $$\sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2$$
3. Standardize and Scale:
   $$y_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma_i + \beta_i$$

### The Systems Problem:
* **Two Full Reductions**: The GPU must sweep through all $d$ numbers twice (first to compute $\mu$, then to compute $\sigma^2$).
* **Memory Bandwidth Choke**: Normalization is an **element-wise memory-bound kernel**. The arithmetic intensity (FLOPs per byte loaded from VRAM) is tiny.

---

## 2. The Key Discovery: Shift Invariance is Unnecessary

Zhang & Sennrich (2019) hypothesized that the regularizing effect of LayerNorm does **not** come from re-centering the mean ($\mu = 0$), but rather from the **scaling of the activation energy (magnitude)**!

By discarding the mean-centering step:
$$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}$$

$$\mathbf{y} = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \odot \mathbf{\gamma}$$

Notice:
* No mean calculation $\mu$.
* No subtraction $(x - \mu)$.
* No learnable bias parameter $\beta$.

```
LayerNorm:  [ Load x ] ──► [ Sum for Mean ] ──► [ Diff & Square for Var ] ──► [ Normalize & Scale + Bias ]
RMSNorm:    [ Load x ] ──► [ Square & Sum (Single Pass) ] ──────────────► [ Scale by Gamma ]
```

---

## 3. Benchmark & Computational Gains

| Property | LayerNorm (GPT-2) | RMSNorm (Axiom-LM / LLaMA-3) |
| :--- | :--- | :--- |
| **Passes over Memory** | 2 passes | 1 pass |
| **Parameters per Layer** | $2 \times d_{\text{model}}$ ($\gamma, \beta$) | $1 \times d_{\text{model}}$ ($\gamma$ only) |
| **FLOPs per Token** | $7d$ operations | $4d$ operations |
| **Kernel Latency** | $1.0\times$ (Baseline) | **$0.7\times$ (~30% faster)** |
| **Training Stability** | Excellent | Identical |

---

## 4. PyTorch Implementation

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # x * (1 / sqrt(mean(x^2) + eps))
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight
```

> **Why `torch.rsqrt`?**
> In GPU shader hardware, computing the reciprocal square root ($1/\sqrt{z}$) is an optimized single-cycle hardware instruction (`rsqrtss` in x86, `rsqrt` in CUDA/MPS).
