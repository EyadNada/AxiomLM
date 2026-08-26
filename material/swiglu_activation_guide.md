# SwiGLU Activation & Gated FFN Guide

A deep dive into **SwiGLU (Swish Gated Linear Unit)** feed-forward networks (Shazeer, 2020), their mathematical formulation, dimensional tuning, and superiority over standard GELU/ReLU MLPs.

---

## 1. Evolution of the Feed-Forward Block (FFN)

In autoregressive Transformers, the MLP/FFN block accounts for **approximately 66% of the parameter count per layer** (excluding embeddings).

### The Evolution:
1. **ReLU FFN (Original 2017 Transformer):**
   $$\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$
2. **GELU FFN (GPT-2 / GPT-3):**
   $$\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$$
3. **SwiGLU FFN (PaLM, LLaMA-1/2/3, Mistral, Axiom-LM):**
   $$\text{FFN}_{\text{SwiGLU}}(x) = \Big( \text{Swish}(x W_{\text{gate}}) \odot (x W_{\text{up}}) \Big) W_{\text{down}}$$

```
GPT-2 (GELU MLP):
Input x ──► Linear (W1: d -> 4d) ──► GELU ──────────────────────────► Linear (W2: 4d -> d) ──► Output

Modern SwiGLU:
Input x ──┬──► Linear (W_gate: d -> hidden) ──► SiLU (Swish) ──┐
          │                                                    ▼
          └──► Linear (W_up:   d -> hidden) ───────────────► Multiply ──► Linear (W_down: hidden -> d) ──► Output
```

---

## 2. Why Gated Linear Units (GLU) Perform Dramatically Better

In a standard MLP, non-linearity is applied independently across each coordinate.
In a Gated Linear Unit:
1. **$x W_{\text{up}}$** represents the raw features extracted from the input.
2. **$\text{SiLU}(x W_{\text{gate}})$** represents a continuous gating valve between $(0, \infty)$ that dynamically attenuates or amplifies specific features.

The element-wise product $\odot$ produces **second-order (bilinear) interactions** between features. This gives the network higher representational capacity per parameter.

---

## 3. Parameter Parity & Dimension Scaling ($\frac{8}{3} d_{\text{model}}$)

A standard GPT-2 MLP has 2 matrices of shape $(d \times 4d)$ and $(4d \times d)$:
$$\text{Params}_{\text{GPT2}} = 2 \times (d \times 4d) = 8 d^2$$

SwiGLU has 3 matrices ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) of shapes $(d \times d_h)$, $(d \times d_h)$, $(d_h \times d)$:
$$\text{Params}_{\text{SwiGLU}} = 3 \times (d \times d_h) = 3 d \cdot d_h$$

To match the exact parameter count ($3 d \cdot d_h = 8 d^2$):
$$d_h = \frac{8}{3} d_{\text{model}} \approx 2.667 \times d_{\text{model}}$$

For $d_{\text{model}} = 768$:
$$d_h = \frac{8}{3} \times 768 = 2048$$

---

## 4. PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 2048):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.silu is x * sigmoid(x)
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)
```

---

## 5. Empirical Results in Literature
* In Noam Shazeer's seminal paper *"GLU Variants Improve Transformer"*, SwiGLU consistently achieved **0.5 to 1.5 lower perplexity points** across every model scale compared to standard GELU or ReLU networks at equal compute budgets.
