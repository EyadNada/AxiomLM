# Rotary Position Embeddings (RoPE) Deep-Dive Guide

A comprehensive, intuitive, and mathematical breakdown of **Rotary Position Embeddings (RoPE)** (Su et al., 2021) as implemented in LLaMA, Mistral, Gemma, and Axiom-LM.

---

## 1. Why Did GPT-2 Need an Upgrade?

In classic GPT-2:
$$\mathbf{h}_i = \text{TokenEmbedding}(x_i) + \text{PositionEmbedding}(i)$$

Where $\text{PositionEmbedding}$ is a lookup table matrix $W_{pe} \in \mathbb{R}^{T_{\text{max}} \times d_{\text{model}}}$.

### The Three Fatal Flaws of Absolute Positional Embeddings:
1. **Hard Length Wall**: If trained with $T=1024$, the model has **no embedding vector** for position $1025$. It cannot generate beyond its fixed training window.
2. **Ignores Relative Distance**: Adding an arbitrary vector at pos $i$ does not mathematically ensure that the attention dot product $Q_i K_j^T$ depends cleanly on $(i - j)$.
3. **Wasted Parameters**: Consumes thousands of learned weights that must be stored and trained.

---

## 2. The Core RoPE Intuition: Rotation in the Complex Plane

Instead of **adding** a vector, RoPE **rotates** the Query and Key vectors in 2D pairs.

Consider a 2D vector $\mathbf{x} = (x_1, x_2) = r e^{i \phi}$.

At position $m$, rotate it by angle $m\theta$:
$$\mathbf{x}'_m = \mathbf{x} \cdot e^{i m \theta} = r e^{i (\phi + m\theta)}$$

Now, compute the inner product (attention score) between Query at position $m$ and Key at position $n$:
$$\langle \mathbf{q}_m, \mathbf{k}_n \rangle = \text{Re} \left( \mathbf{q}'_m \cdot (\mathbf{k}'_n)^* \right) = \text{Re} \left( (\mathbf{q} e^{i m \theta}) \cdot (\mathbf{k} e^{-i n \theta}) \right) = \text{Re} \left( \mathbf{q} \mathbf{k}^* e^{i (m - n) \theta} \right)$$

Look at the exponent: **the position indices $m$ and $n$ completely vanish, leaving only their relative distance $(m - n)$!**

```
Position m=0:   ───► (Angle 0)
Position m=1:   ───/ (Angle θ)
Position m=2:   ───| (Angle 2θ)
Position m=3:   ───\ (Angle 3θ)

Distance between pos 3 and pos 1:  3θ - 1θ = 2θ  (Invariant to absolute shift!)
```

---

## 3. High-Dimensional RoPE Formulation

In a model with head dimension $d_k = 64$, we divide the 64 numbers into 32 pairs of 2D coordinates $(x_{2i}, x_{2i+1})$.

Each pair rotates at a different frequency:
$$\theta_i = 10000^{-2i/d_k}, \quad i \in [0, 1, \dots, d_k/2 - 1]$$

* **Low channels ($i=0$)**: Rotate rapidly (captures fine-grained local word order like "the cat" vs "cat the").
* **High channels ($i=31$)**: Rotate very slowly (captures macro-level paragraph context and structure).

$$\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$

---

## 4. PyTorch Implementation in Axiom-LM

```python
import torch

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # dim must be head dimension (e.g. 64)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    # Complex exponential: e^(i * m * theta)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    # Reshape into complex numbers (pairs of real coordinates)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Rotate by complex multiplication
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

---

## 5. Summary of Advantages

1. **True Relative Attention**: The model naturally learns distance penalties.
2. **Context Length Extrapolation**: Can extend context window dynamically with techniques like YaRN or RoPE scaling.
3. **No Learned Parameters**: Reduces model checkpoint size.
