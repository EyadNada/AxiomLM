# The Muon Matrix Optimizer Guide

A mathematical and conceptual guide to **Muon (Momentum Orthogonalized by Newton-Schulz)**, the next-generation matrix optimizer developed by Keller Jordan, Jeremy Bernstein, and the nanogpt/modded-nanogpt research collective.

---

## 1. Why Do We Need a New Optimizer Beyond AdamW?

AdamW has been the king of deep learning optimizers for nearly a decade. However, AdamW has a fundamental structural limitation:

* **Scalar-Wise Updates**: AdamW updates every parameter scalar $w_{ij}$ independently based on its historical variance $\sqrt{v_{ij}}$.
* **Ignores Matrix Geometry**: Neural networks are composed of 2D linear transformations $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$. In matrix space, updating entries independently can lead to ill-conditioned transformations, collapsed singular values, and slow convergence.

---

## 2. The Core Insight: Orthogonal Updates

An **orthogonal matrix** $O$ preserves vector norms and angles ($\|O x\| = \|x\|$).

When updating a weight matrix $W$:
$$W_{t+1} = W_t - \eta \cdot U_t$$

If the update step $U_t$ is **orthogonalized** (all its singular values are equal to 1), every direction in activation space receives an equal, balanced gradient signal. No singular dimension is over-emphasized or squashed.

---

## 3. How to Orthogonalize Fast: The Newton-Schulz Iteration

Computing exact SVD ($U \Sigma V^T$) on a GPU is extremely slow and memory-intensive ($O(d^3)$ with large constant factor).

**The Solution:** The **Newton-Schulz Iteration** is a purely iterative algebraic algorithm that uses only **fast matrix multiplications (`matmul`)** to drive any matrix toward its nearest orthogonal polar factor!

### The Algorithm:
1. Accumulate standard heavy-ball momentum:
   $$M_t = \beta M_{t-1} + G_t$$
2. Scale $M_t$ by its Frobenius norm:
   $$X_0 = \frac{M_t}{\|M_t\|_F}$$
3. Perform 5 iterations of the cubic polynomial:
   $$X_{k+1} = a X_k + b (X_k X_k^T) X_k + c (X_k X_k^T)^2 X_k$$
   *(or standard Schulz iteration: $X_{k+1} = \frac{1}{2} X_k (3 I - X_k^T X_k)$)*
4. Apply the orthogonalized update $X_5$ to the weight matrix.

```
Gradient G ──► Momentum M ──► Scale X0 ──► [ Matmul Triplet Iteration ] (x5) ──► Orthogonal Update U ──► Weight W
```

---

## 4. PyTorch Reference Implementation

```python
import torch

@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # normalize Frobenius norm
    if G.size(0) > G.size(1):
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X.type_as(G)
```

---

## 5. Performance Gains

* **Training Speedup**: Reaches the same validation loss in **~35% to 50% fewer steps** compared to standard AdamW.
* **Compatibility**: Muon is applied to 2D internal weight matrices (MLP projections, attention matrices), while 1D parameters (biases, RMSNorm scales, embeddings) use standard AdamW.
