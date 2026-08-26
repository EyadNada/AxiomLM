# Newton-Schulz Iteration: Spectral Analysis and Matrix Orthogonalization

A rigorous mathematical and hardware-level analysis of the **Newton-Schulz iteration** used for polar decomposition and gradient orthogonalization in next-generation matrix optimizers like **Muon**.

---

## 1. The Mathematical Objective: Polar Decomposition

Given a real gradient or momentum matrix $G \in \mathbb{R}^{m \times n}$ (assume without loss of generality $m \le n$), the **polar decomposition** factors $G$ into:
$$G = U \cdot H$$

where:
* $U \in \mathbb{R}^{m \times n}$ is a semi-orthogonal matrix satisfying $U U^T = I_m$.
* $H = (G G^T)^{1/2} \in \mathbb{R}^{m \times m}$ is a symmetric positive semi-definite matrix.

The matrix $U$ represents the **zeroth matrix power** of $G$:
$$U = G (G^T G)^{-1/2} = (G G^T)^{-1/2} G$$

Geometrically, $U$ is the **unique closest orthogonal matrix to $G$** under the Frobenius norm:
$$U = \arg\min_{Q Q^T = I} \|G - Q\|_F$$

---

## 2. Why Iterative Approximation Beats Exact SVD on Accelerators

| Method | Computational Complexity | FLOP Pattern | Hardware Utilization (Tensor Cores / AMX) | Memory Overhead |
| :--- | :---: | :---: | :---: | :---: |
| **Exact SVD ($U \Sigma V^T$)** | $O(m^2 n)$ (Large constant) | Bidiagonalization + QR Sweeps | < 15% (Memory-bound branching) | High (Intermediate Householder buffers) |
| **5-Step Newton-Schulz** | $15 m^2 n$ FLOPs | Pure GEMM Matmul Triplets | **> 85% (Compute-bound systolic tiling)** | **Minimal ($O(m^2)$ scratch space)** |

Accelerators (NVIDIA GPUs, Apple Silicon M-series) are designed specifically for dense General Matrix Multiplications (`GEMM`). By turning orthogonalization into **a fixed sequence of matrix multiplications**, Newton-Schulz runs orders of magnitude faster in wall-clock time than exact SVD.

---

## 3. Polynomial Derivation: From 3rd to 5th Order

The classic Newton-Schulz iteration computes the inverse square root of a matrix:
$$X_{k+1} = \frac{1}{2} X_k (3 I - X_k^T X_k)$$

This corresponds to the cubic polynomial $p(x) = \frac{3}{2}x - \frac{1}{2}x^3$. While effective, it requires 10–15 iterations to converge for ill-conditioned matrices.

### The Quintic (5th-Order) Minimax Formulation
To minimize iterations to just **5 steps**, Keller Jordan and Jeremy Bernstein derived the optimal 5th-order odd polynomial:
$$p(x) = a x + b x^3 + c x^5$$

subject to the constraints:
1. $p(0) = 0$ (preserves null spaces).
2. $p(1) = 1$ (fixed point at unit singular values: $a + b + c = 1 \cdot \text{scale}$).
3. $p'(1) = 0$ (superlinear local convergence).
4. $\max_{x \in (0, 1]} |p(x) - 1|$ is minimized.

Solving this Chebyshev-equioscillation problem yields:
$$a = 3.4445, \quad b = -4.7750, \quad c = 2.0315$$

### Singular Value Evolution Across 5 Iterations

Let singular values $\sigma_0 \in (0, 1]$:
* **Step 0**: Normalized input $\sigma_0 \in (0.01, 1.00)$
* **Step 1**: $\sigma_1 = p(\sigma_0) \in (0.034, 1.05)$
* **Step 2**: $\sigma_2 = p(\sigma_1) \in (0.11, 1.02)$
* **Step 3**: $\sigma_3 = p(\sigma_2) \in (0.37, 1.005)$
* **Step 4**: $\sigma_4 = p(\sigma_3) \in (0.89, 1.001)$
* **Step 5**: $\sigma_5 = p(\sigma_4) \in [0.9995, 1.0005]$

In just 5 algebraic steps, every singular value across all non-zero dimensions is compressed tightly to $1.00 \pm 0.0005$.

---

## 4. Matrix Evaluation via Nested Matmuls

To evaluate $a X + b (X X^T) X + c (X X^T)^2 X$ using minimal memory and operations:

```
Let A = X @ X.T          # Shape: (m, m) - 1 matmul
Let B = b * A + c * A @ A # Shape: (m, m) - 1 matmul
Let X_next = a * X + B @ X # Shape: (m, n) - 1 matmul
```

* **Total Matmuls per Step**: 3 matrix multiplies.
* **Total Matmuls across 5 Steps**: 15 fast matrix multiplications.
* For a $768 \times 768$ weight matrix in BF16, all 5 steps complete in **< 0.15 milliseconds** on Apple Silicon M-series GPU.

---

## 5. Numerical Stability and Precision Handling

1. **Frobenius Normalization**:
   $$X_0 = \frac{G}{\|G\|_F + \epsilon}$$
   Guarantees that the maximum singular value $\sigma_{\max}(X_0) \le 1.0$, preventing polynomial explosion.
2. **Dimension Transposition**:
   If $m > n$, we transpose $X = G^T$ so that intermediate square matrix $A = X X^T$ has dimension $n \times n$ (always operating on the smaller dimension), minimizing FLOPs.
3. **Precision**:
   Iterations execute natively in `bfloat16` or `float32`. Even in BF16, the contractive nature of the polynomial $p(x)$ self-corrects rounding errors toward the fixed point $\sigma = 1.0$.

---

## 6. Summary: Why Muon Outperforms Scalar Optimizers

1. **Isotropic Energy Distribution**: By normalizing all singular values to 1, gradient signal is distributed evenly across all eigen-directions.
2. **Eliminates Directional Stalling**: Prevents ill-conditioned layers where ill-aligned gradients cause standard AdamW to oscillate.
3. **Hardware Alignment**: Transforms complex linear algebra (SVD) into pure systolic matrix multiplication.
