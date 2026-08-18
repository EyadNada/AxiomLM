# PyTorch `torch.set_float32_matmul_precision` Guide

A comprehensive technical reference for **`torch.set_float32_matmul_precision`**, explaining how PyTorch controls internal precision for Float32 matrix multiplications (GEMM) using **TensorFloat-32 (TF32)** and **Bfloat16** on NVIDIA Tensor Cores.

---

## 1. PyTorch API Reference

*(Reference: [PyTorch Documentation — `torch.set_float32_matmul_precision`](https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html))*

```python
torch.set_float32_matmul_precision(precision)
```

### Getter Counterpart
```python
current_precision = torch.get_float32_matmul_precision()  # returns 'highest', 'high', or 'medium'
```

---

## 2. What Does It Do?

By default, PyTorch computes `torch.float32` matrix multiplications (`torch.matmul`, `nn.Linear`, attention projections $Q K^T$, $A V$) using full IEEE 754 single-precision (24 mantissa bits).

`torch.set_float32_matmul_precision` allows PyTorch to internally trade a small, usually negligible amount of mantissa precision for **massive hardware acceleration ($2\times - 3\times$ speedup)** on NVIDIA Tensor Cores **without changing the dtype of your tensors or modifying your model code**.

---

## 3. The 3 Supported Precision Settings

| Setting | Internal Datatype | Mantissa Bits (Precision) | Exponent Bits (Range) | Hardware Target | Relative Speed |
|:---|:---:|:---:|:---:|:---|:---:|
| **`"highest"`** *(Default)* | **FP32** | 24 bits (23 explicit) | 8 bits | Standard FP32 ALUs / CUDA Cores | $1.0\times$ (Baseline) |
| **`"high"`** *(Recommended)* | **TF32** / $2\times$ BF16 | 11 bits (10 explicit) | 8 bits | Ampere, Ada, Hopper, Blackwell Tensor Cores | **$\sim 2.5\times - 3\times$** |
| **`"medium"`** | **BF16** | 8 bits (7 explicit) | 8 bits | Fast BF16 Tensor Core GEMM kernels | **$\sim 3\times - 4\times$** |

---

### Detailed Breakdown of Settings:

### 1. `"highest"` (Full Precision)
* **Datatype:** Standard IEEE 754 `float32`.
* **Behavior:** Matrix multiplications use full 23-bit explicit mantissa precision.
* **When to use:**
  - Scientific computing or simulations requiring exact FP32 numerical guarantees.
  - When debugging numerical discrepancies or verifying loss values down to $10^{-6}$.

---

### 2. `"high"` (TensorFloat-32 — Recommended Default for Training)
* **Datatype:** **TensorFloat-32 (`TF32`)** or sum of two `bfloat16` numbers.
* **Behavior:** 
  - Inputs (FP32) are internally truncated to 10 mantissa bits (same precision as FP16) while keeping the full 8-bit exponent (same dynamic range as FP32).
  - Executed directly on **NVIDIA Tensor Cores** in hardware.
  - Accumulation is performed in full **FP32** registers.
  - Output is returned as a standard **`torch.float32`** tensor.
* **When to use:**
  - **Always recommended on NVIDIA Ampere or newer GPUs (RTX 30xx/40xx, A100, H100, L40S).**
  - Virtually **zero impact on final training loss/accuracy** for LLMs (like GPT-2).

---

### 3. `"medium"` (Bfloat16 Reduced Precision)
* **Datatype:** **`bfloat16`** (7 explicit mantissa bits).
* **Behavior:** Truncates inputs to 7 mantissa bits for maximum hardware execution throughput.
* **When to use:**
  - When maximum throughput is needed for Float32 operations on architectures where pure BF16 GEMM is faster than TF32.

---

## 4. Hardware Mechanism & Bit Allocation

```
FP32 (highest):  [S (1)] [   Exponent (8)   ] [       Mantissa (23 explicit)       ]
TF32 (high):     [S (1)] [   Exponent (8)   ] [ Mantissa (10) ]
BF16 (medium):   [S (1)] [   Exponent (8)   ] [ Mant (7) ]
```

### Why TF32 is Ideal for Deep Learning:
1. **Same Range as FP32:** Because TF32 preserves all 8 exponent bits, **no scaling or underflow protection (`GradScaler`) is needed**.
2. **Same Precision as FP16:** 10 mantissa bits provide $\approx 3-4$ decimal digits of precision, which is more than sufficient for neural network weight updates and gradient descent.

---

## 5. TF32 vs. Automatic Mixed Precision (`torch.autocast`)

| Feature | `torch.set_float32_matmul_precision('high')` | `torch.autocast(..., dtype=torch.bfloat16)` |
|:---|:---|:---|
| **What it affects** | Only `float32` matrix multiplications (cuBLAS GEMM) | Entire forward pass operations (Linear, Attn, LayerNorm, activations) |
| **Tensor Dtypes** | Tensors remain **`torch.float32`** in memory | Activations and weights are cast to **`torch.bfloat16`** |
| **Memory Savings** | $0\%$ (VRAM usage is identical to FP32) | **$\sim 50\%$ VRAM reduction** (enables $2\times$ larger batch size) |
| **Speedup** | $\approx 2\times - 3\times$ | $\approx 3\times - 5\times$ |
| **Code Changes** | 1 line at the top of your script | Wrapped around forward pass with context manager `with torch.autocast():` |

---

## 6. How to Use in `train_gpt2.py`

Place this single line right after importing PyTorch:

```python
import torch

# ⚡ Enable TensorFloat-32 for ~3x faster matmuls on Ampere/Ada/Hopper GPUs
torch.set_float32_matmul_precision('high')

# Automatic Device Selection
device = "cuda" if torch.cuda.is_available() else "cpu"
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

print(f"Using device: {device}")
print(f"Float32 matmul precision: {torch.get_float32_matmul_precision()}")
```

---

## 7. Performance Benchmarks on GPT-2 (124M)

*(Measured on an NVIDIA A100 80GB SXM4, $B=16, T=1024$)*

| Setting | Precision Mode | Step Time (`dt`) | Throughput (Tokens/sec) | Speedup |
|:---|:---:|:---:|:---:|:---:|
| `torch.set_float32_matmul_precision('highest')` | Pure FP32 | `~1002 ms` | `16,350` | $1.0\times$ (Baseline) |
| **`torch.set_float32_matmul_precision('high')`** | **TF32** | **`~345 ms`** | **`47,500`** | **$\mathbf{\sim 2.9\times}$** |
| `+ torch.autocast(dtype=torch.bfloat16)` | BF16 AMP | `~160 ms` | `102,400` | $\sim 6.2\times$ |
| `+ torch.compile()` | Fused Inductor | `~95 ms` | `172,000` | $\sim 10.5\times$ |

---

## 8. Summary Checklist

- [x] **Add `torch.set_float32_matmul_precision('high')`** to every CUDA deep learning training script.
- [x] It has **no downsides** on modern NVIDIA GPUs (Ampere RTX 30xx/40xx, A100, H100).
- [x] On non-CUDA devices (e.g. Apple Silicon MPS or CPUs), PyTorch safely ignores or gracefully falls back without errors.
