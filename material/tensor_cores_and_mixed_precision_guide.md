# Mixed Precision Training & NVIDIA Tensor Cores Guide

This document serves as a comprehensive reference and paper breakdown of the foundational research behind **Tensor Cores** and **Mixed Precision Training**, detailing the mathematics, hardware architecture, number formats, and practical PyTorch implementation for training models like **GPT-2 (124M)**.

---

## 1. Paper Overview & Reference

* **Title:** *Mixed Precision Training*
* **Authors:** Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu (NVIDIA & Baidu)
* **Venue:** ICLR 2018 / [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)
* **Core Contribution:** Introduced a robust methodology to train deep neural networks using half-precision floating-point numbers without losing model accuracy or requiring hyperparameter modifications.

---

## 2. What are Tensor Cores?

### Standard CUDA Cores vs. Tensor Cores

* **CUDA Core (ALU):** Performs **1 operation per clock cycle** on scalar values (e.g., $1 \times 1$ multiply or add). A matrix multiplication $Y = A \cdot B$ is broken down into millions of individual scalar operations across CUDA threads.
* **Tensor Core (MMA Unit):** Performs a specialized hardware matrix multiply-and-accumulate operation in a single warp clock cycle:

$$\mathbf{D} = \mathbf{A} \times \mathbf{B} + \mathbf{C}$$

Where $\mathbf{A}$ and $\mathbf{B}$ are $4 \times 4$ (or $16 \times 16$) matrix tiles in half precision (FP16/BF16), and $\mathbf{C}$ and $\mathbf{D}$ are accumulator matrices in FP16 or full-precision FP32.

```
       Matrix A (FP16/BF16)         Matrix B (FP16/BF16)
          [ 4 x 4 Tile ]       x       [ 4 x 4 Tile ]
                                +
                     Matrix C (FP32 Accumulator)
                                =
                     Matrix D (FP32 Result)
             ==> Executed in hardware concurrently per cycle
```

### Generational Architecture Milestones

| Architecture | Representative GPU | Key Tensor Core Innovations |
|:---|:---|:---|
| **Volta** (2017) | V100 | 1st Gen: FP16 input, FP32 accumulate ($4\times4\times4$ operations/cycle). |
| **Turing** (2018) | T4, RTX 2080 | 2nd Gen: Added INT8/INT4 precision for inference speedup. |
| **Ampere** (2020) | A100, RTX 3090 | 3rd Gen: **TF32** (TensorFloat-32), **BF16**, FP64, 2:4 Structured Sparsity. |
| **Ada / Hopper** (2022) | H100, RTX 4090 | 4th Gen: **FP8** (E4M3, E5M2), Transformer Engine, DPX instructions. |
| **Blackwell** (2024) | B200 | 5th Gen: **FP4**, 2nd Gen Transformer Engine, microscopic scaling factors. |

---

## 3. Number Formats Breakdown

Understanding the bit allocations (Sign, Exponent, Mantissa) explains why each format behaves differently during training:

```
FP32:   [S (1)] [   Exponent (8)   ] [       Mantissa / Fraction (23)       ]
FP16:   [S (1)] [  Exp (5)  ] [   Mantissa (10)   ]
BF16:   [S (1)] [   Exponent (8)   ] [ Mantissa (7) ]
TF32:   [S (1)] [   Exponent (8)   ] [   Mantissa (10)   ]
FP8(E4):[S (1)] [ Exp (4) ] [ Mant (3) ]
FP8(E5):[S (1)] [  Exp (5)  ] [ Mant (2) ]
```

### Detailed Format Comparison

| Format | Total Bits | Exponent (Range) | Mantissa (Precision) | Min Positive Value | Max Value | Primary Use Case |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **FP32** (Single) | 32 | 8 bits | 23 bits | $\approx 1.18 \times 10^{-38}$ | $\approx 3.40 \times 10^{38}$ | Baseline, Master weights, Reductions |
| **FP16** (Half) | 16 | 5 bits | 10 bits | $\approx 6.10 \times 10^{-5}$ ($2^{-14}$) | $65,504$ | Fast math (Requires Loss Scaling) |
| **BF16** (Bfloat16) | 16 | 8 bits | 7 bits | $\approx 1.18 \times 10^{-38}$ | $\approx 3.40 \times 10^{38}$ | **Modern default**: Same range as FP32, no loss scaling needed |
| **TF32** (TensorFloat) | 19 | 8 bits | 10 bits | $\approx 1.18 \times 10^{-38}$ | $\approx 3.40 \times 10^{38}$ | Drop-in acceleration on Ampere+ without changing code |
| **FP8 (E4M3)** | 8 | 4 bits | 3 bits | $\approx 0.00195$ | $448$ | Forward pass activations & weights |
| **FP8 (E5M2)** | 8 | 5 bits | 2 bits | $\approx 6.10 \times 10^{-5}$ | $57,344$ | Backward pass gradients |

---

## 4. The 3 Pillars of Mixed Precision Training (Micikevicius et al.)

When naively training networks in FP16, models fail to converge due to precision limitations. The paper introduced **3 essential techniques**:

### Pillar 1: FP32 Master Copy of Weights

* **The Problem:** Weight updates $\theta \leftarrow \theta - \gamma \cdot g$ often add very small numbers. If $\gamma \cdot g < 2^{-11} \cdot \theta$, in FP16 addition the delta underflows to zero, meaning weights never change.
* **The Solution:**
  1. Maintain a master copy of weights in **FP32**.
  2. For Forward Pass: Cast weights to **FP16**.
  3. For Backward Pass: Compute activations & gradients in **FP16**.
  4. In Optimizer: Update the master **FP32** weights using FP32 arithmetic.

```
+-------------------------------------------------------+
|                 Forward Pass (FP16)                   |
|   Inputs (FP16) x Weights (FP16) -> Activations(FP16) |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
|                 Backward Pass (FP16)                  |
|     Gradients w.r.t Activations & Weights (FP16)      |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
|               Optimizer Step (FP32)                   |
| Master FP32 Weights <- Update(Master FP32, Grad FP32) |
+-------------------------------------------------------+
                           |
                           v
                 Cast back to FP16
```

---

### Pillar 2: Loss Scaling (Essential for FP16)

* **The Problem:** In deep neural networks, more than $50\%$ of gradient values have magnitudes smaller than $2^{-14} \approx 6.1 \times 10^{-5}$ (the smallest normalized number representable in FP16). They underflow to zero, starving early layers of gradient signal.
* **The Solution:** Scale the loss by a constant factor $S$ (e.g., $S = 1024$ or dynamic scaling):
  $$\text{Loss}_{\text{scaled}} = \text{Loss} \times S$$
* By the chain rule, all gradients $\nabla_\theta \text{Loss}_{\text{scaled}} = S \cdot \nabla_\theta \text{Loss}$ are shifted up into the representable FP16 range.
* Before the optimizer update, unscale the gradients:
  $$g = \frac{g_{\text{scaled}}}{S}$$

```
                Gradient Distribution Before vs. After Scaling
  Count ^
        |            [ Underflow Zone ]
        |   <--- Below 2^-14 (Lost) --->|<----- FP16 Range ----->
        |
        |          ░░░░░░               |
        |        ░░░░░░░░░░░            |
        |      ░░░░░░░░░░░░░░           |
        |    ░░░░░░░░░░░░░░░░░░         |  Original Gradients (Unscaled)
        +-------------------------------+---------------------------> Exponent
        |                               |
        |      Scaled Gradients by S    |       ░░░░░░
        |         (Shifted Right) =====>|     ░░░░░░░░░░░
        |                               |   ░░░░░░░░░░░░░░
        |                               | ░░░░░░░░░░░░░░░░░░
        +-------------------------------+---------------------------> Exponent
                                      2^-14
```

---

### Pillar 3: FP32 Accumulation & Arithmetic

* When performing dot products $\sum_{i} a_i b_i$ in Tensor Cores, multiplications happen in half precision, but the summation is accumulated into **FP32 registers** before writing back to memory.
* Sensitive reductions (e.g., Softmax, LayerNorm, BatchNorm) are computed in FP32 to prevent overflow when summing exponents.

---

## 5. Practical Implementation in PyTorch for GPT-2 (124M)

### 1. Enabling TensorFloat-32 (TF32) for Ampere+ GPUs
TF32 runs standard `torch.float32` matrix multiplications on Tensor Cores at $\sim 8\times$ the speed of standard FP32 with zero code changes:

```python
import torch

# Allows PyTorch to use TF32 internally for matmuls and convolutions
torch.set_float32_matmul_precision('high')  # Options: 'highest' (FP32), 'high' (TF32), 'medium' (bfloat16)
```

---

### 2. Automatic Mixed Precision with Bfloat16 (Recommended for Modern GPUs)
Since **BF16** shares the same 8-bit exponent as FP32, **Loss Scaling is not required**, making training completely stable:

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT(GPTConfig()).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for step in range(num_steps):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    
    optimizer.zero_grad()
    
    # Run forward pass in bfloat16 mixed precision
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    
    # Backward pass computes gradients
    loss.backward()
    
    optimizer.step()
```

---

### 3. Automatic Mixed Precision with FP16 & `GradScaler` (For Older GPUs)
For hardware without native BF16 support (e.g., Turing / Volta GPUs like V100, T4, RTX 2080):

```python
scaler = torch.cuda.amp.GradScaler()

for step in range(num_steps):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    
    optimizer.zero_grad()
    
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits, loss = model(x, y)
    
    # Scale loss and backpropagate
    scaler.scale(loss).backward()
    
    # Unscale gradients and step optimizer (skips step if Inf/NaN detected)
    scaler.step(optimizer)
    
    # Update scale factor for next iteration
    scaler.update()
```

---

## 6. Tensor Core Dimension & Alignment Rules (The Rule of 8 / 16 / 64)

For Tensor Cores to achieve maximum hardware throughput, matrix dimensions ($M, N, K$) must be aligned to multiples of the warp tile size:

* **FP16 / BF16:** Dimensions should be multiples of **8** (ideally multiples of **64** or **128** for cache efficiency).
* **TF32:** Dimensions should be multiples of **4** (ideally **16**).
* **FP8:** Dimensions should be multiples of **16** (ideally **32** or **64**).

### GPT-2 Vocabulary Padding Example
In original GPT-2, the vocabulary size is **`50,257`** (a prime number, not divisible by 8 or 64).
* $50,257 \div 64 = 785.265625$ (Causes memory misalignments and prevents full Tensor Core warp utilization).
* In modern implementations (like NanoGPT / Karpathy), `vocab_size` is padded up to the nearest multiple of 64: **`50,304`** ($50,304 = 64 \times 786$), yielding an immediate **$\sim 5-10\%$ speedup** on the final linear projection layer `lm_head`.

---

## 7. Summary Cheat Sheet

| Technique | When to Use | PyTorch Code | Speedup Factor |
|:---|:---|:---|:---:|
| **TF32** | Ampere/Ada/Hopper GPUs | `torch.set_float32_matmul_precision('high')` | $\approx 2\times - 3\times$ |
| **AMP (BF16)** | Modern GPUs (A100, H100, RTX 30/40) | `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` | $\approx 3\times - 5\times$ |
| **AMP (FP16)** | Legacy GPUs (V100, T4) | `torch.autocast(...)` + `torch.cuda.amp.GradScaler()` | $\approx 3\times - 4\times$ |
| **Vocab Padding** | Any Transformer model | Pad `vocab_size` to multiple of $64$ (e.g. $50304$) | $\approx 1.05\times - 1.1\times$ |
