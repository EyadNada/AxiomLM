# Model FLOPs Utilization (MFU) & Hardware Roofline Guide

A comprehensive mathematical, systems, and engineering guide to **Model FLOPs Utilization (MFU)**, **Hardware Peak TFLOPs Estimation**, and **Roofline Efficiency Analysis** for Transformer Pretraining.

---

## 1. Motivation: Beyond "Tokens Per Second"

In large language model (LLM) engineering, measuring throughput solely in **tokens/second** ($\text{tok/s}$) is insufficient:
* Tokens/sec changes with sequence length $T$, hidden dimension $d_{\text{model}}$, parameter count $P$, batch size $B$, and hardware generation.
* Comparing $10,000\text{ tok/s}$ on an Apple M3 Max to $50,000\text{ tok/s}$ on an NVIDIA H100 provides zero insight into how efficiently each chip's silicon is being exploited.

**MFU (Model FLOPs Utilization)** is the industry-standard efficiency metric (used by DeepMind, OpenAI, Meta, and Anthropic). It quantifies the fraction of theoretical peak floating-point operations (FLOPs/sec) actively performing useful model forward/backward compute:

$$\text{MFU} = \frac{\text{Achieved FLOPs/sec}}{\text{Theoretical Peak Hardware FLOPs/sec}} \times 100\%$$

---

## 2. Mathematical Derivation of Transformer FLOPs

To calculate MFU, we must count the exact number of floating-point operations executed per token during one training step (Forward + Backward pass).

### A. Dense GEMM Operations ($6P$ Rule)
For every parameter $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ in a linear layer, multiplying an activation vector $x \in \mathbb{R}^{d_{\text{in}}}$ requires:
* $d_{\text{in}}$ multiplications
* $d_{\text{in}}$ additions
* **Total:** $2 \times d_{\text{in}} \times d_{\text{out}} = 2 \times N_{\text{params}}$ FLOPs.

During training:
1. **Forward Pass:** Computes activations $\mathbf{Y} = \mathbf{X}\mathbf{W}$ $\rightarrow 2P$ FLOPs/token.
2. **Backward Pass (Activation Gradient):** $\nabla_{\mathbf{X}} = \nabla_{\mathbf{Y}}\mathbf{W}^T$ $\rightarrow 2P$ FLOPs/token.
3. **Backward Pass (Weight Gradient):** $\nabla_{\mathbf{W}} = \mathbf{X}^T \nabla_{\mathbf{Y}}$ $\rightarrow 2P$ FLOPs/token.

$$\text{Total Dense Compute} = 2P + 2P + 2P = 6P \text{ FLOPs per token}$$

*(Where $P$ is the number of active non-embedding Transformer parameters).*

---

### B. Attention FLOPs (Context-Dependent Overhead)
Standard Causal Self-Attention contains additional operations that scale quadratically with context length $T$:
1. **Query-Key Matrix Multiply ($\mathbf{Q}\mathbf{K}^T$):**
   * Shape: $(B, N_h, T, d_k) \times (B, N_h, d_k, T) \rightarrow 2 \times B \times N_h \times T \times T \times d_k$
   * Since $N_h \times d_k = d_{\text{model}}$, FLOPs $= 2 B T^2 d_{\text{model}}$.
2. **Attention-Value Matrix Multiply ($\mathbf{A}\mathbf{V}$):**
   * Shape: $(B, N_h, T, T) \times (B, N_h, T, d_k) \rightarrow 2 B T^2 d_{\text{model}}$ FLOPs.
3. **Backward Pass of Attention:**
   * Computing $\nabla_{\mathbf{Q}}, \nabla_{\mathbf{K}}, \nabla_{\mathbf{V}}$ takes $2\times$ forward compute $\rightarrow 8 B T^2 d_{\text{model}}$ FLOPs.

$$\text{Attention FLOPs per Layer} = 12 B T^2 d_{\text{model}}$$
$$\text{Attention FLOPs per Token (across $L$ layers)} = 12 L \cdot d_{\text{model}} \cdot T$$

---

### C. Total FLOPs per Training Step
For a batch containing $\text{Tokens} = B \times T$ tokens:

$$\text{FLOPs}_{\text{step}} = \left( 6P + 12 L \cdot d_{\text{model}} \cdot T \right) \times \text{Tokens}$$

$$\text{Achieved FLOPs/sec} = \frac{\text{FLOPs}_{\text{step}}}{\Delta t_{\text{step}}}$$

---

## 3. Theoretical Peak Hardware TFLOPs Reference

| Hardware Accelerator | Peak BF16 / FP16 TFLOPs | Memory Bandwidth (GB/s) | Arithmetic Intensity Ceiling |
| :--- | :---: | :---: | :---: |
| **Apple M1 (Base)** | **~5.2 TFLOPs** | ~68 GB/s | 76.5 FLOPs/byte |
| **Apple M2 (Base)** | **~7.1 TFLOPs** | ~100 GB/s | 71.0 FLOPs/byte |
| **Apple M3 Max (40-core GPU)** | **~38.4 TFLOPs** | ~400 GB/s | 96.0 FLOPs/byte |
| **Apple M4 Max (40-core GPU)** | **~46.0 TFLOPs** | ~546 GB/s | 84.2 FLOPs/byte |
| **NVIDIA RTX 3090** | **~71.0 TFLOPs** | ~936 GB/s | 75.8 FLOPs/byte |
| **NVIDIA RTX 4090** | **~165.2 TFLOPs** | ~1,008 GB/s | 163.8 FLOPs/byte |
| **NVIDIA A100 (SXM4 80GB)** | **~312.0 TFLOPs** | ~2,039 GB/s | 153.0 FLOPs/byte |
| **NVIDIA H100 (SXM5 80GB)** | **~989.0 TFLOPs** | ~3,350 GB/s | 295.2 FLOPs/byte |

---

## 4. The Roofline Model: Memory-Bound vs Compute-Bound

The **Roofline Model** defines the operational boundary of any deep learning kernel on a given hardware accelerator:

```
Attained Performance (TFLOPs/s)
  ^
  |                  /------------------------- Peak Compute Roofline (Tensor Cores / AMX)
  |                 /
  |                /  <--- Memory Bandwidth Bound Region
  |               /
  |              /
  |             /
  +------------+-----------------------------> Arithmetic Intensity (FLOPs / Byte)
```

1. **Memory-Bandwidth Bound ($\text{FLOPs/Byte} < \text{Ceiling}$):**
   * The GPU ALUs are constantly waiting for weights and activations to transfer from High-Bandwidth Memory (HBM) to on-chip SRAM cache.
   * *Examples:* Un-fused LayerNorm, standard Softmax, AdamW element-wise parameter updates.
2. **Compute Bound ($\text{FLOPs/Byte} \ge \text{Ceiling}$):**
   * Memory transfers are saturated; matrix multiplication units (Tensor Cores / Metal SIMD groups) operate at maximum speed.
   * *Examples:* Large dense GEMMs ($QKV$ projections, SwiGLU MLP layers, FlashAttention-2 SRAM tiled loops).

---

## 5. Typical MFU Baselines in LLM Pretraining

| System Implementation | Typical MFU % | Bottlenecks / Characteristics |
| :--- | :---: | :--- |
| **Naive Eager PyTorch FP32** | **12% – 20%** | Full 32-bit floats, quadratic Softmax memory roundtrips, CPU-GPU sync stalls. |
| **BF16 Mixed Precision + SDPA** | **35% – 48%** | Tensor core acceleration, FlashAttention SRAM tiling. *(Axiom-LM target)* |
| **Custom Fused Kernels + DDP** | **48% – 58%** | Fused RMSNorm/SwiGLU, overlapped communication and computation (nanoGPT / Megatron). |
| **Theoretical Limit** | **~60% – 65%** | Irreducible non-GEMM overheads (token routing, optimizer updates, kernel launches). |

---

## 6. Python Implementation in Axiom-LM

```python
def estimate_hardware_peak_tflops(device: str) -> float:
    """Estimates theoretical peak BF16/FP16 TFLOPs for the active hardware."""
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0).lower()
        if "h100" in gpu_name:
            return 989.0
        elif "a100" in gpu_name:
            return 312.0
        elif "4090" in gpu_name:
            return 165.2
        elif "3090" in gpu_name:
            return 71.0
        return 70.0  # Default CUDA fallback
    elif device == "mps":
        # Apple Silicon MPS estimate (~10 TFLOPs base / 35 TFLOPs Pro/Max)
        return 10.0
    return 2.0  # CPU fallback


def calculate_mfu(
    model: torch.nn.Module,
    tokens_per_sec: float,
    seq_len: int,
    peak_tflops: float,
) -> float:
    """
    Computes MFU percentage: (6P + 12*L*d_model*T) * tokens_per_sec / Peak_FLOPs
    """
    cfg = model.config
    P = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flops_per_token = 6 * P + 12 * cfg.n_layer * cfg.n_embd * seq_len
    achieved_flops = flops_per_token * tokens_per_sec
    peak_flops = peak_tflops * 1e12
    return (achieved_flops / peak_flops) * 100.0
```
