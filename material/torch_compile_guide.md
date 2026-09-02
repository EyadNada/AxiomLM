# Introduction to `torch.compile` — Comprehensive Guide

A technical reference and deep-dive for **`torch.compile`**, based on the official PyTorch 2.0+ tutorial by William Wen and Andrej Karpathy's *"Let's reproduce GPT-2 (124M)"* training optimizations.

---

## 1. PyTorch API Reference

*(Reference: [PyTorch Tutorials — Introduction to `torch.compile`](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) & [PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.compile.html))*

```python
import torch

# Basic 1-line usage:
compiled_model = torch.compile(model)
```

### Full Signature & Parameters

```python
torch.compile(
    model=None,
    *,
    fullgraph=False,
    dynamic=None,
    backend="inductor",
    mode=None,
    options=None,
    disable=False
)
```

| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| **`model`** | `nn.Module` or `callable` | *required* | The PyTorch model, layer, or Python function to compile. |
| **`mode`** | `str` | `None` / `"default"` | Optimization preset: `"default"`, `"reduce-overhead"`, or `"max-autotune"`. |
| **`dynamic`** | `bool` or `None` | `None` | If `True`, enables dynamic shape support (avoids recompilation when batch size or sequence length varies). |
| **`fullgraph`** | `bool` | `False` | If `True`, requires the entire model to be captured in a single graph without graph breaks (raises error if Python code cannot be traced). |
| **`backend`** | `str` or `callable` | `"inductor"` | The compiler backend. PyTorch's primary production backend is **TorchInductor** (compiles to **OpenAI Triton** for GPUs and C++/OpenMP for CPUs). |
| **`disable`** | `bool` | `False` | Turn off compilation for easy debugging without changing model references. |

---

## 2. Why `torch.compile`? (Eager Mode vs. Graph Mode)

### Standard PyTorch: Eager Mode
PyTorch's popularity comes from **eager execution** (Python-first, imperative execution, dynamic debuggability). However, eager execution introduces two major performance bottlenecks:

1. **Python Overhead & Kernel Launch Overhead:**
   Every PyTorch operation (`+`, `*`, `torch.matmul`, `F.gelu`) makes a Python-to-C++ call and launches an individual CUDA kernel on the GPU. When operations are fast, CPU launch overhead dominates GPU compute time.

2. **Memory Bandwidth Bottleneck (Memory Wall):**
   Modern GPUs (A100, H100, RTX 4090) perform math calculations far faster than they can read/write data from Global GPU Memory (VRAM / HBM).
   In eager mode, intermediate activations are written to VRAM and read back for every single operation:
   $$\text{LayerNorm} \xrightarrow{\text{write VRAM}} \text{VRAM} \xrightarrow{\text{read VRAM}} \text{QKV Projection} \xrightarrow{\text{write VRAM}} \dots$$

### The Solution: JIT Compilation with `torch.compile`
`torch.compile` intercepts your model's operations, creates an optimized computational graph, and **fuses multiple operations into a single GPU kernel**. Data stays in fast on-chip **SRAM (Cache)** instead of constantly round-tripping to VRAM.

```
Eager Mode:
[Read VRAM] -> [Bias Add] -> [Write VRAM] -> [Read VRAM] -> [GELU] -> [Write VRAM]

Compiled (Fused Kernel):
[Read VRAM once] -> [Bias Add + GELU in On-Chip SRAM] -> [Write VRAM once]
```

---

## 3. The 3 Under-the-Hood Technologies

`torch.compile` is built on a clean 3-layer architecture:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. TorchDynamo (Graph Capture)                              │
│    Intercepts CPython frame evaluation, extracts PyTorch FX │
│    computational subgraphs cleanly without breaking.        │
└──────────────────────────────┬──────────────────────────────┘
                               │ FX Graph
┌──────────────────────────────▼──────────────────────────────┐
│ 2. AOTAutograd (Ahead-of-Time Autograd)                     │
│    Captures forward & backward passes ahead-of-time         │
│    into joint functional FX computation graphs.             │
└──────────────────────────────┬──────────────────────────────┘
                               │ Forward & Backward Graphs
┌──────────────────────────────▼──────────────────────────────┐
│ 3. TorchInductor (Code Generation / Compiler Backend)       │
│    Compiles FX graphs into optimized OpenAI Triton kernels  │
│    for GPUs (and C++/OpenMP for CPUs).                      │
└─────────────────────────────────────────────────────────────┘
```

### 1. TorchDynamo (Python-level capture)
- Uses CPython Frame Evaluation API (PEP 523) to inspect bytecode right before execution.
- If it encounters unsupported Python constructs (e.g. arbitrary external libraries, file I/O, `print`), it introduces a **graph break**: it compiles the supported parts and cleanly executes the unsupported parts in standard Python eager mode.

### 2. AOTAutograd (Joint Graph Capture)
- Traces both the **forward pass** and the **backward pass** (gradients) before execution begins.
- Gives the backend compiler full visibility across both forward activations and backward gradient calculations to optimize memory reuse.

### 3. TorchInductor (Deep Learning Compiler Backend)
- Generates high-performance **OpenAI Triton** code for NVIDIA/AMD GPUs.
- Performs automatic loop fusion, memory allocation planning, and kernel scheduling.

---

## 4. Comparison: `torch.compile` vs. TorchScript vs. FX Tracing

| Feature | `torch.jit.trace` | `torch.jit.script` | `torch.fx` | `torch.compile` (PyTorch 2.0+) |
|:---|:---|:---|:---|:---|
| **Handles Dynamic Control Flow (`if`/`while`)** |  Silently freezes branching |  Requires strict Python subset |  Breaks on arbitrary control flow | ** Fully supported (falls back or handles cleanly)** |
| **Code Changes Required** | None / Low | High (type annotations, rewrite loops) | Medium | **Zero (single wrapper `torch.compile(m)`)** |
| **Graph Breaks** |  Fails silently or gives wrong results |  Throws compilation error |  Throws compilation error | ** Graceful subgraphs / graph breaks** |
| **Backend Code Generation** | Custom C++ JIT IR | Custom C++ JIT IR | Python AST / Interpreted | **OpenAI Triton (State-of-the-Art GPU Kernels)** |
| **Status** | Legacy / Maintenance | Legacy / Maintenance | Specialized Tooling | **Modern Standard (Recommended for all new work)** |

---

## 5. Compilation Modes (`mode=...`)

```python
# 1. Default Mode
model = torch.compile(model)

# 2. Reduce Overhead (Best for small batches / latency-critical inference)
model = torch.compile(model, mode="reduce-overhead")

# 3. Max Autotune (Best for maximum training throughput)
model = torch.compile(model, mode="max-autotune")
```

### Modes Explained:

1. **`"default"`:**
   - Balanced compilation time and speedup.
   - Fuses elementwise kernels, reduces memory bandwidth, optimizes memory layout.

2. **`"reduce-overhead"`:**
   - Leverages **CUDA Graphs** to eliminate CPU launch overhead.
   - Excellent for small batch sizes where GPU kernels run faster than CPU can submit them.
   - *Requirement:* Requires static tensor sizes and shapes (no variable batch size/sequence length).

3. **`"max-autotune"`:**
   - Takes significantly longer to compile initially (minutes).
   - Profiles multiple candidate matrix multiplication (GEMM) algorithms and Triton kernel configurations directly on your hardware to pick the fastest one.
   - Produces the absolute highest throughput for long training runs.

---

## 6. Real-World Speedup Demonstration

In training transformer models like **GPT-2 (124M)**:

| Optimization Step | Typical Step Time (ms) | Tokens / Sec | Relative Speedup |
|:---|:---:|:---:|:---:|
| 1. Baseline FP32 (Eager) | $\sim 1000\text{ ms}$ | $\sim 16\text{k tok/s}$ | $1.0\times$ |
| 2. + TF32 (`torch.set_float32_matmul_precision('high')`) | $\sim 400\text{ ms}$ | $\sim 40\text{k tok/s}$ | $\sim 2.5\times$ |
| 3. + AMP BF16 (`torch.autocast(dtype=torch.bfloat16)`) | $\sim 250\text{ ms}$ | $\sim 65\text{k tok/s}$ | $\sim 4.0\times$ |
| 4. + FlashAttention-2 (`F.scaled_dot_product_attention`) | $\sim 180\text{ ms}$ | $\sim 90\text{k tok/s}$ | $\sim 5.5\times$ |
| 5. **+ `torch.compile(model)`** | **$\sim 120\text{ ms}$** | **$\sim 135\text{k tok/s}$** | **$\sim 8.3\times$** |

*(Note: Compilation incurs an initial "warmup" delay on step 0/1 as kernels are compiled and cached, after which steady-state execution is dramatically faster).*

---

## 7. How to Use in `train_gpt2.py`

### 1. Wrapping the Model
```python
model = GPT(GPTConfig())
model.to(device)

# Compile the model (typically after moving to device)
model = torch.compile(model)
```

### 2. Warmup Step
The first forward & backward pass triggers JIT compilation. Expect step 0 to take a few seconds:
```
step  0, loss: 10.9913, dt: 5432.03ms (compiling kernels...)
step  1, loss:  9.5335, dt:  120.15ms (full compiled speed!)
```

### 3. Hardware Support & Platform Notes

- **NVIDIA GPUs (Ampere, Ada, Hopper, Blackwell — RTX 30/40xx, A100, H100):**
  - Full native support via OpenAI Triton. Maximum speedup achieved.
- **Apple Silicon (MPS - Metal Performance Shaders):**
  - PyTorch 2.0+ `torch.compile` is primarily designed for CUDA/Triton and CPU.
  - On macOS / MPS, `torch.compile` support is experimental and will fallback or require specific backends (`backend="aot_eager"`). For pure training on Mac MPS, eager mode or standard PyTorch ops are commonly used.
- **CPUs (x86 / ARM):**
  - TorchInductor compiles to multithreaded C++ kernels via OpenMP, giving $1.2\times - 1.8\times$ CPU speedups.

---

## 8. Summary Checklist for GPT-2 Pretraining

1. Move model to GPU: `model.to(device)`
2. Enable fast matmuls: `torch.set_float32_matmul_precision('high')`
3. Wrap forward pass with mixed precision: `with torch.autocast(device_type=device, dtype=torch.bfloat16):`
4. Use fused FlashAttention in Transformer blocks (`F.scaled_dot_product_attention`).
5. Compile the model: `model = torch.compile(model)`.
6. Use decoupled AdamW: `torch.optim.AdamW(..., fused=True)`.
