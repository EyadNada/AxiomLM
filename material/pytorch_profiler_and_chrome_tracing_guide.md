# PyTorch Profiler & Chrome/Perfetto Trace Analysis Guide

A practical engineering guide to profiling large language model training loops, identifying hardware bottlenecks, and visualizing GPU execution timelines using `torch.profiler`.

---

## 1. Why Profile with `torch.profiler`?

Simple wall-clock timers (`time.time()`) only provide gross latency averages. They cannot answer critical systems questions:
1. Is the GPU/MPS accelerator sitting idle waiting for CPU batch loading?
2. Are un-fused elementwise operations thrashing the memory bus?
3. Where are hidden host-device synchronizations occurring?
4. How much time is spent in Forward GEMM vs Attention vs Backward pass vs Optimizer updates?

`torch.profiler` directly hooks into the PyTorch C++ dispatcher, CUDA streams, and Metal execution queues to record microsecond-accurate trace events.

---

## 2. Common Latency Pitfalls & Silent Bottlenecks

### A. Hidden Host-Device Synchronizations
A single CPU-GPU synchronization halts the asynchronous GPU pipeline, forcing the accelerator to flush its queue:
* Calling `.item()` on a loss tensor inside the inner gradient accumulation loop.
* Printing or inspecting tensors (`print(x)`) during training.
* Calling `tensor.cpu()` or `tensor.numpy()` before passing to DataLoader.

**Solution:** Accumulate loss tensors on device (`loss_accum += loss.detach()`), and synchronize only once per optimization step.

---

### B. Memory Allocation Stalls (VRAM Thrashing)
Creating dynamic tensors or modifying shape inside the forward loop triggers PyTorch's caching allocator (`cudaMalloc` / `mps::alloc`):
* Concatenating tensors dynamically (`torch.cat`) without preallocation.
* Materializing full $O(T^2)$ causal masks on every forward pass.

**Solution:** Use `scaled_dot_product_attention(is_causal=True)`, reuse persistent buffers, and precompute RoPE frequency grids.

---

### C. Kernel Launch Overhead & CPU Dispatch Latency
When tensor sizes are too small (e.g. batch size $= 1$ or micro-steps without gradient accumulation), the time taken by the CPU to dispatch CUDA/MPS kernels exceeds the kernel runtime, leaving the GPU starved for work.

---

## 3. Profiler Implementation in PyTorch

```python
import torch

def create_profiler(output_dir: str = "./log/profiler_trace"):
    """
    Configures a standard PyTorch profiler with schedule:
    - wait: 1 step (warmup allocator)
    - warmup: 1 step (JIT and graph optimization)
    - active: 3 steps (record detailed trace)
    """
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        activities.append(torch.profiler.ProfilerActivity.CPU)

    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )
```

---

## 4. How to Inspect Trace Files (`trace.json`)

1. Run the training script with the profiler enabled:
   ```bash
   python3 brain/train_gpt2.py --profile
   ```
2. Open Google Chrome and navigate to:
   * **`chrome://tracing`** or **[ui.perfetto.dev](https://ui.perfetto.dev)**
3. Drag and drop the generated `.json` trace file.

### Trace Timeline Breakdown
```
[ CPU Thread ]  ---- aten::linear ---- aten::sdpa ---- aten::rmsnorm ---- opt::step ----
[ GPU Stream ]  ====== gemm_kernel ====== flash_attn_fwd ====== rmsnorm_fwd ===== ns_matmul =====
```

* **GEMM Bands:** Large solid blocks indicate compute-bound Tensor Core / Metal AMX operations.
* **Gaps between Bands:** Reveal CPU dispatch bubbles, data loader stalls, or synchronous barriers.
