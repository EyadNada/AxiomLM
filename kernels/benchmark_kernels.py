import os
import sys
import time
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from brain.train_gpt2 import RMSNorm, SwiGLUMLP, GPTConfig
from kernels import fused_rmsnorm, fused_swiglu, FusedRMSNorm, FusedSwiGLUMLP


def benchmark_rmsnorm(device: str = "cpu", B: int = 8, T: int = 1024, D: int = 768, num_iters: int = 100):
    print(f"\n=======================================================")
    print(f"  Benchmark 1: RMSNorm (Forward + Backward)")
    print(f"  Tensor Shape: ({B}, {T}, {D}) | Device: {device.upper()}")
    print(f"=======================================================")

    x = torch.randn(B, T, D, device=device, requires_grad=True)
    w = torch.randn(D, device=device, requires_grad=True)

    # 1. Warmup
    for _ in range(10):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        out = x * rms * w
        out.sum().backward()
        x.grad = None
        w.grad = None

    # Benchmark Standard PyTorch
    if device == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        out = x * rms * w
        loss = out.sum()
        loss.backward()
        x.grad = None
        w.grad = None
    if device == "mps":
        torch.mps.synchronize()
    t1 = time.perf_counter()
    ref_time_ms = ((t1 - t0) / num_iters) * 1000.0

    # Benchmark Fused RMSNorm
    for _ in range(10):
        out = fused_rmsnorm(x, w, 1e-6)
        out.sum().backward()
        x.grad = None
        w.grad = None

    if device == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        out = fused_rmsnorm(x, w, 1e-6)
        loss = out.sum()
        loss.backward()
        x.grad = None
        w.grad = None
    if device == "mps":
        torch.mps.synchronize()
    t1 = time.perf_counter()
    fused_time_ms = ((t1 - t0) / num_iters) * 1000.0

    speedup = ref_time_ms / fused_time_ms if fused_time_ms > 0 else 1.0
    print(f"  PyTorch Standard RMSNorm: {ref_time_ms:.3f} ms / pass")
    print(f"  Axiom Fused RMSNorm:     {fused_time_ms:.3f} ms / pass")
    print(f"  Speedup:                  {speedup:.2f}x")


def benchmark_swiglu(device: str = "cpu", B: int = 8, T: int = 1024, D: int = 2048, num_iters: int = 100):
    print(f"\n=======================================================")
    print(f"  Benchmark 2: SwiGLU Activation (Forward + Backward)")
    print(f"  Tensor Shape: ({B}, {T}, {D}) | Device: {device.upper()}")
    print(f"=======================================================")

    gate = torch.randn(B, T, D, device=device, requires_grad=True)
    up = torch.randn(B, T, D, device=device, requires_grad=True)

    # Warmup
    for _ in range(10):
        out = F.silu(gate) * up
        out.sum().backward()
        gate.grad = None
        up.grad = None

    if device == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        out = F.silu(gate) * up
        loss = out.sum()
        loss.backward()
        gate.grad = None
        up.grad = None
    if device == "mps":
        torch.mps.synchronize()
    t1 = time.perf_counter()
    ref_time_ms = ((t1 - t0) / num_iters) * 1000.0

    # Fused SwiGLU
    for _ in range(10):
        out = fused_swiglu(gate, up)
        out.sum().backward()
        gate.grad = None
        up.grad = None

    if device == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        out = fused_swiglu(gate, up)
        loss = out.sum()
        loss.backward()
        gate.grad = None
        up.grad = None
    if device == "mps":
        torch.mps.synchronize()
    t1 = time.perf_counter()
    fused_time_ms = ((t1 - t0) / num_iters) * 1000.0

    speedup = ref_time_ms / fused_time_ms if fused_time_ms > 0 else 1.0
    print(f"  PyTorch Standard SwiGLU:  {ref_time_ms:.3f} ms / pass")
    print(f"  Axiom Fused SwiGLU:       {fused_time_ms:.3f} ms / pass")
    print(f"  Speedup:                  {speedup:.2f}x")


if __name__ == "__main__":
    benchmark_rmsnorm(device="cpu", B=8, T=1024, D=768, num_iters=100)
    benchmark_swiglu(device="cpu", B=8, T=1024, D=2048, num_iters=100)
    if torch.backends.mps.is_available():
        benchmark_rmsnorm(device="mps", B=4, T=512, D=768, num_iters=50)
        benchmark_swiglu(device="mps", B=4, T=512, D=2048, num_iters=50)
