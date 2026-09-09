import os
import sys
import time
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from axiomlm.models.modules import RMSNorm, SwiGLUMLP
from axiomlm.models.transformer import ModelConfig, GPTConfig
from axiomlm.kernels import fused_rmsnorm, fused_swiglu, FusedRMSNorm, FusedSwiGLUMLP


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



def benchmark_rope(device="mps", B=8, n_head=32, T=1024, head_dim=128, num_iters=100):
    print("=" * 55)
    print(f"  Benchmark 3: RoPE (Forward + Backward)")
    print(f"  Tensor Shape: ({B}, {n_head}, {T}, {head_dim}) | Device: {device.upper()}")
    print("=" * 55)
    
    from axiomlm.models.modules import precompute_rope_frequencies
    from axiomlm.kernels.ops import fused_apply_rope
    
    x = torch.randn(B, n_head, T, head_dim, device=device, requires_grad=True)
    freqs_cis = precompute_rope_frequencies(head_dim, max_seq_len=2048).to(device)
    grad_out = torch.randn_like(x)
    
    # Eager implementation (the fallback inside fused_apply_rope if we force HAS_CUSTOM_KERNELS=False)
    # Actually, we can just use the standard eager math directly here to avoid hacking globals
    def eager_rope(x_in, freqs):
        orig_dtype = x_in.dtype
        b, nh, t, hd = x_in.shape
        x_complex = torch.view_as_complex(x_in.float().reshape(b, nh, t, -1, 2))
        freqs_slice = freqs[0 : t, :].view(1, 1, t, -1)
        x_rotated = torch.view_as_real(x_complex * freqs_slice).reshape(b, nh, t, hd)
        return x_rotated.to(orig_dtype)
        
    def run_eager():
        out = eager_rope(x, freqs_cis)
        out.sum().backward()
        
    def run_fused():
        out = fused_apply_rope(x, freqs_cis, start_pos=0)
        out.sum().backward()
        
    for _ in range(5):
        run_eager()
        run_fused()
        
    if device == "mps":
        torch.mps.synchronize()
    import time
    t0 = time.perf_counter()
    for _ in range(num_iters):
        run_eager()
    if device == "mps":
        torch.mps.synchronize()
    t1 = time.perf_counter()
    
    for _ in range(num_iters):
        run_fused()
    if device == "mps":
        torch.mps.synchronize()
    t2 = time.perf_counter()
    
    t_eager = (t1 - t0) * 1000 / num_iters
    t_fused = (t2 - t1) * 1000 / num_iters
    
    print(f"  PyTorch Standard RoPE: {t_eager:.3f} ms / pass")
    print(f"  Axiom Fused RoPE:      {t_fused:.3f} ms / pass")
    print(f"  Speedup:               {t_eager/t_fused:.2f}x\n")



def benchmark_ce(device="mps", B=8, T=1024, V=50257, num_iters=50):
    print("=" * 55)
    print(f"  Benchmark 4: Cross Entropy (Forward + Backward)")
    print(f"  Tensor Shape: ({B*T}, {V}) | Device: {device.upper()}")
    print("=" * 55)
    
    from axiomlm.kernels.ops import fused_cross_entropy
    
    logits = torch.randn(B * T, V, device=device, requires_grad=True)
    targets = torch.randint(0, V, (B * T,), device=device)
    
    def run_eager():
        out = F.cross_entropy(logits, targets, ignore_index=-1)
        out.backward()
        
    def run_fused():
        out = fused_cross_entropy(logits, targets, ignore_index=-1)
        out.backward()
        
    for _ in range(3):
        run_eager()
        run_fused()
        
    if device == "mps":
        torch.mps.synchronize()
    import time
    t0 = time.perf_counter()
    for _ in range(num_iters):
        run_eager()
    if device == "mps":
        torch.mps.synchronize()
    t1 = time.perf_counter()
    
    for _ in range(num_iters):
        run_fused()
    if device == "mps":
        torch.mps.synchronize()
    t2 = time.perf_counter()
    
    t_eager = (t1 - t0) * 1000 / num_iters
    t_fused = (t2 - t1) * 1000 / num_iters
    
    print(f"  PyTorch Standard CE: {t_eager:.3f} ms / pass")
    print(f"  Axiom Fused CE:      {t_fused:.3f} ms / pass")
    print(f"  Speedup:             {t_eager/t_fused:.2f}x\n")

if __name__ == "__main__":
    benchmark_rmsnorm(device="cpu", B=8, T=1024, D=768, num_iters=100)
    benchmark_swiglu(device="cpu", B=8, T=1024, D=2048, num_iters=100)
    if torch.backends.mps.is_available():
        benchmark_rmsnorm(device="mps", B=4, T=512, D=768, num_iters=50)
        benchmark_swiglu(device="mps", B=4, T=512, D=2048, num_iters=50)
        benchmark_rope(device="mps", B=4, n_head=32, T=512, head_dim=64, num_iters=50)
        benchmark_ce(device="mps", B=4, T=512, V=50257, num_iters=50)
