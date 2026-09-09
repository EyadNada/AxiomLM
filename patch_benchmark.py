import re

with open("axiomlm/kernels/benchmark_kernels.py", "r") as f:
    content = f.read()

rope_benchmark = """
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
        
    t_eager = benchmark_func(run_eager, num_iters=num_iters)
    t_fused = benchmark_func(run_fused, num_iters=num_iters)
    
    print(f"  PyTorch Standard RoPE: {t_eager:.3f} ms / pass")
    print(f"  Axiom Fused RoPE:      {t_fused:.3f} ms / pass")
    print(f"  Speedup:               {t_eager/t_fused:.2f}x\\n")

"""

# Insert benchmark_rope function before the main block
content = content.replace("if __name__ == \"__main__\":", rope_benchmark + "\nif __name__ == \"__main__\":")

# Add call to benchmark_rope in main block
content = content.replace('benchmark_swiglu(device="mps", B=4, T=512, D=2048, num_iters=100)',
                          'benchmark_swiglu(device="mps", B=4, T=512, D=2048, num_iters=100)\n    benchmark_rope(device="mps", B=4, n_head=32, T=512, head_dim=64, num_iters=100)')

with open("axiomlm/kernels/benchmark_kernels.py", "w") as f:
    f.write(content)
