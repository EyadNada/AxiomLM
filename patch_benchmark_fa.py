with open("axiomlm/kernels/benchmark_kernels.py", "r") as f:
    content = f.read()

fa_benchmark = """
def benchmark_fa(device="mps", B=4, n_head=12, T=1024, head_dim=64, num_iters=50):
    print("=" * 55)
    print(f"  Benchmark 5: Flash Attention (Forward Only)")
    print(f"  Tensor Shape: ({B}, {n_head}, {T}, {head_dim}) | Device: {device.upper()}")
    print("=" * 55)
    
    q = torch.randn(B, n_head, T, head_dim, device=device)
    k = torch.randn(B, n_head, T, head_dim, device=device)
    v = torch.randn(B, n_head, T, head_dim, device=device)
    
    # We call the metal mod directly for fused, and F.scaled_dot_product_attention for eager
    import axiomlm.kernels.ops as ops
    
    def run_eager():
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        
    def run_fused():
        out = ops.fused_sdpa(q, k, v, is_causal=True)
        
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
    
    print(f"  Apple MPSGraph Attention: {t_eager:.3f} ms / pass")
    print(f"  Axiom Metal FlashAttn:    {t_fused:.3f} ms / pass")
    print(f"  Speedup:                  {t_eager/t_fused:.2f}x\\n")
"""

content = content.replace("if __name__ == \"__main__\":", fa_benchmark + "\nif __name__ == \"__main__\":")
content = content.replace('benchmark_ce(device="mps", B=4, T=512, V=50257, num_iters=50)',
                          'benchmark_ce(device="mps", B=4, T=512, V=50257, num_iters=50)\n        benchmark_fa(device="mps", B=4, n_head=12, T=1024, head_dim=64, num_iters=50)')

with open("axiomlm/kernels/benchmark_kernels.py", "w") as f:
    f.write(content)
