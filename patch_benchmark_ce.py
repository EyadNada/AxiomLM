with open("axiomlm/kernels/benchmark_kernels.py", "r") as f:
    content = f.read()

ce_benchmark = """
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
    print(f"  Speedup:             {t_eager/t_fused:.2f}x\\n")
"""

content = content.replace("if __name__ == \"__main__\":", ce_benchmark + "\nif __name__ == \"__main__\":")
content = content.replace('benchmark_rope(device="mps", B=4, n_head=32, T=512, head_dim=64, num_iters=50)',
                          'benchmark_rope(device="mps", B=4, n_head=32, T=512, head_dim=64, num_iters=50)\n        benchmark_ce(device="mps", B=4, T=512, V=50257, num_iters=50)')

with open("axiomlm/kernels/benchmark_kernels.py", "w") as f:
    f.write(content)
