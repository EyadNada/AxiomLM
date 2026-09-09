import re

with open("tests/test_kernels.py", "r") as f:
    content = f.read()

rope_test = """
    def test_fused_rope_parity(self):
        \"\"\"Test FusedRoPE forward and backward pass matches PyTorch reference exactly.\"\"\"
        if not torch.backends.mps.is_available():
            self.skipTest("MPS not available")
            
        from axiomlm.models.modules import precompute_rope_frequencies, apply_rope
        from axiomlm.kernels.ops import fused_apply_rope
        
        device = torch.device("mps")
        B, n_head, T, head_dim = 2, 4, 128, 64
        
        x = torch.randn(B, n_head, T, head_dim, device=device, requires_grad=True)
        freqs_cis = precompute_rope_frequencies(head_dim, max_seq_len=256).to(device)
        
        # Reference implementation
        x_ref = x.clone().detach().requires_grad_(True)
        out_ref = apply_rope(x_ref, freqs_cis, start_pos=0)
        
        # Fused implementation
        x_fused = x.clone().detach().requires_grad_(True)
        out_fused = fused_apply_rope(x_fused, freqs_cis, start_pos=0)
        
        max_diff = (out_ref - out_fused).abs().max().item()
        self.assertLess(max_diff, 1e-5, f"RoPE forward max diff {max_diff}")
        
        # Backward
        grad_out = torch.randn_like(out_ref)
        out_ref.backward(grad_out)
        out_fused.backward(grad_out)
        
        max_grad_diff = (x_ref.grad - x_fused.grad).abs().max().item()
        self.assertLess(max_grad_diff, 1e-4, f"RoPE backward max diff {max_grad_diff}")
"""

content = content.replace("if __name__ == '__main__':", rope_test + "\nif __name__ == '__main__':")

with open("tests/test_kernels.py", "w") as f:
    f.write(content)
