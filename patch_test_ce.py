with open("tests/test_kernels.py", "r") as f:
    content = f.read()

ce_test = """
    def test_fused_cross_entropy(self):
        \"\"\"Test FusedCrossEntropy forward and backward matches PyTorch exactly.\"\"\"
        if not torch.backends.mps.is_available():
            self.skipTest("MPS not available")
            
        from axiomlm.kernels.ops import fused_cross_entropy
        
        device = torch.device("mps")
        B, T, V = 2, 64, 1024
        ignore_index = -1
        
        logits = torch.randn(B * T, V, device=device, requires_grad=True)
        targets = torch.randint(-1, V, (B * T,), device=device)
        
        # Reference implementation
        logits_ref = logits.clone().detach().requires_grad_(True)
        loss_ref = torch.nn.functional.cross_entropy(logits_ref, targets, ignore_index=ignore_index)
        
        # Fused implementation
        logits_fused = logits.clone().detach().requires_grad_(True)
        loss_fused = fused_cross_entropy(logits_fused, targets, ignore_index=ignore_index)
        
        diff = abs(loss_ref.item() - loss_fused.item())
        self.assertLess(diff, 1e-4, f"CE forward diff {diff}")
        
        # Backward
        loss_ref.backward()
        loss_fused.backward()
        
        max_grad_diff = (logits_ref.grad - logits_fused.grad).abs().max().item()
        self.assertLess(max_grad_diff, 1e-4, f"CE backward diff {max_grad_diff}")
"""

content = content.replace("if __name__ == '__main__':", ce_test + "\nif __name__ == '__main__':")

with open("tests/test_kernels.py", "w") as f:
    f.write(content)
