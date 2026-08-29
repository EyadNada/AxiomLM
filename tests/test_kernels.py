import os
import sys
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from kernels import (
    fused_rmsnorm,
    FusedRMSNorm,
    fused_swiglu,
    FusedSwiGLUMLP,
    _NEON_MOD,
)
from brain.train_gpt2 import RMSNorm, SwiGLUMLP, GPTConfig


class TestCustomKernels(unittest.TestCase):
    """Rigorous mathematical correctness and gradient tests for custom fused kernels."""

    def setUp(self):
        torch.manual_seed(42)
        self.devices = ["cpu"]
        if torch.backends.mps.is_available():
            self.devices.append("mps")

    def test_neon_module_loaded(self):
        """Verify native ARM NEON C++ extension is loaded and functional."""
        self.assertIsNotNone(_NEON_MOD, "Native NEON kernel module should be compiled and loaded.")
        self.assertTrue(hasattr(_NEON_MOD, "rmsnorm_forward_neon"))
        self.assertTrue(hasattr(_NEON_MOD, "rmsnorm_backward_neon"))
        self.assertTrue(hasattr(_NEON_MOD, "swiglu_forward_neon"))
        self.assertTrue(hasattr(_NEON_MOD, "swiglu_backward_neon"))

    def test_fused_rmsnorm_forward_parity(self):
        """Test FusedRMSNorm produces exact numerical match with standard RMSNorm."""
        for device in self.devices:
            B, T, D = 4, 32, 768
            x = torch.randn(B, T, D, device=device)
            weight = torch.randn(D, device=device)

            ref_norm = RMSNorm(D).to(device)
            ref_norm.weight.data.copy_(weight)
            ref_out = ref_norm(x)

            custom_out = fused_rmsnorm(x, weight, eps=1e-6)

            max_diff = (ref_out - custom_out).abs().max().item()
            self.assertLess(max_diff, 1e-5, f"RMSNorm forward diff {max_diff} on {device} exceeds tolerance")

    def test_fused_rmsnorm_backward_gradients(self):
        """Test FusedRMSNorm backward pass gradients match PyTorch autograd exactly."""
        for device in self.devices:
            B, T, D = 2, 8, 64
            x1 = torch.randn(B, T, D, device=device, requires_grad=True)
            w1 = torch.randn(D, device=device, requires_grad=True)

            x2 = x1.detach().clone().requires_grad_(True)
            w2 = w1.detach().clone().requires_grad_(True)

            # Reference backward
            rms_ref = torch.rsqrt(x1.pow(2).mean(-1, keepdim=True) + 1e-6)
            y_ref = x1 * rms_ref * w1
            loss_ref = (y_ref * 2.5).sum()
            loss_ref.backward()

            # Fused backward
            y_custom = fused_rmsnorm(x2, w2, eps=1e-6)
            loss_custom = (y_custom * 2.5).sum()
            loss_custom.backward()

            grad_x_diff = (x1.grad - x2.grad).abs().max().item()
            grad_w_diff = (w1.grad - w2.grad).abs().max().item()

            self.assertLess(grad_x_diff, 1e-4, f"RMSNorm grad_x diff {grad_x_diff} on {device}")
            self.assertLess(grad_w_diff, 1e-4, f"RMSNorm grad_w diff {grad_w_diff} on {device}")

    def test_fused_rmsnorm_gradcheck(self):
        """Use torch.autograd.gradcheck for double-precision gradient verification."""
        x = torch.randn(2, 4, 16, dtype=torch.float64, requires_grad=True)
        weight = torch.randn(16, dtype=torch.float64, requires_grad=True)
        test_passed = torch.autograd.gradcheck(fused_rmsnorm, (x, weight, 1e-6), eps=1e-6, atol=1e-4)
        self.assertTrue(test_passed, "Gradcheck for FusedRMSNorm failed")

    def test_fused_swiglu_forward_parity(self):
        """Test FusedSwiGLU forward pass matches reference SiLU(gate) * up."""
        for device in self.devices:
            B, T, D = 4, 32, 512
            gate = torch.randn(B, T, D, device=device)
            up = torch.randn(B, T, D, device=device)

            ref_out = F.silu(gate) * up
            custom_out = fused_swiglu(gate, up)

            max_diff = (ref_out - custom_out).abs().max().item()
            self.assertLess(max_diff, 1e-5, f"SwiGLU forward diff {max_diff} on {device} exceeds tolerance")

    def test_fused_swiglu_backward_gradients(self):
        """Test FusedSwiGLU backward pass gradients match PyTorch autograd exactly."""
        for device in self.devices:
            B, T, D = 2, 8, 64
            g1 = torch.randn(B, T, D, device=device, requires_grad=True)
            u1 = torch.randn(B, T, D, device=device, requires_grad=True)

            g2 = g1.detach().clone().requires_grad_(True)
            u2 = u1.detach().clone().requires_grad_(True)

            # Reference
            y_ref = F.silu(g1) * u1
            loss_ref = (y_ref * 3.0).sum()
            loss_ref.backward()

            # Fused
            y_custom = fused_swiglu(g2, u2)
            loss_custom = (y_custom * 3.0).sum()
            loss_custom.backward()

            grad_g_diff = (g1.grad - g2.grad).abs().max().item()
            grad_u_diff = (u1.grad - u2.grad).abs().max().item()

            self.assertLess(grad_g_diff, 1e-4, f"SwiGLU grad_gate diff {grad_g_diff} on {device}")
            self.assertLess(grad_u_diff, 1e-4, f"SwiGLU grad_up diff {grad_u_diff} on {device}")

    def test_fused_swiglu_gradcheck(self):
        """Use torch.autograd.gradcheck for double-precision SwiGLU verification."""
        gate = torch.randn(2, 4, 16, dtype=torch.float64, requires_grad=True)
        up = torch.randn(2, 4, 16, dtype=torch.float64, requires_grad=True)
        test_passed = torch.autograd.gradcheck(fused_swiglu, (gate, up), eps=1e-6, atol=1e-4)
        self.assertTrue(test_passed, "Gradcheck for FusedSwiGLU failed")

    def test_fused_swiglu_mlp_module(self):
        """Test full FusedSwiGLUMLP module vs standard SwiGLUMLP module."""
        for device in self.devices:
            config = GPTConfig(n_embd=768, mlp_type="swiglu")
            ref_mlp = SwiGLUMLP(config).to(device)
            fused_mlp = FusedSwiGLUMLP(config).to(device)

            # Copy identical weights
            fused_mlp.w_gate.weight.data.copy_(ref_mlp.w_gate.weight.data)
            fused_mlp.w_up.weight.data.copy_(ref_mlp.w_up.weight.data)
            fused_mlp.w_down.weight.data.copy_(ref_mlp.w_down.weight.data)
            if ref_mlp.w_gate.bias is not None:
                fused_mlp.w_gate.bias.data.copy_(ref_mlp.w_gate.bias.data)
                fused_mlp.w_up.bias.data.copy_(ref_mlp.w_up.bias.data)
                fused_mlp.w_down.bias.data.copy_(ref_mlp.w_down.bias.data)

            x = torch.randn(2, 16, 768, device=device)
            ref_out = ref_mlp(x)
            fused_out = fused_mlp(x)

            diff = (ref_out - fused_out).abs().max().item()
            self.assertLess(diff, 1e-5, f"FusedSwiGLUMLP output diff {diff} on {device}")


if __name__ == "__main__":
    unittest.main()
