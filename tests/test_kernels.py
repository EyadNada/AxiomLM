import os
import sys
import platform
import unittest
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from axiomlm.kernels import (  # noqa: E402
    fused_rmsnorm,
    fused_swiglu,
    FusedSwiGLUMLP,
    fused_sdpa,
    flash_attention_mps,
    gqa_flash_decode_mps,
    _NEON_MOD,
    _METAL_MOD,
)
from axiomlm.kernels import triton_kernels as _triton_kernels  # noqa: E402
from axiomlm.models import RMSNorm, SwiGLUMLP, GPTConfig  # noqa: E402

IS_ARM64 = platform.machine().lower() in ["arm64", "aarch64"]


class TestCustomKernels(unittest.TestCase):
    """Rigorous mathematical correctness and gradient tests for custom fused kernels."""

    def setUp(self):
        torch.manual_seed(42)
        self.devices = ["cpu"]
        if torch.backends.mps.is_available():
            self.devices.append("mps")

    @unittest.skipUnless(
        IS_ARM64,
        "Native ARM NEON SIMD kernels require ARM64 / Apple Silicon architecture",
    )
    def test_neon_module_loaded(self):
        """Verify native ARM NEON C++ extension is loaded and functional."""
        self.assertIsNotNone(
            _NEON_MOD, "Native NEON kernel module should be compiled and loaded."
        )
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
            self.assertLess(
                max_diff,
                1e-5,
                f"RMSNorm forward diff {max_diff} on {device} exceeds tolerance",
            )

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

            self.assertLess(
                grad_x_diff, 1e-4, f"RMSNorm grad_x diff {grad_x_diff} on {device}"
            )
            self.assertLess(
                grad_w_diff, 1e-4, f"RMSNorm grad_w diff {grad_w_diff} on {device}"
            )

    def test_fused_rmsnorm_gradcheck(self):
        """Use torch.autograd.gradcheck for double-precision gradient verification."""
        x = torch.randn(2, 4, 16, dtype=torch.float64, requires_grad=True)
        weight = torch.randn(16, dtype=torch.float64, requires_grad=True)
        test_passed = torch.autograd.gradcheck(
            fused_rmsnorm, (x, weight, 1e-6), eps=1e-6, atol=1e-4
        )
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
            self.assertLess(
                max_diff,
                1e-5,
                f"SwiGLU forward diff {max_diff} on {device} exceeds tolerance",
            )

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

            self.assertLess(
                grad_g_diff, 1e-4, f"SwiGLU grad_gate diff {grad_g_diff} on {device}"
            )
            self.assertLess(
                grad_u_diff, 1e-4, f"SwiGLU grad_up diff {grad_u_diff} on {device}"
            )

    def test_fused_swiglu_gradcheck(self):
        """Use torch.autograd.gradcheck for double-precision SwiGLU verification."""
        gate = torch.randn(2, 4, 16, dtype=torch.float64, requires_grad=True)
        up = torch.randn(2, 4, 16, dtype=torch.float64, requires_grad=True)
        test_passed = torch.autograd.gradcheck(
            fused_swiglu, (gate, up), eps=1e-6, atol=1e-4
        )
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
            self.assertLess(
                diff, 1e-5, f"FusedSwiGLUMLP output diff {diff} on {device}"
            )

    def test_fused_sdpa_forward_parity(self):
        """Test Fused SDPA forward pass matches reference PyTorch SDPA."""
        for device in self.devices:
            B, H, T, D = 2, 4, 16, 64
            q = torch.randn(B, H, T, D, device=device)
            k = torch.randn(B, H, T, D, device=device)
            v = torch.randn(B, H, T, D, device=device)

            ref_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            custom_out = fused_sdpa(q, k, v, is_causal=True)

            max_diff = (ref_out - custom_out).abs().max().item()
            self.assertLess(
                max_diff,
                1e-4,
                f"SDPA forward diff {max_diff} on {device} exceeds tolerance",
            )

    def test_fused_sdpa_backward_gradients(self):
        """Test Fused SDPA backward pass gradients match reference."""
        for device in self.devices:
            B, H, T, D = 2, 4, 16, 64
            q1 = torch.randn(B, H, T, D, device=device, requires_grad=True)
            k1 = torch.randn(B, H, T, D, device=device, requires_grad=True)
            v1 = torch.randn(B, H, T, D, device=device, requires_grad=True)

            q2 = q1.detach().clone().requires_grad_(True)
            k2 = k1.detach().clone().requires_grad_(True)
            v2 = v1.detach().clone().requires_grad_(True)

            # Reference
            y_ref = F.scaled_dot_product_attention(q1, k1, v1, is_causal=True)
            loss_ref = (y_ref * 2.0).sum()
            loss_ref.backward()

            # Fused
            y_custom = fused_sdpa(q2, k2, v2, is_causal=True)
            loss_custom = (y_custom * 2.0).sum()
            loss_custom.backward()

            grad_q_diff = (q1.grad - q2.grad).abs().max().item()
            grad_k_diff = (k1.grad - k2.grad).abs().max().item()
            grad_v_diff = (v1.grad - v2.grad).abs().max().item()

            self.assertLess(
                grad_q_diff, 1e-4, f"SDPA grad_q diff {grad_q_diff} on {device}"
            )
            self.assertLess(
                grad_k_diff, 1e-4, f"SDPA grad_k diff {grad_k_diff} on {device}"
            )
            self.assertLess(
                grad_v_diff, 1e-4, f"SDPA grad_v diff {grad_v_diff} on {device}"
            )

    def test_metal_flash_attention_parity(self):
        """Test custom Metal tiled SRAM FlashAttention kernel (flash_attention_forward_mps) matches reference."""
        if "mps" not in self.devices or _METAL_MOD is None:
            self.skipTest("MPS Metal module not available")
        B, H, T, D = 2, 4, 32, 64
        q = torch.randn(B, H, T, D, device="mps", dtype=torch.float32)
        k = torch.randn(B, H, T, D, device="mps", dtype=torch.float32)
        v = torch.randn(B, H, T, D, device="mps", dtype=torch.float32)

        ref_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        metal_out = flash_attention_mps(q, k, v, is_causal=True)
        max_diff = (ref_out - metal_out).abs().max().item()
        self.assertLess(
            max_diff, 1e-4, f"Metal FlashAttention diff {max_diff} exceeds tolerance"
        )

    def test_triton_wrapper_names_alignment(self):
        """Verify triton_kernels exports both triton_* and short aliases expected by ops.py."""
        for fn_name in [
            "triton_rmsnorm_forward",
            "triton_rmsnorm_backward",
            "triton_swiglu_forward",
            "triton_swiglu_backward",
            "rmsnorm_forward",
            "rmsnorm_backward",
            "swiglu_forward",
            "swiglu_backward",
            "triton_fused_sdpa_forward",
        ]:
            self.assertTrue(
                hasattr(_triton_kernels, fn_name),
                f"Missing {fn_name} in triton_kernels.py",
            )

    def test_quant_packing_roundtrip(self):
        """Verify Q4_G64 and Q8_G64 packing and dequantization precision."""
        from axiomlm.kernels import (
            pack_linear_q4_g64,
            dequantize_q4_g64,
            pack_linear_q8_g64,
            dequantize_q8_g64,
        )

        torch.manual_seed(42)
        N, K = 256, 768
        w = torch.randn(N, K, dtype=torch.float16)

        # Q4_G64
        packed_q4, sb_q4 = pack_linear_q4_g64(w)
        w_dq_q4 = dequantize_q4_g64(packed_q4, sb_q4, K)
        cos_sim_q4 = torch.cosine_similarity(
            w.float().view(-1), w_dq_q4.float().view(-1), dim=0
        ).item()
        self.assertGreater(
            cos_sim_q4, 0.995, f"Q4 Cosine similarity too low: {cos_sim_q4}"
        )

        # Q8_G64
        packed_q8, sb_q8 = pack_linear_q8_g64(w)
        w_dq_q8 = dequantize_q8_g64(packed_q8, sb_q8, K)
        cos_sim_q8 = torch.cosine_similarity(
            w.float().view(-1), w_dq_q8.float().view(-1), dim=0
        ).item()
        self.assertGreater(
            cos_sim_q8, 0.9999, f"Q8 Cosine similarity too low: {cos_sim_q8}"
        )

    def test_metal_qmv_parity(self):
        """Verify Q4_G64 and Q8_G64 Metal GEMV matches reference dequantized GEMV on MPS."""
        if "mps" not in self.devices or _METAL_MOD is None:
            self.skipTest("MPS Metal module not available")

        from axiomlm.kernels import (
            pack_linear_q4_g64,
            dequantize_q4_g64,
            pack_linear_q8_g64,
            dequantize_q8_g64,
            qmv_q4_g64,
            qmv_q8_g64,
        )

        torch.manual_seed(999)
        device = torch.device("mps")
        N, K = 768, 768
        w = torch.randn(N, K, dtype=torch.float16)
        x = torch.randn(1, K, dtype=torch.float16, device=device)

        # Test Q4
        p_q4, sb_q4 = pack_linear_q4_g64(w)
        y_q4 = qmv_q4_g64(x, p_q4.to(device), sb_q4.to(device))
        y_ref_q4 = F.linear(x, dequantize_q4_g64(p_q4, sb_q4, K).to(device))
        cos_q4 = torch.cosine_similarity(
            y_q4.float().view(-1), y_ref_q4.float().view(-1), dim=0
        ).item()
        self.assertGreater(
            cos_q4, 0.9999, f"Metal Q4 GEMV cosine sim too low: {cos_q4}"
        )

        # Test Q8
        p_q8, sb_q8 = pack_linear_q8_g64(w)
        y_q8 = qmv_q8_g64(x, p_q8.to(device), sb_q8.to(device))
        y_ref_q8 = F.linear(x, dequantize_q8_g64(p_q8, sb_q8, K).to(device))
        cos_q8 = torch.cosine_similarity(
            y_q8.float().view(-1), y_ref_q8.float().view(-1), dim=0
        ).item()
        self.assertGreater(
            cos_q8, 0.99999, f"Metal Q8 GEMV cosine sim too low: {cos_q8}"
        )

    def test_quantized_model_decode_runner(self):
        """Verify Transformer.quantize('q4' / 'q8') executes through MPSQuantizedDecodeRunner."""
        if "mps" not in self.devices or _METAL_MOD is None:
            self.skipTest("MPS Metal module not available")

        from axiomlm.models.transformer import Transformer, ModelConfig
        from axiomlm.engine.paged_cache import StaticKVCache

        device = torch.device("mps")
        cfg = ModelConfig(
            arch="modern",
            n_layer=2,
            n_head=4,
            n_kv_head=2,
            n_embd=256,
            vocab_size=512,
            block_size=512,
            use_fused_kernels=True,
        )
        model = Transformer(cfg).to(device=device, dtype=torch.float16).eval()
        cache = StaticKVCache(
            max_batch_size=1,
            max_seq_len=256,
            n_layer=cfg.n_layer,
            n_kv_head=cfg.n_kv_head,
            head_dim=cfg.n_embd // cfg.n_head,
            device=device,
            dtype=torch.float16,
        )

        curr_token = torch.tensor([[10]], device=device, dtype=torch.long)
        with torch.no_grad():
            # 1. FP16 Decode step
            logits_fp16, _, _ = model(curr_token, kv_caches=[cache] * cfg.n_layer)
            self.assertIsNotNone(model._mps_decode_runner)

            # 2. Q8 Decode step
            model.quantize("q8")
            cache.reset()
            logits_q8, _, _ = model(curr_token, kv_caches=[cache] * cfg.n_layer)
            self.assertIsNotNone(model._mps_quant_decode_runner)

            cos_q8 = torch.cosine_similarity(
                logits_fp16.float().view(-1), logits_q8.float().view(-1), dim=0
            ).item()
            self.assertGreater(
                cos_q8, 0.999, f"Q8 model decode cos sim too low: {cos_q8}"
            )

            # 3. Q4 Decode step
            model.quantize("q4")
            cache.reset()
            logits_q4, _, _ = model(curr_token, kv_caches=[cache] * cfg.n_layer)
            self.assertIsNotNone(model._mps_quant_decode_runner)

            cos_q4 = torch.cosine_similarity(
                logits_fp16.float().view(-1), logits_q4.float().view(-1), dim=0
            ).item()
            self.assertGreater(
                cos_q4, 0.98, f"Q4 model decode cos sim too low: {cos_q4}"
            )

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available")
    def test_metal_gqa_flash_decode_parity(self):
        """Test Metal Split-KV Flash-Decode against PyTorch SDPA with 3:1 GQA."""
        device = torch.device("mps")
        n_head = 12
        n_kv_head = 4
        head_dim = 64
        max_seq_len = 1024

        for seq_len in [32, 64, 256, 1024]:
            q = torch.randn(1, n_head, 1, head_dim, dtype=torch.float16, device=device)
            k_cache = torch.randn(
                1, n_kv_head, max_seq_len, head_dim, dtype=torch.float16, device=device
            )
            v_cache = torch.randn(
                1, n_kv_head, max_seq_len, head_dim, dtype=torch.float16, device=device
            )

            k_act = k_cache[:, :, :seq_len, :]
            v_act = v_cache[:, :, :seq_len, :]
            ref = torch.nn.functional.scaled_dot_product_attention(
                q, k_act, v_act, enable_gqa=True
            )

            for n_splits in [1, 2, 4, 8]:
                out = gqa_flash_decode_mps(
                    q, k_cache, v_cache, active_len=seq_len, num_splits=n_splits
                )
                torch.mps.synchronize()

                cos_sim = torch.nn.functional.cosine_similarity(
                    ref.flatten().float(), out.flatten().float(), dim=0
                ).item()
                max_diff = (ref - out).abs().max().item()

                self.assertGreater(
                    cos_sim,
                    0.9999,
                    f"Low cosine similarity: {cos_sim} at T={seq_len}, S={n_splits}",
                )
                self.assertLess(
                    max_diff,
                    0.05,
                    f"Excessive max diff: {max_diff} at T={seq_len}, S={n_splits}",
                )


if __name__ == "__main__":
    unittest.main()
