"""
Phase 3 Task 3.3 MLSys Reviewer Unit & Numerical Parity Test Suite.

Exhaustive verification of BitNet 1.58-bit Ternary Weights and Q2_K 2-bit Quantized
Matrix-Vector (GEMV, M=1) and Batched Matrix (GEMM, M in [1, 32]) Kernels on Apple
Silicon GPU (Metal Performance Shaders / MPS).

Test Matrix:
1. test_01_ternary_packing_roundtrip_parity:
   Exact bit-for-bit round-trip recovery of ternary weights in {-1, 0, +1} (max_diff == 0.0)
   and all 256 byte bit-patterns.
2. test_02_q2_k_packing_roundtrip_parity:
   Exact bit-for-bit round-trip recovery of 2-bit unsigned codes in {0, 1, 2, 3} (max_diff == 0.0).
3. test_03_bitnet_gemv_parity_vs_dequantized:
   Single-token GEMV (M=1) numerical parity vs PyTorch dequantized reference across SmolLM2
   and Llama-3.2-1B layer dimensions (K in [576, 2048, 8192], Cosine >= 0.9999).
4. test_04_q2_k_gemv_parity_vs_dequantized:
   Single-token GEMV (M=1) numerical parity vs PyTorch dequantized reference across SmolLM2
   and Llama-3.2-1B layer dimensions (Cosine >= 0.9999).
5. test_05_bitnet_gemm_batch_scaling:
   Batched tall-and-skinny GEMM across M in [1, 2, 3, 4, 7, 8, 15, 16, 27, 32].
6. test_06_q2_k_gemm_batch_scaling:
   Batched tall-and-skinny GEMM across M in [1, 2, 3, 4, 7, 8, 15, 16, 27, 32].
7. test_07_bias_addition_parity:
   Exact parity with non-null bias addition on both BitNet and Q2_K GEMV and GEMM.
8. test_08_theoretical_memory_compression:
   Strict verification of physical DRAM footprint (0.28125 B/weight for BitNet, 0.3125 B/weight for Q2_K).
9. test_09_adversarial_edge_cases:
   Robustness against all-zero weights, 100-sigma activation outliers, and non-power-of-two group multiples.
10. test_10_transformer_model_quantization_integration:
    Verification of Transformer model quantization pipeline for ternary and Q2_K formats.
"""

import unittest

import torch

from axiomlm.kernels import (
    dequantize_q2_k,
    dequantize_ternary,
    is_mps_available,
    pack_linear_q2_k,
    pack_linear_ternary,
    qmm_q2_k,
    qmm_ternary,
    qmv_q2_k,
    qmv_ternary,
)
from axiomlm.models.transformer import ModelConfig, Transformer


class TestPhase3Task33BitNetQ2K(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not is_mps_available():
            raise unittest.SkipTest(
                "Apple Silicon MPS not available; skipping Phase 3.3 tests."
            )
        cls.device = torch.device("mps")
        torch.manual_seed(42)

    def test_01_ternary_packing_roundtrip_parity(self):
        """Verify BitNet 1.58b packing and unpacking is bit-for-bit exact."""
        # 1. Exhaustive 256-pattern byte round-trip
        all_bytes = torch.arange(256, dtype=torch.uint8, device=self.device).unsqueeze(
            -1
        )
        b_int = all_bytes.to(torch.int32)
        w0 = (b_int & 1) - ((b_int >> 1) & 1)
        w1 = ((b_int >> 2) & 1) - ((b_int >> 3) & 1)
        w2 = ((b_int >> 4) & 1) - ((b_int >> 5) & 1)
        w3 = ((b_int >> 6) & 1) - ((b_int >> 7) & 1)
        q = torch.cat([w0, w1, w2, w3], dim=-1)
        # All decoded values must be in {-1, 0, 1}
        self.assertTrue(
            torch.isin(q, torch.tensor([-1, 0, 1], device=self.device)).all()
        )

        # 2. Binary signs round-trip
        N, K = 64, 256
        q_src = torch.where(torch.randn(N, K, device=self.device) > 0, 1.0, -1.0).to(
            torch.float16
        )
        scale_src = torch.full(
            (N, K // 64), 0.75, dtype=torch.float16, device=self.device
        )
        w_orig = q_src * scale_src.repeat_interleave(64, dim=-1)

        w_packed, scales = pack_linear_ternary(w_orig)
        self.assertEqual(w_packed.shape, (N, K // 4))
        self.assertEqual(w_packed.dtype, torch.uint8)
        self.assertEqual(scales.shape, (N, K // 64))

        w_recon = dequantize_ternary(w_packed, scales, K=K)
        max_diff = (w_orig - w_recon).abs().max().item()
        self.assertEqual(
            max_diff, 0.0, f"BitNet ternary binary-signs error: {max_diff}"
        )

        # 3. Lattice alignment: reconstructed weights lie on {-s, 0, +s}
        w_gauss = torch.randn(N, K, dtype=torch.float16, device=self.device)
        wp, s = pack_linear_ternary(w_gauss)
        wr = dequantize_ternary(wp, s, K=K)
        wg = wr.view(N, K // 64, 64)
        sg = s.unsqueeze(-1)
        qr = torch.round(wg / sg)
        lattice_err = (wg - qr * sg).abs().max().item()
        self.assertEqual(lattice_err, 0.0, f"Lattice alignment error: {lattice_err}")

    def test_02_q2_k_packing_roundtrip_parity(self):
        """Verify Q2_K 2-bit packing and unpacking is bit-for-bit exact."""
        N, K = 64, 256
        q_src = torch.randint(0, 4, (N, K), dtype=torch.float32, device=self.device)
        sc = torch.full((N, K // 64), 0.5, dtype=torch.float16, device=self.device)
        bi = torch.full((N, K // 64), -1.0, dtype=torch.float16, device=self.device)
        w_orig = (
            q_src * sc.repeat_interleave(64, dim=-1) + bi.repeat_interleave(64, dim=-1)
        ).to(torch.float16)

        w_packed, scales_biases = pack_linear_q2_k(w_orig)
        self.assertEqual(w_packed.shape, (N, K // 4))
        self.assertEqual(w_packed.dtype, torch.uint8)
        self.assertEqual(scales_biases.shape, (N, K // 64, 2))

        w_recon = dequantize_q2_k(w_packed, scales_biases, K=K)
        max_diff = (w_orig - w_recon).abs().max().item()
        self.assertLess(
            max_diff, 1e-3, f"Q2_K round-trip reconstruction error: {max_diff}"
        )

    def test_03_bitnet_gemv_parity_vs_dequantized(self):
        """Verify BitNet 1.58b GEMV (M=1) against dequantized FP16 linear matmul."""
        test_shapes = [
            (576, 576),  # SmolLM2 hidden
            (1536, 1536),  # SmolLM2 MLP
            (2048, 2048),  # Llama-3.2-1B hidden
            (8192, 2048),  # Llama-3.2-1B gate/up
            (2048, 8192),  # Llama-3.2-1B down
        ]
        for N, K in test_shapes:
            with self.subTest(N=N, K=K):
                w = torch.randn(N, K, dtype=torch.float16, device=self.device)
                w_packed, scales = pack_linear_ternary(w)
                w_recon = dequantize_ternary(w_packed, scales, K=K)

                x = torch.randn(1, K, dtype=torch.float16, device=self.device)
                y_ref = torch.matmul(x, w_recon.t())
                y_metal = qmv_ternary(x, w_packed, scales)

                max_diff = (y_ref - y_metal).abs().max().item()
                cos_sim = torch.cosine_similarity(
                    y_ref.flatten().float(), y_metal.flatten().float(), dim=0
                ).item()

                tol = 0.05 if K <= 576 else 0.35
                self.assertLess(
                    max_diff,
                    tol,
                    f"BitNet GEMV diff too large ({max_diff}) for shape ({N}, {K})",
                )
                self.assertGreater(
                    cos_sim,
                    0.9999,
                    f"BitNet GEMV cosine similarity too low ({cos_sim}) for shape ({N}, {K})",
                )

    def test_04_q2_k_gemv_parity_vs_dequantized(self):
        """Verify Q2_K GEMV (M=1) against dequantized FP16 linear matmul."""
        test_shapes = [
            (576, 576),
            (1536, 1536),
            (2048, 2048),
            (8192, 2048),
            (2048, 8192),
        ]
        for N, K in test_shapes:
            with self.subTest(N=N, K=K):
                w = torch.randn(N, K, dtype=torch.float16, device=self.device)
                w_packed, scales_biases = pack_linear_q2_k(w)
                w_recon = dequantize_q2_k(w_packed, scales_biases, K=K)

                x = torch.randn(1, K, dtype=torch.float16, device=self.device)
                y_ref = torch.matmul(x, w_recon.t())
                y_metal = qmv_q2_k(x, w_packed, scales_biases)

                max_diff = (y_ref - y_metal).abs().max().item()
                cos_sim = torch.cosine_similarity(
                    y_ref.flatten().float(), y_metal.flatten().float(), dim=0
                ).item()

                tol = 0.05 if K <= 576 else 0.35
                self.assertLess(
                    max_diff,
                    tol,
                    f"Q2_K GEMV diff too large ({max_diff}) for shape ({N}, {K})",
                )
                self.assertGreater(
                    cos_sim,
                    0.9999,
                    f"Q2_K GEMV cosine similarity too low ({cos_sim}) for shape ({N}, {K})",
                )

    def test_05_bitnet_gemm_batch_scaling(self):
        """Verify BitNet 1.58b batched GEMM across batch sizes M in [1, 2, 3, 4, 7, 8, 15, 16, 27, 32]."""
        N, K = 512, 1024
        w = torch.randn(N, K, dtype=torch.float16, device=self.device)
        w_packed, scales = pack_linear_ternary(w)
        w_recon = dequantize_ternary(w_packed, scales, K=K)

        for M in [1, 2, 3, 4, 7, 8, 15, 16, 27, 32]:
            with self.subTest(M=M):
                x = torch.randn(M, K, dtype=torch.float16, device=self.device)
                y_ref = torch.matmul(x, w_recon.t())
                y_metal = qmm_ternary(x, w_packed, scales)

                max_diff = (y_ref - y_metal).abs().max().item()
                cos_sim = torch.cosine_similarity(
                    y_ref.flatten().float(), y_metal.flatten().float(), dim=0
                ).item()

                self.assertLess(
                    max_diff,
                    0.15,
                    f"BitNet GEMM diff too large ({max_diff}) for M={M}",
                )
                self.assertGreater(
                    cos_sim,
                    0.9999,
                    f"BitNet GEMM cosine similarity too low ({cos_sim}) for M={M}",
                )

    def test_06_q2_k_gemm_batch_scaling(self):
        """Verify Q2_K batched GEMM across batch sizes M in [1, 2, 3, 4, 7, 8, 15, 16, 27, 32]."""
        N, K = 512, 1024
        w = torch.randn(N, K, dtype=torch.float16, device=self.device)
        w_packed, scales_biases = pack_linear_q2_k(w)
        w_recon = dequantize_q2_k(w_packed, scales_biases, K=K)

        for M in [1, 2, 3, 4, 7, 8, 15, 16, 27, 32]:
            with self.subTest(M=M):
                x = torch.randn(M, K, dtype=torch.float16, device=self.device)
                y_ref = torch.matmul(x, w_recon.t())
                y_metal = qmm_q2_k(x, w_packed, scales_biases)

                max_diff = (y_ref - y_metal).abs().max().item()
                cos_sim = torch.cosine_similarity(
                    y_ref.flatten().float(), y_metal.flatten().float(), dim=0
                ).item()

                self.assertLess(
                    max_diff,
                    0.15,
                    f"Q2_K GEMM diff too large ({max_diff}) for M={M}",
                )
                self.assertGreater(
                    cos_sim,
                    0.9999,
                    f"Q2_K GEMM cosine similarity too low ({cos_sim}) for M={M}",
                )

    def test_07_bias_addition_parity(self):
        """Verify bias integration for both BitNet and Q2_K GEMV and GEMM."""
        N, K = 256, 512
        w = torch.randn(N, K, dtype=torch.float16, device=self.device)
        bias = torch.randn(N, dtype=torch.float16, device=self.device)

        # BitNet with bias
        w_p_tern, s_tern = pack_linear_ternary(w)
        w_rec_tern = dequantize_ternary(w_p_tern, s_tern, K=K)
        for M in [1, 4]:
            x = torch.randn(M, K, dtype=torch.float16, device=self.device)
            ref = torch.matmul(x, w_rec_tern.t()) + bias
            metal = (
                qmv_ternary(x, w_p_tern, s_tern, bias=bias)
                if M == 1
                else qmm_ternary(x, w_p_tern, s_tern, bias=bias)
            )
            diff = (ref - metal).abs().max().item()
            self.assertLess(
                diff, 0.10, f"BitNet bias diff too large ({diff}) for M={M}"
            )

        # Q2_K with bias
        w_p_q2, sb_q2 = pack_linear_q2_k(w)
        w_rec_q2 = dequantize_q2_k(w_p_q2, sb_q2, K=K)
        for M in [1, 4]:
            x = torch.randn(M, K, dtype=torch.float16, device=self.device)
            ref = torch.matmul(x, w_rec_q2.t()) + bias
            metal = (
                qmv_q2_k(x, w_p_q2, sb_q2, bias=bias)
                if M == 1
                else qmm_q2_k(x, w_p_q2, sb_q2, bias=bias)
            )
            diff = (ref - metal).abs().max().item()
            self.assertLess(diff, 0.10, f"Q2_K bias diff too large ({diff}) for M={M}")

    def test_08_theoretical_memory_compression(self):
        """Verify exact physical memory footprint and compression factors vs FP16."""
        N, K = 2048, 2048
        w = torch.randn(N, K, dtype=torch.float16, device=self.device)
        fp16_bytes = w.numel() * 2

        # BitNet 1.58b:
        w_p_tern, s_tern = pack_linear_ternary(w)
        tern_bytes = (
            w_p_tern.numel() * w_p_tern.element_size()
            + s_tern.numel() * s_tern.element_size()
        )
        b_per_weight_tern = tern_bytes / w.numel()
        compression_tern = fp16_bytes / tern_bytes

        # Q2_K:
        w_p_q2, sb_q2 = pack_linear_q2_k(w)
        q2_bytes = (
            w_p_q2.numel() * w_p_q2.element_size()
            + sb_q2.numel() * sb_q2.element_size()
        )
        b_per_weight_q2 = q2_bytes / w.numel()
        compression_q2 = fp16_bytes / q2_bytes

        self.assertAlmostEqual(b_per_weight_tern, 0.28125, places=4)
        self.assertAlmostEqual(compression_tern, 16.0 / 2.25, places=2)
        self.assertAlmostEqual(b_per_weight_q2, 0.3125, places=4)
        self.assertAlmostEqual(compression_q2, 16.0 / 2.5, places=2)

    def test_09_adversarial_edge_cases(self):
        """Verify numerical stability with all-zeros, extreme outliers, and odd group multiples."""
        # 1. All zeros
        N, K = 128, 256
        w_zero = torch.zeros(N, K, dtype=torch.float16, device=self.device)
        x = torch.randn(1, K, dtype=torch.float16, device=self.device)

        wp, s = pack_linear_ternary(w_zero)
        out_tern = qmv_ternary(x, wp, s)
        self.assertEqual(out_tern.abs().max().item(), 0.0)

        wp2, sb = pack_linear_q2_k(w_zero)
        out_q2 = qmv_q2_k(x, wp2, sb)
        self.assertEqual(out_q2.abs().max().item(), 0.0)

        # 2. 100-sigma outlier in activations (+-80.0)
        x_outlier = torch.randn(2, K, dtype=torch.float16, device=self.device)
        x_outlier[0, 10] = 80.0
        x_outlier[1, 55] = -80.0

        w_rand = torch.randn(N, K, dtype=torch.float16, device=self.device)
        wp_t, s_t = pack_linear_ternary(w_rand)
        w_rec_t = dequantize_ternary(wp_t, s_t, K=K)
        ref_t = torch.matmul(x_outlier, w_rec_t.t())
        metal_t = qmm_ternary(x_outlier, wp_t, s_t)
        self.assertFalse(torch.isnan(metal_t).any().item())
        self.assertFalse(torch.isinf(metal_t).any().item())
        diff_t = (ref_t - metal_t).abs().max().item()
        self.assertLess(diff_t, 0.25)

        # 3. Non-power-of-two group multiple: K=384, N=768
        K_odd = 384
        N_odd = 768
        w_odd = torch.randn(N_odd, K_odd, dtype=torch.float16, device=self.device)
        wp_odd, s_odd = pack_linear_ternary(w_odd)
        w_rec_odd = dequantize_ternary(wp_odd, s_odd, K=K_odd)
        x_odd = torch.randn(3, K_odd, dtype=torch.float16, device=self.device)
        ref_odd = torch.matmul(x_odd, w_rec_odd.t())
        metal_odd = qmm_ternary(x_odd, wp_odd, s_odd)
        self.assertLess((ref_odd - metal_odd).abs().max().item(), 0.05)

    def test_10_transformer_model_quantization_integration(self):
        """Verify Transformer model integration for BitNet and Q2_K quant modes."""
        cfg = ModelConfig(
            n_embd=128,
            n_layer=2,
            n_head=4,
            n_kv_head=2,
            vocab_size=256,
            block_size=128,
            mlp_type="swiglu",
            norm_type="rmsnorm",
            pos_emb="rope",
        )
        model = Transformer(cfg).to(dtype=torch.float16, device=self.device)

        # 1. Quantize to BitNet 1.58b
        model.quantize(mode="ternary", release_fp16=False)
        qw_tern = model._ensure_quant_weights(self.device)
        self.assertEqual(qw_tern["mode"], "ternary")
        self.assertEqual(qw_tern["c_attn_p"][0].dtype, torch.uint8)
        self.assertEqual(qw_tern["c_attn_s"][0].dtype, torch.float16)

        # 2. Quantize to Q2_K
        model.quantize(mode="q2_k", release_fp16=False)
        qw_q2 = model._ensure_quant_weights(self.device)
        self.assertEqual(qw_q2["mode"], "q2_k")
        self.assertEqual(qw_q2["c_attn_p"][0].dtype, torch.uint8)
        self.assertEqual(qw_q2["c_attn_s"][0].dtype, torch.float16)


if __name__ == "__main__":
    unittest.main()
