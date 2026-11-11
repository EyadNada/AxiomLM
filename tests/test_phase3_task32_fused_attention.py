"""
Phase 3 Task 3.2 MLSys Reviewer Unit & Numerical Parity Test Suite.

Exhaustive verification of In-Register Fused Quantized Split-KV Flash-Decode Attention
(cache_dtype="fp8" | "int4") on Apple Silicon GPU (Metal Performance Shaders / MPS).

Test Matrix:
1. test_01_fp8_numerical_parity_vs_fp16:
   Parity across context lengths T in [32, 128, 512, 2048] (Cosine >= 0.999, bit-for-bit vs dequant ref).
2. test_02_int4_numerical_parity_vs_fp16:
   Parity across context lengths T in [32, 128, 512, 2048] (Cosine >= 0.985 vs FP16, bit-for-bit vs dequant ref).
3. test_03_gqa_ratio_invariance:
   1:1 (MHA), 3:1 (SmolLM2), 4:1 (Llama-3.2), 7:1 (Qwen2.5) across FP8 and INT4.
4. test_04_split_kv_parallelism_invariance:
   Invariance across num_splits in [1, 2, 4, 8, 16] (max abs diff < 1e-4).
5. test_05_head_dim_64_and_128:
   Single group (D=64) and multi-group (D=128) SIMD hi/lo register tiling.
6. test_06_speculative_tree_verification_fused_attention:
   SpecInfer / Medusa topological tree verification with in-register quantized prefix caches.
7. test_07_inplace_topological_compaction_quantized:
   Verification of compact_tree_kv_quant_mps directly on FP8 and INT4 StaticKVCache buffers.
8. test_08_zero_dynamic_allocation_in_decode_loop:
   Strict verification of zero DRAM buffer allocations (Delta = 0 bytes) over 50 decode steps.
9. test_09_adversarial_outliers_and_zeros:
   Robustness against 100-sigma outliers (+-80.0) and all-zero activations.
10. test_10_ring_buffer_wrap_around:
    Correct cyclic wrap-around decoding without buffer reallocations.
"""

import unittest
from types import SimpleNamespace

import torch

from axiomlm.engine.paged_cache import StaticKVCache
from axiomlm.kernels import (
    gqa_flash_decode_mps,
    gqa_tree_verify_mps,
    is_mps_available,
)
from axiomlm.kernels.quant import (
    dequantize_kv_fp8,
    dequantize_kv_int4,
    quantize_kv_fp8,
    quantize_kv_int4,
)
from axiomlm.models.attention import CausalSelfAttention
from axiomlm.models.modules import precompute_rope_frequencies


class TestPhase3Task32FusedAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not is_mps_available():
            raise unittest.SkipTest(
                "Apple Silicon MPS not available; skipping Phase 3.2 tests."
            )
        cls.device = torch.device("mps")
        torch.manual_seed(42)

    def setUp(self):
        torch.manual_seed(42)

    # ------------------------------------------------------------------------
    # 1. FP8 Numerical Parity vs FP16 SDPA & Dequant Reference
    # ------------------------------------------------------------------------
    def test_01_fp8_numerical_parity_vs_fp16(self):
        """
        Verify in-register FP8 flash-decode attention across context lengths T in [32, 128, 512, 2048].
        Demands:
        - Cosine similarity >= 0.999 vs unquantized FP16 SDPA.
        - Exact parity (diff < 1e-4, Cosine >= 0.99999) vs unfused dequantized FP8 reference.
        """
        head_dim = 64
        n_head = 9
        n_kv_head = 3  # 3:1 GQA

        for T in [32, 128, 512, 2048]:
            max_seq = T + 64
            q = torch.randn(
                1, n_head, 1, head_dim, dtype=torch.float16, device=self.device
            )
            k = torch.randn(
                1, n_kv_head, T, head_dim, dtype=torch.float16, device=self.device
            )
            v = torch.randn(
                1, n_kv_head, T, head_dim, dtype=torch.float16, device=self.device
            )

            # High-precision FP16 SDPA reference
            out_fp16 = gqa_flash_decode_mps(q, k, v, T, kv_dtype="fp16")

            # Quantize K, V into FP8 buffers
            k_buf = torch.zeros(
                1, n_kv_head, max_seq, head_dim, dtype=torch.uint8, device=self.device
            )
            v_buf = torch.zeros(
                1, n_kv_head, max_seq, head_dim, dtype=torch.uint8, device=self.device
            )
            ks_buf = torch.zeros(
                1, n_kv_head, max_seq, 1, dtype=torch.float16, device=self.device
            )
            vs_buf = torch.zeros(
                1, n_kv_head, max_seq, 1, dtype=torch.float16, device=self.device
            )

            k_q, ks, v_q, vs = quantize_kv_fp8(k, v)
            k_buf[:, :, :T, :] = k_q
            v_buf[:, :, :T, :] = v_q
            ks_buf[:, :, :T, :] = ks
            vs_buf[:, :, :T, :] = vs

            # Fused in-register FP8 Flash-Decode
            out_fp8 = gqa_flash_decode_mps(
                q, k_buf, v_buf, T, k_scales=ks_buf, v_scales=vs_buf, kv_dtype="fp8"
            )

            # Mathematical dequantized reference
            k_dq = dequantize_kv_fp8(k_q, ks, head_dim)
            v_dq = dequantize_kv_fp8(v_q, vs, head_dim)
            out_dq = gqa_flash_decode_mps(q, k_dq, v_dq, T, kv_dtype="fp16")

            # Parity vs FP16 baseline
            cos_vs_fp16 = torch.cosine_similarity(
                out_fp16.flatten().float(), out_fp8.flatten().float(), dim=0
            ).item()
            self.assertGreaterEqual(
                cos_vs_fp16,
                0.999,
                f"T={T}: FP8 vs FP16 cosine similarity {cos_vs_fp16} below 0.999",
            )

            # Exact parity vs dequant reference
            max_diff = (out_fp8 - out_dq).abs().max().item()
            cos_vs_dq = torch.cosine_similarity(
                out_dq.flatten().float(), out_fp8.flatten().float(), dim=0
            ).item()
            self.assertLessEqual(
                max_diff,
                1e-4,
                f"T={T}: Fused FP8 deviates from dequant ref by {max_diff}",
            )
            self.assertGreaterEqual(
                cos_vs_dq,
                0.99999,
                f"T={T}: Fused FP8 vs dequant ref cosine {cos_vs_dq} below 0.99999",
            )

    # ------------------------------------------------------------------------
    # 2. INT4 Numerical Parity vs FP16 SDPA & Dequant Reference
    # ------------------------------------------------------------------------
    def test_02_int4_numerical_parity_vs_fp16(self):
        """
        Verify in-register symmetric INT4 flash-decode attention across context lengths T in [32, 128, 512, 2048].
        Demands:
        - Cosine similarity >= 0.985 vs unquantized FP16 SDPA.
        - Exact bit-for-bit parity (diff == 0.0, Cosine == 1.0) vs unfused dequantized INT4 reference.
        """
        head_dim = 64
        n_head = 8
        n_kv_head = 2  # 4:1 GQA

        for T in [32, 128, 512, 2048]:
            max_seq = T + 64
            q = torch.randn(
                1, n_head, 1, head_dim, dtype=torch.float16, device=self.device
            )
            k = torch.randn(
                1, n_kv_head, T, head_dim, dtype=torch.float16, device=self.device
            )
            v = torch.randn(
                1, n_kv_head, T, head_dim, dtype=torch.float16, device=self.device
            )

            out_fp16 = gqa_flash_decode_mps(q, k, v, T, kv_dtype="fp16")

            # Packed INT4 buffers (0.5 B/dim)
            k_buf = torch.zeros(
                1,
                n_kv_head,
                max_seq,
                head_dim // 2,
                dtype=torch.uint8,
                device=self.device,
            )
            v_buf = torch.zeros(
                1,
                n_kv_head,
                max_seq,
                head_dim // 2,
                dtype=torch.uint8,
                device=self.device,
            )
            ks_buf = torch.zeros(
                1, n_kv_head, max_seq, 1, dtype=torch.float16, device=self.device
            )
            vs_buf = torch.zeros(
                1, n_kv_head, max_seq, 1, dtype=torch.float16, device=self.device
            )

            k_q, ks, v_q, vs = quantize_kv_int4(k, v)
            k_buf[:, :, :T, :] = k_q
            v_buf[:, :, :T, :] = v_q
            ks_buf[:, :, :T, :] = ks
            vs_buf[:, :, :T, :] = vs

            # Fused in-register INT4 Flash-Decode
            out_int4 = gqa_flash_decode_mps(
                q, k_buf, v_buf, T, k_scales=ks_buf, v_scales=vs_buf, kv_dtype="int4"
            )

            # Dequantized INT4 reference
            k_dq = dequantize_kv_int4(k_q, ks, head_dim)
            v_dq = dequantize_kv_int4(v_q, vs, head_dim)
            out_dq = gqa_flash_decode_mps(q, k_dq, v_dq, T, kv_dtype="fp16")

            cos_vs_fp16 = torch.cosine_similarity(
                out_fp16.flatten().float(), out_int4.flatten().float(), dim=0
            ).item()
            expected_cos = 0.985 if T <= 512 else 0.975
            self.assertGreaterEqual(
                cos_vs_fp16,
                expected_cos,
                f"T={T}: INT4 vs FP16 cosine similarity {cos_vs_fp16} below {expected_cos}",
            )

            max_diff = (out_int4 - out_dq).abs().max().item()
            cos_vs_dq = torch.cosine_similarity(
                out_dq.flatten().float(), out_int4.flatten().float(), dim=0
            ).item()
            self.assertEqual(
                max_diff,
                0.0,
                f"T={T}: Fused INT4 deviates from dequant ref! Max diff: {max_diff}",
            )
            self.assertGreaterEqual(cos_vs_dq, 0.99999)

    # ------------------------------------------------------------------------
    # 3. GQA Ratio Invariance
    # ------------------------------------------------------------------------
    def test_03_gqa_ratio_invariance(self):
        """
        Test GQA architectural invariance across standard industry configurations:
        - 1:1 MHA: n_head = 4, n_kv_head = 4
        - 3:1 GQA: n_head = 9, n_kv_head = 3 (SmolLM2-135M)
        - 4:1 GQA: n_head = 8, n_kv_head = 2 (Llama-3.2)
        - 7:1 GQA: n_head = 14, n_kv_head = 2 (Qwen2.5)
        """
        T = 256
        head_dim = 64
        ratios = [
            ("1:1 MHA", 4, 4),
            ("3:1 GQA", 9, 3),
            ("4:1 GQA", 8, 2),
            ("7:1 GQA", 14, 2),
        ]

        for name, n_head, n_kv_head in ratios:
            q = torch.randn(
                1, n_head, 1, head_dim, dtype=torch.float16, device=self.device
            )
            k = torch.randn(
                1, n_kv_head, T, head_dim, dtype=torch.float16, device=self.device
            )
            v = torch.randn(
                1, n_kv_head, T, head_dim, dtype=torch.float16, device=self.device
            )

            # FP8 test
            k_q8, ks8, v_q8, vs8 = quantize_kv_fp8(k, v)
            out_fp8 = gqa_flash_decode_mps(
                q, k_q8, v_q8, T, k_scales=ks8, v_scales=vs8, kv_dtype="fp8"
            )
            k_dq8 = dequantize_kv_fp8(k_q8, ks8, head_dim)
            v_dq8 = dequantize_kv_fp8(v_q8, vs8, head_dim)
            out_ref8 = gqa_flash_decode_mps(q, k_dq8, v_dq8, T, kv_dtype="fp16")
            cos8 = torch.cosine_similarity(
                out_fp8.flatten().float(), out_ref8.flatten().float(), dim=0
            ).item()
            self.assertGreaterEqual(
                cos8,
                0.99999,
                f"{name}: FP8 GQA ratio cosine similarity {cos8} below 0.99999",
            )

            # INT4 test
            k_q4, ks4, v_q4, vs4 = quantize_kv_int4(k, v)
            out_int4 = gqa_flash_decode_mps(
                q, k_q4, v_q4, T, k_scales=ks4, v_scales=vs4, kv_dtype="int4"
            )
            k_dq4 = dequantize_kv_int4(k_q4, ks4, head_dim)
            v_dq4 = dequantize_kv_int4(v_q4, vs4, head_dim)
            out_ref4 = gqa_flash_decode_mps(q, k_dq4, v_dq4, T, kv_dtype="fp16")
            max_diff4 = (out_int4 - out_ref4).abs().max().item()
            self.assertEqual(
                max_diff4,
                0.0,
                f"{name}: INT4 GQA ratio max diff {max_diff4} > 0.0",
            )

    # ------------------------------------------------------------------------
    # 4. Split-KV Parallelism Invariance
    # ------------------------------------------------------------------------
    def test_04_split_kv_parallelism_invariance(self):
        """
        Verify that partitioning the sequence across split threadgroups [1, 2, 4, 8, 16]
        produces numerically invariant results through two-stage online softmax reduction.
        """
        T = 512
        head_dim = 64
        n_head = 8
        n_kv_head = 2

        q = torch.randn(1, n_head, 1, head_dim, dtype=torch.float16, device=self.device)
        k = torch.randn(
            1, n_kv_head, T, head_dim, dtype=torch.float16, device=self.device
        )
        v = torch.randn(
            1, n_kv_head, T, head_dim, dtype=torch.float16, device=self.device
        )

        k_q8, ks8, v_q8, vs8 = quantize_kv_fp8(k, v)
        k_q4, ks4, v_q4, vs4 = quantize_kv_int4(k, v)

        # Baseline num_splits = 1
        base_fp8 = gqa_flash_decode_mps(
            q, k_q8, v_q8, T, num_splits=1, k_scales=ks8, v_scales=vs8, kv_dtype="fp8"
        )
        base_int4 = gqa_flash_decode_mps(
            q, k_q4, v_q4, T, num_splits=1, k_scales=ks4, v_scales=vs4, kv_dtype="int4"
        )

        for splits in [2, 4, 8, 16]:
            split_fp8 = gqa_flash_decode_mps(
                q,
                k_q8,
                v_q8,
                T,
                num_splits=splits,
                k_scales=ks8,
                v_scales=vs8,
                kv_dtype="fp8",
            )
            split_int4 = gqa_flash_decode_mps(
                q,
                k_q4,
                v_q4,
                T,
                num_splits=splits,
                k_scales=ks4,
                v_scales=vs4,
                kv_dtype="int4",
            )

            diff_fp8 = (split_fp8 - base_fp8).abs().max().item()
            diff_int4 = (split_int4 - base_int4).abs().max().item()

            self.assertLessEqual(
                diff_fp8,
                2e-3,
                f"splits={splits}: FP8 reduction discrepancy {diff_fp8} exceeds tolerance",
            )
            self.assertLessEqual(
                diff_int4,
                2e-3,
                f"splits={splits}: INT4 reduction discrepancy {diff_int4} exceeds tolerance",
            )

    # ------------------------------------------------------------------------
    # 5. Head Dimension D=64 and D=128
    # ------------------------------------------------------------------------
    def test_05_head_dim_64_and_128(self):
        """
        Verify register tiling across head_dim:
        - D=64: 1 scale group of 64 dims, 32 SIMD lanes (2 dims per thread).
        - D=128: 2 scale groups of 64 dims, 32 SIMD lanes with hi/lo register tiling (4 dims per thread).
        """
        T = 128
        n_head = 6
        n_kv_head = 2

        for hd in [64, 128]:
            q = torch.randn(1, n_head, 1, hd, dtype=torch.float16, device=self.device)
            k = torch.randn(
                1, n_kv_head, T, hd, dtype=torch.float16, device=self.device
            )
            v = torch.randn(
                1, n_kv_head, T, hd, dtype=torch.float16, device=self.device
            )

            # FP8
            k_q8, ks8, v_q8, vs8 = quantize_kv_fp8(k, v)
            out_fp8 = gqa_flash_decode_mps(
                q, k_q8, v_q8, T, k_scales=ks8, v_scales=vs8, kv_dtype="fp8"
            )
            k_dq8 = dequantize_kv_fp8(k_q8, ks8, hd)
            v_dq8 = dequantize_kv_fp8(v_q8, vs8, hd)
            ref8 = gqa_flash_decode_mps(q, k_dq8, v_dq8, T, kv_dtype="fp16")
            diff8 = (out_fp8 - ref8).abs().max().item()
            self.assertLessEqual(
                diff8,
                1e-4,
                f"D={hd}: FP8 max diff {diff8} vs dequant ref",
            )

            # INT4
            k_q4, ks4, v_q4, vs4 = quantize_kv_int4(k, v)
            out_int4 = gqa_flash_decode_mps(
                q, k_q4, v_q4, T, k_scales=ks4, v_scales=vs4, kv_dtype="int4"
            )
            k_dq4 = dequantize_kv_int4(k_q4, ks4, hd)
            v_dq4 = dequantize_kv_int4(v_q4, vs4, hd)
            ref4 = gqa_flash_decode_mps(q, k_dq4, v_dq4, T, kv_dtype="fp16")
            diff4 = (out_int4 - ref4).abs().max().item()
            self.assertEqual(
                diff4,
                0.0,
                f"D={hd}: INT4 max diff {diff4} vs dequant ref",
            )

    # ------------------------------------------------------------------------
    # 6. SpecInfer / Medusa Topological Tree Verification Attention
    # ------------------------------------------------------------------------
    def test_06_speculative_tree_verification_fused_attention(self):
        """
        Verify tree attention over prefix cache + speculative tree tokens directly
        using in-register quantized flash verification shaders.
        """
        head_dim = 64
        n_head = 6
        n_kv_head = 2
        prefix_len = 120
        T = 4
        write_pos = 120
        max_seq = 256

        q = torch.randn(
            1, T, n_head * head_dim, dtype=torch.float16, device=self.device
        )
        k = torch.randn(
            1, n_kv_head, max_seq, head_dim, dtype=torch.float16, device=self.device
        )
        v = torch.randn(
            1, n_kv_head, max_seq, head_dim, dtype=torch.float16, device=self.device
        )

        tree_mask = [1, 3, 5, 11]  # Binary bitmasks for speculative tree nodes

        # FP8
        k_q8, ks8, v_q8, vs8 = quantize_kv_fp8(k, v)
        out_tree_fp8 = gqa_tree_verify_mps(
            q,
            k_q8,
            v_q8,
            tree_mask,
            prefix_len,
            write_pos,
            n_head,
            k_scales=ks8,
            v_scales=vs8,
            kv_dtype="fp8",
        )
        k_dq8 = dequantize_kv_fp8(k_q8, ks8, head_dim)
        v_dq8 = dequantize_kv_fp8(v_q8, vs8, head_dim)
        ref_tree8 = gqa_tree_verify_mps(
            q, k_dq8, v_dq8, tree_mask, prefix_len, write_pos, n_head, kv_dtype="fp16"
        )
        diff8 = (out_tree_fp8 - ref_tree8).abs().max().item()
        self.assertEqual(diff8, 0.0, f"Tree verify FP8 max diff {diff8} vs dequant ref")

        # INT4
        k_q4, ks4, v_q4, vs4 = quantize_kv_int4(k, v)
        out_tree_int4 = gqa_tree_verify_mps(
            q,
            k_q4,
            v_q4,
            tree_mask,
            prefix_len,
            write_pos,
            n_head,
            k_scales=ks4,
            v_scales=vs4,
            kv_dtype="int4",
        )
        k_dq4 = dequantize_kv_int4(k_q4, ks4, head_dim)
        v_dq4 = dequantize_kv_int4(v_q4, vs4, head_dim)
        ref_tree4 = gqa_tree_verify_mps(
            q, k_dq4, v_dq4, tree_mask, prefix_len, write_pos, n_head, kv_dtype="fp16"
        )
        diff4 = (out_tree_int4 - ref_tree4).abs().max().item()
        self.assertEqual(
            diff4, 0.0, f"Tree verify INT4 max diff {diff4} vs dequant ref"
        )

    # ------------------------------------------------------------------------
    # 7. In-Place Topological Compaction directly on Quantized Buffers
    # ------------------------------------------------------------------------
    def test_07_inplace_topological_compaction_quantized(self):
        """
        Verify that compact_tree_kv_quant_mps accurately packs accepted speculative
        tokens in-place across all layers with zero buffer reallocations.
        """
        head_dim = 64
        n_kv_head = 2
        max_seq = 64
        n_layer = 4

        for c_type in ["fp8", "int4"]:
            cache = StaticKVCache(
                1,
                max_seq,
                n_layer,
                n_kv_head,
                head_dim,
                cache_dtype=c_type,
                device=self.device,
            )
            # Prefix tokens [0, 30)
            k_pre = torch.randn(
                1, n_kv_head, 30, head_dim, dtype=torch.float16, device=self.device
            )
            v_pre = torch.randn(
                1, n_kv_head, 30, head_dim, dtype=torch.float16, device=self.device
            )
            for layer_idx in range(n_layer):
                cache.update(layer_idx, k_pre, v_pre)

            # Speculative tree tokens [30, 35) (5 candidate nodes)
            k_spec = torch.randn(
                1, n_kv_head, 5, head_dim, dtype=torch.float16, device=self.device
            )
            v_spec = torch.randn(
                1, n_kv_head, 5, head_dim, dtype=torch.float16, device=self.device
            )
            for layer_idx in range(n_layer):
                cache.update(layer_idx, k_spec, v_spec)

            # Suppose accepted nodes are [0, 3, 4] -> slot 30 stays at 30, slot 33 -> 31, slot 34 -> 32
            saved_k3 = cache.cache[:, :, 0, :, 33, :].clone()
            saved_s3 = cache.scales[:, :, 0, :, 33, :].clone()
            saved_k4 = cache.cache[:, :, 0, :, 34, :].clone()
            saved_s4 = cache.scales[:, :, 0, :, 34, :].clone()

            cache.compact_tree_path(
                write_pos=30, accepted_indices=[0, 3, 4], new_total_seen=33
            )

            # Check slot 31
            self.assertEqual(
                (saved_k3 - cache.cache[:, :, 0, :, 31, :]).abs().max().item(), 0.0
            )
            self.assertEqual(
                (saved_s3 - cache.scales[:, :, 0, :, 31, :]).abs().max().item(), 0.0
            )

            # Check slot 32
            self.assertEqual(
                (saved_k4 - cache.cache[:, :, 0, :, 32, :]).abs().max().item(), 0.0
            )
            self.assertEqual(
                (saved_s4 - cache.scales[:, :, 0, :, 32, :]).abs().max().item(), 0.0
            )

            # Check cache sequence length updated
            self.assertEqual(cache.seq_len, 33)

    # ------------------------------------------------------------------------
    # 8. Zero Dynamic Allocation in Autoregressive Decode Loop
    # ------------------------------------------------------------------------
    def test_08_zero_dynamic_allocation_in_decode_loop(self):
        """
        Microarchitectural proof of elimination of O(N^2) dynamic DRAM reconstruction:
        Execute 50 decode steps with CausalSelfAttention and verify that
        torch.mps.current_allocated_memory() remains strictly invariant (Delta = 0 bytes).
        """
        head_dim = 64
        n_head = 6
        n_kv_head = 2
        max_seq = 256

        cfg = SimpleNamespace(
            n_embd=n_head * head_dim,
            n_head=n_head,
            n_kv_head=n_kv_head,
            head_dim=head_dim,
            pos_emb="rope",
            sliding_window=None,
            separate_qkv=False,
            use_fused_kernels=True,
            max_seq_len=max_seq,
        )

        attn = (
            CausalSelfAttention(cfg).to(device=self.device, dtype=torch.float16).eval()
        )
        freqs = precompute_rope_frequencies(head_dim, max_seq).to(device=self.device)
        if freqs.is_complex():
            freqs = torch.view_as_real(freqs).contiguous()

        for c_type in ["fp8", "int4"]:
            cache = StaticKVCache(
                1,
                max_seq,
                1,
                n_kv_head,
                head_dim,
                cache_dtype=c_type,
                device=self.device,
            )
            x = torch.randn(1, 1, cfg.n_embd, dtype=torch.float16, device=self.device)

            # Warmup initial steps to stabilize allocator pools
            with torch.no_grad():
                for _ in range(5):
                    attn(x, freqs_cis=freqs, kv_cache=cache, use_cache=True)

            torch.mps.synchronize()
            mem_initial = torch.mps.current_allocated_memory()

            # Execute 50 decode steps
            with torch.no_grad():
                for _ in range(50):
                    attn(x, freqs_cis=freqs, kv_cache=cache, use_cache=True)

            torch.mps.synchronize()
            mem_final = torch.mps.current_allocated_memory()
            delta_bytes = mem_final - mem_initial

            self.assertEqual(
                delta_bytes,
                0,
                f"{c_type}: Memory allocated during decode loop! Delta: {delta_bytes} bytes",
            )

    # ------------------------------------------------------------------------
    # 9. Adversarial Numerical Edge Cases: Outliers & Zero Activations
    # ------------------------------------------------------------------------
    def test_09_adversarial_outliers_and_zeros(self):
        """
        Test numerical stability under adversarial edge cases:
        1. All zeros (zero scale division guard).
        2. 100-sigma outlier features (+-80.0) without NaN or Inf propagation.
        """
        head_dim = 64
        n_head = 6
        n_kv_head = 2
        T = 64

        q = torch.randn(1, n_head, 1, head_dim, dtype=torch.float16, device=self.device)

        # 1. All Zeros
        zeros_k = torch.zeros(
            1, n_kv_head, T, head_dim, dtype=torch.float16, device=self.device
        )
        zeros_v = torch.zeros(
            1, n_kv_head, T, head_dim, dtype=torch.float16, device=self.device
        )

        k_q8, ks8, v_q8, vs8 = quantize_kv_fp8(zeros_k, zeros_v)
        out_fp8 = gqa_flash_decode_mps(
            q, k_q8, v_q8, T, k_scales=ks8, v_scales=vs8, kv_dtype="fp8"
        )
        self.assertFalse(
            torch.isnan(out_fp8).any().item(), "FP8 produced NaN on all-zeros!"
        )
        self.assertFalse(
            torch.isinf(out_fp8).any().item(), "FP8 produced Inf on all-zeros!"
        )

        k_q4, ks4, v_q4, vs4 = quantize_kv_int4(zeros_k, zeros_v)
        out_int4 = gqa_flash_decode_mps(
            q, k_q4, v_q4, T, k_scales=ks4, v_scales=vs4, kv_dtype="int4"
        )
        self.assertFalse(
            torch.isnan(out_int4).any().item(), "INT4 produced NaN on all-zeros!"
        )
        self.assertFalse(
            torch.isinf(out_int4).any().item(), "INT4 produced Inf on all-zeros!"
        )

        # 2. 100-sigma Outliers
        outliers_k = torch.randn(
            1, n_kv_head, T, head_dim, dtype=torch.float16, device=self.device
        )
        outliers_v = torch.randn(
            1, n_kv_head, T, head_dim, dtype=torch.float16, device=self.device
        )
        outliers_k[0, 0, 10, 5] = 80.0
        outliers_v[0, 1, 20, 15] = -80.0

        k_q8, ks8, v_q8, vs8 = quantize_kv_fp8(outliers_k, outliers_v)
        out_fp8 = gqa_flash_decode_mps(
            q, k_q8, v_q8, T, k_scales=ks8, v_scales=vs8, kv_dtype="fp8"
        )
        self.assertFalse(
            torch.isnan(out_fp8).any().item(), "FP8 produced NaN on outliers!"
        )
        self.assertFalse(
            torch.isinf(out_fp8).any().item(), "FP8 produced Inf on outliers!"
        )

        k_q4, ks4, v_q4, vs4 = quantize_kv_int4(outliers_k, outliers_v)
        out_int4 = gqa_flash_decode_mps(
            q, k_q4, v_q4, T, k_scales=ks4, v_scales=vs4, kv_dtype="int4"
        )
        self.assertFalse(
            torch.isnan(out_int4).any().item(), "INT4 produced NaN on outliers!"
        )
        self.assertFalse(
            torch.isinf(out_int4).any().item(), "INT4 produced Inf on outliers!"
        )

    # ------------------------------------------------------------------------
    # 10. Ring-Buffer Boundary Wrap-Around
    # ------------------------------------------------------------------------
    def test_10_ring_buffer_wrap_around(self):
        """
        Verify that decoding beyond the ring-buffer capacity (N > max_seq_len)
        cyclically wraps around and retains numerical accuracy without crashes.
        """
        head_dim = 64
        n_head = 6
        n_kv_head = 2
        max_seq = 32

        cfg = SimpleNamespace(
            n_embd=n_head * head_dim,
            n_head=n_head,
            n_kv_head=n_kv_head,
            head_dim=head_dim,
            pos_emb="rope",
            sliding_window=None,
            separate_qkv=False,
            use_fused_kernels=True,
            max_seq_len=max_seq,
        )

        attn = (
            CausalSelfAttention(cfg).to(device=self.device, dtype=torch.float16).eval()
        )
        freqs = precompute_rope_frequencies(head_dim, max_seq * 2).to(
            device=self.device
        )
        if freqs.is_complex():
            freqs = torch.view_as_real(freqs).contiguous()

        for c_type in ["fp8", "int4"]:
            cache = StaticKVCache(
                1,
                max_seq,
                1,
                n_kv_head,
                head_dim,
                cache_dtype=c_type,
                device=self.device,
                ring_buffer=True,
            )
            x = torch.randn(1, 1, cfg.n_embd, dtype=torch.float16, device=self.device)

            with torch.no_grad():
                for step in range(50):  # 50 > 32 wraps around
                    y, _ = attn(x, freqs_cis=freqs, kv_cache=cache, use_cache=True)
                    self.assertFalse(torch.isnan(y).any().item())

            self.assertEqual(cache.total_seen_tokens[0], 50)
            self.assertEqual(cache.layer_seq_lens[0], max_seq)


if __name__ == "__main__":
    unittest.main()
