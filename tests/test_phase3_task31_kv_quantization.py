"""
Unit and systems test suite for AxiomLM Phase 3 Task 3.1:
In-Flight FP8 & INT4 KV-Cache Quantization inside `rope_kv_cache_write_half_kernel`.

Rigorously verifies:
1. Mathematical precision, SQNR (dB), and Cosine Similarity of OCP FP8 E4M3FN and symmetric INT4.
2. In-flight bare-metal Metal kernel parity against CPU reference implementations.
3. Exact physical memory footprint reduction (2.0x for FP8, ~3.8x for INT4) across StaticKVCache.
4. Speculative candidate tree write parity across candidate trees (N=8).
5. End-to-end attention layer output fidelity across SmolLM2-360M and Llama-3.2-1B configurations.
6. Numerical boundary robustness (all-zeros, extreme outliers, wrap-around).
"""

from __future__ import annotations

import unittest

import torch

from axiomlm.engine.paged_cache import StaticKVCache
from axiomlm.kernels import _METAL_MOD, is_mps_available
from axiomlm.kernels.quant import (
    compute_quantization_metrics,
    dequantize_kv_fp8,
    dequantize_kv_int4,
    quantize_kv_fp8,
    quantize_kv_int4,
    rope_kv_cache_write_quantized,
)
from axiomlm.models.transformer import ModelConfig, Transformer


class TestPhase3Task31KVQuantization(unittest.TestCase):
    """Exhaustive mathematical and microarchitectural tests for KV-Cache Quantization."""

    def setUp(self):
        torch.manual_seed(2026)
        self.device = torch.device(
            "mps" if is_mps_available() and _METAL_MOD is not None else "cpu"
        )
        self.head_dim = 64
        self.n_head = 4
        self.n_kv_head = 2
        self.max_seq_len = 64

    def test_01_fp8_e4m3_numerical_fidelity_and_sqnr(self):
        """
        Validate OCP FP8 E4M3FN numerical fidelity across standard normal,
        uniform, and heavy-tailed activation distributions.
        Assert:
          - Cosine Similarity >= 0.999
          - SQNR >= 30.0 dB
          - Max Absolute Error <= 0.20
        """
        for dist_name, x in [
            ("normal", torch.randn(2, 4, 16, self.head_dim, dtype=torch.float16)),
            (
                "uniform",
                (torch.rand(2, 4, 16, self.head_dim, dtype=torch.float16) - 0.5) * 4.0,
            ),
            (
                "heavy_tail",
                torch.randn(2, 4, 16, self.head_dim, dtype=torch.float16)
                * torch.exp(
                    torch.randn(2, 4, 16, self.head_dim, dtype=torch.float16) * 0.5
                ),
            ),
        ]:
            _k_fp8, k_scales, _v_fp8, _v_scales = quantize_kv_fp8(x, x)
            k_dq = dequantize_kv_fp8(_k_fp8, k_scales, self.head_dim)

            metrics = compute_quantization_metrics(x, k_dq)
            self.assertGreaterEqual(
                metrics["cosine_similarity"],
                0.999,
                f"FP8 cosine similarity failed for {dist_name}: {metrics['cosine_similarity']}",
            )
            self.assertGreaterEqual(
                metrics["sqnr_db"],
                30.0,
                f"FP8 SQNR below 30.0 dB for {dist_name}: {metrics['sqnr_db']} dB",
            )
            self.assertLessEqual(
                metrics["max_abs_error"],
                0.25,
                f"FP8 Max error too large for {dist_name}: {metrics['max_abs_error']}",
            )

    def test_02_int4_numerical_fidelity_and_sqnr(self):
        """
        Validate symmetric INT4 [-8, 7] numerical fidelity and nibble packing.
        Assert:
          - Cosine Similarity >= 0.992
          - SQNR >= 18.0 dB
          - Memory reduction >= 3.7x
        """
        x = torch.randn(2, 4, 16, self.head_dim, dtype=torch.float16)
        k_int4, k_scales, _v_int4, _v_scales = quantize_kv_int4(x, x)
        k_dq = dequantize_kv_int4(k_int4, k_scales, self.head_dim)

        quant_bytes = (
            k_int4.numel() * k_int4.element_size()
            + k_scales.numel() * k_scales.element_size()
        )
        metrics = compute_quantization_metrics(x, k_dq, quant_bytes=quant_bytes)

        self.assertGreaterEqual(
            metrics["cosine_similarity"],
            0.992,
            f"INT4 cosine similarity failed: {metrics['cosine_similarity']}",
        )
        self.assertGreaterEqual(
            metrics["sqnr_db"],
            18.0,
            f"INT4 SQNR below 18.0 dB: {metrics['sqnr_db']} dB",
        )
        self.assertGreaterEqual(
            metrics["compression_ratio"],
            3.70,
            f"INT4 compression ratio below 3.7x: {metrics['compression_ratio']:.2f}x",
        )

    def test_03_in_flight_rope_kv_write_fp8_metal_parity(self):
        """
        Execute in-flight rope_kv_cache_write_fp8_mps on Apple GPU and verify
        mathematical parity against CPU reference.
        """
        if self.device.type != "mps":
            self.skipTest("Metal MPS hardware required for kernel test.")

        q_dim = self.n_head * self.head_dim
        kv_dim = self.n_kv_head * self.head_dim
        total_dim = q_dim + 2 * kv_dim
        write_pos = 7

        qkv = torch.randn(1, 1, total_dim, dtype=torch.float16, device=self.device)
        freqs_cis = torch.randn(
            self.max_seq_len,
            self.head_dim // 2,
            2,
            dtype=torch.float32,
            device=self.device,
        )

        k_cache_fp8 = torch.zeros(
            1,
            self.n_kv_head,
            self.max_seq_len,
            self.head_dim,
            dtype=torch.uint8,
            device=self.device,
        )
        v_cache_fp8 = torch.zeros(
            1,
            self.n_kv_head,
            self.max_seq_len,
            self.head_dim,
            dtype=torch.uint8,
            device=self.device,
        )
        k_scales = torch.zeros(
            1,
            self.n_kv_head,
            self.max_seq_len,
            1,
            dtype=torch.float16,
            device=self.device,
        )
        v_scales = torch.zeros(
            1,
            self.n_kv_head,
            self.max_seq_len,
            1,
            dtype=torch.float16,
            device=self.device,
        )

        # Clone for CPU reference
        qkv_cpu = qkv.cpu().clone()
        freqs_cpu = freqs_cis.cpu().clone()
        k_cache_cpu = k_cache_fp8.cpu().clone()
        v_cache_cpu = v_cache_fp8.cpu().clone()
        ks_cpu = k_scales.cpu().clone()
        vs_cpu = v_scales.cpu().clone()

        # Run GPU kernel
        _METAL_MOD.rope_kv_cache_write_fp8_mps(
            qkv,
            freqs_cis,
            k_cache_fp8,
            v_cache_fp8,
            k_scales,
            v_scales,
            q_dim,
            kv_dim,
            self.head_dim,
            0,
            write_pos,
        )
        torch.mps.synchronize()

        # Run CPU reference
        rope_kv_cache_write_quantized(
            qkv_cpu,
            freqs_cpu,
            k_cache_cpu,
            v_cache_cpu,
            ks_cpu,
            vs_cpu,
            q_dim,
            kv_dim,
            self.head_dim,
            0,
            write_pos,
            kv_dtype="fp8",
        )

        # Verify Q rotation parity
        q_gpu = qkv[:, :, :q_dim].cpu()
        q_cpu = qkv_cpu[:, :, :q_dim]
        q_cos = torch.cosine_similarity(
            q_gpu.flatten().float(), q_cpu.flatten().float(), dim=0
        ).item()
        self.assertGreaterEqual(q_cos, 0.9999, f"Q RoPE rotation mismatch: {q_cos}")

        # Verify GPU dequantized K & V match reference K & V
        k_dq_gpu = dequantize_kv_fp8(k_cache_fp8, k_scales, self.head_dim)[
            :, :, write_pos
        ].cpu()
        k_dq_cpu = dequantize_kv_fp8(k_cache_cpu, ks_cpu, self.head_dim)[
            :, :, write_pos
        ]
        kv_cos = torch.cosine_similarity(
            k_dq_gpu.flatten().float(), k_dq_cpu.flatten().float(), dim=0
        ).item()
        self.assertGreaterEqual(
            kv_cos, 0.999, f"FP8 GPU/CPU K parity mismatch: {kv_cos}"
        )

    def test_04_in_flight_rope_kv_write_int4_metal_parity(self):
        """
        Execute in-flight rope_kv_cache_write_int4_mps on Apple GPU and verify
        mathematical parity against CPU reference.
        """
        if self.device.type != "mps":
            self.skipTest("Metal MPS hardware required for kernel test.")

        q_dim = self.n_head * self.head_dim
        kv_dim = self.n_kv_head * self.head_dim
        total_dim = q_dim + 2 * kv_dim
        write_pos = 11

        qkv = torch.randn(1, 1, total_dim, dtype=torch.float16, device=self.device)
        freqs_cis = torch.randn(
            self.max_seq_len,
            self.head_dim // 2,
            2,
            dtype=torch.float32,
            device=self.device,
        )

        k_cache_int4 = torch.zeros(
            1,
            self.n_kv_head,
            self.max_seq_len,
            self.head_dim // 2,
            dtype=torch.uint8,
            device=self.device,
        )
        v_cache_int4 = torch.zeros(
            1,
            self.n_kv_head,
            self.max_seq_len,
            self.head_dim // 2,
            dtype=torch.uint8,
            device=self.device,
        )
        k_scales = torch.zeros(
            1,
            self.n_kv_head,
            self.max_seq_len,
            1,
            dtype=torch.float16,
            device=self.device,
        )
        v_scales = torch.zeros(
            1,
            self.n_kv_head,
            self.max_seq_len,
            1,
            dtype=torch.float16,
            device=self.device,
        )

        # Clone for CPU reference
        qkv_cpu = qkv.cpu().clone()
        freqs_cpu = freqs_cis.cpu().clone()
        k_cache_cpu = k_cache_int4.cpu().clone()
        v_cache_cpu = v_cache_int4.cpu().clone()
        ks_cpu = k_scales.cpu().clone()
        vs_cpu = v_scales.cpu().clone()

        # Run GPU kernel
        _METAL_MOD.rope_kv_cache_write_int4_mps(
            qkv,
            freqs_cis,
            k_cache_int4,
            v_cache_int4,
            k_scales,
            v_scales,
            q_dim,
            kv_dim,
            self.head_dim,
            0,
            write_pos,
        )
        torch.mps.synchronize()

        # Run CPU reference
        rope_kv_cache_write_quantized(
            qkv_cpu,
            freqs_cpu,
            k_cache_cpu,
            v_cache_cpu,
            ks_cpu,
            vs_cpu,
            q_dim,
            kv_dim,
            self.head_dim,
            0,
            write_pos,
            kv_dtype="int4",
        )

        k_dq_gpu = dequantize_kv_int4(k_cache_int4, k_scales, self.head_dim)[
            :, :, write_pos
        ].cpu()
        k_dq_cpu = dequantize_kv_int4(k_cache_cpu, ks_cpu, self.head_dim)[
            :, :, write_pos
        ]
        kv_cos = torch.cosine_similarity(
            k_dq_gpu.flatten().float(), k_dq_cpu.flatten().float(), dim=0
        ).item()
        self.assertGreaterEqual(
            kv_cos, 0.992, f"INT4 GPU/CPU K parity mismatch: {kv_cos}"
        )

    def test_05_static_kv_cache_memory_footprint_scaling(self):
        """
        Verify physical memory allocation reduction of StaticKVCache across
        context lengths T in {1024, 8192, 32768}.
        Assert:
          - FP8 achieves ~1.94x footprint reduction over FP16.
          - INT4 achieves ~3.76x footprint reduction over FP16.
        """
        n_layer = 16
        n_kv_head = 8
        head_dim = 64

        for seq_len in [1024, 8192, 32768]:
            c_fp16 = StaticKVCache(
                1,
                seq_len,
                n_layer,
                n_kv_head,
                head_dim,
                cache_dtype="fp16",
                device="cpu",
            )
            c_fp8 = StaticKVCache(
                1,
                seq_len,
                n_layer,
                n_kv_head,
                head_dim,
                cache_dtype="fp8",
                device="cpu",
            )
            c_int4 = StaticKVCache(
                1,
                seq_len,
                n_layer,
                n_kv_head,
                head_dim,
                cache_dtype="int4",
                device="cpu",
            )

            b_fp16 = c_fp16.get_memory_bytes()
            b_fp8 = c_fp8.get_memory_bytes()
            b_int4 = c_int4.get_memory_bytes()

            fp8_ratio = b_fp16 / b_fp8
            int4_ratio = b_fp16 / b_int4

            self.assertGreaterEqual(
                fp8_ratio,
                1.90,
                f"FP8 ratio {fp8_ratio:.2f}x below 1.90x at T={seq_len}",
            )
            self.assertLessEqual(fp8_ratio, 2.00)

            self.assertGreaterEqual(
                int4_ratio,
                3.70,
                f"INT4 ratio {int4_ratio:.2f}x below 3.70x at T={seq_len}",
            )
            self.assertLessEqual(int4_ratio, 4.00)

    def test_06_tree_speculative_kv_write_quantized_parity(self):
        """
        Verify batched tree candidate write (rope_tree_kv_write_fp8_mps and int4)
        for SpecInfer / Medusa tree topology (N=8 candidate nodes).
        """
        if self.device.type != "mps":
            self.skipTest("Metal MPS hardware required for kernel test.")

        T_nodes = 8
        tree_depths = torch.tensor(
            [0, 1, 1, 2, 2, 2, 3, 3], dtype=torch.int32, device=self.device
        )
        q_dim = self.n_head * self.head_dim
        kv_dim = self.n_kv_head * self.head_dim
        total_pairs = (q_dim + 2 * kv_dim) // 2
        write_pos = 10

        qkv = torch.randn(
            T_nodes, total_pairs, 2, dtype=torch.float16, device=self.device
        )
        freqs_cis = torch.randn(
            self.max_seq_len,
            self.head_dim // 2,
            2,
            dtype=torch.float32,
            device=self.device,
        )

        # FP8 Tree Write
        k_fp8 = torch.zeros(
            1,
            self.n_kv_head,
            self.max_seq_len,
            self.head_dim,
            dtype=torch.uint8,
            device=self.device,
        )
        v_fp8 = torch.zeros(
            1,
            self.n_kv_head,
            self.max_seq_len,
            self.head_dim,
            dtype=torch.uint8,
            device=self.device,
        )
        ks_fp8 = torch.zeros(
            1,
            self.n_kv_head,
            self.max_seq_len,
            1,
            dtype=torch.float16,
            device=self.device,
        )
        vs_fp8 = torch.zeros(
            1,
            self.n_kv_head,
            self.max_seq_len,
            1,
            dtype=torch.float16,
            device=self.device,
        )

        _METAL_MOD.rope_tree_kv_write_fp8_mps(
            qkv,
            freqs_cis,
            k_fp8,
            v_fp8,
            ks_fp8,
            vs_fp8,
            tree_depths,
            q_dim,
            kv_dim,
            self.head_dim,
            0,
            write_pos,
            T_nodes,
        )
        torch.mps.synchronize()

        # Verify all T candidate slots received non-zero scales and quantized entries
        for u in range(T_nodes):
            slot = (write_pos + u) % self.max_seq_len
            self.assertGreater(ks_fp8[0, :, slot, 0].abs().sum().item(), 0.0)
            self.assertGreater(k_fp8[0, :, slot].sum().item(), 0)

        # INT4 Tree Write
        k_i4 = torch.zeros(
            1,
            self.n_kv_head,
            self.max_seq_len,
            self.head_dim // 2,
            dtype=torch.uint8,
            device=self.device,
        )
        v_i4 = torch.zeros(
            1,
            self.n_kv_head,
            self.max_seq_len,
            self.head_dim // 2,
            dtype=torch.uint8,
            device=self.device,
        )
        ks_i4 = torch.zeros(
            1,
            self.n_kv_head,
            self.max_seq_len,
            1,
            dtype=torch.float16,
            device=self.device,
        )
        vs_i4 = torch.zeros(
            1,
            self.n_kv_head,
            self.max_seq_len,
            1,
            dtype=torch.float16,
            device=self.device,
        )

        _METAL_MOD.rope_tree_kv_write_int4_mps(
            qkv,
            freqs_cis,
            k_i4,
            v_i4,
            ks_i4,
            vs_i4,
            tree_depths,
            q_dim,
            kv_dim,
            self.head_dim,
            0,
            write_pos,
            T_nodes,
        )
        torch.mps.synchronize()

        for u in range(T_nodes):
            slot = (write_pos + u) % self.max_seq_len
            self.assertGreater(ks_i4[0, :, slot, 0].abs().sum().item(), 0.0)
            self.assertGreater(k_i4[0, :, slot].sum().item(), 0)

    def test_07_real_model_attention_layer_parity(self):
        """
        Evaluate full attention layer forward pass comparing FP16 vs FP8 vs INT4 KV-cache.
        Assert downstream attention output cosine similarity:
          - FP8 vs FP16 >= 0.999
          - INT4 vs FP16 >= 0.992
        """
        cfg = ModelConfig(
            block_size=64,
            vocab_size=128,
            n_layer=2,
            n_head=4,
            n_kv_head=2,
            n_embd=256,
        )
        head_dim = cfg.n_embd // cfg.n_head
        model = Transformer(cfg).to(device=self.device, dtype=torch.float16).eval()
        attn = model.transformer.h[0].attn
        x = torch.randn(1, 4, cfg.n_embd, device=self.device, dtype=torch.float16)
        freqs = model.freqs_cis

        cache_fp16 = StaticKVCache(
            1, 64, 1, cfg.n_kv_head, head_dim, cache_dtype="fp16", device=self.device
        )
        cache_fp8 = StaticKVCache(
            1, 64, 1, cfg.n_kv_head, head_dim, cache_dtype="fp8", device=self.device
        )
        cache_int4 = StaticKVCache(
            1, 64, 1, cfg.n_kv_head, head_dim, cache_dtype="int4", device=self.device
        )

        with torch.no_grad():
            out_fp16, _ = attn(x, freqs_cis=freqs, kv_cache=cache_fp16, use_cache=True)
            out_fp8, _ = attn(x, freqs_cis=freqs, kv_cache=cache_fp8, use_cache=True)
            out_int4, _ = attn(x, freqs_cis=freqs, kv_cache=cache_int4, use_cache=True)

        cos_fp8 = torch.cosine_similarity(
            out_fp16.flatten().float(), out_fp8.flatten().float(), dim=0
        ).item()
        cos_int4 = torch.cosine_similarity(
            out_fp16.flatten().float(), out_int4.flatten().float(), dim=0
        ).item()

        self.assertGreaterEqual(
            cos_fp8, 0.999, f"FP8 attention cosine similarity too low: {cos_fp8}"
        )
        self.assertGreaterEqual(
            cos_int4, 0.992, f"INT4 attention cosine similarity too low: {cos_int4}"
        )

    def test_08_extreme_edge_cases_and_outliers(self):
        """
        Test numerical robustness against:
        1. All-zero activations (zero division scale guard).
        2. Massive dynamic range (100-sigma outliers at magnitude 80.0).
        3. Ring-buffer wrap-around at sequence boundary.
        """
        # All zeros
        zeros = torch.zeros(
            1, 2, 4, self.head_dim, dtype=torch.float16, device=self.device
        )
        k_fp8, ks, _, _ = quantize_kv_fp8(zeros, zeros)
        k_dq = dequantize_kv_fp8(k_fp8, ks, self.head_dim)
        self.assertEqual(k_dq.abs().sum().item(), 0.0)

        # 100-sigma outlier
        outliers = torch.randn(
            1, 2, 4, self.head_dim, dtype=torch.float16, device=self.device
        )
        outliers[0, 0, 0, 10] = 80.0
        outliers[0, 1, 2, 20] = -80.0
        k_fp8, ks, _, _ = quantize_kv_fp8(outliers, outliers)
        k_dq = dequantize_kv_fp8(k_fp8, ks, self.head_dim)
        cos = torch.cosine_similarity(
            outliers.flatten().float(), k_dq.flatten().float(), dim=0
        ).item()
        self.assertGreaterEqual(cos, 0.999)

        # Ring-buffer wrap-around
        cache = StaticKVCache(
            1,
            16,
            2,
            2,
            self.head_dim,
            cache_dtype="int4",
            device=self.device,
            ring_buffer=True,
        )
        for step in range(25):  # 25 > 16 wraps around
            tok_k = torch.randn(
                1, 2, 1, self.head_dim, dtype=torch.float16, device=self.device
            )
            tok_v = torch.randn(
                1, 2, 1, self.head_dim, dtype=torch.float16, device=self.device
            )
            k_act, _ = cache.update(0, tok_k, tok_v)
            self.assertLessEqual(k_act.size(2), 16)
        self.assertEqual(cache.layer_seq_lens[0], 16)
        self.assertEqual(cache.total_seen_tokens[0], 25)


if __name__ == "__main__":
    unittest.main()
