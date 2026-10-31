"""
Metal Kernel Boundary, Tiling, and Ring-Buffer Wrap Tests.
Validates non-power-of-2 context lengths for Split-KV Flash-Decode ($B_r=16$),
circular ring-buffer modulo wrapping, and non-power-of-2 vocab argmax reductions.
"""

import unittest
import torch
import torch.nn.functional as F

from axiomlm.models.transformer import Transformer, ModelConfig
from axiomlm.engine.paged_cache import StaticKVCache

try:
    from axiomlm.kernels import _METAL_MOD
except ImportError:
    _METAL_MOD = None


class TestMetalBoundaryAndTiling(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            self.skipTest("Metal GPU / MPS required for hardware boundary tests")
        if _METAL_MOD is None:
            self.skipTest("Metal compiled extension module not available")
        self.device = torch.device("mps")

    def test_non_power_of_two_context_lengths_flash_decode(self):
        """
        Tests Split-KV Flash-Decode across odd context lengths:
        P in {1, 3, 7, 13, 17, 29, 31, 33, 65, 127}.
        Guarantees that register tiling (Br=16) and split reductions correctly
        mask out unaligned tail tokens without memory corruption or NaNs.
        """
        odd_lengths = [1, 3, 7, 13, 17, 29, 31, 33, 65, 127]
        H_q, H_kv, D = 12, 4, 64
        max_seq = 256

        for active_len in odd_lengths:
            q = torch.randn(1, H_q, 1, D, device=self.device, dtype=torch.float16)
            k_cache = torch.randn(
                1, H_kv, max_seq, D, device=self.device, dtype=torch.float16
            )
            v_cache = torch.randn(
                1, H_kv, max_seq, D, device=self.device, dtype=torch.float16
            )

            # 1. Custom Metal Split-KV Flash-Decode
            out_custom = _METAL_MOD.gqa_flash_decode_mps(
                q, k_cache, v_cache, active_len, 0
            )
            self.assertFalse(
                torch.isnan(out_custom).any(),
                f"NaN in custom Flash-Decode at active_len={active_len}",
            )

            # 2. PyTorch Reference Scaled Dot-Product Attention
            k_slice = k_cache[:, :, :active_len, :]
            v_slice = v_cache[:, :, :active_len, :]
            # Repeat KV heads 3x for 12 Q heads
            k_rep = k_slice.repeat_interleave(3, dim=1)
            v_rep = v_slice.repeat_interleave(3, dim=1)

            out_ref = F.scaled_dot_product_attention(
                q, k_rep, v_rep, is_causal=False, scale=0.125
            )

            # Verify maximum absolute difference
            max_diff = (out_custom - out_ref).abs().max().item()
            self.assertLess(
                max_diff,
                5e-3,
                f"Flash-Decode difference {max_diff} exceeds tolerance at active_len={active_len}",
            )

    def test_ring_buffer_wrap_around_boundary(self):
        """
        Tests circular ring-buffer boundary crossing when generated tokens exceed max_seq_len.
        Ensures modulo arithmetic in RoPE KV cache write wraps safely to slot 0.
        """
        window_size = 32
        config = ModelConfig(
            arch="modern",
            n_layer=2,
            n_head=4,
            n_kv_head=2,
            n_embd=128,
            vocab_size=500,
            block_size=window_size,
            norm_type="rmsnorm",
            mlp_type="swiglu",
            pos_emb="rope",
            use_fused_kernels=True,
        )
        model = Transformer(config).to(device=self.device, dtype=torch.float16).eval()
        cache = StaticKVCache(
            1,
            window_size,
            2,
            2,
            32,
            device=self.device,
            dtype=torch.float16,
            ring_buffer=True,
        )

        prompt = torch.tensor([[10, 20, 30]], device=self.device, dtype=torch.long)
        # Prefill
        logits, _, _ = model(prompt, kv_caches=[cache] * 2)
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # Generate 64 tokens (crossing the 32-token ring-buffer boundary twice)
        for step in range(64):
            logits, _, _ = model(next_tok, kv_caches=[cache] * 2)
            self.assertFalse(
                torch.isnan(logits).any(),
                f"NaN encountered at step {step} during ring-buffer wrap",
            )
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # Verify that total_seen_tokens is 3 + 64 = 67, while layer_seq_lens is capped at 32
        self.assertEqual(cache.total_seen_tokens[0], 67)
        self.assertEqual(cache.layer_seq_lens[0], 32)

    def test_non_power_of_two_vocab_argmax(self):
        """
        Tests GPU argmax reductions on the non-power-of-2 vocab size V=50,304.
        Verifies that max values placed at boundary indices (0, 50302, 50303)
        are accurately detected without out-of-bounds reads.
        """
        vocab_size = 50304
        test_indices = [0, 1, 1023, 2048, 50000, 50302, 50303]

        for target_idx in test_indices:
            logits = torch.zeros(
                (1, 1, vocab_size), device=self.device, dtype=torch.float16
            )
            logits[0, 0, target_idx] = 100.0  # Dominant logit

            pred_idx = _METAL_MOD.argmax_last_token_mps(logits)
            self.assertEqual(
                pred_idx,
                target_idx,
                f"argmax_last_token_mps failed to identify target at index {target_idx} (got {pred_idx})",
            )


if __name__ == "__main__":
    unittest.main()
