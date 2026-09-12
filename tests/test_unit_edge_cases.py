import unittest
import torch

from axiomlm.engine.paged_cache import PagedKVCache, SequenceContext
from axiomlm.optim.muon import zeropower_via_newtonschulz5
from axiomlm.kernels import fused_sdpa


class TestKernelEdgeCases(unittest.TestCase):
    def setUp(self):
        self.devices = ["cpu"]
        if torch.cuda.is_available():
            self.devices.append("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.devices.append("mps")

    def test_muon_newton_schulz_zeros(self):
        """Ensure Muon Newton-Schulz root finding doesn't return NaNs on zero matrices (checks eps stability)."""
        G = torch.zeros(128, 128)
        # Should return all zeros, or near zero, but definitely not NaNs
        X = zeropower_via_newtonschulz5(G)
        self.assertFalse(
            torch.isnan(X).any(), "Newton-Schulz returned NaNs for zero matrix"
        )

    def test_muon_newton_schulz_extreme_values(self):
        """Ensure Muon doesn't diverge to Infinity with huge gradients."""
        G = torch.randn(128, 128) * 1e10
        X = zeropower_via_newtonschulz5(G)
        self.assertFalse(
            torch.isnan(X).any(), "Newton-Schulz returned NaNs for huge matrix"
        )
        self.assertFalse(
            torch.isinf(X).any(), "Newton-Schulz returned Infs for huge matrix"
        )

        # Verify it actually orthogonalized the matrix despite scale
        # X @ X.T should be approximately the identity matrix
        ortho = X @ X.T
        identity = torch.eye(128, dtype=X.dtype, device=X.device)
        # We allow a larger tolerance because bfloat16/float32 precision limits at extreme scaling
        max_diff = (ortho - identity).abs().max().item()
        self.assertLess(max_diff, 0.5, "Newton-Schulz failed to orthogonalize properly")

    def test_fused_sdpa_zeros_and_large_values(self):
        """Ensure fused_sdpa handles extreme values and padding masks without crashing."""
        for device in self.devices:
            B, H, T, D = 2, 4, 16, 64
            # Zero tensors
            q = torch.zeros(B, H, T, D, device=device)
            k = torch.zeros(B, H, T, D, device=device)
            v = torch.zeros(B, H, T, D, device=device)

            out = fused_sdpa(q, k, v, is_causal=True)
            self.assertFalse(
                torch.isnan(out).any(), f"fused_sdpa NaNs on zeros on {device}"
            )

            # Extreme logits which cause attention softmax saturation
            q = torch.randn(B, H, T, D, device=device) * 1e4
            k = torch.randn(B, H, T, D, device=device) * 1e4
            v = torch.randn(B, H, T, D, device=device)

            out = fused_sdpa(q, k, v, is_causal=True)
            self.assertFalse(
                torch.isnan(out).any(), f"fused_sdpa NaNs on extreme logits on {device}"
            )


class TestPagedKVCache(unittest.TestCase):
    def test_paged_cache_allocation_and_reconstruction(self):
        """Verify blocks are allocated correctly and sequence reconstruction is exact."""
        block_size = 16
        n_layer = 2
        n_kv_head = 4
        head_dim = 64

        cache = PagedKVCache(
            num_blocks=10,
            block_size=block_size,
            n_layer=n_layer,
            n_kv_head=n_kv_head,
            head_dim=head_dim,
        )

        self.assertEqual(len(cache.free_blocks), 10)

        ctx = SequenceContext(cache)

        # Append 20 tokens (Requires 2 blocks: 16 + 4)
        k1 = torch.randn(1, 20, n_kv_head, head_dim)
        v1 = torch.randn(1, 20, n_kv_head, head_dim)

        ctx.append_kv(layer_idx=0, k=k1, v=v1)

        # Should have allocated 2 blocks
        self.assertEqual(len(ctx.block_table), 2)
        self.assertEqual(len(cache.free_blocks), 8)  # 10 - 2 = 8
        self.assertEqual(ctx.seq_len, 20)

        # Reconstruct and check exact match
        k_recon, v_recon = ctx.get_reconstructed_cache(layer_idx=0)
        self.assertTrue(torch.allclose(k1, k_recon))
        self.assertTrue(torch.allclose(v1, v_recon))

        # Append 12 more tokens (Total 32, fills the second block perfectly)
        k2 = torch.randn(1, 12, n_kv_head, head_dim)
        v2 = torch.randn(1, 12, n_kv_head, head_dim)
        ctx.append_kv(layer_idx=0, k=k2, v=v2)

        self.assertEqual(len(ctx.block_table), 2)  # No new blocks needed yet
        self.assertEqual(len(cache.free_blocks), 8)
        self.assertEqual(ctx.seq_len, 32)

        # Append 1 more token (Requires 3rd block)
        k3 = torch.randn(1, 1, n_kv_head, head_dim)
        v3 = torch.randn(1, 1, n_kv_head, head_dim)
        ctx.append_kv(layer_idx=0, k=k3, v=v3)

        self.assertEqual(len(ctx.block_table), 3)
        self.assertEqual(len(cache.free_blocks), 7)
        self.assertEqual(ctx.seq_len, 33)

        # Verify full concatenation
        k_full = torch.cat([k1, k2, k3], dim=1)
        v_full = torch.cat([v1, v2, v3], dim=1)

        k_recon_full, v_recon_full = ctx.get_reconstructed_cache(layer_idx=0)
        self.assertTrue(torch.allclose(k_full, k_recon_full))
        self.assertTrue(torch.allclose(v_full, v_recon_full))

    def test_paged_cache_oom(self):
        """Verify the cache correctly raises RuntimeError when running out of blocks."""
        cache = PagedKVCache(
            num_blocks=2,  # Only 2 blocks available
            block_size=16,
            n_layer=1,
            n_kv_head=1,
            head_dim=64,
        )

        ctx = SequenceContext(cache)
        k = torch.randn(1, 40, 1, 64)  # 40 tokens requires 3 blocks (16, 16, 8)
        v = torch.randn(1, 40, 1, 64)

        with self.assertRaises(RuntimeError) as context:
            ctx.append_kv(layer_idx=0, k=k, v=v)

        self.assertTrue("PagedKVCache Out of Memory" in str(context.exception))


if __name__ == "__main__":
    unittest.main()
