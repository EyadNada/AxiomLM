"""
Adversarial and Edge-Case Tests for Speculative Decoding.
Tests extreme boundary conditions, cache rollback/rewind integrity,
high-entropy zero-match fallbacks, and context window limits.
"""

import unittest
import torch

from axiomlm.models.transformer import Transformer, ModelConfig
from axiomlm.engine.paged_cache import StaticKVCache
from axiomlm.engine.speculative import (
    find_candidate_ngram,
    SpeculativePromptLookupDecoder,
)


class TestSpeculativeAdversarial(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = (
            torch.device("mps")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else torch.device("cpu")
        )

    def _create_toy_model(self, block_size: int = 128) -> Transformer:
        config = ModelConfig(
            arch="modern",
            n_layer=4,
            n_head=4,
            n_kv_head=2,
            n_embd=128,
            vocab_size=1000,
            block_size=block_size,
            norm_type="rmsnorm",
            mlp_type="swiglu",
            pos_emb="rope",
            use_fused_kernels=(self.device.type == "mps"),
        )
        return Transformer(config).to(device=self.device, dtype=torch.float16).eval()

    def test_adversarial_zero_match_random_entropy(self):
        """
        Adversarial Test: High-entropy prompt with strictly non-repeating tokens.
        Proves candidate n-gram search returns empty list [] and the speculative engine
        falls back to single-token greedy generation with zero corruption.
        """
        # Create strictly non-repeating token sequence
        prompt_tokens = list(range(100, 160))  # 60 unique tokens
        candidates = find_candidate_ngram(
            prompt_tokens, ngram_size=3, max_spec_tokens=4
        )
        self.assertEqual(
            candidates, [], "Expected 0 candidates on non-repeating prompt"
        )

        model = self._create_toy_model()

        # Run greedy baseline
        cache_ref = StaticKVCache(
            1, 128, 4, 2, 32, device=self.device, dtype=torch.float16
        )
        with torch.no_grad():
            prompt_tensor = torch.tensor(
                [prompt_tokens], device=self.device, dtype=torch.long
            )
            out, _, _ = model(prompt_tensor, kv_caches=[cache_ref] * 4)
            next_tok = out[:, -1, :].argmax(dim=-1, keepdim=True)

            ref_tokens = list(prompt_tokens)
            ref_tokens.append(int(next_tok.item()))

            for _ in range(15):
                out, _, _ = model(next_tok, kv_caches=[cache_ref] * 4)
                next_tok = out[:, -1, :].argmax(dim=-1, keepdim=True)
                ref_tokens.append(int(next_tok.item()))

        # Run speculative decoder (Python loop)
        decoder = SpeculativePromptLookupDecoder(ngram_size=3, max_spec_tokens=3)
        spec_tokens_py, stats_py = decoder.generate(
            model,
            prompt_tokens,
            max_new_tokens=16,
            device=self.device,
            use_cpp_loop=False,
        )
        self.assertEqual(
            spec_tokens_py,
            ref_tokens,
            "Python speculative loop diverged on zero-match high-entropy text!",
        )

        # Run speculative decoder (Pure C++ loop if MPS available)
        if self.device.type == "mps":
            spec_tokens_cpp, stats_cpp = decoder.generate(
                model,
                prompt_tokens,
                max_new_tokens=16,
                device=self.device,
                use_cpp_loop=True,
            )
            self.assertEqual(
                spec_tokens_cpp,
                ref_tokens,
                "Pure C++ speculative loop diverged on zero-match high-entropy text!",
            )

    def test_acceptance_and_rejection_permutations(self):
        """
        Tests speculative decoding under varied repetition patterns that provoke
        different acceptance numbers (0, 1, 2, K accepted tokens).
        Ensures strict token-for-token mathematical parity with greedy decoding.
        """
        patterns = [
            # Pattern A: Repetitive blocks with periodic noise
            [10, 20, 30, 40, 50, 60] * 5,
            # Pattern B: Alternating pairs
            [100, 200, 100, 200, 300, 400] * 6,
            # Pattern C: Long repetitive motif with unexpected suffix
            [1, 2, 3, 4, 5, 6, 7, 8, 9] * 4 + [1, 2, 3, 99],
        ]

        model = self._create_toy_model()

        for idx, prompt in enumerate(patterns):
            # Compute greedy reference
            cache_ref = StaticKVCache(
                1, 128, 4, 2, 32, device=self.device, dtype=torch.float16
            )
            with torch.no_grad():
                pt_tensor = torch.tensor([prompt], device=self.device, dtype=torch.long)
                out, _, _ = model(pt_tensor, kv_caches=[cache_ref] * 4)
                next_tok = out[:, -1, :].argmax(dim=-1, keepdim=True)
                ref_gen = list(prompt)
                ref_gen.append(int(next_tok.item()))

                for _ in range(16):
                    out, _, _ = model(next_tok, kv_caches=[cache_ref] * 4)
                    next_tok = out[:, -1, :].argmax(dim=-1, keepdim=True)
                    ref_gen.append(int(next_tok.item()))

            decoder = SpeculativePromptLookupDecoder(ngram_size=3, max_spec_tokens=3)

            # Python Loop
            spec_py, stats_py = decoder.generate(
                model, prompt, max_new_tokens=17, device=self.device, use_cpp_loop=False
            )
            self.assertEqual(
                spec_py[: len(ref_gen)],
                ref_gen,
                f"Pattern {idx} diverged in Python speculative loop",
            )

            # C++ Loop
            if self.device.type == "mps":
                spec_cpp, stats_cpp = decoder.generate(
                    model,
                    prompt,
                    max_new_tokens=17,
                    device=self.device,
                    use_cpp_loop=True,
                )
                self.assertEqual(
                    spec_cpp[: len(ref_gen)],
                    ref_gen,
                    f"Pattern {idx} diverged in C++ speculative loop",
                )

    def test_kv_cache_rewind_and_rollback_integrity(self):
        """
        Verifies that when speculative draft tokens are rejected, the KV-cache
        is cleanly rewound without ghost tokens corrupting subsequent steps.
        """
        model = self._create_toy_model()
        code_motif = [11, 22, 33, 44, 55, 66, 77, 88] * 4
        cache = StaticKVCache(1, 128, 4, 2, 32, device=self.device, dtype=torch.float16)

        decoder = SpeculativePromptLookupDecoder(ngram_size=3, max_spec_tokens=4)
        gen_tokens, stats = decoder.generate(
            model,
            code_motif,
            max_new_tokens=20,
            cache=cache,
            device=self.device,
            use_cpp_loop=False,
        )

        # In autoregressive generation, the final predicted token is returned
        # but its KV pairs are only processed if another forward pass occurs.
        # Thus, total_seen_tokens is exactly len(generated) - 1.
        expected_seen = len(gen_tokens) - 1
        self.assertEqual(
            cache.total_seen_tokens[0],
            expected_seen,
            f"KV cache seen count ({cache.total_seen_tokens[0]}) != expected ({expected_seen}) after rejection rollbacks",
        )

    def test_context_window_boundary_speculation(self):
        """
        Boundary Test: Speculative decoding approaching and reaching physical cache block_size.
        Ensures speculation does not attempt to write past block_size.
        """
        block_size = 48
        model = self._create_toy_model(block_size=block_size)
        pattern = [1, 2, 3, 4, 5] * 7  # 35 tokens
        cache = StaticKVCache(
            1, block_size, 4, 2, 32, device=self.device, dtype=torch.float16
        )

        decoder = SpeculativePromptLookupDecoder(ngram_size=3, max_spec_tokens=4)

        # Request generation that fills the cache up to capacity (35 + 13 = 48)
        gen_tokens, stats = decoder.generate(
            model,
            pattern,
            max_new_tokens=13,
            cache=cache,
            device=self.device,
            use_cpp_loop=False,
        )
        self.assertEqual(len(gen_tokens), block_size)
        self.assertEqual(cache.total_seen_tokens[0], block_size - 1)


if __name__ == "__main__":
    unittest.main()
