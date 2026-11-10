"""
Unit and systems test suite for AxiomLM Phase 2 Task 2.4:
Sampling Without Replacement (Algorithm 2) & Power-Law Rejection Bounds.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from axiomlm.engine.medusa import MedusaSpeculativeDecoder
from axiomlm.kernels import _METAL_MOD, is_mps_available
from axiomlm.models.transformer import ModelConfig, Transformer
from axiomlm.speculative.sampling import (
    SequoiaTreeVerifier,
    fit_power_law_rejection_bound,
    sample_without_replacement,
)


class TestPhase2Task24SamplingWithoutReplacement(unittest.TestCase):
    """Rigorous mathematical and systems verification for Sequoia Algorithm 2."""

    def setUp(self):
        torch.manual_seed(2026)
        self.device = torch.device(
            "mps" if is_mps_available() and _METAL_MOD is not None else "cpu"
        )
        self.vocab_size = 64
        self.temperatures = [0.0, 0.2, 0.6, 1.0]

    def test_01_zero_duplicate_invariant_across_temperatures(self):
        """
        Validate zero-duplicate invariant of sample_without_replacement:
        For any T in {0.2, 0.6, 1.0} and any k <= V, the sampled tokens
        contain strictly zero duplicate elements (len(set(tokens)) == k).
        """
        V = 32
        logits = torch.randn(V, device=self.device)

        for T in [0.2, 0.6, 1.0]:
            for k in [1, 2, 4, 8, 16, 32]:
                for trial in range(50):
                    samples = sample_without_replacement(
                        logits, num_samples=k, temperature=T
                    )
                    self.assertEqual(samples.size(-1), k)
                    unique_tokens = set(samples.tolist())
                    self.assertEqual(
                        len(unique_tokens),
                        k,
                        f"Duplicate detected at T={T}, k={k}: {samples.tolist()}",
                    )

        # Edge cases: k=0, k > V
        empty_sample = sample_without_replacement(
            logits, num_samples=0, temperature=0.6
        )
        self.assertEqual(empty_sample.size(-1), 0)

        clamped_sample = sample_without_replacement(
            logits, num_samples=V + 10, temperature=0.6
        )
        self.assertEqual(clamped_sample.size(-1), V)
        self.assertEqual(len(set(clamped_sample.tolist())), V)

    def test_02_mathematical_parity_to_greedy_at_zero_temperature(self):
        """
        Validate exact mathematical equivalence at T = 0.0:
        1. sample_without_replacement matches torch.topk(logits, k).indices.
        2. SequoiaTreeVerifier accepts c_1 if and only if c_1 == argmax(P).
        3. Fallback token on rejection is identically argmax(P).
        """
        V = 64
        logits = torch.randn(V, device=self.device)

        # 1. Top-k parity
        for k in [1, 2, 4, 8]:
            res = sample_without_replacement(logits, num_samples=k, temperature=0.0)
            expected = torch.topk(logits, k=k, dim=-1).indices
            self.assertTrue(
                torch.equal(res, expected),
                f"Mismatch between sample_without_replacement and topk at k={k}",
            )

        # 2. Verifier parity
        verifier = SequoiaTreeVerifier(temperature=0.0)
        target_logits = torch.randn(V, device=self.device)
        best_token = target_logits.argmax(dim=-1).item()

        # Match case
        match_res = verifier.verify_candidates(
            target_logits=target_logits,
            candidates=[best_token, (best_token + 1) % V],
            temperature=0.0,
        )
        self.assertTrue(match_res.accepted)
        self.assertEqual(match_res.accepted_token, best_token)
        self.assertEqual(match_res.accepted_candidate_idx, 0)
        self.assertFalse(match_res.is_fallback)

        # Mismatch case
        wrong_token = (best_token + 1) % V
        mismatch_res = verifier.verify_candidates(
            target_logits=target_logits,
            candidates=[wrong_token],
            temperature=0.0,
        )
        self.assertFalse(mismatch_res.accepted)
        self.assertEqual(mismatch_res.accepted_token, best_token)
        self.assertIsNone(mismatch_res.accepted_candidate_idx)
        self.assertTrue(mismatch_res.is_fallback)

    def test_03_exact_distribution_preservation_lossless_guarantee(self):
        """
        Empirical statistical proof of Sequoia Algorithm 2 Theorem F.2:
        Under speculative verification with residual fallback sampling,
        the emitted token distribution is identical to the target model's
        true distribution P(x) (Total Variation Distance < 3*sigma).
        """
        V = 8
        torch.manual_seed(42)
        gen = torch.Generator(device="cpu").manual_seed(42)

        # Define distinct target and draft logits
        target_logits = torch.tensor(
            [2.5, 1.8, 1.2, 0.5, -0.2, -1.0, -1.5, -2.0], device="cpu"
        )
        draft_logits = torch.tensor(
            [1.0, 2.2, 0.8, 1.5, -0.5, -0.8, -1.2, -1.8], device="cpu"
        )

        T = 0.8
        p_target = F.softmax(target_logits / T, dim=-1).numpy()

        verifier = SequoiaTreeVerifier(temperature=T, generator=gen)
        num_trials = 40_000
        counts = [0] * V
        K = 2  # Propose 2 candidate tokens without replacement

        for _ in range(num_trials):
            # Draft K candidates without replacement from draft logits
            cands = sample_without_replacement(
                draft_logits, num_samples=K, temperature=T, generator=gen
            ).tolist()

            res = verifier.verify_candidates(
                target_logits=target_logits,
                candidates=cands,
                draft_logits=draft_logits,
                temperature=T,
            )
            counts[res.accepted_token] += 1

        p_emp = [c / num_trials for c in counts]

        # Calculate Total Variation Distance: D_TV = 0.5 * sum(|p_target - p_emp|)
        tv_dist = 0.5 * sum(abs(p_target[i] - p_emp[i]) for i in range(V))

        # Expected MC standard error for V=8, N=40,000 is ~0.005
        # 3-sigma tolerance = 0.018
        self.assertLess(
            tv_dist,
            0.020,
            f"Total Variation Distance {tv_dist:.4f} exceeds 3-sigma MC error threshold",
        )

        # Calculate KL divergence: D_KL(P_target || P_emp)
        kl_div = sum(
            p_target[i]
            * (
                torch.log(torch.tensor(p_target[i])) - torch.log(torch.tensor(p_emp[i]))
            ).item()
            for i in range(V)
            if p_target[i] > 1e-6 and p_emp[i] > 1e-6
        )
        self.assertLess(
            kl_div,
            0.005,
            f"KL Divergence {kl_div:.5f} exceeds statistical divergence threshold",
        )

    def test_04_power_law_rejection_decay(self):
        """
        Validate power-law decay of rejection rates r_k <= a * k^(-b):
        1. Exponent b > 0 (rejection rates decrease with candidate depth/width).
        2. Goodness-of-fit R^2 >= 0.80.
        3. Fit behaves gracefully on synthetic power-law sequences.
        """
        # Synthetic power-law rejection rates: r_k = 0.8 * k^(-0.7)
        true_b = 0.7
        true_a = 0.8
        rates = [true_a * (k ** (-true_b)) for k in range(1, 9)]

        fit_res = fit_power_law_rejection_bound(rates)
        self.assertTrue(fit_res["valid"])
        self.assertAlmostEqual(fit_res["exponent_b"], true_b, delta=0.01)
        self.assertAlmostEqual(fit_res["scale_a"], true_a, delta=0.01)
        self.assertGreaterEqual(fit_res["r_squared"], 0.99)

        # Monotonically decaying empirical simulation
        empirical_rejections = [0.65, 0.42, 0.28, 0.20, 0.15, 0.11]
        fit_emp = fit_power_law_rejection_bound(empirical_rejections)
        self.assertTrue(fit_emp["valid"])
        self.assertGreater(fit_emp["exponent_b"], 0.0)
        self.assertGreaterEqual(fit_emp["r_squared"], 0.90)

    def test_05_end_to_end_medusa_speculative_decoder_across_temperatures(self):
        """
        Validate end-to-end MedusaSpeculativeDecoder execution across all 4 temperatures:
        T in {0.0, 0.2, 0.6, 1.0} on hardware device.
        Verifies:
        1. Exact token length generated.
        2. Valid token range within vocabulary.
        3. Rejection statistics and power-law fit populated.
        4. Parity of greedy generation across runs with seed.
        """
        config = ModelConfig(
            block_size=128,
            vocab_size=self.vocab_size,
            n_layer=2,
            n_head=4,
            n_embd=64,
        )
        model = Transformer(config).to(self.device).eval()
        model.add_medusa_heads(num_heads=3)

        prompt = [1, 5, 12, 18]
        max_new = 16

        for T in self.temperatures:
            gen = torch.Generator(device="cpu").manual_seed(2026)
            decoder = MedusaSpeculativeDecoder(num_heads=3, temperature=T)

            tokens, stats = decoder.generate(
                model=model,
                prompt_tokens=prompt,
                max_new_tokens=max_new,
                device=self.device,
                generator=gen,
            )

            # Verification assertions
            self.assertEqual(len(tokens), len(prompt) + max_new)
            self.assertTrue(all(0 <= tok < self.vocab_size for tok in tokens))
            self.assertEqual(stats["new_tokens"], max_new)
            self.assertEqual(stats["temperature"], T)
            self.assertGreater(stats["tokens_per_second"], 0.0)
            self.assertIn("power_law_fit", stats)

            # Greedy reproducibility test
            if T == 0.0:
                tokens_rep, _ = decoder.generate(
                    model=model,
                    prompt_tokens=prompt,
                    max_new_tokens=max_new,
                    device=self.device,
                )
                self.assertEqual(
                    tokens,
                    tokens_rep,
                    "Greedy decoding (T=0.0) must produce identical deterministic tokens across runs",
                )


if __name__ == "__main__":
    unittest.main()
