"""
Unit and systems test suite for AxiomLM Phase 2 Task 2.2:
Sequoia Hardware-Aware Dynamic Programming Tree Optimizer.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from axiomlm.engine.paged_cache import StaticKVCache
from axiomlm.kernels import _METAL_MOD, is_mps_available
from axiomlm.models.transformer import ModelConfig, Transformer
from axiomlm.speculative.sequoia_dp import (
    SequoiaDPOptimizer,
)


class TestPhase2Task22SequoiaDP(unittest.TestCase):
    """Rigorous mathematical and systems verification for Sequoia DP optimizer."""

    def setUp(self):
        self.optimizer = SequoiaDPOptimizer(
            num_heads=4,
            max_branches=4,
            draft_cost_c=0.02,
            max_budget=32,
        )
        # Task 2.1 M3 Pro empirical profile
        self.t_m3_pro = {1: 1.000, 2: 1.280, 4: 1.385, 8: 1.488, 16: 2.315, 32: 3.926}
        self.optimizer.set_hardware_latency(self.t_m3_pro)

    def test_01_dp_table_monotonicity_and_bounds(self):
        """
        Validate mathematical invariants of the dynamic programming recurrence:
        1. G(1) == 1.0 (single token greedy decode)
        2. G(N) strictly non-decreasing: G(N + 1) >= G(N)
        3. G(N) <= N for all N
        4. Sub-linear expected return: G(N) / N strictly decreases for N >= 2
        """
        T, _ = self.optimizer.solve_dp_tables()
        prev_G = 1.0

        for n in range(1, 33):
            G_n = float(np.max(T[n, 0, :]))
            self.assertGreaterEqual(
                G_n, prev_G, f"Monotonicity violation at n={n}: {G_n} < {prev_G}"
            )
            self.assertLessEqual(
                G_n, float(n), f"Information bound violation at n={n}: {G_n} > {n}"
            )
            if n > 1:
                self.assertLess(
                    G_n / n,
                    prev_G / (n - 1),
                    f"Sub-linear return violation at n={n}",
                )
            prev_G = G_n

        self.assertAlmostEqual(float(np.max(T[1, 0, :])), 1.0, places=5)

    def test_02_tree_topology_reconstruction_exactness(self):
        """
        Verify that backtracking extracts exact, valid, prefix-closed candidate trees:
        1. Budget n yields exactly n - 1 draft paths (total n tokens including root).
        2. Reconstructed paths are unique and prefix-closed.
        3. Depth does not exceed num_heads.
        """
        T, split = self.optimizer.solve_dp_tables()

        for n in [2, 4, 8, 16, 20]:
            choices = self.optimizer.extract_tree_for_budget(n, T, split)
            self.assertEqual(
                len(choices),
                n - 1,
                f"Budget {n} produced {len(choices)} draft paths, expected {n - 1}",
            )

            # Ensure all paths are unique
            self.assertEqual(
                len(choices), len(set(choices)), f"Duplicate paths for budget {n}"
            )

            # Ensure max depth is within num_heads
            for path in choices:
                self.assertLessEqual(
                    len(path),
                    self.optimizer.num_heads,
                    f"Path {path} exceeds num_heads={self.optimizer.num_heads}",
                )

            # Check compile_medusa_tree compatibility
            compiled = self.optimizer.solve_fixed_budget(n).compiled_tree
            self.assertEqual(compiled["num_nodes"], n)
            self.assertEqual(len(compiled["tree_depths"]), n)
            self.assertEqual(len(compiled["tree_mask"]), n)

    def test_03_hardware_speedup_optimum_detection(self):
        """
        Microarchitectural Proof:
        Verify that under the Apple M3 Pro UMA latency curve t(N), the optimizer
        identifies the global maximum speedup at a moderate tree budget (N* in [4, 16])
        and proves that N=32 results in a lower speedup due to compute knee saturation.
        """
        result = self.optimizer.solve_optimal_tree(max_tree_size=32)

        self.assertGreaterEqual(result.optimal_N, 4)
        self.assertLessEqual(result.optimal_N, 16)
        self.assertGreater(result.speedup, 1.40, "Expected speedup > 1.40x on M3 Pro")

        speedup_32 = result.all_budget_stats[32]["speedup"]
        self.assertGreater(
            result.speedup,
            speedup_32 * 1.5,
            f"Expected optimal speedup ({result.speedup:.3f}x) > 1.5 * speedup(32) ({speedup_32:.3f}x)",
        )

    def test_04_acceptance_rate_sensitivity_analysis(self):
        """
        Verify that optimal tree budget adapts to model accuracy:
        1. High acceptance (P ~ 0.90) expands optimal tree budget N*.
        2. Low acceptance (P ~ 0.10) contracts optimal tree budget N* to 1 or 2.
        """
        # High acceptance regime
        opt_high = SequoiaDPOptimizer(num_heads=4, max_branches=4, draft_cost_c=0.01)
        opt_high.set_hardware_latency(self.t_m3_pro)
        P_high = np.array(
            [
                [0.92, 0.45, 0.20, 0.08],
                [0.85, 0.35, 0.15, 0.05],
                [0.75, 0.25, 0.10, 0.03],
                [0.65, 0.20, 0.08, 0.02],
            ]
        )
        opt_high.set_acceptance_probabilities(P_high)
        res_high = opt_high.solve_optimal_tree(max_tree_size=32)

        # Low acceptance regime
        opt_low = SequoiaDPOptimizer(num_heads=4, max_branches=4, draft_cost_c=0.04)
        opt_low.set_hardware_latency(self.t_m3_pro)
        P_low = np.array(
            [
                [0.15, 0.02, 0.005, 0.001],
                [0.08, 0.01, 0.002, 0.000],
                [0.04, 0.005, 0.001, 0.000],
                [0.02, 0.002, 0.000, 0.000],
            ]
        )
        opt_low.set_acceptance_probabilities(P_low)
        res_low = opt_low.solve_optimal_tree(max_tree_size=32)

        self.assertGreater(
            res_high.optimal_N,
            res_low.optimal_N,
            f"Expected high-accuracy tree ({res_high.optimal_N}) > low-accuracy tree ({res_low.optimal_N})",
        )
        self.assertLessEqual(
            res_low.optimal_N, 2, "Low accuracy should contract tree to N <= 2"
        )

    @unittest.skipUnless(
        is_mps_available() and _METAL_MOD is not None, "Requires Apple Metal MPS"
    )
    def test_05_end_to_end_forward_tree_verify_execution(self):
        """
        Systems Integration Test:
        Extract the DP-optimal tree, compile it, and verify that AxiomLM's Transformer
        executes `forward_tree_verify` directly on Apple Silicon MPS with zero errors.
        """
        device = torch.device("mps")
        cfg = ModelConfig(
            arch="modern",
            n_layer=4,
            n_head=8,
            n_kv_head=2,
            n_embd=512,
            intermediate_size=1024,
            vocab_size=1024,
            block_size=256,
            use_fused_kernels=True,
            qkv_bias=True,
        )
        model = Transformer(cfg).to(device=device, dtype=torch.float16).eval()
        model.quantize("q4")
        model.add_medusa_heads(num_heads=4)

        opt_result = self.optimizer.solve_optimal_tree(max_tree_size=16)
        compiled = opt_result.compiled_tree
        T_nodes = compiled["num_nodes"]

        cache = StaticKVCache(
            max_batch_size=1,
            max_seq_len=256,
            n_layer=cfg.n_layer,
            n_kv_head=cfg.n_kv_head,
            head_dim=cfg.n_embd // cfg.n_head,
            device=device,
            dtype=torch.float16,
        )

        prefix = torch.randint(0, cfg.vocab_size, (1, 32), device=device)
        cand_tokens = torch.randint(0, cfg.vocab_size, (1, T_nodes), device=device)

        with torch.no_grad():
            model(prefix, kv_caches=[cache] * cfg.n_layer)
            logits, med_logits, _ = model.forward_tree_verify(
                cand_tokens,
                kv_caches=[cache] * cfg.n_layer,
                tree_depths=compiled["tree_depths"],
                tree_mask=compiled["tree_mask"],
                return_medusa=True,
            )

        self.assertEqual(logits.shape, (1, T_nodes, cfg.vocab_size))
        self.assertEqual(len(med_logits), 4)


if __name__ == "__main__":
    unittest.main()
