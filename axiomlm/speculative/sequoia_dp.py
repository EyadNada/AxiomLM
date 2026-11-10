"""
AxiomLM Sequoia Hardware-Aware Dynamic Programming Tree Optimizer.

Implements Sequoia's exact Dynamic Programming tree optimization algorithm
(Chen et al., CMU / Together AI / Meta FAIR, 2024; arXiv:2402.12374) adapted for
Apple Silicon Unified Memory Architectures (UMA) and Medusa multi-head self-drafting.

Solves the constrained optimization problem:
    Tree* = argmax_{T in Trees} Speedup(T)
where:
    Speedup(T) = G(|T|) / (t(|T|) + d(T) * c)
    - G(|T|) is expected accepted tokens under empirical positional branch probabilities P[d, b].
    - t(|T|) is empirical hardware verification latency relative to single-token decode (from Task 2.1).
    - d(T) is tree depth and c is relative draft cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class OptimalTreeResult:
    """Encapsulates the global optimal tree topology and performance metrics."""

    optimal_N: int
    max_depth: int
    expected_accepted_tokens: float
    verification_cost_t: float
    draft_cost: float
    speedup: float
    tree_choices: list[tuple[int, ...]]
    compiled_tree: dict[str, Any]
    all_budget_stats: dict[int, dict[str, float]]


def interpolate_hardware_latency(
    t_measured: dict[int, float], max_N: int = 32
) -> dict[int, float]:
    """
    Interpolates measured hardware latency profile across all integer budgets N in [1, max_N].
    Uses monotonic piecewise linear interpolation between measured anchor points,
    ensuring t(1) == 1.0 and t(N1) <= t(N2) for N1 < N2.
    """
    t_interp: dict[int, float] = {}
    known_keys = sorted(t_measured.keys())
    if 1 not in known_keys:
        known_keys.insert(0, 1)
        t_measured[1] = 1.0

    for n in range(1, max_N + 1):
        if n in t_measured:
            t_interp[n] = float(t_measured[n])
        else:
            # Find bounding keys
            lower_k = max([k for k in known_keys if k <= n], default=known_keys[0])
            upper_k = min([k for k in known_keys if k >= n], default=known_keys[-1])
            if lower_k == upper_k:
                t_interp[n] = float(t_measured[lower_k])
            else:
                alpha = (n - lower_k) / (upper_k - lower_k)
                t_interp[n] = float(
                    t_measured[lower_k]
                    + alpha * (t_measured[upper_k] - t_measured[lower_k])
                )

    return t_interp


class SequoiaDPOptimizer:
    """
    Sequoia Hardware-Aware Dynamic Programming Tree Optimizer.
    """

    def __init__(
        self,
        num_heads: int = 4,
        max_branches: int = 4,
        draft_cost_c: float = 0.02,
        max_budget: int = 32,
    ):
        """
        Args:
            num_heads: Number of speculative drafting heads (maximum tree depth).
            max_branches: Maximum branching factor per node (top-k candidates per head).
            draft_cost_c: Relative draft execution latency cost (c = t_draft / t_verify(1)).
            max_budget: Maximum candidate tree size (constrained by Metal 32-bit bitmask).
        """
        self.num_heads = int(num_heads)
        self.max_branches = int(max_branches)
        self.draft_cost_c = float(draft_cost_c)
        self.max_budget = min(int(max_budget), 32)

        # Default hardware latency profile for Apple M3 Pro (MARLIN Q4 from Task 2.1)
        self.t_hardware: dict[int, float] = interpolate_hardware_latency(
            {1: 1.000, 2: 1.280, 4: 1.385, 8: 1.488, 16: 2.315, 32: 3.926},
            max_N=self.max_budget,
        )

        # Default empirical acceptance probabilities: [num_heads, max_branches]
        # P[d, b] is probability that b-th candidate from head d is accepted given parent is accepted.
        self.P: np.ndarray = np.zeros(
            (self.num_heads, self.max_branches), dtype=np.float64
        )
        base_decay = [0.65, 0.45, 0.30, 0.20]
        for d in range(self.num_heads):
            p_head = base_decay[d] if d < len(base_decay) else (0.20 * (0.7 ** (d - 3)))
            for b in range(self.max_branches):
                self.P[d, b] = p_head * (0.35**b)

    def set_hardware_latency(self, t_profile: dict[int, float] | list[float]) -> None:
        """
        Updates the empirical hardware latency profile t(N) = Latency(N) / Latency(1).
        """
        if isinstance(t_profile, list):
            t_dict = {i + 1: float(val) for i, val in enumerate(t_profile)}
        else:
            t_dict = {int(k): float(v) for k, v in t_profile.items()}
        self.t_hardware = interpolate_hardware_latency(t_dict, max_N=self.max_budget)

    def set_acceptance_probabilities(
        self,
        P: np.ndarray | list[list[float]] | list[float],
    ) -> None:
        """
        Sets empirical positional acceptance probabilities P[d, b].
        Can be 2D array [num_heads, max_branches] or 1D list [max_branches].
        """
        arr = np.array(P, dtype=np.float64)
        if arr.ndim == 1:
            # Broadcast across all heads with geometric depth penalty
            self.max_branches = len(arr)
            self.P = np.zeros((self.num_heads, self.max_branches), dtype=np.float64)
            for d in range(self.num_heads):
                depth_factor = 0.85**d
                self.P[d, :] = arr * depth_factor
        elif arr.ndim == 2:
            self.num_heads = arr.shape[0]
            self.max_branches = arr.shape[1]
            self.P = arr
        else:
            raise ValueError(f"Expected 1D or 2D array for P, got shape {arr.shape}")

    def solve_dp_tables(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Executes Sequoia's 3D dynamic programming recurrence:
            T[n, d, b] = max(
                T[n, d, b - 1],
                max_{1 <= m < n} ( T[n - m, d, b - 1] + P[d, b] * max_j T[m, d + 1, j] )
            )
        Returns:
            T: Expected accepted tokens table [max_budget + 1, num_heads + 1, max_branches + 1]
            split: Optimal budget partition decisions [max_budget + 1, num_heads + 1, max_branches + 1]
        """
        N = self.max_budget
        D = self.num_heads
        B = self.max_branches

        T = np.zeros((N + 1, D + 1, B + 1), dtype=np.float64)
        split = np.zeros((N + 1, D + 1, B + 1), dtype=np.int32)

        # Base cases:
        # n = 1: subtree only contains root token -> generates exactly 1.0 token
        for d in range(D + 1):
            for b in range(B + 1):
                T[1, d, b] = 1.0

        # b = 0: no branches permitted from this node -> generates exactly 1.0 token
        for n in range(1, N + 1):
            for d in range(D + 1):
                T[n, d, 0] = 1.0

        # d = D: maximum depth reached (no subsequent Medusa head) -> generates exactly 1.0 token
        for n in range(1, N + 1):
            for b in range(B + 1):
                T[n, D, b] = 1.0

        # DP recurrence over budget n, depth d, and branch index b
        for n in range(2, N + 1):
            for d in range(D - 1, -1, -1):
                for b in range(1, B + 1):
                    # Option 1: Do not allocate budget to branch b
                    best_val = T[n, d, b - 1]
                    best_m = 0

                    p_b = float(self.P[d, b - 1])

                    # Option 2: Allocate m nodes (1 <= m < n) to branch b's subtree
                    for m in range(1, n):
                        sub_val = float(np.max(T[m, d + 1, :]))
                        val = T[n - m, d, b - 1] + p_b * sub_val
                        if val > best_val:
                            best_val = val
                            best_m = m

                    T[n, d, b] = best_val
                    split[n, d, b] = best_m

        return T, split

    def _reconstruct_tree_nodes(
        self,
        T: np.ndarray,
        split: np.ndarray,
        n: int,
        d: int,
        b: int,
        current_path: tuple[int, ...],
        tree_nodes: list[tuple[int, ...]],
    ) -> None:
        """
        Backtracks through the optimal split table to extract the concrete top-k draft paths.
        """
        if n <= 1 or b <= 0 or d >= self.num_heads:
            return

        m = int(split[n, d, b])

        # Recurse on remaining branches b - 1 with residual budget (n - m)
        self._reconstruct_tree_nodes(
            T, split, n - m, d, b - 1, current_path, tree_nodes
        )

        if m > 0:
            child_path = current_path + (b - 1,)
            tree_nodes.append(child_path)
            # Find optimal branch count for child subtree at depth d + 1
            best_j = int(np.argmax(T[m, d + 1, :]))
            self._reconstruct_tree_nodes(
                T, split, m, d + 1, best_j, child_path, tree_nodes
            )

    def extract_tree_for_budget(
        self,
        n: int,
        T: np.ndarray,
        split: np.ndarray,
    ) -> list[tuple[int, ...]]:
        """
        Reconstructs the optimal list of top-k draft paths for candidate budget n.
        """
        if n <= 1:
            return []
        best_b = int(np.argmax(T[n, 0, :]))
        tree_nodes: list[tuple[int, ...]] = []
        self._reconstruct_tree_nodes(T, split, n, 0, best_b, (), tree_nodes)
        return tree_nodes

    def solve_optimal_tree(
        self,
        max_tree_size: int | None = None,
    ) -> OptimalTreeResult:
        """
        Identifies the provably optimal candidate tree topology (N*, d*) maximizing
        wall-clock speedup across all candidate budgets N in [1, max_tree_size].
        """
        limit_N = min(max_tree_size or self.max_budget, self.max_budget)
        T, split = self.solve_dp_tables()

        best_speedup = 0.0
        best_N = 1
        best_choices: list[tuple[int, ...]] = []
        all_stats: dict[int, dict[str, float]] = {}

        for n in range(1, limit_N + 1):
            G_n = float(np.max(T[n, 0, :]))
            t_n = self.t_hardware.get(n, 1.0)
            choices_n = self.extract_tree_for_budget(n, T, split)
            depth_n = max([len(p) for p in choices_n], default=0)

            draft_cost = depth_n * self.draft_cost_c
            effective_latency = t_n + draft_cost
            speedup_n = G_n / effective_latency

            all_stats[n] = {
                "N": float(n),
                "expected_tokens": round(G_n, 4),
                "verification_t": round(t_n, 4),
                "draft_cost": round(draft_cost, 4),
                "effective_latency": round(effective_latency, 4),
                "speedup": round(speedup_n, 4),
            }

            if speedup_n > best_speedup:
                best_speedup = speedup_n
                best_N = n
                best_choices = choices_n

        from ..engine.medusa import compile_medusa_tree

        compiled = compile_medusa_tree(best_choices, self.num_heads)
        opt_stats = all_stats[best_N]

        return OptimalTreeResult(
            optimal_N=best_N,
            max_depth=compiled["max_depth"],
            expected_accepted_tokens=opt_stats["expected_tokens"],
            verification_cost_t=opt_stats["verification_t"],
            draft_cost=opt_stats["draft_cost"],
            speedup=opt_stats["speedup"],
            tree_choices=best_choices,
            compiled_tree=compiled,
            all_budget_stats=all_stats,
        )

    def solve_fixed_budget(self, N: int) -> OptimalTreeResult:
        """
        Extracts the optimal tree topology for a strictly fixed budget N.
        """
        N = min(max(1, N), self.max_budget)
        T, split = self.solve_dp_tables()
        choices = self.extract_tree_for_budget(N, T, split)
        from ..engine.medusa import compile_medusa_tree

        compiled = compile_medusa_tree(choices, self.num_heads)

        G_n = float(np.max(T[N, 0, :]))
        t_n = self.t_hardware.get(N, 1.0)
        draft_cost = compiled["max_depth"] * self.draft_cost_c
        speedup = G_n / (t_n + draft_cost)

        return OptimalTreeResult(
            optimal_N=N,
            max_depth=compiled["max_depth"],
            expected_accepted_tokens=round(G_n, 4),
            verification_cost_t=round(t_n, 4),
            draft_cost=round(draft_cost, 4),
            speedup=round(speedup, 4),
            tree_choices=choices,
            compiled_tree=compiled,
            all_budget_stats={},
        )
