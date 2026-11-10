"""
AxiomLM Phase 2 Task 2.2: Sequoia Dynamic Programming Tree Optimization Benchmark.

Loads the empirical hardware verification latency curves t(N) profiled in Task 2.1
across all architectures (SmolLM2-360M Q4/FP16, Qwen2.5-0.5B Q4, Llama-3.2-1B Q4),
runs Sequoia's exact DP solver to find the provably optimal tree topologies (N*, d*),
and evaluates expected token generation speedup.

Outputs:
  - benchmarks/phase2_task22_sequoia_dp_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import numpy as np

from axiomlm.speculative.sequoia_dp import SequoiaDPOptimizer


def run_sequoia_dp_benchmarks(
    task21_report_path: str,
    output_path: str,
    draft_cost_c: float = 0.02,
) -> dict[str, Any]:
    print("=" * 65)
    print("AXIOMLM PHASE 2 TASK 2.2: SEQUOIA DP TREE OPTIMIZATION BENCHMARK")
    print(f"Loading empirical verification latency from: {task21_report_path}")
    print(f"Draft Head Latency Cost c: {draft_cost_c:.3f}")
    print("=" * 65)

    with open(task21_report_path, "r") as f:
        task21_data = json.load(f)

    profiles = task21_data.get("profiles", {})
    results: dict[str, Any] = {
        "metadata": {
            "task": "Phase 2 Task 2.2: Sequoia Hardware-Aware Dynamic Programming Tree Optimization",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_hardware": task21_data.get("metadata", {}).get("hardware", {}),
            "draft_cost_c": draft_cost_c,
        },
        "optimal_trees": {},
    }

    # Positional draft head acceptance profiles P[head, rank] for testing:
    # 1. Standard Self-Distilled Medusa (realistic calibrated draft heads)
    p_medusa_standard = np.array(
        [
            [0.65, 0.18, 0.06, 0.02],
            [0.45, 0.12, 0.03, 0.01],
            [0.30, 0.07, 0.015, 0.005],
            [0.20, 0.04, 0.008, 0.002],
        ]
    )

    # 2. High-Accuracy Domain (e.g. Code / Structured JSON)
    p_medusa_code = np.array(
        [
            [0.85, 0.35, 0.15, 0.05],
            [0.72, 0.25, 0.08, 0.02],
            [0.58, 0.16, 0.04, 0.01],
            [0.42, 0.10, 0.02, 0.005],
        ]
    )

    for profile_name, data in profiles.items():
        if not profile_name.endswith("_q4") and not profile_name.endswith("_fp16"):
            continue

        # Extract empirical t(N) map
        t_map: dict[int, float] = {}
        for n_str, entry in data.items():
            n_val = int(n_str)
            t_map[n_val] = float(entry["normalized_cost_t_N"])

        print(f"\n>>> Solving for: {profile_name}")
        print(f"    Anchors: {t_map}")

        optimizer = SequoiaDPOptimizer(
            num_heads=4,
            max_branches=4,
            draft_cost_c=draft_cost_c,
            max_budget=32,
        )
        optimizer.set_hardware_latency(t_map)

        # Evaluate standard domain
        optimizer.set_acceptance_probabilities(p_medusa_standard)
        res_standard = optimizer.solve_optimal_tree(max_tree_size=32)

        # Evaluate code domain
        optimizer.set_acceptance_probabilities(p_medusa_code)
        res_code = optimizer.solve_optimal_tree(max_tree_size=32)

        print(
            f"    [Standard Text] Optimal N* = {res_standard.optimal_N:2d} (Depth {res_standard.max_depth}) | "
            f"Expected Tokens = {res_standard.expected_accepted_tokens:.3f} | Speedup = {res_standard.speedup:.3f}x"
        )
        print(
            f"    [Code / Syntax] Optimal N* = {res_code.optimal_N:2d} (Depth {res_code.max_depth}) | "
            f"Expected Tokens = {res_code.expected_accepted_tokens:.3f} | Speedup = {res_code.speedup:.3f}x"
        )

        results["optimal_trees"][profile_name] = {
            "standard_text": {
                "optimal_N": res_standard.optimal_N,
                "max_depth": res_standard.max_depth,
                "expected_tokens": res_standard.expected_accepted_tokens,
                "verification_cost_t": res_standard.verification_cost_t,
                "draft_cost": res_standard.draft_cost,
                "wall_clock_speedup": res_standard.speedup,
                "tree_choices": [list(choice) for choice in res_standard.tree_choices],
                "budget_sweep": res_standard.all_budget_stats,
            },
            "code_structured": {
                "optimal_N": res_code.optimal_N,
                "max_depth": res_code.max_depth,
                "expected_tokens": res_code.expected_accepted_tokens,
                "verification_cost_t": res_code.verification_cost_t,
                "draft_cost": res_code.draft_cost,
                "wall_clock_speedup": res_code.speedup,
                "tree_choices": [list(choice) for choice in res_code.tree_choices],
                "budget_sweep": res_code.all_budget_stats,
            },
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[SUCCESS] Sequoia DP report saved to: {output_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Sequoia Dynamic Programming Tree Optimization."
    )
    parser.add_argument(
        "--task21-report",
        type=str,
        default="benchmarks/phase2_task21_verification_latency_report.json",
        help="Path to Task 2.1 verification latency report.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="benchmarks/phase2_task22_sequoia_dp_report.json",
        help="Path to output JSON report.",
    )
    parser.add_argument(
        "--draft-cost-c",
        type=float,
        default=0.02,
        help="Relative draft head latency cost c = t_draft / t_verify(1).",
    )
    args = parser.parse_args()

    run_sequoia_dp_benchmarks(
        task21_report_path=args.task21_report,
        output_path=args.output_file,
        draft_cost_c=args.draft_cost_c,
    )


if __name__ == "__main__":
    main()
