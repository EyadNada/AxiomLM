"""
AxiomLM Phase 2 Task 2.3: Medusa-2 KL-Divergence Self-Distillation Benchmark.

Trains auxiliary Medusa heads under:
  1. Baseline: Pure Cross-Entropy loss
  2. Medusa-2: Cross-Entropy + Soft KL-Divergence Self-Distillation against frozen base logits.

Empirically measures:
  - Loss trajectory (CE loss and KL loss across optimization steps)
  - Resulting Positional Acceptance Probability Matrix P[head, rank]
  - Acceptance gain of soft distillation over pure cross-entropy
  - Feeds empirical P[d, b] into Sequoia DP optimizer to verify end-to-end tree speedup.

Outputs:
  - benchmarks/phase2_task23_medusa_distillation_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import numpy as np
import torch

from axiomlm.engine.medusa import MedusaSpeculativeDecoder, train_medusa_heads
from axiomlm.engine.paged_cache import StaticKVCache
from axiomlm.models.medusa import evaluate_medusa_acceptance_matrix
from axiomlm.models.transformer import ModelConfig, Transformer
from axiomlm.speculative.sequoia_dp import SequoiaDPOptimizer


def run_medusa_distillation_benchmark(
    output_path: str = "benchmarks/phase2_task23_medusa_distillation_report.json",
    epochs: int = 25,
    lr: float = 2e-2,
    num_heads: int = 4,
    device_str: str = "mps",
) -> dict[str, Any]:
    print("=" * 65)
    print("AXIOMLM PHASE 2 TASK 2.3: MEDUSA-2 SELF-DISTILLATION BENCHMARK")
    print(f"Device: {device_str} | Heads: {num_heads} | Epochs: {epochs} | LR: {lr}")
    print("=" * 65)

    device = torch.device(device_str)
    torch.manual_seed(2026)

    # Representative model configuration
    cfg = ModelConfig(
        arch="modern",
        n_layer=4,
        n_head=8,
        n_kv_head=2,
        n_embd=512,
        intermediate_size=1024,
        vocab_size=1024,
        block_size=512,
        use_fused_kernels=True,
        qkv_bias=True,
    )

    base_model = Transformer(cfg).to(device=device, dtype=torch.float16).eval()
    base_model.add_medusa_heads(num_heads=num_heads)

    # Generate synthetic corpus from the backbone model
    prompt = [10, 20, 30, 40, 50]
    head_dim = cfg.n_embd // cfg.n_head
    cache = StaticKVCache(
        max_batch_size=1,
        max_seq_len=512,
        n_layer=cfg.n_layer,
        n_kv_head=cfg.n_kv_head,
        head_dim=head_dim,
        device=device,
        dtype=torch.float16,
    )

    curr = torch.tensor([prompt], device=device)
    with torch.no_grad():
        out, _, _ = base_model(curr, kv_caches=[cache] * cfg.n_layer)

    tokens = list(prompt)
    for _ in range(128):
        next_t = out[:, -1, :].argmax(dim=-1).item()
        tokens.append(next_t)
        curr = torch.tensor([[next_t]], device=device)
        with torch.no_grad():
            out, _, _ = base_model(curr, kv_caches=[cache] * cfg.n_layer)

    train_tokens = torch.tensor([tokens], device=device, dtype=torch.long)

    # 1. Train with Pure Cross-Entropy (Baseline)
    print("\n>>> 1. Training Baseline Medusa (Pure Hard Cross-Entropy)...")
    torch.manual_seed(2026)
    model_ce = Transformer(cfg).to(device=device, dtype=torch.float16).eval()
    model_ce.load_state_dict(base_model.state_dict(), strict=False)
    model_ce.add_medusa_heads(num_heads=num_heads)

    stats_ce = train_medusa_heads(
        model=model_ce,
        input_ids=train_tokens,
        epochs=epochs,
        lr=lr,
        freeze_backbone=True,
        use_distillation=False,
        return_stats=True,
    )
    assert isinstance(stats_ce, dict)

    # 2. Train with Medusa-2 Self-Distillation (CE + Soft KL)
    print(
        "\n>>> 2. Training Medusa-2 (KL Self-Distillation against Frozen Base Logits)..."
    )
    torch.manual_seed(2026)
    model_distill = Transformer(cfg).to(device=device, dtype=torch.float16).eval()
    model_distill.load_state_dict(base_model.state_dict(), strict=False)
    model_distill.add_medusa_heads(num_heads=num_heads)

    stats_distill = train_medusa_heads(
        model=model_distill,
        input_ids=train_tokens,
        epochs=epochs,
        lr=lr,
        freeze_backbone=True,
        use_distillation=True,
        temperature=2.0,
        distill_weight=0.5,
        return_stats=True,
    )
    assert isinstance(stats_distill, dict)

    # 3. Evaluate Positional Acceptance Matrix P[head, rank]
    print("\n>>> 3. Evaluating Empirical Acceptance Matrix P[head, rank]...")
    with torch.no_grad():
        base_logits_eval, med_ce_eval, _ = model_ce.forward_with_medusa(train_tokens)
        _, med_distill_eval, _ = model_distill.forward_with_medusa(train_tokens)

    eval_ce = evaluate_medusa_acceptance_matrix(
        med_ce_eval, base_logits_eval, max_topk=4
    )
    eval_distill = evaluate_medusa_acceptance_matrix(
        med_distill_eval, base_logits_eval, max_topk=4
    )

    p_ce = eval_ce["acceptance_matrix"]
    p_distill = eval_distill["acceptance_matrix"]

    print("\nBaseline Pure CE Acceptance Matrix P[d, b]:")
    print(np.round(p_ce, 4))

    print("\nMedusa-2 Self-Distilled Acceptance Matrix P[d, b]:")
    print(np.round(p_distill, 4))

    # 4. Feed empirical P[d, b] into Sequoia DP Optimizer to verify speedup
    t_profile_m3 = {1: 1.000, 2: 1.280, 4: 1.385, 8: 1.488, 16: 2.315, 32: 3.926}
    opt_ce = SequoiaDPOptimizer(num_heads=num_heads, max_branches=4, draft_cost_c=0.02)
    opt_ce.set_hardware_latency(t_profile_m3)
    opt_ce.set_acceptance_probabilities(p_ce)
    res_ce = opt_ce.solve_optimal_tree(max_tree_size=16)

    opt_distill = SequoiaDPOptimizer(
        num_heads=num_heads, max_branches=4, draft_cost_c=0.02
    )
    opt_distill.set_hardware_latency(t_profile_m3)
    opt_distill.set_acceptance_probabilities(p_distill)
    res_distill = opt_distill.solve_optimal_tree(max_tree_size=16)

    print("\n>>> 4. Sequoia DP Optimization Results:")
    print(
        f"  Baseline CE:      N*={res_ce.optimal_N:2d} | "
        f"G(N*)={res_ce.expected_accepted_tokens:.3f} | Speedup={res_ce.speedup:.3f}x"
    )
    print(
        f"  Medusa-2 Distill: N*={res_distill.optimal_N:2d} | "
        f"G(N*)={res_distill.expected_accepted_tokens:.3f} | Speedup={res_distill.speedup:.3f}x"
    )

    # 5. Measure real wall-clock decoding throughput with MedusaSpeculativeDecoder
    print("\n>>> 5. Wall-Clock Speculative Decoding Run...")
    dec_ce = MedusaSpeculativeDecoder(num_heads=num_heads)
    _, speed_ce = dec_ce.generate(
        model=model_ce,
        prompt_tokens=prompt,
        max_new_tokens=32,
        device=device,
        tree_choices=res_ce.tree_choices,
    )

    dec_distill = MedusaSpeculativeDecoder(num_heads=num_heads)
    _, speed_distill = dec_distill.generate(
        model=model_distill,
        prompt_tokens=prompt,
        max_new_tokens=32,
        device=device,
        tree_choices=res_distill.tree_choices,
    )

    print(
        f"  Baseline CE Tok/s:      {speed_ce['tokens_per_second']:.1f} tok/s | "
        f"Acceptance: {speed_ce['acceptance_rate_pct']:.1f}%"
    )
    print(
        f"  Medusa-2 Distill Tok/s: {speed_distill['tokens_per_second']:.1f} tok/s | "
        f"Acceptance: {speed_distill['acceptance_rate_pct']:.1f}%"
    )

    report = {
        "metadata": {
            "task": "Phase 2 Task 2.3: Medusa-2 KL-Divergence Self-Distillation",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": str(device),
            "hyperparameters": {
                "epochs": epochs,
                "learning_rate": lr,
                "num_heads": num_heads,
                "temperature": 2.0,
                "distill_weight": 0.5,
            },
        },
        "training_curves": {
            "baseline_ce_losses": stats_ce["losses"],
            "distill_combined_losses": stats_distill["losses"],
            "distill_ce_history": stats_distill["ce_history"],
            "distill_kl_history": stats_distill["kl_history"],
        },
        "empirical_acceptance_matrix": {
            "baseline_ce": p_ce.tolist(),
            "medusa_2_distill": p_distill.tolist(),
            "top1_acceptance_ce": eval_ce["top1_accuracies"],
            "top1_acceptance_distill": eval_distill["top1_accuracies"],
        },
        "sequoia_dp_projection": {
            "baseline_ce": {
                "optimal_N": res_ce.optimal_N,
                "expected_tokens": res_ce.expected_accepted_tokens,
                "speedup": res_ce.speedup,
            },
            "medusa_2_distill": {
                "optimal_N": res_distill.optimal_N,
                "expected_tokens": res_distill.expected_accepted_tokens,
                "speedup": res_distill.speedup,
            },
        },
        "wall_clock_decoding": {
            "baseline_ce": speed_ce,
            "medusa_2_distill": speed_distill,
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[SUCCESS] Report successfully generated at: {output_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Medusa-2 KL-Divergence Self-Distillation Benchmark."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="benchmarks/phase2_task23_medusa_distillation_report.json",
        help="Path to output JSON report.",
    )
    parser.add_argument(
        "--epochs", type=int, default=25, help="Number of training epochs."
    )
    parser.add_argument("--lr", type=float, default=2e-2, help="Learning rate.")
    parser.add_argument(
        "--num-heads", type=int, default=4, help="Number of Medusa heads."
    )
    args = parser.parse_args()

    run_medusa_distillation_benchmark(
        output_path=args.output_file,
        epochs=args.epochs,
        lr=args.lr,
        num_heads=args.num_heads,
    )


if __name__ == "__main__":
    main()
