"""
AxiomLM Phase 2 Task 2.4 Empirical Benchmark Campaign:
Sampling Without Replacement (Algorithm 2) & Power-Law Rejection Scaling.

Evaluates speculative decoding performance across temperatures:
    T in {0.0, 0.2, 0.6, 1.0}
on Apple Silicon Unified Memory Architecture (M3 Pro).

Measures:
- Wall-clock decode throughput (tok/s) and latency (ms/tok).
- Speculative acceptance rates (%) and mean accepted tokens per step.
- Empirical rejection rates across candidate ranks r_k for k in {1, 2, 3}.
- Sequoia power-law fit: r_k <= a * k^(-b) (exponent b, goodness-of-fit R^2).
- Token diversity / Shannon entropy across sampling temperatures.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from typing import Any

import numpy as np
import torch

from axiomlm.engine.medusa import (
    MedusaSpeculativeDecoder,
    train_medusa_heads,
)
from axiomlm.engine.paged_cache import StaticKVCache
from axiomlm.kernels import _METAL_MOD, is_mps_available
from axiomlm.models.transformer import ModelConfig, Transformer
from axiomlm.speculative.sampling import fit_power_law_rejection_bound


def get_hardware_telemetry() -> dict[str, Any]:
    """Captures Apple Silicon hardware profile."""
    telemetry: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "mps_available": is_mps_available(),
        "metal_module_loaded": _METAL_MOD is not None,
    }
    try:
        cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
        telemetry["chip"] = subprocess.check_output(cmd).decode().strip()
    except Exception:
        telemetry["chip"] = "Apple Silicon"
    return telemetry


def compute_entropy(token_ids: list[int], vocab_size: int) -> float:
    """Computes Shannon entropy of generated tokens in bits."""
    if len(token_ids) == 0:
        return 0.0
    counts = np.bincount(token_ids, minlength=vocab_size)
    probs = counts[counts > 0] / len(token_ids)
    return float(-np.sum(probs * np.log2(probs)))


def run_sampling_temperature_benchmark(
    num_trials: int = 5,
    max_new_tokens: int = 48,
    temperatures: list[float] | None = None,
) -> dict[str, Any]:
    """
    Executes the definitive Phase 2 Task 2.4 benchmark campaign across temperatures.
    """
    if temperatures is None:
        temperatures = [0.0, 0.2, 0.6, 1.0]

    device = torch.device(
        "mps" if is_mps_available() and _METAL_MOD is not None else "cpu"
    )
    print("=== AxiomLM Phase 2 Task 2.4 Benchmark: Sampling Without Replacement ===")
    print(f"Target Device: {device} | MPS Available: {is_mps_available()}")
    print(f"Temperatures evaluated: {temperatures} | Trials per temp: {num_trials}")

    # Standard benchmark architecture: 4 layers, 8 heads, 256 embd, 512 vocab
    vocab_size = 512
    config = ModelConfig(
        block_size=512,
        vocab_size=vocab_size,
        n_layer=4,
        n_head=8,
        n_embd=256,
    )
    torch.manual_seed(42)
    model = Transformer(config).to(device).eval()
    model.add_medusa_heads(num_heads=3)

    # Initialize weights to non-trivial states
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.orthogonal_(p)

    prompt_tokens = [12, 45, 89, 102, 33]

    # Generate training corpus from the backbone model
    tokens = list(prompt_tokens)
    head_dim = config.n_embd // config.n_head
    n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
    cache = StaticKVCache(
        max_batch_size=1,
        max_seq_len=256,
        n_layer=config.n_layer,
        n_kv_head=n_kv_head,
        head_dim=head_dim,
        device=device,
        dtype=torch.float32,
    )
    curr = torch.tensor([prompt_tokens], device=device)
    with torch.no_grad():
        out, _, _ = model(curr, kv_caches=[cache] * config.n_layer)
    for _ in range(96):
        next_t = out[:, -1, :].argmax(dim=-1).item()
        tokens.append(next_t)
        curr = torch.tensor([[next_t]], device=device)
        with torch.no_grad():
            out, _, _ = model(curr, kv_caches=[cache] * config.n_layer)

    train_tokens = torch.tensor([tokens], device=device, dtype=torch.long)
    print("\nTraining Medusa heads with Self-Distillation for temperature benchmark...")
    train_medusa_heads(
        model=model,
        input_ids=train_tokens,
        epochs=20,
        lr=2e-3,
        use_distillation=True,
        distill_weight=0.5,
        temperature=2.0,
    )
    print("Self-distillation training complete. Ready for temperature sweep.\n")

    results_by_temp: dict[str, Any] = {}

    for T in temperatures:
        print(f"\nEvaluating Temperature T = {T:.1f}...")
        throughput_list: list[float] = []
        latency_list: list[float] = []
        acceptance_list: list[float] = []
        mean_acc_len_list: list[float] = []
        entropy_list: list[float] = []
        all_rank_rejections: list[list[float]] = []

        tree_choices = [(0,), (1,), (2,), (0, 0), (0, 1), (1, 0)]
        decoder = MedusaSpeculativeDecoder(
            num_heads=3, temperature=T, tree_choices=tree_choices
        )

        # Warmup pass
        decoder.generate(
            model=model,
            prompt_tokens=prompt_tokens,
            max_new_tokens=16,
            device=device,
            temperature=T,
        )
        if device.type == "mps":
            torch.mps.synchronize()

        for trial in range(num_trials):
            gen = torch.Generator(device="cpu").manual_seed(2026 + trial)
            tokens, stats = decoder.generate(
                model=model,
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                device=device,
                temperature=T,
                generator=gen,
            )
            if device.type == "mps":
                torch.mps.synchronize()

            throughput_list.append(stats["tokens_per_second"])
            latency_list.append(stats["ms_per_token"])
            acceptance_list.append(stats["acceptance_rate_pct"])
            mean_acc_len_list.append(stats["mean_accepted_length"])
            gen_new = tokens[len(prompt_tokens) :]
            entropy_list.append(compute_entropy(gen_new, vocab_size))

            rank_rejs = stats.get("rejection_rates_per_rank", [])
            for r_idx, val in enumerate(rank_rejs):
                while len(all_rank_rejections) <= r_idx:
                    all_rank_rejections.append([])
                all_rank_rejections[r_idx].append(val)

        mean_tp = float(np.mean(throughput_list))
        std_tp = float(np.std(throughput_list))
        mean_lat = float(np.mean(latency_list))
        std_lat = float(np.std(latency_list))
        mean_acc = float(np.mean(acceptance_list))
        std_acc = float(np.std(acceptance_list))
        mean_len = float(np.mean(mean_acc_len_list))
        std_len = float(np.std(mean_acc_len_list))
        mean_ent = float(np.mean(entropy_list))
        std_ent = float(np.std(entropy_list))

        avg_rejs = [
            float(np.mean(vals)) for vals in all_rank_rejections if len(vals) > 0
        ]
        power_law = fit_power_law_rejection_bound(avg_rejs)

        print(
            f"  Throughput: {mean_tp:.2f} ± {std_tp:.2f} tok/s | "
            f"Latency: {mean_lat:.2f} ± {std_lat:.2f} ms/tok"
        )
        print(
            f"  Acceptance Rate: {mean_acc:.2f} ± {std_acc:.2f}% | "
            f"Accepted/Step: {mean_len:.2f} ± {std_len:.2f}"
        )
        print(f"  Entropy: {mean_ent:.3f} ± {std_ent:.3f} bits")
        if power_law["valid"]:
            print(
                f"  Power-Law Fit: exponent b = {power_law['exponent_b']:.3f}, "
                f"scale a = {power_law['scale_a']:.3f}, R^2 = {power_law['r_squared']:.3f}"
            )

        results_by_temp[str(T)] = {
            "temperature": T,
            "throughput_tok_per_sec": {"mean": mean_tp, "std": std_tp},
            "latency_ms_per_tok": {"mean": mean_lat, "std": std_lat},
            "acceptance_rate_pct": {"mean": mean_acc, "std": std_acc},
            "mean_accepted_length": {"mean": mean_len, "std": std_len},
            "shannon_entropy_bits": {"mean": mean_ent, "std": std_ent},
            "rejection_rates_per_rank": avg_rejs,
            "power_law_fit": power_law,
            "raw_trials": {
                "throughput": throughput_list,
                "latency_ms": latency_list,
                "acceptance_pct": acceptance_list,
            },
        }

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware_telemetry": get_hardware_telemetry(),
        "benchmark_config": {
            "num_trials": num_trials,
            "max_new_tokens": max_new_tokens,
            "temperatures": temperatures,
            "vocab_size": vocab_size,
            "model_config": config.__dict__,
        },
        "results_by_temperature": results_by_temp,
    }

    report_path = os.path.join(
        os.path.dirname(__file__), "phase2_task24_sampling_temperature_report.json"
    )
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[✓] Saved complete benchmark report to: {report_path}")

    return report


if __name__ == "__main__":
    run_sampling_temperature_benchmark()
