"""
AxiomLM Phase 2 Task 2.1: Hardware-Aware Verification Latency Profiling (Apple M3 Pro).

Empirically measures the normalized verification latency cost function:
    t(N) = Latency(N) / Latency(1)
for N in [1, 2, 4, 8, 16, 32, 64] across real model architectures
(SmolLM2-360M, Qwen2.5-0.5B, Llama-3.2-1B) under:
  1. Topology-Aware Tree Verification (forward_tree_verify via gqa_tree_verify_mps)
  2. Batched Forward Verification (forward via QMM / batched decode)
  3. FP16 vs MARLIN Q4 comparison to expose the quantization verification gap.

Outputs:
  - benchmarks/phase2_task21_verification_latency_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import time
from typing import Any

import torch

from axiomlm.engine.paged_cache import StaticKVCache
from axiomlm.models.transformer import ModelConfig, Transformer


def get_hardware_info() -> dict[str, Any]:
    soc_name = "Apple Silicon"
    try:
        res = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            soc_name = res.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        soc_name = "Apple Silicon"

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "soc": soc_name,
        "torch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
    }


def create_model_configs() -> dict[str, ModelConfig]:
    return {
        "SmolLM2-360M": ModelConfig(
            arch="modern",
            n_layer=24,
            n_head=15,
            n_kv_head=5,
            n_embd=960,
            intermediate_size=2560,
            vocab_size=49152,
            block_size=512,
            use_fused_kernels=True,
            qkv_bias=True,
        ),
        "Qwen2.5-0.5B": ModelConfig(
            arch="modern",
            n_layer=24,
            n_head=14,
            n_kv_head=2,
            n_embd=896,
            intermediate_size=4864,
            vocab_size=151936,
            block_size=512,
            use_fused_kernels=True,
            qkv_bias=True,
        ),
        "Llama-3.2-1B": ModelConfig(
            arch="modern",
            n_layer=16,
            n_head=32,
            n_kv_head=8,
            n_embd=2048,
            intermediate_size=8192,
            vocab_size=128256,
            block_size=512,
            use_fused_kernels=True,
            qkv_bias=True,
        ),
    }


def build_candidate_tree(N: int) -> tuple[list[int], list[int]]:
    """
    Constructs a valid branching candidate tree topology of size N (N <= 32).
    Generates depth offsets and 32-bit ancestor bitmasks.
    """
    tree_depths = [0] * N
    tree_mask = [0] * N
    tree_mask[0] = 1  # Root attends to itself

    if N == 1:
        return tree_depths, tree_mask

    # Branching structure: root has 4 children, each child has up to 3 children, etc.
    parent = [-1] * N
    for i in range(1, N):
        if i <= 4:
            p = 0
        elif i <= 16:
            p = 1 + (i - 5) % 4
        else:
            p = 5 + (i - 17) % 12
        parent[i] = p
        tree_depths[i] = tree_depths[p] + 1
        tree_mask[i] = tree_mask[p] | (1 << i)

    return tree_depths, tree_mask


def profile_verification_for_config(
    model_name: str,
    cfg: ModelConfig,
    quant_mode: str | None,
    eval_tree: bool,
    prefix_len: int = 128,
    warmup_iters: int = 10,
    timed_iters: int = 30,
) -> dict[str, Any]:
    device = torch.device("mps")
    print("\n=======================================================")
    print(
        f"Profiling: {model_name} | Mode: {quant_mode or 'FP16'} | Prefix Len: {prefix_len}"
    )
    print("=======================================================")

    torch.manual_seed(42)
    model = Transformer(cfg).to(device=device, dtype=torch.float16).eval()
    if quant_mode is not None:
        model.quantize(quant_mode)

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

    # Prefill prefix
    prefix_tokens = torch.randint(
        0, min(1000, cfg.vocab_size), (1, prefix_len), device=device
    )
    with torch.no_grad():
        model(prefix_tokens, kv_caches=[cache] * cfg.n_layer)

    results_by_N: dict[str, Any] = {}
    base_latency_ms: float | None = None

    N_values = [1, 2, 4, 8, 16, 32] if eval_tree else [1, 2, 4, 8, 16, 32, 64]

    for N in N_values:
        cand_tokens = torch.randint(0, min(1000, cfg.vocab_size), (1, N), device=device)
        tree_depths, tree_mask = build_candidate_tree(N) if eval_tree else ([], [])

        # Warmup loop
        for _ in range(warmup_iters):
            with torch.no_grad():
                if eval_tree:
                    model.forward_tree_verify(
                        cand_tokens,
                        kv_caches=[cache] * cfg.n_layer,
                        tree_depths=tree_depths,
                        tree_mask=tree_mask,
                        return_medusa=False,
                    )
                else:
                    model(
                        cand_tokens,
                        kv_caches=[cache] * cfg.n_layer,
                        return_all_logits=True,
                    )
        torch.mps.synchronize()

        # Timed measurement loop
        latencies_ms: list[float] = []
        for _ in range(timed_iters):
            torch.mps.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                if eval_tree:
                    model.forward_tree_verify(
                        cand_tokens,
                        kv_caches=[cache] * cfg.n_layer,
                        tree_depths=tree_depths,
                        tree_mask=tree_mask,
                        return_medusa=False,
                    )
                else:
                    model(
                        cand_tokens,
                        kv_caches=[cache] * cfg.n_layer,
                        return_all_logits=True,
                    )
            torch.mps.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        latencies_ms.sort()
        mean_ms = sum(latencies_ms) / len(latencies_ms)
        var = sum((x - mean_ms) ** 2 for x in latencies_ms) / (len(latencies_ms) - 1)
        std_ms = math.sqrt(var) if len(latencies_ms) > 1 else 0.0

        p50_ms = latencies_ms[len(latencies_ms) // 2]
        p95_ms = latencies_ms[int(len(latencies_ms) * 0.95)]
        p99_ms = latencies_ms[int(len(latencies_ms) * 0.99)]

        if N == 1:
            base_latency_ms = mean_ms

        t_N = (
            mean_ms / base_latency_ms
            if base_latency_ms and base_latency_ms > 0
            else 1.0
        )
        cand_rate = N / (mean_ms / 1000.0)
        speedup_potential = N / t_N
        incremental_cost = (
            (mean_ms - base_latency_ms) / (N - 1)
            if (N > 1 and base_latency_ms)
            else 0.0
        )

        print(
            f"  N={N:2d} | Mean={mean_ms:6.2f} ± {std_ms:4.2f} ms | "
            f"t(N)={t_N:5.2f}x | Rate={cand_rate:7.1f} cand/s | "
            f"Max Pot={speedup_potential:5.2f}x | Δt={incremental_cost:5.3f} ms/cand"
        )

        results_by_N[str(N)] = {
            "N": N,
            "mean_latency_ms": round(mean_ms, 3),
            "std_latency_ms": round(std_ms, 3),
            "p50_latency_ms": round(p50_ms, 3),
            "p95_latency_ms": round(p95_ms, 3),
            "p99_latency_ms": round(p99_ms, 3),
            "normalized_cost_t_N": round(t_N, 4),
            "verification_rate_cand_per_s": round(cand_rate, 1),
            "ideal_speedup_ceiling": round(speedup_potential, 3),
            "incremental_cost_ms_per_cand": round(incremental_cost, 4),
        }

    # Clean up model from MPS memory
    del model
    del cache
    torch.mps.empty_cache()

    return results_by_N


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile verification latency scaling t(N) on Apple Silicon."
    )
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=128,
        help="Committed prefix length in KV cache.",
    )
    parser.add_argument(
        "--warmup-iters", type=int, default=10, help="Number of warmup iterations."
    )
    parser.add_argument(
        "--timed-iters",
        type=int,
        default=30,
        help="Number of timed benchmark iterations.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="benchmarks/phase2_task21_verification_latency_report.json",
        help="Path to output JSON report.",
    )
    args = parser.parse_args()

    hw_info = get_hardware_info()
    print("=" * 60)
    print("AXIOMLM PHASE 2 TASK 2.1: VERIFICATION LATENCY BENCHMARK")
    print(f"Device: {hw_info['soc']} ({hw_info['processor']})")
    print(
        f"Prefix Length: {args.prefix_len} | Warmup: {args.warmup_iters} | Timed: {args.timed_iters}"
    )
    print("=" * 60)

    configs = create_model_configs()
    report: dict[str, Any] = {
        "metadata": {
            "task": "Phase 2 Task 2.1: Empirical Hardware Verification Latency Curve t(N)",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hardware": hw_info,
            "benchmark_params": {
                "prefix_len": args.prefix_len,
                "warmup_iters": args.warmup_iters,
                "timed_iters": args.timed_iters,
            },
        },
        "profiles": {},
    }

    # 1. Profile SmolLM2-360M Tree Verification Q4
    report["profiles"]["SmolLM2-360M_tree_q4"] = profile_verification_for_config(
        "SmolLM2-360M",
        configs["SmolLM2-360M"],
        quant_mode="q4",
        eval_tree=True,
        prefix_len=args.prefix_len,
        warmup_iters=args.warmup_iters,
        timed_iters=args.timed_iters,
    )

    # 2. Profile SmolLM2-360M Tree Verification FP16
    report["profiles"]["SmolLM2-360M_tree_fp16"] = profile_verification_for_config(
        "SmolLM2-360M",
        configs["SmolLM2-360M"],
        quant_mode=None,
        eval_tree=True,
        prefix_len=args.prefix_len,
        warmup_iters=args.warmup_iters,
        timed_iters=args.timed_iters,
    )

    # 3. Profile Qwen2.5-0.5B Tree Verification Q4
    report["profiles"]["Qwen2.5-0.5B_tree_q4"] = profile_verification_for_config(
        "Qwen2.5-0.5B",
        configs["Qwen2.5-0.5B"],
        quant_mode="q4",
        eval_tree=True,
        prefix_len=args.prefix_len,
        warmup_iters=args.warmup_iters,
        timed_iters=args.timed_iters,
    )

    # 4. Profile Llama-3.2-1B Tree Verification Q4
    report["profiles"]["Llama-3.2-1B_tree_q4"] = profile_verification_for_config(
        "Llama-3.2-1B",
        configs["Llama-3.2-1B"],
        quant_mode="q4",
        eval_tree=True,
        prefix_len=args.prefix_len,
        warmup_iters=args.warmup_iters,
        timed_iters=args.timed_iters,
    )

    # 5. Profile SmolLM2-360M Batched Verification Q4 (N up to 64)
    report["profiles"]["SmolLM2-360M_batched_q4"] = profile_verification_for_config(
        "SmolLM2-360M",
        configs["SmolLM2-360M"],
        quant_mode="q4",
        eval_tree=False,
        prefix_len=args.prefix_len,
        warmup_iters=args.warmup_iters,
        timed_iters=args.timed_iters,
    )

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[SUCCESS] Verification latency profile saved to {args.output_file}")


if __name__ == "__main__":
    main()
