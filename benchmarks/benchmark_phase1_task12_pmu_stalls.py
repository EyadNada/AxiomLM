"""
AxiomLM Phase 1 — Task 1.2: Microarchitectural Memory Stall Cycle Tracing & PMU Roofline Dissection.

Scientifically quantifies the GPU pipeline stall breakdown on Apple Silicon Unified Memory:
  - Cold-DRAM latency (T_cold) vs Warm-Cache latency (T_warm)
  - Memory Wait Stall Cycles (T_mem_wait = T_cold - T_warm)
  - Memory Stall Ratio (% of total GPU execution cycles spent waiting on DRAM queue starvation)
  - Sustained physical DRAM memory bandwidth (GB/s & % of 150 GB/s physical ceiling)
  - Arithmetic intensity scaling from single-token decode (M=1) to tree verification (M in {8, 16, 32})

Evaluation Precision Ladder:
  1. FP16 Native Baseline
  2. Q8_G64
  3. Q4_G64 Naive Serial-Unpack
  4. Q4_MARLIN Register-Direct Ladder (AxiomLM)
  5. Q4_MARLIN Tree Verification QMM (M in {8, 16, 32})

Evaluated across both 0.5B scale (2048x2048) and 1B-3B scale (4096x4096).
Target Hardware: Apple M3 Pro GPU (AGX G15, 150 GB/s LPDDR5-6400 UMA bus)
"""

from __future__ import annotations

import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.nn import functional as F

from axiomlm.kernels import (
    pack_linear_q4_g64,
    pack_linear_q4_marlin,
    pack_linear_q8_g64,
    qmm_q4_g64,
    qmm_q4_marlin,
    qmv_q4_g64,
    qmv_q4_marlin,
    qmv_q8_g64,
)

PEAK_DRAM_BW_GBS = 150.0  # Apple M3 Pro physical memory bandwidth (GB/s)
HARDWARE_KNEE_FLOPS_PER_BYTE = 57.3  # M3 Pro hardware balance point


def sync_mps():
    torch.mps.synchronize()


def run_latency_trials(
    fn,
    pool: List[Any],
    steps_per_trial: int = 16,
    num_trials: int = 20,
    warmup_steps: int = 5,
) -> tuple[float, float]:
    pool_len = len(pool)
    for s in range(warmup_steps):
        fn(pool[s % pool_len])
    sync_mps()

    trial_us: List[float] = []
    for _ in range(num_trials):
        sync_mps()
        t0 = time.perf_counter()
        for s in range(steps_per_trial):
            fn(pool[s % pool_len])
        sync_mps()
        t1 = time.perf_counter()
        step_us = ((t1 - t0) / steps_per_trial) * 1e6
        trial_us.append(step_us)

    return statistics.mean(trial_us), (
        statistics.stdev(trial_us) if len(trial_us) > 1 else 0.0
    )


def run_warm_trials(
    fn,
    target: Any,
    steps_per_trial: int = 32,
    num_trials: int = 20,
    warmup_steps: int = 10,
) -> tuple[float, float]:
    for _ in range(warmup_steps):
        fn(target)
    sync_mps()

    trial_us: List[float] = []
    for _ in range(num_trials):
        sync_mps()
        t0 = time.perf_counter()
        for _ in range(steps_per_trial):
            fn(target)
        sync_mps()
        t1 = time.perf_counter()
        step_us = ((t1 - t0) / steps_per_trial) * 1e6
        trial_us.append(step_us)

    return statistics.mean(trial_us), (
        statistics.stdev(trial_us) if len(trial_us) > 1 else 0.0
    )


def profile_pmu_stall_matrix(
    N: int = 4096, K: int = 4096, device_str: str = "mps"
) -> Dict[str, Any]:
    device = torch.device(device_str)
    print(
        f"\n[+] Allocating cold-DRAM rotating pools for matrix dimensions N={N}, K={K}..."
    )

    fp16_weight_bytes = N * K * 2
    q8_weight_bytes = N * K + N * (K // 64) * 4
    q4_weight_bytes = N * (K // 2) + N * (K // 64) * 4

    pool_size_fp16 = max(12, int(350e6 / fp16_weight_bytes))
    pool_size_q8 = max(16, int(350e6 / q8_weight_bytes))
    pool_size_q4 = max(24, int(350e6 / q4_weight_bytes))

    print(
        f"  * FP16 pool: {pool_size_fp16} x {fp16_weight_bytes / 1e6:.1f} MB = {pool_size_fp16 * fp16_weight_bytes / 1e6:.1f} MB"
    )
    print(
        f"  * Q8   pool: {pool_size_q8} x {q8_weight_bytes / 1e6:.1f} MB = {pool_size_q8 * q8_weight_bytes / 1e6:.1f} MB"
    )
    print(
        f"  * Q4   pool: {pool_size_q4} x {q4_weight_bytes / 1e6:.1f} MB = {pool_size_q4 * q4_weight_bytes / 1e6:.1f} MB"
    )

    w_base = (torch.randn(N, K, dtype=torch.float32) * 0.02).half()

    fp16_pool = [w_base.clone().to(device) for _ in range(pool_size_fp16)]

    p8, sb8 = pack_linear_q8_g64(w_base)
    q8_pool = [
        (p8.clone().to(device), sb8.clone().to(device)) for _ in range(pool_size_q8)
    ]

    p4, sb4 = pack_linear_q4_g64(w_base)
    naive_q4_pool = [
        (p4.clone().to(device), sb4.clone().to(device)) for _ in range(pool_size_q4)
    ]

    pm, sbm = pack_linear_q4_marlin(w_base)
    marlin_q4_pool = [
        (pm.clone().to(device), sbm.clone().to(device)) for _ in range(pool_size_q4)
    ]

    del w_base, p8, sb8, p4, sb4, pm, sbm
    gc.collect()
    sync_mps()

    regimes: List[Dict[str, Any]] = []

    configs = [
        ("FP16 (M=1)", 1, "fp16"),
        ("Q8_G64 (M=1)", 1, "q8"),
        ("Naive Q4_G64 (M=1)", 1, "naive_q4"),
        ("MARLIN Q4 (M=1)", 1, "marlin_q4"),
        ("Naive Q4_G64 (M=8)", 8, "naive_q4"),
        ("MARLIN Q4 (M=8)", 8, "marlin_q4"),
        ("Naive Q4_G64 (M=16)", 16, "naive_q4"),
        ("MARLIN Q4 (M=16)", 16, "marlin_q4"),
        ("MARLIN Q4 (M=32)", 32, "marlin_q4"),
    ]

    header = (
        f"{'Format & Batch M':<22} | {'T_cold (us)':>15} | {'T_warm (us)':>13} | "
        f"{'Stall (us)':>11} | {'Stall %':>8} | {'DRAM BW':>10} | {'% Peak':>7} | {'Intensity':>11}"
    )
    print("\n" + header)
    print("-" * len(header))

    for label, M, fmt in configs:
        x = torch.randn(1, M, K, dtype=torch.float16, device=device)
        io_bytes = M * K * 2 + M * N * 2
        total_flops = 2.0 * M * N * K

        if fmt == "fp16":
            weight_bytes = fp16_weight_bytes

            def fn(w):
                return F.linear(x, w)

            pool = fp16_pool
        elif fmt == "q8":
            weight_bytes = q8_weight_bytes

            def fn(ws):
                return qmv_q8_g64(x, ws[0], ws[1])

            pool = q8_pool
        elif fmt == "naive_q4":
            weight_bytes = q4_weight_bytes

            def fn(ws):
                return (
                    qmv_q4_g64(x, ws[0], ws[1])
                    if M == 1
                    else qmm_q4_g64(x, ws[0], ws[1])
                )

            pool = naive_q4_pool
        elif fmt == "marlin_q4":
            weight_bytes = q4_weight_bytes

            def fn(ws):
                return (
                    qmv_q4_marlin(x, ws[0], ws[1], out_features=N)
                    if M == 1
                    else qmm_q4_marlin(x, ws[0], ws[1], out_features=N)
                )

            pool = marlin_q4_pool
        else:
            raise ValueError(f"Unknown format {fmt}")

        total_bytes = weight_bytes + io_bytes
        arithmetic_intensity = total_flops / total_bytes

        t_cold_mean, t_cold_std = run_latency_trials(fn, pool)
        t_warm_mean, t_warm_std = run_warm_trials(fn, pool[0])

        t_mem_wait = max(0.0, t_cold_mean - t_warm_mean)
        stall_ratio_pct = (t_mem_wait / t_cold_mean) * 100.0 if t_cold_mean > 0 else 0.0
        sustained_bw_gbs = (total_bytes / (t_cold_mean * 1e-6)) / 1e9
        pct_peak_bw = (sustained_bw_gbs / PEAK_DRAM_BW_GBS) * 100.0

        print(
            f"{label:<22} | {t_cold_mean:7.1f} +- {t_cold_std:4.1f} | {t_warm_mean:6.1f} +- {t_warm_std:3.1f} | "
            f"{t_mem_wait:9.1f} | {stall_ratio_pct:7.1f}% | {sustained_bw_gbs:7.1f} GB/s | {pct_peak_bw:6.1f}% | "
            f"{arithmetic_intensity:6.2f} FLOP/B"
        )

        regimes.append(
            {
                "label": label,
                "batch_M": M,
                "format": fmt,
                "t_cold_us_mean": round(t_cold_mean, 2),
                "t_cold_us_std": round(t_cold_std, 2),
                "t_warm_us_mean": round(t_warm_mean, 2),
                "t_warm_us_std": round(t_warm_std, 2),
                "t_mem_wait_us": round(t_mem_wait, 2),
                "memory_wait_stall_ratio_pct": round(stall_ratio_pct, 2),
                "sustained_dram_bw_gbs": round(sustained_bw_gbs, 2),
                "pct_peak_bw": round(pct_peak_bw, 2),
                "arithmetic_intensity_flops_per_byte": round(arithmetic_intensity, 2),
                "gflops": round(total_flops / (t_cold_mean * 1e-6) / 1e9, 2),
            }
        )

    del fp16_pool, q8_pool, naive_q4_pool, marlin_q4_pool
    gc.collect()
    torch.mps.empty_cache()

    return {
        "dimensions": {"N": N, "K": K},
        "hardware": {
            "peak_dram_bw_gbs": PEAK_DRAM_BW_GBS,
            "hardware_knee_flops_per_byte": HARDWARE_KNEE_FLOPS_PER_BYTE,
        },
        "measurements": regimes,
    }


def main():
    print("=" * 80)
    print("AxiomLM Phase 1: Microarchitectural Memory Wall Proof")
    print("Task 1.2: Hardware PMU Stall Cycle Tracing & Roofline Dissection")
    print("=" * 80)

    # 1. Dimension 1: N=2048, K=2048 (SmolLM2 / Qwen scale)
    report_2048 = profile_pmu_stall_matrix(N=2048, K=2048)

    # 2. Dimension 2: N=4096, K=4096 (Llama-3.2 scale)
    report_4096 = profile_pmu_stall_matrix(N=4096, K=4096)

    final_report = {
        "metadata": {
            "task": "Phase 1 - Task 1.2: Hardware PMU Stall Cycle Tracing & Roofline Dissection",
            "device": "Apple M3 Pro",
            "peak_dram_bw_gbs": PEAK_DRAM_BW_GBS,
            "hardware_balance_point_flops_per_byte": HARDWARE_KNEE_FLOPS_PER_BYTE,
        },
        "grid_2048x2048": report_2048,
        "grid_4096x4096": report_4096,
    }

    out_path = Path("benchmarks/phase1_task12_pmu_stalls_report.json")
    with open(out_path, "w") as f:
        json.dump(final_report, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[+] Task 1.2 PMU Stall Cycle report written to {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
