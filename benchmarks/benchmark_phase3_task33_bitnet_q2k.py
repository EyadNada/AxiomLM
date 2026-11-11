"""
Phase 3 Task 3.3 MLSys Rigorous Sub-4-Bit GEMV & GEMM Benchmark Suite.

Benchmarks BitNet 1.58-bit Ternary Weights and Q2_K 2-bit Quantized GEMV (M=1)
and Batched GEMM (M in [1, 32]) against MARLIN Q4, Q8_G64, and FP16 on Apple
Silicon GPU (Metal Performance Shaders / MPS).

Evaluates:
1. Microarchitectural DRAM bandwidth saturation across Llama-3.2-1B and SmolLM2 dimensions.
2. Arithmetic intensity and ALU unpacking throughput scaling across batch sizes M in [1, 2, 4, 8, 16, 32].
3. Model-level physical memory compression factors and theoretical tok/s upper bounds.
4. Latency speedups vs FP16 and Q8_G64.

Exports golden results to benchmarks/phase3_task33_bitnet_q2k_report.json.
"""

import json
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from axiomlm.kernels import (
    is_mps_available,
    pack_linear_q2_k,
    pack_linear_q4_marlin,
    pack_linear_q8_g64,
    pack_linear_ternary,
    qmm_q2_k,
    qmm_q4_marlin,
    qmm_ternary,
    qmv_q2_k,
    qmv_q4_marlin,
    qmv_q8_g64,
    qmv_ternary,
)


def benchmark_kernel(fn, warmup: int = 15, iters: int = 50) -> float:
    """Accurately measures MPS kernel latency in microseconds."""
    torch.mps.synchronize()
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.mps.synchronize()
    t1 = time.perf_counter()

    return ((t1 - t0) / iters) * 1e6


def run_benchmark():
    if not is_mps_available():
        print("[ERROR] Apple Silicon MPS not available. Exiting.")
        return

    device = torch.device("mps")
    peak_dram_bw_gb_s = 150.0  # Apple M3 Pro nominal memory bandwidth

    print("\n" + "=" * 95)
    print(
        "PHASE 3 TASK 3.3: BITNET 1.58-BIT & Q2_K SUB-4-BIT MLSYS COMPREHENSIVE BENCHMARK"
    )
    print(
        f"Hardware: Apple Silicon GPU (MPS) | Nominal DRAM Bandwidth: {peak_dram_bw_gb_s:.1f} GB/s"
    )
    print("=" * 95)

    # ------------------------------------------------------------------------
    # 1. Layer-Level GEMV Benchmark (M=1: Auto-regressive Generation Limit)
    # ------------------------------------------------------------------------
    layer_configs = [
        {"name": "SmolLM2 Attn Proj", "N": 576, "K": 576},
        {"name": "SmolLM2 MLP Gate/Up", "N": 1536, "K": 576},
        {"name": "SmolLM2 MLP Down", "N": 576, "K": 1536},
        {"name": "Llama-3.2-1B Attn Proj", "N": 2048, "K": 2048},
        {"name": "Llama-3.2-1B MLP Gate/Up", "N": 8192, "K": 2048},
        {"name": "Llama-3.2-1B MLP Down", "N": 2048, "K": 8192},
    ]

    print("\n--- 1. SINGLE-TOKEN GEMV (M=1): BANDWIDTH SATURATION & LATENCY ---")
    print(
        f"{'Layer Config':<24} | {'FP16 (us)':<10} | {'Q8 (us)':<8} | {'Q4 (us)':<8} | "
        f"{'Q2_K (us)':<9} | {'BitNet (us)':<11} | {'BitNet BW':<10} | {'BitNet %Peak':<12}"
    )
    print("-" * 95)

    gemv_results: list[dict[str, Any]] = []

    for cfg in layer_configs:
        N, K = cfg["N"], cfg["K"]
        name = cfg["name"]

        # Generate weights and inputs
        w = torch.randn(N, K, dtype=torch.float16, device=device)
        x = torch.randn(1, K, dtype=torch.float16, device=device)

        # Pack into various representations
        w_q8_p, w_q8_sb = pack_linear_q8_g64(w)
        w_q4_p, w_q4_sb = pack_linear_q4_marlin(w, clip_search=False)
        w_q2_p, w_q2_sb = pack_linear_q2_k(w)
        w_tern_p, w_tern_s = pack_linear_ternary(w)

        # Latency measurements (us)
        lat_fp16 = benchmark_kernel(lambda _x=x, _w=w: torch.matmul(_x, _w.t()))
        lat_q8 = benchmark_kernel(
            lambda _x=x, _p=w_q8_p, _sb=w_q8_sb: qmv_q8_g64(_x, _p, _sb)
        )
        lat_q4 = benchmark_kernel(
            lambda _x=x, _p=w_q4_p, _sb=w_q4_sb, _N=N: qmv_q4_marlin(_x, _p, _sb, _N)
        )
        lat_q2 = benchmark_kernel(
            lambda _x=x, _p=w_q2_p, _sb=w_q2_sb: qmv_q2_k(_x, _p, _sb)
        )
        lat_tern = benchmark_kernel(
            lambda _x=x, _p=w_tern_p, _s=w_tern_s: qmv_ternary(_x, _p, _s)
        )

        # Memory footprints (bytes)
        bytes_fp16 = N * K * 2
        bytes_q8 = N * (K // 4) * 4 + N * (K // 64) * 4  # packed + scale/bias
        bytes_q4 = N * (K // 8) * 4 + N * (K // 64) * 4  # packed + scale/bias
        bytes_q2 = N * (K // 4) * 1 + N * (K // 64) * 4  # 0.25 B/weight + scale/bias
        bytes_tern = N * (K // 4) * 1 + N * (K // 64) * 2  # 0.25 B/weight + scale

        bw_fp16 = (bytes_fp16 / (lat_fp16 * 1e-6)) / 1e9
        bw_q8 = (bytes_q8 / (lat_q8 * 1e-6)) / 1e9
        bw_q4 = (bytes_q4 / (lat_q4 * 1e-6)) / 1e9
        bw_q2 = (bytes_q2 / (lat_q2 * 1e-6)) / 1e9
        bw_tern = (bytes_tern / (lat_tern * 1e-6)) / 1e9

        pct_peak_tern = (bw_tern / peak_dram_bw_gb_s) * 100.0

        print(
            f"{name:<24} | {lat_fp16:<10.1f} | {lat_q8:<8.1f} | {lat_q4:<8.1f} | "
            f"{lat_q2:<9.1f} | {lat_tern:<11.1f} | {bw_tern:<7.1f} GB/s | {pct_peak_tern:<11.1f}%"
        )

        gemv_results.append(
            {
                "layer_name": name,
                "N": N,
                "K": K,
                "latency_us": {
                    "fp16": round(lat_fp16, 2),
                    "q8": round(lat_q8, 2),
                    "q4_marlin": round(lat_q4, 2),
                    "q2_k": round(lat_q2, 2),
                    "bitnet_1_58b": round(lat_tern, 2),
                },
                "bandwidth_gb_s": {
                    "fp16": round(bw_fp16, 2),
                    "q8": round(bw_q8, 2),
                    "q4_marlin": round(bw_q4, 2),
                    "q2_k": round(bw_q2, 2),
                    "bitnet_1_58b": round(bw_tern, 2),
                },
                "bandwidth_utilization_pct": round(pct_peak_tern, 1),
                "speedup_vs_fp16": round(lat_fp16 / max(lat_tern, 1e-6), 2),
                "speedup_vs_q8": round(lat_q8 / max(lat_tern, 1e-6), 2),
                "speedup_vs_q4": round(lat_q4 / max(lat_tern, 1e-6), 2),
                "memory_footprint_mb": {
                    "fp16": round(bytes_fp16 / (1024 * 1024), 3),
                    "q8": round(bytes_q8 / (1024 * 1024), 3),
                    "q4_marlin": round(bytes_q4 / (1024 * 1024), 3),
                    "q2_k": round(bytes_q2 / (1024 * 1024), 3),
                    "bitnet_1_58b": round(bytes_tern / (1024 * 1024), 3),
                },
            }
        )

    # ------------------------------------------------------------------------
    # 2. Batched GEMM Benchmark (M in [1, 2, 4, 8, 16, 32]: Arithmetic Scaling)
    # ------------------------------------------------------------------------
    batch_sizes = [1, 2, 4, 8, 16, 32]
    bench_N, bench_K = 2048, 2048  # Llama-3.2-1B hidden layer size

    print(f"\n--- 2. BATCHED GEMM SCALING (Llama-3.2-1B: N={bench_N}, K={bench_K}) ---")
    print(
        f"{'M (tokens)':<10} | {'FP16 (us)':<10} | {'Q4 (us)':<8} | {'Q2_K (us)':<9} | "
        f"{'BitNet (us)':<11} | {'BitNet Speedup':<14} | {'BitNet GFLOP/s':<14}"
    )
    print("-" * 95)

    gemm_results: list[dict[str, Any]] = []
    w_bench = torch.randn(bench_N, bench_K, dtype=torch.float16, device=device)
    w_bench_q4_p, w_bench_q4_sb = pack_linear_q4_marlin(w_bench, clip_search=False)
    w_bench_q2_p, w_bench_q2_sb = pack_linear_q2_k(w_bench)
    w_bench_tern_p, w_bench_tern_s = pack_linear_ternary(w_bench)

    for M in batch_sizes:
        x_m = torch.randn(M, bench_K, dtype=torch.float16, device=device)

        lat_fp16 = benchmark_kernel(
            lambda _xm=x_m, _w=w_bench: torch.matmul(_xm, _w.t())
        )
        lat_q4 = benchmark_kernel(
            lambda _xm=x_m, _p=w_bench_q4_p, _sb=w_bench_q4_sb, _N=bench_N: (
                qmm_q4_marlin(_xm, _p, _sb, _N)
            )
        )
        lat_q2 = benchmark_kernel(
            lambda _xm=x_m, _p=w_bench_q2_p, _sb=w_bench_q2_sb: qmm_q2_k(_xm, _p, _sb)
        )
        lat_tern = benchmark_kernel(
            lambda _xm=x_m, _p=w_bench_tern_p, _s=w_bench_tern_s: qmm_ternary(
                _xm, _p, _s
            )
        )

        flops = 2.0 * M * bench_N * bench_K
        gflops_tern = (flops / (lat_tern * 1e-6)) / 1e9
        speedup = lat_fp16 / max(lat_tern, 1e-6)

        print(
            f"{M:<10} | {lat_fp16:<10.1f} | {lat_q4:<8.1f} | {lat_q2:<9.1f} | "
            f"{lat_tern:<11.1f} | {speedup:<13.2f}x | {gflops_tern:<12.1f} GFLOP/s"
        )

        gemm_results.append(
            {
                "batch_size_M": M,
                "latency_us": {
                    "fp16": round(lat_fp16, 2),
                    "q4_marlin": round(lat_q4, 2),
                    "q2_k": round(lat_q2, 2),
                    "bitnet_1_58b": round(lat_tern, 2),
                },
                "speedup_vs_fp16": round(speedup, 2),
                "speedup_vs_q4": round(lat_q4 / max(lat_tern, 1e-6), 2),
                "throughput_gflops": round(gflops_tern, 2),
            }
        )

    # ------------------------------------------------------------------------
    # 3. Model-Level Memory Footprint & Physical Generation Limits
    # ------------------------------------------------------------------------
    # Llama-3.2-1B parameters: 1.23 billion
    # 16 layers, n_embd=2048, intermediate=8192, vocab=128256
    llama_params = 1_235_814_400
    model_footprints = {
        "fp16": llama_params * 2.0 / (1024**3),
        "q8": llama_params * 1.0625 / (1024**3),
        "q4_marlin": llama_params * 0.5625 / (1024**3),
        "q2_k": llama_params * 0.3125 / (1024**3),
        "bitnet_1_58b": llama_params * 0.28125 / (1024**3),
    }

    # Physical DRAM read limit (tok/s) = Bandwidth / Footprint
    theoretical_tok_s = {
        k: round(peak_dram_bw_gb_s / v, 1) for k, v in model_footprints.items()
    }

    print(
        "\n--- 3. FULL-MODEL FOOTPRINT & PHYSICAL DRAM TOK/S UPPER BOUND (Llama-3.2-1B) ---"
    )
    print(
        f"{'Format':<14} | {'Bytes/Weight':<14} | {'Model Size (GB)':<16} | {'Max DRAM tok/s':<16} | {'Compression':<12}"
    )
    print("-" * 80)
    for fmt, sz in model_footprints.items():
        bpw = sz * (1024**3) / llama_params
        comp = model_footprints["fp16"] / sz
        print(
            f"{fmt:<14} | {bpw:<14.5f} | {sz:<16.2f} | {theoretical_tok_s[fmt]:<16.1f} | {comp:<11.2f}x"
        )

    # ------------------------------------------------------------------------
    # 4. Export Golden Report
    # ------------------------------------------------------------------------
    report = {
        "benchmark_metadata": {
            "device": "Apple Silicon MPS (Metal Performance Shaders)",
            "hardware": "Apple M3 Pro",
            "nominal_dram_bandwidth_gb_s": peak_dram_bw_gb_s,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "gemv_single_token_layer_benchmarks": gemv_results,
        "gemm_batched_scaling_benchmarks": gemm_results,
        "full_model_theoretical_limits_llama_3_2_1b": {
            "parameters": llama_params,
            "model_size_gb": {k: round(v, 3) for k, v in model_footprints.items()},
            "physical_dram_tok_s_limit": theoretical_tok_s,
            "compression_factors_vs_fp16": {
                k: round(model_footprints["fp16"] / v, 2)
                for k, v in model_footprints.items()
            },
        },
    }

    out_path = os.path.join(
        os.path.dirname(__file__), "phase3_task33_bitnet_q2k_report.json"
    )
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 95)
    print(f"[SUCCESS] Golden benchmark report saved to: {out_path}")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    run_benchmark()
