"""
AxiomLM Phase 1 — Task 1.3: Empirical Bandwidth Saturation Curve Benchmark & Publication Vector Generator.

Microarchitectural proof of the Quantization Bandwidth Collapse on Apple Silicon Unified Memory:
  - Systematically profiles sustained physical DRAM bandwidth (GB/s & % of 150 GB/s peak)
    across the bitwidth ladder:
      * 16-bit (Native FP16)
      * 8-bit  (Q8_G64)
      * 4-bit  (Naive Q4_G64 serial unpack)
      * 4-bit  (AxiomLM MARLIN register-direct ladder)
    across matrix candidate / batch sizes M in {1, 2, 4, 8, 16, 32, 64}.
  - Cold-DRAM rotating pool (>350 MB) guarantees zero L2/SLC cache residency.
  - Generates publication-ready vector graphic: benchmarks/bandwidth_saturation_curve.svg
  - Generates comprehensive report: benchmarks/phase1_task13_bandwidth_saturation_report.json

Target Hardware: Apple M3 Pro GPU (AGX G15, 150 GB/s LPDDR5-6400 UMA bus)
"""

from __future__ import annotations

import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")  # Headless rendering
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from axiomlm.kernels import (
    pack_linear_q4_g64,
    pack_linear_q4_marlin,
    pack_linear_q8_g64,
    qmm_q8_g64,
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
) -> tuple[float, float, float, float]:
    """
    Measures latency with cold-DRAM cache busting across a rotating pool.
    Returns (mean_us, stdev_us, min_us, max_us).
    """
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

    return (
        statistics.mean(trial_us),
        statistics.stdev(trial_us) if len(trial_us) > 1 else 0.0,
        min(trial_us),
        max(trial_us),
    )


def generate_saturation_curve(
    N: int = 4096, K: int = 4096, device_str: str = "mps"
) -> Dict[str, Any]:
    device = torch.device(device_str)
    print(f"\n[+] Allocating cold-DRAM rotating pools for N={N}, K={K}...")

    fp16_weight_bytes = N * K * 2
    q8_weight_bytes = N * K + N * (K // 64) * 4
    q4_weight_bytes = N * (K // 2) + N * (K // 64) * 4

    pool_size_fp16 = max(12, int(350e6 / fp16_weight_bytes))
    pool_size_q8 = max(18, int(350e6 / q8_weight_bytes))
    pool_size_q4 = max(36, int(350e6 / q4_weight_bytes))

    print(
        f"  * FP16 pool: {pool_size_fp16} matrices ({pool_size_fp16 * fp16_weight_bytes / 1e6:.1f} MB)"
    )
    print(
        f"  * Q8   pool: {pool_size_q8} matrices ({pool_size_q8 * q8_weight_bytes / 1e6:.1f} MB)"
    )
    print(
        f"  * Q4   pool: {pool_size_q4} matrices ({pool_size_q4 * q4_weight_bytes / 1e6:.1f} MB)"
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

    m_values = [1, 2, 4, 8, 16, 32, 64]
    formats = [
        ("FP16 (16-bit)", "fp16", fp16_weight_bytes, fp16_pool),
        ("Q8_G64 (8-bit)", "q8", q8_weight_bytes, q8_pool),
        ("Naive Q4_G64 (4-bit)", "naive_q4", q4_weight_bytes, naive_q4_pool),
        ("AxiomLM MARLIN (4-bit)", "marlin_q4", q4_weight_bytes, marlin_q4_pool),
    ]

    results_by_format: Dict[str, List[Dict[str, Any]]] = {fmt[0]: [] for fmt in formats}

    header = (
        f"{'M':>3} | {'Format':<22} | {'Latency (us)':>15} | "
        f"{'DRAM BW (GB/s)':>15} | {'% Peak':>7} | {'GFLOP/s':>9} | {'FLOP/Byte':>10}"
    )
    print("\n" + header)
    print("-" * len(header))

    for M in m_values:
        x = torch.randn(1, M, K, dtype=torch.float16, device=device)
        io_bytes = M * K * 2 + M * N * 2
        total_flops = 2.0 * M * N * K

        for fmt_label, fmt_key, weight_bytes, pool in formats:
            if fmt_key == "fp16":

                def fn(w):
                    return F.linear(x, w)

            elif fmt_key == "q8":

                def fn(ws):
                    return (
                        qmv_q8_g64(x, ws[0], ws[1])
                        if M == 1
                        else qmm_q8_g64(x, ws[0], ws[1])
                    )

            elif fmt_key == "naive_q4":

                def fn(ws):
                    return (
                        qmv_q4_g64(x, ws[0], ws[1])
                        if M == 1
                        else qmm_q4_g64(x, ws[0], ws[1])
                    )

            elif fmt_key == "marlin_q4":

                def fn(ws):
                    return (
                        qmv_q4_marlin(x, ws[0], ws[1], out_features=N)
                        if M == 1
                        else qmm_q4_marlin(x, ws[0], ws[1], out_features=N)
                    )

            total_bytes = weight_bytes + io_bytes
            arithmetic_intensity = total_flops / total_bytes

            mean_us, std_us, min_us, max_us = run_latency_trials(fn, pool)
            sustained_bw_gbs = (total_bytes / (mean_us * 1e-6)) / 1e9
            pct_peak = (sustained_bw_gbs / PEAK_DRAM_BW_GBS) * 100.0
            gflops = total_flops / (mean_us * 1e-6) / 1e9

            print(
                f"{M:3d} | {fmt_label:<22} | {mean_us:7.1f} +- {std_us:4.1f} | "
                f"{sustained_bw_gbs:12.1f} GB/s | {pct_peak:6.1f}% | {gflops:9.1f} | {arithmetic_intensity:8.2f}"
            )

            results_by_format[fmt_label].append(
                {
                    "M": M,
                    "latency_us_mean": round(mean_us, 2),
                    "latency_us_std": round(std_us, 2),
                    "latency_us_min": round(min_us, 2),
                    "latency_us_max": round(max_us, 2),
                    "sustained_bw_gbs": round(sustained_bw_gbs, 2),
                    "pct_peak_bw": round(pct_peak, 2),
                    "gflops": round(gflops, 2),
                    "arithmetic_intensity_flops_per_byte": round(
                        arithmetic_intensity, 2
                    ),
                }
            )

    del fp16_pool, q8_pool, naive_q4_pool, marlin_q4_pool
    gc.collect()
    torch.mps.empty_cache()

    return {
        "dimensions": {"N": N, "K": K},
        "m_values": m_values,
        "peak_bandwidth_gbs": PEAK_DRAM_BW_GBS,
        "hardware_knee_flops_per_byte": HARDWARE_KNEE_FLOPS_PER_BYTE,
        "curves": results_by_format,
    }


def plot_bandwidth_saturation(
    data: Dict[str, Any], output_prefix: str = "benchmarks/bandwidth_saturation_curve"
):
    curves = data["curves"]
    m_values = data["m_values"]
    peak_bw = data["peak_bandwidth_gbs"]

    # Publication-quality matplotlib styling (ACM/IEEE MLSys style)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "lines.linewidth": 2.2,
            "lines.markersize": 7,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
        }
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Color palette
    colors = {
        "FP16 (16-bit)": "#1f77b4",  # Deep Blue
        "Q8_G64 (8-bit)": "#2ca02c",  # Forest Green
        "Naive Q4_G64 (4-bit)": "#d62728",  # Crimson Red
        "AxiomLM MARLIN (4-bit)": "#ff7f0e",  # High-contrast Amber
    }
    markers = {
        "FP16 (16-bit)": "o",
        "Q8_G64 (8-bit)": "s",
        "Naive Q4_G64 (4-bit)": "^",
        "AxiomLM MARLIN (4-bit)": "D",
    }

    # Subplot 1: Sustained DRAM Bandwidth vs Batch / Tree Size M
    for label, points in curves.items():
        bw = [p["sustained_bw_gbs"] for p in points]
        ax1.plot(
            m_values,
            bw,
            label=label,
            color=colors[label],
            marker=markers[label],
        )

    ax1.axhline(
        peak_bw,
        color="#7f7f7f",
        linestyle=":",
        linewidth=1.8,
        label=f"Physical DRAM Ceiling ({peak_bw:.0f} GB/s)",
    )
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(m_values)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_xlabel("Batch / Tree Verification Tokens (M)")
    ax1.set_ylabel("Sustained DRAM Bandwidth (GB/s)")
    ax1.set_title("(a) Sustained Memory Bandwidth vs Tree Tokens M")
    ax1.set_ylim(0, 165)
    ax1.grid(True)
    ax1.legend(loc="lower left", framealpha=0.9)

    # Subplot 2: Bandwidth Efficiency (% Peak) vs Arithmetic Intensity
    for label, points in curves.items():
        intensities = [p["arithmetic_intensity_flops_per_byte"] for p in points]
        pct = [p["pct_peak_bw"] for p in points]
        ax2.plot(
            intensities,
            pct,
            label=label,
            color=colors[label],
            marker=markers[label],
        )

    ax2.axvline(
        data["hardware_knee_flops_per_byte"],
        color="#8c564b",
        linestyle="--",
        linewidth=1.8,
        label=f"Roofline Balance Knee ({data['hardware_knee_flops_per_byte']:.1f} FLOP/B)",
    )
    ax2.set_xscale("log")
    ax2.set_xlabel("Arithmetic Intensity (FLOPs / Byte)")
    ax2.set_ylabel("DRAM Bandwidth Utilization (% Peak)")
    ax2.set_title("(b) Bandwidth Efficiency across Roofline Spectrum")
    ax2.set_ylim(0, 100)
    ax2.grid(True)
    ax2.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()

    # Save SVG (vector format for LaTeX manuscript) and PNG
    svg_path = f"{output_prefix}.svg"
    png_path = f"{output_prefix}.png"
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.savefig(png_path, format="png", bbox_inches="tight")
    plt.close()

    print(f"\n[+] Publication vector figure generated at {svg_path}")
    print(f"[+] High-resolution PNG generated at {png_path}")


def main():
    print("=" * 80)
    print("AxiomLM Phase 1: Microarchitectural Memory Wall Proof")
    print("Task 1.3: Empirical Bandwidth Saturation Curve & Figure Generator")
    print("=" * 80)

    data = generate_saturation_curve(N=4096, K=4096)

    # Export report JSON
    json_path = Path("benchmarks/phase1_task13_bandwidth_saturation_report.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n[+] Master Task 1.3 JSON report written to {json_path}")

    # Generate vector artifacts
    plot_bandwidth_saturation(
        data, output_prefix="benchmarks/bandwidth_saturation_curve"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
