"""
AxiomLM Phase 3 Task 3.1 Empirical Benchmark Campaign:
In-Flight FP8 & INT4 KV-Cache Quantization Microarchitectural Profiling.

Evaluates in-flight KV-cache quantization on Apple Silicon Unified Memory Architecture (M3 Pro).

Measures:
1. Bare-metal Metal kernel latency (mean, std, P50, P95, P99 in microseconds) across FP16, FP8, and INT4.
2. Physical DRAM write bytes and effective write bandwidth (GB/s).
3. Physical KV-cache memory footprint scaling from T=512 to T=32,768 (32K) across SmolLM2-360M, Qwen2.5-0.5B, and Llama-3.2-1B.
4. Numerical fidelity metrics: Signal-to-Quantization-Noise Ratio (SQNR in dB), Cosine Similarity, MSE, and Max Error.
5. Export golden publication artifact `benchmarks/phase3_task31_kv_quant_report.json`.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from typing import Any

# Add workspace root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from axiomlm.engine.paged_cache import StaticKVCache
from axiomlm.kernels import _METAL_MOD, is_mps_available
from axiomlm.kernels.quant import (
    compute_quantization_metrics,
    dequantize_kv_fp8,
    dequantize_kv_int4,
    quantize_kv_fp8,
    quantize_kv_int4,
    rope_kv_cache_write_quantized,
)


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
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        telemetry["chip"] = "Apple Silicon"
    return telemetry


def benchmark_kernel_latencies(
    device: torch.device,
    n_head: int = 32,
    n_kv_head: int = 8,
    head_dim: int = 64,
    max_seq_len: int = 4096,
    num_iterations: int = 100,
    warmup: int = 20,
) -> dict[str, Any]:
    """
    Profiles execution latency and write bandwidth of rope_kv_cache_write
    across FP16, FP8, and INT4 on Apple GPU with hardware synchronization barriers.
    """
    q_dim = n_head * head_dim
    kv_dim = n_kv_head * head_dim
    total_dim = q_dim + 2 * kv_dim

    qkv = torch.randn(1, 1, total_dim, dtype=torch.float16, device=device)
    freqs_cis = torch.randn(
        max_seq_len, head_dim // 2, 2, dtype=torch.float32, device=device
    )

    # 1. FP16 Buffers
    k_fp16 = torch.zeros(
        1, n_kv_head, max_seq_len, head_dim, dtype=torch.float16, device=device
    )
    v_fp16 = torch.zeros(
        1, n_kv_head, max_seq_len, head_dim, dtype=torch.float16, device=device
    )

    # 2. FP8 Buffers
    k_fp8 = torch.zeros(
        1, n_kv_head, max_seq_len, head_dim, dtype=torch.uint8, device=device
    )
    v_fp8 = torch.zeros(
        1, n_kv_head, max_seq_len, head_dim, dtype=torch.uint8, device=device
    )
    k_scales_fp8 = torch.zeros(
        1,
        n_kv_head,
        max_seq_len,
        max(1, head_dim // 64),
        dtype=torch.float16,
        device=device,
    )
    v_scales_fp8 = torch.zeros(
        1,
        n_kv_head,
        max_seq_len,
        max(1, head_dim // 64),
        dtype=torch.float16,
        device=device,
    )

    # 3. INT4 Buffers
    k_int4 = torch.zeros(
        1, n_kv_head, max_seq_len, head_dim // 2, dtype=torch.uint8, device=device
    )
    v_int4 = torch.zeros(
        1, n_kv_head, max_seq_len, head_dim // 2, dtype=torch.uint8, device=device
    )
    k_scales_int4 = torch.zeros(
        1,
        n_kv_head,
        max_seq_len,
        max(1, head_dim // 64),
        dtype=torch.float16,
        device=device,
    )
    v_scales_int4 = torch.zeros(
        1,
        n_kv_head,
        max_seq_len,
        max(1, head_dim // 64),
        dtype=torch.float16,
        device=device,
    )

    results: dict[str, Any] = {}

    configs = [
        (
            "FP16 (Baseline)",
            "fp16",
            k_fp16,
            v_fp16,
            None,
            None,
            2 * n_kv_head * head_dim * 2,
        ),
        (
            "FP8 (E4M3 In-Flight)",
            "fp8",
            k_fp8,
            v_fp8,
            k_scales_fp8,
            v_scales_fp8,
            2 * n_kv_head * head_dim * 1 + 2 * n_kv_head * (head_dim // 64) * 2,
        ),
        (
            "INT4 (Packed In-Flight)",
            "int4",
            k_int4,
            v_int4,
            k_scales_int4,
            v_scales_int4,
            2 * n_kv_head * (head_dim // 2) * 1 + 2 * n_kv_head * (head_dim // 64) * 2,
        ),
    ]

    for label, mode, k_buf, v_buf, ks, vs, write_bytes in configs:
        # Warmup
        for w in range(warmup):
            write_pos = w % max_seq_len
            rope_kv_cache_write_quantized(
                qkv,
                freqs_cis,
                k_buf,
                v_buf,
                ks,
                vs,
                q_dim,
                kv_dim,
                head_dim,
                0,
                write_pos,
                kv_dtype=mode,
            )
        if device.type == "mps":
            torch.mps.synchronize()

        # Timed iterations
        latencies_us = []
        for it in range(num_iterations):
            write_pos = (warmup + it) % max_seq_len
            t0 = time.perf_counter()
            rope_kv_cache_write_quantized(
                qkv,
                freqs_cis,
                k_buf,
                v_buf,
                ks,
                vs,
                q_dim,
                kv_dim,
                head_dim,
                0,
                write_pos,
                kv_dtype=mode,
            )
            if device.type == "mps":
                torch.mps.synchronize()
            t1 = time.perf_counter()
            latencies_us.append((t1 - t0) * 1e6)

        arr = np.array(latencies_us)
        mean_us = float(np.mean(arr))
        std_us = float(np.std(arr))
        p50_us = float(np.percentile(arr, 50))
        p95_us = float(np.percentile(arr, 95))
        p99_us = float(np.percentile(arr, 99))
        write_bw_gb_s = (write_bytes / (mean_us * 1e-6)) / 1e9

        results[mode] = {
            "label": label,
            "mean_latency_us": round(mean_us, 2),
            "std_latency_us": round(std_us, 2),
            "p50_us": round(p50_us, 2),
            "p95_us": round(p95_us, 2),
            "p99_us": round(p99_us, 2),
            "write_bytes_per_token": write_bytes,
            "write_bandwidth_gb_s": round(write_bw_gb_s, 2),
        }

    return results


def benchmark_memory_scaling() -> dict[str, Any]:
    """
    Computes exact physical memory footprint and savings scaling across
    context lengths T in [512, 1024, 2048, 4096, 8192, 16384, 32768]
    for SmolLM2-360M, Qwen2.5-0.5B, and Llama-3.2-1B.
    """
    models = {
        "SmolLM2-360M": {"n_layer": 30, "n_kv_head": 3, "head_dim": 64},
        "Qwen2.5-0.5B": {"n_layer": 24, "n_kv_head": 2, "head_dim": 64},
        "Llama-3.2-1B": {"n_layer": 16, "n_kv_head": 8, "head_dim": 64},
    }
    context_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]

    scaling_results: dict[str, Any] = {}

    for model_name, params in models.items():
        n_layer = params["n_layer"]
        n_kv = params["n_kv_head"]
        hd = params["head_dim"]

        model_data = []
        for T in context_lengths:
            c_fp16 = StaticKVCache(
                1, T, n_layer, n_kv, hd, cache_dtype="fp16", device="cpu"
            )
            c_fp8 = StaticKVCache(
                1, T, n_layer, n_kv, hd, cache_dtype="fp8", device="cpu"
            )
            c_int4 = StaticKVCache(
                1, T, n_layer, n_kv, hd, cache_dtype="int4", device="cpu"
            )

            b_fp16 = c_fp16.get_memory_bytes()
            b_fp8 = c_fp8.get_memory_bytes()
            b_int4 = c_int4.get_memory_bytes()

            mb_fp16 = b_fp16 / (1024 * 1024)
            mb_fp8 = b_fp8 / (1024 * 1024)
            mb_int4 = b_int4 / (1024 * 1024)

            saved_fp8_mb = mb_fp16 - mb_fp8
            saved_int4_mb = mb_fp16 - mb_int4

            ratio_fp8 = b_fp16 / b_fp8
            ratio_int4 = b_fp16 / b_int4

            model_data.append(
                {
                    "context_length": T,
                    "fp16_mb": round(mb_fp16, 2),
                    "fp8_mb": round(mb_fp8, 2),
                    "int4_mb": round(mb_int4, 2),
                    "saved_fp8_mb": round(saved_fp8_mb, 2),
                    "saved_int4_mb": round(saved_int4_mb, 2),
                    "compression_fp8_x": round(ratio_fp8, 2),
                    "compression_int4_x": round(ratio_int4, 2),
                }
            )

        scaling_results[model_name] = model_data

    return scaling_results


def benchmark_numerical_fidelity(
    device: torch.device, num_samples: int = 1000
) -> dict[str, Any]:
    """
    Evaluates SQNR (dB), Cosine Similarity, MSE, and Max Error distributions
    across 1,000 activation vectors.
    """
    torch.manual_seed(42)
    head_dim = 64
    x = torch.randn(num_samples, head_dim, dtype=torch.float16, device=device)

    # FP8 evaluation
    k_fp8, ks_fp8, _, _ = quantize_kv_fp8(x, x)
    k_dq_fp8 = dequantize_kv_fp8(k_fp8, ks_fp8, head_dim)
    m_fp8 = compute_quantization_metrics(x, k_dq_fp8)

    # INT4 evaluation
    k_i4, ks_i4, _, _ = quantize_kv_int4(x, x)
    k_dq_i4 = dequantize_kv_int4(k_i4, ks_i4, head_dim)
    m_i4 = compute_quantization_metrics(x, k_dq_i4)

    return {
        "fp8_metrics": {
            "sqnr_db": round(m_fp8["sqnr_db"], 2),
            "cosine_similarity": round(m_fp8["cosine_similarity"], 5),
            "mse": round(m_fp8["mse"], 6),
            "max_abs_error": round(m_fp8["max_abs_error"], 4),
        },
        "int4_metrics": {
            "sqnr_db": round(m_i4["sqnr_db"], 2),
            "cosine_similarity": round(m_i4["cosine_similarity"], 5),
            "mse": round(m_i4["mse"], 6),
            "max_abs_error": round(m_i4["max_abs_error"], 4),
        },
    }


def main():
    print("=" * 80)
    print(" AxiomLM: Phase 3 Task 3.1 Empirical Benchmark Campaign")
    print(" In-Flight FP8 & INT4 KV-Cache Quantization Microarchitectural Profiling")
    print("=" * 80)

    telemetry = get_hardware_telemetry()
    print(f" • Platform: {telemetry['chip']} ({telemetry['platform']})")
    print(
        f" • PyTorch:  {telemetry['torch_version']} | Metal MPS: {telemetry['mps_available']}"
    )
    print("-" * 80)

    device = torch.device(
        "mps" if is_mps_available() and _METAL_MOD is not None else "cpu"
    )

    # 1. Microarchitectural Latency & Bandwidth
    print("\n[1/3] Benchmarking In-Flight Kernel Latency and DRAM Write Traffic...")
    latencies = benchmark_kernel_latencies(device)
    print(
        f"  {'Format':<25} {'Mean Latency':<16} {'Std':<10} {'P50':<10} {'P99':<10} {'Write Traffic':<14} {'Write BW'}"
    )
    print("  " + "-" * 95)
    for v in latencies.values():
        print(
            f"  {v['label']:<25} {v['mean_latency_us']:>6.2f} us       {v['std_latency_us']:>6.2f} us   {v['p50_us']:>6.2f} us   {v['p99_us']:>6.2f} us   {v['write_bytes_per_token']:>6} B/tok      {v['write_bandwidth_gb_s']:>6.2f} GB/s"
        )

    # 2. Numerical Fidelity
    print("\n[2/3] Profiling Numerical Fidelity Across 1,000 Activation Vectors...")
    fidelity = benchmark_numerical_fidelity(device)
    print(
        f"  • FP8  (E4M3):  SQNR = {fidelity['fp8_metrics']['sqnr_db']:>5.2f} dB | Cosine Sim = {fidelity['fp8_metrics']['cosine_similarity']:.5f} | MSE = {fidelity['fp8_metrics']['mse']:.6f} | Max Err = {fidelity['fp8_metrics']['max_abs_error']:.4f}"
    )
    print(
        f"  • INT4 (Symmetric): SQNR = {fidelity['int4_metrics']['sqnr_db']:>5.2f} dB | Cosine Sim = {fidelity['int4_metrics']['cosine_similarity']:.5f} | MSE = {fidelity['int4_metrics']['mse']:.6f} | Max Err = {fidelity['int4_metrics']['max_abs_error']:.4f}"
    )

    # 3. Memory Footprint Scaling
    print("\n[3/3] Profiling KV-Cache Physical Memory Scaling up to 32K Context...")
    scaling = benchmark_memory_scaling()

    print("\n  -- Llama-3.2-1B (16 Layers, 8 KV Heads, Head Dim 64) --")
    print(
        f"  {'Context Length':<16} {'FP16 Footprint':<16} {'FP8 Footprint':<16} {'INT4 Footprint':<16} {'RAM Saved (INT4)':<18} {'Compression'}"
    )
    print("  " + "-" * 95)
    for row in scaling["Llama-3.2-1B"]:
        print(
            f"  {row['context_length']:<16} {row['fp16_mb']:>8.2f} MB      {row['fp8_mb']:>8.2f} MB      {row['int4_mb']:>8.2f} MB      {row['saved_int4_mb']:>10.2f} MB        {row['compression_int4_x']:>4.2f}x"
        )

    # Save artifact
    output_path = os.path.join("benchmarks", "phase3_task31_kv_quant_report.json")
    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware_telemetry": telemetry,
        "kernel_latencies": latencies,
        "numerical_fidelity": fidelity,
        "memory_scaling": scaling,
    }

    with open(output_path, "w") as f:
        json.dump(report_data, f, indent=2)

    print("\n" + "=" * 80)
    print(f" Successfully exported golden benchmark artifact to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
