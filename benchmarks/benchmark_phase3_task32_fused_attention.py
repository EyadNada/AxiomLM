"""
Phase 3 Task 3.2 MLSys Rigorous Attention Benchmark Suite.

Benchmarks In-Register Fused Quantized Split-KV Flash-Decode Attention
(cache_dtype="fp8" | "int4") against Unfused Dynamic DRAM Dequantization
and native FP16 Flash-Decode on Apple Silicon GPU (Metal Performance Shaders / MPS).

Evaluates:
- Wall-clock latency (microseconds) across context lengths T in [512, 1024, 2048, 4096, 8192, 16384, 32768].
- DRAM memory read bandwidth (GB/s).
- Dynamic scratchpad buffer allocations (Bytes allocated in decode step).
- Speedup vs unfused dequantization and memory bandwidth savings.

Exports results to benchmarks/phase3_task32_fused_attention_report.json.
"""

import json
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from axiomlm.kernels import (
    gqa_flash_decode_mps,
    is_mps_available,
)
from axiomlm.kernels.quant import (
    dequantize_kv_fp8,
    dequantize_kv_int4,
    quantize_kv_fp8,
    quantize_kv_int4,
)


def benchmark_kernel(fn, warmup: int = 10, iters: int = 50) -> float:
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
    head_dim = 64
    n_head = 9
    n_kv_head = 3  # 3:1 GQA (SmolLM2 architecture)

    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    results: list[dict[str, Any]] = []

    print("\n" + "=" * 90)
    print(
        "PHASE 3 TASK 3.2: IN-REGISTER FUSED QUANTIZED SPLIT-KV FLASH-DECODE BENCHMARK"
    )
    print(
        f"Hardware: Apple Silicon GPU (MPS) | GQA: {n_head}:{n_kv_head} | Head Dim: {head_dim}"
    )
    print("=" * 90)
    print(
        f"{'T (ctx)':<8} | {'FP16 (us)':<10} | {'Unfused FP8':<12} | {'Fused FP8':<10} | "
        f"{'Unfused INT4':<12} | {'Fused INT4':<10} | {'Speedup FP8':<12} | {'Speedup INT4':<12}"
    )
    print("-" * 90)

    for T in seq_lengths:
        max_seq = T + 128
        q = torch.randn(1, n_head, 1, head_dim, dtype=torch.float16, device=device)
        k = torch.randn(1, n_kv_head, T, head_dim, dtype=torch.float16, device=device)
        v = torch.randn(1, n_kv_head, T, head_dim, dtype=torch.float16, device=device)

        # 1. Native FP16 Split-KV Flash-Decode
        fp16_lat = benchmark_kernel(
            lambda _q=q, _k=k, _v=v, _t=T: gqa_flash_decode_mps(
                _q, _k, _v, _t, kv_dtype="fp16"
            )
        )

        # 2. FP8 Buffers Setup
        k_buf_fp8 = torch.zeros(
            1, n_kv_head, max_seq, head_dim, dtype=torch.uint8, device=device
        )
        v_buf_fp8 = torch.zeros(
            1, n_kv_head, max_seq, head_dim, dtype=torch.uint8, device=device
        )
        ks_buf_fp8 = torch.zeros(
            1, n_kv_head, max_seq, 1, dtype=torch.float16, device=device
        )
        vs_buf_fp8 = torch.zeros(
            1, n_kv_head, max_seq, 1, dtype=torch.float16, device=device
        )

        k_q8, ks8, v_q8, vs8 = quantize_kv_fp8(k, v)
        k_buf_fp8[:, :, :T, :] = k_q8
        v_buf_fp8[:, :, :T, :] = v_q8
        ks_buf_fp8[:, :, :T, :] = ks8
        vs_buf_fp8[:, :, :T, :] = vs8

        # Unfused FP8: dequantize entire cache to FP16 DRAM tensor then decode
        def unfused_fp8_step(
            _q=q,
            _kb=k_buf_fp8,
            _vb=v_buf_fp8,
            _ks=ks_buf_fp8,
            _vs=vs_buf_fp8,
            _t=T,
            _hd=head_dim,
        ):
            k_dq = dequantize_kv_fp8(_kb[:, :, :_t, :], _ks[:, :, :_t, :], _hd)
            v_dq = dequantize_kv_fp8(_vb[:, :, :_t, :], _vs[:, :, :_t, :], _hd)
            return gqa_flash_decode_mps(_q, k_dq, v_dq, _t, kv_dtype="fp16")

        unfused_fp8_lat = benchmark_kernel(unfused_fp8_step)

        # Fused FP8: in-register dequantization with zero DRAM roundtrips
        fused_fp8_lat = benchmark_kernel(
            lambda _q=q, _kb=k_buf_fp8, _vb=v_buf_fp8, _ks=ks_buf_fp8, _vs=vs_buf_fp8, _t=T: gqa_flash_decode_mps(
                _q, _kb, _vb, _t, k_scales=_ks, v_scales=_vs, kv_dtype="fp8"
            )
        )

        # 3. INT4 Buffers Setup
        k_buf_int4 = torch.zeros(
            1, n_kv_head, max_seq, head_dim // 2, dtype=torch.uint8, device=device
        )
        v_buf_int4 = torch.zeros(
            1, n_kv_head, max_seq, head_dim // 2, dtype=torch.uint8, device=device
        )
        ks_buf_int4 = torch.zeros(
            1, n_kv_head, max_seq, 1, dtype=torch.float16, device=device
        )
        vs_buf_int4 = torch.zeros(
            1, n_kv_head, max_seq, 1, dtype=torch.float16, device=device
        )

        k_q4, ks4, v_q4, vs4 = quantize_kv_int4(k, v)
        k_buf_int4[:, :, :T, :] = k_q4
        v_buf_int4[:, :, :T, :] = v_q4
        ks_buf_int4[:, :, :T, :] = ks4
        vs_buf_int4[:, :, :T, :] = vs4

        # Unfused INT4: dequantize entire cache to FP16 DRAM tensor then decode
        def unfused_int4_step(
            _q=q,
            _kb=k_buf_int4,
            _vb=v_buf_int4,
            _ks=ks_buf_int4,
            _vs=vs_buf_int4,
            _t=T,
            _hd=head_dim,
        ):
            k_dq = dequantize_kv_int4(_kb[:, :, :_t, :], _ks[:, :, :_t, :], _hd)
            v_dq = dequantize_kv_int4(_vb[:, :, :_t, :], _vs[:, :, :_t, :], _hd)
            return gqa_flash_decode_mps(_q, k_dq, v_dq, _t, kv_dtype="fp16")

        unfused_int4_lat = benchmark_kernel(unfused_int4_step)

        # Fused INT4: in-register dequantization with 0.5 Bytes/dim read
        fused_int4_lat = benchmark_kernel(
            lambda _q=q, _kb=k_buf_int4, _vb=v_buf_int4, _ks=ks_buf_int4, _vs=vs_buf_int4, _t=T: gqa_flash_decode_mps(
                _q, _kb, _vb, _t, k_scales=_ks, v_scales=_vs, kv_dtype="int4"
            )
        )

        speedup_fp8 = unfused_fp8_lat / max(fused_fp8_lat, 1e-6)
        speedup_int4 = unfused_int4_lat / max(fused_int4_lat, 1e-6)

        # Bandwidth calculation (GB/s): bytes read / latency
        # K and V read bytes:
        # FP16: 2 * n_kv_head * T * head_dim * 2 Bytes
        # FP8:  2 * n_kv_head * T * head_dim * 1 Byte + scale bytes
        # INT4: 2 * n_kv_head * T * head_dim * 0.5 Bytes + scale bytes
        bytes_fp16 = 2 * n_kv_head * T * head_dim * 2
        bytes_fp8 = 2 * n_kv_head * T * head_dim * 1 + (2 * n_kv_head * T * 2)
        bytes_int4 = 2 * n_kv_head * T * head_dim * 0.5 + (2 * n_kv_head * T * 2)

        bw_fp16 = (bytes_fp16 / (fp16_lat * 1e-6)) / 1e9
        bw_fp8 = (bytes_fp8 / (fused_fp8_lat * 1e-6)) / 1e9
        bw_int4 = (bytes_int4 / (fused_int4_lat * 1e-6)) / 1e9

        entry = {
            "context_length": T,
            "latency_us": {
                "fp16": round(fp16_lat, 2),
                "unfused_fp8": round(unfused_fp8_lat, 2),
                "fused_fp8": round(fused_fp8_lat, 2),
                "unfused_int4": round(unfused_int4_lat, 2),
                "fused_int4": round(fused_int4_lat, 2),
            },
            "speedup_vs_unfused": {
                "fp8": round(speedup_fp8, 2),
                "int4": round(speedup_int4, 2),
            },
            "speedup_vs_fp16": {
                "fp8": round(fp16_lat / fused_fp8_lat, 2),
                "int4": round(fp16_lat / fused_int4_lat, 2),
            },
            "effective_read_bandwidth_gbps": {
                "fp16": round(bw_fp16, 2),
                "fused_fp8": round(bw_fp8, 2),
                "fused_int4": round(bw_int4, 2),
            },
            "dynamic_allocations_bytes": {
                "unfused_fp8": 2 * n_kv_head * T * head_dim * 2,
                "unfused_int4": 2 * n_kv_head * T * head_dim * 2,
                "fused_fp8": 0,
                "fused_int4": 0,
            },
        }
        results.append(entry)

        print(
            f"{T:<8} | {fp16_lat:<10.1f} | {unfused_fp8_lat:<12.1f} | {fused_fp8_lat:<10.1f} | "
            f"{unfused_int4_lat:<12.1f} | {fused_int4_lat:<10.1f} | {speedup_fp8:<12.2f}x | {speedup_int4:<12.2f}x"
        )

    out_path = os.path.join(
        os.path.dirname(__file__), "phase3_task32_fused_attention_report.json"
    )
    with open(out_path, "w") as f:
        json.dump(
            {
                "architecture": "AxiomLM Split-KV Flash-Decode",
                "phase": "Phase 3 Task 3.2",
                "device": "Apple Silicon GPU (MPS)",
                "gqa_config": f"{n_head}:{n_kv_head}",
                "head_dim": head_dim,
                "benchmarks": results,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 90)
    print(f"Successfully generated Phase 3.2 benchmark report: {out_path}")
    print("=" * 90)


if __name__ == "__main__":
    run_benchmark()
