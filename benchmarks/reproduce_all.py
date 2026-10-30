"""
AxiomLM Master Reproducibility Benchmark Suite.
One-click unified harness that benchmarks AxiomLM, Apple MLX, and llama.cpp (llama-bench)
on identical hardware under strict parameter parity (same architecture, context, prompt, and decode lengths).
Outputs structured JSON telemetry and publication-ready Markdown tables.
"""

import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
from datetime import datetime
from typing import Dict, Any

from benchmarks.hardware_telemetry import collect_system_telemetry
from benchmarks.benchmark_llama_cpp import run_llama_cpp_benchmarks
from benchmarks.benchmark_mlx import run_mlx_benchmarks
from benchmarks.benchmark_axiomlm import run_axiomlm_benchmarks


def generate_markdown_comparison_table(results: Dict[str, Any]) -> str:
    """Generates a clean Markdown comparison table from the combined benchmark run."""
    lines = []
    lines.append(
        "| Framework / Engine | Precision | Decode Speed (tok/s) | Latency (ms/tok) | Speedup vs. MLX | Speedup vs. llama.cpp |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    # Baselines
    mlx_f16 = results["mlx"]["f16"]["decode_tok_s"]
    llama_f16 = results["llama_cpp"]["f16"]["decode_tok_s"]
    llama_q4 = results["llama_cpp"]["q4_0"]["decode_tok_s"]

    # 1. PyTorch MPS Cached
    pt = results["axiomlm"]["pytorch_mps_cached"]
    lines.append(
        f"| **PyTorch MPS Eager (Cached)** | FP16 | `{pt['decode_tok_s']:.1f} ± {pt['decode_stddev']:.1f}` | `{pt['decode_latency_ms']:.2f} ms` | `{(pt['decode_tok_s'] / mlx_f16):.2f}x` | `{(pt['decode_tok_s'] / llama_f16):.2f}x` |"
    )

    # 2. Apple MLX
    mlx = results["mlx"]["f16"]
    lines.append(
        f"| **Apple MLX Baseline** | FP16 | `{mlx['decode_tok_s']:.1f} ± {mlx['decode_stddev']:.1f}` | `{mlx['decode_latency_ms']:.2f} ms` | `1.00x` | `{(mlx['decode_tok_s'] / llama_f16):.2f}x` |"
    )

    # 3. llama.cpp Metal FP16
    ll_f16 = results["llama_cpp"]["f16"]
    lines.append(
        f"| **llama.cpp Metal Baseline** | FP16 | `{ll_f16['decode_tok_s']:.1f} ± {ll_f16['decode_stddev']:.1f}` | `{ll_f16['decode_latency_ms']:.2f} ms` | `{(ll_f16['decode_tok_s'] / mlx_f16):.2f}x` | `1.00x` |"
    )

    # 4. AxiomLM FP16
    ax_f16 = results["axiomlm"]["axiomlm_fp16"]
    lines.append(
        f"| **AxiomLM Modern Decode Runner** | **FP16** | **`{ax_f16['decode_tok_s']:.1f} ± {ax_f16['decode_stddev']:.1f}`** | **`{ax_f16['decode_latency_ms']:.2f} ms`** | **`{(ax_f16['decode_tok_s'] / mlx_f16):.2f}x`** | **`{(ax_f16['decode_tok_s'] / llama_f16):.2f}x`** |"
    )

    # 5. llama.cpp Metal Q4_0
    ll_q4 = results["llama_cpp"]["q4_0"]
    lines.append(
        f"| **llama.cpp Metal Baseline** | Q4_0 | `{ll_q4['decode_tok_s']:.1f} ± {ll_q4['decode_stddev']:.1f}` | `{ll_q4['decode_latency_ms']:.2f} ms` | — | `1.00x` |"
    )

    # 6. AxiomLM Q8_G64
    ax_q8 = results["axiomlm"]["axiomlm_q8_g64"]
    lines.append(
        f"| **AxiomLM Quantized Metal GEMV** | **Q8_G64** | **`{ax_q8['decode_tok_s']:.1f} ± {ax_q8['decode_stddev']:.1f}`** | **`{ax_q8['decode_latency_ms']:.2f} ms`** | — | **`{(ax_q8['decode_tok_s'] / llama_q4):.2f}x`** |"
    )

    # 7. AxiomLM Q4_G64
    ax_q4 = results["axiomlm"]["axiomlm_q4_g64"]
    lines.append(
        f"| **AxiomLM Quantized Metal GEMV** | **Q4_G64** | **`{ax_q4['decode_tok_s']:.1f} ± {ax_q4['decode_stddev']:.1f}`** | **`{ax_q4['decode_latency_ms']:.2f} ms`** | — | **`{(ax_q4['decode_tok_s'] / llama_q4):.2f}x`** |"
    )

    # 8. AxiomLM Speculative Q4
    ax_spec = results["axiomlm"]["axiomlm_speculative_q4"]
    lines.append(
        f"| **AxiomLM Speculative Batched QMM** | **Q4_G64 (K={ax_spec['k']})** | **`{ax_spec['decode_tok_s']:.1f} ± {ax_spec['decode_stddev']:.1f}`** | **`{ax_spec['decode_latency_ms']:.2f} ms`** | — | **`{(ax_spec['decode_tok_s'] / llama_q4):.2f}x`** |"
    )

    return "\n".join(lines)


def run_full_reproducibility_suite(
    prompt_tokens: int = 512,
    gen_tokens: int = 128,
    repetitions: int = 3,
    output_dir: str = "benchmarks/results",
) -> Dict[str, Any]:
    """Runs the complete reproducibility suite across all three frameworks."""
    start_time = datetime.now()
    timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")

    print("\n" + "#" * 80)
    print("AXIOMLM EMPIRICAL BENCHMARK REPRODUCIBILITY SUITE")
    print(
        f"Timestamp: {start_time.isoformat()} | Prompt: {prompt_tokens} | Gen: {gen_tokens} | Reps: {repetitions}"
    )
    print("#" * 80)

    # 1. Telemetry
    print("\n[Stage 1/4] Capturing Hardware & Software Telemetry...")
    telemetry = collect_system_telemetry()
    hw = telemetry["hardware"]
    print(
        f"  • SoC Target: {hw['soc_name']} ({hw['cpu_cores']} CPU Cores, {hw['gpu_cores']} GPU Cores)"
    )
    print(
        f"  • Memory: {hw['ram_gb']} GB Unified Memory ({hw['peak_bandwidth_gb_s']} GB/s peak bandwidth)"
    )
    print(f"  • Operating System: macOS {hw['os_version']} (Darwin {hw['os_release']})")
    print(
        f"  • Git Commit: {telemetry['git']['commit_hash'][:10]} (Branch: {telemetry['git']['branch']})"
    )

    # 2. llama.cpp (llama-bench)
    print("\n[Stage 2/4] Executing llama.cpp (llama-bench) Metal Baselines...")
    llama_results = run_llama_cpp_benchmarks(prompt_tokens, gen_tokens, repetitions)

    # 3. Apple MLX
    print("\n[Stage 3/4] Executing Apple MLX Native GPU Baselines...")
    mlx_results = run_mlx_benchmarks(prompt_tokens, gen_tokens, repetitions)

    # 4. AxiomLM
    print("\n[Stage 4/4] Executing AxiomLM Custom Metal Engine Suite...")
    axiomlm_results = run_axiomlm_benchmarks(prompt_tokens, gen_tokens, repetitions)

    # Combine
    combined = {
        "metadata": {
            "timestamp": start_time.isoformat(),
            "prompt_tokens": prompt_tokens,
            "gen_tokens": gen_tokens,
            "repetitions": repetitions,
        },
        "telemetry": telemetry,
        "llama_cpp": llama_results,
        "mlx": mlx_results,
        "axiomlm": axiomlm_results,
    }

    # Generate Markdown Table
    md_table = generate_markdown_comparison_table(combined)
    combined["markdown_table"] = md_table

    # Save Results
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, f"benchmark_{timestamp_str}.json")
    latest_file = os.path.join(output_dir, "latest_benchmark_run.json")

    with open(report_file, "w") as f:
        json.dump(combined, f, indent=2)
    with open(latest_file, "w") as f:
        json.dump(combined, f, indent=2)

    print("\n" + "=" * 80)
    print("CONTROLLED BENCHMARK COMPARISON MATRIX (1:1 PARAMETER PARITY)")
    print("=" * 80)
    print(md_table)
    print("=" * 80)
    print("\nArtifacts successfully written to:")
    print(f"  • Timestamped Run Log: {report_file}")
    print(f"  • Latest Snapshot:     {latest_file}\n")

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-click master benchmark reproduction suite"
    )
    parser.add_argument(
        "--prompt", type=int, default=512, help="Prompt token count (prefill)"
    )
    parser.add_argument(
        "--gen", type=int, default=128, help="Generation token count (decode)"
    )
    parser.add_argument(
        "--repetitions", type=int, default=2, help="Repetitions for each test"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="benchmarks/results",
        help="Directory for JSON logs",
    )
    args = parser.parse_args()

    run_full_reproducibility_suite(args.prompt, args.gen, args.repetitions, args.outdir)
