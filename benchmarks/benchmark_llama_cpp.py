"""
Standardized llama.cpp (llama-bench) Benchmark Harness.
Runs llama-bench with exact prompt and generation length controls against
the architecture-parity AxiomLM 124M GGUF models on Apple Metal GPU.
"""

import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import shutil
import subprocess
from typing import Any

from benchmarks.export_gguf import prepare_benchmark_models


def run_llama_bench(
    model_path: str,
    prompt_tokens: int = 512,
    gen_tokens: int = 128,
    repetitions: int = 3,
    n_gpu_layers: int = 99,
) -> dict[str, Any]:
    """Executes llama-bench with JSON output and parses exact throughput and latencies."""
    llama_bench_bin = shutil.which("llama-bench")
    if not llama_bench_bin:
        raise RuntimeError(
            "llama-bench executable not found. Install via brew install llama.cpp"
        )

    cmd = [
        llama_bench_bin,
        "-m",
        model_path,
        "-p",
        str(prompt_tokens),
        "-n",
        str(gen_tokens),
        "-r",
        str(repetitions),
        "-ngl",
        str(n_gpu_layers),
        "-o",
        "json",
    ]

    res = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
    )

    # llama-bench outputs metadata lines followed by JSON array
    stdout_text = res.stdout.strip()
    json_start = stdout_text.find("[")
    if json_start == -1:
        raise RuntimeError(
            f"Could not find JSON array in llama-bench output: {stdout_text[:500]}"
        )

    json_str = stdout_text[json_start:]
    bench_data = json.loads(json_str)

    results: dict[str, Any] = {
        "model_path": model_path,
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "raw_entries": bench_data,
        "prompt_tok_s": None,
        "prompt_stddev": None,
        "decode_tok_s": None,
        "decode_stddev": None,
        "decode_latency_ms": None,
    }

    for entry in bench_data:
        test_type = entry.get("test", "")
        # In llama-bench, pp<N> is prompt processing (prefill), tg<N> is token generation (decode)
        if (
            f"pp{prompt_tokens}" in test_type
            or entry.get("n_prompt") == prompt_tokens
            and entry.get("n_gen") == 0
        ):
            results["prompt_tok_s"] = entry.get("avg_ts")
            results["prompt_stddev"] = entry.get("stddev_ts")
        elif f"tg{gen_tokens}" in test_type or entry.get("n_gen") == gen_tokens:
            results["decode_tok_s"] = entry.get("avg_ts")
            results["decode_stddev"] = entry.get("stddev_ts")
            if results["decode_tok_s"] and results["decode_tok_s"] > 0:
                results["decode_latency_ms"] = round(
                    1000.0 / results["decode_tok_s"], 3
                )

    return results


def run_llama_cpp_benchmarks(
    prompt_tokens: int = 512,
    gen_tokens: int = 128,
    repetitions: int = 3,
) -> dict[str, Any]:
    """Runs standardized llama-bench suite across both FP16 and Q4_0 precision targets."""
    print("=" * 70)
    print(
        f"[llama.cpp Benchmark] Running llama-bench on Metal (P={prompt_tokens}, N={gen_tokens}, R={repetitions})"
    )
    print("=" * 70)

    models = prepare_benchmark_models()
    f16_model = models["f16_path"]
    q4_model = models["q4_path"]

    print("\n>>> Running llama.cpp FP16 Baseline...")
    f16_results = run_llama_bench(f16_model, prompt_tokens, gen_tokens, repetitions)
    print(
        f"  • FP16 Prefill Speed: {f16_results['prompt_tok_s']:.2f} tok/s (± {f16_results['prompt_stddev']:.2f})"
    )
    print(
        f"  • FP16 Decode Speed:  {f16_results['decode_tok_s']:.2f} tok/s (± {f16_results['decode_stddev']:.2f}) [Latency: {f16_results['decode_latency_ms']} ms/tok]"
    )

    print("\n>>> Running llama.cpp Q4_0 Baseline...")
    q4_results = run_llama_bench(q4_model, prompt_tokens, gen_tokens, repetitions)
    print(
        f"  • Q4_0 Prefill Speed: {q4_results['prompt_tok_s']:.2f} tok/s (± {q4_results['prompt_stddev']:.2f})"
    )
    print(
        f"  • Q4_0 Decode Speed:  {q4_results['decode_tok_s']:.2f} tok/s (± {q4_results['decode_stddev']:.2f}) [Latency: {q4_results['decode_latency_ms']} ms/tok]"
    )

    return {
        "engine": "llama.cpp",
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "f16": f16_results,
        "q4_0": q4_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run llama.cpp benchmarks with standardized prompt/gen lengths"
    )
    parser.add_argument(
        "--prompt", type=int, default=512, help="Prompt token count (prefill)"
    )
    parser.add_argument(
        "--gen", type=int, default=128, help="Generation token count (decode)"
    )
    parser.add_argument(
        "--repetitions", type=int, default=3, help="Benchmark repetitions"
    )
    args = parser.parse_args()

    suite_results = run_llama_cpp_benchmarks(args.prompt, args.gen, args.repetitions)
    print("\nBenchmark Suite Completed Successfully.")
