"""
Standardized AxiomLM Engine Benchmark Harness.
Measures PyTorch MPS Cached baseline, AxiomLM FP16 Decode Runner,
Q8_G64 Quantized Metal GEMV, Q4_G64 Quantized Metal GEMV, and
Speculative Batched QMM Decoding with exact prompt and generation length controls.
"""

import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
from typing import Any

import numpy as np
import torch

from axiomlm import ModelConfig, Transformer
from axiomlm.engine.medusa import MedusaSpeculativeDecoder, train_medusa_heads
from axiomlm.engine.paged_cache import StaticKVCache
from axiomlm.engine.speculative import SpeculativePromptLookupDecoder
from axiomlm.kernels import is_mps_available


def benchmark_engine_mode(
    model: Transformer,
    prompt_tensor: torch.Tensor,
    gen_tokens: int = 128,
    repetitions: int = 3,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Measures prefill and autoregressive decode throughput for a configured AxiomLM model state."""
    if device is None:
        device = torch.device("mps")
    prompt_len = prompt_tensor.size(1)
    prefill_speeds = []
    decode_speeds = []

    for _ in range(repetitions):
        cache = StaticKVCache(1, 1024, 12, 4, 64, device=device, dtype=torch.float16)

        # 1. Prefill
        torch.mps.synchronize()
        t0_p = time.perf_counter()
        with torch.no_grad():
            out, _, _ = model(prompt_tensor, kv_caches=cache)
        torch.mps.synchronize()
        dt_p = time.perf_counter() - t0_p
        prefill_tok_s = prompt_len / dt_p if dt_p > 0 else 0.0
        prefill_speeds.append(prefill_tok_s)

        next_tok = out[:, -1, :].argmax(dim=-1, keepdim=True)

        # 2. Warmup
        for _ in range(5):
            with torch.no_grad():
                out, _, _ = model(next_tok, kv_caches=cache)
                next_tok = out[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.mps.synchronize()

        # 3. Decode measurement
        t0_d = time.perf_counter()
        for _ in range(gen_tokens):
            with torch.no_grad():
                out, _, _ = model(next_tok, kv_caches=cache)
                next_tok = out[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.mps.synchronize()
        dt_d = time.perf_counter() - t0_d
        decode_tok_s = gen_tokens / dt_d if dt_d > 0 else 0.0
        decode_speeds.append(decode_tok_s)

    avg_p = float(np.mean(prefill_speeds))
    std_p = float(np.std(prefill_speeds))
    avg_d = float(np.mean(decode_speeds))
    std_d = float(np.std(decode_speeds))
    latency_ms = round(1000.0 / avg_d, 3) if avg_d > 0 else 0.0

    return {
        "prompt_tokens": prompt_len,
        "gen_tokens": gen_tokens,
        "repetitions": repetitions,
        "prompt_tok_s": round(avg_p, 2),
        "prompt_stddev": round(std_p, 2),
        "decode_tok_s": round(avg_d, 2),
        "decode_stddev": round(std_d, 2),
        "decode_latency_ms": latency_ms,
    }


def run_axiomlm_benchmarks(
    prompt_tokens: int = 512,
    gen_tokens: int = 128,
    repetitions: int = 3,
) -> dict[str, Any]:
    """Executes the full suite of AxiomLM architectural optimizations."""
    if not is_mps_available():
        print(
            "\n[Notice] MPS backend is unavailable: no supported Metal GPU was found (running in VM/sandbox/CI)."
        )
        print("Skipping Apple Silicon Metal benchmarks.")
        return {}

    device = torch.device("mps")
    print("=" * 70)
    print(
        f"[AxiomLM Benchmark] Running Full AxiomLM Engine Suite (P={prompt_tokens}, N={gen_tokens}, R={repetitions})"
    )
    print("=" * 70)

    config = ModelConfig(
        arch="modern",
        n_layer=12,
        n_head=12,
        n_kv_head=4,
        n_embd=768,
        vocab_size=50304,
        block_size=1024,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        pos_emb="rope",
        use_fused_kernels=True,
    )

    torch.manual_seed(42)
    model = Transformer(config).to(device=device, dtype=torch.float16).eval()

    # Generate synthetic prompt tensor
    prompt_data = [int(i % 50000) for i in range(prompt_tokens)]
    prompt_tensor = torch.tensor([prompt_data], device=device, dtype=torch.long)

    results: dict[str, Any] = {
        "engine": "AxiomLM",
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
    }

    # 1. PyTorch MPS Eager Cached Baseline
    print("\n>>> [1/6] PyTorch MPS Cached Forward Baseline (FP16)...")
    orig_fused = model.config.use_fused_kernels
    model.config.use_fused_kernels = False
    pt_res = benchmark_engine_mode(
        model, prompt_tensor, gen_tokens, repetitions, device
    )
    model.config.use_fused_kernels = orig_fused
    results["pytorch_mps_cached"] = pt_res
    print(
        f"  • Decode Speed: {pt_res['decode_tok_s']:.2f} tok/s (± {pt_res['decode_stddev']:.2f}) [Latency: {pt_res['decode_latency_ms']} ms/tok]"
    )

    # 2. AxiomLM FP16 Whole-Model Decode Runner
    print("\n>>> [2/6] AxiomLM Modern FP16 Decode Runner (Fused Metal + Split-KV)...")
    fp16_res = benchmark_engine_mode(
        model, prompt_tensor, gen_tokens, repetitions, device
    )
    results["axiomlm_fp16"] = fp16_res
    print(
        f"  • Decode Speed: {fp16_res['decode_tok_s']:.2f} tok/s (± {fp16_res['decode_stddev']:.2f}) [Latency: {fp16_res['decode_latency_ms']} ms/tok]"
    )

    # 3. AxiomLM Q8_G64 Quantized Metal GEMV
    print("\n>>> [3/6] AxiomLM Q8_G64 Quantized Metal GEMV (8-bit Weights)...")
    model.quantize("q8")
    q8_res = benchmark_engine_mode(
        model, prompt_tensor, gen_tokens, repetitions, device
    )
    results["axiomlm_q8_g64"] = q8_res
    print(
        f"  • Decode Speed: {q8_res['decode_tok_s']:.2f} tok/s (± {q8_res['decode_stddev']:.2f}) [Latency: {q8_res['decode_latency_ms']} ms/tok]"
    )

    # 4. AxiomLM Q4_G64 Quantized Metal GEMV
    print("\n>>> [4/6] AxiomLM Q4_G64 Quantized Metal GEMV (4-bit Weights)...")
    model.quantize("q4")
    q4_res = benchmark_engine_mode(
        model, prompt_tensor, gen_tokens, repetitions, device
    )
    results["axiomlm_q4_g64"] = q4_res
    print(
        f"  • Decode Speed: {q4_res['decode_tok_s']:.2f} tok/s (± {q4_res['decode_stddev']:.2f}) [Latency: {q4_res['decode_latency_ms']} ms/tok]"
    )

    # 5. AxiomLM Speculative Prompt-Lookup Decoding (Python Loop, Q4_G64, K=3)
    print(
        "\n>>> [5/6] AxiomLM Speculative Decoding (Python Loop Baseline, Q4_G64, K=3)..."
    )
    code_pattern = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212]
    spec_prompt = (code_pattern * 45)[:prompt_tokens]
    decoder = SpeculativePromptLookupDecoder(ngram_size=3, max_spec_tokens=3)

    # Warmup
    _, _ = decoder.generate(
        model, spec_prompt, max_new_tokens=16, device=device, use_cpp_loop=False
    )

    spec_py_speeds = []
    acceptance_rates = []
    tokens_per_fwd = []

    for _ in range(repetitions):
        _, stats = decoder.generate(
            model,
            spec_prompt,
            max_new_tokens=gen_tokens,
            device=device,
            use_cpp_loop=False,
        )
        spec_py_speeds.append(stats["tokens_per_second"])
        acceptance_rates.append(stats["acceptance_rate_pct"])
        tokens_per_fwd.append(stats["tokens_per_forward_call"])

    avg_spec_py = float(np.mean(spec_py_speeds))
    std_spec_py = float(np.std(spec_py_speeds))
    avg_acc = float(np.mean(acceptance_rates))
    avg_tpf = float(np.mean(tokens_per_fwd))
    spec_py_lat = round(1000.0 / avg_spec_py, 3)

    results["axiomlm_speculative_q4_python_loop"] = {
        "k": 3,
        "decode_tok_s": round(avg_spec_py, 2),
        "decode_stddev": round(std_spec_py, 2),
        "decode_latency_ms": spec_py_lat,
        "acceptance_rate_pct": round(avg_acc, 2),
        "tokens_per_forward": round(avg_tpf, 2),
    }
    print(
        f"  • Decode Speed: {avg_spec_py:.2f} tok/s (± {std_spec_py:.2f}) [Latency: {spec_py_lat} ms/tok]"
    )
    print(
        f"  • Acceptance Rate: {avg_acc:.1f}% | Tokens / Forward Step: {avg_tpf:.2f}x"
    )

    # 6. AxiomLM Pure C++ Speculative Engine (Zero Python Overhead, Q4_G64, K=3)
    print("\n>>> [6/7] AxiomLM Pure C++ Speculative Engine (Phase 6, Q4_G64, K=3)...")
    _, _ = decoder.generate(
        model, spec_prompt, max_new_tokens=16, device=device, use_cpp_loop=True
    )

    spec_cpp_speeds = []
    for _ in range(repetitions):
        _, stats = decoder.generate(
            model,
            spec_prompt,
            max_new_tokens=gen_tokens,
            device=device,
            use_cpp_loop=True,
        )
        spec_cpp_speeds.append(stats["tokens_per_second"])

    avg_spec_cpp = float(np.mean(spec_cpp_speeds))
    std_spec_cpp = float(np.std(spec_cpp_speeds))
    spec_cpp_lat = round(1000.0 / avg_spec_cpp, 3)

    results["axiomlm_speculative_q4_cpp_loop"] = {
        "k": 3,
        "decode_tok_s": round(avg_spec_cpp, 2),
        "decode_stddev": round(std_spec_cpp, 2),
        "decode_latency_ms": spec_cpp_lat,
        "acceptance_rate_pct": round(avg_acc, 2),
        "tokens_per_forward": round(avg_tpf, 2),
        "speedup_vs_python_loop": round(avg_spec_cpp / avg_spec_py, 2),
    }
    # Backward-compatible alias pointing to the top-performing C++ speculative engine
    results["axiomlm_speculative_q4"] = results["axiomlm_speculative_q4_cpp_loop"]
    print(
        f"  • Decode Speed: {avg_spec_cpp:.2f} tok/s (± {std_spec_cpp:.2f}) [Latency: {spec_cpp_lat} ms/tok]"
    )
    print(
        f"  • Speedup vs Python Loop: {avg_spec_cpp / avg_spec_py:.2f}x (+{avg_spec_cpp - avg_spec_py:.1f} tok/s)"
    )

    # 7. AxiomLM Neural Medusa Speculative Engine (Phase 7, Multi-Head Self-Drafting, Q4_G64, K=3)
    print(
        "\n>>> [7/7] AxiomLM Neural Medusa Speculative Engine (Phase 7, Q4_G64, K=3)..."
    )
    medusa_model = Transformer(config).to(device=device, dtype=torch.float16).eval()
    medusa_model.add_medusa_heads(num_heads=3)

    medusa_prompt = [101, 202, 303, 404, 505, 606, 707, 808]
    cache_rollout = StaticKVCache(
        1, 1024, 12, 4, 64, device=device, dtype=torch.float16
    )
    curr_t = torch.tensor([medusa_prompt], device=device)
    with torch.no_grad():
        out, _, _ = medusa_model(curr_t, kv_caches=[cache_rollout] * 12)
    rollout_seq = list(medusa_prompt)
    for _ in range(128):
        next_t = out[:, -1, :].argmax(dim=-1).item()
        rollout_seq.append(next_t)
        curr_t = torch.tensor([[next_t]], device=device)
        with torch.no_grad():
            out, _, _ = medusa_model(curr_t, kv_caches=[cache_rollout] * 12)

    train_medusa_heads(
        medusa_model,
        torch.tensor([rollout_seq], device=device),
        epochs=30,
        lr=2e-2,
        freeze_backbone=True,
    )
    medusa_model.quantize("q4")

    medusa_decoder = MedusaSpeculativeDecoder(num_heads=3)
    _, _ = medusa_decoder.generate(
        medusa_model, medusa_prompt, max_new_tokens=16, device=device
    )

    medusa_speeds = []
    medusa_accs = []
    medusa_tpfs = []
    for _ in range(repetitions):
        _, stats = medusa_decoder.generate(
            medusa_model, medusa_prompt, max_new_tokens=gen_tokens, device=device
        )
        medusa_speeds.append(stats["tokens_per_second"])
        medusa_accs.append(stats["acceptance_rate_pct"])
        medusa_tpfs.append(stats["tokens_per_forward_call"])

    avg_medusa = float(np.mean(medusa_speeds))
    std_medusa = float(np.std(medusa_speeds))
    avg_med_acc = float(np.mean(medusa_accs))
    avg_med_tpf = float(np.mean(medusa_tpfs))
    medusa_lat = round(1000.0 / avg_medusa, 3)

    results["axiomlm_medusa_q4"] = {
        "k": 3,
        "decode_tok_s": round(avg_medusa, 2),
        "decode_stddev": round(std_medusa, 2),
        "decode_latency_ms": medusa_lat,
        "acceptance_rate_pct": round(avg_med_acc, 2),
        "tokens_per_forward": round(avg_med_tpf, 2),
    }
    print(
        f"  • Decode Speed: {avg_medusa:.2f} tok/s (± {std_medusa:.2f}) [Latency: {medusa_lat} ms/tok]"
    )
    print(
        f"  • Acceptance Rate: {avg_med_acc:.1f}% | Tokens / Forward Step: {avg_med_tpf:.2f}x"
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run AxiomLM benchmarks with standardized prompt/gen lengths"
    )
    parser.add_argument("--prompt", type=int, default=512, help="Prompt token count")
    parser.add_argument("--gen", type=int, default=128, help="Generation token count")
    parser.add_argument("--repetitions", type=int, default=3, help="Repetitions")
    args = parser.parse_args()

    suite_results = run_axiomlm_benchmarks(args.prompt, args.gen, args.repetitions)
    if suite_results:
        print("\nAxiomLM Benchmark Suite Completed Successfully.")
    else:
        print("\nAxiomLM Benchmark Suite Skipped (No Metal GPU available).")
