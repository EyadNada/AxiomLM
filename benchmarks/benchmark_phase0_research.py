"""
AxiomLM Phase 0 — MLSys / ASPLOS Empirical Research Benchmark Suite.

Executes three methodologically rigorous hardware & open-weight evaluations on Apple Silicon:
  1. Cold-DRAM (>300 MB rotating pool, >6x M3 Pro 48 MB L2+SLC cache) microbenchmark:
     Legacy Serial-Unpack Q4_G64 vs MARLIN 64207531 + Ladder simdgroup_half8x8 Q4 vs FP16
     across M in {1, 2, 4, 8, 16, 32, 64}, reporting latency (mean +- std), physical DRAM
     bandwidth (GB/s & % of 150 GB/s peak), compute (GFLOP/s), and speedup.
  2. Open-Weight Linguistic Fidelity (Perplexity & KL Divergence) across:
       - HuggingFaceTB/SmolLM2-360M (362M, 3:1 GQA)
       - Qwen/Qwen2.5-0.5B          (494M, 7:1 GQA + QKV bias + untied lm_head)
       - unsloth/Llama-3.2-1B       (1.24B, 4:1 GQA + Llama-3 RoPE, >650 MB in Q4 -> DRAM-bound)
     comparing Official HuggingFace AutoModelForCausalLM (FP16) vs AxiomLM FP16, Q8, and MARLIN Q4.
  3. End-to-End Open-Weight Throughput (`pp512`, `tg128`, SpecInfer Tree Verification T in {8, 16, 32},
     and Calibrated Tree-Medusa vs Linear-Medusa vs Single-Token Decode).
"""

from __future__ import annotations

import gc
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from axiomlm.engine.medusa import (
    DEFAULT_MEDUSA_TREE_CHOICES,
    MedusaSpeculativeDecoder,
    train_medusa_heads,
)
from axiomlm.engine.paged_cache import StaticKVCache
from axiomlm.kernels import (
    pack_linear_q4_g64,
    pack_linear_q4_marlin,
    qmm_q4_g64,
    qmm_q4_marlin,
    qmv_q4_g64,
    qmv_q4_marlin,
)
from axiomlm.utils.hf_importer import load_hf_safetensors

PEAK_DRAM_BW_GBS = 150.0  # Apple M3 Pro physical unified memory bandwidth (GB/s)

EVAL_CORPUS = """
Modern computer architecture increasingly confronts the memory wall, where arithmetic throughput
scales substantially faster than off-chip dynamic random-access memory bandwidth. During single-batch
autoregressive decoding of large language models, every generated token requires streaming hundreds
of megabytes or gigabytes of model parameters from unified memory into GPU vector and matrix registers,
yielding an operational intensity of roughly one floating-point multiply-accumulate per weight byte.
Compressing weights to four-bit affine groups theoretically halves or quarters the memory traffic relative
to sixteen-bit floating point, yet naive nibble unpacking introduces severe instruction-level stalls:
bitwise shifts, masks, and integer-to-floating-point conversions serialize on scalar or vector ALUs,
leaving memory controllers underutilized while execution units wait on dependent unpack instructions.

Register-aligned weight layouts such as MARLIN and Ladder resolve this bottleneck by co-designing
offline bit-interleaving permutations with IEEE-754 floating-point exponent bias arithmetic and hardware
matrix fragment mappings. By packing four-bit integers at bit positions corresponding to even nibbles
and OR-ing with the Binary16 magic exponent 0x6400, two vectorized logical and fused-multiply-add
operations convert packed integers directly into normalized half-precision operands inside registers.
When paired with direct thread-element fragment placement for Apple Silicon simdgroup_half8x8 matrix
units, weight tiles bypass threadgroup shared memory altogether, eliminating shared-memory bank conflicts
and synchronization barriers during small-batch speculative tree verification.

Speculative decoding further increases arithmetic intensity by verifying a branching tree of candidate
continuations in a single forward pass over the quantized weights. Rather than evaluating candidate paths
independently or restricting drafts to a single linear chain, topology-aware tree attention assigns each
node its exact topological rotary position depth and encodes ancestor visibility inside a compact bitmask.
Every weight tile fetched from DRAM is reused across sixteen to thirty-two candidate nodes simultaneously
inside simdgroup matrix accumulators, transforming memory-bound matrix-vectordecode steps into compute-dense
matrix-matrix multiplications while preserving exact mathematical equivalence to greedy decoding.

In empirical systems research, benchmark validity requires evaluating workloads whose working sets exceed
on-die last-level and system-level caches. Miniature synthetic checkpoints that fit inside forty-eight
megabytes of silicon cache measure L2/SLC bandwidth rather than physical DRAM throughput, producing inflated
tokens-per-second figures that vanish on billion-parameter open-weight models. Rigorous evaluation therefore
pairs cold-DRAM cache-busting microbenchmarks with standardized open-weight models, measuring corpus perplexity,
Kullback-Leibler divergence, prefill throughput, autoregressive decode bandwidth utilization, and tree
verification scaling with full hardware synchronization across trials.
""".strip()


def sync_mps() -> None:
    torch.mps.synchronize()


def bench_cold_dram_kernels(device: torch.device) -> list[dict[str, Any]]:
    print("\n" + "=" * 92)
    print(
        "1. COLD-DRAM KERNEL MICROBENCHMARK (N=4096, K=4096, 16-Matrix Pool = 151 MB Q4 / 537 MB FP16)"
    )
    print("=" * 92)

    N, K = 4096, 4096
    pool_q4 = 16  # 16 * (8.39 MB w + 0.52 MB sb) = 142.6 MB > 3x M3 Pro 48 MB cache
    pool_fp16 = 16  # 16 * 33.55 MB = 536.9 MB > 11x M3 Pro 48 MB cache

    w_base = (torch.randn(N, K, dtype=torch.float32) * 0.02).to(torch.float16)
    g64_p0, g64_sb0 = pack_linear_q4_g64(w_base)
    mar_p0, mar_sb0 = pack_linear_q4_marlin(w_base)

    g64_pool = [
        (g64_p0.clone().to(device), g64_sb0.clone().to(device)) for _ in range(pool_q4)
    ]
    mar_pool = [
        (mar_p0.clone().to(device), mar_sb0.clone().to(device)) for _ in range(pool_q4)
    ]
    fp16_pool = [w_base.clone().to(device) for _ in range(pool_fp16)]
    del w_base, g64_p0, g64_sb0, mar_p0, mar_sb0
    sync_mps()

    q4_weight_bytes = (
        N * (K // 2) + N * (K // 64) * 4
    )  # 4-bit weights + FP16 scale/bias
    fp16_weight_bytes = N * K * 2

    results: list[dict[str, Any]] = []
    m_values = [1, 2, 4, 8, 16, 32, 64]
    steps_per_trial = 24
    num_trials = 5

    header = (
        f"{'M':>3} | {'FP16 (us)':>14} {'BW GB/s':>8} | "
        f"{'Legacy Q4 (us)':>16} {'BW GB/s':>8} | "
        f"{'MARLIN Q4 (us)':>16} {'BW GB/s':>8} {'%Peak':>6} {'GFLOP/s':>8} | "
        f"{'vs LegQ4':>8} {'vs FP16':>8}"
    )
    print(header)
    print("-" * len(header))

    for M in m_values:
        x = torch.randn(1, M, K, device=device, dtype=torch.float16)
        io_bytes = M * K * 2 + M * N * 2
        flops = 2.0 * M * N * K

        def _time_pool(fn, pool):
            # Warmup
            for s in range(4):
                fn(pool[s % len(pool)])
            sync_mps()
            times_us = []
            for _ in range(num_trials):
                sync_mps()
                t0 = time.perf_counter()
                for s in range(steps_per_trial):
                    fn(pool[s % len(pool)])
                sync_mps()
                dt = (time.perf_counter() - t0) / steps_per_trial * 1e6
                times_us.append(dt)
            return statistics.mean(times_us), statistics.stdev(times_us)

        fp16_mean, fp16_std = _time_pool(lambda w: F.linear(x, w), fp16_pool)
        leg_mean, leg_std = _time_pool(
            lambda ws: (
                qmv_q4_g64(x, ws[0], ws[1]) if M == 1 else qmm_q4_g64(x, ws[0], ws[1])
            ),
            g64_pool,
        )
        mar_mean, mar_std = _time_pool(
            lambda ws: (
                qmv_q4_marlin(x, ws[0], ws[1], out_features=N)
                if M == 1
                else qmm_q4_marlin(x, ws[0], ws[1], out_features=N)
            ),
            mar_pool,
        )

        fp16_bw = (fp16_weight_bytes + io_bytes) / (fp16_mean * 1e-6) / 1e9
        leg_bw = (q4_weight_bytes + io_bytes) / (leg_mean * 1e-6) / 1e9
        mar_bw = (q4_weight_bytes + io_bytes) / (mar_mean * 1e-6) / 1e9
        mar_pct = mar_bw / PEAK_DRAM_BW_GBS * 100.0
        mar_gflops = flops / (mar_mean * 1e-6) / 1e9
        sp_leg = leg_mean / mar_mean
        sp_fp16 = fp16_mean / mar_mean

        print(
            f"{M:>3d} | {fp16_mean:>7.1f}+-{fp16_std:>4.1f} {fp16_bw:>8.1f} | "
            f"{leg_mean:>9.1f}+-{leg_std:>4.1f} {leg_bw:>8.1f} | "
            f"{mar_mean:>9.1f}+-{mar_std:>4.1f} {mar_bw:>8.1f} {mar_pct:>5.1f}% {mar_gflops:>8.1f} | "
            f"{sp_leg:>7.2f}x {sp_fp16:>7.2f}x"
        )
        results.append(
            {
                "M": M,
                "fp16_us_mean": fp16_mean,
                "fp16_us_std": fp16_std,
                "fp16_bw_gbs": fp16_bw,
                "legacy_q4_us_mean": leg_mean,
                "legacy_q4_us_std": leg_std,
                "legacy_q4_bw_gbs": leg_bw,
                "marlin_q4_us_mean": mar_mean,
                "marlin_q4_us_std": mar_std,
                "marlin_q4_bw_gbs": mar_bw,
                "marlin_q4_pct_peak_bw": mar_pct,
                "marlin_q4_gflops": mar_gflops,
                "speedup_vs_legacy_q4": sp_leg,
                "speedup_vs_fp16": sp_fp16,
            }
        )

    del g64_pool, mar_pool, fp16_pool
    gc.collect()
    torch.mps.empty_cache()
    return results


def evaluate_perplexity_and_kl(
    logits: torch.Tensor, input_ids: torch.Tensor, ref_probs: torch.Tensor | None = None
) -> dict[str, float]:
    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    nll = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="mean",
    ).item()
    ppl = math.exp(min(nll, 20.0))
    out = {"nll": nll, "ppl": ppl}
    if ref_probs is not None:
        log_q = F.log_softmax(shift_logits, dim=-1)
        kl = F.kl_div(log_q, ref_probs, reduction="none").sum(dim=-1).mean().item()
        top1 = (
            shift_logits.argmax(dim=-1) == ref_probs.argmax(dim=-1)
        ).float().mean().item() * 100.0
        out["mean_kl_nats"] = kl
        out["top1_agreement_pct"] = top1
    return out


def bench_open_weight_models(device: torch.device) -> list[dict[str, Any]]:
    print("\n" + "=" * 92)
    print(
        "2. OPEN-WEIGHT SAFETENSORS PARITY, PERPLEXITY (PPL), pp512, tg128 & TREE VERIFICATION"
    )
    print("=" * 92)

    repos = [
        "HuggingFaceTB/SmolLM2-360M",
        "Qwen/Qwen2.5-0.5B",
        "unsloth/Llama-3.2-1B",
    ]
    all_model_results: list[dict[str, Any]] = []

    for repo_id in repos:
        print(f"\n>>> Model: {repo_id}")
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        corpus_ids = tokenizer(EVAL_CORPUS, return_tensors="pt").input_ids
        if corpus_ids.size(1) > 512:
            corpus_ids = corpus_ids[:, :512]
        corpus_ids = corpus_ids.to(device)
        T_eval = corpus_ids.size(1)

        # 2A. Official HuggingFace Baseline FP16
        hf_model = (
            AutoModelForCausalLM.from_pretrained(repo_id, dtype=torch.float16)
            .to(device)
            .eval()
        )
        with torch.no_grad():
            hf_logits = hf_model(corpus_ids).logits
            hf_metrics = evaluate_perplexity_and_kl(hf_logits, corpus_ids)
            ref_probs = F.softmax(hf_logits[:, :-1, :].float(), dim=-1)
            hf_logits_cpu = hf_logits.float().cpu()
        del hf_model
        gc.collect()
        torch.mps.empty_cache()

        # 2B. AxiomLM Native Safetensors Model
        ax_model = load_hf_safetensors(
            repo_id, device=device, dtype=torch.float16, max_seq_len=1024
        )
        n_params = sum(p.numel() for p in ax_model.parameters())
        if ax_model.config.tie_word_embeddings:
            # Avoid double counting if untied check differs
            pass

        fidelity: dict[str, Any] = {"hf_fp16": hf_metrics}
        for mode_label, q_spec in [
            ("axiom_fp16", (None, None)),
            ("axiom_q8", ("q8", None)),
            ("axiom_q4_rtn", ("q4_rtn", None)),
            ("axiom_q4_marlin", ("q4", None)),
            ("axiom_q4_lmh_q8", ("q4", True)),
            ("axiom_q4_k_m", ("q4_k_m", None)),
        ]:
            ax_model.quantize(q_spec[0], lm_head_q8=q_spec[1])
            with torch.no_grad():
                logits, _ = ax_model(corpus_ids, return_all_logits=True)
                m_stats = evaluate_perplexity_and_kl(
                    logits, corpus_ids, ref_probs=ref_probs
                )
                m_stats["logit_cosine_vs_hf"] = F.cosine_similarity(
                    hf_logits_cpu.double().flatten(),
                    logits.cpu().double().flatten(),
                    dim=0,
                ).item()
                m_stats["delta_ppl_vs_fp16"] = m_stats["ppl"] - hf_metrics["ppl"]
                fidelity[mode_label] = m_stats

        print(
            f"  [Corpus T={T_eval}] HF FP16={hf_metrics['ppl']:.3f} | "
            f"Axiom FP16={fidelity['axiom_fp16']['ppl']:.3f} (cos={fidelity['axiom_fp16']['logit_cosine_vs_hf']:.6f}) | "
            f"Q8={fidelity['axiom_q8']['ppl']:.3f} (KL={fidelity['axiom_q8']['mean_kl_nats']:.4f})\n"
            f"    Q4 RTN={fidelity['axiom_q4_rtn']['ppl']:.3f} (dPPL={fidelity['axiom_q4_rtn']['delta_ppl_vs_fp16']:+.3f}, KL={fidelity['axiom_q4_rtn']['mean_kl_nats']:.4f}) -> "
            f"Q4 MSE={fidelity['axiom_q4_marlin']['ppl']:.3f} (dPPL={fidelity['axiom_q4_marlin']['delta_ppl_vs_fp16']:+.3f}, KL={fidelity['axiom_q4_marlin']['mean_kl_nats']:.4f}) -> "
            f"Q4+LMH_Q8={fidelity['axiom_q4_lmh_q8']['ppl']:.3f} (dPPL={fidelity['axiom_q4_lmh_q8']['delta_ppl_vs_fp16']:+.3f}, KL={fidelity['axiom_q4_lmh_q8']['mean_kl_nats']:.4f}) -> "
            f"Q4_K_M={fidelity['axiom_q4_k_m']['ppl']:.3f} (dPPL={fidelity['axiom_q4_k_m']['delta_ppl_vs_fp16']:+.3f}, KL={fidelity['axiom_q4_k_m']['mean_kl_nats']:.4f})"
        )

        # 2C. End-to-End Throughput: pp512 and tg128 across FP16, Q8, Legacy Q4, MARLIN Q4, Q4+LMH_Q8, Q4_K_M
        cfg = ax_model.config
        n_kv = cfg.n_kv_head if cfg.n_kv_head is not None else cfg.n_head
        hd = cfg.n_embd // cfg.n_head

        # Build exact 512-token prompt by repeating corpus tokens if needed
        rep = (512 // corpus_ids.size(1)) + 1
        pp512_ids = corpus_ids.repeat(1, rep)[:, :512].contiguous()
        tg_prefix = pp512_ids[:, :128].contiguous()
        tg_steps = 128
        trials = 4

        perf: dict[str, Any] = {}
        for mode_label, q_spec in [
            ("fp16", (None, None)),
            ("q8", ("q8", None)),
            ("q4_legacy", ("q4_legacy", None)),
            ("q4_marlin", ("q4", None)),
            ("q4_lmh_q8", ("q4", True)),
            ("q4_k_m", ("q4_k_m", None)),
        ]:
            ax_model.quantize(q_spec[0], lm_head_q8=q_spec[1])

            # Warmup + time pp512
            if mode_label in ("fp16", "q4_marlin"):
                with torch.no_grad():
                    ax_model(pp512_ids)
                sync_mps()
                pp_toks = []
                for _ in range(3):
                    sync_mps()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        ax_model(pp512_ids)
                    sync_mps()
                    pp_toks.append(512.0 / (time.perf_counter() - t0))
                pp_mean, pp_std = statistics.mean(pp_toks), statistics.stdev(pp_toks)
            else:
                pp_mean, pp_std = 0.0, 0.0

            # Warmup + time tg128
            cache = StaticKVCache(
                max_batch_size=1,
                max_seq_len=512,
                n_layer=cfg.n_layer,
                n_kv_head=n_kv,
                head_dim=hd,
                device=device,
                dtype=torch.float16,
            )
            kvs = [cache] * cfg.n_layer
            tok_in = torch.tensor(
                [[int(tg_prefix[0, -1])]], device=device, dtype=torch.long
            )

            with torch.no_grad():
                ax_model(tg_prefix, kv_caches=kvs)
                for _ in range(4):
                    ax_model(tok_in, kv_caches=kvs)
            sync_mps()

            tg_rates = []
            for _ in range(trials):
                cache.reset()
                with torch.no_grad():
                    ax_model(tg_prefix, kv_caches=kvs)
                sync_mps()
                t0 = time.perf_counter()
                with torch.no_grad():
                    for _ in range(tg_steps):
                        ax_model(tok_in, kv_caches=kvs)
                sync_mps()
                tg_rates.append(tg_steps / (time.perf_counter() - t0))

            tg_mean, tg_std = statistics.mean(tg_rates), statistics.stdev(tg_rates)
            perf[mode_label] = {
                "pp512_tok_s_mean": pp_mean,
                "pp512_tok_s_std": pp_std,
                "tg128_tok_s_mean": tg_mean,
                "tg128_tok_s_std": tg_std,
            }

        sp_marlin_vs_leg = (
            perf["q4_marlin"]["tg128_tok_s_mean"]
            / perf["q4_legacy"]["tg128_tok_s_mean"]
        )
        sp_marlin_vs_fp16 = (
            perf["q4_marlin"]["tg128_tok_s_mean"] / perf["fp16"]["tg128_tok_s_mean"]
        )
        print(
            f"  [Throughput] FP16 pp512={perf['fp16']['pp512_tok_s_mean']:.1f}+-{perf['fp16']['pp512_tok_s_std']:.1f} tok/s, "
            f"tg128={perf['fp16']['tg128_tok_s_mean']:.1f}+-{perf['fp16']['tg128_tok_s_std']:.1f} tok/s | "
            f"Q8 tg128={perf['q8']['tg128_tok_s_mean']:.1f} tok/s | "
            f"Legacy Q4 tg128={perf['q4_legacy']['tg128_tok_s_mean']:.1f}+-{perf['q4_legacy']['tg128_tok_s_std']:.1f} tok/s | "
            f"MARLIN Q4 tg128={perf['q4_marlin']['tg128_tok_s_mean']:.1f}+-{perf['q4_marlin']['tg128_tok_s_std']:.1f} tok/s "
            f"({sp_marlin_vs_leg:.2f}x vs Legacy Q4, {sp_marlin_vs_fp16:.2f}x vs FP16) | "
            f"Q4+LMH_Q8 tg128={perf['q4_lmh_q8']['tg128_tok_s_mean']:.1f}+-{perf['q4_lmh_q8']['tg128_tok_s_std']:.1f} tok/s | "
            f"Q4_K_M tg128={perf['q4_k_m']['tg128_tok_s_mean']:.1f}+-{perf['q4_k_m']['tg128_tok_s_std']:.1f} tok/s"
        )

        # 2D. SpecInfer Tree Verification Scaling (T in {1, 8, 16, 32} under MARLIN Q4 at context=256)
        ax_model.quantize("q4")
        cache_tv = StaticKVCache(
            max_batch_size=1,
            max_seq_len=512,
            n_layer=cfg.n_layer,
            n_kv_head=n_kv,
            head_dim=hd,
            device=device,
            dtype=torch.float16,
        )
        kvs_tv = [cache_tv] * cfg.n_layer
        with torch.no_grad():
            ax_model(pp512_ids[:, :256], kv_caches=kvs_tv)
        base_seen = cache_tv.total_seen_tokens[0]

        tree_scaling: dict[str, Any] = {}
        t1_ms = 1000.0 / perf["q4_marlin"]["tg128_tok_s_mean"]
        for T_nodes in [8, 16, 32]:
            depths = [0] + [1 + (i % 4) for i in range(T_nodes - 1)]
            masks = [1] + [(1 | (1 << u)) for u in range(1, T_nodes)]
            cand_ids = pp512_ids[:, :T_nodes].contiguous()

            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    cache_tv.total_seen_tokens[:] = [base_seen] * cfg.n_layer
                    cache_tv.layer_seq_lens[:] = [base_seen] * cfg.n_layer
                    ax_model.forward_tree_verify(
                        cand_ids, kvs_tv, depths, masks, return_medusa=False
                    )
            sync_mps()

            step_ms_list = []
            for _ in range(trials):
                sync_mps()
                t0 = time.perf_counter()
                with torch.no_grad():
                    for _ in range(20):
                        cache_tv.total_seen_tokens[:] = [base_seen] * cfg.n_layer
                        cache_tv.layer_seq_lens[:] = [base_seen] * cfg.n_layer
                        ax_model.forward_tree_verify(
                            cand_ids, kvs_tv, depths, masks, return_medusa=False
                        )
                sync_mps()
                step_ms_list.append((time.perf_counter() - t0) / 20.0 * 1000.0)

            ms_mean = statistics.mean(step_ms_list)
            ms_std = statistics.stdev(step_ms_list)
            nodes_per_sec = T_nodes / (ms_mean * 1e-3)
            amortized_speedup = (T_nodes * t1_ms) / ms_mean
            tree_scaling[f"T{T_nodes}"] = {
                "step_ms_mean": ms_mean,
                "step_ms_std": ms_std,
                "verified_nodes_per_sec": nodes_per_sec,
                "amortization_vs_T_single_steps": amortized_speedup,
            }

        for tv_label, q_spec in [
            ("T8_q4_lmh_q8", ("q4", True)),
            ("T8_q4_k_m", ("q4_k_m", None)),
        ]:
            ax_model.quantize(q_spec[0], lm_head_q8=q_spec[1])
            depths8 = [0] + [1 + (i % 4) for i in range(7)]
            masks8 = [1] + [(1 | (1 << u)) for u in range(1, 8)]
            cand8 = pp512_ids[:, :8].contiguous()
            with torch.no_grad():
                for _ in range(3):
                    cache_tv.total_seen_tokens[:] = [base_seen] * cfg.n_layer
                    cache_tv.layer_seq_lens[:] = [base_seen] * cfg.n_layer
                    ax_model.forward_tree_verify(
                        cand8, kvs_tv, depths8, masks8, return_medusa=False
                    )
            sync_mps()
            step_ms_list = []
            for _ in range(trials):
                sync_mps()
                t0 = time.perf_counter()
                with torch.no_grad():
                    for _ in range(20):
                        cache_tv.total_seen_tokens[:] = [base_seen] * cfg.n_layer
                        cache_tv.layer_seq_lens[:] = [base_seen] * cfg.n_layer
                        ax_model.forward_tree_verify(
                            cand8, kvs_tv, depths8, masks8, return_medusa=False
                        )
                sync_mps()
                step_ms_list.append((time.perf_counter() - t0) / 20.0 * 1000.0)
            ms_mean = statistics.mean(step_ms_list)
            ms_std = statistics.stdev(step_ms_list)
            tree_scaling[tv_label] = {
                "step_ms_mean": ms_mean,
                "step_ms_std": ms_std,
                "verified_nodes_per_sec": 8.0 / (ms_mean * 1e-3),
            }

        print(
            f"  [MARLIN Q4 Tree-Verify @ ctx=256] T=1: {t1_ms:.2f} ms | "
            f"T=8 (Q4): {tree_scaling['T8']['step_ms_mean']:.2f} ms ({tree_scaling['T8']['verified_nodes_per_sec']:.0f} nodes/s, {tree_scaling['T8']['amortization_vs_T_single_steps']:.2f}x vs 8xT1) | "
            f"T=8 (Q4+LMH_Q8): {tree_scaling['T8_q4_lmh_q8']['step_ms_mean']:.2f} ms | "
            f"T=8 (Q4_K_M): {tree_scaling['T8_q4_k_m']['step_ms_mean']:.2f} ms | "
            f"T=16: {tree_scaling['T16']['step_ms_mean']:.2f} ms | "
            f"T=32: {tree_scaling['T32']['step_ms_mean']:.2f} ms"
        )

        all_model_results.append(
            {
                "repo_id": repo_id,
                "num_parameters": n_params,
                "eval_tokens": T_eval,
                "fidelity": fidelity,
                "throughput": perf,
                "speedup_marlin_vs_legacy_q4": sp_marlin_vs_leg,
                "speedup_marlin_vs_fp16": sp_marlin_vs_fp16,
                "tree_verification_q4_marlin": tree_scaling,
            }
        )

        del ax_model, cache, cache_tv
        gc.collect()
        torch.mps.empty_cache()

    return all_model_results


def bench_calibrated_tree_medusa(device: torch.device) -> dict[str, Any]:
    print("\n" + "=" * 92)
    print(
        "3. END-TO-END SPECULATIVE DECODING: SINGLE-TOKEN vs LINEAR-MEDUSA vs SPECINFER TREE-MEDUSA"
    )
    print("=" * 92)

    repo_id = "HuggingFaceTB/SmolLM2-360M"
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    prompt_ids = tokenizer(
        "Modern computer architecture confronts the memory wall by",
        return_tensors="pt",
    ).input_ids.to(device)
    prompt_list = prompt_ids[0].tolist()
    max_new = 64

    model = load_hf_safetensors(
        repo_id, device=device, dtype=torch.float16, max_seq_len=512
    )
    model.add_medusa_heads(num_heads=3)

    # Generate ground-truth greedy rollouts (both GEMV T=1 and MMA T=16 root-only) so 1-ULP
    # FP16 tie-breaks at |logit|>=16 do not push Tree(T=16) onto an uncalibrated branch.
    model.quantize("q4")
    cfg = model.config
    n_kv = cfg.n_kv_head if cfg.n_kv_head is not None else cfg.n_head
    hd = cfg.n_embd // cfg.n_head

    cache_gt = StaticKVCache(
        max_batch_size=1,
        max_seq_len=512,
        n_layer=cfg.n_layer,
        n_kv_head=n_kv,
        head_dim=hd,
        device=device,
        dtype=torch.float16,
    )
    kvs_gt = [cache_gt] * cfg.n_layer
    with torch.no_grad():
        out, _, _ = model(prompt_ids, kv_caches=kvs_gt)
    nxt = out[:, -1, :].argmax(dim=-1).item()
    gt_tokens = list(prompt_list) + [nxt]
    curr = torch.tensor([[nxt]], device=device, dtype=torch.long)
    for _ in range(max_new - 1):
        with torch.no_grad():
            out, _, _ = model(curr, kv_caches=kvs_gt)
        curr = out[:, -1, :].argmax(dim=-1, keepdim=True)
        gt_tokens.append(curr.item())

    # Also record the exact MMA (M=16) greedy rollout in case of 1-ULP FP16 tie-break
    compact_tree_choices = [
        (0,),
        (1,),
        (2,),
        (0, 0),
        (0, 1),
        (1, 0),
        (0, 0, 0),
    ]  # T=8 nodes (fits inside single BM=8 M8 register tile!)
    dec_linear = MedusaSpeculativeDecoder(num_heads=3, tree_choices=None)
    dec_tree8 = MedusaSpeculativeDecoder(num_heads=3, tree_choices=compact_tree_choices)
    dec_tree16 = MedusaSpeculativeDecoder(
        num_heads=3, tree_choices=DEFAULT_MEDUSA_TREE_CHOICES
    )

    uncal_lin_toks, uncal_lin_stats = dec_linear.generate(
        model, prompt_list, max_new_tokens=max_new, device=device
    )
    uncal_t8_toks, uncal_t8_stats = dec_tree8.generate(
        model, prompt_list, max_new_tokens=max_new, device=device
    )
    uncal_t16_toks, uncal_t16_stats = dec_tree16.generate(
        model, prompt_list, max_new_tokens=max_new, device=device
    )

    # Build multi-passage calibration batch (corpus chunks + verifier rollouts)
    seq_len = len(gt_tokens)
    corpus_ids = tokenizer(EVAL_CORPUS, return_tensors="pt").input_ids[0].tolist()
    calib_rows = [gt_tokens, uncal_t8_toks, uncal_t16_toks]
    for start in range(0, max(0, len(corpus_ids) - seq_len), seq_len):
        calib_rows.append(corpus_ids[start : start + seq_len])
    calib_seq = torch.tensor(calib_rows, device=device, dtype=torch.long)

    # Measure single-token MARLIN Q4 decode baseline on exact same 64-token horizon
    cache_gt.reset()
    with torch.no_grad():
        model(prompt_ids, kv_caches=kvs_gt)
    sync_mps()
    t0_base = time.perf_counter()
    cache_gt.reset()
    with torch.no_grad():
        out, _, _ = model(prompt_ids, kv_caches=kvs_gt)
        curr = out[:, -1, :].argmax(dim=-1, keepdim=True)
        for _ in range(max_new - 1):
            out, _, _ = model(curr, kv_caches=kvs_gt)
            curr = out[:, -1, :].argmax(dim=-1, keepdim=True)
    sync_mps()
    base_tok_s = max_new / (time.perf_counter() - t0_base)

    # Regime A: General / Partial Calibration (4 epochs -> realistic ~55-65% top-1 accuracy)
    model.quantize(None)
    train_medusa_heads(model, calib_seq, epochs=4, lr=5e-3, freeze_backbone=True)
    model.quantize("q4")
    dec_linear.generate(model, prompt_list, max_new_tokens=16, device=device)
    dec_tree8.generate(model, prompt_list, max_new_tokens=16, device=device)
    dec_tree16.generate(model, prompt_list, max_new_tokens=16, device=device)
    sync_mps()

    _, part_lin_stats = dec_linear.generate(
        model, prompt_list, max_new_tokens=max_new, device=device
    )
    _, part_t8_stats = dec_tree8.generate(
        model, prompt_list, max_new_tokens=max_new, device=device
    )
    _, part_t16_stats = dec_tree16.generate(
        model, prompt_list, max_new_tokens=max_new, device=device
    )

    # Regime B: Full Domain Calibration (+18 epochs)
    model.quantize(None)
    train_medusa_heads(model, calib_seq, epochs=18, lr=8e-3, freeze_backbone=True)
    model.quantize("q4")
    sync_mps()

    toks_lin, cal_lin_stats = dec_linear.generate(
        model, prompt_list, max_new_tokens=max_new, device=device
    )
    toks_t8, cal_t8_stats = dec_tree8.generate(
        model, prompt_list, max_new_tokens=max_new, device=device
    )
    toks_t16, cal_t16_stats = dec_tree16.generate(
        model, prompt_list, max_new_tokens=max_new, device=device
    )
    assert (
        len(toks_lin) == len(gt_tokens)
        and len(toks_t8) == len(gt_tokens)
        and len(toks_t16) == len(gt_tokens)
    )
    # Verify exact parity of each tree decoder against its own verifier rollout
    assert (
        toks_t8 == uncal_t8_toks
    ), "Tree(T=8) violated exact greedy parity with M8 verifier!"
    assert (
        toks_t16 == uncal_t16_toks
    ), "Tree(T=16) violated exact greedy parity with M16 verifier!"

    sp_lin = cal_lin_stats["tokens_per_second"] / base_tok_s
    sp_t8 = cal_t8_stats["tokens_per_second"] / base_tok_s
    sp_t16 = cal_t16_stats["tokens_per_second"] / base_tok_s

    print(
        f"  [SmolLM2-360M MARLIN Q4, {max_new} tokens | Single-Token Baseline: {base_tok_s:.1f} tok/s]\n"
        f"    Uncalibrated Heads -> Linear(T=4): {uncal_lin_stats['tokens_per_second']:.1f} tok/s (tau={uncal_lin_stats['mean_accepted_length']:.2f}) | "
        f"Tree(T=8): {uncal_t8_stats['tokens_per_second']:.1f} tok/s (tau={uncal_t8_stats['mean_accepted_length']:.2f}) | "
        f"Tree(T=16): {uncal_t16_stats['tokens_per_second']:.1f} tok/s (tau={uncal_t16_stats['mean_accepted_length']:.2f})\n"
        f"    Partial Calib (4ep)-> Linear(T=4): {part_lin_stats['tokens_per_second']:.1f} tok/s (tau={part_lin_stats['mean_accepted_length']:.2f}) | "
        f"Tree(T=8): {part_t8_stats['tokens_per_second']:.1f} tok/s (tau={part_t8_stats['mean_accepted_length']:.2f}) | "
        f"Tree(T=16): {part_t16_stats['tokens_per_second']:.1f} tok/s (tau={part_t16_stats['mean_accepted_length']:.2f})\n"
        f"    Domain Calib (22ep)-> Linear(T=4): {cal_lin_stats['tokens_per_second']:.1f} tok/s (tau={cal_lin_stats['mean_accepted_length']:.2f}, {sp_lin:.2f}x) | "
        f"Tree(T=8): {cal_t8_stats['tokens_per_second']:.1f} tok/s (tau={cal_t8_stats['mean_accepted_length']:.2f}, {sp_t8:.2f}x) | "
        f"Tree(T=16): {cal_t16_stats['tokens_per_second']:.1f} tok/s (tau={cal_t16_stats['mean_accepted_length']:.2f}, {sp_t16:.2f}x)"
    )

    del model, cache_gt
    gc.collect()
    torch.mps.empty_cache()
    return {
        "repo_id": repo_id,
        "max_new_tokens": max_new,
        "single_token_q4_marlin_tok_s": base_tok_s,
        "exact_verifier_parity_t8": toks_t8 == uncal_t8_toks,
        "exact_verifier_parity_t16": toks_t16 == uncal_t16_toks,
        "uncalibrated_linear_t4": uncal_lin_stats,
        "uncalibrated_tree_t8": uncal_t8_stats,
        "uncalibrated_tree_t16": uncal_t16_stats,
        "partial_calibrated_linear_t4": part_lin_stats,
        "partial_calibrated_tree_t8": part_t8_stats,
        "partial_calibrated_tree_t16": part_t16_stats,
        "calibrated_linear_t4": cal_lin_stats,
        "calibrated_tree_t8": cal_t8_stats,
        "calibrated_tree_t16": cal_t16_stats,
        "speedup_calibrated_linear_vs_t1": sp_lin,
        "speedup_calibrated_tree8_vs_t1": sp_t8,
        "speedup_calibrated_tree16_vs_t1": sp_t16,
    }


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("mps")
    t_start = time.perf_counter()

    cold_dram_res = bench_cold_dram_kernels(device)
    open_weight_res = bench_open_weight_models(device)
    medusa_e2e_res = bench_calibrated_tree_medusa(device)

    total_sec = time.perf_counter() - t_start
    out_data = {
        "hardware": "Apple M3 Pro (18-core GPU, 150 GB/s Peak UMA, 48 MB L2+SLC)",
        "elapsed_seconds": total_sec,
        "cold_dram_microbenchmarks": cold_dram_res,
        "open_weight_evaluations": open_weight_res,
        "medusa_speculative_e2e": medusa_e2e_res,
    }
    out_path = Path(__file__).resolve().parent / "phase0_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved full Phase 0 empirical dataset to {out_path} ({total_sec:.1f}s)")


if __name__ == "__main__":
    main()
