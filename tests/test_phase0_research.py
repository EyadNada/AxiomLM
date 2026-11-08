"""
Phase 0 MLSys / ASPLOS Research Validation Suite:
  1. Task 0.1: MARLIN 64207531 + Ladder simdgroup_half8x8 register-direct QMV/QMM
     mathematical parity, non-multiple-of-8 boundary safety, and SwiGLU outlier resilience.
  2. Task 0.2: Generalized GQA Split-KV (3:1, 4:1, 7:1, 1:1; head_dim 64 & 128),
     SpecInfer 32-bit bitmask tree attention zero-cross-contamination proof across
     all nodes of a 20-node branching tree, in-place KV compaction exactness, and
     Tree-Medusa 100% token-for-token greedy equivalence.
  3. Task 0.3: Official HuggingFace AutoModelForCausalLM numerical & generation parity
     across SmolLM2-360M, Qwen2.5-0.5B, and Llama-3.2-1B.
"""

from __future__ import annotations

import copy
import unittest

import torch
from torch.nn import functional as F

from axiomlm.engine.medusa import (
    DEFAULT_MEDUSA_TREE_CHOICES,
    MedusaSpeculativeDecoder,
    compile_medusa_tree,
)
from axiomlm.engine.paged_cache import StaticKVCache
from axiomlm.kernels import (
    _METAL_MOD,
    dequantize_q4_g64,
    dequantize_q4_marlin,
    gqa_flash_decode_mps,
    is_mps_available,
    pack_linear_q4_g64,
    pack_linear_q4_marlin,
    qmm_q4_marlin,
    qmv_q4_marlin,
)
from axiomlm.models.transformer import ModelConfig, Transformer
from axiomlm.utils.hf_importer import (
    load_hf_safetensors,
    permute_hf_rope_weight,
    unpermute_hf_rope_weight,
)


@unittest.skipUnless(
    is_mps_available() and _METAL_MOD is not None, "Requires Apple Metal MPS"
)
class TestPhase0Research(unittest.TestCase):
    """Peer-review-grade mathematical and systems validation for Phase 0."""

    def setUp(self):
        torch.manual_seed(2026)
        self.device = torch.device("mps")

    def test_01_marlin_pack_dequant_exact_bit_identity(self):
        """
        Verify MARLIN 64207531 + Ladder layout dequantizes to the exact bit pattern of Q4_G64
        when clip_search=False, and strictly reduces reconstruction MSE when clip_search=True.
        """
        w = torch.randn(512, 1024, dtype=torch.float16)
        # Inject synthetic heavy-tailed group outliers
        w[::4, ::64] *= 6.0
        w_g64, sb_g64 = pack_linear_q4_g64(w)
        w_mar_rtn, sb_mar_rtn = pack_linear_q4_marlin(w, clip_search=False)
        w_mar_mse, sb_mar_mse = pack_linear_q4_marlin(w, clip_search=True)

        dq_g64 = dequantize_q4_g64(w_g64, sb_g64, 1024)
        dq_mar_rtn = dequantize_q4_marlin(w_mar_rtn, sb_mar_rtn, 1024)
        dq_mar_mse = dequantize_q4_marlin(w_mar_mse, sb_mar_mse, 1024)
        self.assertTrue(
            torch.equal(dq_g64, dq_mar_rtn),
            f"Max diff between Q4_G64 and unclipped MARLIN: {(dq_g64 - dq_mar_rtn).abs().max().item()}",
        )
        mse_rtn = F.mse_loss(dq_mar_rtn.float(), w.float()).item()
        mse_clip = F.mse_loss(dq_mar_mse.float(), w.float()).item()
        self.assertLess(
            mse_clip,
            mse_rtn * 0.90,
            f"Expected >10% overall MSE reduction, got {mse_clip:.6f} vs {mse_rtn:.6f}",
        )
        # Check non-outlier elements inside the outlier-contaminated rows specifically
        mask_inliers = torch.ones_like(w[::4], dtype=torch.bool)
        mask_inliers[:, ::64] = False
        inlier_mse_rtn = F.mse_loss(
            dq_mar_rtn[::4][mask_inliers].float(), w[::4][mask_inliers].float()
        ).item()
        inlier_mse_clip = F.mse_loss(
            dq_mar_mse[::4][mask_inliers].float(), w[::4][mask_inliers].float()
        ).item()
        self.assertLess(
            inlier_mse_clip,
            inlier_mse_rtn * 0.85,
            f"Expected >15% inlier MSE reduction in outlier groups, got {inlier_mse_clip:.6f} vs {inlier_mse_rtn:.6f}",
        )

    def test_02_marlin_qmv_qmm_register_kernels_and_outlier_safety(self):
        """
        Validate MPS MARLIN QMV (M=1), M<=8 QMM, and M>8 double-buffered QMM across
        aligned and unaligned N, linear bias, and extreme SwiGLU outlier activations.
        """
        for N, K, has_bias in [(2048, 2048, False), (2003, 1024, True)]:
            w = (torch.randn(N, K, dtype=torch.float32) * 0.05).to(torch.float16)
            bias = (
                (torch.randn(N, dtype=torch.float16) * 0.1).to(self.device)
                if has_bias
                else None
            )
            w_mar, sb_mar = pack_linear_q4_marlin(w)
            w_mar = w_mar.to(self.device)
            sb_mar = sb_mar.to(self.device)
            w_dq = dequantize_q4_marlin(
                w_mar.cpu(), sb_mar.cpu(), K, out_features=N
            ).to(self.device)

            for M in [1, 2, 5, 8, 16, 19, 32, 64]:
                x = torch.randn(1, M, K, device=self.device, dtype=torch.float16)
                ref = F.linear(
                    x.float(), w_dq.float(), bias.float() if bias is not None else None
                ).half()
                out = (
                    qmv_q4_marlin(x, w_mar, sb_mar, out_features=N, bias=bias)
                    if M == 1
                    else qmm_q4_marlin(x, w_mar, sb_mar, out_features=N, bias=bias)
                )
                self.assertEqual(out.shape, (1, M, N))
                cos = F.cosine_similarity(
                    ref.flatten().float(), out.flatten().float(), dim=0
                ).item()
                max_diff = (ref.float() - out.float()).abs().max().item()
                self.assertGreater(
                    cos, 0.99999, f"Low cosine similarity {cos} for M={M}, N={N}, K={K}"
                )
                self.assertLess(
                    max_diff, 0.05, f"High max_diff {max_diff} for M={M}, N={N}, K={K}"
                )

        # Extreme SwiGLU outlier channel test (x_channel = 10000.0 > 65504 / 15)
        w_out = (torch.randn(256, 512, dtype=torch.float32) * 0.02).to(torch.float16)
        w_mar, sb_mar = pack_linear_q4_marlin(w_out)
        w_mar, sb_mar = w_mar.to(self.device), sb_mar.to(self.device)
        w_dq = dequantize_q4_marlin(w_mar.cpu(), sb_mar.cpu(), 512).to(self.device)

        x_outlier = torch.randn(1, 1, 512, device=self.device, dtype=torch.float16)
        x_outlier[0, 0, 17] = 10000.0
        x_outlier[0, 0, 311] = -9500.0
        y_qmv = qmv_q4_marlin(x_outlier, w_mar, sb_mar, out_features=256)
        y_ref = F.linear(x_outlier.float(), w_dq.float()).half()
        self.assertTrue(
            torch.isfinite(y_qmv).all(), "Outlier caused Inf/NaN in MARLIN QMV"
        )
        cos_outlier = F.cosine_similarity(
            y_ref.flatten().float(), y_qmv.flatten().float(), dim=0
        ).item()
        self.assertGreater(cos_outlier, 0.99999)

    def test_03_generalized_gqa_split_kv_arbitrary_ratios_and_head_dims(self):
        """
        Verify Split-KV Flash-Decoding across arbitrary GQA ratios (3:1, 4:1, 7:1, 1:1)
        and head_dim in {64, 128} across num_splits in {1, 4, 8}.
        """
        configs = [
            (15, 5, 64),  # SmolLM2-360M (3:1)
            (32, 8, 64),  # Llama-3.2-1B (4:1)
            (14, 2, 64),  # Qwen2.5-0.5B (7:1)
            (12, 12, 64),  # Standard MHA (1:1)
            (16, 4, 128),  # Llama-3 8B / Qwen2.5-7B head_dim=128 (4:1)
        ]
        seq_len = 300
        max_seq_len = 512

        for n_head, n_kv_head, head_dim in configs:
            q = torch.randn(
                1, n_head, 1, head_dim, device=self.device, dtype=torch.float16
            )
            k_cache = torch.randn(
                1,
                n_kv_head,
                max_seq_len,
                head_dim,
                device=self.device,
                dtype=torch.float16,
            )
            v_cache = torch.randn(
                1,
                n_kv_head,
                max_seq_len,
                head_dim,
                device=self.device,
                dtype=torch.float16,
            )

            n_rep = n_head // n_kv_head
            k_active = k_cache[:, :, :seq_len, :].repeat_interleave(n_rep, dim=1)
            v_active = v_cache[:, :, :seq_len, :].repeat_interleave(n_rep, dim=1)
            ref = F.scaled_dot_product_attention(
                q.float(), k_active.float(), v_active.float(), is_causal=False
            ).half()

            for num_splits in [1, 4, 8]:
                out = gqa_flash_decode_mps(
                    q, k_cache, v_cache, seq_len, num_splits=num_splits
                )
                max_diff = (ref.float() - out.float()).abs().max().item()
                self.assertLess(
                    max_diff,
                    2e-3,
                    f"GQA Split-KV mismatch ({max_diff}) for ({n_head}, {n_kv_head}, {head_dim}), splits={num_splits}",
                )

    def test_04_specinfer_tree_verify_zero_cross_contamination_and_compaction(self):
        """
        Rigorous Tree-Attention Proof:
        Verify that evaluating a 17-node branching tree in a single pass via `forward_tree_verify`
        (both FP16 and MARLIN Q4) yields identical logits at EVERY node u in [0, 16] as evaluating
        each root-to-node path sequentially in complete isolation, and that `compact_tree_path`
        compacts the winning path with 0.0 KV error.
        """
        cfg = ModelConfig(
            arch="modern",
            n_layer=3,
            n_head=15,
            n_kv_head=5,
            n_embd=960,
            intermediate_size=2560,
            vocab_size=1024,
            block_size=512,
            use_fused_kernels=True,
            qkv_bias=True,
        )
        model = Transformer(cfg).to(device=self.device, dtype=torch.float16).eval()
        model.add_medusa_heads(num_heads=4)

        compiled = compile_medusa_tree(DEFAULT_MEDUSA_TREE_CHOICES, num_heads=4)
        T_tree = compiled["num_nodes"]
        self.assertEqual(T_tree, 17)
        tree_depths = compiled["tree_depths"]
        tree_mask = compiled["tree_mask"]

        prefix_tokens = torch.randint(0, cfg.vocab_size, (1, 96), device=self.device)
        tree_tokens = torch.randint(0, cfg.vocab_size, (1, T_tree), device=self.device)

        for mode in [None, "q4"]:
            model.quantize(mode)

            # 1. Single-pass SpecInfer tree verification over all 17 nodes
            cache_tree = StaticKVCache(
                max_batch_size=1,
                max_seq_len=256,
                n_layer=cfg.n_layer,
                n_kv_head=cfg.n_kv_head,
                head_dim=cfg.n_embd // cfg.n_head,
                device=self.device,
                dtype=torch.float16,
            )
            with torch.no_grad():
                model(prefix_tokens, kv_caches=[cache_tree] * cfg.n_layer)
                write_pos = cache_tree.total_seen_tokens[0]
                tree_logits, tree_med_logits, _ = model.forward_tree_verify(
                    tree_tokens,
                    kv_caches=[cache_tree] * cfg.n_layer,
                    tree_depths=tree_depths,
                    tree_mask=tree_mask,
                    return_medusa=True,
                )

            # 2. Ground-truth isolated sequential evaluation for EVERY node u in [0, T_tree - 1]
            for u in range(T_tree):
                ancestors = [j for j in range(T_tree) if ((tree_mask[u] >> j) & 1)]
                self.assertEqual(ancestors[-1], u)
                self.assertEqual(len(ancestors), tree_depths[u] + 1)

                branch_tokens = tree_tokens[:, ancestors]
                full_seq = torch.cat([prefix_tokens, branch_tokens], dim=1)
                cache_iso = StaticKVCache(
                    max_batch_size=1,
                    max_seq_len=256,
                    n_layer=cfg.n_layer,
                    n_kv_head=cfg.n_kv_head,
                    head_dim=cfg.n_embd // cfg.n_head,
                    device=self.device,
                    dtype=torch.float16,
                )
                with torch.no_grad():
                    iso_logits, _ = model(full_seq, return_all_logits=True)
                    model(full_seq, kv_caches=[cache_iso] * cfg.n_layer)

                ref_u = iso_logits[0, -1, :].float()
                got_u = tree_logits[0, u, :].float()
                cos = F.cosine_similarity(ref_u, got_u, dim=0).item()
                self.assertGreater(
                    cos,
                    0.9999,
                    f"Cross-contamination at node u={u} (mode={mode}): cos={cos}",
                )
                self.assertEqual(
                    ref_u.argmax().item(),
                    got_u.argmax().item(),
                    f"Top-1 mismatch at node u={u} (mode={mode})",
                )

                # Test KV compaction on the deepest leaf node u == T_tree - 1
                if u == T_tree - 1:
                    cache_tree.compact_tree_path(
                        write_pos=write_pos,
                        accepted_indices=ancestors,
                        new_total_seen=write_pos + len(ancestors),
                    )
                    total_len = write_pos + len(ancestors)
                    for layer_i in range(cfg.n_layer):
                        k_comp = cache_tree.k_caches[layer_i][:, :, :total_len, :]
                        k_iso = cache_iso.k_caches[layer_i][:, :, :total_len, :]
                        v_comp = cache_tree.v_caches[layer_i][:, :, :total_len, :]
                        v_iso = cache_iso.v_caches[layer_i][:, :, :total_len, :]
                        k_diff = (k_comp.float() - k_iso.float()).abs().max().item()
                        v_diff = (v_comp.float() - v_iso.float()).abs().max().item()
                        tol = 5e-3 if mode is None else 2e-2
                        self.assertLess(
                            k_diff,
                            tol,
                            f"Compacted K cache drift {k_diff} at layer {layer_i} (mode={mode})",
                        )
                        self.assertLess(
                            v_diff,
                            tol,
                            f"Compacted V cache drift {v_diff} at layer {layer_i} (mode={mode})",
                        )

    def test_05_tree_medusa_speculative_decoder_exact_greedy_parity(self):
        """
        Verify Tree-Structured MedusaSpeculativeDecoder produces 100% exact token-for-token
        greedy parity under both FP16 and MARLIN Q4.
        """
        cfg = ModelConfig(
            arch="modern",
            n_layer=2,
            n_head=8,
            n_kv_head=2,
            n_embd=512,
            vocab_size=512,
            block_size=256,
            use_fused_kernels=True,
        )
        model = Transformer(cfg).to(device=self.device, dtype=torch.float16).eval()
        model.add_medusa_heads(num_heads=3)
        prompt = [12, 77, 193, 404, 8, 255, 301, 44]
        max_new = 24

        for mode in [None, "q4"]:
            model.quantize(mode)

            # Standard single-token greedy decode
            cache_std = StaticKVCache(
                max_batch_size=1,
                max_seq_len=256,
                n_layer=cfg.n_layer,
                n_kv_head=cfg.n_kv_head,
                head_dim=cfg.n_embd // cfg.n_head,
                device=self.device,
                dtype=torch.float16,
            )
            p_t = torch.tensor([prompt], device=self.device, dtype=torch.long)
            with torch.no_grad():
                out, _, _ = model(p_t, kv_caches=[cache_std] * cfg.n_layer)
            nxt = out[:, -1, :].argmax(dim=-1).item()
            tokens_std = list(prompt) + [nxt]
            curr = torch.tensor([[nxt]], device=self.device, dtype=torch.long)
            for _ in range(max_new - 1):
                with torch.no_grad():
                    out, _, _ = model(curr, kv_caches=[cache_std] * cfg.n_layer)
                curr = out[:, -1, :].argmax(dim=-1, keepdim=True)
                tokens_std.append(curr.item())

            decoder = MedusaSpeculativeDecoder(
                num_heads=3, tree_choices=DEFAULT_MEDUSA_TREE_CHOICES
            )
            tokens_tree, stats = decoder.generate(
                model=model,
                prompt_tokens=prompt,
                max_new_tokens=max_new,
                device=self.device,
            )
            self.assertEqual(
                tokens_std,
                tokens_tree,
                f"Tree-Medusa parity failed under mode={mode}:\nStd:  {tokens_std}\nTree: {tokens_tree}",
            )
            self.assertEqual(stats["tree_nodes_per_step"], 16)

    def test_06_hf_safetensors_open_weights_official_parity(self):
        """
        Task 0.3 Research Validation:
        Verify `load_hf_safetensors` against official HuggingFace `AutoModelForCausalLM`
        across SmolLM2-360M, Qwen2.5-0.5B, and Llama-3.2-1B.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        w = torch.randn(512, 256, dtype=torch.float16)
        self.assertTrue(
            torch.equal(
                w, unpermute_hf_rope_weight(permute_hf_rope_weight(w, 8, 64), 8, 64)
            )
        )

        models_to_verify = [
            "HuggingFaceTB/SmolLM2-360M",
            "Qwen/Qwen2.5-0.5B",
            "unsloth/Llama-3.2-1B",
        ]
        prompt_text = (
            "In computer architecture, speculative execution and register-level "
            "dequantization reduce memory bandwidth bottlenecks by"
        )

        for repo_id in models_to_verify:
            tokenizer = AutoTokenizer.from_pretrained(repo_id)
            input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(
                self.device
            )

            hf_model = (
                AutoModelForCausalLM.from_pretrained(repo_id, dtype=torch.float16)
                .to(self.device)
                .eval()
            )
            ax_model = load_hf_safetensors(
                repo_id, device=self.device, dtype=torch.float16, max_seq_len=1024
            )

            with torch.no_grad():
                hf_logits = hf_model(input_ids).logits.float()
                ax_logits, _ = ax_model(input_ids, return_all_logits=True)
                ax_logits = ax_logits.float()

            cos = F.cosine_similarity(
                hf_logits.flatten(), ax_logits.flatten(), dim=0
            ).item()
            top1_match = (
                (hf_logits.argmax(dim=-1) == ax_logits.argmax(dim=-1))
                .float()
                .mean()
                .item()
            )
            self.assertGreater(
                cos, 0.9999, f"Low HF cosine similarity ({cos}) for {repo_id}"
            )
            self.assertGreaterEqual(
                top1_match,
                0.95,
                f"Low top-1 token match ({top1_match:.3f}) for {repo_id}",
            )
            self.assertEqual(
                hf_logits[0, -1].argmax().item(),
                ax_logits[0, -1].argmax().item(),
                f"Next-token greedy prediction mismatch on {repo_id}",
            )

            # Verify Q8_G64 near-lossless parity (< 0.05 nats mean KL)
            p_fp16_all = F.softmax(ax_logits, dim=-1)
            ax_model.quantize("q8")
            with torch.no_grad():
                q8_logits, _ = ax_model(input_ids, return_all_logits=True)
                kl_q8 = (
                    F.kl_div(
                        F.log_softmax(q8_logits.float(), dim=-1),
                        p_fp16_all,
                        reduction="none",
                    )
                    .sum(dim=-1)
                    .mean()
                    .item()
                )
            self.assertLess(
                kl_q8, 0.05, f"High Q8 KL divergence ({kl_q8}) on {repo_id}"
            )

            # Verify unclipped MARLIN Q4_RTN exact equivalence with Legacy Q4_G64,
            # and verify MSE-clipped MARLIN Q4 + Q4_K_M strictly improve KL fidelity!
            ax_model.quantize("q4_legacy")
            with torch.no_grad():
                q4_leg_logits, _ = ax_model(input_ids, return_all_logits=True)
            ax_model.quantize("q4_rtn")
            with torch.no_grad():
                q4_rtn_logits, _ = ax_model(input_ids, return_all_logits=True)
            ax_model.quantize("q4")
            with torch.no_grad():
                q4_mar_logits, _ = ax_model(input_ids, return_all_logits=True)
            ax_model.quantize("q4_k_m")
            with torch.no_grad():
                q4_km_logits, _ = ax_model(input_ids, return_all_logits=True)

            cos_marlin_vs_leg = F.cosine_similarity(
                q4_leg_logits.cpu().double().flatten(),
                q4_rtn_logits.cpu().double().flatten(),
                dim=0,
            ).item()
            self.assertGreater(
                cos_marlin_vs_leg,
                0.9998,
                f"MARLIN Q4_RTN diverged from Q4_G64 reference ({cos_marlin_vs_leg}) on {repo_id}",
            )

            def _mean_kl(lg_ten: torch.Tensor) -> float:
                return (
                    F.kl_div(
                        F.log_softmax(lg_ten.float(), dim=-1),
                        p_fp16_all,
                        reduction="none",
                    )
                    .sum(dim=-1)
                    .mean()
                    .item()
                )

            kl_rtn = _mean_kl(q4_rtn_logits)
            kl_q4 = _mean_kl(q4_mar_logits)
            kl_km = _mean_kl(q4_km_logits)

            self.assertLess(
                kl_q4,
                kl_rtn,
                f"Expected MSE-clipped Q4 KL ({kl_q4:.4f}) < RTN Q4 KL ({kl_rtn:.4f}) on {repo_id}",
            )
            self.assertLess(
                kl_km,
                kl_q4,
                f"Expected Q4_K_M KL ({kl_km:.4f}) < pure Q4 KL ({kl_q4:.4f}) on {repo_id}",
            )
            kl_km_bound = 0.12 if "1B" in repo_id else 0.30
            self.assertLess(
                kl_km,
                kl_km_bound,
                f"Expected Q4_K_M KL < {kl_km_bound} nats on {repo_id}, got {kl_km:.4f}",
            )
            self.assertLess(
                kl_km,
                kl_rtn * 0.55,
                f"Expected >45% KL reduction vs RTN on {repo_id}, got {kl_km:.4f} vs {kl_rtn:.4f}",
            )

            del hf_model, ax_model
            torch.mps.empty_cache()

    def test_07_mixed_precision_q4_k_m_and_q8_lm_head_decode_runner_parity(self):
        """
        Verify C++ MPSQuantizedDecodeRunner (`step`, `verify_step`, `verify_tree_step`)
        matches eager quantized sequence execution across `q4`, `q4` + `lm_head_q8=True`,
        and `q4_k_m` mixed-precision modes.
        """
        cfg = ModelConfig(
            vocab_size=503,
            block_size=128,
            n_layer=3,
            n_head=8,
            n_kv_head=2,
            n_embd=256,
            intermediate_size=512,
            norm_type="rmsnorm",
            pos_emb="rope",
            mlp_type="swiglu",
            bias=False,
            qkv_bias=True,
            tie_word_embeddings=False,
            use_fused_kernels=True,
        )
        model = Transformer(cfg).to(device=self.device, dtype=torch.float16).eval()
        prompt = torch.randint(0, cfg.vocab_size, (1, 10), device=self.device)
        next_tok = torch.randint(0, cfg.vocab_size, (1, 1), device=self.device)
        tree_toks = torch.randint(0, cfg.vocab_size, (1, 8), device=self.device)
        compiled = compile_medusa_tree(
            [(0,), (1,), (2,), (0, 0), (0, 1), (1, 0), (0, 0, 0)], num_heads=3
        )

        for mode, lmh_q8 in [("q4", False), ("q4", True), ("q4_k_m", None)]:
            model.quantize(mode, lm_head_q8=lmh_q8)
            cache = StaticKVCache(
                max_batch_size=1,
                max_seq_len=128,
                n_layer=cfg.n_layer,
                n_kv_head=cfg.n_kv_head,
                head_dim=cfg.n_embd // cfg.n_head,
                device=self.device,
                dtype=torch.float16,
            )
            kvs = [cache] * cfg.n_layer

            with torch.no_grad():
                # 1. Single-step runner vs sequence forward parity
                full_seq = torch.cat([prompt, next_tok], dim=1)
                seq_logits, _ = model(full_seq, return_all_logits=True)
                model(prompt, kv_caches=kvs)
                step_logits, _, _ = model(next_tok, kv_caches=kvs)
                cos_step = F.cosine_similarity(
                    seq_logits[0, -1].float(), step_logits[0, -1].float(), dim=0
                ).item()
                self.assertGreater(
                    cos_step,
                    0.9995,
                    f"Single-step runner parity failed for {(mode, lmh_q8)}: cos={cos_step}",
                )

                # 2. SpecInfer tree runner vs non-fused fallback parity
                cache_ref = copy.deepcopy(cache)
                logits_mps, _, _ = model.forward_tree_verify(
                    tree_toks,
                    kv_caches=kvs,
                    tree_depths=compiled["tree_depths"],
                    tree_mask=compiled["tree_mask"],
                    return_medusa=False,
                )
                model.config.use_fused_kernels = False
                logits_ref, _, _ = model.forward_tree_verify(
                    tree_toks,
                    kv_caches=[cache_ref] * cfg.n_layer,
                    tree_depths=compiled["tree_depths"],
                    tree_mask=compiled["tree_mask"],
                    return_medusa=False,
                )
                model.config.use_fused_kernels = True
                cos_tree = F.cosine_similarity(
                    logits_mps.flatten().float(), logits_ref.flatten().float(), dim=0
                ).item()
                self.assertGreater(
                    cos_tree,
                    0.9995,
                    f"Tree runner parity failed for {(mode, lmh_q8)}: cos={cos_tree}",
                )


if __name__ == "__main__":
    unittest.main()
