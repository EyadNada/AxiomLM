"""
Tests for AxiomLM Speculative Prompt-Lookup Decoding Engine.
"""

import unittest
import torch
from axiomlm.models.transformer import Transformer, ModelConfig
from axiomlm.engine.paged_cache import StaticKVCache
from axiomlm.engine.speculative import (
    find_candidate_ngram,
    SpeculativePromptLookupDecoder,
)


class TestSpeculativePromptLookup(unittest.TestCase):
    def test_find_candidate_ngram(self):
        # Case 1: Simple repetition
        # tokens: [1, 2, 3, 4, 5, 1, 2, 3] -> matching [1, 2, 3] finds draft [4, 5]
        tokens = [1, 2, 3, 4, 5, 1, 2, 3]
        draft = find_candidate_ngram(tokens, ngram_size=3, max_spec_tokens=4)
        self.assertEqual(draft, [4, 5])

        # Case 2: Max spec tokens limits length
        tokens = [10, 20, 30, 40, 50, 60, 70, 10, 20, 30]
        draft = find_candidate_ngram(tokens, ngram_size=3, max_spec_tokens=2)
        self.assertEqual(draft, [40, 50])

        # Case 3: No match
        tokens = [1, 2, 3, 4, 5, 6, 7]
        draft = find_candidate_ngram(tokens, ngram_size=3, max_spec_tokens=3)
        self.assertEqual(draft, [])

        # Case 4: Sequence shorter than ngram + 1
        draft = find_candidate_ngram([1, 2], ngram_size=3)
        self.assertEqual(draft, [])

        # Case 5: Match occurs at the very end (no tokens follow match before target)
        tokens = [1, 2, 3, 1, 2, 3]
        draft = find_candidate_ngram(tokens, ngram_size=3)
        self.assertEqual(draft, [])

    def test_speculative_parity_greedy(self):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        torch.manual_seed(42)
        config = ModelConfig(
            arch="modern",
            n_layer=2,
            n_head=4,
            n_kv_head=2,
            n_embd=256,
            vocab_size=512,
            block_size=512,
            use_fused_kernels=True,
        )
        dtype = torch.float16 if device.type == "mps" else torch.float32
        model = Transformer(config).to(device=device, dtype=dtype).eval()

        pattern = [10, 25, 30, 45, 50, 12, 78, 34, 99, 120]
        prompt = pattern * 5 + pattern[:3]
        max_new_tokens = 32

        # 1. Standard autoregressive greedy generation
        cache_std = StaticKVCache(
            max_batch_size=1,
            max_seq_len=512,
            n_layer=config.n_layer,
            n_kv_head=config.n_kv_head,
            head_dim=config.n_embd // config.n_head,
            device=device,
            dtype=dtype,
        )
        p_tensor = torch.tensor([prompt], device=device, dtype=torch.long)
        with torch.no_grad():
            out, _, _ = model(p_tensor, kv_caches=[cache_std] * config.n_layer)
        next_t = out[:, -1, :].argmax(dim=-1).item()
        tokens_std = list(prompt) + [next_t]

        curr = torch.tensor([[next_t]], device=device, dtype=torch.long)
        for _ in range(max_new_tokens - 1):
            with torch.no_grad():
                out, _, _ = model(curr, kv_caches=[cache_std] * config.n_layer)
            curr = out[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens_std.append(curr.item())

        # 2. Speculative prompt lookup decoding
        decoder = SpeculativePromptLookupDecoder(ngram_size=2, max_spec_tokens=3)
        tokens_spec, stats = decoder.generate(
            model=model,
            prompt_tokens=prompt,
            max_new_tokens=max_new_tokens,
            device=device,
        )

        # 3. Assert mathematical exact parity
        self.assertEqual(len(tokens_std), len(tokens_spec))
        self.assertEqual(tokens_std, tokens_spec)
        self.assertGreater(stats["drafted_tokens"], 0)
        self.assertGreater(stats["accepted_draft_tokens"], 0)
        self.assertLess(stats["forward_calls"], max_new_tokens)

    def test_quantized_q4_speculative_parity(self):
        if not torch.backends.mps.is_available():
            return
        device = torch.device("mps")
        torch.manual_seed(42)
        config = ModelConfig(
            arch="modern",
            n_layer=2,
            n_head=4,
            n_kv_head=2,
            n_embd=256,
            vocab_size=512,
            block_size=512,
            use_fused_kernels=True,
        )
        model = Transformer(config).to(device=device, dtype=torch.float16).eval()
        model.quantize("q4")

        pattern = [10, 25, 30, 45, 50, 12, 78, 34, 99, 120]
        prompt = pattern * 5 + pattern[:3]
        max_new_tokens = 24

        # 1. Standard autoregressive greedy generation under Q4
        cache_std = StaticKVCache(
            max_batch_size=1,
            max_seq_len=512,
            n_layer=config.n_layer,
            n_kv_head=config.n_kv_head,
            head_dim=config.n_embd // config.n_head,
            device=device,
            dtype=torch.float16,
        )
        p_tensor = torch.tensor([prompt], device=device, dtype=torch.long)
        with torch.no_grad():
            out, _, _ = model(p_tensor, kv_caches=[cache_std] * config.n_layer)
        next_t = out[:, -1, :].argmax(dim=-1).item()
        tokens_std = list(prompt) + [next_t]

        curr = torch.tensor([[next_t]], device=device, dtype=torch.long)
        for _ in range(max_new_tokens - 1):
            with torch.no_grad():
                out, _, _ = model(curr, kv_caches=[cache_std] * config.n_layer)
            curr = out[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens_std.append(curr.item())

        # 2. Speculative prompt lookup decoding under Q4 with Metal verify_step
        decoder = SpeculativePromptLookupDecoder(ngram_size=2, max_spec_tokens=3)
        tokens_spec, stats = decoder.generate(
            model=model,
            prompt_tokens=prompt,
            max_new_tokens=max_new_tokens,
            device=device,
        )

        # 3. Assert mathematical exact parity
        self.assertEqual(len(tokens_std), len(tokens_spec))
        self.assertEqual(tokens_std, tokens_spec)
        self.assertGreater(stats["drafted_tokens"], 0)
        self.assertGreater(stats["accepted_draft_tokens"], 0)
        self.assertLess(stats["forward_calls"], max_new_tokens)

    def test_verify_step_numerical_parity(self):
        if not torch.backends.mps.is_available():
            return
        device = torch.device("mps")
        config = ModelConfig(
            vocab_size=512,
            block_size=256,
            n_layer=2,
            n_head=12,
            n_kv_head=4,
            n_embd=768,
            norm_type="rmsnorm",
            mlp_type="swiglu",
            pos_emb="rope",
            use_fused_kernels=True,
        )
        torch.manual_seed(42)
        model = Transformer(config).to(device=device, dtype=torch.float16).eval()

        prompt = [10, 20, 30, 40]
        prompt_t = torch.tensor([prompt], device=device, dtype=torch.long)

        for mode in [None, "q4", "q8"]:
            model.quantize(mode)
            c_run = StaticKVCache(1, 256, 2, 4, 64, device=device, dtype=torch.float16)
            c_step = StaticKVCache(1, 256, 2, 4, 64, device=device, dtype=torch.float16)
            with torch.no_grad():
                model(prompt_t, kv_caches=[c_run] * 2)
                model(prompt_t, kv_caches=[c_step] * 2)

            for T in [2, 3, 4]:
                cand = torch.randint(0, 512, (1, T), device=device, dtype=torch.long)
                k_saved = [k.clone() for k in c_step.k_caches]
                v_saved = [v.clone() for v in c_step.v_caches]
                seen_saved = c_step.total_seen_tokens[0]

                step_logits = []
                for t in range(T):
                    with torch.no_grad():
                        l_s, _, _ = model(cand[:, t : t + 1], kv_caches=[c_step] * 2)
                        step_logits.append(l_s.clone())
                expected = torch.cat(step_logits, dim=1)

                with torch.no_grad():
                    actual, _, _ = model(
                        cand, kv_caches=[c_run] * 2, return_all_logits=True
                    )

                diff = (actual - expected).abs().max().item()
                self.assertLess(
                    diff,
                    0.02,
                    f"Mode {mode} T={T} verify_step diff too high: {diff}",
                )

                for i in range(2):
                    c_step.k_caches[i].copy_(k_saved[i])
                    c_step.v_caches[i].copy_(v_saved[i])
                    c_run.k_caches[i].copy_(k_saved[i])
                    c_run.v_caches[i].copy_(v_saved[i])
                c_step.total_seen_tokens[:] = [seen_saved] * 2
                c_run.total_seen_tokens[:] = [seen_saved] * 2
                c_step.layer_seq_lens[:] = [seen_saved] * 2
                c_run.layer_seq_lens[:] = [seen_saved] * 2


if __name__ == "__main__":
    unittest.main()
