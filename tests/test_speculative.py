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


if __name__ == "__main__":
    unittest.main()
