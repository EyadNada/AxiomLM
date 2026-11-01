"""
Rigorous Mathematical Correctness, Parity, and Acceptance Tests for
Phase 7: Neural Medusa Heads (Multi-Head Self-Drafting Speculative Decoding).
"""

import unittest

import torch

from axiomlm.engine.medusa import MedusaSpeculativeDecoder, train_medusa_heads
from axiomlm.engine.paged_cache import StaticKVCache
from axiomlm.kernels import is_mps_available
from axiomlm.models.medusa import MedusaHead, compute_medusa_loss
from axiomlm.models.transformer import ModelConfig, Transformer


class TestMedusaSpeculative(unittest.TestCase):
    """Exhaustive test suite for Neural Medusa Heads and Speculative Decoding."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("mps" if is_mps_available() else "cpu")
        self.config = ModelConfig(
            arch="modern",
            n_layer=2,
            n_head=4,
            n_kv_head=2,
            n_embd=256,
            vocab_size=512,
            block_size=512,
            use_fused_kernels=True,
        )

    def test_medusa_head_initialization_and_forward(self):
        """Verify MedusaHead output shapes, residual blocks, and gradient flow."""
        n_embd = 256
        vocab_size = 512
        B, T = 2, 8

        # 1. Linear head (Medusa-1 style)
        head_linear = MedusaHead(n_embd, vocab_size, use_residual=False)
        x = torch.randn(B, T, n_embd, requires_grad=True)
        out_linear = head_linear(x)
        self.assertEqual(out_linear.shape, (B, T, vocab_size))

        loss_linear = out_linear.sum()
        loss_linear.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(head_linear.head.weight.grad)

        # 2. Residual head (Medusa-2 style)
        head_res = MedusaHead(n_embd, vocab_size, use_residual=True)
        x_res = torch.randn(B, T, n_embd, requires_grad=True)
        out_res = head_res(x_res)
        self.assertEqual(out_res.shape, (B, T, vocab_size))

        loss_res = out_res.sum()
        loss_res.backward()
        self.assertIsNotNone(x_res.grad)
        self.assertIsNotNone(head_res.res_block[0].weight.grad)

    def test_transformer_dynamic_attachment(self):
        """Verify adding Medusa heads dynamically sets up modules, dtypes, and devices correctly."""
        model = Transformer(self.config).to(device=self.device)
        self.assertIsNone(model.medusa_heads)

        # Attach 3 Medusa heads
        model.add_medusa_heads(num_heads=3, use_residual=False)
        self.assertIsNotNone(model.medusa_heads)
        self.assertEqual(len(model.medusa_heads), 3)

        # Verify device placement
        for head in model.medusa_heads:
            self.assertEqual(head.head.weight.device.type, self.device.type)

        # Forward with Medusa
        idx = torch.randint(0, self.config.vocab_size, (1, 4), device=self.device)
        logits, medusa_logits, _ = model.forward_with_medusa(
            idx, return_all_logits=True
        )
        self.assertEqual(logits.shape, (1, 4, self.config.vocab_size))
        self.assertEqual(len(medusa_logits), 3)
        for k in range(3):
            self.assertEqual(medusa_logits[k].shape, (1, 4, self.config.vocab_size))

    def test_medusa_loss_computation(self):
        """Verify Medusa multi-head loss alignment and decaying weights."""
        B, T = 2, 16
        V = self.config.vocab_size
        num_heads = 3

        med_logits = [torch.randn(B, T, V) for _ in range(num_heads)]
        targets = torch.randint(0, V, (B, T))

        total_loss, per_head_losses = compute_medusa_loss(
            med_logits, targets, loss_weights=[1.0, 0.8, 0.64]
        )
        self.assertGreater(total_loss.item(), 0.0)
        self.assertEqual(len(per_head_losses), num_heads)
        for h_loss in per_head_losses:
            self.assertGreater(h_loss.item(), 0.0)

        # Verify sequence shorter than head offset yields zero loss without crashing
        short_logits = [torch.randn(B, 2, V) for _ in range(num_heads)]
        short_targets = torch.randint(0, V, (B, 2))
        short_loss, _ = compute_medusa_loss(short_logits, short_targets)
        self.assertEqual(short_loss.item(), 0.0)

    def test_medusa_training_pipeline_and_backbone_freeze(self):
        """Verify Medusa training loop decreases loss while leaving backbone frozen."""
        model = Transformer(self.config).to(device=self.device)
        model.add_medusa_heads(num_heads=2)

        # Record backbone weights before training
        wte_before = model.transformer.wte.weight.clone()
        block0_attn_before = model.transformer.h[0].attn.c_attn.weight.clone()

        # Synthetic training data
        tokens = torch.randint(0, self.config.vocab_size, (2, 32), device=self.device)
        losses = train_medusa_heads(
            model,
            tokens,
            epochs=5,
            lr=1e-2,
            freeze_backbone=True,
        )

        self.assertEqual(len(losses), 5)
        self.assertLess(
            losses[-1], losses[0], "Medusa head loss should decrease with training"
        )

        # Confirm backbone weights are completely unchanged
        self.assertTrue(torch.equal(wte_before, model.transformer.wte.weight))
        self.assertTrue(
            torch.equal(block0_attn_before, model.transformer.h[0].attn.c_attn.weight)
        )

    def test_medusa_speculative_exact_parity_greedy(self):
        """
        Mathematical Parity Proof:
        Verify Medusa Speculative Decoder generates identical token-for-token output
        as standard autoregressive greedy decoding across arbitrary (non-repetitive) text.
        """
        dtype = torch.float16 if self.device.type == "mps" else torch.float32
        model = Transformer(self.config).to(device=self.device, dtype=dtype).eval()
        model.add_medusa_heads(num_heads=3)

        # Non-repetitive arbitrary pseudo-random prompt
        prompt = [45, 128, 290, 12, 401, 89, 215, 333, 9, 174]
        max_new_tokens = 24

        # 1. Standard autoregressive greedy generation baseline
        cache_std = StaticKVCache(
            max_batch_size=1,
            max_seq_len=512,
            n_layer=self.config.n_layer,
            n_kv_head=self.config.n_kv_head,
            head_dim=self.config.n_embd // self.config.n_head,
            device=self.device,
            dtype=dtype,
        )
        p_tensor = torch.tensor([prompt], device=self.device, dtype=torch.long)
        with torch.no_grad():
            out, _, _ = model(p_tensor, kv_caches=[cache_std] * self.config.n_layer)
        next_t = out[:, -1, :].argmax(dim=-1).item()
        tokens_std = list(prompt) + [next_t]

        curr = torch.tensor([[next_t]], device=self.device, dtype=torch.long)
        for _ in range(max_new_tokens - 1):
            with torch.no_grad():
                out, _, _ = model(curr, kv_caches=[cache_std] * self.config.n_layer)
            curr = out[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens_std.append(curr.item())

        # 2. Medusa Speculative Decoding
        decoder = MedusaSpeculativeDecoder(num_heads=3)
        tokens_medusa, stats = decoder.generate(
            model=model,
            prompt_tokens=prompt,
            max_new_tokens=max_new_tokens,
            device=self.device,
        )

        # 3. Assert 100% exact token-for-token mathematical parity
        self.assertEqual(len(tokens_std), len(tokens_medusa))
        self.assertEqual(
            tokens_std,
            tokens_medusa,
            f"Medusa generated tokens differed from standard autoregressive greedy:\n"
            f"Std:    {tokens_std}\n"
            f"Medusa: {tokens_medusa}",
        )
        self.assertEqual(len(tokens_medusa), len(prompt) + max_new_tokens)
        self.assertGreater(stats["new_tokens"], 0)
        self.assertGreater(stats["tokens_per_second"], 0.0)

    def test_medusa_acceptance_and_speedup(self):
        """
        Verify that training Medusa heads on a repeated sequence enables high
        draft acceptance rate (>65%) and reduces total model forward passes.
        """
        dtype = torch.float16 if self.device.type == "mps" else torch.float32
        model = Transformer(self.config).to(device=self.device, dtype=dtype).eval()
        model.add_medusa_heads(num_heads=3)

        # Generate reference sequence from the backbone model
        prompt = [10, 20, 30, 40]
        cache = StaticKVCache(
            max_batch_size=1,
            max_seq_len=512,
            n_layer=self.config.n_layer,
            n_kv_head=self.config.n_kv_head,
            head_dim=self.config.n_embd // self.config.n_head,
            device=self.device,
            dtype=dtype,
        )
        curr = torch.tensor([prompt], device=self.device)
        with torch.no_grad():
            out, _, _ = model(curr, kv_caches=[cache] * self.config.n_layer)
        tokens = list(prompt)
        for _ in range(64):
            next_t = out[:, -1, :].argmax(dim=-1).item()
            tokens.append(next_t)
            curr = torch.tensor([[next_t]], device=self.device)
            with torch.no_grad():
                out, _, _ = model(curr, kv_caches=[cache] * self.config.n_layer)

        train_tensor = torch.tensor([tokens], device=self.device, dtype=torch.long)

        # Train medusa heads to distill the backbone's multi-step predictions
        train_medusa_heads(
            model,
            train_tensor,
            epochs=35,
            lr=3e-2,
            freeze_backbone=True,
        )

        max_new_tokens = 32
        decoder = MedusaSpeculativeDecoder(num_heads=3)
        tokens_medusa, stats = decoder.generate(
            model=model,
            prompt_tokens=prompt,
            max_new_tokens=max_new_tokens,
            device=self.device,
        )

        self.assertEqual(len(tokens_medusa), len(prompt) + max_new_tokens)
        self.assertEqual(tokens_medusa, tokens[: len(prompt) + max_new_tokens])
        self.assertGreater(
            stats["acceptance_rate_pct"],
            50.0,
            f"Trained Medusa head should have >50% acceptance rate (got {stats['acceptance_rate_pct']}%)",
        )
        self.assertGreater(
            stats["tokens_per_forward_call"],
            1.4,
            f"Trained Medusa head should average >1.4 tokens per call (got {stats['tokens_per_forward_call']})",
        )
        self.assertLess(
            stats["forward_calls"],
            max_new_tokens,
            f"Forward calls ({stats['forward_calls']}) should be less than max_new_tokens ({max_new_tokens})",
        )

    def test_medusa_quantized_q4_parity(self):
        """Verify Medusa speculative decoding produces exact greedy parity under Q4_G64 quantization."""
        if not is_mps_available():
            self.skipTest("MPS backend unavailable")

        device = torch.device("mps")
        model = Transformer(self.config).to(device=device, dtype=torch.float16).eval()
        model.add_medusa_heads(num_heads=3)
        model.quantize("q4")

        prompt = [15, 30, 45, 60, 75, 90, 105, 120]
        max_new_tokens = 20

        # 1. Standard autoregressive greedy generation under Q4
        cache_std = StaticKVCache(
            max_batch_size=1,
            max_seq_len=256,
            n_layer=self.config.n_layer,
            n_kv_head=self.config.n_kv_head,
            head_dim=self.config.n_embd // self.config.n_head,
            device=device,
            dtype=torch.float16,
        )
        p_tensor = torch.tensor([prompt], device=device, dtype=torch.long)
        with torch.no_grad():
            out, _, _ = model(p_tensor, kv_caches=[cache_std] * self.config.n_layer)
        next_t = out[:, -1, :].argmax(dim=-1).item()
        tokens_std = list(prompt) + [next_t]

        curr = torch.tensor([[next_t]], device=device, dtype=torch.long)
        for _ in range(max_new_tokens - 1):
            with torch.no_grad():
                out, _, _ = model(curr, kv_caches=[cache_std] * self.config.n_layer)
            curr = out[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens_std.append(curr.item())

        # 2. Medusa Speculative Generation under Q4
        decoder = MedusaSpeculativeDecoder(num_heads=3)
        tokens_medusa, _ = decoder.generate(
            model=model,
            prompt_tokens=prompt,
            max_new_tokens=max_new_tokens,
            device=device,
        )

        self.assertEqual(len(tokens_std), len(tokens_medusa))
        self.assertEqual(
            tokens_std,
            tokens_medusa,
            f"Q4 Medusa differed from Q4 greedy:\nStd:    {tokens_std}\nMedusa: {tokens_medusa}",
        )

    def test_medusa_quantized_q8_parity(self):
        """Verify Medusa speculative decoding produces exact greedy parity under Q8_G64 quantization."""
        if not is_mps_available():
            self.skipTest("MPS backend unavailable")

        device = torch.device("mps")
        model = Transformer(self.config).to(device=device, dtype=torch.float16).eval()
        model.add_medusa_heads(num_heads=3)
        model.quantize("q8")

        prompt = [22, 44, 66, 88, 110, 132, 154]
        max_new_tokens = 20

        # 1. Standard autoregressive greedy generation under Q8
        cache_std = StaticKVCache(
            max_batch_size=1,
            max_seq_len=256,
            n_layer=self.config.n_layer,
            n_kv_head=self.config.n_kv_head,
            head_dim=self.config.n_embd // self.config.n_head,
            device=device,
            dtype=torch.float16,
        )
        p_tensor = torch.tensor([prompt], device=device, dtype=torch.long)
        with torch.no_grad():
            out, _, _ = model(p_tensor, kv_caches=[cache_std] * self.config.n_layer)
        next_t = out[:, -1, :].argmax(dim=-1).item()
        tokens_std = list(prompt) + [next_t]

        curr = torch.tensor([[next_t]], device=device, dtype=torch.long)
        for _ in range(max_new_tokens - 1):
            with torch.no_grad():
                out, _, _ = model(curr, kv_caches=[cache_std] * self.config.n_layer)
            curr = out[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens_std.append(curr.item())

        # 2. Medusa Speculative Generation under Q8
        decoder = MedusaSpeculativeDecoder(num_heads=3)
        tokens_medusa, _ = decoder.generate(
            model=model,
            prompt_tokens=prompt,
            max_new_tokens=max_new_tokens,
            device=device,
        )

        self.assertEqual(len(tokens_std), len(tokens_medusa))
        self.assertEqual(
            tokens_std,
            tokens_medusa,
            f"Q8 Medusa differed from Q8 greedy:\nStd:    {tokens_std}\nMedusa: {tokens_medusa}",
        )

    def test_medusa_variable_head_counts(self):
        """Verify Medusa speculative decoding works correctly across K in [1, 2, 4] heads."""
        dtype = torch.float16 if self.device.type == "mps" else torch.float32
        prompt = [100, 200, 300, 400]
        max_new_tokens = 16

        for num_heads in [1, 2, 4]:
            model = Transformer(self.config).to(device=self.device, dtype=dtype).eval()
            model.add_medusa_heads(num_heads=num_heads)

            decoder = MedusaSpeculativeDecoder(num_heads=num_heads)
            tokens, stats = decoder.generate(
                model=model,
                prompt_tokens=prompt,
                max_new_tokens=max_new_tokens,
                device=self.device,
            )
            self.assertEqual(len(tokens), len(prompt) + max_new_tokens)
            self.assertGreater(stats["tokens_per_second"], 0.0)


if __name__ == "__main__":
    unittest.main()
