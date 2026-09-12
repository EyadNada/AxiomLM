import unittest
import torch
import numpy as np
import tempfile
import os

from axiomlm.models.transformer import Transformer, ModelConfig
from axiomlm.dengine.dataloader import DataLoaderLite
from axiomlm.engine.inference import InferenceEngine


class DummyTokenizer:
    def encode(self, text):
        return [0, 1, 2]

    def decode(self, tokens):
        return "mock"


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"

    def test_data_to_model_pipeline(self):
        """
        Integration Test: DataLoaderLite -> Transformer.
        Ensures batch fetching feeds perfectly into the model without shape mismatches.
        """
        B, T = 2, 32
        vocab_size = 256  # Tiny vocab for fast test

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a tiny dummy shard
            dummy_tokens = np.random.randint(0, vocab_size, size=1000, dtype=np.uint16)
            dummy_path = os.path.join(tmpdir, "train_0000.bin")
            dummy_tokens.tofile(dummy_path)

            # 1. Initialize DataLoader
            loader = DataLoaderLite(B=B, T=T, split="train", data_dir=tmpdir)
            x, y = loader.next_batch()

            self.assertEqual(x.shape, (B, T))
            self.assertEqual(y.shape, (B, T))

            # 2. Initialize Model
            config = ModelConfig(
                arch="modern",
                block_size=64,
                vocab_size=vocab_size,
                n_layer=2,
                n_head=2,
                n_embd=64,
            )
            model = Transformer(config).to(self.device)

            # 3. Forward Pass Integration
            logits, loss = model(x.to(self.device), targets=y.to(self.device))

            # Verify outputs
            self.assertEqual(logits.shape, (B, T, vocab_size))
            self.assertIsNotNone(loss)
            self.assertTrue(torch.is_tensor(loss))
            self.assertFalse(torch.isnan(loss).any())

    def test_inference_engine_kv_cache_integration(self):
        """
        Integration Test: InferenceEngine -> Transformer -> PagedKVCache.
        Ensures the engine correctly passes updated cache state between auto-regressive steps.
        """
        vocab_size = 256
        config = ModelConfig(
            arch="modern",
            block_size=64,
            vocab_size=vocab_size,
            n_layer=2,
            n_head=2,
            n_embd=64,
        )
        model = Transformer(config).to(self.device)
        tokenizer = DummyTokenizer()

        engine = InferenceEngine(model=model, device=self.device, tokenizer=tokenizer)

        # Intercept the Transformer forward pass to spy on the kv_caches kwarg
        original_forward = model.forward

        cache_steps = []

        def spy_forward(idx, targets=None, kv_caches=None):
            if kv_caches is not None and kv_caches[0] is not None:
                # Record the seq_len of the cache before this forward pass
                cache_steps.append(kv_caches[0].seq_len)
            return original_forward(idx, targets=targets, kv_caches=kv_caches)

        model.forward = spy_forward

        # Generate 5 tokens
        # Prompt gives 3 tokens ("mock"), then it should generate 5 more tokens
        # Prefill step evaluates 3 tokens, then generation step evaluating 1 token 5 times.
        # Cache lengths before model execution:
        # Step 1 (Prefill, 3 tokens): cache empty -> seq_len = 0
        # Step 2 (Decode token 1): cache has 3 tokens -> seq_len = 3
        # Step 3 (Decode token 2): cache has 4 tokens -> seq_len = 4
        # Step 4 (Decode token 3): cache has 5 tokens -> seq_len = 5
        # Step 5 (Decode token 4): cache has 6 tokens -> seq_len = 6
        # Step 6 (Decode token 5): cache has 7 tokens -> seq_len = 7

        engine.generate("Hello", max_tokens=5)

        # Assert the cache seq_len strictly increased across decode steps
        self.assertEqual(len(cache_steps), 5)  # 1 prefill + 5 decodes
        self.assertEqual(cache_steps, [0, 3, 4, 5, 6])


if __name__ == "__main__":
    unittest.main()
