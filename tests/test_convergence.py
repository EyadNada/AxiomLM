import unittest
import torch
from axiomlm import GPT, GPTConfig


class TestModelConvergence(unittest.TestCase):
    def test_overfit_single_batch(self):
        """
        Acceptance Test: Mathematically prove the model can overfit a single batch.
        This guarantees that gradients, loss calculation, and optimizer steps are correctly wired.
        """
        torch.manual_seed(42)

        cfg = GPTConfig(
            vocab_size=100,
            block_size=16,
            n_layer=2,
            n_head=2,
            n_embd=32,
            n_kv_head=2,
            norm_type="rmsnorm",
            pos_emb="rope",
            mlp_type="swiglu",
        )
        model = GPT(cfg)
        model.train()

        # Simple AdamW optimizer for the test
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

        # Deterministic batch of data (B=2, T=16)
        x = torch.randint(0, 100, (2, 16))
        y = torch.roll(x, shifts=-1, dims=1)  # Target is next token

        initial_loss = None
        final_loss = None

        for step in range(100):
            optimizer.zero_grad(set_to_none=True)
            logits, loss = model(x, targets=y)
            loss.backward()
            optimizer.step()

            if step == 0:
                initial_loss = loss.item()
            if step == 99:
                final_loss = loss.item()

        self.assertIsNotNone(initial_loss)
        self.assertIsNotNone(final_loss)
        self.assertLess(
            final_loss,
            initial_loss,
            f"Loss did not decrease! {initial_loss} -> {final_loss}",
        )
        self.assertLess(
            final_loss, 0.5, f"Model failed to overfit! Final loss: {final_loss}"
        )


if __name__ == "__main__":
    unittest.main()
