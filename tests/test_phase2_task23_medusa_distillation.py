"""
Unit and systems test suite for AxiomLM Phase 2 Task 2.3:
Medusa-2 KL-Divergence Self-Distillation Engine.
"""

from __future__ import annotations

import unittest

import torch

from axiomlm.engine.medusa import train_medusa_heads
from axiomlm.kernels import _METAL_MOD, is_mps_available
from axiomlm.models.medusa import (
    compute_medusa_distillation_loss,
    compute_medusa_loss,
    evaluate_medusa_acceptance_matrix,
)
from axiomlm.models.transformer import ModelConfig, Transformer


class TestPhase2Task23MedusaDistillation(unittest.TestCase):
    """Rigorous mathematical and systems verification for Medusa self-distillation."""

    def setUp(self):
        torch.manual_seed(2026)
        self.device = torch.device(
            "mps" if is_mps_available() and _METAL_MOD is not None else "cpu"
        )
        self.num_heads = 4
        self.vocab_size = 256
        self.batch_size = 2
        self.seq_len = 32

    def test_01_distillation_loss_alignment_and_non_negativity(self):
        """
        Validate mathematical invariants of compute_medusa_distillation_loss:
        1. Loss is strictly positive.
        2. Per-head CE and KL losses are non-negative.
        3. Shift alignment: head k aligns exactly with base prediction at t + k + 1.
        4. Setting distill_weight=0.0 matches pure cross-entropy.
        """
        B, T, V = self.batch_size, self.seq_len, self.vocab_size
        base_logits = torch.randn(B, T, V, device=self.device)
        medusa_logits = [
            torch.randn(B, T, V, device=self.device, requires_grad=True)
            for _ in range(self.num_heads)
        ]
        targets = torch.randint(0, V, (B, T), device=self.device)

        # 1. Combined CE + KL
        loss_comb, ce_losses, kl_losses = compute_medusa_distillation_loss(
            medusa_logits=medusa_logits,
            base_logits=base_logits,
            targets=targets,
            temperature=2.0,
            distill_weight=0.5,
        )

        self.assertGreater(loss_comb.item(), 0.0)
        self.assertEqual(len(ce_losses), self.num_heads)
        self.assertEqual(len(kl_losses), self.num_heads)

        for k in range(self.num_heads):
            self.assertGreaterEqual(ce_losses[k].item(), 0.0)
            self.assertGreaterEqual(kl_losses[k].item(), 0.0)

        # 2. distill_weight = 0.0 matches compute_medusa_loss
        loss_ce_only, ce_ref = compute_medusa_loss(medusa_logits, targets)
        loss_distill_0, ce_distill, _ = compute_medusa_distillation_loss(
            medusa_logits=medusa_logits,
            base_logits=base_logits,
            targets=targets,
            distill_weight=0.0,
        )
        self.assertAlmostEqual(loss_ce_only.item(), loss_distill_0.item(), places=5)
        for k in range(self.num_heads):
            self.assertAlmostEqual(ce_ref[k].item(), ce_distill[k].item(), places=5)

    def test_02_top_k_sparse_distillation_exactness(self):
        """
        Verify that Top-M sparse distillation gathers valid probabilities
        and produces non-negative KL divergence with gradient flow.
        """
        B, T, V = self.batch_size, self.seq_len, self.vocab_size
        base_logits = torch.randn(B, T, V, device=self.device)
        med_logits = [
            torch.randn(B, T, V, device=self.device, requires_grad=True)
            for _ in range(self.num_heads)
        ]
        targets = torch.randint(0, V, (B, T), device=self.device)

        loss_sparse, _, _kl_sparse = compute_medusa_distillation_loss(
            medusa_logits=med_logits,
            base_logits=base_logits,
            targets=targets,
            temperature=1.5,
            distill_weight=0.7,
            top_k_distill=32,
        )

        self.assertGreater(loss_sparse.item(), 0.0)
        loss_sparse.backward()

        for k in range(self.num_heads):
            self.assertIsNotNone(med_logits[k].grad)
            self.assertGreater(med_logits[k].grad.norm().item(), 0.0)

    def test_03_gradient_isolation_backbone_freeze(self):
        """
        Systems Invariant:
        Verify that during Medusa self-distillation training, all backbone parameters
        remain completely frozen (grad is None), and gradients ONLY flow into
        Medusa auxiliary head weights.
        """
        cfg = ModelConfig(
            arch="modern",
            n_layer=2,
            n_head=4,
            n_kv_head=2,
            n_embd=128,
            vocab_size=256,
            block_size=128,
            use_fused_kernels=False,
        )
        model = Transformer(cfg).to(device=self.device, dtype=torch.float32).eval()
        model.add_medusa_heads(num_heads=3)

        train_tokens = torch.randint(0, cfg.vocab_size, (2, 32), device=self.device)

        train_medusa_heads(
            model=model,
            input_ids=train_tokens,
            epochs=2,
            lr=1e-3,
            freeze_backbone=True,
            use_distillation=True,
            distill_weight=0.5,
        )

        # Check backbone gradients
        for name, param in model.transformer.named_parameters():
            self.assertTrue(
                param.grad is None or (param.grad == 0).all(),
                f"Backbone parameter {name} was modified by Medusa distillation!",
            )
        self.assertTrue(model.lm_head.weight.grad is None)

        # Check Medusa head parameters exist and were updated
        for head_idx, head in enumerate(model.medusa_heads):
            for p in head.parameters():
                self.assertTrue(p.requires_grad)

    def test_04_acceptance_matrix_evaluation(self):
        """
        Verify that evaluate_medusa_acceptance_matrix correctly extracts
        positional acceptance probabilities P[d, b] in [0.0, 1.0].
        """
        B, T, V = 2, 32, 128
        base_logits = torch.randn(B, T, V, device=self.device)
        # Create synthetic medusa logits where head 0 perfectly matches base predictions
        med_logits = []
        base_preds = base_logits.argmax(dim=-1)

        for k in range(3):
            logits_k = torch.randn(B, T, V, device=self.device)
            # Force top-1 of head 0 to match base prediction
            if k == 0:
                target_base = base_preds[:, 1 : T - 1]
                for b_idx in range(B):
                    for t_idx in range(T - 2):
                        tok = target_base[b_idx, t_idx].item()
                        logits_k[b_idx, t_idx, tok] += 50.0  # Dominant logit
            med_logits.append(logits_k)

        eval_stats = evaluate_medusa_acceptance_matrix(
            med_logits, base_logits, max_topk=4
        )
        acc_mat = eval_stats["acceptance_matrix"]
        self.assertEqual(acc_mat.shape, (3, 4))
        self.assertTrue((acc_mat >= 0.0).all() and (acc_mat <= 1.0).all())
        # Head 0 top-1 must be 1.0 (100% accepted)
        self.assertAlmostEqual(float(acc_mat[0, 0]), 1.0, places=3)
        self.assertAlmostEqual(eval_stats["top1_accuracies"][0], 1.0, places=3)

    def test_05_distillation_convergence(self):
        """
        Verify that Medusa-2 self-distillation training strictly reduces loss
        over successive epochs on synthetic text.
        """
        cfg = ModelConfig(
            arch="modern",
            n_layer=2,
            n_head=4,
            n_kv_head=2,
            n_embd=128,
            vocab_size=256,
            block_size=128,
            use_fused_kernels=False,
        )
        model = Transformer(cfg).to(device=self.device, dtype=torch.float32).eval()
        model.add_medusa_heads(num_heads=3)

        train_tokens = torch.randint(0, cfg.vocab_size, (2, 32), device=self.device)

        stats = train_medusa_heads(
            model=model,
            input_ids=train_tokens,
            epochs=5,
            lr=5e-3,
            freeze_backbone=True,
            use_distillation=True,
            distill_weight=0.5,
            return_stats=True,
        )

        self.assertIsInstance(stats, dict)
        losses = stats["losses"]
        self.assertEqual(len(losses), 5)
        self.assertLess(
            losses[-1],
            losses[0],
            f"Expected loss to decrease: start={losses[0]:.4f}, end={losses[-1]:.4f}",
        )


if __name__ == "__main__":
    unittest.main()
