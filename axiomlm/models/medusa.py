"""
AxiomLM Neural Medusa Heads Architecture & Loss Functions.

Implements auxiliary multi-token prediction heads that attach to the final
hidden state of the Transformer to enable self-drafting speculative decoding
without relying on syntactic prompt repetition.
Reference: Cai et al., 2024 ("Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads").
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class MedusaHead(nn.Module):
    """
    A single auxiliary Medusa self-drafting prediction head.
    Predicts tokens at offset (t + k + 2) directly from token t's top-layer representation.
    """

    def __init__(
        self,
        n_embd: int,
        vocab_size: int,
        use_residual: bool = False,
    ):
        super().__init__()
        self.use_residual = use_residual
        if use_residual:
            self.res_block = nn.Sequential(
                nn.Linear(n_embd, n_embd, bias=False),
                nn.SiLU(),
            )
        else:
            self.res_block = None

        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Hidden state tensor of shape (B, T, n_embd) or (B, 1, n_embd)
        Returns:
            Logits tensor of shape (B, T, vocab_size)
        """
        if self.use_residual and self.res_block is not None:
            x = x + self.res_block(x)
        return self.head(x)


def compute_medusa_loss(
    medusa_logits: list[torch.Tensor],
    targets: torch.Tensor,
    loss_weights: list[float] | None = None,
    ignore_index: int = -1,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Computes cross-entropy loss for each Medusa head with decaying weights.

    Head k (0-indexed) predicts tokens at offset (t + k + 2).
    Therefore, medusa_logits[k][:, :-k-2] predicts targets[:, k+2:].

    Args:
        medusa_logits: List of K tensors, each of shape (B, T, vocab_size).
        targets: Target token IDs of shape (B, T).
        loss_weights: Decay factors for each head (default: 0.8^k).
        ignore_index: Target index to ignore in cross-entropy.

    Returns:
        Tuple of (total_medusa_loss, list_of_per_head_losses).
    """
    num_heads = len(medusa_logits)
    if num_heads == 0:
        return torch.tensor(0.0, device=targets.device), []

    if loss_weights is None:
        loss_weights = [0.8**k for k in range(num_heads)]

    total_loss = torch.tensor(0.0, device=targets.device, dtype=torch.float32)
    per_head_losses = []

    _B, T = targets.shape

    for k in range(num_heads):
        # Head k predicts token at offset (k + 2) relative to current position.
        # Shift: logits[:, : T - (k + 2)] align with targets[:, (k + 2) :]
        shift = k + 2
        if T <= shift:
            # Sequence too short for this head's prediction offset
            zero_loss = torch.tensor(0.0, device=targets.device)
            per_head_losses.append(zero_loss)
            continue

        logits_k = medusa_logits[k][:, : T - shift, :].contiguous()
        targets_k = targets[:, shift:].contiguous()

        loss_k = F.cross_entropy(
            logits_k.view(-1, logits_k.size(-1)),
            targets_k.view(-1),
            ignore_index=ignore_index,
        )
        per_head_losses.append(loss_k)
        total_loss = total_loss + (loss_weights[k] * loss_k)

    return total_loss, per_head_losses


def compute_medusa_distillation_loss(
    medusa_logits: list[torch.Tensor],
    base_logits: torch.Tensor,
    targets: torch.Tensor,
    loss_weights: list[float] | None = None,
    temperature: float = 2.0,
    distill_weight: float = 0.5,
    ignore_index: int = -1,
    top_k_distill: int | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """
    Computes Medusa-2 self-distillation loss combining hard cross-entropy
    and soft KL-divergence against frozen base model logits.

    Head k predicts token at offset (t + k + 2) from representation at step t.
    The base model predicts token at (t + k + 2) from representation at step (t + k + 1).
    Therefore:
      - medusa_logits[k][:, : T - (k + 2), :] aligns with base_logits[:, (k + 1) : T - 1, :]
      - and both predict targets[:, (k + 2) :]

    Args:
        medusa_logits: List of K tensors, each of shape (B, T, vocab_size).
        base_logits: Frozen base model logits of shape (B, T, vocab_size).
        targets: Ground truth target token IDs of shape (B, T).
        loss_weights: Weight decay factors per head (default: 0.8^k).
        temperature: Softmax distillation temperature tau.
        distill_weight: Interpolation lambda between CE and KL (0.0 = pure CE, 1.0 = pure KL).
        ignore_index: Index to ignore in cross-entropy.
        top_k_distill: If specified, restricts KL divergence to top-M base logits for efficiency.

    Returns:
        Tuple of (total_loss, per_head_ce_losses, per_head_kl_losses).
    """
    num_heads = len(medusa_logits)
    if num_heads == 0:
        zero = torch.tensor(0.0, device=targets.device)
        return zero, [], []

    if loss_weights is None:
        loss_weights = [0.8**k for k in range(num_heads)]

    total_loss = torch.tensor(0.0, device=targets.device, dtype=torch.float32)
    per_head_ce: list[torch.Tensor] = []
    per_head_kl: list[torch.Tensor] = []

    _B, T, V = base_logits.shape
    base_logits_detached = base_logits.detach()

    for k in range(num_heads):
        shift = k + 2
        if T <= shift:
            zero_loss = torch.tensor(0.0, device=targets.device)
            per_head_ce.append(zero_loss)
            per_head_kl.append(zero_loss)
            continue

        # Slices:
        # medusa slice: [:, : T - (k + 2)]
        # base slice:   [:, (k + 1) : T - 1]
        # target slice: [:, (k + 2) :]
        med_slice = medusa_logits[k][:, : T - shift, :].contiguous()
        base_slice = base_logits_detached[:, (k + 1) : T - 1, :].contiguous()
        target_slice = targets[:, shift:].contiguous()

        # 1. Hard Cross-Entropy Loss
        ce_loss_k = F.cross_entropy(
            med_slice.view(-1, V),
            target_slice.view(-1),
            ignore_index=ignore_index,
        )

        # 2. Soft KL-Divergence Distillation Loss in float32 for numerical stability on MPS
        med_f32 = med_slice.float()
        base_f32 = base_slice.float()
        if top_k_distill is not None and 0 < top_k_distill < V:
            top_idx = torch.topk(base_f32, k=top_k_distill, dim=-1).indices
            base_cand = torch.gather(base_f32, -1, top_idx)
            med_cand = torch.gather(med_f32, -1, top_idx)
            p_base = F.softmax(base_cand.view(-1, top_k_distill) / temperature, dim=-1)
            log_q_med = F.log_softmax(
                med_cand.view(-1, top_k_distill) / temperature, dim=-1
            )
            kl_loss_k = (temperature**2) * F.kl_div(
                log_q_med, p_base, reduction="batchmean"
            )
        else:
            p_base = F.softmax(base_f32.view(-1, V) / temperature, dim=-1)
            log_q_med = F.log_softmax(med_f32.view(-1, V) / temperature, dim=-1)
            kl_loss_k = (temperature**2) * F.kl_div(
                log_q_med, p_base, reduction="batchmean"
            )

        # 3. Interpolated Head Loss
        loss_k = (1.0 - distill_weight) * ce_loss_k + distill_weight * kl_loss_k

        per_head_ce.append(ce_loss_k)
        per_head_kl.append(kl_loss_k)
        total_loss = total_loss + (loss_weights[k] * loss_k)

    return total_loss, per_head_ce, per_head_kl


def evaluate_medusa_acceptance_matrix(
    medusa_logits: list[torch.Tensor],
    base_logits: torch.Tensor,
    max_topk: int = 4,
) -> dict[str, Any]:
    """
    Evaluates the empirical positional acceptance matrix P[head, rank] by comparing
    each Medusa head's top-k candidate predictions against the frozen base model's
    genuine greedy predictions.

    Args:
        medusa_logits: List of K tensors, each of shape (B, T, vocab_size).
        base_logits: Base model logits of shape (B, T, vocab_size).
        max_topk: Number of candidate ranks per head to evaluate (e.g. 4).

    Returns:
        Dict containing:
          - acceptance_matrix: np.ndarray of shape [K, max_topk] with empirical P[d, b]
          - top1_accuracies: list of float (top-1 acceptance per head)
          - total_tokens_evaluated: int
    """
    num_heads = len(medusa_logits)
    _B, T, _V = base_logits.shape
    base_preds = base_logits.argmax(dim=-1)  # (B, T)

    acc_matrix = np.zeros((num_heads, max_topk), dtype=np.float64)
    total_tokens_per_head = [0] * num_heads

    for k in range(num_heads):
        shift = k + 2
        if T <= shift:
            continue

        # Medusa head k at step t predicts target at t + k + 2.
        # The base model's greedy prediction for t + k + 2 is produced at step (t + k + 1).
        med_slice = medusa_logits[k][:, : T - shift, :]  # (B, T_eval, V)
        base_target_tokens = base_preds[:, (k + 1) : T - 1]  # (B, T_eval)

        T_eval = med_slice.size(1)
        if T_eval == 0:
            continue

        total_tokens_per_head[k] = _B * T_eval
        # Top-k predictions from Medusa head k
        topk_preds = torch.topk(
            med_slice, k=max_topk, dim=-1
        ).indices  # (B, T_eval, max_topk)

        for b in range(max_topk):
            rank_b_preds = topk_preds[:, :, b]
            matches = (rank_b_preds == base_target_tokens).sum().item()
            acc_matrix[k, b] = matches / (_B * T_eval)

    top1_accs = [float(acc_matrix[k, 0]) for k in range(num_heads)]

    return {
        "acceptance_matrix": acc_matrix,
        "top1_accuracies": top1_accs,
        "total_tokens_evaluated": (
            total_tokens_per_head[0] if total_tokens_per_head else 0
        ),
    }
