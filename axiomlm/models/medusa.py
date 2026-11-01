"""
AxiomLM Neural Medusa Heads Architecture & Loss Functions.

Implements auxiliary multi-token prediction heads that attach to the final
hidden state of the Transformer to enable self-drafting speculative decoding
without relying on syntactic prompt repetition.
Reference: Cai et al., 2024 ("Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads").
"""

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
