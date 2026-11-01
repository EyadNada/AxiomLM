"""
AxiomLM Neural Medusa Speculative Decoding Engine.

Provides self-drafting multi-token speculative decoding powered by auxiliary
prediction heads on Apple Silicon Metal. Completely decouples speculative speedups
from prompt repetition, enabling high acceptance rates on arbitrary non-repetitive text
with exact token-for-token mathematical parity to greedy decoding.
"""

import time
from typing import Any

import torch

from ..models.medusa import compute_medusa_loss
from ..models.transformer import Transformer
from .paged_cache import StaticKVCache


class MedusaSpeculativeDecoder:
    """
    Neural Medusa Speculative Decoder for AxiomLM Transformers.

    Utilizes K auxiliary linear heads attached to the final hidden state
    to self-draft K candidate tokens in parallel, then verifies them in a single
    batched verification pass (qmm_q4_g64 / verify_step).
    """

    def __init__(
        self,
        num_heads: int = 3,
        temperature: float = 0.0,
    ):
        self.num_heads = num_heads
        self.temperature = temperature

    def generate(
        self,
        model: Transformer,
        prompt_tokens: list[int],
        max_new_tokens: int = 64,
        cache: StaticKVCache | None = None,
        device: torch.device | None = None,
    ) -> tuple[list[int], dict[str, Any]]:
        """
        Executes Medusa neural speculative decoding.

        Args:
            model: AxiomLM Transformer model with medusa_heads attached.
            prompt_tokens: List of initial prompt token IDs.
            max_new_tokens: Maximum number of tokens to generate.
            cache: Pre-allocated StaticKVCache (created if None).
            device: Target torch device (defaults to model device).

        Returns:
            Tuple of (generated_tokens, stats_dict).
        """
        if model.medusa_heads is None or len(model.medusa_heads) == 0:
            raise ValueError(
                "Model does not have Medusa heads attached. Call model.add_medusa_heads() before decoding."
            )

        if device is None:
            device = next(model.parameters()).device

        config = model.config
        head_dim = config.n_embd // config.n_head
        n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        max_cache_len = max(config.block_size, len(prompt_tokens) + max_new_tokens + 16)

        if cache is None:
            cache = StaticKVCache(
                max_batch_size=1,
                max_seq_len=max_cache_len,
                n_layer=config.n_layer,
                n_kv_head=n_kv_head,
                head_dim=head_dim,
                device=device,
                dtype=next(model.parameters()).dtype,
            )

        n_L = config.n_layer
        kv_caches = [cache] * n_L
        k_heads = len(model.medusa_heads)

        # 1. Prefill stage: Process initial prompt and generate first draft
        prompt_tensor = torch.tensor([prompt_tokens], device=device, dtype=torch.long)
        with torch.no_grad():
            base_logits, medusa_logits, _ = model.forward_with_medusa(
                prompt_tensor, kv_caches=kv_caches, return_all_logits=False
            )

        # Base model's genuine next token
        next_tok = base_logits[:, -1, :].argmax(dim=-1).item()
        generated = list(prompt_tokens) + [next_tok]
        target_total = len(prompt_tokens) + max_new_tokens

        # Initial Medusa draft candidates from the prompt's final hidden state
        draft_candidates = [
            medusa_logits[k][:, -1, :].argmax(dim=-1).item() for k in range(k_heads)
        ]

        total_drafted = 0
        total_accepted = 0
        model_forward_calls = 1

        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()

        # 2. Medusa Speculative Generation Loop
        with torch.no_grad():
            while len(generated) < target_total:
                # Number of candidates to verify in this step
                k_cand = min(len(draft_candidates), target_total - len(generated))
                if k_cand <= 0:
                    break

                curr_cand = draft_candidates[:k_cand]
                # Candidate verification sequence: current committed token + candidate draft tokens
                verif_seq = [generated[-1]] + curr_cand

                verif_tensor = torch.tensor(
                    [verif_seq], device=device, dtype=torch.long
                )
                checkpoint_seen = cache.total_seen_tokens[0]

                # Single-pass batched forward verification (computes all blocks, LN, lm_head & medusa_heads)
                logits, med_logits_all, _ = model.forward_with_medusa(
                    verif_tensor, kv_caches=kv_caches, return_all_logits=True
                )
                model_forward_calls += 1
                total_drafted += k_cand

                # Model's genuine greedy predictions at each position in verif_seq
                preds = logits[0].argmax(dim=-1).tolist()
                accepted = 0
                all_matched = True
                last_accepted_idx = 0

                for j in range(k_cand):
                    pred_tok = preds[j]
                    if pred_tok == curr_cand[j]:
                        # Candidate matched!
                        generated.append(pred_tok)
                        accepted += 1
                        last_accepted_idx = j
                        if len(generated) >= target_total:
                            all_matched = False
                            break
                    else:
                        # Mismatch: accept model's genuine prediction and reject remaining candidates
                        generated.append(pred_tok)
                        accepted += 1
                        last_accepted_idx = j
                        all_matched = False
                        break

                if all_matched and len(generated) < target_total:
                    # All K candidates matched perfectly! Accept bonus token from last verified position
                    bonus_tok = preds[k_cand]
                    generated.append(bonus_tok)
                    accepted += 1
                    last_accepted_idx = k_cand - 1

                # Credit accepted draft tokens (excluding the 1 base guaranteed token)
                total_accepted += max(0, accepted - 1)

                # O(1) KV-Cache rollback: advance only the accepted token count
                active = min(checkpoint_seen + accepted, cache.max_seq_len)
                cache.total_seen_tokens[:] = [checkpoint_seen + accepted] * n_L
                cache.layer_seq_lens[:] = [active] * n_L

                if len(generated) >= target_total:
                    break

                # Extract new Medusa draft candidates from the last accepted position's hidden state
                # med_logits_all[k] has shape (1, T_verif, vocab_size)
                if med_logits_all and last_accepted_idx < med_logits_all[0].size(1):
                    draft_candidates = [
                        med_logits_all[k][0, last_accepted_idx].argmax(dim=-1).item()
                        for k in range(k_heads)
                    ]
                else:
                    # Fallback single step if index boundary exceeded
                    curr_single = torch.tensor(
                        [[generated[-1]]], device=device, dtype=torch.long
                    )
                    _, fallback_med_logits, _ = model.forward_with_medusa(
                        curr_single, kv_caches=kv_caches, return_all_logits=False
                    )
                    model_forward_calls += 1
                    draft_candidates = [
                        fallback_med_logits[k][:, -1, :].argmax(dim=-1).item()
                        for k in range(k_heads)
                    ]

        if device.type == "mps":
            torch.mps.synchronize()
        elapsed_sec = time.perf_counter() - t0

        new_tokens_count = len(generated) - len(prompt_tokens)
        tok_sec = new_tokens_count / elapsed_sec if elapsed_sec > 0 else 0.0
        acc_rate = (
            (total_accepted / total_drafted * 100.0) if total_drafted > 0 else 0.0
        )

        stats = {
            "new_tokens": new_tokens_count,
            "elapsed_seconds": elapsed_sec,
            "tokens_per_second": tok_sec,
            "ms_per_token": (
                (elapsed_sec / new_tokens_count * 1000.0)
                if new_tokens_count > 0
                else 0.0
            ),
            "forward_calls": model_forward_calls,
            "tokens_per_forward_call": (
                new_tokens_count / model_forward_calls
                if model_forward_calls > 0
                else 0.0
            ),
            "drafted_tokens": total_drafted,
            "accepted_draft_tokens": total_accepted,
            "acceptance_rate_pct": acc_rate,
        }

        return generated, stats


def train_medusa_heads(
    model: Transformer,
    input_ids: torch.Tensor,
    targets: torch.Tensor | None = None,
    epochs: int = 1,
    lr: float = 1e-3,
    loss_weights: list[float] | None = None,
    freeze_backbone: bool = True,
) -> list[float]:
    """
    Lightweight training loop for Medusa auxiliary heads while keeping the Transformer backbone frozen.

    Args:
        model: Transformer model with medusa_heads attached.
        input_ids: Input tokens of shape (B, T).
        targets: Target tokens of shape (B, T) (defaults to input_ids shifted by 1).
        epochs: Number of optimization epochs.
        lr: Learning rate for Medusa heads.
        loss_weights: Decay factors for head losses (default: 0.8^k).
        freeze_backbone: If True, freezes all backbone weights and only updates Medusa heads.

    Returns:
        List of loss values per epoch/step.
    """
    if model.medusa_heads is None:
        raise ValueError("Model does not have Medusa heads attached.")

    if targets is None:
        targets = input_ids.clone()

    if freeze_backbone:
        for p in model.transformer.parameters():
            p.requires_grad = False
        model.lm_head.weight.requires_grad = False
        for head in model.medusa_heads:
            for p in head.parameters():
                p.requires_grad = True

    # Optimize Medusa heads in float32 for numerical stability on MPS / half-precision runners
    orig_dtype = next(model.medusa_heads.parameters()).dtype
    model.medusa_heads.to(dtype=torch.float32)

    optimizer = torch.optim.AdamW(model.medusa_heads.parameters(), lr=lr)
    losses = []

    model.train()
    for ep in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        _, medusa_logits, _ = model.forward_with_medusa(input_ids)
        loss, _ = compute_medusa_loss(medusa_logits, targets, loss_weights=loss_weights)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    model.medusa_heads.to(dtype=orig_dtype)
    model.eval()
    return losses
