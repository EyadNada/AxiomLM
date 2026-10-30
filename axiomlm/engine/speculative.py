"""
AxiomLM Speculative Prompt-Lookup Decoding Engine.

Provides zero-parameter speculative decoding via prompt n-gram matching (Prompt Lookup Decoding).
Enables multi-token verification per forward pass on Apple Silicon Metal with mathematical parity
to standard autoregressive greedy decoding.
"""

import time
from typing import List, Optional, Tuple, Dict, Any
import torch
from ..models.transformer import Transformer
from .paged_cache import StaticKVCache


def find_candidate_ngram(
    input_ids: List[int],
    ngram_size: int = 3,
    max_spec_tokens: int = 4,
) -> List[int]:
    """
    Finds candidate draft tokens from past sequence context via N-Gram matching.

    Args:
        input_ids: List of integer token IDs generated so far.
        ngram_size: Context matching window length (N).
        max_spec_tokens: Maximum number of candidate tokens (K) to draft.

    Returns:
        List of draft candidate tokens, or empty list if no match found.
    """
    if len(input_ids) < ngram_size + 1:
        return []

    target_ngram = input_ids[-ngram_size:]
    search_end = len(input_ids) - ngram_size
    # Search backwards from the most recent history to favor temporal locality
    for i in range(search_end - 1, -1, -1):
        if input_ids[i : i + ngram_size] == target_ngram:
            start = i + ngram_size
            end = min(start + max_spec_tokens, search_end)
            if start < end:
                return input_ids[start:end]
    return []


class SpeculativePromptLookupDecoder:
    """
    Prompt-Lookup Speculative Decoder for AxiomLM Transformers on Apple Silicon MPS.
    """

    def __init__(
        self,
        ngram_size: int = 3,
        max_spec_tokens: int = 4,
    ):
        self.ngram_size = ngram_size
        self.max_spec_tokens = max_spec_tokens

    def generate(
        self,
        model: Transformer,
        prompt_tokens: List[int],
        max_new_tokens: int = 64,
        cache: Optional[StaticKVCache] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        Executes speculative prompt lookup decoding.

        Args:
            model: AxiomLM Transformer model in eval mode.
            prompt_tokens: Initial prompt token IDs.
            max_new_tokens: Number of tokens to generate.
            cache: Pre-allocated StaticKVCache (if None, one is created automatically).
            device: Target torch device (defaults to model device).

        Returns:
            Tuple of (generated_tokens, stats_dict)
        """
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

        # 1. Prefill prompt
        prompt_tensor = torch.tensor([prompt_tokens], device=device, dtype=torch.long)
        with torch.no_grad():
            logits, _, _ = model(prompt_tensor, kv_caches=kv_caches)

        next_tok = logits[:, -1, :].argmax(dim=-1).item()
        generated = list(prompt_tokens) + [next_tok]
        target_total = len(prompt_tokens) + max_new_tokens

        total_drafted = 0
        total_accepted = 0
        model_forward_calls = 0

        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            while len(generated) < target_total:
                candidates = find_candidate_ngram(
                    generated,
                    ngram_size=self.ngram_size,
                    max_spec_tokens=self.max_spec_tokens,
                )

                if not candidates:
                    # Single-token fallback decode step
                    curr = torch.tensor(
                        [[generated[-1]]], device=device, dtype=torch.long
                    )
                    logits, _, _ = model(curr, kv_caches=kv_caches)
                    next_t = logits[:, -1, :].argmax(dim=-1).item()
                    generated.append(next_t)
                    model_forward_calls += 1
                else:
                    # Multi-token speculative verification step
                    k_cand = len(candidates)
                    verif_seq = [generated[-1]] + candidates
                    verif_tensor = torch.tensor(
                        [verif_seq], device=device, dtype=torch.long
                    )
                    checkpoint_seen = cache.total_seen_tokens[0]

                    logits, _, _ = model(
                        verif_tensor, kv_caches=kv_caches, return_all_logits=True
                    )
                    model_forward_calls += 1
                    total_drafted += k_cand

                    preds = logits[0].argmax(dim=-1).tolist()
                    accepted = 0

                    for j in range(k_cand):
                        pred_tok = preds[j]
                        if pred_tok == candidates[j]:
                            generated.append(pred_tok)
                            accepted += 1
                            if len(generated) >= target_total:
                                break
                        else:
                            # Mismatch: accept model's genuine prediction and discard rest
                            generated.append(pred_tok)
                            accepted += 1
                            break
                    else:
                        # All candidates matched! Accept bonus token from last verified position
                        if len(generated) < target_total:
                            bonus_tok = preds[k_cand]
                            generated.append(bonus_tok)
                            accepted += 1

                    total_accepted += max(0, accepted - 1)

                    # Rollback / advance KV cache pointers to exact accepted position
                    active = min(checkpoint_seen + accepted, cache.max_seq_len)
                    cache.total_seen_tokens[:] = [checkpoint_seen + accepted] * n_L
                    cache.layer_seq_lens[:] = [active] * n_L

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
