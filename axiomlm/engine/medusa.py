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

from ..models.medusa import (
    compute_medusa_distillation_loss,
    compute_medusa_loss,
    evaluate_medusa_acceptance_matrix,
)
from ..models.transformer import Transformer
from .paged_cache import StaticKVCache

DEFAULT_MEDUSA_TREE_CHOICES: list[tuple[int, ...]] = [
    (0,),
    (1,),
    (2,),
    (3,),
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 1),
    (2, 0),
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    (0, 1, 1),
    (1, 0, 0),
    (0, 0, 0, 0),
]


def compile_medusa_tree(
    tree_choices: list[tuple[int, ...]],
    num_heads: int,
) -> dict[str, Any]:
    """
    Compiles a list of top-k Medusa draft rank tuples into SpecInfer / Medusa
    topology metadata for single-pass tree verification (T <= 32):
      - tree_depths: list[int] of length T (relative RoPE offset per node)
      - tree_mask:   list[int] of length T (32-bit ancestor bitmask per node)
      - parent:      list[int] of length T (-1 for root 0)
      - children:    list[list[int]] adjacency list for greedy verification walk
      - head_idx:    list[int] (-1 for root 0, depth - 1 for candidate nodes)
      - topk_rank:   list[int] (-1 for root 0, rank index within head top-k)
      - max_topk_per_head: list[int] of length num_heads
      - max_depth:   int
    """
    prefix_closed: set[tuple[int, ...]] = set()
    for choice in tree_choices:
        t_choice = tuple(int(r) for r in choice)
        if len(t_choice) == 0 or len(t_choice) > num_heads:
            continue
        for prefix_len in range(1, len(t_choice) + 1):
            prefix_closed.add(t_choice[:prefix_len])

    sorted_paths = sorted(prefix_closed, key=lambda p: (len(p), p))
    if len(sorted_paths) > 31:
        sorted_paths = sorted_paths[:31]

    T = 1 + len(sorted_paths)
    node_id_map: dict[tuple[int, ...], int] = {(): 0}
    tree_depths = [0] * T
    tree_mask = [0] * T
    parent = [-1] * T
    children: list[list[int]] = [[] for _ in range(T)]
    head_idx = [-1] * T
    topk_rank = [-1] * T
    max_topk_per_head = [1] * num_heads

    tree_mask[0] = 1  # Root attends to itself (and full KV prefix)
    max_depth = 0

    for idx_in_list, path in enumerate(sorted_paths):
        u = idx_in_list + 1
        node_id_map[path] = u
        d = len(path)
        max_depth = max(max_depth, d)
        par_u = node_id_map[path[:-1]]
        h = d - 1
        rk = path[-1]

        tree_depths[u] = d
        parent[u] = par_u
        children[par_u].append(u)
        head_idx[u] = h
        topk_rank[u] = rk
        tree_mask[u] = tree_mask[par_u] | (1 << u)
        max_topk_per_head[h] = max(max_topk_per_head[h], rk + 1)

    return {
        "num_nodes": T,
        "tree_depths": tree_depths,
        "tree_mask": tree_mask,
        "parent": parent,
        "children": children,
        "head_idx": head_idx,
        "topk_rank": topk_rank,
        "max_topk_per_head": max_topk_per_head,
        "max_depth": max_depth,
    }


class MedusaSpeculativeDecoder:
    """
    Neural Medusa Speculative Decoder for AxiomLM Transformers.

    Supports both linear chain verification and SpecInfer topology-aware candidate
    tree verification (up to T=32 nodes per pass with 32-bit ancestor bitmasks and
    in-place KV-cache compaction).
    """

    def __init__(
        self,
        num_heads: int = 3,
        temperature: float = 0.0,
        tree_choices: list[tuple[int, ...]] | None = None,
    ):
        self.num_heads = num_heads
        self.temperature = temperature
        self.tree_choices = tree_choices

    def generate(
        self,
        model: Transformer,
        prompt_tokens: list[int],
        max_new_tokens: int = 64,
        cache: StaticKVCache | None = None,
        device: torch.device | None = None,
        tree_choices: list[tuple[int, ...]] | None = None,
    ) -> tuple[list[int], dict[str, Any]]:
        """
        Executes Medusa neural speculative decoding (linear or tree-structured).

        Args:
            model: AxiomLM Transformer model with medusa_heads attached.
            prompt_tokens: List of initial prompt token IDs.
            max_new_tokens: Maximum number of tokens to generate.
            cache: Pre-allocated StaticKVCache (created if None).
            device: Target torch device (defaults to model device).
            tree_choices: Optional candidate tree topology overriding self.tree_choices.

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
        max_cache_len = max(config.block_size, len(prompt_tokens) + max_new_tokens + 32)

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
        active_choices = tree_choices if tree_choices is not None else self.tree_choices
        compiled_tree = (
            compile_medusa_tree(active_choices, k_heads)
            if active_choices is not None
            else None
        )
        use_tree = compiled_tree is not None and compiled_tree["num_nodes"] > 1

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

        def _extract_tree_topk(
            med_list: list[torch.Tensor], pos_idx: int
        ) -> list[list[int]]:
            assert compiled_tree is not None
            res: list[list[int]] = []
            for h in range(k_heads):
                req_k = min(compiled_tree["max_topk_per_head"][h], med_list[h].size(-1))
                vals = torch.topk(
                    med_list[h][0, pos_idx], k=req_k, dim=-1
                ).indices.tolist()
                res.append(vals)
            return res

        # Initial Medusa draft candidates from the prompt's final hidden state
        draft_candidates = [
            medusa_logits[k][:, -1, :].argmax(dim=-1).item() for k in range(k_heads)
        ]
        tree_topk = _extract_tree_topk(medusa_logits, -1) if use_tree else []

        total_drafted = 0
        total_accepted = 0
        model_forward_calls = 1

        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()

        # 2. Medusa Speculative Generation Loop
        with torch.no_grad():
            while len(generated) < target_total:
                checkpoint_seen = cache.total_seen_tokens[0]

                if (
                    use_tree
                    and compiled_tree is not None
                    and checkpoint_seen + compiled_tree["num_nodes"]
                    <= cache.max_seq_len
                ):
                    T_tree = compiled_tree["num_nodes"]
                    verif_seq = [0] * T_tree
                    verif_seq[0] = generated[-1]
                    h_indices = compiled_tree["head_idx"]
                    r_indices = compiled_tree["topk_rank"]
                    for u in range(1, T_tree):
                        h_u = h_indices[u]
                        r_u = r_indices[u]
                        verif_seq[u] = tree_topk[h_u][min(r_u, len(tree_topk[h_u]) - 1)]

                    verif_tensor = torch.tensor(
                        [verif_seq], device=device, dtype=torch.long
                    )
                    logits, _, _ = model.forward_tree_verify(
                        verif_tensor,
                        kv_caches=kv_caches,
                        tree_depths=compiled_tree["tree_depths"],
                        tree_mask=compiled_tree["tree_mask"],
                        return_medusa=False,
                    )
                    model_forward_calls += 1
                    step_max_draft = min(
                        compiled_tree["max_depth"], target_total - len(generated)
                    )
                    total_drafted += step_max_draft

                    preds = logits[0].argmax(dim=-1).tolist()
                    children = compiled_tree["children"]
                    curr_u = 0
                    accepted_nodes = [0]

                    while len(generated) < target_total:
                        pred_tok = preds[curr_u]
                        matched_child = None
                        for child_u in children[curr_u]:
                            if verif_seq[child_u] == pred_tok:
                                matched_child = child_u
                                break
                        generated.append(pred_tok)
                        if matched_child is None or len(generated) >= target_total:
                            break
                        curr_u = matched_child
                        accepted_nodes.append(curr_u)

                    accepted_count = len(accepted_nodes)
                    total_accepted += max(0, accepted_count - 1)
                    cache.compact_tree_path(
                        write_pos=checkpoint_seen,
                        accepted_indices=accepted_nodes,
                        new_total_seen=checkpoint_seen + accepted_count,
                    )

                    if len(generated) >= target_total:
                        break

                    # Lazy terminal-leaf drafting: evaluate Medusa heads ONLY on accepted leaf `curr_u` (M=1)
                    leaf_norm = model._last_verify_norm[:, curr_u : curr_u + 1, :]
                    med_logits_leaf = model.eval_medusa_heads(leaf_norm)
                    tree_topk = _extract_tree_topk(med_logits_leaf, 0)
                    continue

                # Linear chain verification path
                k_cand = min(len(draft_candidates), target_total - len(generated))
                if k_cand <= 0:
                    break

                curr_cand = draft_candidates[:k_cand]
                # Candidate verification sequence: current committed token + candidate draft tokens
                verif_seq = [generated[-1]] + curr_cand

                verif_tensor = torch.tensor(
                    [verif_seq], device=device, dtype=torch.long
                )

                # Single-pass batched forward verification; defer Medusa heads to terminal accepted position
                logits, _, _ = model.forward_with_medusa(
                    verif_tensor,
                    kv_caches=kv_caches,
                    return_all_logits=True,
                    return_medusa=False,
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
                    last_accepted_idx = k_cand

                # Credit accepted draft tokens (excluding the 1 base guaranteed token)
                total_accepted += max(0, accepted - 1)

                # O(1) KV-Cache rollback: advance only the accepted token count
                active = min(checkpoint_seen + accepted, cache.max_seq_len)
                cache.total_seen_tokens[:] = [checkpoint_seen + accepted] * n_L
                cache.layer_seq_lens[:] = [active] * n_L

                if len(generated) >= target_total:
                    break

                # Lazy terminal-leaf drafting: evaluate Medusa heads ONLY at `last_accepted_idx` (M=1)
                last_norm = getattr(model, "_last_verify_norm", None)
                if last_norm is not None and last_accepted_idx < last_norm.size(1):
                    med_logits_leaf = model.eval_medusa_heads(
                        last_norm[:, last_accepted_idx : last_accepted_idx + 1, :]
                    )
                    draft_candidates = [
                        med_logits_leaf[k][0, 0].argmax(dim=-1).item()
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
            "mean_accepted_length": (
                new_tokens_count / model_forward_calls
                if model_forward_calls > 0
                else 0.0
            ),
            "tree_nodes_per_step": (
                compiled_tree["num_nodes"]
                if use_tree and compiled_tree is not None
                else (1 + k_heads)
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
    use_distillation: bool = False,
    temperature: float = 2.0,
    distill_weight: float = 0.5,
    top_k_distill: int | None = None,
    return_stats: bool = False,
) -> list[float] | dict[str, Any]:
    """
    Lightweight training loop for Medusa auxiliary heads with optional KL-divergence
    self-distillation against the frozen base model logits.

    Args:
        model: Transformer model with medusa_heads attached.
        input_ids: Input tokens of shape (B, T).
        targets: Target tokens of shape (B, T) (defaults to input_ids shifted by 1).
        epochs: Number of optimization epochs.
        lr: Learning rate for Medusa heads.
        loss_weights: Decay factors for head losses (default: 0.8^k).
        freeze_backbone: If True, freezes all backbone weights and only updates Medusa heads.
        use_distillation: If True, computes Medusa-2 KL-divergence self-distillation against base logits.
        temperature: Distillation temperature tau.
        distill_weight: Weight lambda for KL divergence (0.0 = pure CE, 1.0 = pure KL).
        top_k_distill: If specified, restricts KL to top-M base logits for large vocabularies.
        return_stats: If True, returns dict with loss histories and empirical acceptance matrix.

    Returns:
        List of loss values per epoch (or detailed stats dict if return_stats=True).
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
    losses: list[float] = []
    ce_history: list[float] = []
    kl_history: list[float] = []

    model.train()
    last_base_logits = None
    last_med_logits = None

    for ep in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        base_logits, medusa_logits, _ = model.forward_with_medusa(input_ids)
        last_base_logits = base_logits
        last_med_logits = medusa_logits

        if use_distillation:
            loss, per_ce, per_kl = compute_medusa_distillation_loss(
                medusa_logits=medusa_logits,
                base_logits=base_logits,
                targets=targets,
                loss_weights=loss_weights,
                temperature=temperature,
                distill_weight=distill_weight,
                top_k_distill=top_k_distill,
            )
            ce_history.append(sum(c.item() for c in per_ce))
            kl_history.append(sum(k.item() for k in per_kl))
        else:
            loss, per_ce = compute_medusa_loss(
                medusa_logits=medusa_logits,
                targets=targets,
                loss_weights=loss_weights,
            )
            ce_history.append(sum(c.item() for c in per_ce))

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    model.medusa_heads.to(dtype=orig_dtype)
    model._quant_weights_cache = None
    model.eval()

    if return_stats:
        acc_stats = (
            evaluate_medusa_acceptance_matrix(
                last_med_logits, last_base_logits, max_topk=4
            )
            if (last_base_logits is not None and last_med_logits is not None)
            else {}
        )
        return {
            "losses": losses,
            "ce_history": ce_history,
            "kl_history": kl_history,
            "acceptance_stats": acc_stats,
        }

    return losses
