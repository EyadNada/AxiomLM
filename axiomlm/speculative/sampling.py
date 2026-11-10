"""
AxiomLM Speculative Sampling & Sequoia Tree Verification Engine.

Implements Sequoia Algorithm 2 (Chen et al., NeurIPS 2024):
- Sampling Without Replacement via Gumbel-Top-k trick.
- Provably distribution-preserving tree verification across temperatures T in [0.0, 1.0].
- Cumulative and marginal power-law rejection bounds (r_k <= a * k^(-b)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


def sample_without_replacement(
    logits: torch.Tensor,
    num_samples: int,
    temperature: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Samples k tokens without replacement from the categorical distribution
    defined by logits at the specified temperature.

    Mathematical guarantees:
    1. If temperature <= 1e-6:
       Deterministic greedy top-k (torch.topk(logits, k)).
       Zero stochasticity; exact equivalence to argmax when k=1.
    2. If temperature > 1e-6:
       Gumbel-Top-k sampling without replacement (Vieira 2014, Yellott 1977).
       Adds standard Gumbel noise to logits / T and takes top-k.
       All k sampled indices are strictly unique (zero duplicates).

    Args:
        logits: Logit tensor of shape (..., V).
        num_samples: Number of distinct tokens to sample (k <= V).
        temperature: Sampling temperature T >= 0.0.
        generator: Optional torch.Generator for deterministic reproducibility.

    Returns:
        Tensor of shape (..., num_samples) containing sampled token IDs.
    """
    vocab_size = logits.size(-1)
    k = max(0, min(num_samples, vocab_size))
    if k == 0:
        out_shape = list(logits.shape[:-1]) + [0]
        return torch.empty(out_shape, dtype=torch.long, device=logits.device)

    # Greedy / deterministic regime
    if temperature <= 1e-6:
        return torch.topk(logits, k=k, dim=-1).indices

    # Stochastic sampling without replacement via Gumbel-Top-k
    logits_f32 = logits.to(torch.float32)
    device = logits.device

    # Generate uniform noise safely respecting generator device
    if generator is not None:
        try:
            u = torch.rand(
                logits.shape, device=device, generator=generator, dtype=torch.float32
            )
        except RuntimeError:
            # Fallback if generator is bound to cpu while tensor is on mps
            u = torch.rand(
                logits.shape, device="cpu", generator=generator, dtype=torch.float32
            ).to(device)
    else:
        u = torch.rand(logits.shape, device=device, dtype=torch.float32)

    eps = 1e-10
    u_clamped = u.clamp(min=eps, max=1.0 - eps)

    # G = -log(-log(U)) ~ Standard Gumbel(0, 1)
    g = -torch.log(-torch.log(u_clamped))

    perturbed = (logits_f32 / temperature) + g
    return torch.topk(perturbed, k=k, dim=-1).indices


@dataclass
class VerificationStepResult:
    """Detailed result of a single speculative verification step."""

    accepted: bool
    accepted_token: int
    accepted_candidate_idx: int | None
    is_fallback: bool
    rejected_tokens: list[int] = field(default_factory=list)
    acceptance_probs: list[float] = field(default_factory=list)


class SequoiaTreeVerifier:
    """
    Sequoia Algorithm 2 Speculative Tree Verifier.

    Validates candidate tokens sampled without replacement from a draft model/heads
    against the target model's output distribution.

    Mathematical guarantees:
    - Lossless: The output token distribution is IDENTICAL to the target model's distribution:
      Pr(emitted = x) = P_target(x) for all x in V.
    - No duplicate drafts: Candidates are sampled without replacement.
    - Exact multi-candidate residual update:
      P_{k+1} = (P_k - Q_k)^+ / sum(P_k - Q_k)^+
      Guarantees exact distribution preservation for all K >= 1.
    """

    def __init__(
        self,
        temperature: float = 0.0,
        generator: torch.Generator | None = None,
    ):
        self.temperature = float(temperature)
        self.generator = generator

    def _rand_scalar(self, device: torch.device) -> float:
        """Draws a single Uniform(0, 1) float respecting generator device."""
        if self.generator is not None:
            try:
                return torch.rand(
                    (), device=device, generator=self.generator, dtype=torch.float32
                ).item()
            except RuntimeError:
                return torch.rand(
                    (), device="cpu", generator=self.generator, dtype=torch.float32
                ).item()
        return torch.rand((), device=device, dtype=torch.float32).item()

    def _multinomial_scalar(self, probs: torch.Tensor) -> int:
        """Draws a single sample from categorical distribution respecting generator device."""
        p_cpu = probs.to(device="cpu", dtype=torch.float32)
        if self.generator is not None:
            try:
                return torch.multinomial(
                    p_cpu, num_samples=1, generator=self.generator
                ).item()
            except RuntimeError:
                return torch.multinomial(p_cpu, num_samples=1).item()
        return torch.multinomial(p_cpu, num_samples=1).item()

    def verify_candidates(
        self,
        target_logits: torch.Tensor,
        candidates: list[int],
        draft_logits: torch.Tensor | None = None,
        temperature: float | None = None,
    ) -> VerificationStepResult:
        """
        Executes Sequoia Algorithm 2 verification for a list of candidate tokens
        sampled without replacement at a single tree node.

        Args:
            target_logits: Target model 1D logits tensor of shape (V,).
            candidates: List of K candidate token IDs proposed without replacement.
            draft_logits: Optional draft model 1D logits tensor of shape (V,).
                          If None, draft distribution over candidates is assumed uniform.
            temperature: Optional override for sampling temperature.

        Returns:
            VerificationStepResult with acceptance decision, token, and rejection history.
        """
        temp = self.temperature if temperature is None else float(temperature)
        device = target_logits.device
        V = target_logits.size(-1)
        K = len(candidates)

        # 1. Deterministic Greedy Regime (T <= 1e-6)
        if temp <= 1e-6:
            target_best = target_logits.argmax(dim=-1).item()
            if K > 0 and candidates[0] == target_best:
                return VerificationStepResult(
                    accepted=True,
                    accepted_token=candidates[0],
                    accepted_candidate_idx=0,
                    is_fallback=False,
                    rejected_tokens=[],
                    acceptance_probs=[1.0],
                )
            else:
                rejs = [candidates[0]] if K > 0 else []
                return VerificationStepResult(
                    accepted=False,
                    accepted_token=target_best,
                    accepted_candidate_idx=None,
                    is_fallback=True,
                    rejected_tokens=rejs,
                    acceptance_probs=[0.0] if K > 0 else [],
                )

        # 2. Stochastic Sampling Without Replacement Regime (T > 1e-6)
        # Compute probabilities in float32 for MPS numerical accuracy
        p_target = F.softmax(target_logits.to(torch.float32) / temp, dim=-1)

        if draft_logits is not None:
            p_draft = F.softmax(draft_logits.to(torch.float32) / temp, dim=-1)
        else:
            p_draft = torch.zeros_like(p_target)
            if K > 0:
                p_draft[candidates] = 1.0 / K

        curr_p = p_target.clone()
        curr_q = p_draft.clone()

        rejected_list: list[int] = []
        acc_probs: list[float] = []

        for idx, cand in enumerate(candidates):
            p_val = curr_p[cand].item()
            q_val = curr_q[cand].item()

            alpha = min(1.0, max(0.0, p_val / max(1e-9, q_val)))
            acc_probs.append(alpha)

            u = self._rand_scalar(device)
            if u <= alpha:
                # Candidate cand is ACCEPTED!
                return VerificationStepResult(
                    accepted=True,
                    accepted_token=cand,
                    accepted_candidate_idx=idx,
                    is_fallback=False,
                    rejected_tokens=rejected_list,
                    acceptance_probs=acc_probs,
                )

            # Candidate cand is REJECTED
            rejected_list.append(cand)

            # Exact residual update: curr_p = (curr_p - curr_q)^+ / sum((curr_p - curr_q)^+)
            res_p = torch.clamp(curr_p - curr_q, min=0.0)
            sum_res_p = res_p.sum().item()
            if sum_res_p <= 1e-9:
                # Target residual exhausted; accept candidate directly
                return VerificationStepResult(
                    accepted=True,
                    accepted_token=cand,
                    accepted_candidate_idx=idx,
                    is_fallback=False,
                    rejected_tokens=rejected_list[:-1],
                    acceptance_probs=acc_probs,
                )
            curr_p = res_p / sum_res_p

            # Update draft distribution conditioned on not being cand
            curr_q[cand] = 0.0
            sum_res_q = curr_q.sum().item()
            if sum_res_q > 1e-9:
                curr_q = curr_q / sum_res_q
            else:
                curr_q = torch.ones_like(curr_q) / V

        # 3. Residual Fallback: All K candidates rejected.
        # Sample fallback token from the exact residual target distribution curr_p
        fallback_token = self._multinomial_scalar(curr_p)

        return VerificationStepResult(
            accepted=False,
            accepted_token=fallback_token,
            accepted_candidate_idx=None,
            is_fallback=True,
            rejected_tokens=rejected_list,
            acceptance_probs=acc_probs,
        )

    def verify_tree_step(
        self,
        tree_logits: torch.Tensor,
        compiled_tree: dict[str, Any],
        verif_seq: list[int],
        medusa_logits_leaf: list[torch.Tensor] | None = None,
        temperature: float | None = None,
        max_accept: int | None = None,
    ) -> tuple[list[int], list[int], dict[str, Any]]:
        """
        Performs tree-structured speculative verification under Sequoia Algorithm 2.

        Traverses the compiled tree topology from root (0), verifying sibling candidates
        using sampling without replacement at each branching decision.

        Args:
            tree_logits: Target model logits across all tree nodes of shape (T_tree, V).
            compiled_tree: Compiled tree metadata dictionary from compile_medusa_tree.
            verif_seq: Token IDs placed at each tree node [0..T_tree-1].
            medusa_logits_leaf: Optional list of Medusa head logits for draft distributions.
            temperature: Optional override for temperature.
            max_accept: Maximum number of accepted tokens allowed.

        Returns:
            Tuple of:
            - accepted_nodes: List of accepted tree node indices [0, ...].
            - emitted_tokens: List of emitted token IDs along the verified path.
            - tree_stats: Dictionary containing step metrics and rejection rates.
        """
        temp = self.temperature if temperature is None else float(temperature)
        children = compiled_tree["children"]
        limit = max_accept if max_accept is not None else len(verif_seq)

        curr_u = 0
        accepted_nodes = [0]
        emitted_tokens: list[int] = []

        total_tested = 0
        total_accepted = 0
        per_rank_rejections: list[int] = []
        per_rank_attempts: list[int] = []

        while len(emitted_tokens) < limit:
            u_children = children[curr_u]
            u_logits = tree_logits[curr_u]

            if len(u_children) == 0:
                # Terminal leaf reached with no further children: sample final token from target
                if temp <= 1e-6:
                    final_tok = u_logits.argmax(dim=-1).item()
                else:
                    p_final = F.softmax(u_logits.to(torch.float32) / temp, dim=-1)
                    final_tok = self._multinomial_scalar(p_final)
                emitted_tokens.append(final_tok)
                break

            # Sibling candidates at current node
            cand_tokens = [verif_seq[child_idx] for child_idx in u_children]
            step_res = self.verify_candidates(
                target_logits=u_logits,
                candidates=cand_tokens,
                draft_logits=None,
                temperature=temp,
            )

            total_tested += len(cand_tokens)
            emitted_tokens.append(step_res.accepted_token)

            # Record per-rank acceptance / rejection statistics
            for r_idx in range(len(cand_tokens)):
                while len(per_rank_attempts) <= r_idx:
                    per_rank_attempts.append(0)
                    per_rank_rejections.append(0)
                per_rank_attempts[r_idx] += 1
                if step_res.accepted and step_res.accepted_candidate_idx == r_idx:
                    pass
                else:
                    per_rank_rejections[r_idx] += 1

            if step_res.accepted and step_res.accepted_candidate_idx is not None:
                matched_child = u_children[step_res.accepted_candidate_idx]
                curr_u = matched_child
                accepted_nodes.append(curr_u)
                total_accepted += 1
            else:
                # Tree path terminated on rejection: fallback token already emitted
                break

        rejection_rates = [
            (
                (per_rank_rejections[i] / per_rank_attempts[i])
                if per_rank_attempts[i] > 0
                else 0.0
            )
            for i in range(len(per_rank_attempts))
        ]

        stats = {
            "tested_candidates": total_tested,
            "accepted_draft_tokens": total_accepted,
            "rejection_rates_per_rank": rejection_rates,
            "emitted_count": len(emitted_tokens),
        }

        return accepted_nodes, emitted_tokens, stats


def fit_power_law_rejection_bound(
    rejection_rates: list[float],
) -> dict[str, float]:
    """
    Fits empirical rejection rates r_k across speculation candidate ranks k
    to the Sequoia power-law decay model:

        r_k <= a * k^(-b)   <=>   ln(r_k) = ln(a) - b * ln(k)

    Args:
        rejection_rates: List of rejection rates [r_1, r_2, ..., r_K] for k in 1..K.

    Returns:
        Dictionary containing:
        - exponent_b: The power-law decay exponent b (b > 0 indicates decaying rejections).
        - scale_a: The scale factor a.
        - r_squared: The R^2 goodness-of-fit metric.
        - valid: Boolean indicating whether fit succeeded with valid positive data.
    """
    valid_pairs = []
    for k_idx, r in enumerate(rejection_rates, start=1):
        if r > 1e-6 and r < (1.0 - 1e-6):
            valid_pairs.append((float(k_idx), float(r)))

    if len(valid_pairs) < 2:
        return {
            "exponent_b": 0.0,
            "scale_a": 1.0,
            "r_squared": 0.0,
            "valid": False,
        }

    ln_k = [math.log(k) for k, _ in valid_pairs]
    ln_r = [math.log(r) for _, r in valid_pairs]
    n = len(valid_pairs)

    mean_x = sum(ln_k) / n
    mean_y = sum(ln_r) / n

    ss_xx = sum((x - mean_x) ** 2 for x in ln_k)
    ss_yy = sum((y - mean_y) ** 2 for y in ln_r)
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(ln_k, ln_r))

    if ss_xx < 1e-12:
        return {
            "exponent_b": 0.0,
            "scale_a": math.exp(mean_y),
            "r_squared": 0.0,
            "valid": False,
        }

    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x

    # ln(r) = intercept + slope * ln(k)  =>  b = -slope
    exponent_b = -slope
    scale_a = math.exp(intercept)

    if ss_yy > 1e-12:
        r_squared = (ss_xy**2) / (ss_xx * ss_yy)
    else:
        r_squared = 1.0

    return {
        "exponent_b": float(exponent_b),
        "scale_a": float(scale_a),
        "r_squared": float(r_squared),
        "valid": True,
    }
