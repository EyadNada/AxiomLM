"""
AxiomLM Causal Self-Attention with Grouped-Query Attention (GQA) & KV-Cache.
"""
from typing import Tuple, Optional, Any, List, Union
import torch
import torch.nn as nn
from torch.nn import functional as F

from .modules import apply_rope, repeat_kv


class CausalSelfAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention with support for:
    - Multi-Head Attention (MHA) and Grouped-Query Attention (GQA)
    - Rotary Position Embeddings (RoPE)
    - FlashAttention / Scaled Dot-Product Attention (SDPA)
    - Key-Value (KV) cache for O(1) autoregressive generation
    """
    def __init__(self, config: Any):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if getattr(config, "n_kv_head", None) is not None else config.n_head
        self.separate_qkv = (self.n_kv_head != self.n_head)
        self.n_rep = self.n_head // self.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.pos_emb = getattr(config, "pos_emb", "learned")

        # Projections
        if not self.separate_qkv:
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        else:
            q_dim = self.n_head * self.head_dim
            kv_dim = self.n_kv_head * self.head_dim
            self.c_attn = nn.Linear(config.n_embd, q_dim + 2 * kv_dim, bias=config.bias)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # type: ignore

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
        B, T, C = x.size()

        if not self.separate_qkv:
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        else:
            qkv = self.c_attn(x)
            q_dim = self.n_head * self.head_dim
            kv_dim = self.n_kv_head * self.head_dim
            q = qkv[:, :, :q_dim]
            k = qkv[:, :, q_dim : q_dim + kv_dim]
            v = qkv[:, :, q_dim + kv_dim :]
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        start_pos = kv_cache[0].shape[2] if kv_cache is not None else 0

        # Apply RoPE if configured
        if self.pos_emb == "rope" and freqs_cis is not None:
            q = apply_rope(q, freqs_cis, start_pos=start_pos)
            k = apply_rope(k, freqs_cis, start_pos=start_pos)

        # Update Key-Value Cache
        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)
            new_kv_cache = (k, v)
        elif use_cache:
            new_kv_cache = (k, v)
        else:
            new_kv_cache = None

        # GQA broadcasting
        k_rep = repeat_kv(k, self.n_rep)
        v_rep = repeat_kv(v, self.n_rep)

        is_causal = (T > 1) and (start_pos == 0)
        y = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=is_causal)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        if kv_cache is not None or use_cache:
            return y, new_kv_cache if new_kv_cache is not None else (k, v)
        return y, None
