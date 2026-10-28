"""
AxiomLM Causal Self-Attention with Grouped-Query Attention (GQA) & KV-Cache.
"""

from typing import Tuple, Optional, Any, Union
import torch
import torch.nn as nn
from torch.nn import functional as F

from .modules import apply_rope, repeat_kv

try:
    from ..kernels import fused_sdpa
    from ..kernels.ops import _METAL_MOD

    HAS_CUSTOM_KERNELS = True
except (ImportError, ValueError):
    try:
        from kernels import fused_sdpa
        from kernels.ops import _METAL_MOD

        HAS_CUSTOM_KERNELS = True
    except ImportError:
        HAS_CUSTOM_KERNELS = False
        _METAL_MOD = None


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
        self.n_kv_head = (
            config.n_kv_head
            if getattr(config, "n_kv_head", None) is not None
            else config.n_head
        )
        self.separate_qkv = self.n_kv_head != self.n_head
        self.n_rep = self.n_head // self.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.pos_emb = getattr(config, "pos_emb", "learned")
        self.sliding_window = getattr(config, "sliding_window", None)
        self.use_fused_kernels = (
            getattr(config, "use_fused_kernels", False) and HAS_CUSTOM_KERNELS
        )

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
    ) -> Union[
        torch.Tensor, Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]
    ]:
        B, T, C = x.size()

        start_pos = 0
        if kv_cache is not None:
            if isinstance(kv_cache, tuple):
                start_pos = kv_cache[0].shape[2]
            elif hasattr(kv_cache, "total_seen_tokens"):
                start_pos = kv_cache.total_seen_tokens[getattr(self, "layer_idx", 0)]
            else:
                start_pos = kv_cache.seq_len

        qkv = self.c_attn(x)
        q_dim = self.n_head * self.head_dim
        kv_dim = self.n_kv_head * self.head_dim

        # Fast-path for T=1 single-token autoregressive decode on MPS
        if (
            T == 1
            and not torch.is_grad_enabled()
            and self.pos_emb == "rope"
            and freqs_cis is not None
            and self.use_fused_kernels
            and _METAL_MOD is not None
            and x.device.type == "mps"
            and not freqs_cis.is_complex()
        ):
            if (
                B == 1
                and kv_cache is not None
                and hasattr(kv_cache, "k_caches")
                and kv_cache.max_batch_size == 1
            ):
                layer_idx = getattr(self, "layer_idx", 0)
                seen = kv_cache.total_seen_tokens[layer_idx]
                max_seq = kv_cache.max_seq_len
                write_pos = (
                    (seen % max_seq)
                    if (kv_cache.ring_buffer or seen >= max_seq)
                    else seen
                )
                new_seen = seen + 1
                active_len = new_seen if new_seen <= max_seq else max_seq
                k_buf = kv_cache.k_caches[layer_idx]
                v_buf = kv_cache.v_caches[layer_idx]
                _METAL_MOD.rope_kv_cache_write_mps(
                    qkv,
                    freqs_cis,
                    k_buf,
                    v_buf,
                    q_dim,
                    kv_dim,
                    self.head_dim,
                    seen % freqs_cis.size(0),
                    write_pos,
                )
                kv_cache.total_seen_tokens[layer_idx] = new_seen
                kv_cache.layer_seq_lens[layer_idx] = active_len
                q = qkv[:, :, :q_dim].view(1, self.n_head, 1, self.head_dim)
                k = k_buf if active_len == max_seq else k_buf[:, :, :active_len, :]
                v = v_buf if active_len == max_seq else v_buf[:, :, :active_len, :]
                new_kv_cache = kv_cache
                kv_already_updated = True
            else:
                qk_view = qkv[:, :, : q_dim + kv_dim].view(
                    B, self.n_head + self.n_kv_head, 1, self.head_dim
                )
                _METAL_MOD.apply_rope_inplace_mps(
                    qk_view, freqs_cis, start_pos % freqs_cis.size(0)
                )
                q = qkv[:, :, :q_dim].view(B, self.n_head, 1, self.head_dim)
                k = qkv[:, :, q_dim : q_dim + kv_dim].view(
                    B, self.n_kv_head, 1, self.head_dim
                )
                v = qkv[:, :, q_dim + kv_dim :].view(
                    B, self.n_kv_head, 1, self.head_dim
                )
                kv_already_updated = False
        else:
            kv_already_updated = False
            if not self.separate_qkv:
                q, k, v = qkv.split(self.n_embd, dim=2)
                if T == 1:
                    q = q.view(B, self.n_head, 1, self.head_dim)
                    k = k.view(B, self.n_head, 1, self.head_dim)
                    v = v.view(B, self.n_head, 1, self.head_dim)
                else:
                    q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                    k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                    v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            else:
                q = qkv[:, :, :q_dim]
                k = qkv[:, :, q_dim : q_dim + kv_dim]
                v = qkv[:, :, q_dim + kv_dim :]
                if T == 1:
                    q = q.view(B, self.n_head, 1, self.head_dim)
                    k = k.view(B, self.n_kv_head, 1, self.head_dim)
                    v = v.view(B, self.n_kv_head, 1, self.head_dim)
                else:
                    q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                    k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
                    v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

            # Apply RoPE if configured
            if self.pos_emb == "rope" and freqs_cis is not None:
                if self.use_fused_kernels:
                    q = apply_rope(q, freqs_cis, start_pos=start_pos)
                    k = apply_rope(k, freqs_cis, start_pos=start_pos)
                else:
                    orig_dtype = q.dtype
                    freqs_slice = freqs_cis[start_pos : start_pos + T]
                    if not freqs_slice.is_complex():
                        freqs_slice = torch.view_as_complex(
                            freqs_slice.float().contiguous()
                        )
                    freqs_slice = freqs_slice.view(1, 1, T, -1).to(q.device)
                    q_c = torch.view_as_complex(
                        q.float().reshape(B, self.n_head, T, -1, 2)
                    )
                    k_c = torch.view_as_complex(
                        k.float().reshape(B, self.n_kv_head, T, -1, 2)
                    )
                    q = (
                        torch.view_as_real(q_c * freqs_slice)
                        .reshape(B, self.n_head, T, self.head_dim)
                        .to(orig_dtype)
                    )
                    k = (
                        torch.view_as_real(k_c * freqs_slice)
                        .reshape(B, self.n_kv_head, T, self.head_dim)
                        .to(orig_dtype)
                    )

        # Update Key-Value Cache
        if kv_already_updated:
            pass
        elif kv_cache is not None:
            if isinstance(kv_cache, tuple):
                k_prev, v_prev = kv_cache
                k = torch.cat([k_prev, k], dim=2)
                v = torch.cat([v_prev, v], dim=2)
                new_kv_cache = (k, v)
            elif hasattr(kv_cache, "update"):
                # Fast zero-allocation StaticKVCache / RingBufferKVCache
                k, v = kv_cache.update(getattr(self, "layer_idx", 0), k, v)
                new_kv_cache = kv_cache
            else:
                # Assuming kv_cache is a SequenceContext
                # It expects [1, seq_len, n_kv_head, head_dim]. Currently k is [B, n_kv_head, seq_len, head_dim]
                kv_cache.append_kv(
                    getattr(self, "layer_idx", 0), k.transpose(1, 2), v.transpose(1, 2)
                )
                k_rec, v_rec = kv_cache.get_reconstructed_cache(
                    getattr(self, "layer_idx", 0)
                )
                # Reconstructed shape: [1, total_seq_len, n_kv_head, head_dim]. Need to transpose back to [1, n_kv_head, total_seq_len, head_dim]
                k = k_rec.transpose(1, 2)
                v = v_rec.transpose(1, 2)
                new_kv_cache = kv_cache
        elif use_cache:
            new_kv_cache = (k, v)
        else:
            new_kv_cache = None

        is_causal = (T > 1) and (start_pos == 0)

        # Sliding Window Attention (SWA)
        if self.sliding_window is not None and is_causal:
            k_rep = repeat_kv(k, self.n_rep)
            v_rep = repeat_kv(v, self.n_rep)
            if self.use_fused_kernels:
                y = fused_sdpa(
                    q, k_rep, v_rep, is_causal=True, sliding_window=self.sliding_window
                )
            else:
                # Build custom sliding window mask for F.scaled_dot_product_attention
                causal_mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
                window_mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril(
                    diagonal=-self.sliding_window
                )
                attn_mask = causal_mask & ~window_mask
                y = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=attn_mask)
        else:
            if self.use_fused_kernels:
                y = fused_sdpa(q, k, v, is_causal=is_causal, sliding_window=None)
            else:
                # Use native zero-copy GQA broadcasting when n_rep > 1
                if self.n_rep > 1:
                    y = F.scaled_dot_product_attention(
                        q, k, v, is_causal=is_causal, enable_gqa=True
                    )
                elif k.size(1) != q.size(1):
                    k_rep = repeat_kv(k, self.n_rep)
                    v_rep = repeat_kv(v, self.n_rep)
                    y = F.scaled_dot_product_attention(
                        q, k_rep, v_rep, is_causal=is_causal
                    )
                else:
                    y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        if T == 1:
            y = y.view(B, 1, C)
        else:
            y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        if kv_cache is not None or use_cache:
            return y, new_kv_cache if new_kv_cache is not None else (k, v)
        return y, None
