"""
AxiomLM Architectural Primitives & Modern Transformer Components.
"""
from typing import Tuple, Optional, Any
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    from ..kernels import FusedRMSNorm, FusedSwiGLUMLP
    HAS_CUSTOM_KERNELS = True
except (ImportError, ValueError):
    try:
        from kernels import FusedRMSNorm, FusedSwiGLUMLP
        HAS_CUSTOM_KERNELS = True
    except ImportError:
        HAS_CUSTOM_KERNELS = False
        FusedRMSNorm = None  # type: ignore
        FusedSwiGLUMLP = None  # type: ignore


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm).
    Replaces LayerNorm by removing mean centering, improving throughput and numerical stability.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def precompute_rope_frequencies(head_dim: int, max_seq_len: int = 2048, theta: float = 10000.0) -> torch.Tensor:
    """
    Precomputes complex rotary frequency tensors for Rotary Position Embeddings (RoPE).
    Returns complex64 tensor of shape (max_seq_len, head_dim // 2).
    """
    assert head_dim % 2 == 0, f"head_dim ({head_dim}) must be even for complex RoPE"
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    """
    Applies Rotary Position Embeddings (RoPE) to query or key tensors via complex multiplication.
    Args:
        x: Tensor of shape (B, n_head, T, head_dim)
        freqs_cis: Complex tensor of shape (max_seq_len, head_dim // 2)
        start_pos: Offset for KV-cache decoding step
    Returns:
        Rotated tensor of same shape and dtype as x.
    """
    orig_dtype = x.dtype
    B, n_head, T, head_dim = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(B, n_head, T, -1, 2))
    freqs_slice = freqs_cis[start_pos : start_pos + T, :].view(1, 1, T, -1).to(x.device)
    x_rotated = torch.view_as_real(x_complex * freqs_slice).reshape(B, n_head, T, head_dim)
    return x_rotated.to(orig_dtype)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expands Key/Value heads to match Query heads for Grouped-Query Attention (GQA).
    Input: (B, n_kv_head, T, head_dim) -> Output: (B, n_head, T, head_dim)
    """
    if n_rep == 1:
        return x
    B, n_kv_head, T, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, n_kv_head, n_rep, T, head_dim)
        .reshape(B, n_kv_head * n_rep, T, head_dim)
    )


class SwiGLUMLP(nn.Module):
    """
    SwiGLU Gated Feed-Forward Network.
    Computes: SwiGLU(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
    """
    def __init__(self, config: Any, hidden_dim: Optional[int] = None, bias: bool = False):
        super().__init__()
        if hasattr(config, "n_embd"):
            n_embd = config.n_embd
            bias = getattr(config, "bias", False)
        else:
            n_embd = int(config)
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * n_embd) / 3)
            hidden_dim = 64 * ((hidden_dim + 64 - 1) // 64)
        self.w_gate = nn.Linear(n_embd, hidden_dim, bias=bias)
        self.w_up = nn.Linear(n_embd, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, n_embd, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MLP(nn.Module):
    """Classic GPT-2 Feed-Forward Network with GELU activation."""
    def __init__(self, config: Any):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
