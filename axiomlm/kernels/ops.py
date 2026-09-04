"""
Axiom-LM Fused Custom Operators & Autograd Bindings.

Provides high-performance drop-in replacements for:
1. FusedRMSNorm (Function & Module)
2. FusedSwiGLU (Function & Module)

Automatically dispatches to:
- Apple Silicon ARM NEON SIMD C++ extension (macOS MPS/CPU)
- OpenAI Triton JIT kernels (NVIDIA CUDA)
- Vectorized PyTorch reference fallback
"""

import math
from typing import Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .build_kernels import load_neon_module

# Load native compiled NEON module
_NEON_MOD = load_neon_module()

try:
    from .triton_kernels import (
        HAS_TRITON,
        triton_rmsnorm_forward,
        triton_swiglu_forward,
        triton_fused_sdpa_forward,
    )
except ImportError:
    HAS_TRITON = False


# ----------------------------------------------------------------------------
# 1. Fused RMSNorm Autograd Function
# ----------------------------------------------------------------------------

class FusedRMSNormFunction(torch.autograd.Function):
    """
    Fused Root Mean Square Normalization Autograd Function.
    Calculates forward normalization and backward gradients in single SRAM passes.
    """
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.contiguous().view(-1, orig_shape[-1])
        weight_flat = weight.contiguous()

        device = x.device
        if device.type == "cuda" and HAS_TRITON:
            out, rsqrt_cache = triton_rmsnorm_forward(x_flat, weight_flat, eps=eps)
        elif _NEON_MOD is not None and device.type == "cpu" and x.dtype == torch.float32:
            out, rsqrt_cache = _NEON_MOD.rmsnorm_forward_neon(x_flat, weight_flat, eps)
        else:
            # High-efficiency native PyTorch single-pass on MPS/other devices
            mean_sq = x_flat.pow(2).mean(-1, keepdim=True)
            rsqrt_cache = torch.rsqrt(mean_sq + eps)
            out = x_flat * rsqrt_cache * weight_flat

        ctx.save_for_backward(x_flat, weight_flat, rsqrt_cache)
        ctx.eps = eps
        ctx.orig_shape = orig_shape
        return out.view(orig_shape)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], None]:
        x_flat, weight_flat, rsqrt_cache = ctx.saved_tensors
        grad_out_flat = grad_output.contiguous().view(-1, ctx.orig_shape[-1])
        device = grad_output.device

        if _NEON_MOD is not None and device.type == "cpu" and grad_output.dtype == torch.float32:
            grad_x, grad_weight = _NEON_MOD.rmsnorm_backward_neon(
                grad_out_flat, x_flat, weight_flat, rsqrt_cache
            )
        else:
            # Exact analytical backward derivation
            D = x_flat.size(-1)
            r_val = rsqrt_cache if rsqrt_cache.dim() == 2 else rsqrt_cache.unsqueeze(-1)
            dy_w = grad_out_flat * weight_flat
            inner = (dy_w * x_flat).sum(-1, keepdim=True)
            scale = (inner / D) * (r_val * r_val * r_val)
            grad_x = (dy_w * r_val) - (x_flat * scale)
            grad_weight = (grad_out_flat * x_flat * r_val).sum(0)

        return grad_x.view(ctx.orig_shape), grad_weight, None


def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Functional interface for Fused RMSNorm."""
    return FusedRMSNormFunction.apply(x, weight, eps)


class FusedRMSNorm(nn.Module):
    """
    Fused Root Mean Square Normalization Layer.
    Drop-in replacement for standard RMSNorm with kernel acceleration.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_rmsnorm(x, self.weight, self.eps)


# ----------------------------------------------------------------------------
# 2. Fused SwiGLU Autograd Function
# ----------------------------------------------------------------------------

class FusedSwiGLUFunction(torch.autograd.Function):
    """
    Fused SwiGLU (Swish Gated Linear Unit) Autograd Function.
    Calculates y = SiLU(gate) * up in a single fused pass.
    """
    @staticmethod
    def forward(ctx: Any, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        device = gate.device
        if device.type == "cuda" and HAS_TRITON:
            out = triton_swiglu_forward(gate.contiguous(), up.contiguous())
        else:
            out = F.silu(gate) * up

        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate, up = ctx.saved_tensors
        sig_g = torch.sigmoid(gate)
        silu_g = gate * sig_g
        grad_up = grad_output * silu_g
        d_silu_dg = sig_g * (1.0 + gate * (1.0 - sig_g))
        grad_gate = grad_output * up * d_silu_dg
        return grad_gate, grad_up


def fused_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Functional interface for Fused SwiGLU."""
    return FusedSwiGLUFunction.apply(gate, up)


class FusedSwiGLUMLP(nn.Module):
    """
    Fused SwiGLU Multi-Layer Perceptron (LLaMA-3 spec) with custom fused kernel.
    """
    def __init__(self, config: Any):
        super().__init__()
        hidden_dim = int(2 * (4 * config.n_embd) / 3)
        hidden_dim = 64 * ((hidden_dim + 64 - 1) // 64)
        bias = getattr(config, "bias", False)
        self.w_gate = nn.Linear(config.n_embd, hidden_dim, bias=bias)
        self.w_up = nn.Linear(config.n_embd, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, config.n_embd, bias=bias)
        setattr(self.w_down, "NANOGPT_SCALE_INIT", 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w_gate(x)
        up = self.w_up(x)
        return self.w_down(fused_swiglu(gate, up))

# ----------------------------------------------------------------------------
# 3. Fused SDPA (Scaled Dot-Product Attention) Autograd Function
# ----------------------------------------------------------------------------

class FusedSDPAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        device = q.device
        if device.type == "cuda" and HAS_TRITON:
            out = triton_fused_sdpa_forward(q, k, v, is_causal=is_causal)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        ctx.save_for_backward(q, k, v)
        ctx.is_causal = is_causal
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        q, k, v = ctx.saved_tensors
        # Using PyTorch's highly optimized SDPA backward for exact parity
        with torch.enable_grad():
            q_ = q.detach().requires_grad_(True)
            k_ = k.detach().requires_grad_(True)
            v_ = v.detach().requires_grad_(True)
            out = F.scaled_dot_product_attention(q_, k_, v_, is_causal=ctx.is_causal)
            out.backward(grad_output)
        return q_.grad, k_.grad, v_.grad, None

def fused_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
    """Functional interface for Fused SDPA (FlashAttention)."""
    return FusedSDPAFunction.apply(q, k, v, is_causal)
