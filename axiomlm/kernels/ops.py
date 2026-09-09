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

from .build_kernels import load_neon_module, load_metal_module

# Load native compiled modules
_NEON_MOD = load_neon_module()
_METAL_MOD = load_metal_module()

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
    @staticmethod
    def forward(ctx, x, weight, eps):
        ctx.eps = eps
        
        if x.device.type == "cuda" and HAS_TRITON:
            y, rsqrt = _triton_kernels.rmsnorm_forward(x, weight, eps)
            ctx.save_for_backward(x, weight, rsqrt)
            return y
        elif _NEON_MOD is not None and x.device.type == "cpu" and x.dtype == torch.float32:
            orig_shape = x.shape
            x_flat = x.contiguous().view(-1, orig_shape[-1])
            weight_flat = weight.contiguous()
            y, rsqrt = _NEON_MOD.rmsnorm_forward_neon(x_flat, weight_flat, eps)
            ctx.save_for_backward(x, weight, rsqrt)
            ctx.orig_shape = orig_shape
            return y.view(orig_shape)
        elif _METAL_MOD is not None and x.device.type == "mps" and x.dtype == torch.float32:
            orig_shape = x.shape
            x_flat = x.contiguous().view(-1, orig_shape[-1])
            weight_flat = weight.contiguous()
            y, rsqrt = _METAL_MOD.rmsnorm_forward_mps(x_flat, weight_flat, eps)
            ctx.save_for_backward(x, weight, rsqrt)
            ctx.orig_shape = orig_shape
            return y.view(orig_shape)
            
        # Fallback to standard PyTorch eager execution (avoid custom autograd for eager fallback!)
        raise RuntimeError("FusedRMSNormFunction.apply called without available kernel.")

    @staticmethod
    def backward(ctx, grad_y):
        x, weight, rsqrt = ctx.saved_tensors
        eps = ctx.eps
        
        if grad_y.device.type == "cuda" and HAS_TRITON:
            grad_x, grad_w = _triton_kernels.rmsnorm_backward(grad_y, x, weight, rsqrt, eps)
            return grad_x, grad_w, None
        elif _NEON_MOD is not None and grad_y.device.type == "cpu" and grad_y.dtype == torch.float32:
            grad_out_flat = grad_y.contiguous().view(-1, ctx.orig_shape[-1])
            x_flat = x.contiguous().view(-1, ctx.orig_shape[-1])
            weight_flat = weight.contiguous()
            grad_x, grad_w = _NEON_MOD.rmsnorm_backward_neon(grad_out_flat, x_flat, weight_flat, rsqrt)
            return grad_x.view(ctx.orig_shape), grad_w, None
        elif _METAL_MOD is not None and grad_y.device.type == "mps" and grad_y.dtype == torch.float32:
            grad_out_flat = grad_y.contiguous().view(-1, ctx.orig_shape[-1])
            x_flat = x.contiguous().view(-1, ctx.orig_shape[-1])
            weight_flat = weight.contiguous()
            grad_x, grad_w = _METAL_MOD.rmsnorm_backward_mps(grad_out_flat, x_flat, weight_flat, rsqrt)
            return grad_x.view(ctx.orig_shape), grad_w, None

        raise RuntimeError("RMSNorm backward kernel missing for device.")


def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Functional interface for Fused RMSNorm."""
    device = x.device
    if (device.type == "cuda" and HAS_TRITON) or (_NEON_MOD is not None and device.type == "cpu" and x.dtype == torch.float32) or (_METAL_MOD is not None and device.type == "mps" and x.dtype == torch.float32):
        return FusedRMSNormFunction.apply(x, weight, eps)
    
    # Fallback to standard PyTorch eager execution to avoid Python autograd overhead
    mean_sq = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(mean_sq + eps) * weight


class FusedRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_rmsnorm(x, self.weight, self.eps)


# ----------------------------------------------------------------------------
# 2. Fused SwiGLU Autograd Function
# ----------------------------------------------------------------------------

class FusedSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        if gate.device.type == "cuda" and HAS_TRITON:
            out = _triton_kernels.swiglu_forward(gate, up)
            ctx.save_for_backward(gate, up)
            return out
        elif _NEON_MOD is not None and gate.device.type == "cpu" and gate.dtype == torch.float32:
            out = _NEON_MOD.swiglu_forward_neon(gate, up)
            ctx.save_for_backward(gate, up)
            return out
        elif _METAL_MOD is not None and gate.device.type == "mps" and gate.dtype == torch.float32:
            out = _METAL_MOD.swiglu_forward_mps(gate, up)
            ctx.save_for_backward(gate, up)
            return out
            
        raise RuntimeError("FusedSwiGLUFunction.apply called without available kernel.")

    @staticmethod
    def backward(ctx, grad_y):
        gate, up = ctx.saved_tensors
        grad_y = grad_y.contiguous()
        gate = gate.contiguous()
        up = up.contiguous()
        
        if grad_y.device.type == "cuda" and HAS_TRITON:
            grad_gate, grad_up = _triton_kernels.swiglu_backward(grad_y, gate, up)
            return grad_gate, grad_up
        elif _NEON_MOD is not None and grad_y.device.type == "cpu" and grad_y.dtype == torch.float32:
            grad_gate, grad_up = _NEON_MOD.swiglu_backward_neon(grad_y, gate, up)
            return grad_gate, grad_up
        elif _METAL_MOD is not None and grad_y.device.type == "mps" and grad_y.dtype == torch.float32:
            grad_gate, grad_up = _METAL_MOD.swiglu_backward_mps(grad_y, gate, up)
            return grad_gate, grad_up

        raise RuntimeError("SwiGLU backward kernel missing for device.")


def fused_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Functional interface for Fused SwiGLU activation."""
    device = gate.device
    
    # Avoid Python autograd wrapper unless we actually have a compiled C++ or Triton kernel active for this device
    # Note: We intentionally bypass _NEON_MOD for SwiGLU on CPU because PyTorch's native C++ F.silu is highly 
    # optimized, faster, and perfectly accurate, whereas our custom NEON kernel uses a less precise exp approximation.
    if (device.type == "cuda" and HAS_TRITON) or (_METAL_MOD is not None and device.type == "mps" and gate.dtype == torch.float32):
        return FusedSwiGLUFunction.apply(gate, up)
        
    # Standard PyTorch eager fallback (native C++ autograd)
    return F.silu(gate) * up


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
    def forward(ctx: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False, sliding_window: Optional[int] = None) -> torch.Tensor:
        device = q.device
        if device.type == "cuda" and HAS_TRITON:
            out = triton_fused_sdpa_forward(q, k, v, is_causal=is_causal, sliding_window=sliding_window)
        else:
            if sliding_window is not None and is_causal:
                T = q.size(2)
                causal_mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
                window_mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril(diagonal=-sliding_window)
                attn_mask = causal_mask & ~window_mask
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            else:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        ctx.save_for_backward(q, k, v)
        ctx.is_causal = is_causal
        ctx.sliding_window = sliding_window
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        q, k, v = ctx.saved_tensors
        # Using PyTorch's highly optimized SDPA backward for exact parity
        with torch.enable_grad():
            q_ = q.detach().requires_grad_(True)
            k_ = k.detach().requires_grad_(True)
            v_ = v.detach().requires_grad_(True)
            if ctx.sliding_window is not None and ctx.is_causal:
                T = q_.size(2)
                causal_mask = torch.ones(T, T, dtype=torch.bool, device=q_.device).tril()
                window_mask = torch.ones(T, T, dtype=torch.bool, device=q_.device).tril(diagonal=-ctx.sliding_window)
                attn_mask = causal_mask & ~window_mask
                out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=attn_mask)
            else:
                out = F.scaled_dot_product_attention(q_, k_, v_, is_causal=ctx.is_causal)
            out.backward(grad_output)
        return q_.grad, k_.grad, v_.grad, None, None

def fused_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False, sliding_window: Optional[int] = None) -> torch.Tensor:
    """Functional interface for Fused SDPA (FlashAttention) with Sliding Window Support."""
    return FusedSDPAFunction.apply(q, k, v, is_causal, sliding_window)


# ----------------------------------------------------------------------------
# 4. Fused Rotary Position Embeddings (RoPE) Autograd Function
# ----------------------------------------------------------------------------

class FusedRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, freqs_cis):
        ctx.save_for_backward(freqs_cis)
        if _METAL_MOD is not None and x.device.type == "mps" and x.dtype == torch.float32:
            freqs_real = torch.view_as_real(freqs_cis).contiguous()
            out = _METAL_MOD.apply_rope_mps(x.contiguous(), freqs_real, True)
            return out
        raise RuntimeError("FusedRoPEFunction called without available kernel.")

    @staticmethod
    def backward(ctx, grad_out):
        freqs_cis, = ctx.saved_tensors
        if _METAL_MOD is not None and grad_out.device.type == "mps" and grad_out.dtype == torch.float32:
            freqs_real = torch.view_as_real(freqs_cis).contiguous()
            grad_x = _METAL_MOD.apply_rope_mps(grad_out.contiguous(), freqs_real, False)
            return grad_x, None
        raise RuntimeError("FusedRoPEFunction backward kernel missing for device.")

def fused_apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    """Functional interface for Fused RoPE."""
    B, n_head, T, head_dim = x.shape
    freqs_slice = freqs_cis[start_pos : start_pos + T, :]
    
    if _METAL_MOD is not None and x.device.type == "mps" and x.dtype == torch.float32:
        return FusedRoPEFunction.apply(x, freqs_slice)
        
    # Standard PyTorch eager fallback
    orig_dtype = x.dtype
    x_complex = torch.view_as_complex(x.float().reshape(B, n_head, T, -1, 2))
    freqs_slice_expanded = freqs_slice.view(1, 1, T, -1).to(x.device)
    x_rotated = torch.view_as_real(x_complex * freqs_slice_expanded).reshape(B, n_head, T, head_dim)
    return x_rotated.to(orig_dtype)
