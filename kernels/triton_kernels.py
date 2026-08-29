"""
OpenAI Triton Fused Kernel Implementations for CUDA GPUs (Axiom-LM Engine).

This module provides fused JIT-compiled kernels for:
1. Fused RMSNorm (Forward & Backward)
2. Fused SwiGLU (Forward & Backward)

These kernels fuse reductions, nonlinear activations, and arithmetic operations
into single SRAM passes, avoiding DRAM roundtrips on NVIDIA Tensor Core GPUs.
"""

# pyright: reportMissingImports=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportOptionalCall=false

from typing import Tuple, Optional, Any
import torch

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
    HAS_TRITON = True
except (ImportError, ModuleNotFoundError):
    HAS_TRITON = False
    triton = None  # type: ignore
    tl = None      # type: ignore


if HAS_TRITON:
    @triton.jit  # type: ignore
    def _rmsnorm_forward_kernel(
        X_ptr,          # Pointer to input tensor X (M, N)
        Y_ptr,          # Pointer to output tensor Y (M, N)
        W_ptr,          # Pointer to weights W (N,)
        R_ptr,          # Pointer to rsqrt cache (M,)
        stride_x_row,   # Stride between rows of X
        stride_y_row,   # Stride between rows of Y
        N,              # Hidden dimension D
        eps,            # Epsilon for numerical stability
        BLOCK_SIZE: tl.constexpr,  # type: ignore
    ):
        row_idx = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        # Load row of X
        x_offsets = row_idx * stride_x_row + cols
        x = tl.load(X_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        # Compute mean square: mean(x^2)
        x_sq = x * x
        mean_sq = tl.sum(x_sq, axis=0) / N
        rsqrt_val = tl.rsqrt(mean_sq + eps)

        # Save rsqrt value for backward pass
        if R_ptr is not None:
            tl.store(R_ptr + row_idx, rsqrt_val)

        # Fused scale and affine transform
        y = x * rsqrt_val * w

        # Store result
        y_offsets = row_idx * stride_y_row + cols
        tl.store(Y_ptr + y_offsets, y, mask=mask)

    @triton.jit  # type: ignore
    def _rmsnorm_backward_kernel(
        DY_ptr,         # Pointer to incoming gradient dY (M, N)
        X_ptr,          # Pointer to forward input X (M, N)
        W_ptr,          # Pointer to weights W (N,)
        R_ptr,          # Pointer to cached rsqrt (M,)
        DX_ptr,         # Pointer to output input grad dX (M, N)
        DW_ptr,         # Pointer to output weight grad dW (M, N) - partials
        stride_dy_row,
        stride_x_row,
        stride_dx_row,
        stride_dw_row,
        N,
        BLOCK_SIZE: tl.constexpr,  # type: ignore
    ):
        row_idx = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        # Load dY, X, W, and cached rsqrt
        dy = tl.load(DY_ptr + row_idx * stride_dy_row + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X_ptr + row_idx * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        rsqrt_val = tl.load(R_ptr + row_idx).to(tl.float32)

        # x_norm = x * rsqrt_val
        x_norm = x * rsqrt_val

        # dW partial = dy * x_norm
        dw = dy * x_norm
        tl.store(DW_ptr + row_idx * stride_dw_row + cols, dw, mask=mask)

        # dX computation:
        # dX = rsqrt_val * (dy * w - x_norm * mean(dy * w * x_norm))
        dy_w = dy * w
        inner_prod = tl.sum(dy_w * x_norm, axis=0) / N
        dx = rsqrt_val * (dy_w - x_norm * inner_prod)

        tl.store(DX_ptr + row_idx * stride_dx_row + cols, dx, mask=mask)

    @triton.jit  # type: ignore
    def _swiglu_forward_kernel(
        Gate_ptr,       # Pointer to Gate tensor (N,)
        Up_ptr,         # Pointer to Up tensor (N,)
        Out_ptr,        # Pointer to Output tensor (N,)
        N_elements,     # Total element count
        BLOCK_SIZE: tl.constexpr,  # type: ignore
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_elements

        g = tl.load(Gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(Up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        # SiLU(g) = g * sigmoid(g) = g / (1 + exp(-g))
        sig_g = tl.sigmoid(g)
        silu_g = g * sig_g
        out = silu_g * u

        tl.store(Out_ptr + offsets, out, mask=mask)

    @triton.jit  # type: ignore
    def _swiglu_backward_kernel(
        DY_ptr,         # Upstream grad dY (N,)
        Gate_ptr,       # Gate input (N,)
        Up_ptr,         # Up input (N,)
        DGate_ptr,      # Output dGate (N,)
        DUp_ptr,        # Output dUp (N,)
        N_elements,
        BLOCK_SIZE: tl.constexpr,  # type: ignore
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_elements

        dy = tl.load(DY_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(Gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(Up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        sig_g = tl.sigmoid(g)
        silu_g = g * sig_g

        # dUp = dY * silu(g)
        dup = dy * silu_g

        # dGate = dY * u * d(silu(g))/dg = dY * u * (sig_g * (1 + g * (1 - sig_g)))
        dsilu_dg = sig_g * (1.0 + g * (1.0 - sig_g))
        dgate = dy * u * dsilu_dg

        tl.store(DUp_ptr + offsets, dup, mask=mask)
        tl.store(DGate_ptr + offsets, dgate, mask=mask)


def triton_rmsnorm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton implementation of RMSNorm forward pass."""
    if not HAS_TRITON or triton is None:
        raise RuntimeError("OpenAI Triton is not installed or available on this system.")
    M = x.numel() // x.size(-1)
    N = x.size(-1)
    x_2d = x.contiguous().view(M, N)
    y_2d = torch.empty_like(x_2d)
    rsqrt_cache = torch.empty(M, device=x.device, dtype=torch.float32)

    BLOCK_SIZE = triton.next_power_of_2(N)
    _rmsnorm_forward_kernel[(M,)](
        x_2d, y_2d, weight, rsqrt_cache,
        x_2d.stride(0), y_2d.stride(0),
        N, eps, BLOCK_SIZE=BLOCK_SIZE
    )
    return y_2d.view_as(x), rsqrt_cache


def triton_swiglu_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Triton implementation of SwiGLU forward pass."""
    if not HAS_TRITON or triton is None:
        raise RuntimeError("OpenAI Triton is not installed or available on this system.")
    assert gate.shape == up.shape
    out = torch.empty_like(gate)
    N = gate.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _swiglu_forward_kernel[grid](gate, up, out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out
