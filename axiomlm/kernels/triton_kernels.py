"""
OpenAI Triton Fused Kernel Implementations for CUDA GPUs (Axiom-LM Engine).

This module provides fused JIT-compiled kernels for:
1. Fused RMSNorm (Forward & Backward)
2. Fused SwiGLU (Forward & Backward)

These kernels fuse reductions, nonlinear activations, and arithmetic operations
into single SRAM passes, avoiding DRAM roundtrips on NVIDIA Tensor Core GPUs.
"""

from typing import Tuple, Optional, Any
import torch

# Type-checker stubs for platforms without NVIDIA Triton (e.g. Apple Silicon / macOS)
triton: Any = None
tl: Any = None
HAS_TRITON: bool = False

try:
    import triton as _triton  # type: ignore
    import triton.language as _tl  # type: ignore
    triton = _triton
    tl = _tl
    HAS_TRITON = True
except (ImportError, ModuleNotFoundError, Exception):
    class _TritonLanguageStub:
        constexpr: Any = int
        def __getattr__(self, name: str) -> Any:
            return None

    class _TritonStub:
        def __getattr__(self, name: str) -> Any:
            return None
        def jit(self, fn: Any = None, **kwargs: Any) -> Any:
            if fn is not None:
                return fn
            return lambda f: f

    triton = _TritonStub()
    tl = _TritonLanguageStub()
    HAS_TRITON = False


# ----------------------------------------------------------------------------
# 1. Fused RMSNorm Triton GPU Kernels
# ----------------------------------------------------------------------------

@triton.jit
def _rmsnorm_forward_kernel(
    X_ptr: Any,          # Pointer to input tensor X (M, N)
    Y_ptr: Any,          # Pointer to output tensor Y (M, N)
    W_ptr: Any,          # Pointer to weights W (N,)
    R_ptr: Any,          # Pointer to rsqrt cache (M,)
    stride_x_row: int,   # Stride between rows of X
    stride_y_row: int,   # Stride between rows of Y
    N: int,              # Hidden dimension D
    eps: float,          # Epsilon for numerical stability
    BLOCK_SIZE: Any,     # Tile size (tl.constexpr)
) -> None:
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


@triton.jit
def _rmsnorm_backward_kernel(
    DY_ptr: Any,         # Pointer to incoming gradient dY (M, N)
    X_ptr: Any,          # Pointer to forward input X (M, N)
    W_ptr: Any,          # Pointer to weights W (N,)
    R_ptr: Any,          # Pointer to cached rsqrt (M,)
    DX_ptr: Any,         # Pointer to output input grad dX (M, N)
    DW_ptr: Any,         # Pointer to output weight grad dW (M, N) - partials
    stride_dy_row: int,
    stride_x_row: int,
    stride_dx_row: int,
    stride_dw_row: int,
    N: int,
    BLOCK_SIZE: Any,
) -> None:
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


# ----------------------------------------------------------------------------
# 2. Fused SwiGLU Triton GPU Kernels
# ----------------------------------------------------------------------------

@triton.jit
def _swiglu_forward_kernel(
    Gate_ptr: Any,       # Pointer to Gate tensor (N,)
    Up_ptr: Any,         # Pointer to Up tensor (N,)
    Out_ptr: Any,        # Pointer to Output tensor (N,)
    N_elements: int,     # Total element count
    BLOCK_SIZE: Any,
) -> None:
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


@triton.jit
def _swiglu_backward_kernel(
    DY_ptr: Any,         # Upstream grad dY (N,)
    Gate_ptr: Any,       # Gate input (N,)
    Up_ptr: Any,         # Up input (N,)
    DGate_ptr: Any,      # Output dGate (N,)
    DUp_ptr: Any,        # Output dUp (N,)
    N_elements: int,
    BLOCK_SIZE: Any,
) -> None:
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

# ----------------------------------------------------------------------------
# 3. Fused Scaled Dot-Product Attention (Tiled SDPA)
# ----------------------------------------------------------------------------

@triton.jit
def _fused_sdpa_forward_kernel(
    Q, K, V, sm_scale,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Offsets
    qkv_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qkv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qkv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    q = tl.load(Q_block_ptr)
    q = (q * sm_scale).to(tl.float16)

    # loop over K, V
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(tl.advance(K_block_ptr, (0, start_n)))
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        
        if IS_CAUSAL:
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = start_n + tl.arange(0, BLOCK_N)
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
            
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp(qk)
        l_ij = tl.sum(p, 1)
        
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        acc = acc * alpha[:, None]
        v = tl.load(tl.advance(V_block_ptr, (start_n, 0)))
        p = p.to(tl.float16)
        acc += tl.dot(p, v)
        
        m_i = m_ij
        
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.dtype.element_ty))

def triton_fused_sdpa_forward(q, k, v, is_causal=False):
    # q, k, v are shape [B, H, N_CTX, D_HEAD]
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_DMODEL = q.shape[-1]
    
    sm_scale = 1.0 / (BLOCK_DMODEL ** 0.5)
    
    out = torch.empty_like(q)
    
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    
    _fused_sdpa_forward_kernel[grid](
        q, k, v, sm_scale,
        out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        q.shape[0], q.shape[1], q.shape[2],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
        IS_CAUSAL=is_causal,
        num_warps=4,
        num_stages=2,
    )
    return out
