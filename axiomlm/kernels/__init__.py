"""
Axiom-LM Custom Low-Level GPU & SIMD Kernel Suite.
Exports FusedRMSNorm, FusedSwiGLU, and their respective functional APIs.
"""

from .ops import (
    FusedRMSNormFunction,
    fused_rmsnorm,
    FusedRMSNorm,
    FusedSwiGLUFunction,
    fused_swiglu,
    fused_sdpa,
    FusedSwiGLUMLP,
    _NEON_MOD,
    HAS_TRITON,
)

__all__ = [
    "FusedRMSNormFunction",
    "fused_rmsnorm",
    "FusedRMSNorm",
    "FusedSwiGLUFunction",
    "fused_swiglu",
    "FusedSwiGLUMLP",
    "_NEON_MOD",
    "HAS_TRITON",
]
