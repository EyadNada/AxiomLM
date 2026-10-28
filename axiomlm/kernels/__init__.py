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
    FusedSDPAFunction,
    fused_sdpa,
    flash_attention_mps,
    FusedRoPEFunction,
    fused_apply_rope,
    FusedCrossEntropyFunction,
    fused_cross_entropy,
    FusedSwiGLUMLP,
    _NEON_MOD,
    _METAL_MOD,
    HAS_TRITON,
)

__all__ = [
    "FusedRMSNormFunction",
    "fused_rmsnorm",
    "FusedRMSNorm",
    "FusedSwiGLUFunction",
    "fused_swiglu",
    "FusedSDPAFunction",
    "fused_sdpa",
    "flash_attention_mps",
    "FusedRoPEFunction",
    "fused_apply_rope",
    "FusedCrossEntropyFunction",
    "fused_cross_entropy",
    "FusedSwiGLUMLP",
    "_NEON_MOD",
    "_METAL_MOD",
    "HAS_TRITON",
]
