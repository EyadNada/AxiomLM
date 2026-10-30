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
    gqa_flash_decode_mps,
    FusedRoPEFunction,
    fused_apply_rope,
    FusedCrossEntropyFunction,
    fused_cross_entropy,
    FusedSwiGLUMLP,
    _NEON_MOD,
    _METAL_MOD,
    HAS_TRITON,
)
from .quant import (
    pack_linear_q4_g64,
    dequantize_q4_g64,
    pack_linear_q8_g64,
    dequantize_q8_g64,
    qmv_q4_g64,
    qmm_q4_g64,
    qmv_q8_g64,
    qmm_q8_g64,
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
    "gqa_flash_decode_mps",
    "FusedRoPEFunction",
    "fused_apply_rope",
    "FusedCrossEntropyFunction",
    "fused_cross_entropy",
    "FusedSwiGLUMLP",
    "_NEON_MOD",
    "_METAL_MOD",
    "HAS_TRITON",
    "pack_linear_q4_g64",
    "dequantize_q4_g64",
    "pack_linear_q8_g64",
    "dequantize_q8_g64",
    "qmv_q4_g64",
    "qmm_q4_g64",
    "qmv_q8_g64",
    "qmm_q8_g64",
]
