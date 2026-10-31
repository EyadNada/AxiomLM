"""
Axiom-LM Custom Low-Level GPU & SIMD Kernel Suite.
Exports FusedRMSNorm, FusedSwiGLU, and their respective functional APIs.
"""

import torch

from .ops import (
    _METAL_MOD,
    _NEON_MOD,
    HAS_TRITON,
    FusedCrossEntropyFunction,
    FusedRMSNorm,
    FusedRMSNormFunction,
    FusedRoPEFunction,
    FusedSDPAFunction,
    FusedSwiGLUFunction,
    FusedSwiGLUMLP,
    flash_attention_mps,
    fused_apply_rope,
    fused_cross_entropy,
    fused_rmsnorm,
    fused_sdpa,
    fused_swiglu,
    gqa_flash_decode_mps,
)
from .quant import (
    dequantize_q4_g64,
    dequantize_q8_g64,
    pack_linear_q4_g64,
    pack_linear_q8_g64,
    qmm_q4_g64,
    qmm_q8_g64,
    qmv_q4_g64,
    qmv_q8_g64,
)

_IS_MPS_FUNCTIONAL = None


def is_mps_available() -> bool:
    """
    Bulletproof check for functional Apple Metal Performance Shaders (MPS).
    Returns False on headless CI VMs / containers / sandboxes where PyTorch raises:
      RuntimeError: MPS backend is unavailable: no supported Metal GPU was found
    """
    global _IS_MPS_FUNCTIONAL
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return False

    if _IS_MPS_FUNCTIONAL is not None:
        return _IS_MPS_FUNCTIONAL

    try:
        t = torch.zeros(1, device="mps")
        del t
        _IS_MPS_FUNCTIONAL = True
        return True
    except (RuntimeError, Exception):
        _IS_MPS_FUNCTIONAL = False
        return False


__all__ = [
    "HAS_TRITON",
    "_METAL_MOD",
    "_NEON_MOD",
    "FusedCrossEntropyFunction",
    "FusedRMSNorm",
    "FusedRMSNormFunction",
    "FusedRoPEFunction",
    "FusedSDPAFunction",
    "FusedSwiGLUFunction",
    "FusedSwiGLUMLP",
    "dequantize_q4_g64",
    "dequantize_q8_g64",
    "flash_attention_mps",
    "fused_apply_rope",
    "fused_cross_entropy",
    "fused_rmsnorm",
    "fused_sdpa",
    "fused_swiglu",
    "gqa_flash_decode_mps",
    "is_mps_available",
    "pack_linear_q4_g64",
    "pack_linear_q8_g64",
    "qmm_q4_g64",
    "qmm_q8_g64",
    "qmv_q4_g64",
    "qmv_q8_g64",
]
