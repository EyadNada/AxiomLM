"""
Axiom-LM Fast Register-Level Quantized GEMV and Weight Packing.
Optimized for Apple Silicon GPU SIMD-groups with group size G=64.
Supports:
- Q4_G64: 4.5 bits/weight (8 x 4-bit nibbles in uint32 + half2 scale/bias)
- Q8_G64: 8.5 bits/weight (4 x 8-bit bytes in uint32 + half2 scale/bias)
"""

from typing import Tuple
import torch
import torch.nn.functional as F
from .build_kernels import load_metal_module

_METAL_MOD = load_metal_module()


def pack_linear_q4_g64(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Packs a 2D weight matrix (N, K) where K is a multiple of 64 into:
      w_packed: (N, K // 8) of dtype torch.int32
      scales_biases: (N, K // 64, 2) of dtype torch.float16

    Affine quantization formula:
      w ~= q * scale + bias,  q in [0, 15]
    """
    assert weight.ndim == 2, f"Expected 2D weight matrix, got shape {weight.shape}"
    N, K = weight.shape
    assert K % 64 == 0, f"K ({K}) must be an exact multiple of group size 64"
    device = weight.device

    w_groups = weight.view(N, K // 64, 64)
    w_min = w_groups.amin(dim=-1, keepdim=True)
    w_max = w_groups.amax(dim=-1, keepdim=True)

    denom = torch.clamp(w_max - w_min, min=1e-8)
    scale = (denom / 15.0).to(torch.float16)
    bias = w_min.to(torch.float16)

    q = torch.clamp(
        torch.round((w_groups - bias.float()) / torch.clamp(scale.float(), min=1e-8)),
        0,
        15,
    ).to(torch.int32)

    q_sub = q.view(N, K // 8, 8)
    packed = torch.zeros((N, K // 8), dtype=torch.int32, device=device)
    for shift_idx in range(8):
        packed |= (q_sub[:, :, shift_idx] & 0xF) << (shift_idx * 4)

    scales_biases = torch.stack([scale.squeeze(-1), bias.squeeze(-1)], dim=-1)
    return packed, scales_biases


def dequantize_q4_g64(
    w_packed: torch.Tensor, scales_biases: torch.Tensor, K: int
) -> torch.Tensor:
    """Dequantizes Q4_G64 packed representation back into FP16 (N, K) tensor."""
    N = w_packed.shape[0]
    device = w_packed.device

    q_unpacked = torch.empty((N, K // 8, 8), dtype=torch.float32, device=device)
    for shift_idx in range(8):
        q_unpacked[:, :, shift_idx] = ((w_packed >> (shift_idx * 4)) & 0xF).float()

    q = q_unpacked.view(N, K // 64, 64)
    scale = scales_biases[:, :, 0:1].float()
    bias = scales_biases[:, :, 1:2].float()

    return (q * scale + bias).view(N, K).to(torch.float16)


def pack_linear_q8_g64(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Packs a 2D weight matrix (N, K) where K is a multiple of 64 into:
      w_packed: (N, K // 4) of dtype torch.int32
      scales_biases: (N, K // 64, 2) of dtype torch.float16

    Affine quantization formula:
      w ~= q * scale + bias,  q in [0, 255]
    """
    assert weight.ndim == 2, f"Expected 2D weight matrix, got shape {weight.shape}"
    N, K = weight.shape
    assert K % 64 == 0, f"K ({K}) must be an exact multiple of group size 64"
    device = weight.device

    w_groups = weight.view(N, K // 64, 64)
    w_min = w_groups.amin(dim=-1, keepdim=True)
    w_max = w_groups.amax(dim=-1, keepdim=True)

    denom = torch.clamp(w_max - w_min, min=1e-8)
    scale = (denom / 255.0).to(torch.float16)
    bias = w_min.to(torch.float16)

    q = torch.clamp(
        torch.round((w_groups - bias.float()) / torch.clamp(scale.float(), min=1e-8)),
        0,
        255,
    ).to(torch.int32)

    q_sub = q.view(N, K // 4, 4)
    packed = torch.zeros((N, K // 4), dtype=torch.int32, device=device)
    for shift_idx in range(4):
        packed |= (q_sub[:, :, shift_idx] & 0xFF) << (shift_idx * 8)

    scales_biases = torch.stack([scale.squeeze(-1), bias.squeeze(-1)], dim=-1)
    return packed, scales_biases


def dequantize_q8_g64(
    w_packed: torch.Tensor, scales_biases: torch.Tensor, K: int
) -> torch.Tensor:
    """Dequantizes Q8_G64 packed representation back into FP16 (N, K) tensor."""
    N = w_packed.shape[0]
    device = w_packed.device

    q_unpacked = torch.empty((N, K // 4, 4), dtype=torch.float32, device=device)
    for shift_idx in range(4):
        q_unpacked[:, :, shift_idx] = ((w_packed >> (shift_idx * 8)) & 0xFF).float()

    q = q_unpacked.view(N, K // 64, 64)
    scale = scales_biases[:, :, 0:1].float()
    bias = scales_biases[:, :, 1:2].float()

    return (q * scale + bias).view(N, K).to(torch.float16)


def qmv_q4_g64(
    x: torch.Tensor, w_packed: torch.Tensor, scales_biases: torch.Tensor
) -> torch.Tensor:
    """
    Fast matrix-vector multiply y = x @ W^T with 4-bit weights on Apple Silicon Metal GPU.
    """
    if x.device.type == "mps" and _METAL_MOD is not None:
        return _METAL_MOD.qmv_q4_g64_mps(x, w_packed, scales_biases)

    # Fallback dequantized matmul
    K = x.size(-1)
    w_dq = dequantize_q4_g64(w_packed, scales_biases, K).to(x.device)
    return F.linear(x, w_dq)


def qmv_q8_g64(
    x: torch.Tensor, w_packed: torch.Tensor, scales_biases: torch.Tensor
) -> torch.Tensor:
    """
    Fast matrix-vector multiply y = x @ W^T with 8-bit weights on Apple Silicon Metal GPU.
    """
    if x.device.type == "mps" and _METAL_MOD is not None:
        return _METAL_MOD.qmv_q8_g64_mps(x, w_packed, scales_biases)

    # Fallback dequantized matmul
    K = x.size(-1)
    w_dq = dequantize_q8_g64(w_packed, scales_biases, K).to(x.device)
    return F.linear(x, w_dq)
