"""
Axiom-LM Fast Register-Level Quantized GEMV and Weight Packing.
Optimized for Apple Silicon GPU SIMD-groups with group size G=64.
Supports:
- Q4_G64: 4.5 bits/weight (8 x 4-bit nibbles in uint32 + half2 scale/bias)
- Q8_G64: 8.5 bits/weight (4 x 8-bit bytes in uint32 + half2 scale/bias)
"""

import torch
import torch.nn.functional as F

from .build_kernels import load_metal_module

_METAL_MOD = load_metal_module()


def pack_linear_q4_g64(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


def pack_linear_q8_g64(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

    packed = torch.empty((N, K // 4), dtype=torch.int32, device=device)
    scales_biases = torch.empty((N, K // 64, 2), dtype=torch.float16, device=device)
    shifts = torch.tensor([0, 8, 16, 24], dtype=torch.int32, device=device)
    chunk_rows = 16384
    for r0 in range(0, N, chunk_rows):
        r1 = min(r0 + chunk_rows, N)
        wg = weight[r0:r1].view(r1 - r0, K // 64, 64)
        w_min = wg.amin(dim=-1, keepdim=True)
        w_max = wg.amax(dim=-1, keepdim=True)

        denom = torch.clamp(w_max - w_min, min=1e-8)
        sc = (denom.float() / 255.0).to(torch.float16)
        bi = w_min.to(torch.float16)

        q_c = torch.clamp(
            torch.round((wg.float() - bi.float()) / torch.clamp(sc.float(), min=1e-8)),
            0.0,
            255.0,
        ).to(torch.int32)

        q_sub = q_c.view(r1 - r0, K // 4, 4)
        packed[r0:r1] = ((q_sub & 0xFF) << shifts).sum(dim=-1, dtype=torch.int32)
        scales_biases[r0:r1, :, 0] = sc.squeeze(-1)
        scales_biases[r0:r1, :, 1] = bi.squeeze(-1)

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


def dequant_q8_rows(
    indices: torch.Tensor,
    w_packed: torch.Tensor,
    scales_biases: torch.Tensor,
) -> torch.Tensor:
    """
    Dequantizes rows of a Q8_G64 weight matrix specified by `indices`.
    indices: (B, T) or (N,)
    w_packed: (V, K // 4) int32
    scales_biases: (V, K // 64, 2) half
    Returns: (*indices.shape, K) of dtype float16
    """
    orig_shape = indices.shape
    flat_idx = indices.reshape(-1)
    sel_p = w_packed[flat_idx]
    sel_sb = scales_biases[flat_idx]
    num_tokens = flat_idx.size(0)
    K_packed = w_packed.size(1)
    K = K_packed * 4

    q_unpacked = torch.empty(
        (num_tokens, K_packed, 4), dtype=torch.float32, device=w_packed.device
    )
    for shift_idx in range(4):
        q_unpacked[:, :, shift_idx] = ((sel_p >> (shift_idx * 8)) & 0xFF).float()

    q = q_unpacked.view(num_tokens, K // 64, 64)
    scale = sel_sb[:, :, 0:1].float()
    bias = sel_sb[:, :, 1:2].float()

    out = (q * scale + bias).view(num_tokens, K).to(torch.float16)
    return out.view(*orig_shape, K)


def qmv_q4_g64(
    x: torch.Tensor, w_packed: torch.Tensor, scales_biases: torch.Tensor
) -> torch.Tensor:
    """
    Fast matrix-vector multiply y = x @ W^T with 4-bit weights on Apple Silicon Metal GPU.
    Automatically uses batched QMM kernel if M > 1 tokens are provided.
    """
    if x.device.type == "mps" and _METAL_MOD is not None:
        K = x.size(-1)
        M = x.numel() // K
        if M > 1:
            return _METAL_MOD.qmm_q4_g64_mps(x, w_packed, scales_biases)
        return _METAL_MOD.qmv_q4_g64_mps(x, w_packed, scales_biases)

    # Fallback dequantized matmul
    K = x.size(-1)
    w_dq = dequantize_q4_g64(w_packed, scales_biases, K).to(x.device)
    return F.linear(x, w_dq)


def qmm_q4_g64(
    x: torch.Tensor, w_packed: torch.Tensor, scales_biases: torch.Tensor
) -> torch.Tensor:
    """
    Batched matrix multiplication y = x @ W^T with 4-bit weights on Apple Silicon Metal GPU.
    Loads 4-bit weights from DRAM once and computes across M candidate tokens in registers.
    """
    if x.device.type == "mps" and _METAL_MOD is not None:
        return _METAL_MOD.qmm_q4_g64_mps(x, w_packed, scales_biases)

    K = x.size(-1)
    w_dq = dequantize_q4_g64(w_packed, scales_biases, K).to(x.device)
    return F.linear(x, w_dq)


def qmv_q8_g64(
    x: torch.Tensor, w_packed: torch.Tensor, scales_biases: torch.Tensor
) -> torch.Tensor:
    """
    Fast matrix-vector multiply y = x @ W^T with 8-bit weights on Apple Silicon Metal GPU.
    Automatically uses batched QMM kernel if M > 1 tokens are provided.
    """
    if x.device.type == "mps" and _METAL_MOD is not None:
        K = x.size(-1)
        M = x.numel() // K
        if M > 1:
            return _METAL_MOD.qmm_q8_g64_mps(x, w_packed, scales_biases)
        return _METAL_MOD.qmv_q8_g64_mps(x, w_packed, scales_biases)

    # Fallback dequantized matmul
    K = x.size(-1)
    w_dq = dequantize_q8_g64(w_packed, scales_biases, K).to(x.device)
    return F.linear(x, w_dq)


def qmm_q8_g64(
    x: torch.Tensor, w_packed: torch.Tensor, scales_biases: torch.Tensor
) -> torch.Tensor:
    """
    Batched matrix multiplication y = x @ W^T with 8-bit weights on Apple Silicon Metal GPU.
    Loads 8-bit weights from DRAM once and computes across M candidate tokens in registers.
    """
    if x.device.type == "mps" and _METAL_MOD is not None:
        return _METAL_MOD.qmm_q8_g64_mps(x, w_packed, scales_biases)

    K = x.size(-1)
    w_dq = dequantize_q8_g64(w_packed, scales_biases, K).to(x.device)
    return F.linear(x, w_dq)


def pack_linear_q4_marlin(
    weight: torch.Tensor,
    clip_search: bool = True,
    col_importance: torch.Tensor | None = None,
    grid_steps: int = 13,
    min_shrink: float = 0.72,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Packs a 2D weight matrix (N, K) where K is a multiple of 64 into the unified
    Apple Silicon MARLIN 64207531 + Ladder simdgroup_half8x8 register-aligned layout:
      w_marlin:  (ceil(N / 8), K // 64, 32, 2) of dtype torch.int32
      sb_marlin: (ceil(N / 8), K // 64, 8, 2)  of dtype torch.float16

    When `clip_search=True` (default), performs a chunked vectorized asymmetric
    shrinkage search minimizing importance-weighted group MSE in actual FP16
    (scale, bias) space with zero changes to the packed layout or Metal shaders.
    Set `clip_search=False` for exact bit-for-bit Round-To-Nearest parity with Q4_G64.
    """
    assert weight.ndim == 2, f"Expected 2D weight matrix, got shape {weight.shape}"
    N, K = weight.shape
    assert K % 64 == 0, f"K ({K}) must be an exact multiple of group size 64"
    device = weight.device

    N_tiles = (N + 7) // 8
    N_pad = N_tiles * 8
    K_groups = K // 64

    if N_pad != N:
        w_pad = torch.zeros((N_pad, K), dtype=weight.dtype, device=device)
        w_pad[:N] = weight
    else:
        w_pad = weight

    w_groups = w_pad.view(N_pad, K_groups, 64)

    if not clip_search and col_importance is None:
        w_min = w_groups.amin(dim=-1, keepdim=True)
        w_max = w_groups.amax(dim=-1, keepdim=True)
        denom = torch.clamp(w_max - w_min, min=1e-8)
        scale = (denom / 15.0).to(torch.float16)
        bias = w_min.to(torch.float16)
        q = torch.clamp(
            torch.round(
                (w_groups - bias.float()) / torch.clamp(scale.float(), min=1e-8)
            ),
            0,
            15,
        ).to(torch.int32)
    else:
        scale = torch.empty((N_pad, K_groups, 1), dtype=torch.float16, device=device)
        bias = torch.empty((N_pad, K_groups, 1), dtype=torch.float16, device=device)
        q = torch.empty((N_pad, K_groups, 64), dtype=torch.int32, device=device)

        if col_importance is not None:
            assert (
                col_importance.numel() == K
            ), f"Expected col_importance of length {K}, got {col_importance.numel()}"
            ig_full = col_importance.to(device=device, dtype=torch.float32).view(
                1, K_groups, 64
            )
            ig_full = ig_full * ig_full
            ig_full = ig_full / ig_full.mean(dim=-1, keepdim=True).clamp(min=1e-8)
        else:
            ig_full = None

        alphas = torch.linspace(1.0, min_shrink, steps=max(2, grid_steps)).tolist()
        chunk_rows = 8192
        for r0 in range(0, N_pad, chunk_rows):
            r1 = min(r0 + chunk_rows, N_pad)
            wg = w_groups[r0:r1].float()
            mn0 = wg.amin(dim=-1, keepdim=True)
            mx0 = wg.amax(dim=-1, keepdim=True)
            if ig_full is not None:
                mu = (wg * ig_full).mean(dim=-1, keepdim=True)
            else:
                mu = wg.mean(dim=-1, keepdim=True)

            best_err = torch.full(
                (r1 - r0, K_groups), float("inf"), dtype=torch.float32, device=device
            )
            cur_mn = mn0.clone()
            for a in alphas:
                cand_mn = mu + a * (mn0 - mu)
                sc = ((mx0 - cand_mn).clamp(min=1e-8) / 15.0).to(torch.float16).float()
                bi = cand_mn.to(torch.float16).float()
                qc = ((wg - bi) / sc.clamp(min=1e-8)).round().clamp(0.0, 15.0)
                diff = wg - (qc * sc + bi)
                err = (
                    (ig_full * (diff * diff)).sum(dim=-1)
                    if ig_full is not None
                    else (diff * diff).sum(dim=-1)
                )
                better = err < best_err
                best_err = torch.where(better, err, best_err)
                cur_mn = torch.where(better.unsqueeze(-1), cand_mn, cur_mn)

            best_err.fill_(float("inf"))
            best_sc = ((mx0 - cur_mn).clamp(min=1e-8) / 15.0).to(torch.float16)
            best_bi = cur_mn.to(torch.float16)
            for a in alphas:
                cand_mx = mu + a * (mx0 - mu)
                sc_h = ((cand_mx - cur_mn).clamp(min=1e-8) / 15.0).to(torch.float16)
                bi_h = cur_mn.to(torch.float16)
                sc = sc_h.float()
                bi = bi_h.float()
                qc = ((wg - bi) / sc.clamp(min=1e-8)).round().clamp(0.0, 15.0)
                diff = wg - (qc * sc + bi)
                err = (
                    (ig_full * (diff * diff)).sum(dim=-1)
                    if ig_full is not None
                    else (diff * diff).sum(dim=-1)
                )
                better = err < best_err
                best_err = torch.where(better, err, best_err)
                best_sc = torch.where(better.unsqueeze(-1), sc_h, best_sc)
                best_bi = torch.where(better.unsqueeze(-1), bi_h, best_bi)

            scale[r0:r1] = best_sc
            bias[r0:r1] = best_bi
            q[r0:r1] = (
                ((wg - best_bi.float()) / best_sc.float().clamp(min=1e-8))
                .round()
                .clamp(0.0, 15.0)
                .to(torch.int32)
            )

    lanes = torch.arange(32, device=device)
    r_idx = ((lanes & 16) >> 2) | ((lanes & 6) >> 1)
    c0_idx = ((lanes & 8) >> 1) | ((lanes & 1) << 1)
    c1_idx = c0_idx + 1

    # Reshape (N_tiles, c=8, K_groups, word=2, s_sub=4, r=8) -> (N_tiles, K_groups, c=8, r=8, word=2, s_sub=4)
    q_cr = q.view(N_tiles, 8, K_groups, 2, 4, 8).permute(0, 2, 1, 5, 3, 4)
    q_c0 = q_cr[:, :, c0_idx, r_idx, :, :]
    q_c1 = q_cr[:, :, c1_idx, r_idx, :, :]

    shifts = torch.tensor([0, 4, 8, 12], dtype=torch.int32, device=device)
    lo16 = ((q_c0 & 0xF) << shifts).sum(dim=-1, dtype=torch.int32)
    hi16 = (((q_c1 & 0xF) << shifts).sum(dim=-1, dtype=torch.int32)) << 16
    w_marlin = (lo16 | hi16).contiguous()

    sb_marlin = (
        torch.stack([scale.squeeze(-1), bias.squeeze(-1)], dim=-1)
        .view(N_tiles, 8, K_groups, 2)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    return w_marlin, sb_marlin


def dequantize_q4_marlin(
    w_marlin: torch.Tensor,
    sb_marlin: torch.Tensor,
    K: int,
    out_features: int | None = None,
) -> torch.Tensor:
    """Dequantizes MARLIN + Ladder Q4 layout back into an FP16 (N, K) tensor."""
    N_tiles, K_groups, _, _ = w_marlin.shape
    assert K_groups * 64 == K, f"Expected K={K_groups * 64}, got {K}"
    device = w_marlin.device

    lanes = torch.arange(32, device=device)
    r_idx = ((lanes & 16) >> 2) | ((lanes & 6) >> 1)
    c0_idx = ((lanes & 8) >> 1) | ((lanes & 1) << 1)
    c1_idx = c0_idx + 1

    shifts_lo = torch.tensor([0, 4, 8, 12], dtype=torch.int32, device=device)
    shifts_hi = torch.tensor([16, 20, 24, 28], dtype=torch.int32, device=device)
    w_exp = w_marlin.unsqueeze(-1)
    q_c0 = ((w_exp >> shifts_lo) & 0xF).float()
    q_c1 = ((w_exp >> shifts_hi) & 0xF).float()

    q_cr = torch.empty(
        (N_tiles, K_groups, 8, 8, 2, 4), dtype=torch.float32, device=device
    )
    q_cr[:, :, c0_idx, r_idx, :, :] = q_c0
    q_cr[:, :, c1_idx, r_idx, :, :] = q_c1

    q = q_cr.permute(0, 2, 1, 4, 5, 3).reshape(N_tiles * 8, K_groups, 64)
    sb = sb_marlin.permute(0, 2, 1, 3).reshape(N_tiles * 8, K_groups, 2)
    w_dq = (
        (q * sb[:, :, 0:1].float() + sb[:, :, 1:2].float())
        .reshape(N_tiles * 8, K)
        .to(torch.float16)
    )
    if out_features is not None and out_features < w_dq.shape[0]:
        w_dq = w_dq[:out_features]
    return w_dq


def qmv_q4_marlin(
    x: torch.Tensor,
    w_marlin: torch.Tensor,
    sb_marlin: torch.Tensor,
    out_features: int | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Register-aligned MARLIN + Ladder Q4 GEMV/GEMM y = x @ W^T (+ bias) on Apple Silicon.
    Automatically routes M > 1 to zero-barrier simdgroup_multiply_accumulate QMM kernels.
    """
    n_out = out_features if out_features is not None else -1
    if x.device.type == "mps" and x.dtype == torch.float16 and _METAL_MOD is not None:
        K = x.size(-1)
        M = x.numel() // K
        if M > 1:
            return _METAL_MOD.qmm_q4_marlin_mps(x, w_marlin, sb_marlin, n_out, bias)
        return _METAL_MOD.qmv_q4_marlin_mps(x, w_marlin, sb_marlin, n_out, bias)

    K = x.size(-1)
    w_dq = dequantize_q4_marlin(w_marlin, sb_marlin, K, out_features=out_features).to(
        device=x.device, dtype=x.dtype
    )
    b_dq = bias.to(device=x.device, dtype=x.dtype) if bias is not None else None
    return F.linear(x, w_dq, b_dq)


def qmm_q4_marlin(
    x: torch.Tensor,
    w_marlin: torch.Tensor,
    sb_marlin: torch.Tensor,
    out_features: int | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Zero-barrier register-direct MARLIN + Ladder Q4 batched GEMM y = x @ W^T (+ bias).
    Supports arbitrary batch/candidate sizes M >= 1 without Threadgroup shared memory.
    """
    n_out = out_features if out_features is not None else -1
    if x.device.type == "mps" and x.dtype == torch.float16 and _METAL_MOD is not None:
        return _METAL_MOD.qmm_q4_marlin_mps(x, w_marlin, sb_marlin, n_out, bias)

    K = x.size(-1)
    w_dq = dequantize_q4_marlin(w_marlin, sb_marlin, K, out_features=out_features).to(
        device=x.device, dtype=x.dtype
    )
    b_dq = bias.to(device=x.device, dtype=x.dtype) if bias is not None else None
    return F.linear(x, w_dq, b_dq)
