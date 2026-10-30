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

from typing import Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .build_kernels import load_neon_module, load_metal_module

# Load native compiled modules
_NEON_MOD = load_neon_module()
_METAL_MOD = load_metal_module()

try:
    from . import triton_kernels as _triton_kernels
    from .triton_kernels import HAS_TRITON
except ImportError:
    HAS_TRITON = False
    _triton_kernels = None


# ----------------------------------------------------------------------------
# 1. Fused RMSNorm Autograd Function
# ----------------------------------------------------------------------------


class FusedRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        ctx.eps = eps

        if x.device.type == "cuda" and HAS_TRITON:
            y, rsqrt = _triton_kernels.triton_rmsnorm_forward(x, weight, eps)
            ctx.save_for_backward(x, weight, rsqrt)
            return y
        elif (
            _NEON_MOD is not None
            and x.device.type == "cpu"
            and x.dtype == torch.float32
        ):
            orig_shape = x.shape
            x_flat = x.contiguous().view(-1, orig_shape[-1])
            weight_flat = weight.contiguous()
            y, rsqrt = _NEON_MOD.rmsnorm_forward_neon(x_flat, weight_flat, eps)
            ctx.save_for_backward(x, weight, rsqrt)
            ctx.orig_shape = orig_shape
            return y.view(orig_shape)
        elif (
            _METAL_MOD is not None
            and x.device.type == "mps"
            and x.dtype in (torch.float32, torch.float16)
        ):
            orig_shape = x.shape
            x_flat = x.contiguous().view(-1, orig_shape[-1])
            weight_flat = weight.to(x.dtype).contiguous()
            y, rsqrt = _METAL_MOD.rmsnorm_forward_mps(x_flat, weight_flat, eps)
            ctx.save_for_backward(x, weight, rsqrt)
            ctx.orig_shape = orig_shape
            return y.view(orig_shape)

        # Fallback to standard PyTorch eager execution (avoid custom autograd for eager fallback!)
        raise RuntimeError(
            "FusedRMSNormFunction.apply called without available kernel."
        )

    @staticmethod
    def backward(ctx, grad_y):
        x, weight, rsqrt = ctx.saved_tensors
        eps = ctx.eps

        if grad_y.device.type == "cuda" and HAS_TRITON:
            grad_x, grad_w = _triton_kernels.triton_rmsnorm_backward(
                grad_y, x, weight, rsqrt, eps
            )
            return grad_x, grad_w, None
        elif (
            _NEON_MOD is not None
            and grad_y.device.type == "cpu"
            and grad_y.dtype == torch.float32
        ):
            grad_out_flat = grad_y.contiguous().view(-1, ctx.orig_shape[-1])
            x_flat = x.contiguous().view(-1, ctx.orig_shape[-1])
            weight_flat = weight.contiguous()
            grad_x, grad_w = _NEON_MOD.rmsnorm_backward_neon(
                grad_out_flat, x_flat, weight_flat, rsqrt
            )
            return grad_x.view(ctx.orig_shape), grad_w, None
        elif (
            _METAL_MOD is not None
            and grad_y.device.type == "mps"
            and grad_y.dtype in (torch.float32, torch.float16)
        ):
            orig_dtype = grad_y.dtype
            grad_out_flat = grad_y.float().contiguous().view(-1, ctx.orig_shape[-1])
            x_flat = x.float().contiguous().view(-1, ctx.orig_shape[-1])
            weight_flat = weight.float().contiguous()
            grad_x, grad_w = _METAL_MOD.rmsnorm_backward_mps(
                grad_out_flat, x_flat, weight_flat, rsqrt.float()
            )
            return (
                grad_x.to(orig_dtype).view(ctx.orig_shape),
                grad_w.to(weight.dtype),
                None,
            )

        raise RuntimeError("RMSNorm backward kernel missing for device.")


def fused_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Functional interface for Fused RMSNorm."""
    device = x.device
    if (
        (device.type == "cuda" and HAS_TRITON)
        or (_NEON_MOD is not None and device.type == "cpu" and x.dtype == torch.float32)
        or (
            _METAL_MOD is not None
            and device.type == "mps"
            and x.dtype in (torch.float32, torch.float16)
        )
    ):
        if (
            not torch.is_grad_enabled()
            and _METAL_MOD is not None
            and device.type == "mps"
            and x.dtype in (torch.float32, torch.float16)
        ):
            x_c = x if x.is_contiguous() else x.contiguous()
            weight_flat = (
                weight
                if (weight.dtype == x.dtype and weight.is_contiguous())
                else weight.to(x.dtype).contiguous()
            )
            return _METAL_MOD.rmsnorm_infer_mps(x_c, weight_flat, eps)
        return FusedRMSNormFunction.apply(x, weight, eps)

    # Fallback to standard PyTorch eager execution to avoid Python autograd overhead
    mean_sq = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(mean_sq + eps) * weight


class FusedRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: Optional[int] = None,
        eps: float = 1e-6,
        dim: Optional[int] = None,
    ):
        super().__init__()
        resolved_dim = hidden_size if hidden_size is not None else dim
        if resolved_dim is None:
            raise ValueError("Specify hidden_size (or dim) for FusedRMSNorm.")
        self.weight = nn.Parameter(torch.ones(resolved_dim))
        self.eps = eps
        self._cached_weight: Optional[torch.Tensor] = None

    def train(self, mode: bool = True):
        self._cached_weight = None
        return super().train(mode)

    def _apply(self, fn):
        self._cached_weight = None
        return super()._apply(fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training and not torch.is_grad_enabled():
            if (
                self._cached_weight is None
                or self._cached_weight.device != x.device
                or self._cached_weight.dtype != x.dtype
            ):
                self._cached_weight = self.weight.to(
                    device=x.device, dtype=x.dtype
                ).contiguous()
            if (
                _METAL_MOD is not None
                and x.device.type == "mps"
                and x.dtype in (torch.float32, torch.float16)
                and x.is_contiguous()
            ):
                return _METAL_MOD.rmsnorm_infer_mps(x, self._cached_weight, self.eps)
            return fused_rmsnorm(x, self._cached_weight, self.eps)
        return fused_rmsnorm(x, self.weight, self.eps)


# ----------------------------------------------------------------------------
# 2. Fused SwiGLU Autograd Function
# ----------------------------------------------------------------------------


class FusedSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        if gate.device.type == "cuda" and HAS_TRITON:
            out = _triton_kernels.triton_swiglu_forward(gate, up)
            ctx.save_for_backward(gate, up)
            return out
        elif (
            _NEON_MOD is not None
            and gate.device.type == "cpu"
            and gate.dtype == torch.float32
        ):
            out = _NEON_MOD.swiglu_forward_neon(gate, up)
            ctx.save_for_backward(gate, up)
            return out
        elif (
            _METAL_MOD is not None
            and gate.device.type == "mps"
            and gate.dtype in (torch.float32, torch.float16)
        ):
            out = _METAL_MOD.swiglu_forward_mps(gate.contiguous(), up.contiguous())
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
            grad_gate, grad_up = _triton_kernels.triton_swiglu_backward(
                grad_y, gate, up
            )
            return grad_gate, grad_up
        elif (
            _NEON_MOD is not None
            and grad_y.device.type == "cpu"
            and grad_y.dtype == torch.float32
        ):
            grad_gate, grad_up = _NEON_MOD.swiglu_backward_neon(grad_y, gate, up)
            return grad_gate, grad_up
        elif (
            _METAL_MOD is not None
            and grad_y.device.type == "mps"
            and grad_y.dtype in (torch.float32, torch.float16)
        ):
            orig_dtype = grad_y.dtype
            grad_gate, grad_up = _METAL_MOD.swiglu_backward_mps(
                grad_y.float().contiguous(),
                gate.float().contiguous(),
                up.float().contiguous(),
            )
            return grad_gate.to(orig_dtype), grad_up.to(orig_dtype)

        raise RuntimeError("SwiGLU backward kernel missing for device.")


def fused_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Functional interface for Fused SwiGLU activation."""
    device = gate.device

    # Avoid Python autograd wrapper unless we actually have a compiled C++ or Triton kernel active for this device
    # Note: We intentionally bypass _NEON_MOD for SwiGLU on CPU because PyTorch's native C++ F.silu is highly
    # optimized, faster, and perfectly accurate, whereas our custom NEON kernel uses a less precise exp approximation.
    if (device.type == "cuda" and HAS_TRITON) or (
        _METAL_MOD is not None
        and device.type == "mps"
        and gate.dtype in (torch.float32, torch.float16)
    ):
        if (
            not torch.is_grad_enabled()
            and _METAL_MOD is not None
            and device.type == "mps"
        ):
            return _METAL_MOD.swiglu_forward_mps(gate.contiguous(), up.contiguous())
        return FusedSwiGLUFunction.apply(gate, up)

    # Standard PyTorch eager fallback (native C++ autograd)
    return F.silu(gate) * up


class FusedSwiGLUMLP(nn.Module):
    """
    Fused SwiGLU Multi-Layer Perceptron (LLaMA-3 spec) with custom fused kernel
    and single-dispatch Gate+Up GEMM fusion during inference.
    """

    def __init__(self, config: Any):
        super().__init__()
        hidden_dim = int(2 * (4 * config.n_embd) / 3)
        hidden_dim = 64 * ((hidden_dim + 64 - 1) // 64)
        self.hidden_dim = hidden_dim
        bias = getattr(config, "bias", False)
        self.w_gate = nn.Linear(config.n_embd, hidden_dim, bias=bias)
        self.w_up = nn.Linear(config.n_embd, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, config.n_embd, bias=bias)
        setattr(self.w_down, "NANOGPT_SCALE_INIT", 1)
        self._cached_gate_up_w: Optional[torch.Tensor] = None
        self._cached_gate_up_b: Optional[torch.Tensor] = None

    def train(self, mode: bool = True):
        self._cached_gate_up_w = None
        self._cached_gate_up_b = None
        return super().train(mode)

    def _apply(self, fn):
        self._cached_gate_up_w = None
        self._cached_gate_up_b = None
        return super()._apply(fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training and not torch.is_grad_enabled():
            if (
                self._cached_gate_up_w is None
                or self._cached_gate_up_w.device != x.device
                or self._cached_gate_up_w.dtype != x.dtype
            ):
                self._cached_gate_up_w = torch.cat(
                    [self.w_gate.weight, self.w_up.weight], dim=0
                ).to(device=x.device, dtype=x.dtype)
                if self.w_gate.bias is not None and self.w_up.bias is not None:
                    self._cached_gate_up_b = torch.cat(
                        [self.w_gate.bias, self.w_up.bias], dim=0
                    ).to(device=x.device, dtype=x.dtype)
            gate_up = F.linear(x, self._cached_gate_up_w, self._cached_gate_up_b)
            if (
                _METAL_MOD is not None
                and x.device.type == "mps"
                and x.dtype in (torch.float32, torch.float16)
                and (self.hidden_dim % 4 == 0)
                and gate_up.is_contiguous()
            ):
                return self.w_down(_METAL_MOD.swiglu_gate_up_mps(gate_up))
            gate, up = gate_up.split(self.hidden_dim, dim=-1)
            return self.w_down(fused_swiglu(gate, up))
        gate = self.w_gate(x)
        up = self.w_up(x)
        return self.w_down(fused_swiglu(gate, up))


# ----------------------------------------------------------------------------
# 3. Fused SDPA (Scaled Dot-Product Attention) Autograd Function
# ----------------------------------------------------------------------------


class FusedSDPAFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
        sliding_window: Optional[int] = None,
    ) -> torch.Tensor:
        device = q.device
        if device.type == "cuda" and HAS_TRITON:
            out = _triton_kernels.triton_fused_sdpa_forward(
                q, k, v, is_causal=is_causal, sliding_window=sliding_window
            )
        elif (
            _METAL_MOD is not None
            and device.type == "mps"
            and q.dtype == torch.float32
            and sliding_window is None
            and q.size(1) == k.size(1)
            and q.size(2) == k.size(2)
            and q.size(3) <= 128
        ):
            out = _METAL_MOD.flash_attention_forward_mps(
                q.contiguous(), k.contiguous(), v.contiguous(), is_causal
            )
        else:
            if sliding_window is not None and is_causal:
                T = q.size(2)
                causal_mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
                window_mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril(
                    diagonal=-sliding_window
                )
                attn_mask = causal_mask & ~window_mask
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            else:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        ctx.save_for_backward(q, k, v)
        ctx.is_causal = is_causal
        ctx.sliding_window = sliding_window
        return out

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        q, k, v = ctx.saved_tensors
        # Using PyTorch's highly optimized SDPA backward for exact parity
        with torch.enable_grad():
            q_ = q.detach().requires_grad_(True)
            k_ = k.detach().requires_grad_(True)
            v_ = v.detach().requires_grad_(True)
            if ctx.sliding_window is not None and ctx.is_causal:
                T = q_.size(2)
                causal_mask = torch.ones(
                    T, T, dtype=torch.bool, device=q_.device
                ).tril()
                window_mask = torch.ones(T, T, dtype=torch.bool, device=q_.device).tril(
                    diagonal=-ctx.sliding_window
                )
                attn_mask = causal_mask & ~window_mask
                out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=attn_mask)
            else:
                out = F.scaled_dot_product_attention(
                    q_, k_, v_, is_causal=ctx.is_causal
                )
            out.backward(grad_output)
        return q_.grad, k_.grad, v_.grad, None, None


def fused_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    """Functional interface for Fused SDPA (FlashAttention) with Sliding Window & Zero-Copy GQA Support."""
    enable_gqa = q.size(1) != k.size(1)
    if (
        not torch.is_grad_enabled()
        and not (q.device.type == "cuda" and HAS_TRITON)
        and sliding_window is None
    ):
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, enable_gqa=enable_gqa
        )
    if enable_gqa and (
        sliding_window is not None or (q.device.type == "cuda" and HAS_TRITON)
    ):
        n_rep = q.size(1) // k.size(1)
        B, n_kv_head, T_k, head_dim = k.shape
        k = (
            k[:, :, None, :, :]
            .expand(B, n_kv_head, n_rep, T_k, head_dim)
            .reshape(B, n_kv_head * n_rep, T_k, head_dim)
        )
        v = (
            v[:, :, None, :, :]
            .expand(B, n_kv_head, n_rep, T_k, head_dim)
            .reshape(B, n_kv_head * n_rep, T_k, head_dim)
        )
    elif enable_gqa:
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, enable_gqa=True
        )
    return FusedSDPAFunction.apply(q, k, v, is_causal, sliding_window)


def flash_attention_mps(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
) -> torch.Tensor:
    """Direct dispatch to the custom Metal tiled SRAM FlashAttention forward kernel."""
    if _METAL_MOD is not None and q.device.type == "mps" and q.dtype == torch.float32:
        if q.size(1) != k.size(1):
            n_rep = q.size(1) // k.size(1)
            B, n_kv_head, T_k, head_dim = k.shape
            k = (
                k[:, :, None, :, :]
                .expand(B, n_kv_head, n_rep, T_k, head_dim)
                .reshape(B, n_kv_head * n_rep, T_k, head_dim)
            )
            v = (
                v[:, :, None, :, :]
                .expand(B, n_kv_head, n_rep, T_k, head_dim)
                .reshape(B, n_kv_head * n_rep, T_k, head_dim)
            )
        return _METAL_MOD.flash_attention_forward_mps(
            q.contiguous(), k.contiguous(), v.contiguous(), is_causal
        )
    return F.scaled_dot_product_attention(
        q, k, v, is_causal=is_causal, enable_gqa=(q.size(1) != k.size(1))
    )


def gqa_flash_decode_mps(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    active_len: int,
    num_splits: int = 0,
) -> torch.Tensor:
    """Fast Split-KV Flash-Decode for 3:1 GQA on Apple Silicon MPS (FP16)."""
    if (
        _METAL_MOD is not None
        and hasattr(_METAL_MOD, "gqa_flash_decode_mps")
        and q.device.type == "mps"
        and q.dtype == torch.float16
    ):
        return _METAL_MOD.gqa_flash_decode_mps(
            q.contiguous(),
            k_cache.contiguous(),
            v_cache.contiguous(),
            active_len,
            num_splits,
        )
    # PyTorch fallback for CPU or non-FP16 testing
    orig_shape = q.shape
    if q.dim() == 3:
        q_4d = q.view(1, 12, 1, 64)
    elif q.dim() == 2:
        q_4d = q.view(1, 12, 1, 64)
    elif q.dim() == 1:
        q_4d = q.view(1, 12, 1, 64)
    else:
        q_4d = q
    k_in = k_cache if k_cache.dim() == 4 else k_cache.unsqueeze(0)
    v_in = v_cache if v_cache.dim() == 4 else v_cache.unsqueeze(0)
    k_active = k_in[:, :, :active_len, :]
    v_active = v_in[:, :, :active_len, :]
    out = F.scaled_dot_product_attention(q_4d, k_active, v_active, enable_gqa=True)
    return out.view(orig_shape)


class FusedRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, freqs_cis):
        if freqs_cis.is_complex():
            freqs_real = torch.view_as_real(freqs_cis).contiguous()
        else:
            freqs_real = (
                freqs_cis if freqs_cis.is_contiguous() else freqs_cis.contiguous()
            )
        ctx.save_for_backward(freqs_real)
        if (
            _METAL_MOD is not None
            and x.device.type == "mps"
            and x.dtype in (torch.float32, torch.float16)
        ):
            out = _METAL_MOD.apply_rope_mps(x.contiguous(), freqs_real, True)
            return out
        raise RuntimeError("FusedRoPEFunction called without available kernel.")

    @staticmethod
    def backward(ctx, grad_out):
        (freqs_real,) = ctx.saved_tensors
        if (
            _METAL_MOD is not None
            and grad_out.device.type == "mps"
            and grad_out.dtype in (torch.float32, torch.float16)
        ):
            grad_x = _METAL_MOD.apply_rope_mps(grad_out.contiguous(), freqs_real, False)
            return grad_x, None
        raise RuntimeError("FusedRoPEFunction backward kernel missing for device.")


def fused_apply_rope(
    x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int = 0
) -> torch.Tensor:
    """Functional interface for Fused RoPE."""
    B, n_head, T, head_dim = x.shape
    freqs_slice = freqs_cis[start_pos : start_pos + T]

    if (
        _METAL_MOD is not None
        and x.device.type == "mps"
        and x.dtype in (torch.float32, torch.float16)
    ):
        if not torch.is_grad_enabled():
            if freqs_slice.is_complex():
                freqs_real = torch.view_as_real(freqs_slice).contiguous()
            else:
                freqs_real = (
                    freqs_slice
                    if freqs_slice.is_contiguous()
                    else freqs_slice.contiguous()
                )
            x_c = x if x.is_contiguous() else x.contiguous()
            return _METAL_MOD.apply_rope_mps(x_c, freqs_real, True)
        return FusedRoPEFunction.apply(x, freqs_slice)

    # Standard PyTorch eager fallback
    orig_dtype = x.dtype
    x_complex = torch.view_as_complex(x.float().reshape(B, n_head, T, -1, 2))
    if not freqs_slice.is_complex():
        freqs_slice = torch.view_as_complex(freqs_slice.float().contiguous())
    freqs_slice_expanded = freqs_slice.view(1, 1, T, -1).to(x.device)
    x_rotated = torch.view_as_real(x_complex * freqs_slice_expanded).reshape(
        B, n_head, T, head_dim
    )
    return x_rotated.to(orig_dtype)


# ----------------------------------------------------------------------------
# 5. Fused Cross Entropy
# ----------------------------------------------------------------------------


class FusedCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, ignore_index):
        ctx.save_for_backward(logits, targets)
        ctx.ignore_index = ignore_index
        losses = _METAL_MOD.cross_entropy_forward_mps(
            logits.contiguous(), targets.contiguous(), ignore_index
        )
        return losses

    @staticmethod
    def backward(ctx, grad_losses):
        logits, targets = ctx.saved_tensors
        ignore_index = ctx.ignore_index
        grad_logits = _METAL_MOD.cross_entropy_backward_mps(
            logits.contiguous(),
            targets.contiguous(),
            grad_losses.contiguous(),
            ignore_index,
        )
        return grad_logits, None, None


def fused_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    if (
        _METAL_MOD is not None
        and logits.device.type == "mps"
        and logits.dtype == torch.float32
    ):
        orig_shape = logits.shape
        logits_flat = logits.view(-1, orig_shape[-1])
        targets_flat = targets.view(-1).to(torch.int32)

        losses = FusedCrossEntropyFunction.apply(
            logits_flat, targets_flat, ignore_index
        )

        if reduction == "mean":
            valid_mask = (targets_flat != ignore_index).float()
            num_valid = valid_mask.sum().clamp(min=1.0)
            return (losses * valid_mask).sum() / num_valid
        elif reduction == "sum":
            valid_mask = (targets_flat != ignore_index).float()
            return (losses * valid_mask).sum()
        else:
            return losses.view(targets.shape)

    return F.cross_entropy(
        logits, targets, ignore_index=ignore_index, reduction=reduction
    )
