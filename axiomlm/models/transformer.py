"""
AxiomLM Transformer Architecture & Model Configuration.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from ..kernels.ops import fused_cross_entropy
from .attention import CausalSelfAttention
from .medusa import MedusaHead
from .modules import (
    MLP,
    RMSNorm,
    SwiGLUMLP,
    apply_rope,
    precompute_rope_frequencies,
    repeat_kv,
)

try:
    from ..kernels import FusedRMSNorm, FusedSwiGLUMLP
    from ..kernels.ops import _METAL_MOD

    HAS_CUSTOM_KERNELS = True
except (ImportError, ValueError):
    try:
        from kernels import FusedRMSNorm, FusedSwiGLUMLP
        from kernels.ops import _METAL_MOD

        HAS_CUSTOM_KERNELS = True
    except ImportError:
        HAS_CUSTOM_KERNELS = False
        FusedRMSNorm = None  # type: ignore
        FusedSwiGLUMLP = None  # type: ignore
        _METAL_MOD = None  # type: ignore


@dataclass
class ModelConfig:
    """
    Configuration specification for AxiomLM Transformer models.
    Supports both Classic GPT-2 and Modern LLaMA-3 architectures.
    """

    block_size: int = 1024
    vocab_size: int = 50304  # Padded to multiple of 64 for SIMD/Tensor Core efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_kv_head: int | None = None  # GQA: None for full MHA, or int (e.g. 4)
    norm_type: str = "rmsnorm"  # "rmsnorm" (modern) or "layernorm" (classic)
    pos_emb: str = "rope"  # "rope" (modern) or "learned" (classic)
    mlp_type: str = "swiglu"  # "swiglu" (modern) or "gelu" (classic)
    bias: bool = False  # Modern architectures omit linear biases
    rope_theta: float = 10000.0  # Base frequency for RoPE
    sliding_window: int | None = None  # Sliding Window Attention (SWA) size
    use_fused_kernels: bool = False  # Use fused Triton / Metal / NEON operators
    grad_checkpoint: bool = False  # Enable gradient checkpointing
    arch: str | None = None  # Optional convenience preset ("modern" or "classic")
    quant_mode: str | None = (
        None  # Quantized decode mode: None (FP16), "q4" (MARLIN Q4), "q4_legacy", or "q8"
    )
    num_medusa_heads: int = (
        0  # Number of auxiliary Medusa speculative self-drafting heads (e.g. 3)
    )
    intermediate_size: int | None = (
        None  # Explicit SwiGLU intermediate size (HF parity)
    )
    rms_norm_eps: float = (
        1e-6  # RMSNorm epsilon (1e-5 for Llama-3/SmolLM2, 1e-6 for Qwen2)
    )
    tie_word_embeddings: bool = True  # Tie wte.weight and lm_head.weight
    qkv_bias: bool = (
        False  # Enable QKV attention bias (e.g. Qwen2.5) without MLP/norm bias
    )
    rope_scaling: dict[str, Any] | None = (
        None  # Optional RoPE scaling dict (e.g. llama3)
    )

    def __post_init__(self):
        if self.arch == "classic":
            self.norm_type = "layernorm"
            self.pos_emb = "learned"
            self.mlp_type = "gelu"
            self.bias = True
            self.n_kv_head = None
        elif self.arch == "modern":
            self.norm_type = "rmsnorm"
            self.pos_emb = "rope"
            self.mlp_type = "swiglu"
            self.bias = False
            if self.n_kv_head is None:
                self.n_kv_head = 4


# Backward compatibility alias
GPTConfig = ModelConfig


class Block(nn.Module):
    """
    A single Transformer decoder block.
    Configurable for Pre-LN (LayerNorm) or Pre-RMSNorm with SwiGLU / GELU MLP.
    """

    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.norm_type = config.norm_type
        self.mlp_type = config.mlp_type
        self.use_fused_kernels = config.use_fused_kernels and HAS_CUSTOM_KERNELS
        self.layer_idx = layer_idx

        # 1. Pre-Attention Normalization
        if (
            self.use_fused_kernels
            and self.norm_type == "rmsnorm"
            and FusedRMSNorm is not None
        ):
            self.ln_1 = FusedRMSNorm(config.n_embd, eps=config.rms_norm_eps)
        elif self.norm_type == "rmsnorm":
            self.ln_1 = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        else:
            self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)

        # 2. Attention
        self.attn = CausalSelfAttention(config)
        self.attn.layer_idx = layer_idx

        # 3. Pre-MLP Normalization
        if (
            self.use_fused_kernels
            and self.norm_type == "rmsnorm"
            and FusedRMSNorm is not None
        ):
            self.ln_2 = FusedRMSNorm(config.n_embd, eps=config.rms_norm_eps)
        elif self.norm_type == "rmsnorm":
            self.ln_2 = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        else:
            self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)

        # 4. Feed-Forward Network
        if (
            self.use_fused_kernels
            and self.mlp_type == "swiglu"
            and FusedSwiGLUMLP is not None
        ):
            self.mlp = FusedSwiGLUMLP(config)
        elif self.mlp_type == "swiglu":
            self.mlp = SwiGLUMLP(config)
        else:
            self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_out, new_kv_cache = self.attn(
            self.ln_1(x), freqs_cis=freqs_cis, kv_cache=kv_cache, use_cache=use_cache
        )
        if (
            not self.training
            and not torch.is_grad_enabled()
            and self.use_fused_kernels
            and _METAL_MOD is not None
            and x.device.type == "mps"
            and hasattr(self.ln_2, "_cached_weight")
            and x.is_contiguous()
            and attn_out.is_contiguous()
        ):
            if (
                self.ln_2._cached_weight is None
                or self.ln_2._cached_weight.device != x.device
                or self.ln_2._cached_weight.dtype != x.dtype
            ):
                self.ln_2._cached_weight = self.ln_2.weight.to(
                    device=x.device, dtype=x.dtype
                ).contiguous()
            norm2 = _METAL_MOD.add_rmsnorm_infer_mps(
                x, attn_out, self.ln_2._cached_weight, self.ln_2.eps
            )
            x = x + self.mlp(norm2)
            return x, new_kv_cache
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv_cache


class Transformer(nn.Module):
    """
    AxiomLM High-Performance Autoregressive Decoder Transformer.
    """

    def __init__(self, config: ModelConfig, init_weights: bool = True):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList(
                    [Block(config, layer_idx=i) for i in range(config.n_layer)]
                ),
            )
        )

        # Position embeddings
        if config.pos_emb == "learned":
            self.transformer["wpe"] = nn.Embedding(config.block_size, config.n_embd)

        # Final Layer Normalization
        use_fused = config.use_fused_kernels and HAS_CUSTOM_KERNELS
        if use_fused and config.norm_type == "rmsnorm" and FusedRMSNorm is not None:
            self.transformer["ln_f"] = FusedRMSNorm(
                config.n_embd, eps=config.rms_norm_eps
            )
        elif config.norm_type == "rmsnorm":
            self.transformer["ln_f"] = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        else:
            self.transformer["ln_f"] = nn.LayerNorm(config.n_embd, bias=config.bias)

        # Output projection head (weight-tied to token embedding table when enabled)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.transformer.wte.weight = self.lm_head.weight

        # Precompute complex RoPE frequencies
        if config.pos_emb == "rope":
            head_dim = config.n_embd // config.n_head
            freqs = precompute_rope_frequencies(
                head_dim=head_dim,
                max_seq_len=config.block_size,
                theta=config.rope_theta,
                rope_scaling=config.rope_scaling,
            )
            self.register_buffer("freqs_cis", freqs, persistent=False)
            self.register_buffer(
                "freqs_real",
                torch.view_as_real(freqs).contiguous(),
                persistent=False,
            )
        else:
            self.freqs_cis = None
            self.freqs_real = None

        # Neural Medusa speculative heads (auxiliary self-drafting)
        if getattr(config, "num_medusa_heads", 0) > 0:
            self.medusa_heads = nn.ModuleList(
                [
                    MedusaHead(config.n_embd, config.vocab_size)
                    for _ in range(config.num_medusa_heads)
                ]
            )
        else:
            self.medusa_heads = None

        # Initialize weights (can be skipped when loading open-weight Safetensors)
        if init_weights:
            self.apply(self._init_weights)
        self._mps_decode_cache_id: int | None = None
        self._mps_decode_dtype: torch.dtype | None = None
        self._mps_decode_runner: Any | None = None
        self._quant_mode: str | None = config.quant_mode
        self._lm_head_q8: bool | None = None
        self._mps_quant_decode_runner: Any | None = None
        self._mps_quant_cache_id: int | None = None
        self._quant_weights_cache: dict | None = None

    def quantize(
        self,
        mode: str | None = "q4",
        lm_head_q8: bool | None = None,
        release_fp16: bool = False,
    ):
        """
        Enables register-level quantized execution:
        - 'q4': MARLIN 64207531 + Ladder 4-bit with analytical importance-MSE scale clipping (4.0 bpw)
        - 'q4_k_m': GGUF Q4_K_M mixed-precision (MSE-clipped MARLIN Q4 + Q8_G64 w_down & lm_head)
        - 'q4_rtn': Raw min-max Round-To-Nearest MARLIN Q4 (for ablation baselines)
        - 'q4_legacy': Legacy serial-unpack Q4_G64 (retained for A/B benchmarking)
        - 'q8': 8-bit group-64 affine quantization
        - None: Native FP16 / FP32

        Args:
            mode: Quantization mode name or None.
            lm_head_q8: Optional override to decouple `lm_head` into Q8_G64 while keeping
                all Transformer blocks in 4-bit MARLIN (`mode='q4'`).
            release_fp16: If True, evicts uncompressed FP16 weights on MPS once quantized
                buffers are packed, collapsing UMA memory footprint (>4x reduction).
        """
        if getattr(self, "_fp16_released", False) and mode is None:
            raise RuntimeError(
                "Cannot unquantize model after FP16 weights have been released in Transformer.quantize(..., release_fp16=True)."
            )
        assert mode in [
            None,
            "q4",
            "q4_k_m",
            "q4_rtn",
            "q4_legacy",
            "q8",
        ], (
            f"Invalid quant mode: {mode}. Choose 'q4', 'q4_k_m', 'q4_rtn', "
            f"'q4_legacy', 'q8', or None."
        )
        self._quant_mode = mode
        self._lm_head_q8 = lm_head_q8
        self._mps_quant_decode_runner = None
        self._mps_quant_cache_id = None
        self._quant_weights_cache = None

        if release_fp16 and mode is not None:
            device = next(self.parameters()).device
            self._ensure_quant_weights(device)
            w_dtype = torch.float16
            for b in self.transformer.h:
                b.attn.c_attn.weight.data = torch.empty(0, dtype=w_dtype, device=device)
                b.attn.c_proj.weight.data = torch.empty(0, dtype=w_dtype, device=device)
                b.mlp.w_gate.weight.data = torch.empty(0, dtype=w_dtype, device=device)
                b.mlp.w_up.weight.data = torch.empty(0, dtype=w_dtype, device=device)
                b.mlp.w_down.weight.data = torch.empty(0, dtype=w_dtype, device=device)
            self.lm_head.weight.data = torch.empty(0, dtype=w_dtype, device=device)
            self.transformer.wte.weight.data = torch.empty(
                0, dtype=w_dtype, device=device
            )
            if self.medusa_heads is not None:
                for head in self.medusa_heads:
                    head.head.weight.data = torch.empty(0, dtype=w_dtype, device=device)
                    if head.res_block is not None:
                        head.res_block[0].weight.data = torch.empty(
                            0, dtype=w_dtype, device=device
                        )
            self._fp16_released = True
            if self._quant_weights_cache is not None:
                qw = self._quant_weights_cache
                if qw.get("wte_p") is not None and qw["wte_p"].numel() > 0:
                    qw["wte_emb"] = torch.empty(0, device=device, dtype=w_dtype)
            import gc

            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
        return self

    def train(self, mode: bool = True):
        if getattr(self, "_fp16_released", False) and mode:
            raise RuntimeError(
                "Cannot switch to training mode after FP16 weights have been released in Transformer.quantize(..., release_fp16=True)."
            )
        self._mps_decode_cache_id = None
        self._mps_decode_runner = None
        self._mps_quant_decode_runner = None
        self._mps_quant_cache_id = None
        self._quant_weights_cache = None
        return super().train(mode)

    def _apply(self, fn):
        self._mps_decode_cache_id = None
        self._mps_decode_runner = None
        self._mps_quant_decode_runner = None
        self._mps_quant_cache_id = None
        self._quant_weights_cache = None
        # Preserve complex64 dtype of freqs_cis and float32 dtype of freqs_real when model.to(dtype=torch.float16) is called
        freqs_backup = self.freqs_cis
        freqs_real_backup = self.freqs_real
        self.freqs_cis = None
        self.freqs_real = None
        ret = super()._apply(fn)
        if freqs_backup is not None:
            target_dev = next(self.parameters()).device
            self.freqs_cis = freqs_backup.to(device=target_dev, dtype=torch.complex64)
            self.freqs_real = freqs_real_backup.to(
                device=target_dev, dtype=torch.float32
            ).contiguous()
        return ret

    def add_medusa_heads(
        self,
        num_heads: int = 3,
        use_residual: bool = False,
    ):
        """
        Dynamically attaches auxiliary Medusa multi-token self-drafting prediction heads.
        Head k predicts token (t + k + 2) directly from hidden state h_t.
        """
        self.config.num_medusa_heads = num_heads
        self.medusa_heads = nn.ModuleList(
            [
                MedusaHead(
                    self.config.n_embd,
                    self.config.vocab_size,
                    use_residual=use_residual,
                )
                for _ in range(num_heads)
            ]
        )
        for head in self.medusa_heads:
            self._init_weights(head)
        target_dev = next(self.parameters()).device
        target_dtype = self.lm_head.weight.dtype
        self.medusa_heads.to(device=target_dev, dtype=target_dtype)
        self._quant_weights_cache = None
        return self

    def eval_medusa_heads(self, x_norm: torch.Tensor) -> list[torch.Tensor]:
        """
        Evaluates attached Medusa draft heads on normalized hidden states `x_norm`
        (typically shape `(1, 1, n_embd)` for lazy terminal-leaf drafting).
        Automatically dispatches MARLIN Q4 / Q8 register kernels when quantized inference is active.
        """
        if self.medusa_heads is None or len(self.medusa_heads) == 0:
            return []
        if self._can_use_quant_forward():
            qw = self._ensure_quant_weights(x_norm.device)
            m_packs = qw.get("medusa_packs")
            if m_packs is not None and len(m_packs) == len(self.medusa_heads):
                out_list: list[torch.Tensor] = []
                for hp in m_packs:
                    h_cur = x_norm
                    if hp["res_p"] is not None and hp["res_s"] is not None:
                        r_out = self._qlin(
                            h_cur, hp["res_p"], hp["res_s"], self.config.n_embd
                        )
                        h_cur = h_cur + F.silu(r_out)
                    out_list.append(
                        self._qlin(
                            h_cur,
                            hp["head_p"],
                            hp["head_s"],
                            self.config.vocab_size,
                        )
                    )
                return out_list
        m_dtype = next(self.medusa_heads.parameters()).dtype
        x_m = x_norm if x_norm.dtype == m_dtype else x_norm.to(dtype=m_dtype)
        return [head(x_m) for head in self.medusa_heads]

    def forward_with_medusa(
        self,
        idx: torch.Tensor,
        kv_caches: list[Any] | None = None,
        return_all_logits: bool = False,
        return_medusa: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[Any] | None]:
        """
        Forward pass producing both primary next-token logits and Medusa self-drafting logits.

        Returns:
            Tuple of (base_logits, medusa_logits_list, new_kv_caches)
        """
        device = idx.device
        B, T = idx.size()

        # Ultra-fast C++ Whole-Model Decode / Speculative Verification Graph Runner for T in [1, 32], B=1 on MPS
        if (
            1 <= T <= 32
            and B == 1
            and not self.training
            and not torch.is_grad_enabled()
            and device.type == "mps"
            and kv_caches is not None
            and len(kv_caches) == self.config.n_layer
            and hasattr(kv_caches[0], "total_seen_tokens")
            and (T == 1 or kv_caches[0].total_seen_tokens[0] > 0)
        ):
            cache0 = kv_caches[0]
            runner = self.get_decode_runner(cache0)
            if runner is not None and hasattr(runner, "get_last_norm"):
                seen = cache0.total_seen_tokens[0]
                max_seq = cache0.max_seq_len
                write_pos = (
                    (seen % max_seq)
                    if (cache0.ring_buffer or seen >= max_seq)
                    else seen
                )
                new_seen = seen + T
                active_len = min(new_seen, max_seq)
                rope_pos = seen % self.freqs_real.size(0)

                if T == 1:
                    logits = runner.step(idx, rope_pos, write_pos, active_len)
                else:
                    logits = runner.verify_step(idx, rope_pos, write_pos, active_len)
                if not return_all_logits and T > 1:
                    logits = logits[:, -1:, :]
                n_L = self.config.n_layer
                cache0.total_seen_tokens[:] = [new_seen] * n_L
                cache0.layer_seq_lens[:] = [active_len] * n_L

                x_norm_full = runner.get_last_norm(T)
                self._last_verify_norm = x_norm_full
                medusa_logits: list[torch.Tensor] = []
                if return_medusa and self.medusa_heads is not None:
                    x_norm = (
                        x_norm_full[:, -1:, :]
                        if (not return_all_logits and T > 1)
                        else x_norm_full
                    )
                    medusa_logits = self.eval_medusa_heads(x_norm)

                return (
                    logits,
                    medusa_logits,
                    kv_caches,
                )

        if self._can_use_quant_forward():
            logits, x_norm, new_kv_caches = self._forward_quantized_sequence(
                idx, kv_caches=kv_caches, return_all_logits=return_all_logits
            )
            self._last_verify_norm = x_norm
            medusa_logits = (
                self.eval_medusa_heads(x_norm)
                if (return_medusa and self.medusa_heads is not None)
                else []
            )
            return logits, medusa_logits, new_kv_caches

        if self.config.pos_emb == "learned":
            pos = torch.arange(0, T, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)
            tok_emb = self.transformer.wte(idx)
            x = tok_emb + pos_emb
        else:
            x = self.transformer.wte(idx)

        new_kv_caches = []
        for i, block in enumerate(self.transformer.h):
            block_kv = (
                kv_caches[i] if (kv_caches is not None and i < len(kv_caches)) else None
            )
            freqs_arg = (
                self.freqs_real
                if (self.config.use_fused_kernels and self.freqs_real is not None)
                else self.freqs_cis
            )
            x, new_kv = block(
                x,
                freqs_cis=freqs_arg,
                kv_cache=block_kv,
                use_cache=(kv_caches is not None),
            )
            if new_kv is not None:
                new_kv_caches.append(new_kv)

        if kv_caches is not None and not return_all_logits:
            x_target = x if T == 1 else x[:, -1:, :]
        else:
            x_target = x

        x_norm = self.transformer.ln_f(x_target)
        self._last_verify_norm = x_norm
        logits = self.lm_head(x_norm)
        medusa_logits = (
            self.eval_medusa_heads(x_norm)
            if (return_medusa and self.medusa_heads is not None)
            else []
        )

        return (
            logits,
            medusa_logits,
            (new_kv_caches if kv_caches is not None else None),
        )

    def forward_tree_verify(
        self,
        idx: torch.Tensor,
        kv_caches: list[Any],
        tree_depths: list[int],
        tree_mask: list[int],
        return_medusa: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[Any]]:
        """
        SpecInfer / Medusa topology-aware candidate tree verification pass (T in [1, 32]).
        Evaluates an entire branching candidate tree in a single weight pass using 32-bit
        ancestor bitmasks (`tree_mask[u]`) and per-node topological RoPE depths (`tree_depths[u]`).
        Writes uncompacted tree KV states into `[write_pos, write_pos + T)`; caller commits the
        accepted path via `cache.compact_tree_path(write_pos, accepted_indices, new_total_seen)`.
        """
        device = idx.device
        B, T = idx.size()
        assert B == 1 and 1 <= T <= 32, f"Expected B=1 and T in [1, 32], got {(B, T)}"
        assert len(tree_depths) == T and len(tree_mask) == T
        cache0 = kv_caches[0]
        seen = cache0.total_seen_tokens[0]
        max_seq = cache0.max_seq_len
        write_pos = (
            (seen % max_seq) if (cache0.ring_buffer or seen >= max_seq) else seen
        )
        assert (
            write_pos + T <= max_seq and seen == write_pos
        ), f"Tree verification requires contiguous tail capacity: write_pos={write_pos}, T={T}, max_seq={max_seq}"
        rope_pos = seen % self.freqs_real.size(0)

        if (
            not self.training
            and not torch.is_grad_enabled()
            and device.type == "mps"
            and (
                self._quant_mode in ("q4", "q4_k_m", "q4_rtn", "q4_legacy", "q8")
                or self.transformer.wte.weight.dtype == torch.float16
            )
        ):
            runner = self.get_decode_runner(cache0)
            if runner is not None and hasattr(runner, "verify_tree_step"):
                logits = runner.verify_tree_step(
                    idx,
                    rope_pos,
                    write_pos,
                    seen,
                    [int(d) for d in tree_depths],
                    [int(m) & 0xFFFFFFFF for m in tree_mask],
                )
                x_norm = runner.get_last_norm(T)
                self._last_verify_norm = x_norm
                medusa_logits = (
                    self.eval_medusa_heads(x_norm)
                    if (return_medusa and self.medusa_heads is not None)
                    else []
                )
                return logits, medusa_logits, kv_caches

        # Exact eager / CPU / non-fused fallback with explicit boolean tree mask
        use_quant = self._can_use_quant_forward()
        qw = self._ensure_quant_weights(device) if use_quant else None
        if use_quant and qw is not None:
            if qw["wte_emb"].numel() > 0:
                x = qw["wte_emb"].index_select(0, idx.view(-1)).view(1, T, -1)
            else:
                from ..kernels.quant import dequant_q8_rows

                x = (
                    dequant_q8_rows(idx, qw["wte_p"], qw["wte_sb"])
                    .to(dtype=torch.float16)
                    .view(1, T, -1)
                )
        else:
            x = self.transformer.wte(idx)

        n_head = self.config.n_head
        n_kv_head = (
            self.config.n_kv_head if self.config.n_kv_head is not None else n_head
        )
        n_rep = n_head // n_kv_head
        head_dim = self.config.n_embd // n_head
        q_dim = n_head * head_dim
        kv_dim = n_kv_head * head_dim

        # Construct (1, 1, T, seen + T) boolean attention mask
        attn_mask = torch.zeros((1, 1, T, seen + T), dtype=torch.bool, device=device)
        if seen > 0:
            attn_mask[:, :, :, :seen] = True
        for u in range(T):
            m_val = int(tree_mask[u])
            for j in range(T):
                if (m_val >> j) & 1:
                    attn_mask[0, 0, u, seen + j] = True

        depth_idx = torch.tensor(
            [(rope_pos + int(d)) % self.freqs_cis.size(0) for d in tree_depths],
            dtype=torch.long,
            device=self.freqs_cis.device,
        )
        freqs_nodes = (
            self.freqs_cis.index_select(0, depth_idx).view(1, 1, T, -1).to(device)
        )

        for i, block in enumerate(self.transformer.h):
            h_norm = block.ln_1(x.to(block.ln_1.weight.dtype))
            if use_quant and qw is not None:
                b_i = qw["c_attn_b"][i] if qw["c_attn_b"][i].numel() > 0 else None
                qkv = self._qlin(
                    h_norm,
                    qw["c_attn_p"][i],
                    qw["c_attn_s"][i],
                    q_dim + 2 * kv_dim,
                    bias=b_i,
                )
            else:
                qkv = block.attn.c_attn(h_norm)

            q = qkv[:, :, :q_dim].view(1, T, n_head, head_dim).transpose(1, 2)
            k = (
                qkv[:, :, q_dim : q_dim + kv_dim]
                .view(1, T, n_kv_head, head_dim)
                .transpose(1, 2)
            )
            v = (
                qkv[:, :, q_dim + kv_dim :]
                .view(1, T, n_kv_head, head_dim)
                .transpose(1, 2)
            )

            orig_dt = q.dtype
            q = (
                torch.view_as_real(
                    torch.view_as_complex(q.float().reshape(1, n_head, T, -1, 2))
                    * freqs_nodes
                )
                .reshape(1, n_head, T, head_dim)
                .to(orig_dt)
            )
            k = (
                torch.view_as_real(
                    torch.view_as_complex(k.float().reshape(1, n_kv_head, T, -1, 2))
                    * freqs_nodes
                )
                .reshape(1, n_kv_head, T, head_dim)
                .to(orig_dt)
            )

            cache0.k_caches[i][:1, :, write_pos : write_pos + T, :] = k.to(
                cache0.cache.dtype
            )
            cache0.v_caches[i][:1, :, write_pos : write_pos + T, :] = v.to(
                cache0.cache.dtype
            )
            k_all = cache0.k_caches[i][:1, :, : seen + T, :].to(q.dtype)
            v_all = cache0.v_caches[i][:1, :, : seen + T, :].to(q.dtype)
            if n_rep > 1:
                k_all = repeat_kv(k_all, n_rep)
                v_all = repeat_kv(v_all, n_rep)

            attn_y = F.scaled_dot_product_attention(
                q,
                k_all,
                v_all,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            attn_y = attn_y.transpose(1, 2).contiguous().view(1, T, q_dim)

            if use_quant and qw is not None:
                attn_out = self._qlin(
                    attn_y, qw["c_proj_p"][i], qw["c_proj_s"][i], q_dim
                )
            else:
                attn_out = block.attn.c_proj(attn_y)

            x = x + attn_out.to(x.dtype)
            h_norm2 = block.ln_2(x.to(block.ln_2.weight.dtype))
            if use_quant and qw is not None:
                gu = self._qlin(
                    h_norm2, qw["gu_p"][i], qw["gu_s"][i], 2 * qw["hidden_dim"]
                )
                g, u_part = gu.chunk(2, dim=-1)
                sw = F.silu(g) * u_part
                mlp_out = self._qlin(sw, qw["wd_p"][i], qw["wd_s"][i], q_dim)
            else:
                mlp_out = block.mlp(h_norm2)
            x = x + mlp_out.to(x.dtype)

        x_norm = self.transformer.ln_f(x.to(self.transformer.ln_f.weight.dtype))
        if use_quant and qw is not None:
            logits = self._qlin(
                x_norm, qw["lmh_p"], qw["lmh_sb"], self.config.vocab_size
            )
        else:
            logits = self.lm_head(x_norm)

        self._last_verify_norm = x_norm
        medusa_logits = (
            self.eval_medusa_heads(x_norm)
            if (return_medusa and self.medusa_heads is not None)
            else []
        )
        return logits, medusa_logits, kv_caches

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _can_use_quant_forward(self) -> bool:
        if self._quant_mode not in ("q4", "q4_k_m", "q4_rtn", "q4_legacy", "q8"):
            return False
        if self.training or torch.is_grad_enabled():
            return False
        if (
            self.config.pos_emb != "rope"
            or self.config.norm_type != "rmsnorm"
            or self.config.mlp_type != "swiglu"
            or self.config.bias
            or self.config.n_embd % 64 != 0
            or len(self.transformer.h) == 0
        ):
            return False
        if (
            getattr(self, "_fp16_released", False)
            or self.transformer.h[0].mlp.w_down.weight.numel() == 0
        ):
            hidden_dim = getattr(self.config, "intermediate_size", None) or (
                self.config.n_embd * 4
            )
        else:
            hidden_dim = self.transformer.h[0].mlp.w_down.weight.size(1)
        return hidden_dim % 64 == 0

    def _ensure_quant_weights(self, device: torch.device) -> dict[str, Any]:
        lmh_q8_flag = getattr(self, "_lm_head_q8", None)
        cache_key = (self._quant_mode, lmh_q8_flag, str(device))
        if (
            self._quant_weights_cache is not None
            and self._quant_weights_cache.get("key") == cache_key
        ):
            return self._quant_weights_cache

        from ..kernels.quant import (
            pack_linear_q4_g64,
            pack_linear_q4_marlin,
            pack_linear_q8_g64,
        )

        mode = self._quant_mode
        use_marlin = mode in ("q4", "q4_k_m", "q4_rtn")
        use_mse_clip = mode in ("q4", "q4_k_m")
        use_q8_lm_head = (
            bool(lmh_q8_flag)
            if lmh_q8_flag is not None
            else (mode in ("q4_k_m", "q8") or bool(self.config.tie_word_embeddings))
        )
        use_q8_w_down = mode in ("q4_k_m", "q8")

        def _pack_proj(
            w_ten: torch.Tensor,
            col_imp: torch.Tensor | None = None,
            force_q8: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if force_q8 or mode == "q8":
                p_t, s_t = pack_linear_q8_g64(w_ten.detach())
                return p_t.to(device).clone(), s_t.to(device).clone()
            if mode == "q4_legacy":
                p_t, s_t = pack_linear_q4_g64(w_ten.detach().cpu())
                return p_t.to(device).clone(), s_t.to(device).clone()
            assert use_marlin
            w_src = w_ten.detach()
            if not use_mse_clip:
                p_t, s_t = pack_linear_q4_marlin(
                    w_src, clip_search=False, col_importance=None
                )
                return p_t.to(device).clone(), s_t.to(device).clone()
            p_t, s_t = pack_linear_q4_marlin(
                w_src,
                clip_search=True,
                col_importance=col_imp,
            )
            return p_t.to(device).clone(), s_t.to(device).clone()

        imp_lnf = (
            self.transformer.ln_f.weight.detach().float().abs()
            if use_mse_clip
            else None
        )
        lmh_p, lmh_sb = _pack_proj(
            self.lm_head.weight, col_imp=imp_lnf, force_q8=use_q8_lm_head
        )

        # Packed Q8_G64 token embeddings (shared directly with lm_head if tie_word_embeddings=True and lm_head is Q8)
        if self.config.tie_word_embeddings and use_q8_lm_head and lmh_p.ndim == 2:
            wte_p = lmh_p
            wte_sb = lmh_sb
        elif self.transformer.wte.weight.numel() > 0:
            wte_p, wte_sb = _pack_proj(
                self.transformer.wte.weight, col_imp=None, force_q8=True
            )
        elif self.config.tie_word_embeddings and self.lm_head.weight.numel() > 0:
            wte_p, wte_sb = _pack_proj(self.lm_head.weight, col_imp=None, force_q8=True)
        else:
            wte_p = torch.empty(0, dtype=torch.int32, device=device)
            wte_sb = torch.empty(0, dtype=torch.float16, device=device)

        w_dtype = torch.float16
        if (
            getattr(self, "_fp16_released", False)
            or self.transformer.wte.weight.numel() == 0
            or (wte_p is not None and wte_p.numel() > 0)
        ):
            wte_emb = torch.empty(0, device=device, dtype=w_dtype)
        else:
            wte_emb = (
                self.transformer.wte.weight.detach()
                .to(device=device, dtype=w_dtype)
                .contiguous()
            )

        c_attn_p, c_attn_s, c_attn_b = [], [], []
        c_proj_p, c_proj_s = [], []
        gu_p, gu_s = [], []
        wd_p, wd_s = [], []
        if (
            getattr(self, "_fp16_released", False)
            or self.transformer.h[0].mlp.w_down.weight.numel() == 0
        ):
            hidden_dim = getattr(self.config, "intermediate_size", None) or (
                self.config.n_embd * 4
            )
        else:
            hidden_dim = self.transformer.h[0].mlp.w_down.weight.size(1)

        for b in self.transformer.h:
            imp_h1 = b.ln_1.weight.detach().float().abs() if use_mse_clip else None
            imp_h2 = b.ln_2.weight.detach().float().abs() if use_mse_clip else None
            imp_sw = (
                (
                    b.mlp.w_gate.weight.detach().float().norm(dim=1)
                    * b.mlp.w_up.weight.detach().float().norm(dim=1)
                )
                if (use_mse_clip and not use_q8_w_down)
                else None
            )

            p, s = _pack_proj(b.attn.c_attn.weight, col_imp=imp_h1)
            c_attn_p.append(p)
            c_attn_s.append(s)
            if b.attn.c_attn.bias is not None:
                c_attn_b.append(
                    b.attn.c_attn.bias.detach()
                    .to(device=device, dtype=w_dtype)
                    .contiguous()
                )
            else:
                c_attn_b.append(torch.empty(0, device=device, dtype=w_dtype))

            p, s = _pack_proj(b.attn.c_proj.weight, col_imp=None)
            c_proj_p.append(p)
            c_proj_s.append(s)

            gate_up_w = torch.cat(
                [b.mlp.w_gate.weight.detach(), b.mlp.w_up.weight.detach()], dim=0
            )
            p, s = _pack_proj(gate_up_w, col_imp=imp_h2)
            gu_p.append(p)
            gu_s.append(s)

            p, s = _pack_proj(
                b.mlp.w_down.weight, col_imp=imp_sw, force_q8=use_q8_w_down
            )
            wd_p.append(p)
            wd_s.append(s)

        medusa_packs: list[dict[str, Any]] = []
        if self.medusa_heads is not None:
            for head in self.medusa_heads:
                res_p, res_s = None, None
                if head.use_residual and head.res_block is not None:
                    res_p, res_s = _pack_proj(head.res_block[0].weight, col_imp=imp_lnf)
                hp, hs = _pack_proj(head.head.weight, col_imp=imp_lnf)
                medusa_packs.append(
                    {
                        "res_p": res_p,
                        "res_s": res_s,
                        "head_p": hp,
                        "head_s": hs,
                    }
                )

        self._quant_weights_cache = {
            "key": cache_key,
            "mode": self._quant_mode,
            "hidden_dim": hidden_dim,
            "wte_emb": wte_emb,
            "wte_p": wte_p,
            "wte_sb": wte_sb,
            "lmh_p": lmh_p,
            "lmh_sb": lmh_sb,
            "c_attn_p": c_attn_p,
            "c_attn_s": c_attn_s,
            "c_attn_b": c_attn_b,
            "c_proj_p": c_proj_p,
            "c_proj_s": c_proj_s,
            "gu_p": gu_p,
            "gu_s": gu_s,
            "wd_p": wd_p,
            "wd_s": wd_s,
            "medusa_packs": medusa_packs,
        }
        return self._quant_weights_cache

    def _qlin(
        self,
        x: torch.Tensor,
        w_p: torch.Tensor,
        w_sb: torch.Tensor,
        out_features: int,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from ..kernels.quant import qmv_q4_g64, qmv_q4_marlin, qmv_q8_g64

        orig_dtype = x.dtype
        x16 = x if x.dtype == torch.float16 else x.to(torch.float16)
        if not x16.is_contiguous():
            x16 = x16.contiguous()
        if w_p.ndim == 4:
            y = qmv_q4_marlin(x16, w_p, w_sb, out_features=out_features, bias=bias)
        elif w_p.size(1) == (x16.size(-1) // 4):
            y = qmv_q8_g64(x16, w_p, w_sb)
            if bias is not None:
                y = y + bias.to(device=y.device, dtype=y.dtype)
        else:
            y = qmv_q4_g64(x16, w_p, w_sb)
            if bias is not None:
                y = y + bias.to(device=y.device, dtype=y.dtype)
        return y if y.dtype == orig_dtype else y.to(orig_dtype)

    def _forward_quantized_sequence(
        self,
        idx: torch.Tensor,
        kv_caches: list[Any] | None = None,
        return_all_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, list[Any] | None]:
        device = idx.device
        B, T = idx.size()
        qw = self._ensure_quant_weights(device)
        if qw["wte_emb"].numel() > 0:
            x = qw["wte_emb"].index_select(0, idx.reshape(-1)).view(B, T, -1)
        else:
            from ..kernels.quant import dequant_q8_rows

            x = (
                dequant_q8_rows(idx, qw["wte_p"], qw["wte_sb"])
                .to(dtype=torch.float16)
                .view(B, T, -1)
            )

        n_head = self.config.n_head
        n_kv_head = (
            self.config.n_kv_head if self.config.n_kv_head is not None else n_head
        )
        n_rep = n_head // n_kv_head
        head_dim = self.config.n_embd // n_head
        q_dim = n_head * head_dim
        kv_dim = n_kv_head * head_dim
        hidden_dim = qw["hidden_dim"]

        new_kv_caches: list[Any] = []
        for i, block in enumerate(self.transformer.h):
            kv_cache = (
                kv_caches[i] if (kv_caches is not None and i < len(kv_caches)) else None
            )
            start_pos = 0
            if kv_cache is not None:
                if isinstance(kv_cache, tuple):
                    start_pos = kv_cache[0].shape[2]
                elif hasattr(kv_cache, "total_seen_tokens"):
                    start_pos = kv_cache.total_seen_tokens[i]
                else:
                    start_pos = kv_cache.seq_len

            h1 = block.ln_1(x.to(block.ln_1.weight.dtype))
            b_i = qw["c_attn_b"][i] if qw["c_attn_b"][i].numel() > 0 else None
            qkv = self._qlin(
                h1, qw["c_attn_p"][i], qw["c_attn_s"][i], q_dim + 2 * kv_dim, bias=b_i
            )
            q = qkv[:, :, :q_dim].view(B, T, n_head, head_dim).transpose(1, 2)
            k = (
                qkv[:, :, q_dim : q_dim + kv_dim]
                .view(B, T, n_kv_head, head_dim)
                .transpose(1, 2)
            )
            v = (
                qkv[:, :, q_dim + kv_dim :]
                .view(B, T, n_kv_head, head_dim)
                .transpose(1, 2)
            )

            freqs_arg = (
                self.freqs_real
                if (self.config.use_fused_kernels and self.freqs_real is not None)
                else self.freqs_cis
            )
            if block.attn.use_fused_kernels and freqs_arg is not None:
                q = apply_rope(q, freqs_arg, start_pos=start_pos)
                k = apply_rope(k, freqs_arg, start_pos=start_pos)
            elif self.freqs_cis is not None:
                orig_dt = q.dtype
                freqs_slice = (
                    self.freqs_cis[start_pos : start_pos + T]
                    .view(1, 1, T, -1)
                    .to(q.device)
                )
                q = (
                    torch.view_as_real(
                        torch.view_as_complex(q.float().reshape(B, n_head, T, -1, 2))
                        * freqs_slice
                    )
                    .reshape(B, n_head, T, head_dim)
                    .to(orig_dt)
                )
                k = (
                    torch.view_as_real(
                        torch.view_as_complex(k.float().reshape(B, n_kv_head, T, -1, 2))
                        * freqs_slice
                    )
                    .reshape(B, n_kv_head, T, head_dim)
                    .to(orig_dt)
                )

            if kv_cache is not None:
                if isinstance(kv_cache, tuple):
                    k = torch.cat([kv_cache[0], k], dim=2)
                    v = torch.cat([kv_cache[1], v], dim=2)
                    new_kv_caches.append((k, v))
                elif hasattr(kv_cache, "update"):
                    k, v = kv_cache.update(i, k, v)
                    new_kv_caches.append(kv_cache)
                else:
                    kv_cache.append_kv(i, k.transpose(1, 2), v.transpose(1, 2))
                    k_rec, v_rec = kv_cache.get_reconstructed_cache(i)
                    k = k_rec.transpose(1, 2)
                    v = v_rec.transpose(1, 2)
                    new_kv_caches.append(kv_cache)

            attn_mask = None
            if T > 1 and start_pos > 0:
                total_k_len = k.size(2)
                q_pos = (start_pos + torch.arange(T, device=q.device)).unsqueeze(1)
                k_pos = torch.arange(total_k_len, device=q.device).unsqueeze(0)
                attn_mask = k_pos <= q_pos
                if self.config.sliding_window is not None:
                    attn_mask = attn_mask & (
                        k_pos >= q_pos - self.config.sliding_window + 1
                    )
                is_causal = False
            elif self.config.sliding_window is not None and T > 1:
                causal_mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
                window_mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril(
                    diagonal=-self.config.sliding_window
                )
                attn_mask = causal_mask & ~window_mask
                is_causal = False
            else:
                is_causal = (T > 1) and (start_pos == 0)

            k_attn = repeat_kv(k, n_rep) if n_rep > 1 else k
            v_attn = repeat_kv(v, n_rep) if n_rep > 1 else v
            y = F.scaled_dot_product_attention(
                q,
                k_attn.to(q.dtype),
                v_attn.to(q.dtype),
                attn_mask=attn_mask,
                is_causal=is_causal,
            )
            y = y.transpose(1, 2).contiguous().view(B, T, q_dim)
            attn_out = self._qlin(y, qw["c_proj_p"][i], qw["c_proj_s"][i], q_dim)
            x = x + attn_out.to(x.dtype)

            h2 = block.ln_2(x.to(block.ln_2.weight.dtype))
            gu = self._qlin(h2, qw["gu_p"][i], qw["gu_s"][i], 2 * hidden_dim)
            if (
                gu.device.type == "mps"
                and gu.dtype == torch.float16
                and _METAL_MOD is not None
                and self.config.use_fused_kernels
            ):
                sw = _METAL_MOD.swiglu_gate_up_mps(gu)
            else:
                g, u_part = gu.chunk(2, dim=-1)
                sw = F.silu(g) * u_part
            mlp_out = self._qlin(sw, qw["wd_p"][i], qw["wd_s"][i], q_dim)
            x = x + mlp_out.to(x.dtype)

        x_target = (
            x[:, -1:, :]
            if (kv_caches is not None and not return_all_logits and T > 1)
            else x
        )
        x_norm = self.transformer.ln_f(x_target.to(self.transformer.ln_f.weight.dtype))
        logits = self._qlin(x_norm, qw["lmh_p"], qw["lmh_sb"], self.config.vocab_size)
        return logits, x_norm, (new_kv_caches if kv_caches is not None else None)

    def get_decode_runner(self, cache: Any) -> Any | None:
        """
        Retrieves or initializes the native C++ Metal decode runner (quantized or FP16)
        bound to the given KV cache instance.
        """
        device = next(self.parameters()).device
        head_dim = self.config.n_embd // self.config.n_head
        if (
            device.type != "mps"
            or _METAL_MOD is None
            or not self.config.use_fused_kernels
            or self.config.pos_emb != "rope"
            or self.config.norm_type != "rmsnorm"
            or self.config.mlp_type != "swiglu"
            or self.config.bias
            or self.config.sliding_window is not None
            or head_dim > 128
            or (head_dim % 2 != 0)
        ):
            return None

        cache0 = cache[0] if isinstance(cache, list) else cache
        if not hasattr(cache0, "k_caches") or cache0.max_batch_size != 1:
            return None

        # 1. Quantized runner (MARLIN Q4 / Q4_K_M / Legacy Q4_G64 / Q8)
        if self._quant_mode in (
            "q4",
            "q4_k_m",
            "q4_rtn",
            "q4_legacy",
            "q8",
        ) and hasattr(_METAL_MOD, "MPSQuantizedDecodeRunner"):
            if not self._can_use_quant_forward():
                return None
            k_ptr = (
                cache0.k_caches[0].data_ptr()
                if hasattr(cache0, "k_caches") and cache0.k_caches
                else None
            )
            runner_mode_key = (self._quant_mode, getattr(self, "_lm_head_q8", None))
            if (
                self._mps_quant_cache_id != id(cache0)
                or getattr(self, "_mps_quant_cache_ptr", None) != k_ptr
                or getattr(self, "_mps_quant_mode", None) != runner_mode_key
                or self._mps_quant_decode_runner is None
            ):
                qw = self._ensure_quant_weights(device)
                w_dtype = torch.float16
                use_q4 = self._quant_mode in ("q4", "q4_k_m", "q4_rtn", "q4_legacy")

                ln1_ws = [
                    b.ln_1.weight.to(device=device, dtype=w_dtype).contiguous()
                    for b in self.transformer.h
                ]
                ln2_ws = [
                    b.ln_2.weight.to(device=device, dtype=w_dtype).contiguous()
                    for b in self.transformer.h
                ]
                lnf_w = self.transformer.ln_f.weight.to(
                    device=device, dtype=w_dtype
                ).contiguous()
                c_attn_bs = [
                    b_ten if b_ten.numel() > 0 else torch.Tensor()
                    for b_ten in qw["c_attn_b"]
                ]
                n_kv_head = (
                    self.config.n_kv_head
                    if self.config.n_kv_head is not None
                    else self.config.n_head
                )

                eps_val = getattr(
                    self.transformer.ln_f, "eps", self.config.rms_norm_eps
                )
                self._mps_quant_decode_runner = _METAL_MOD.MPSQuantizedDecodeRunner(
                    qw["wte_emb"],
                    qw["lmh_p"],
                    qw["lmh_sb"],
                    self.freqs_real.to(device=device, dtype=torch.float32).contiguous(),
                    ln1_ws,
                    qw["c_attn_p"],
                    qw["c_attn_s"],
                    qw["c_proj_p"],
                    qw["c_proj_s"],
                    ln2_ws,
                    qw["gu_p"],
                    qw["gu_s"],
                    qw["wd_p"],
                    qw["wd_s"],
                    lnf_w,
                    cache0.k_caches,
                    cache0.v_caches,
                    self.config.n_head,
                    n_kv_head,
                    head_dim,
                    use_q4,
                    eps_val,
                    c_attn_bs,
                    self.config.vocab_size,
                    qw.get("wte_p", torch.Tensor()),
                    qw.get("wte_sb", torch.Tensor()),
                )
                self._mps_quant_cache_id = id(cache0)
                self._mps_quant_cache_ptr = k_ptr
                self._mps_quant_mode = runner_mode_key
            return self._mps_quant_decode_runner

        # 2. Standard FP16 / FP32 MPSDecodeRunner
        if hasattr(_METAL_MOD, "MPSDecodeRunner"):
            w_dtype = self.transformer.wte.weight.dtype
            k_ptr = (
                cache0.k_caches[0].data_ptr()
                if hasattr(cache0, "k_caches") and cache0.k_caches
                else None
            )
            if (
                self._mps_decode_cache_id != id(cache0)
                or getattr(self, "_mps_decode_cache_ptr", None) != k_ptr
                or self._mps_decode_dtype != w_dtype
                or self._mps_decode_runner is None
            ):
                ln1_ws = []
                c_attn_ws = []
                c_attn_bs = []
                c_proj_ws = []
                ln2_ws = []
                gu_ws = []
                down_ws = []
                for b in self.transformer.h:
                    if (
                        b.ln_1._cached_weight is None
                        or b.ln_1._cached_weight.device != device
                        or b.ln_1._cached_weight.dtype != w_dtype
                    ):
                        b.ln_1._cached_weight = b.ln_1.weight.to(
                            device=device, dtype=w_dtype
                        ).contiguous()
                    if (
                        b.ln_2._cached_weight is None
                        or b.ln_2._cached_weight.device != device
                        or b.ln_2._cached_weight.dtype != w_dtype
                    ):
                        b.ln_2._cached_weight = b.ln_2.weight.to(
                            device=device, dtype=w_dtype
                        ).contiguous()
                    if (
                        b.mlp._cached_gate_up_w is None
                        or b.mlp._cached_gate_up_w.device != device
                        or b.mlp._cached_gate_up_w.dtype != w_dtype
                    ):
                        b.mlp._cached_gate_up_w = torch.cat(
                            [b.mlp.w_gate.weight, b.mlp.w_up.weight], dim=0
                        ).to(device=device, dtype=w_dtype)
                    ln1_ws.append(b.ln_1._cached_weight)
                    c_attn_ws.append(b.attn.c_attn.weight)
                    if b.attn.c_attn.bias is not None:
                        c_attn_bs.append(
                            b.attn.c_attn.bias.to(
                                device=device, dtype=w_dtype
                            ).contiguous()
                        )
                    else:
                        c_attn_bs.append(torch.Tensor())
                    c_proj_ws.append(b.attn.c_proj.weight)
                    ln2_ws.append(b.ln_2._cached_weight)
                    gu_ws.append(b.mlp._cached_gate_up_w)
                    down_ws.append(b.mlp.w_down.weight)

                lnf_mod = self.transformer["ln_f"]
                if (
                    lnf_mod._cached_weight is None
                    or lnf_mod._cached_weight.device != device
                    or lnf_mod._cached_weight.dtype != w_dtype
                ):
                    lnf_mod._cached_weight = lnf_mod.weight.to(
                        device=device, dtype=w_dtype
                    ).contiguous()

                n_kv_head = (
                    self.config.n_kv_head
                    if self.config.n_kv_head is not None
                    else self.config.n_head
                )
                lmh_w = self.lm_head.weight.to(
                    device=device, dtype=w_dtype
                ).contiguous()
                self._mps_decode_runner = _METAL_MOD.MPSDecodeRunner(
                    self.transformer.wte.weight,
                    self.freqs_real.to(device=device, dtype=torch.float32).contiguous(),
                    ln1_ws,
                    c_attn_ws,
                    c_proj_ws,
                    ln2_ws,
                    gu_ws,
                    down_ws,
                    lnf_mod._cached_weight,
                    cache0.k_caches,
                    cache0.v_caches,
                    self.config.n_head,
                    n_kv_head,
                    head_dim,
                    lnf_mod.eps,
                    lmh_w,
                    c_attn_bs,
                )
                self._mps_decode_cache_id = id(cache0)
                self._mps_decode_cache_ptr = k_ptr
                self._mps_decode_dtype = w_dtype
            return self._mps_decode_runner

        return None

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
        return_all_logits: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor | None]
        | tuple[
            torch.Tensor, torch.Tensor | None, list[tuple[torch.Tensor, torch.Tensor]]
        ]
    ):
        device = idx.device
        B, T = idx.size()

        # Ultra-fast C++ Whole-Model Decode / Speculative Verification Graph Runner for T in [1, 32], B=1 on MPS
        if (
            1 <= T <= 32
            and B == 1
            and targets is None
            and not self.training
            and not torch.is_grad_enabled()
            and device.type == "mps"
            and kv_caches is not None
            and len(kv_caches) == self.config.n_layer
            and hasattr(kv_caches[0], "total_seen_tokens")
            and (T == 1 or kv_caches[0].total_seen_tokens[0] > 0)
        ):
            cache0 = kv_caches[0]
            runner = self.get_decode_runner(cache0)
            if runner is not None:
                seen = cache0.total_seen_tokens[0]
                max_seq = cache0.max_seq_len
                write_pos = (
                    (seen % max_seq)
                    if (cache0.ring_buffer or seen >= max_seq)
                    else seen
                )
                new_seen = seen + T
                active_len = min(new_seen, max_seq)
                rope_pos = seen % self.freqs_real.size(0)

                if T == 1:
                    logits = runner.step(idx, rope_pos, write_pos, active_len)
                else:
                    logits = runner.verify_step(idx, rope_pos, write_pos, active_len)
                if not return_all_logits and T > 1:
                    logits = logits[:, -1:, :]
                n_L = self.config.n_layer
                cache0.total_seen_tokens[:] = [new_seen] * n_L
                cache0.layer_seq_lens[:] = [active_len] * n_L
                return logits, None, kv_caches

        if targets is None and self._can_use_quant_forward():
            logits, _, new_kv_caches = self._forward_quantized_sequence(
                idx, kv_caches=kv_caches, return_all_logits=return_all_logits
            )
            if kv_caches is not None:
                return logits, None, new_kv_caches
            return logits, None

        if self.config.pos_emb == "learned":
            assert (
                T <= self.config.block_size
            ), f"Sequence length {T} exceeds block size {self.config.block_size}"
            pos = torch.arange(0, T, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)
            tok_emb = self.transformer.wte(idx)
            x = tok_emb + pos_emb
        else:
            x = self.transformer.wte(idx)

        new_kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []

        for i, block in enumerate(self.transformer.h):
            block_kv = (
                kv_caches[i] if (kv_caches is not None and i < len(kv_caches)) else None
            )
            if self.config.grad_checkpoint and self.training:
                # Custom gradient checkpointing forward
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                x, new_kv = torch_checkpoint(
                    create_custom_forward(block),
                    x,
                    self.freqs_cis,
                    block_kv,
                    use_reentrant=False,
                )
            else:
                freqs_arg = (
                    self.freqs_real
                    if (self.config.use_fused_kernels and self.freqs_real is not None)
                    else self.freqs_cis
                )
                x, new_kv = block(
                    x,
                    freqs_cis=freqs_arg,
                    kv_cache=block_kv,
                    use_cache=(kv_caches is not None),
                )
            if new_kv is not None:
                new_kv_caches.append(new_kv)

        if targets is not None:
            x = self.transformer.ln_f(x)
            logits = self.lm_head(x)
            loss = fused_cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            if kv_caches is not None and not return_all_logits:
                x_last = x if T == 1 else x[:, -1:, :]
                logits = self.lm_head(self.transformer.ln_f(x_last))
            else:
                logits = self.lm_head(self.transformer.ln_f(x))
            loss = None

        if kv_caches is not None:
            return logits, loss, new_kv_caches
        return logits, loss

    def configure_optimizers(
        self,
        weight_decay: float = 0.1,
        learning_rate: float = 6e-4,
        device: str = "cpu",
        optimizer_type: str = "adamw",
        muon_lr: float = 0.02,
        muon_momentum: float = 0.95,
    ) -> list[torch.optim.Optimizer]:
        """
        Configures optimizers with support for Muon (2D Matrix Newton-Schulz) + AdamW dual routing.
        """
        from ..optim.muon import Muon

        if optimizer_type == "muon":
            muon_params = []
            adamw_decay_params = []
            adamw_nodecay_params = []

            for name, p in self.named_parameters():
                if not p.requires_grad:
                    continue
                # Route 2D linear weight matrices to Muon
                if (
                    p.ndim == 2
                    and "wte" not in name
                    and "wpe" not in name
                    and "lm_head" not in name
                ):
                    muon_params.append(p)
                elif "wte" in name or "wpe" in name or "lm_head" in name:
                    adamw_decay_params.append(p)
                elif p.ndim < 2 or "ln" in name or "norm" in name or "bias" in name:
                    adamw_nodecay_params.append(p)
                else:
                    adamw_decay_params.append(p)

            adamw_groups = [
                {"params": adamw_decay_params, "weight_decay": weight_decay},
                {"params": adamw_nodecay_params, "weight_decay": 0.0},
            ]

            use_fused = device == "cuda"
            adamw_extra_args = dict(fused=True) if use_fused else dict()
            optimizer_adamw = torch.optim.AdamW(
                adamw_groups,
                lr=learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8,
                **adamw_extra_args,
            )
            optimizer_muon = Muon(muon_params, lr=muon_lr, momentum=muon_momentum)

            return [optimizer_muon, optimizer_adamw]
        else:
            param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ]
            use_fused = device == "cuda"
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = torch.optim.AdamW(
                optim_groups,
                lr=learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8,
                **extra_args,
            )
            return [optimizer]

    @classmethod
    def from_pretrained(cls, model_type: str = "gpt2") -> "Transformer":
        """Loads official pretrained GPT-2 weights from Hugging Face."""
        from transformers import GPT2LMHeadModel

        config = ModelConfig(
            block_size=1024,
            vocab_size=50257,
            n_layer=12,
            n_head=12,
            n_embd=768,
            n_kv_head=None,
            norm_type="layernorm",
            pos_emb="learned",
            mlp_type="gelu",
            bias=True,
        )
        model = cls(config)
        sd = model.state_dict()

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [
            k
            for k in sd_hf.keys()
            if not k.endswith(".attn.masked_bias") and not k.endswith(".attn.bias")
        ]

        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model


# Backward compatibility alias
GPT = Transformer
