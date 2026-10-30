"""
AxiomLM Transformer Architecture & Model Configuration.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Union, Any
import torch
import torch.nn as nn
from ..kernels.ops import fused_cross_entropy
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .modules import (
    RMSNorm,
    SwiGLUMLP,
    MLP,
    precompute_rope_frequencies,
)
from .attention import CausalSelfAttention

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
    n_kv_head: Optional[int] = None  # GQA: None for full MHA, or int (e.g. 4)
    norm_type: str = "rmsnorm"  # "rmsnorm" (modern) or "layernorm" (classic)
    pos_emb: str = "rope"  # "rope" (modern) or "learned" (classic)
    mlp_type: str = "swiglu"  # "swiglu" (modern) or "gelu" (classic)
    bias: bool = False  # Modern architectures omit linear biases
    rope_theta: float = 10000.0  # Base frequency for RoPE
    sliding_window: Optional[int] = None  # Sliding Window Attention (SWA) size
    use_fused_kernels: bool = False  # Use fused Triton / Metal / NEON operators
    grad_checkpoint: bool = False  # Enable gradient checkpointing
    arch: Optional[str] = None  # Optional convenience preset ("modern" or "classic")
    quant_mode: Optional[str] = (
        None  # Quantized decode mode: None (FP16), "q4" (Q4_G64), or "q8" (Q8_G64)
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
            self.ln_1 = FusedRMSNorm(config.n_embd)
        elif self.norm_type == "rmsnorm":
            self.ln_1 = RMSNorm(config.n_embd)
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
            self.ln_2 = FusedRMSNorm(config.n_embd)
        elif self.norm_type == "rmsnorm":
            self.ln_2 = RMSNorm(config.n_embd)
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
        freqs_cis: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
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

    def __init__(self, config: ModelConfig):
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
            self.transformer["ln_f"] = FusedRMSNorm(config.n_embd)
        elif config.norm_type == "rmsnorm":
            self.transformer["ln_f"] = RMSNorm(config.n_embd)
        else:
            self.transformer["ln_f"] = nn.LayerNorm(config.n_embd, bias=config.bias)

        # Output projection head (weight-tied to token embedding table)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # Precompute complex RoPE frequencies
        if config.pos_emb == "rope":
            head_dim = config.n_embd // config.n_head
            freqs = precompute_rope_frequencies(
                head_dim=head_dim,
                max_seq_len=config.block_size,
                theta=config.rope_theta,
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

        # Initialize weights
        self.apply(self._init_weights)
        self._mps_decode_cache_id: Optional[int] = None
        self._mps_decode_dtype: Optional[torch.dtype] = None
        self._mps_decode_runner: Optional[Any] = None
        self._quant_mode: Optional[str] = config.quant_mode
        self._mps_quant_decode_runner: Optional[Any] = None
        self._mps_quant_cache_id: Optional[int] = None
        self._quant_weights_cache: Optional[dict] = None

    def quantize(self, mode: Optional[str] = "q4"):
        """
        Enables register-level quantized decode (mode='q4' for 4-bit, 'q8' for 8-bit, None for FP16).
        """
        assert mode in [
            None,
            "q4",
            "q8",
        ], f"Invalid quant mode: {mode}. Choose 'q4', 'q8', or None."
        self._quant_mode = mode
        self._mps_quant_decode_runner = None
        self._mps_quant_cache_id = None
        self._quant_weights_cache = None
        return self

    def train(self, mode: bool = True):
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

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        return_all_logits: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[
            torch.Tensor,
            Optional[torch.Tensor],
            List[Tuple[torch.Tensor, torch.Tensor]],
        ],
    ]:
        device = idx.device
        B, T = idx.size()

        # Ultra-fast C++ Whole-Model Decode / Speculative Verification Graph Runner for T in [1, 8], B=1 on MPS
        if (
            1 <= T <= 8
            and B == 1
            and targets is None
            and not self.training
            and not torch.is_grad_enabled()
            and self.config.use_fused_kernels
            and _METAL_MOD is not None
            and device.type == "mps"
            and self.config.pos_emb == "rope"
            and self.config.norm_type == "rmsnorm"
            and self.config.mlp_type == "swiglu"
            and not self.config.bias
            and self.config.sliding_window is None
            and kv_caches is not None
            and len(kv_caches) == self.config.n_layer
            and hasattr(kv_caches[0], "k_caches")
            and kv_caches[0].max_batch_size == 1
            and (T == 1 or kv_caches[0].total_seen_tokens[0] > 0)
        ):
            cache0 = kv_caches[0]
            seen = cache0.total_seen_tokens[0]
            max_seq = cache0.max_seq_len
            write_pos = (
                (seen % max_seq) if (cache0.ring_buffer or seen >= max_seq) else seen
            )
            new_seen = seen + T
            active_len = new_seen if new_seen <= max_seq else max_seq
            rope_pos = seen % self.freqs_real.size(0)

            # Ultra-fast Quantized Metal GEMV Decode Runner (Q4_G64 / Q8_G64)
            if self._quant_mode in ["q4", "q8"] and hasattr(
                _METAL_MOD, "MPSQuantizedDecodeRunner"
            ):
                if (
                    self._mps_quant_cache_id != id(cache0)
                    or self._mps_quant_decode_runner is None
                ):
                    from ..kernels.quant import (
                        pack_linear_q4_g64,
                        pack_linear_q8_g64,
                    )

                    pack_fn = (
                        pack_linear_q4_g64
                        if self._quant_mode == "q4"
                        else pack_linear_q8_g64
                    )
                    use_q4 = self._quant_mode == "q4"
                    w_dtype = torch.float16

                    if (
                        self._quant_weights_cache is None
                        or self._quant_weights_cache.get("mode") != self._quant_mode
                    ):
                        wte_emb = self.transformer.wte.weight.to(
                            device=device, dtype=w_dtype
                        ).contiguous()
                        lmh_p, lmh_sb = pack_fn(self.transformer.wte.weight.cpu())
                        lmh_p, lmh_sb = lmh_p.to(device), lmh_sb.to(device)

                        c_attn_p, c_attn_s = [], []
                        c_proj_p, c_proj_s = [], []
                        gu_p, gu_s = [], []
                        wd_p, wd_s = [], []

                        for b in self.transformer.h:
                            p, s = pack_fn(b.attn.c_attn.weight.cpu())
                            c_attn_p.append(p.to(device))
                            c_attn_s.append(s.to(device))

                            p, s = pack_fn(b.attn.c_proj.weight.cpu())
                            c_proj_p.append(p.to(device))
                            c_proj_s.append(s.to(device))

                            gate_up_w = torch.cat(
                                [b.mlp.w_gate.weight, b.mlp.w_up.weight], dim=0
                            )
                            p, s = pack_fn(gate_up_w.cpu())
                            gu_p.append(p.to(device))
                            gu_s.append(s.to(device))

                            p, s = pack_fn(b.mlp.w_down.weight.cpu())
                            wd_p.append(p.to(device))
                            wd_s.append(s.to(device))

                        self._quant_weights_cache = {
                            "mode": self._quant_mode,
                            "wte_emb": wte_emb,
                            "lmh_p": lmh_p,
                            "lmh_sb": lmh_sb,
                            "c_attn_p": c_attn_p,
                            "c_attn_s": c_attn_s,
                            "c_proj_p": c_proj_p,
                            "c_proj_s": c_proj_s,
                            "gu_p": gu_p,
                            "gu_s": gu_s,
                            "wd_p": wd_p,
                            "wd_s": wd_s,
                        }

                    qw = self._quant_weights_cache
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
                    n_kv_head = (
                        self.config.n_kv_head
                        if self.config.n_kv_head is not None
                        else self.config.n_head
                    )
                    head_dim = self.config.n_embd // self.config.n_head

                    eps_val = getattr(self.transformer.ln_f, "eps", 1e-5)
                    self._mps_quant_decode_runner = _METAL_MOD.MPSQuantizedDecodeRunner(
                        qw["wte_emb"],
                        qw["lmh_p"],
                        qw["lmh_sb"],
                        self.freqs_real.to(device=device, dtype=w_dtype).contiguous(),
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
                    )
                    self._mps_quant_cache_id = id(cache0)

                if T == 1:
                    logits = self._mps_quant_decode_runner.step(
                        idx, rope_pos, write_pos, active_len
                    )
                else:
                    logits = self._mps_quant_decode_runner.verify_step(
                        idx, rope_pos, write_pos, active_len
                    )
                if not return_all_logits and T > 1:
                    logits = logits[:, -1:, :]
                n_L = self.config.n_layer
                cache0.total_seen_tokens[:] = [new_seen] * n_L
                cache0.layer_seq_lens[:] = [active_len] * n_L
                return logits, None, kv_caches

            # Standard FP16 MPSDecodeRunner
            w_dtype = self.transformer.wte.weight.dtype
            if (
                self._mps_decode_cache_id != id(cache0)
                or self._mps_decode_dtype != w_dtype
            ):
                ln1_ws = []
                c_attn_ws = []
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
                head_dim = self.config.n_embd // self.config.n_head
                self._mps_decode_runner = _METAL_MOD.MPSDecodeRunner(
                    self.transformer.wte.weight,
                    self.freqs_real,
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
                )
                self._mps_decode_cache_id = id(cache0)
                self._mps_decode_dtype = w_dtype

            if T == 1:
                logits = self._mps_decode_runner.step(
                    idx, rope_pos, write_pos, active_len
                )
            else:
                logits = self._mps_decode_runner.verify_step(
                    idx, rope_pos, write_pos, active_len
                )
            if not return_all_logits and T > 1:
                logits = logits[:, -1:, :]
            n_L = self.config.n_layer
            cache0.total_seen_tokens[:] = [new_seen] * n_L
            cache0.layer_seq_lens[:] = [active_len] * n_L
            return logits, None, kv_caches

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

        new_kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []

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
    ) -> List[torch.optim.Optimizer]:
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
