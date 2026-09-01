"""
AxiomLM Transformer Architecture & Model Configuration.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union, Dict, Any, cast
import inspect
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
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
    HAS_CUSTOM_KERNELS = True
except (ImportError, ValueError):
    try:
        from kernels import FusedRMSNorm, FusedSwiGLUMLP
        HAS_CUSTOM_KERNELS = True
    except ImportError:
        HAS_CUSTOM_KERNELS = False
        FusedRMSNorm = None  # type: ignore
        FusedSwiGLUMLP = None  # type: ignore


@dataclass
class ModelConfig:
    """
    Configuration specification for AxiomLM Transformer models.
    Supports both Classic GPT-2 and Modern LLaMA-3 architectures.
    """
    block_size: int = 1024
    vocab_size: int = 50304      # Padded to multiple of 64 for SIMD/Tensor Core efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_kv_head: Optional[int] = None # GQA: None for full MHA, or int (e.g. 4)
    norm_type: str = "rmsnorm"   # "rmsnorm" (modern) or "layernorm" (classic)
    pos_emb: str = "rope"        # "rope" (modern) or "learned" (classic)
    mlp_type: str = "swiglu"     # "swiglu" (modern) or "gelu" (classic)
    bias: bool = False           # Modern architectures omit linear biases
    rope_theta: float = 10000.0  # Base frequency for RoPE
    use_fused_kernels: bool = False # Use fused Triton / Metal / NEON operators
    grad_checkpoint: bool = False   # Enable gradient checkpointing
    arch: Optional[str] = None      # Optional convenience preset ("modern" or "classic")

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
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm_type = config.norm_type
        self.mlp_type = config.mlp_type
        self.use_fused_kernels = config.use_fused_kernels and HAS_CUSTOM_KERNELS

        # 1. Pre-Attention Normalization
        if self.use_fused_kernels and self.norm_type == "rmsnorm" and FusedRMSNorm is not None:
            self.ln_1 = FusedRMSNorm(config.n_embd)
        elif self.norm_type == "rmsnorm":
            self.ln_1 = RMSNorm(config.n_embd)
        else:
            self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)

        # 2. Attention
        self.attn = CausalSelfAttention(config)

        # 3. Pre-MLP Normalization
        if self.use_fused_kernels and self.norm_type == "rmsnorm" and FusedRMSNorm is not None:
            self.ln_2 = FusedRMSNorm(config.n_embd)
        elif self.norm_type == "rmsnorm":
            self.ln_2 = RMSNorm(config.n_embd)
        else:
            self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)

        # 4. Feed-Forward Network
        if self.use_fused_kernels and self.mlp_type == "swiglu" and FusedSwiGLUMLP is not None:
            self.mlp = FusedSwiGLUMLP(config.n_embd, bias=config.bias)
        elif self.mlp_type == "swiglu":
            self.mlp = SwiGLUMLP(config.n_embd, bias=config.bias)
        else:
            self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, new_kv_cache = self.attn(self.ln_1(x), freqs_cis=freqs_cis, kv_cache=kv_cache, use_cache=use_cache)
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

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))

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
            freqs = precompute_rope_frequencies(head_dim=head_dim, max_seq_len=config.block_size, theta=config.rope_theta)
            self.register_buffer("freqs_cis", freqs, persistent=False)
        else:
            self.freqs_cis = None

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
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
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, Optional[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]]:
        device = idx.device
        B, T = idx.size()

        if self.config.pos_emb == "learned":
            assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"
            pos = torch.arange(0, T, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)
            tok_emb = self.transformer.wte(idx)
            x = tok_emb + pos_emb
        else:
            x = self.transformer.wte(idx)

        new_kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for i, block in enumerate(self.transformer.h):
            block_kv = kv_caches[i] if (kv_caches is not None and i < len(kv_caches)) else None
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
                x, new_kv = block(x, freqs_cis=self.freqs_cis, kv_cache=block_kv, use_cache=(kv_caches is not None))
            if new_kv is not None:
                new_kv_caches.append(new_kv)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            if kv_caches is not None:
                logits = self.lm_head(x[:, [-1], :])
            else:
                logits = self.lm_head(x)
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
                if p.ndim == 2 and "wte" not in name and "wpe" not in name and "lm_head" not in name:
                    muon_params.append(p)
                elif "wte" in name or "wpe" in name or "lm_head" in name:
                    adamw_decay_params.append(p)
                elif p.ndim < 2 or "ln" in name or "norm" in name or "bias" in name:
                    adamw_nodecay_params.append(p)
                else:
                    adamw_decay_params.append(p)

            adamw_groups = [
                {'params': adamw_decay_params, 'weight_decay': weight_decay},
                {'params': adamw_nodecay_params, 'weight_decay': 0.0},
            ]

            use_fused = (device == "cuda")
            adamw_extra_args = dict(fused=True) if use_fused else dict()
            optimizer_adamw = torch.optim.AdamW(adamw_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, **adamw_extra_args)
            optimizer_muon = Muon(muon_params, lr=muon_lr, momentum=muon_momentum)

            return [optimizer_muon, optimizer_adamw]
        else:
            param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            use_fused = (device == "cuda")
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, **extra_args)
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
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
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
