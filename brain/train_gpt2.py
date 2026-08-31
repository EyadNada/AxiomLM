import os
import math
import time
import inspect
from dataclasses import dataclass
from typing import cast, Callable, Any, overload
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import tiktoken
import numpy as np

try:
    from kernels import FusedRMSNorm, FusedSwiGLUMLP
    HAS_CUSTOM_KERNELS = True
except ImportError:
    HAS_CUSTOM_KERNELS = False
    FusedRMSNorm = None  # type: ignore
    FusedSwiGLUMLP = None  # type: ignore


# -----------------------------------------------------------------------------
# Modern Architecture Modules (LLaMA-3 / Mistral Spec)
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm).
    Replaces LayerNorm by removing mean centering, improving throughput and numerical stability.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def precompute_rope_frequencies(head_dim: int, max_seq_len: int = 2048, theta: float = 10000.0) -> torch.Tensor:
    """
    Precomputes complex rotary frequency tensors for Rotary Position Embeddings (RoPE).
    Returns complex64 tensor of shape (max_seq_len, head_dim // 2).
    """
    assert head_dim % 2 == 0, f"head_dim ({head_dim}) must be even for complex RoPE"
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    """
    Applies rotary position embeddings (RoPE) to query or key tensors.
    x shape: (B, nh, T, head_dim)
    """
    B, nh, T, hs = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(B, nh, T, hs // 2, 2))
    freqs_cis_slice = freqs_cis[start_pos : start_pos + T].unsqueeze(0).unsqueeze(0).to(x.device)
    x_rotated = torch.view_as_real(x_complex * freqs_cis_slice).reshape(B, nh, T, hs)
    return x_rotated.type_as(x)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Broadcasts key/value heads for Grouped-Query Attention (GQA).
    x: (B, n_kv_head, T, head_dim) -> (B, n_head, T, head_dim)
    """
    if n_rep == 1:
        return x
    B, n_kv_head, T, hs = x.shape
    return x[:, :, None, :, :].expand(B, n_kv_head, n_rep, T, hs).reshape(B, n_kv_head * n_rep, T, hs)


class SwiGLUMLP(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU) Feed-Forward Network.
    Uses dimension scaling 2/3 * 4d aligned to multiple of 64.
    """
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(2 * (4 * config.n_embd) / 3)
        hidden_dim = 64 * ((hidden_dim + 64 - 1) // 64)
        self.w_gate = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w_up = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w_down = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        setattr(self.w_down, "NANOGPT_SCALE_INIT", 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# -----------------------------------------------------------------------------
# Unified Transformer Attention & Block Layers
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head for GQA"
        self.n_rep = self.n_head // self.n_kv_head
        self.head_dim = config.n_embd // config.n_head

        # Projections
        if self.n_head == self.n_kv_head:
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
            self.separate_qkv = False
        else:
            # GQA: separate projections for Query and reduced-dimension Key/Value
            self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.bias)
            self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
            self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
            self.separate_qkv = True

        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        setattr(self.c_proj, "NANOGPT_SCALE_INIT", 1)

    def forward(self, x, freqs_cis=None, start_pos=0, kv_cache=None, use_cache=False):
        B, T, C = x.size()

        if not self.separate_qkv:
            qkv = self.c_attn(x)
            q, k, v = qkv.split(C, dim=2)
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        else:
            q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Apply Rotary Position Embeddings (RoPE) if enabled
        if freqs_cis is not None:
            q = apply_rope(q, freqs_cis, start_pos=start_pos)
            k = apply_rope(k, freqs_cis, start_pos=start_pos)

        # KV-Cache management
        if kv_cache is not None and kv_cache[0] is not None and kv_cache[1] is not None:
            k_past, v_past = kv_cache
            k = torch.cat([k_past, k], dim=2)
            v = torch.cat([v_past, v], dim=2)

        new_kv_cache = (k, v) if (use_cache or kv_cache is not None) else None

        # Repeat KV heads for Grouped-Query Attention (GQA)
        k_rep = repeat_kv(k, self.n_rep)
        v_rep = repeat_kv(v, self.n_rep)

        if kv_cache is None or kv_cache[0] is None:
            y = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
        else:
            is_causal = (T > 1)
            y = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=is_causal)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y, new_kv_cache


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        setattr(self.c_proj, "NANOGPT_SCALE_INIT", 1)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Normalization layer (LayerNorm vs RMSNorm vs FusedRMSNorm)
        if config.norm_type == "rmsnorm":
            if getattr(config, "use_fused_kernels", False) and FusedRMSNorm is not None:
                self.ln_1 = FusedRMSNorm(config.n_embd)
                self.ln_2 = FusedRMSNorm(config.n_embd)
            else:
                self.ln_1 = RMSNorm(config.n_embd)
                self.ln_2 = RMSNorm(config.n_embd)
        else:
            self.ln_1 = nn.LayerNorm(config.n_embd)
            self.ln_2 = nn.LayerNorm(config.n_embd)

        self.attn = CausalSelfAttention(config)

        # Feed-Forward layer (GELU MLP vs SwiGLU vs FusedSwiGLU)
        if config.mlp_type == "swiglu":
            if getattr(config, "use_fused_kernels", False) and FusedSwiGLUMLP is not None:
                self.mlp = FusedSwiGLUMLP(config)
            else:
                self.mlp = SwiGLUMLP(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x, freqs_cis=None, start_pos=0, kv_cache=None, use_cache=False):
        attn_out, new_kv_cache = self.attn(
            self.ln_1(x),
            freqs_cis=freqs_cis,
            start_pos=start_pos,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv_cache


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    # Modern Architecture Options (LLaMA-3 Spec)
    n_kv_head: int | None = None          # None for classic MHA, or e.g. 4 for 3x GQA
    norm_type: str = "layernorm"          # "layernorm" (classic) or "rmsnorm" (modern)
    pos_emb: str = "learned"              # "learned" (classic WPE) or "rope" (modern Rotary)
    mlp_type: str = "gelu"                # "gelu" (classic) or "swiglu" (modern)
    rope_theta: float = 10000.0           # Base frequency for RoPE
    bias: bool = True                     # Set False for modern architectures (LLaMA / Mistral)
    use_fused_kernels: bool = False       # Enable low-level fused GPU/SIMD kernels
    grad_checkpoint: bool = False         # Trade compute for activation memory reduction


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Precompute RoPE rotary frequencies if enabled
        if config.pos_emb == "rope":
            head_dim = config.n_embd // config.n_head
            freqs = precompute_rope_frequencies(
                head_dim=head_dim,
                max_seq_len=config.block_size * 2,
                theta=config.rope_theta,
            )
            self.register_buffer("freqs_cis", freqs, persistent=False)
        else:
            self.freqs_cis = None

        if config.norm_type == "rmsnorm":
            ln_f = FusedRMSNorm(config.n_embd) if (config.use_fused_kernels and FusedRMSNorm is not None) else RMSNorm(config.n_embd)
        else:
            ln_f = nn.LayerNorm(config.n_embd)

        # Core transformer dictionary
        transformer_dict = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=ln_f,
        )
        if config.pos_emb == "learned":
            transformer_dict['wpe'] = nn.Embedding(config.block_size, config.n_embd)

        self.transformer = nn.ModuleDict(transformer_dict)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying scheme
        cast(nn.Embedding, self.transformer['wte']).weight = self.lm_head.weight

        # Initialize parameter weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, kv_caches=None, use_cache=False):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"

        past_len = 0
        if kv_caches is not None and kv_caches[0] is not None:
            past_len = kv_caches[0][0].size(2)

        tok_emb = self.transformer['wte'](idx)
        if self.config.pos_emb == "learned":
            pos = torch.arange(past_len, past_len + T, dtype=torch.long, device=idx.device)
            pos_emb = self.transformer['wpe'](pos)
            x = tok_emb + pos_emb
        else:
            x = tok_emb

        freqs_cis = self.freqs_cis if self.config.pos_emb == "rope" else None

        new_kv_caches = [] if (use_cache or kv_caches is not None) else None
        for i, block in enumerate(cast(nn.ModuleList, self.transformer['h'])):
            block_kv = kv_caches[i] if kv_caches is not None else None
            if self.config.grad_checkpoint and self.training and block_kv is None:
                def block_wrapper(t_in):
                    out, _ = block(t_in, freqs_cis=freqs_cis, start_pos=past_len)
                    return out
                x = torch.utils.checkpoint.checkpoint(block_wrapper, x, use_reentrant=False)
                updated_kv = None
            else:
                x, updated_kv = block(
                    x,
                    freqs_cis=freqs_cis,
                    start_pos=past_len,
                    kv_cache=block_kv,
                    use_cache=(use_cache or kv_caches is not None),
                )
            if new_kv_caches is not None:
                new_kv_caches.append(updated_kv)

        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        if use_cache or kv_caches is not None:
            return logits, loss, new_kv_caches
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 weights from Hugging Face."""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"Loading weights from pretrained GPT-2 ({model_type})")

        config_map = {
            'gpt2':         GPTConfig(n_layer=12, n_head=12, n_embd=768, vocab_size=50257, block_size=1024),
            'gpt2-medium':  GPTConfig(n_layer=24, n_head=16, n_embd=1024, vocab_size=50257, block_size=1024),
            'gpt2-large':   GPTConfig(n_layer=36, n_head=20, n_embd=1280, vocab_size=50257, block_size=1024),
            'gpt2-xl':      GPTConfig(n_layer=48, n_head=25, n_embd=1600, vocab_size=50257, block_size=1024),
        }
        config = config_map[model_type]
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith(('.attn.masked_bias', '.attn.bias'))]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"Mismatched state dict keys: {len(sd_keys_hf)} != {len(sd_keys)}"
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

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        device: str,
        optimizer_type: str = "adamw",
        muon_lr: float = 0.02,
    ):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and ('cuda' in device)

        if optimizer_type == "adamw":
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]

            num_decay = sum(p.numel() for p in decay_params)
            num_nodecay = sum(p.numel() for p in nodecay_params)
            print(f"[AdamW] Decayed parameter tensors: {len(decay_params)} ({num_decay:,} params)")
            print(f"[AdamW] Non-decayed parameter tensors: {len(nodecay_params)} ({num_nodecay:,} params)")
            print(f"[AdamW] Using fused AdamW: {use_fused}")

            optimizer = torch.optim.AdamW(
                optim_groups,
                lr=learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8,
                fused=use_fused
            )
            return [optimizer]

        elif optimizer_type == "muon":
            # Dual parameter routing:
            # 1. 2D internal weight matrices (Attention Q/K/V/Out, MLP projections) -> Muon
            # 2. Embedding matrices (wte, wpe) and 1D vectors (RMSNorm/LayerNorm weights, biases) -> AdamW
            embedding_names = {"transformer.wte.weight", "transformer.wpe.weight", "lm_head.weight"}

            muon_params = [p for n, p in param_dict.items() if p.dim() == 2 and n not in embedding_names]
            adamw_decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and n in embedding_names]
            adamw_nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

            num_muon = sum(p.numel() for p in muon_params)
            num_adamw_decay = sum(p.numel() for p in adamw_decay_params)
            num_adamw_nodecay = sum(p.numel() for p in adamw_nodecay_params)

            print(f"[Muon Hybrid] 2D Matrix tensors: {len(muon_params)} ({num_muon:,} params) -> Optimized with Muon (lr={muon_lr})")
            print(f"[Muon Hybrid] Embedding tensors: {len(adamw_decay_params)} ({num_adamw_decay:,} params) -> Optimized with AdamW (lr={learning_rate})")
            print(f"[Muon Hybrid] 1D Vector/Norm tensors: {len(adamw_nodecay_params)} ({num_adamw_nodecay:,} params) -> Optimized with AdamW (lr={learning_rate})")

            optimizer_muon = Muon(
                muon_params,
                lr=muon_lr,
                momentum=0.95,
                nesterov=True,
                ns_steps=5,
                weight_decay=0.0,
            )

            adamw_groups = [
                {'params': adamw_decay_params, 'weight_decay': weight_decay},
                {'params': adamw_nodecay_params, 'weight_decay': 0.0}
            ]
            optimizer_adamw = torch.optim.AdamW(
                adamw_groups,
                lr=learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8,
                fused=use_fused,
            )
            return [optimizer_muon, optimizer_adamw]
        else:
            raise ValueError(f"Unsupported optimizer_type: {optimizer_type}. Choose 'adamw' or 'muon'.")

# -----------------------------------------------------------------------------
# Next-Gen Matrix Optimizer (Muon with Newton-Schulz Polar Decomposition)
# -----------------------------------------------------------------------------

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Computes an approximate polar decomposition (orthogonal factor) using a quintic (5th-order)
    Newton-Schulz iteration:
        X_{k+1} = a * X_k + B @ X_k, where B = b * A + c * A^2, and A = X_k @ X_k^T
    Given matrix G, produces orthogonal matrix U such that U = G (G^T G)^{-1/2}.
    """
    assert len(G.shape) == 2, f"Expected 2D matrix, got shape {G.shape}"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.dtype == torch.bfloat16 else G.float()
    X = X / (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.type_as(G)


class Muon(torch.optim.Optimizer):
    """
    Muon (Momentum Orthogonalized by Newton-Schulz) Matrix Optimizer.
    Optimizes 2D linear weight matrices via normalized orthogonal updates.
    """
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                lr = group['lr']
                momentum = group['momentum']
                nesterov = group['nesterov']
                ns_steps = group['ns_steps']
                weight_decay = group['weight_decay']

                for p in group['params']:
                    if p.grad is None:
                        continue
                    g = p.grad
                    assert g.ndim == 2, f"Muon optimizer requires 2D matrix parameters, got shape {g.shape}"

                    if weight_decay > 0.0:
                        p.mul_(1.0 - lr * weight_decay)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)

                    if nesterov:
                        update_g = g.add(buf, alpha=momentum)
                    else:
                        update_g = buf

                    u = zeropower_via_newtonschulz5(update_g, steps=ns_steps)
                    scale = max(1.0, (p.size(0) / p.size(1)) ** 0.5)
                    p.add_(u, alpha=-lr * scale)

        return loss


# -----------------------------------------------------------------------------
# Data Loader
# -----------------------------------------------------------------------------

class DataLoaderLite:
    """
    High-Performance Multi-Shard Streaming Data Loader.
    Supports single-shard and multi-shard binary uint16 datasets with zero-copy np.memmap,
    deterministic step synchronization across shard boundaries, and multi-GPU DDP rank splitting.
    """
    def __init__(self, B: int, T: int, process_rank: int = 0, num_processes: int = 1, split: str = "train", data_dir: str = "data"):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        self.data_dir = data_dir
        assert split in {"train", "val"}, "split must be 'train' or 'val'"

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        abs_data_dir = data_dir if os.path.isabs(data_dir) else os.path.join(project_root, data_dir)

        # Discover binary shards: check multi-shard pattern first (e.g. train_0000.bin), then single shard (train.bin)
        import glob
        shards = sorted(glob.glob(os.path.join(abs_data_dir, f"{split}_*.bin")))
        if not shards:
            single_path = os.path.join(abs_data_dir, f"{split}.bin")
            if os.path.exists(single_path):
                shards = [single_path]

        self.shards = shards
        self.is_fallback = False

        if self.shards:
            self.shard_lengths = []
            for s_path in self.shards:
                m = np.memmap(s_path, dtype=np.uint16, mode='r')
                self.shard_lengths.append(len(m))
            self.total_tokens = sum(self.shard_lengths)
            print(f"[DataLoaderLite] Loaded {split} ({len(self.shards)} shard{'s' if len(self.shards) > 1 else ''}) from {abs_data_dir} ({self.total_tokens:,} tokens total)")
        else:
            input_path = os.path.join(project_root, "material", "input.txt")
            if not os.path.exists(input_path):
                input_path = "input.txt"
            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()
            enc = tiktoken.get_encoding("gpt2")
            self.tokens = np.array(enc.encode(text), dtype=np.uint16)
            self.total_tokens = len(self.tokens)
            self.shards = []
            self.shard_lengths = [self.total_tokens]
            self.is_fallback = True
            print(f"[DataLoaderLite] Loaded fallback text with {self.total_tokens:,} tokens")

        print(f"[DataLoaderLite] 1 epoch = {self.total_tokens // (B * T * num_processes)} batches")
        self.current_shard_idx = 0
        self._load_shard(0)
        self.reset()

    def _load_shard(self, shard_idx: int):
        """Loads and memory-maps a specific shard index."""
        if self.is_fallback:
            return
        self.current_shard_idx = shard_idx % len(self.shards)
        shard_path = self.shards[self.current_shard_idx]
        self.tokens = np.memmap(shard_path, dtype=np.uint16, mode='r')

    def reset(self):
        """Resets loader position to beginning of first shard."""
        self.current_shard_idx = 0
        self._load_shard(0)
        self.current_position = self.B * self.T * self.process_rank

    def set_step(self, step: int, grad_accum_steps: int = 1):
        """Fast-forwards data loader position across shard boundaries to match resumed training step."""
        tokens_per_step = self.B * self.T * self.num_processes * grad_accum_steps
        if self.total_tokens == 0:
            self.reset()
            return

        global_token_offset = (step * tokens_per_step + self.B * self.T * self.process_rank) % self.total_tokens

        if self.is_fallback or len(self.shards) <= 1:
            max_valid_pos = self.total_tokens - (self.B * self.T * self.num_processes + 1)
            if max_valid_pos > 0:
                self.current_position = global_token_offset % max_valid_pos
            else:
                self.reset()
            return

        # Find shard index and local offset within that shard
        accum_tokens = 0
        for idx, s_len in enumerate(self.shard_lengths):
            if accum_tokens + s_len > global_token_offset:
                self._load_shard(idx)
                local_offset = global_token_offset - accum_tokens
                max_shard_pos = s_len - (self.B * self.T * self.num_processes + 1)
                self.current_position = min(local_offset, max(0, max_shard_pos))
                return
            accum_tokens += s_len

        self.reset()

    def next_batch(self):
        """Yields next (B, T) batch of input x and target y shifted by 1 token."""
        B, T = self.B, self.T
        req_len = B * T * self.num_processes + 1

        # Rotate to next shard if current shard cannot supply a full batch
        if self.current_position + req_len > len(self.tokens):
            if not self.is_fallback and len(self.shards) > 1:
                next_shard = (self.current_shard_idx + 1) % len(self.shards)
                self._load_shard(next_shard)
            self.current_position = self.B * self.T * self.process_rank

        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf_torch = torch.from_numpy(buf.astype(np.int64))
        x = buf_torch[:-1].view(B, T)
        y = buf_torch[1:].view(B, T)
        self.current_position += B * T * self.num_processes

        if self.current_position + req_len > len(self.tokens):
            if not self.is_fallback and len(self.shards) > 1:
                next_shard = (self.current_shard_idx + 1) % len(self.shards)
                self._load_shard(next_shard)
            self.current_position = self.B * self.T * self.process_rank

        return x, y


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Model Unwrapping & Sampling Helpers
# -----------------------------------------------------------------------------

def get_raw_model(model: nn.Module) -> GPT:
    """Safely unwraps DDP (DistributedDataParallel) and torch.compile containers."""
    unwrapped = getattr(model, "module", getattr(model, "_orig_mod", model))
    return cast(GPT, unwrapped)


def sample_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = 50,
    top_p: float | None = None,
    min_p: float | None = None,
    repetition_penalty: float = 1.0,
    prev_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Vectorized logit sampling engine with support for:
    - Temperature scaling
    - Repetition penalty
    - Top-k truncation
    - Top-p (Nucleus) truncation
    - Min-p dynamic thresholding
    """
    logits = logits.clone()

    if repetition_penalty != 1.0 and prev_tokens is not None:
        for b in range(logits.size(0)):
            unique_tokens = torch.unique(prev_tokens[b])
            for tok in unique_tokens:
                if logits[b, tok] > 0:
                    logits[b, tok] /= repetition_penalty
                else:
                    logits[b, tok] *= repetition_penalty

    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        val, _ = torch.topk(logits, k)
        logits[logits < val[:, [-1]]] = -float('Inf')

    probs = F.softmax(logits, dim=-1)

    if min_p is not None and min_p > 0.0:
        p_max = probs.max(dim=-1, keepdim=True).values
        cutoff = p_max * min_p
        probs = torch.where(probs < cutoff, torch.zeros_like(probs), probs)
        probs_sum = probs.sum(dim=-1, keepdim=True)
        probs = torch.where(probs_sum > 0, probs / probs_sum, F.softmax(logits, dim=-1))

    if top_p is not None and top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        sorted_probs[sorted_indices_to_remove] = 0.0
        probs = torch.scatter(torch.zeros_like(probs), -1, sorted_indices, sorted_probs)
        probs_sum = probs.sum(dim=-1, keepdim=True)
        probs = torch.where(probs_sum > 0, probs / probs_sum, F.softmax(logits, dim=-1))

    next_tok = torch.multinomial(probs, 1)
    return next_tok


def generate_samples(
    model: GPT,
    enc,
    device: str,
    prompt: str = "Once upon a time",
    num_samples: int = 2,
    max_length: int = 40,
    temperature: float = 1.0,
    top_k: int | None = 50,
    top_p: float | None = None,
    min_p: float | None = None,
    repetition_penalty: float = 1.0,
):
    """Generates autoregressive text samples using standard eager re-computation (O(T^2))."""
    model.eval()
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0).repeat(num_samples, 1)

    while tokens.size(1) < max_length:
        with torch.no_grad():
            logits, _ = model(tokens)
            next_token = sample_logits(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                prev_tokens=tokens,
            )
            tokens = torch.cat((tokens, next_token), dim=1)

    samples = []
    for i in range(num_samples):
        sample_text = enc.decode(tokens[i, :max_length].tolist())
        samples.append(sample_text)
    return samples


def generate_with_cache(
    model: GPT,
    enc,
    device: str,
    prompt: str = "Once upon a time",
    num_samples: int = 1,
    max_length: int = 40,
    temperature: float = 1.0,
    top_k: int | None = 50,
    top_p: float | None = None,
    min_p: float | None = None,
    repetition_penalty: float = 1.0,
):
    """
    Accelerated autoregressive text generation using per-layer Key-Value caching.
    Reduces compute complexity from O(T^2) to O(1) per decoding step.
    """
    model.eval()
    prompt_tokens = enc.encode(prompt)
    x = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0).repeat(num_samples, 1)

    with torch.no_grad():
        # Prefill phase: pass entire prompt to initialize KV caches
        kv_caches = [None] * model.config.n_layer
        logits, _, kv_caches = model(x, kv_caches=kv_caches)

        next_token = sample_logits(
            logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            prev_tokens=x,
        )
        generated_tokens = torch.cat((x, next_token), dim=1)

        # Decode phase: pass single token (T=1) on each step with cached past
        while generated_tokens.size(1) < max_length:
            logits, _, kv_caches = model(next_token, kv_caches=kv_caches)
            next_token = sample_logits(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                prev_tokens=generated_tokens,
            )
            generated_tokens = torch.cat((generated_tokens, next_token), dim=1)

    samples = []
    for i in range(num_samples):
        sample_text = enc.decode(generated_tokens[i, :max_length].tolist())
        samples.append(sample_text)
    return samples


def benchmark_generation_speed(model: GPT, enc, device: str, prompt: str = "Once upon a time", max_length: int = 100):
    """Benchmarks generation throughput (tokens/sec) comparing Naive O(T^2) vs KV-Cache O(1)."""
    model.eval()
    print(f"\n[Axiom-LM Benchmark] Benchmarking generation to {max_length} tokens on {device}...")

    # Warmup
    _ = generate_samples(model, enc, device, prompt=prompt, num_samples=1, max_length=20)
    _ = generate_with_cache(model, enc, device, prompt=prompt, num_samples=1, max_length=20)

    # 1. Benchmark Naive Eager O(T^2)
    if device == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    elif device == "cuda" and hasattr(torch.cuda, "synchronize"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = generate_samples(model, enc, device, prompt=prompt, num_samples=1, max_length=max_length)
    if device == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    elif device == "cuda" and hasattr(torch.cuda, "synchronize"):
        torch.cuda.synchronize()
    t_naive = time.perf_counter() - t0

    # 2. Benchmark KV-Cache O(1)
    t0 = time.perf_counter()
    _ = generate_with_cache(model, enc, device, prompt=prompt, num_samples=1, max_length=max_length)
    if device == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    elif device == "cuda" and hasattr(torch.cuda, "synchronize"):
        torch.cuda.synchronize()
    t_cache = time.perf_counter() - t0

    prompt_len = len(enc.encode(prompt))
    tokens_generated = max_length - prompt_len
    tok_per_sec_naive = tokens_generated / t_naive
    tok_per_sec_cache = tokens_generated / t_cache
    speedup = t_naive / t_cache

    print(f"  • Naive Eager O(T^2)  : {t_naive*1000:.2f} ms ({tok_per_sec_naive:.2f} tokens/s)")
    print(f"  • KV-Cache O(1)       : {t_cache*1000:.2f} ms ({tok_per_sec_cache:.2f} tokens/s)")
    print(f"  • Speedup Factor      : {speedup:.2f}x faster with KV-Cache\n")
    return {"t_naive": t_naive, "t_cache": t_cache, "speedup": speedup}


# -----------------------------------------------------------------------------
# Hardware Peak Compute & MFU (Model FLOPs Utilization) Engine
# -----------------------------------------------------------------------------

def estimate_hardware_peak_tflops(device: str) -> float:
    """Estimates theoretical peak BF16/FP16 TFLOPs for the active hardware accelerator."""
    if device == "cuda":
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0).lower()
                if "h100" in gpu_name:
                    return 989.0
                elif "a100" in gpu_name:
                    return 312.0
                elif "4090" in gpu_name:
                    return 165.2
                elif "3090" in gpu_name:
                    return 71.0
                elif "4080" in gpu_name:
                    return 97.5
                elif "t4" in gpu_name:
                    return 65.0
                elif "v100" in gpu_name:
                    return 125.0
            except Exception:
                pass
        return 70.0
    elif device == "mps":
        # Apple Silicon MPS: ~10.0 TFLOPs base (M1/M2/M3/M4) / ~35.0 TFLOPs Pro/Max
        return 10.0
    return 2.0  # CPU fallback


def calculate_mfu(
    model: nn.Module,
    tokens_per_sec: float,
    seq_len: int,
    peak_tflops: float,
) -> tuple[float, float]:
    """
    Computes MFU percentage: (6P + 12*L*d_model*T) * tokens_per_sec / Peak_FLOPs.
    Returns (mfu_percentage, achieved_tflops).
    """
    raw_model = get_raw_model(model)
    cfg = raw_model.config
    P = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    flops_per_token = 6 * P + 12 * cfg.n_layer * cfg.n_embd * seq_len
    achieved_flops = flops_per_token * tokens_per_sec
    achieved_tflops = achieved_flops / 1e12
    peak_flops = peak_tflops * 1e12
    mfu_pct = (achieved_flops / peak_flops) * 100.0
    return mfu_pct, achieved_tflops


def create_profiler(output_dir: str = "./log/profiler_trace"):
    """Creates a torch.profiler.profile instance with tensorboard trace handler."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    os.makedirs(output_dir, exist_ok=True)
    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )


def save_checkpoint(
    step: int,
    model: nn.Module,
    optimizers: list,
    optimizer_type: str,
    checkpoint_dir: str,
    is_pause: bool = False,
    keep_step_ckpt: bool = True,
    max_step_ckpts: int = 2,
) -> tuple[str, str | None]:
    """
    Safely saves checkpoint atomically with .bak fallback and step-stamped archiving.
    Prevents file corruption and accidental loss of training progress.
    Retains up to `max_step_ckpts` historical step snapshots to conserve disk space.
    """
    raw_model = get_raw_model(model)
    checkpoint = {
        "step": step,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dicts": [opt.state_dict() for opt in optimizers],
        "optimizer_type": optimizer_type,
        "config": raw_model.config,
    }
    latest_path = os.path.join(checkpoint_dir, "model_latest.pt")
    tmp_path = os.path.join(checkpoint_dir, "model_latest.pt.tmp")
    bak_path = os.path.join(checkpoint_dir, "model_latest.pt.bak")

    # 1. Atomic write to temporary file
    torch.save(checkpoint, tmp_path)

    # 2. Rotate previous latest checkpoint to .bak before replacing
    if os.path.exists(latest_path):
        try:
            if os.path.exists(bak_path):
                os.remove(bak_path)
            os.rename(latest_path, bak_path)
        except OSError:
            pass

    os.rename(tmp_path, latest_path)

    # 3. Save permanent step snapshot for archiving and prune older step snapshots
    step_path = None
    if keep_step_ckpt and (step > 0 or is_pause):
        step_path = os.path.join(checkpoint_dir, f"model_step_{step:04d}.pt")
        try:
            import shutil
            shutil.copy2(latest_path, step_path)
        except Exception:
            torch.save(checkpoint, step_path)

        # Prune older step checkpoints beyond max_step_ckpts
        if max_step_ckpts > 0:
            import glob
            all_step_ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "model_step_*.pt")))
            if len(all_step_ckpts) > max_step_ckpts:
                for old_ckpt in all_step_ckpts[:-max_step_ckpts]:
                    try:
                        os.remove(old_ckpt)
                    except OSError:
                        pass

    return latest_path, step_path


# -----------------------------------------------------------------------------
# Main Training Engine
# -----------------------------------------------------------------------------

def train(
    max_steps: int = 4800,
    total_batch_size: int = 4096,
    eval_interval: int = 50,
    sample_interval: int = 200,
    save_interval: int = 200,
    architecture: str = "classic",
    optimizer_type: str = "adamw",
    muon_lr: float = 0.02,
    resume: str | None = None,
    profile: bool = False,
    use_custom_kernels: bool = False,
    grad_checkpoint: bool = False,
    svd_monitor: bool = False,
    data_dir: str = "data",
):
    # Distributed setup & device detection
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "DDP requires CUDA"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"[Axiom-LM] Using compute device: {device}")

    peak_tflops = estimate_hardware_peak_tflops(device)
    if master_process:
        print(f"[Axiom-LM] Theoretical Peak Hardware Compute: ~{peak_tflops:.1f} TFLOPs")

    # Set seeds for reproducibility
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(1337)

    # Batch size & gradient accumulation setup (e.g. 4,096 tokens per optimizer step for fast iteration)
    B = 2  # Micro-batch size 2 to reduce Unified Memory pressure
    T = 1024
    assert total_batch_size % (B * T * ddp_world_size) == 0, "total_batch_size must be divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"[Axiom-LM] Architecture: {architecture.upper()} | Optimizer: {optimizer_type.upper()} | Batch config: Total={total_batch_size:,} tok | Micro-B={B} | T={T} | GradAccum={grad_accum_steps}")

    # Initialize data loaders with multi-shard support
    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", data_dir=data_dir)
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", data_dir=data_dir)

    # Checkpoint output directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if device == "cuda":
        torch.set_float32_matmul_precision('high')

    # Handle checkpoint resume metadata if specified
    checkpoint_data = None
    start_step = 0
    if resume:
        resume_path = resume if isinstance(resume, str) else os.path.join(checkpoint_dir, "model_latest.pt")
        if os.path.isfile(resume_path):
            if master_process:
                print(f"[Axiom-LM] Loading checkpoint from: {resume_path}")
            checkpoint_data = torch.load(resume_path, map_location=device, weights_only=False)
            start_step = checkpoint_data.get("step", -1) + 1
            if "optimizer_type" in checkpoint_data:
                optimizer_type = checkpoint_data["optimizer_type"]
            if master_process:
                print(f"[Axiom-LM] Resuming training from step {start_step}/{max_steps} (Optimizer: {optimizer_type.upper()})")
        else:
            if master_process:
                print(f"[Axiom-LM] Warning: Checkpoint file '{resume_path}' not found. Starting fresh from step 0.")

    # Initialize Model Architecture (Classic GPT-2 vs Modern LLaMA-3)
    if checkpoint_data is not None and "config" in checkpoint_data:
        config = checkpoint_data["config"]
    elif architecture == "modern":
        config = GPTConfig(
            block_size=1024,
            vocab_size=50304,
            n_layer=12,
            n_head=12,
            n_embd=768,
            n_kv_head=4,
            norm_type="rmsnorm",
            pos_emb="rope",
            mlp_type="swiglu",
            bias=False,
            use_fused_kernels=use_custom_kernels,
            grad_checkpoint=grad_checkpoint,
        )
    else:
        config = GPTConfig(use_fused_kernels=use_custom_kernels, grad_checkpoint=grad_checkpoint)

    model = GPT(config)

    # Load model weights if resuming
    if checkpoint_data is not None and "model_state_dict" in checkpoint_data:
        model.load_state_dict(checkpoint_data["model_state_dict"])
        if master_process:
            print("[Axiom-LM] Successfully loaded model state dict.")

    model.to(device)

    if device == "cuda":
        model = cast(GPT, torch.compile(model))

    if ddp:
        model = cast(GPT, DDP(model, device_ids=[ddp_local_rank]))

    # Precision context
    if device == "cuda":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    elif device == "mps":
        autocast_ctx = torch.autocast(device_type="mps", dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext()

    # Cosine learning rate schedule with linear warmup
    max_adamw_lr = 6e-4
    warmup_steps = min(300, max_steps // 10)

    def get_scheduled_lr(it: int, base_lr: float) -> float:
        min_lr = base_lr * 0.1
        if it < warmup_steps:
            return base_lr * (it + 1) / warmup_steps
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (base_lr - min_lr)

    raw_model = get_raw_model(model)
    optimizers = raw_model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=max_adamw_lr,
        device=device,
        optimizer_type=optimizer_type,
        muon_lr=muon_lr,
    )

    # Restore optimizer state dicts if resuming
    if checkpoint_data is not None and "optimizer_state_dicts" in checkpoint_data:
        for opt, opt_sd in zip(optimizers, checkpoint_data["optimizer_state_dicts"]):
            opt.load_state_dict(opt_sd)
        if master_process:
            print("[Axiom-LM] Successfully restored optimizer states and momentum buffers.")

    # Fast-forward training data loader position when resuming
    if start_step > 0:
        train_loader.set_step(start_step, grad_accum_steps)
        if master_process:
            print(f"[DataLoaderLite] Synchronized dataset position to step {start_step} (token offset: {train_loader.current_position:,})")

    enc = tiktoken.get_encoding('gpt2')

    # Profiler configuration
    if profile:
        trace_dir = os.path.join(project_root, "log", "profiler_trace")
        if master_process:
            print(f"[Axiom-LM Profiler] Recording trace to {trace_dir}...")
        prof_ctx = create_profiler(trace_dir)
        prof_ctx.__enter__()
    else:
        prof_ctx = None

    # Training Loop
    current_step = start_step
    try:
        for step in range(start_step, max_steps):
            current_step = step
            last_step = (step == max_steps - 1)

            # 1. Validation loss evaluation
            if (eval_interval > 0 and step % eval_interval == 0) or last_step:
                model.eval()
                val_loader.reset()
                with torch.no_grad():
                    val_loss_tensor = torch.zeros(1, device=device)
                    val_loss_steps = 20
                    for _ in range(val_loss_steps):
                        x_val, y_val = val_loader.next_batch()
                        x_val, y_val = x_val.to(device), y_val.to(device)
                        with autocast_ctx:
                            _, loss_val = model(x_val, y_val)
                        val_loss_tensor += loss_val / val_loss_steps
                    
                    if ddp:
                        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                    val_loss = val_loss_tensor.item()
                    if master_process:
                        print(f"\n[Val Eval @ Step {step:4d}] validation loss: {val_loss:.4f}", flush=True)

            # 2. Live Text Generation Sampling (KV-Cache Accelerated)
            if master_process and ((sample_interval > 0 and step % sample_interval == 0) or last_step):
                raw_model = get_raw_model(model)
                samples = generate_with_cache(raw_model, enc, device, prompt="Once upon a time", num_samples=2, max_length=45)
                print(f"--- Live Generated Samples (KV-Cache) @ Step {step:4d} ---")
                for idx, s in enumerate(samples, 1):
                    print(f"  [{idx}] {s}")
                print("-" * 50, flush=True)

            # 3. Model Checkpointing
            if not profile and master_process and ((save_interval > 0 and step % save_interval == 0 and step > 0) or last_step):
                latest_p, step_p = save_checkpoint(
                    step=step,
                    model=model,
                    optimizers=optimizers,
                    optimizer_type=optimizer_type,
                    checkpoint_dir=checkpoint_dir,
                    keep_step_ckpt=True,
                )
                print(f"[Axiom-LM] Saved checkpoint to {latest_p}" + (f" (archived {os.path.basename(step_p)})" if step_p else ""), flush=True)

            # 4. Forward / Backward with Micro-Batching (Zero MPS Sync during accumulation)
            model.train()
            for opt in optimizers:
                opt.zero_grad()
            loss_accum_tensor = torch.zeros(1, device=device)
            t0 = time.time()

            for micro_step in range(grad_accum_steps):
                x, y = train_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with autocast_ctx:
                    logits, loss = model(x, y)
                loss = loss / grad_accum_steps
                loss_accum_tensor += loss.detach()
                loss.backward()

            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update learning rates per optimizer group
            current_adamw_lr = get_scheduled_lr(step, max_adamw_lr)
            current_muon_lr = get_scheduled_lr(step, muon_lr)

            if optimizer_type == "muon":
                # optimizers[0] is Muon, optimizers[1] is AdamW
                for param_group in optimizers[0].param_groups:
                    param_group['lr'] = current_muon_lr
                for param_group in optimizers[1].param_groups:
                    param_group['lr'] = current_adamw_lr
            else:
                for param_group in optimizers[0].param_groups:
                    param_group['lr'] = current_adamw_lr

            for opt in optimizers:
                opt.step()

            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()

            t1 = time.time()
            dt = t1 - t0
            tokens_processed = total_batch_size
            tokens_per_sec = tokens_processed / dt
            loss_val = loss_accum_tensor.item()
            mfu_pct, achieved_tflops = calculate_mfu(model, tokens_per_sec, T, peak_tflops)

            if prof_ctx is not None:
                prof_ctx.step()

            if master_process:
                if optimizer_type == "muon":
                    lr_str = f"muon_lr: {current_muon_lr:.4e} | adamw_lr: {current_adamw_lr:.4e}"
                else:
                    lr_str = f"lr: {current_adamw_lr:.4e}"
                print(
                    f"step {step:4d}/{max_steps} | loss: {loss_val:.6f} | {lr_str} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | MFU: {mfu_pct:.1f}% ({achieved_tflops:.2f} TF)",
                    flush=True,
                )
    except KeyboardInterrupt:
        if master_process:
            print(f"\n[Axiom-LM] KeyboardInterrupt caught! Gracefully saving pause snapshot at step {current_step}...")
            latest_p, step_p = save_checkpoint(
                step=current_step,
                model=model,
                optimizers=optimizers,
                optimizer_type=optimizer_type,
                checkpoint_dir=checkpoint_dir,
                is_pause=True,
                keep_step_ckpt=True,
            )
            print(f"[Axiom-LM] Successfully saved pause state to {latest_p}. Resume anytime with '--resume'.", flush=True)
        if prof_ctx is not None:
            prof_ctx.__exit__(None, None, None)
        if ddp:
            destroy_process_group()
        return

    if prof_ctx is not None:
        prof_ctx.__exit__(None, None, None)
        if master_process:
            print(f"[Axiom-LM Profiler] Profiling complete! View trace in chrome://tracing or https://ui.perfetto.dev using files in {trace_dir}")

    if ddp:
        destroy_process_group()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Axiom-LM Pretraining Engine (Classic GPT-2 & Modern LLaMA-3)")
    parser.add_argument("--arch", type=str, default="classic", choices=["classic", "modern"], help="Architecture spec ('classic' GPT-2 or 'modern' LLaMA-3 with RoPE+RMSNorm+SwiGLU+GQA)")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon"], help="Optimizer choice: 'adamw' (standard) or 'muon' (2D Matrix Newton-Schulz + AdamW hybrid)")
    parser.add_argument("--muon_lr", type=float, default=0.02, help="Peak learning rate for Muon matrix optimizer")
    parser.add_argument("--max_steps", type=int, default=4800, help="Total training optimization steps")
    parser.add_argument("--batch_size", type=int, default=4096, help="Total tokens per optimization step")
    parser.add_argument("--eval_interval", type=int, default=50, help="Validation evaluation step interval")
    parser.add_argument("--sample_interval", type=int, default=200, help="Live story sampling step interval")
    parser.add_argument("--save_interval", type=int, default=200, help="Model checkpoint step interval")
    parser.add_argument("--resume", nargs="?", const="checkpoints/model_latest.pt", default=None, help="Resume training from checkpoint file path (defaults to checkpoints/model_latest.pt if flag provided without path)")
    parser.add_argument("--benchmark", action="store_true", help="Run KV-cache vs Naive generation speed benchmark")
    parser.add_argument("--profile", action="store_true", help="Enable PyTorch profiler and export Chrome trace to log/profiler_trace")
    parser.add_argument("--use_custom_kernels", action="store_true", help="Enable custom low-level fused GPU & ARM NEON SIMD kernels")
    parser.add_argument("--grad_checkpoint", action="store_true", help="Enable activation gradient checkpointing for 60-70%% memory reduction")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing binary dataset shards (default: 'data')")
    args = parser.parse_args()

    if args.benchmark:
        device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        enc = tiktoken.get_encoding("gpt2")
        cfg = GPTConfig(
            block_size=1024,
            vocab_size=50304,
            n_layer=12,
            n_head=12,
            n_embd=768,
            n_kv_head=4 if args.arch == "modern" else None,
            norm_type="rmsnorm" if args.arch == "modern" else "layernorm",
            pos_emb="rope" if args.arch == "modern" else "learned",
            mlp_type="swiglu" if args.arch == "modern" else "gelu",
            bias=False if args.arch == "modern" else True,
            use_fused_kernels=args.use_custom_kernels,
            grad_checkpoint=args.grad_checkpoint,
        )
        bm_model = GPT(cfg).to(device)
        benchmark_generation_speed(bm_model, enc, device, prompt="Once upon a time", max_length=100)
    else:
        train_steps = 5 if args.profile and args.max_steps == 4800 else args.max_steps
        train(
            max_steps=train_steps,
            total_batch_size=args.batch_size,
            eval_interval=args.eval_interval,
            sample_interval=args.sample_interval,
            save_interval=args.save_interval,
            architecture=args.arch,
            optimizer_type=args.optimizer,
            muon_lr=args.muon_lr,
            resume=args.resume,
            profile=args.profile,
            use_custom_kernels=args.use_custom_kernels,
            grad_checkpoint=args.grad_checkpoint,
            data_dir=args.data_dir,
        )


if __name__ == "__main__":
    main()