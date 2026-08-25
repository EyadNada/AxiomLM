import os
import math
import time
import inspect
from dataclasses import dataclass
from typing import cast
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import tiktoken
import numpy as np


# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    bias: torch.Tensor

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Q, K, V projections for all heads in a single batched linear layer
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        setattr(self.c_proj, "NANOGPT_SCALE_INIT", 1)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Autoregressive causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x, kv_cache=None, use_cache=False):
        B, T, C = x.size()
        # Project and split into queries, keys, and values: (B, T, 3 * C) -> 3 x (B, T, C)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # Reshape to (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if kv_cache is not None and kv_cache[0] is not None and kv_cache[1] is not None:
            k_past, v_past = kv_cache
            k = torch.cat([k_past, k], dim=2)
            v = torch.cat([v_past, v], dim=2)

        new_kv_cache = (k, v) if (use_cache or kv_cache is not None) else None

        if kv_cache is None or kv_cache[0] is None:
            # Fused scaled dot-product attention with causal mask (FlashAttention / SRAM tiling)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Single-token decode attends to all past keys/values; is_causal=False when T_q == 1
            is_causal = (T > 1)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        # Re-assemble head outputs side-by-side: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y, new_kv_cache


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        setattr(self.c_proj, "NANOGPT_SCALE_INIT", 1)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, kv_cache=None, use_cache=False):
        attn_out, new_kv_cache = self.attn(self.ln_1(x), kv_cache=kv_cache, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv_cache


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
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

        pos = torch.arange(past_len, past_len + T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer['wpe'](pos)
        tok_emb = self.transformer['wte'](idx)
        x = tok_emb + pos_emb

        new_kv_caches = [] if (use_cache or kv_caches is not None) else None
        for i, block in enumerate(cast(nn.ModuleList, self.transformer['h'])):
            block_kv = kv_caches[i] if kv_caches is not None else None
            x, updated_kv = block(x, kv_cache=block_kv, use_cache=(use_cache or kv_caches is not None))
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

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
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

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # 2D parameters (weights, embeddings) decay; 1D parameters (biases, layernorms) do not
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"Decayed parameter tensors: {len(decay_params)} ({num_decay:,} params)")
        print(f"Non-decayed parameter tensors: {len(nodecay_params)} ({num_nodecay:,} params)")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and ('cuda' in device)
        print(f"Using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused
        )
        return optimizer


# -----------------------------------------------------------------------------
# Data Loader
# -----------------------------------------------------------------------------

class DataLoaderLite:
    def __init__(self, B, T, process_rank=0, num_processes=1, split="train"):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        assert split in {"train", "val"}, "split must be 'train' or 'val'"

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        shard_path = os.path.join(project_root, "data", f"{split}.bin")
        if os.path.exists(shard_path):
            self.tokens = np.memmap(shard_path, dtype=np.uint16, mode='r')
            print(f"[DataLoaderLite] Loaded {split} shard from {shard_path} ({len(self.tokens):,} tokens)")
        else:
            input_path = os.path.join(project_root, "material", "input.txt")
            if not os.path.exists(input_path):
                input_path = "input.txt"
            with open(input_path, "r") as f:
                text = f.read()
            enc = tiktoken.get_encoding("gpt2")
            self.tokens = np.array(enc.encode(text), dtype=np.uint16)
            print(f"[DataLoaderLite] Loaded fallback text with {len(self.tokens):,} tokens")

        print(f"[DataLoaderLite] 1 epoch = {len(self.tokens) // (B * T * num_processes)} batches")
        self.reset()

    def reset(self):
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf_torch = torch.from_numpy(buf.astype(np.int64))
        x = buf_torch[:-1].view(B, T)
        y = buf_torch[1:].view(B, T)
        self.current_position += B * T * self.num_processes

        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.reset()
        return x, y


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Model Unwrapping & Sampling Helpers
# -----------------------------------------------------------------------------

def get_raw_model(model: nn.Module) -> GPT:
    """Safely unwraps DDP (DistributedDataParallel) and torch.compile containers."""
    unwrapped = getattr(model, "module", getattr(model, "_orig_mod", model))
    return cast(GPT, unwrapped)


def generate_samples(model: GPT, enc, device, prompt="Once upon a time", num_samples=2, max_length=40):
    """Generates autoregressive text samples using standard eager re-computation (O(T^2))."""
    model.eval()
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0).repeat(num_samples, 1)

    while tokens.size(1) < max_length:
        with torch.no_grad():
            logits, _ = model(tokens)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            tokens = torch.cat((tokens, xcol), dim=1)

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
    top_k: int = 50,
):
    """
    Accelerated autoregressive text generation using per-layer Key-Value caching.
    Reduces compute complexity from O(T^2) to O(1) per decoding step.
    """
    model.eval()
    prompt_tokens = enc.encode(prompt)
    x = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0).repeat(num_samples, 1)

    with torch.no_grad():
        # 1. Prefill phase: pass entire prompt to initialize KV caches
        kv_caches = [None] * model.config.n_layer
        logits, _, kv_caches = model(x, kv_caches=kv_caches)

        next_token_logits = logits[:, -1, :]
        if temperature > 0:
            probs = F.softmax(next_token_logits / temperature, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, min(top_k, probs.size(-1)), dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            next_token = torch.gather(topk_indices, -1, ix)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated_tokens = torch.cat((x, next_token), dim=1)

        # 2. Decode phase: pass single token (T=1) on each step with cached past
        while generated_tokens.size(1) < max_length:
            logits, _, kv_caches = model(next_token, kv_caches=kv_caches)
            next_token_logits = logits[:, -1, :]
            if temperature > 0:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, min(top_k, probs.size(-1)), dim=-1)
                ix = torch.multinomial(topk_probs, 1)
                next_token = torch.gather(topk_indices, -1, ix)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
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
# Main Training Engine
# -----------------------------------------------------------------------------

def train(
    max_steps: int = 4800,
    total_batch_size: int = 4096,
    eval_interval: int = 50,
    sample_interval: int = 200,
    save_interval: int = 200,
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
        print(f"[Axiom-LM] Batch config: Total={total_batch_size:,} tok | Micro-B={B} | T={T} | GradAccum={grad_accum_steps}")

    # Initialize data loaders
    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

    # Checkpoint output directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if device == "cuda":
        torch.set_float32_matmul_precision('high')

    model = GPT(GPTConfig())
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
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = min(300, max_steps // 10)

    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    raw_model = get_raw_model(model)
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)
    enc = tiktoken.get_encoding('gpt2')

    # Training Loop
    for step in range(max_steps):
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
        if master_process and ((save_interval > 0 and step % save_interval == 0 and step > 0) or last_step):
            raw_model = get_raw_model(model)
            checkpoint_path = os.path.join(checkpoint_dir, "model_latest.pt")
            checkpoint = {
                "step": step,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": raw_model.config,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"[Axiom-LM] Saved checkpoint to {checkpoint_path}", flush=True)

        # 4. Forward / Backward with Micro-Batching (Zero MPS Sync during accumulation)
        model.train()
        optimizer.zero_grad()
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

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()

        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

        t1 = time.time()
        dt = t1 - t0
        tokens_processed = total_batch_size
        tokens_per_sec = tokens_processed / dt
        loss_val = loss_accum_tensor.item()

        if master_process:
            print(
                f"step {step:4d}/{max_steps} | loss: {loss_val:.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}",
                flush=True,
            )

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    train()