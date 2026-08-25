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

    def forward(self, x):
        B, T, C = x.size()
        # Project and split into queries, keys, and values: (B, T, 3 * C) -> 3 x (B, T, C)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # Reshape to (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Fused scaled dot-product attention (FlashAttention / SRAM tiling)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # Re-assemble head outputs side-by-side: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


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

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


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

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer['wpe'](pos)
        tok_emb = self.transformer['wte'](idx)
        x = tok_emb + pos_emb

        for block in cast(nn.ModuleList, self.transformer['h']):
            x = block(x)

        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

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
        buf_torch = torch.tensor(buf.astype(np.int64), dtype=torch.long)
        x = buf_torch[:-1].view(B, T)
        y = buf_torch[1:].view(B, T)
        self.current_position += B * T * self.num_processes

        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.reset()
        return x, y


# -----------------------------------------------------------------------------
# Distributed Setup & Device Detection
# -----------------------------------------------------------------------------

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
    print(f"Using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch.mps.manual_seed(1337)

# Batch size & gradient accumulation setup (16,384 tokens per optimizer step)
total_batch_size = 16384
B = 2  # Micro-batch size 2 to reduce Unified Memory pressure
T = 1024
assert total_batch_size % (B * T) == 0, "total_batch_size must be divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"Target batch size: {total_batch_size} tokens | Micro-batch size: {B} | Sequence length: {T}")
print(f"Gradient accumulation steps: {grad_accum_steps}")


# -----------------------------------------------------------------------------
# Training Initialization & Optimization Loop
# -----------------------------------------------------------------------------

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

if device == "cuda":
    torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig())
model.to(device)

if device == "cuda":
    model = cast(GPT, torch.compile(model))

if device == "cuda":
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
elif device == "mps":
    autocast_ctx = torch.autocast(device_type="mps", dtype=torch.bfloat16)
else:
    autocast_ctx = nullcontext()

# Cosine learning rate schedule with warmup
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 320
max_steps = 1600

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # Periodic validation loss evaluation
    if step % 50 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x_val, y_val = val_loader.next_batch()
                x_val, y_val = x_val.to(device), y_val.to(device)
                with autocast_ctx:
                    _, loss_val = model(x_val, y_val)
                loss_val = loss_val / val_loss_steps
                val_loss_accum += loss_val.detach().item()
            if ddp:
                val_loss_tensor = torch.tensor(val_loss_accum, device=device)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                val_loss_accum = val_loss_tensor.item()
            if master_process:
                print(f"[step {step:4d}] validation loss: {val_loss_accum:.4f}", flush=True)

    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with autocast_ctx:
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach().item()
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
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt

    print(
        f"step {step:4d}/{max_steps} | loss: {loss_accum:.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}",
        flush=True,
    )


# -----------------------------------------------------------------------------
# Text Generation & Sampling
# -----------------------------------------------------------------------------

num_return_sequences = 5
max_length = 30

model.eval()

enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

torch.manual_seed(42)
if device == "cuda":
    torch.cuda.manual_seed(42)
elif device == "mps":
    torch.mps.manual_seed(42)

while x.size(1) < max_length:
    with torch.no_grad():
        logits, _ = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

print("\n--- Generated Samples ---", flush=True)
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded, flush=True)

if ddp:
    destroy_process_group()