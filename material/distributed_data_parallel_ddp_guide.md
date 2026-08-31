# Distributed Data Parallel (DDP) & Multi-GPU Scaling Guide

A technical reference and system guide for **PyTorch Distributed Data Parallel (`torch.nn.parallel.DistributedDataParallel`)**, based on the seminal VLDB 2020 paper by Li et al. and Andrej Karpathy's *"Let's reproduce GPT-2 (124M)"* multi-GPU training scaling.

---

> [!NOTE]
> **Single-GPU vs Multi-GPU Architecture in this Repository:**
> The default reference implementation in [`brain/train_gpt2.py`](../brain/train_gpt2.py) is intentionally streamlined and optimized for **a single MPS GPU** (Apple Silicon M-series) or a single CUDA GPU for interactive local development and experimentation.
>
> If you fork this repository and wish to scale training across **multiple GPUs** (e.g. 8x A100 or H100 nodes on cloud services like Lambda Labs, RunPod, AWS, GCP, or Azure), this guide provides the exact theoretical principles, communication mechanisms, and drop-in code blueprints to scale the GPT-2 training pipeline seamlessly with PyTorch DDP.

---

## 1. Academic Paper & Primary References

* **Primary Paper:** *PyTorch Distributed: Experiences on Accelerating Data Parallel Training* — Shen Li, Yanli Zhao, Rohan Varma, Omkar Salpekar, Pieter Noordhuis, Teng Li, Adam Paszke, Jeff Smith, Soumith Chintala (VLDB 2020 / arXiv:2006.15704).
  * Local PDF: [`material/pytorch_distributed_ddp_paper.pdf`](./pytorch_distributed_ddp_paper.pdf)
  * arXiv Link: [arXiv:2006.15704](https://arxiv.org/abs/2006.15704)
* **Official PyTorch API Documentation:** [`torch.nn.parallel.DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
* **PyTorch Distributed Overview:** [`torch.distributed`](https://pytorch.org/docs/stable/distributed.html)

---

## 2. Why DistributedDataParallel (DDP)?

### The Problem with Legacy `torch.nn.DataParallel` (DP)
PyTorch historically provided `torch.nn.DataParallel` (DP), which operated as a single-process, multi-threaded wrapper around a model:
1. **Python GIL Bottleneck:** A single Python process drives all GPUs, suffering severe contention from Python's Global Interpreter Lock (GIL).
2. **Scatter/Gather Overhead:** On every forward pass, the main GPU (GPU 0) must broadcast weights and scatter input mini-batches to all other GPUs, then gather all outputs and compute loss on GPU 0.
3. **GPU 0 Imbalance:** GPU 0 runs out of memory (OOM) much earlier than replica GPUs because it holds the master gradients and optimizer state.

### The Solution: Multi-Process DDP
`torch.nn.parallel.DistributedDataParallel` (DDP) spawns **1 completely independent Python process per GPU** (e.g., 8 processes for an 8-GPU node):
- No Python GIL contention (each process has its own Python interpreter and memory space).
- Model parameters and optimizer states are replicated identically across all GPUs at initialization.
- Every process runs forward and backward passes on its own local slice of data simultaneously.
- Only **gradients** are synchronized across GPUs via ring All-Reduce communication during the backward pass.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DDP ARCHITECTURE (e.g. 4 GPUs)                     │
├───────────────────┬───────────────────┬───────────────────┬─────────────────┤
│    GPU 0 (Rank 0) │    GPU 1 (Rank 1) │    GPU 2 (Rank 2) │  GPU 3 (Rank 3) │
│    [Process 0]    │    [Process 1]    │    [Process 2]    │  [Process 3]    │
├───────────────────┼───────────────────┼───────────────────┼─────────────────┤
│ Local Batch D_0   │ Local Batch D_1   │ Local Batch D_2   │ Local Batch D_3 │
│ Forward Pass      │ Forward Pass      │ Forward Pass      │ Forward Pass    │
│ Loss & Backward   │ Loss & Backward   │ Loss & Backward   │ Loss & Backward │
└─────────┬─────────┴─────────┬─────────┴─────────┬─────────┴────────┬────────┘
          │                   │                   │                  │
          └───────────────────┼───────────────────┼──────────────────┘
                              ▼                   ▼
         ┌─────────────────────────────────────────────────────────┐
         │ Ring All-Reduce Gradient Synchronization (NCCL Backend) │
         │   Gradients are averaged across all ranks in-flight     │
         └────────────────────────────┬────────────────────────────┘
                                      ▼
                   Each GPU performs local AdamW step
                   (Weights stay perfectly synchronized)
```

---

## 3. Core Mechanics & Engineering Innovations (from Li et al., 2020)

### A. Ring All-Reduce Algorithm
Instead of sending all gradients to a central parameter server, GPUs are arranged in a logical ring. Each GPU communicates only with its immediate neighbors.

For $N$ GPUs and gradient tensor of size $M$:
1. **Scatter-Reduce Phase:** Gradients are divided into $N$ chunks. In $N-1$ steps, each GPU transmits and accumulates one chunk to its neighbor.
2. **All-Gather Phase:** In $N-1$ steps, the fully accumulated chunks are circulated around the ring so all GPUs receive the complete averaged gradient.

$$\text{Total Transferred Volume per GPU} = 2 \times \left(\frac{N-1}{N}\right) \times M$$

Notice that the network communication volume per GPU is **independent of the number of GPUs ($N$)** for large $N$, making it bandwidth-optimal.

```
            GPU 0 ────────► GPU 1
              ▲               │
              │   Ring-All    │
              │    Reduce     │
              │               ▼
            GPU 3 ◄──────── GPU 2
```

### B. Overlapping Computation and Communication via Gradient Bucketing
In a naive implementation, one would wait until the backward pass completes and then run `all_reduce` on all gradients. This leaves the GPU interconnect idle during the backward pass.

DDP organizes model parameters into **Buckets** (default `bucket_cap_mb = 25` MB):
1. Parameters in the backward pass are computed in reverse topological order (from output layer to input layer).
2. As soon as all parameter gradients within a 25MB bucket are ready, DDP **immediately launches an asynchronous non-blocking `all_reduce` call** for that bucket on a separate CUDA communication stream.
3. While earlier layer gradients are being computed on the GPU compute stream, later layer gradients are being communicated across the network simultaneously.

```
Time ────────────────────────────────────────────────────────────────────────►
Backward Compute: [Layer 12 Grad] [Layer 11 Grad] [Layer 10 Grad] ... [Layer 1]
NCCL Comm Stream:                  [All-Reduce Bkt 1]   [All-Reduce Bkt 2] ...
```

---

## 4. Multi-GPU Engineering with Gradient Accumulation: `model.no_sync()`

When using large total batch sizes (e.g., $0.5\text{M}$ tokens = $524,288$ tokens) on $N$ GPUs:
$$\text{Total Batch} = B \times T \times \text{world\_size} \times \text{grad\_accum\_steps}$$

If we have 8 GPUs, $B=8$, $T=1024$, then each step processes $8 \times 1024 \times 8 = 65,536$ tokens per micro-step. To reach $524,288$ tokens, we need $\text{grad\_accum\_steps} = 8$.

### The Pitfall: Wasted All-Reduce
By default, DDP triggers gradient all-reduce on **every single `loss.backward()` call**. For micro-steps $0$ through $6$, synchronizing gradients across the network is completely redundant because gradients are only accumulated locally.

### The Fix: `model.no_sync()`
PyTorch provides the `model.no_sync()` context manager:
- During micro-steps $0 \dots (\text{grad\_accum\_steps} - 2)$, wrap the forward/backward passes in `model.no_sync()`.
- On the final micro-step $(\text{grad\_accum\_steps} - 1)$, execute without `no_sync()` so gradients are all-reduced once right before the optimizer step.

```python
for micro_step in range(grad_accum_steps):
    is_last_micro_step = (micro_step == grad_accum_steps - 1)
    
    # Disable gradient sync on all micro-steps except the last one
    sync_context = nullcontext() if is_last_micro_step else model.no_sync()
    
    with sync_context:
        with autocast_ctx:
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss.backward()
```

---

## 5. Sharded Data Loading Across Ranks

In DDP, each GPU process must receive **different, non-overlapping batches of tokens** so that the entire cluster trains on diverse data.

```python
class DistributedDataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split="train"):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        
        # Load token dataset
        with open("material/input.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        self.tokens = torch.tensor(enc.encode(text), dtype=torch.long)
        
        # State: start each rank at its designated offset
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        
        # Advance by total cluster throughput across all processes
        self.current_position += B * T * self.num_processes
        
        # Reset if end of dataset reached
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
            
        return x, y
```

---

## 6. Complete Blueprint: DDP Training Script (`train_gpt2_ddp.py`)

Below is the complete, drop-in multi-GPU DDP training template for scaling GPT-2 across multiple CUDA GPUs on cloud clusters:

```python
import os
import time
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from contextlib import nullcontext

# =============================================================================
# 1. DDP Initialization & Process Setup
# =============================================================================
ddp = int(os.environ.get('RANK', -1)) != -1 # Is this a ddp run?

if ddp:
    assert torch.cuda.is_available(), "CUDA required for multi-GPU DDP"
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0) # Master process handles logging & checkpoints
else:
    # Single GPU / MPS / CPU fallback
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

if master_process:
    print(f"Using device: {device} | DDP: {ddp} | World Size: {ddp_world_size}")

# Set deterministic seed with rank offset
torch.manual_seed(1337 + ddp_rank)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337 + ddp_rank)

# =============================================================================
# 2. Hyperparameters & Batch Configuration
# =============================================================================
total_batch_size = 524288 # 2**19 ~ 0.5M tokens (GPT-3/GPT-2 recipe)
B = 8                     # micro-batch size per GPU
T = 1024                  # sequence context length

assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"Total Batch Size: {total_batch_size} tokens")
    print(f"Per-GPU Micro-Batch: {B} x {T} = {B*T} tokens")
    print(f"Grad Accumulation Steps per GPU: {grad_accum_steps}")

# =============================================================================
# 3. Model Wrapping with DDP
# =============================================================================
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)

if device.startswith("cuda"):
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # Unwrap raw model for helper methods

# =============================================================================
# 4. Training Loop with Distributed Gradient Accumulation
# =============================================================================
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
train_loader = DistributedDataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        
        # Only sync gradients on the final micro-step
        sync_context = nullcontext() if (not ddp or micro_step == grad_accum_steps - 1) else model.no_sync()
        
        with sync_context:
            with autocast_ctx:
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
            
    if ddp:
        # Average loss metric across all GPUs for clean logging
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Update learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    t1 = time.time()
    dt = t1 - t0
    tokens_per_sec = total_batch_size / dt
    
    if master_process:
        print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

# Clean teardown
if ddp:
    dist.destroy_process_group()
```

---

## 7. How to Launch on Multi-GPU Nodes (`torchrun`)

PyTorch provides `torchrun` (the modern replacement for `torch.distributed.launch`) to spawn and supervise multi-GPU processes automatically:

### Single Node with 8 GPUs (e.g. 8x A100/H100 Node):
```bash
torchrun --standalone --nproc_per_node=8 brain/train_gpt2_ddp.py
```

### Multi-Node Cluster (e.g. 2 nodes x 8 GPUs = 16 GPUs):
```bash
# On Node 0 (Master Node, IP: 10.0.0.1):
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=10.0.0.1 --master_port=29500 brain/train_gpt2_ddp.py

# On Node 1 (Worker Node):
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=10.0.0.1 --master_port=29500 brain/train_gpt2_ddp.py
```

---

## 8. Summary Comparison: Single MPS vs Multi-GPU DDP

| Dimension | Single-GPU MPS Reference (`train_gpt2.py`) | Multi-GPU DDP (`train_gpt2_ddp.py`) |
|:---|:---|:---|
| **Primary Target** | Local Apple Silicon Mac (M1/M2/M3/M4) / Single GPU | Multi-GPU CUDA nodes (A100, H100, RTX 4090) & Cloud Clusters |
| **Processes** | 1 Python process | $N$ independent Python processes (1 per GPU) |
| **Communication** | None (Local on-chip unified memory) | NCCL Ring All-Reduce over NVLink / InfiniBand / PCIe |
| **Gradient Sync** | Direct local accumulation | Asynchronous gradient bucketing (`bucket_cap_mb=25`) + `model.no_sync()` |
| **Data Partitioning** | Sequential token slice | Sharded across ranks (`offset = B * T * rank`) |
| **Optimizer Execution** | Single optimizer step | Replicated identical optimizer step per rank |
| **Throughput Scaling** | Baseline ($1\times$) | Near-linear scaling ($\approx 0.92 - 0.98 \times N$) with NVLink |
