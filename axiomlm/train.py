"""
AxiomLM Pretraining Engine with Muon Optimizer & Distributed Data Parallel.
"""
from typing import Optional, List, Tuple, Union, Any
import os
import time
import math
import argparse
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import tiktoken
import numpy as np

from .models.transformer import Transformer, ModelConfig, GPT, GPTConfig
from .optim.muon import Muon
from .optim.schedule import get_lr
from .dengine.dataloader import DataLoaderLite
from .telemetry.profiler import calculate_mfu, estimate_hardware_peak_tflops, create_profiler
from .engine.inference import generate_with_cache, benchmark_generation_speed, sample_logits


def get_raw_model(model: Any) -> Transformer:
    """Unwraps DDP and torch.compile wrappers to get the underlying Transformer module."""
    raw = model.module if hasattr(model, 'module') else model
    raw = raw._orig_mod if hasattr(raw, '_orig_mod') else raw
    return raw


def save_checkpoint(
    step: int,
    model: Any,
    optimizers: List[torch.optim.Optimizer],
    optimizer_type: str,
    checkpoint_dir: str = "checkpoints",
    is_pause: bool = False,
    keep_step_ckpt: bool = True,
    max_step_ckpts: int = 5,
) -> Tuple[str, Optional[str]]:
    """Saves atomic checkpoints containing weights, optimizer states, and configuration."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    raw_model = get_raw_model(model)
    checkpoint_data = {
        "step": step,
        "model": raw_model.state_dict(),
        "config": raw_model.config,
        "optimizer_type": optimizer_type,
        "optimizers": [opt.state_dict() for opt in optimizers],
        "is_pause": is_pause,
        "timestamp": time.time(),
    }

    latest_path = os.path.join(checkpoint_dir, "model_latest.pt")
    tmp_path = os.path.join(checkpoint_dir, f".model_latest_step_{step}.tmp")
    torch.save(checkpoint_data, tmp_path)
    os.replace(tmp_path, latest_path)

    step_path = None
    if keep_step_ckpt:
        step_path = os.path.join(checkpoint_dir, f"model_step_{step}.pt")
        tmp_step_path = os.path.join(checkpoint_dir, f".model_step_{step}.tmp")
        torch.save(checkpoint_data, tmp_step_path)
        os.replace(tmp_step_path, step_path)

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


def train(
    max_steps: int = 4800,
    total_batch_size: int = 4096,
    eval_interval: int = 50,
    sample_interval: int = 200,
    sample_prompt: str = "import torch\n",
    save_interval: int = 25,
    architecture: str = "classic",
    optimizer_type: str = "adamw",
    muon_lr: float = 0.02,
    resume: Optional[str] = None,
    profile: bool = False,
    use_custom_kernels: bool = False,
    grad_checkpoint: bool = False,
    data_dir: str = "data",
) -> None:
    """Pretraining engine for Classic GPT-2 and Modern LLaMA-3 architectures."""
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        # Determine the best backend and device for distributed training
        if torch.cuda.is_available():
            backend = 'nccl'
            ddp_local_rank = int(os.environ['LOCAL_RANK'])
            device = f'cuda:{ddp_local_rank}'
            torch.cuda.set_device(device)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            backend = 'gloo'
            ddp_local_rank = int(os.environ.get('LOCAL_RANK', 0))
            device = 'mps'
        else:
            backend = 'gloo'
            ddp_local_rank = int(os.environ.get('LOCAL_RANK', 0))
            device = 'cpu'
            
        # Initialize process group with multi-node timeouts
        from datetime import timedelta
        init_process_group(backend=backend, init_method="env://", timeout=timedelta(minutes=30))
        
        ddp_rank = int(os.environ['RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
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

    if master_process:
        print(f"[AxiomLM Pretrain Engine] Running on device: {device} (World Size: {ddp_world_size})")

    torch.manual_seed(1337 + ddp_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337 + ddp_rank)

    B = 4 if device in ["cuda", "mps"] else 2
    T = 1024
    assert total_batch_size % (B * T * ddp_world_size) == 0, f"total_batch_size ({total_batch_size}) must be divisible by {B*T*ddp_world_size}"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", data_dir=data_dir)
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", data_dir=data_dir)

    if architecture == "modern":
        config = ModelConfig(
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
        config = ModelConfig(
            block_size=1024,
            vocab_size=50304,
            n_layer=12,
            n_head=12,
            n_embd=768,
            n_kv_head=None,
            norm_type="layernorm",
            pos_emb="learned",
            mlp_type="gelu",
            bias=True,
            use_fused_kernels=use_custom_kernels,
            grad_checkpoint=grad_checkpoint,
        )

    model = Transformer(config).to(device)
    max_adamw_lr = 6e-4
    optimizers = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=max_adamw_lr,
        device=device,
        optimizer_type=optimizer_type,
        muon_lr=muon_lr,
    )

    start_step = 0
    if resume:
        if os.path.isfile(resume):
            if master_process:
                print(f"[AxiomLM] Resuming training from checkpoint: {resume}")
            
            # Compatibility for legacy checkpoints that pickled GPTConfig in __main__
            import __main__
            if not hasattr(__main__, 'GPTConfig'):
                setattr(__main__, 'GPTConfig', ModelConfig)
                
            ckpt = torch.load(resume, map_location=device, weights_only=False)
            start_step = ckpt.get("step", 0)
            raw_model = get_raw_model(model)
            
            # Support both new ('model', 'optimizers') and legacy ('model_state_dict', 'optimizer_state_dicts') keys
            model_key = "model" if "model" in ckpt else "model_state_dict"
            opt_key = "optimizers" if "optimizers" in ckpt else "optimizer_state_dicts"
            
            cleaned_sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ckpt[model_key].items()}
            raw_model.load_state_dict(cleaned_sd, strict=False)

            if opt_key in ckpt and len(ckpt[opt_key]) == len(optimizers):
                for opt, s_dict in zip(optimizers, ckpt[opt_key]):
                    try:
                        opt.load_state_dict(s_dict)
                    except Exception:
                        pass
        else:
            if master_process:
                print(f"[AxiomLM Notice] Checkpoint {resume} not found. Starting from step 0.")

    if ddp:
        if device.startswith("cuda"):
            model = DDP(model, device_ids=[ddp_local_rank])
        else:
            model = DDP(model)

    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.startswith("cuda") else nullcontext()
    enc = tiktoken.get_encoding("gpt2")
    peak_tflops = estimate_hardware_peak_tflops(device)
    checkpoint_dir = "checkpoints"

    prof_ctx = create_profiler() if profile and master_process else None
    if prof_ctx is not None:
        prof_ctx.__enter__()

    try:
        for step in range(start_step, max_steps):
            t0 = time.time()
            last_step = (step == max_steps - 1)

            # 1. Validation Evaluation
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

            # 2. Live Validation Token Sampling
            if master_process and ((sample_interval > 0 and step % sample_interval == 0) or last_step):
                raw_model = get_raw_model(model)
                samples = generate_with_cache(raw_model, enc, device, prompt=sample_prompt, num_samples=2, max_length=45)
                print(f"--- Live Generated Samples (Systems ML / Code) @ Step {step:4d} ---")
                for idx, s in enumerate(samples, 1):
                    print(f"  [{idx}] {s}")
                print("-" * 50, flush=True)

            # 3. Forward / Backward with Micro-Batching
            model.train()
            for opt in optimizers:
                opt.zero_grad()
            loss_accum_tensor = torch.zeros(1, device=device)

            for micro_step in range(grad_accum_steps):
                x, y = train_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with autocast_ctx:
                    logits, loss = model(x, y)
                loss = loss / grad_accum_steps
                loss_accum_tensor += loss.detach()
                loss.backward()

            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update learning rates
            current_adamw_lr = get_lr(step, max_steps=max_steps, max_lr=max_adamw_lr)
            current_muon_lr = get_lr(step, max_steps=max_steps, max_lr=muon_lr)

            if optimizer_type == "muon":
                for param_group in optimizers[0].param_groups:
                    param_group['lr'] = current_muon_lr
                for param_group in optimizers[1].param_groups:
                    param_group['lr'] = current_adamw_lr
            else:
                for param_group in optimizers[0].param_groups:
                    param_group['lr'] = current_adamw_lr

            for opt in optimizers:
                opt.step()

            completed_step = step + 1

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
                lr_str = f"muon_lr: {current_muon_lr:.4e} | adamw_lr: {current_adamw_lr:.4e}" if optimizer_type == "muon" else f"lr: {current_adamw_lr:.4e}"
                print(
                    f"step {step:4d}/{max_steps} | loss: {loss_val:.6f} | {lr_str} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | MFU: {mfu_pct:.1f}% ({achieved_tflops:.2f} TF)",
                    flush=True,
                )

            # 4. Model Checkpointing
            if not profile and master_process and ((save_interval > 0 and completed_step % save_interval == 0) or last_step):
                latest_p, step_p = save_checkpoint(
                    step=completed_step,
                    model=model,
                    optimizers=optimizers,
                    optimizer_type=optimizer_type,
                    checkpoint_dir=checkpoint_dir,
                    keep_step_ckpt=True,
                )
                print(f"[AxiomLM] Saved checkpoint to {latest_p} (step {completed_step})" + (f" (archived {os.path.basename(step_p)})" if step_p else ""), flush=True)

    except (KeyboardInterrupt, SystemExit):
        if master_process:
            last_done = step + 1 if 'loss_val' in locals() else step
            print(f"\n[AxiomLM] Training paused/interrupted! Saving exact completed state at step {last_done}...")
            save_checkpoint(
                step=last_done,
                model=model,
                optimizers=optimizers,
                optimizer_type=optimizer_type,
                checkpoint_dir=checkpoint_dir,
                is_pause=True,
                keep_step_ckpt=True,
            )
        if prof_ctx is not None:
            prof_ctx.__exit__(None, None, None)
        if ddp:
            destroy_process_group()
        return

    if prof_ctx is not None:
        prof_ctx.__exit__(None, None, None)

    if ddp:
        destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="AxiomLM Pretraining CLI")
    parser.add_argument("--arch", type=str, default="modern", choices=["classic", "modern"])
    parser.add_argument("--optimizer", type=str, default="muon", choices=["adamw", "muon"])
    parser.add_argument("--muon_lr", type=float, default=0.02)
    parser.add_argument("--max_steps", type=int, default=4800)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--sample_interval", type=int, default=200)
    parser.add_argument("--sample_prompt", type=str, default="import torch\n")
    parser.add_argument("--save_interval", type=int, default=25)
    parser.add_argument("--resume", nargs="?", const="checkpoints/model_latest.pt", default=None)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--use_custom_kernels", action="store_true")
    parser.add_argument("--grad_checkpoint", action="store_true")
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()

    if args.benchmark:
        device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        enc = tiktoken.get_encoding("gpt2")
        cfg = ModelConfig(
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
        bm_model = Transformer(cfg).to(device)
        benchmark_generation_speed(bm_model, enc, device, prompt=args.sample_prompt, max_length=100)
    else:
        train_steps = 5 if args.profile and args.max_steps == 4800 else args.max_steps
        train(
            max_steps=train_steps,
            total_batch_size=args.batch_size,
            eval_interval=args.eval_interval,
            sample_interval=args.sample_interval,
            sample_prompt=args.sample_prompt,
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
