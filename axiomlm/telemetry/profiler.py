"""
AxiomLM Systems Profiling, Roofline Analysis, and MFU Telemetry.
"""
import os
from typing import Tuple, Any, Optional
import torch


def estimate_hardware_peak_tflops(device: str) -> float:
    """
    Returns theoretical peak FP16/BF16 Tensor Core TFLOPs for the detected hardware.
    """
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).lower()
        if "h100" in gpu_name:
            return 989.0  # H100 SXM FP16 Dense
        elif "a100" in gpu_name:
            return 312.0  # A100 SXM FP16 Dense
        elif "l40" in gpu_name:
            return 181.0  # L40 FP16
        elif "4090" in gpu_name:
            return 165.0  # RTX 4090 FP16
        elif "3090" in gpu_name or "a5000" in gpu_name:
            return 71.0   # RTX 3090 / A5000 FP16
        elif "t4" in gpu_name:
            return 65.0   # T4 FP16
        else:
            return 100.0  # Generic CUDA GPU fallback
    elif device == "mps":
        return 10.0  # Apple Silicon baseline estimation
    else:
        return 2.0   # CPU vector baseline


def calculate_mfu(
    model: Any,
    tokens_per_sec: float,
    seq_len: int = 1024,
    peak_tflops: float = 10.0,
    context_len: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Calculates Model FLOPs Utilization (MFU) based on standard scaling laws:
    FLOPs per token = 6 * N_params + 12 * L * H * Q * T (Attention FLOPs).
    """
    if context_len is not None:
        seq_len = context_len

    raw_model = model.module if hasattr(model, 'module') else model
    config = raw_model.config
    
    # Active parameter count
    N = sum(p.numel() for p in raw_model.parameters())
    L = config.n_layer
    H = config.n_head
    Q = config.n_embd // config.n_head
    T = seq_len

    flops_per_token = 6 * N + 12 * L * H * Q * T
    achieved_flops = flops_per_token * tokens_per_sec
    achieved_tflops = achieved_flops / 1e12

    mfu = (achieved_tflops / peak_tflops) * 100.0 if peak_tflops > 0 else 0.0
    return mfu, achieved_tflops


def create_profiler(log_dir: str = "log/profiler_trace") -> torch.profiler.profile:
    """Initializes PyTorch Profiler with Chrome Trace output."""
    os.makedirs(log_dir, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        
    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )
