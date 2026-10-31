"""
AxiomLM: High-Performance Systems LLM Pretraining & Hardware Kernel Engine.

Modern LLaMA-3 Architecture (RoPE + RMSNorm + SwiGLU + GQA),
Muon Newton-Schulz Optimizer, and Bare-Metal Fused Metal / Triton Kernels.
"""

from . import dengine, engine, models, optim, telemetry
from .dengine.dataloader import DataLoaderLite
from .engine.export import export_checkpoint_to_hf
from .engine.inference import (
    InferenceEngine,
    generate_samples,
    generate_with_cache,
    load_model,
    sample_logits,
)
from .models.attention import CausalSelfAttention
from .models.modules import (
    MLP,
    RMSNorm,
    SwiGLUMLP,
    apply_rope,
    precompute_rope_frequencies,
    repeat_kv,
)
from .models.transformer import (
    GPT,
    Block,
    GPTConfig,
    ModelConfig,
    Transformer,
)
from .optim.muon import Muon, zeropower_via_newtonschulz5
from .optim.schedule import get_lr
from .telemetry.profiler import (
    calculate_mfu,
    create_profiler,
    estimate_hardware_peak_tflops,
)
from .train import train

dENGINE = dengine  # type: ignore

try:
    from . import kernels
    from .kernels import is_mps_available
except ImportError:
    kernels = None  # type: ignore

    def is_mps_available() -> bool:
        return False


__version__ = "0.1.0"

__all__ = [
    "GPT",
    "MLP",
    "Block",
    "CausalSelfAttention",
    "DataLoaderLite",
    "GPTConfig",
    "InferenceEngine",
    "ModelConfig",
    "Muon",
    "RMSNorm",
    "SwiGLUMLP",
    "Transformer",
    "__version__",
    "apply_rope",
    "calculate_mfu",
    "create_profiler",
    "dENGINE",
    "dengine",
    "engine",
    "estimate_hardware_peak_tflops",
    "export_checkpoint_to_hf",
    "generate_samples",
    "generate_with_cache",
    "get_lr",
    "is_mps_available",
    "kernels",
    "load_model",
    "models",
    "optim",
    "precompute_rope_frequencies",
    "repeat_kv",
    "sample_logits",
    "telemetry",
    "train",
    "zeropower_via_newtonschulz5",
]
