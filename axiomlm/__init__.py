"""
AxiomLM: High-Performance Systems LLM Pretraining & Hardware Kernel Engine.

Modern LLaMA-3 Architecture (RoPE + RMSNorm + SwiGLU + GQA),
Muon Newton-Schulz Optimizer, and Bare-Metal Fused Metal / Triton Kernels.
"""

from .models.transformer import (
    Transformer,
    ModelConfig,
    GPT,
    GPTConfig,
    Block,
)
from .models.modules import (
    RMSNorm,
    SwiGLUMLP,
    MLP,
    precompute_rope_frequencies,
    apply_rope,
    repeat_kv,
)
from .models.attention import CausalSelfAttention
from .optim.muon import Muon, zeropower_via_newtonschulz5
from .optim.schedule import get_lr
from .engine.inference import (
    InferenceEngine,
    sample_logits,
    generate_samples,
    generate_with_cache,
    load_model,
)
from .engine.export import export_checkpoint_to_hf
from .dengine.dataloader import DataLoaderLite
from .telemetry.profiler import calculate_mfu, estimate_hardware_peak_tflops, create_profiler
from .train import train

from . import models
from . import optim
from . import engine
from . import dengine
from . import telemetry

dENGINE = dengine  # type: ignore

try:
    from . import kernels
except ImportError:
    kernels = None  # type: ignore

__version__ = "0.1.0"

__all__ = [
    "Transformer",
    "ModelConfig",
    "GPT",
    "GPTConfig",
    "Block",
    "RMSNorm",
    "SwiGLUMLP",
    "MLP",
    "precompute_rope_frequencies",
    "apply_rope",
    "repeat_kv",
    "CausalSelfAttention",
    "Muon",
    "zeropower_via_newtonschulz5",
    "get_lr",
    "InferenceEngine",
    "sample_logits",
    "generate_samples",
    "generate_with_cache",
    "load_model",
    "export_checkpoint_to_hf",
    "DataLoaderLite",
    "calculate_mfu",
    "estimate_hardware_peak_tflops",
    "create_profiler",
    "train",
    "models",
    "optim",
    "kernels",
    "engine",
    "dengine",
    "dENGINE",
    "telemetry",
    "__version__",
]
