"""
AxiomLM Learning Rate Scheduling & Optimization Utilities.
"""
import math
from typing import Optional


def get_lr(
    it: int,
    warmup_steps: int = 300,
    max_steps: int = 4800,
    max_lr: float = 6e-4,
    min_lr: Optional[float] = None,
) -> float:
    """
    Computes scheduled learning rate with linear warmup and cosine decay.
    """
    if min_lr is None:
        min_lr = max_lr * 0.1
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
