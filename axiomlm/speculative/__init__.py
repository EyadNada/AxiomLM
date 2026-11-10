"""
AxiomLM Speculative Decoding Engine & Sequoia Hardware-Aware Tree Optimization.
"""

from .sampling import (
    SequoiaTreeVerifier,
    VerificationStepResult,
    fit_power_law_rejection_bound,
    sample_without_replacement,
)
from .sequoia_dp import OptimalTreeResult, SequoiaDPOptimizer

__all__ = [
    "OptimalTreeResult",
    "SequoiaDPOptimizer",
    "sample_without_replacement",
    "SequoiaTreeVerifier",
    "VerificationStepResult",
    "fit_power_law_rejection_bound",
]
