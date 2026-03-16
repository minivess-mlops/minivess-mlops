"""Active learning interfaces for uncertainty-based sample selection.

Architecture stubs — full implementation in follow-up PR.
Defines ABCs for uncertainty-based sampling strategies and
MONAI Label integration.
"""

from __future__ import annotations

from minivess.active_learning.base import (
    AnnotationRequest,
    MONAILabelAdapter,
    UncertaintySampler,
)

# Supported sampling strategies (implemented in follow-up PR)
SAMPLING_STRATEGIES: list[str] = [
    "max_entropy",
    "max_mc_variance",
    "bald",
    "max_mahalanobis",
]

__all__ = [
    "SAMPLING_STRATEGIES",
    "AnnotationRequest",
    "MONAILabelAdapter",
    "UncertaintySampler",
]
