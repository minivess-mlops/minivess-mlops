"""Centralized seed management for reproducible experiments.

Sets seeds across all random number generators used in the project:
Python ``random``, NumPy, and PyTorch.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seed(seed: int) -> int:
    """Set the global random seed for all frameworks.

    Parameters
    ----------
    seed:
        Integer seed value.

    Returns
    -------
    The seed that was set.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Enable deterministic CUDA/cuDNN operations for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed
