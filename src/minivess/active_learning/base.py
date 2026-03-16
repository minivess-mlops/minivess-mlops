"""Abstract base classes for active learning.

Defines the interfaces for uncertainty-based sample selection and
MONAI Label integration. Concrete implementations will be provided
in a follow-up PR.

Strategies:
    - max_entropy: Select samples with highest predictive entropy
    - max_mc_variance: Select samples with highest MC Dropout variance
    - bald: Bayesian Active Learning by Disagreement
    - max_mahalanobis: Select samples with highest Mahalanobis distance
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


@dataclass
class AnnotationRequest:
    """Request to annotate a specific volume.

    Created by the active learning pipeline when a volume is selected
    for human annotation based on high epistemic uncertainty.
    """

    volume_id: str
    uncertainty_score: float
    source: str  # e.g. "mc_dropout", "mahalanobis", "bald"
    priority: int  # lower = higher priority
    metadata: dict[str, Any] | None = field(default=None)


class UncertaintySampler(ABC):
    """ABC for uncertainty-based sample selection strategies.

    Given per-volume uncertainty scores, select the top-N volumes
    for human annotation to maximize information gain.
    """

    @abstractmethod
    def select_samples(
        self,
        uncertainty_scores: np.ndarray,
        n: int,
    ) -> np.ndarray:
        """Select *n* volume indices with highest uncertainty.

        Args:
            uncertainty_scores: 1-D array of shape ``(n_volumes,)``
                with per-volume uncertainty estimates.
            n: Number of samples to select.

        Returns:
            1-D array of indices into *uncertainty_scores*, sorted
            by descending uncertainty.
        """

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Name of the sampling strategy (e.g. ``'max_entropy'``)."""


class MONAILabelAdapter(ABC):
    """ABC for MONAI Label integration.

    Provides the interface for submitting volumes to MONAI Label
    for annotation and fetching completed annotations back.
    """

    @abstractmethod
    def submit_for_annotation(
        self,
        requests: list[AnnotationRequest],
    ) -> list[str]:
        """Submit annotation requests to MONAI Label.

        Args:
            requests: List of ``AnnotationRequest`` objects.

        Returns:
            List of task IDs from MONAI Label.
        """

    @abstractmethod
    def fetch_annotations(
        self,
        task_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Fetch completed annotations from MONAI Label.

        Args:
            task_ids: Task IDs returned by ``submit_for_annotation``.

        Returns:
            List of annotation result dicts (volume_id, mask_path, status).
        """
