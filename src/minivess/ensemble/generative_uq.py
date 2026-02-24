"""Generative UQ methods for multi-rater segmentation uncertainty.

Implements evaluation infrastructure for Probabilistic U-Net (Kohl 2018),
PHiSeg (Baumgartner 2019), and Stochastic Segmentation Networks (Monteiro
2020). Supports Generalized Energy Distance and Q-Dice metrics for
multi-rater ground truth evaluation (QUBIQ benchmark, Bran 2024).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class GenerativeUQMethod(StrEnum):
    """Generative uncertainty quantification methods."""

    PROB_UNET = "prob_unet"
    PHISEG = "phiseg"
    SSN = "ssn"


@dataclass
class MultiRaterData:
    """Multi-rater annotation data for a single volume.

    Parameters
    ----------
    volume_id:
        Volume identifier.
    rater_masks:
        List of binary segmentation masks from different annotators.
    """

    volume_id: str
    rater_masks: list[NDArray] = field(default_factory=list)

    @property
    def num_raters(self) -> int:
        """Number of raters."""
        return len(self.rater_masks)


@dataclass
class GenerativeUQConfig:
    """Configuration for generative UQ methods.

    Parameters
    ----------
    method:
        Generative UQ method to use.
    latent_dim:
        Dimensionality of the latent space.
    num_samples:
        Number of samples to draw for evaluation.
    """

    method: str = "prob_unet"
    latent_dim: int = 6
    num_samples: int = 16


def _dice_distance(a: NDArray, b: NDArray) -> float:
    """Compute 1 - Dice between two binary masks."""
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    intersection = np.sum(a_bool & b_bool)
    total = np.sum(a_bool) + np.sum(b_bool)
    if total == 0:
        return 0.0
    dice = 2.0 * intersection / total
    return 1.0 - dice


def generalized_energy_distance(
    samples: list[NDArray],
    references: list[NDArray],
) -> float:
    """Compute Generalized Energy Distance (GED) between sample sets.

    GED = 2 * E[d(S, R)] - E[d(S, S')] - E[d(R, R')]

    where d is the Dice distance (1 - Dice), S are model samples,
    and R are rater annotations.

    Parameters
    ----------
    samples:
        List of model-generated segmentation samples.
    references:
        List of rater annotations (ground truth).
    """
    n_s = len(samples)
    n_r = len(references)

    if n_s == 0 or n_r == 0:
        return 0.0

    # E[d(S, R)] — cross term
    cross_sum = sum(
        _dice_distance(samples[i], references[j])
        for i in range(n_s)
        for j in range(n_r)
    )
    cross_term = cross_sum / (n_s * n_r)

    # E[d(S, S')] — sample-sample term
    if n_s > 1:
        ss_sum = sum(
            _dice_distance(samples[i], samples[j])
            for i in range(n_s)
            for j in range(i + 1, n_s)
        )
        ss_term = 2.0 * ss_sum / (n_s * (n_s - 1))
    else:
        ss_term = 0.0

    # E[d(R, R')] — reference-reference term
    if n_r > 1:
        rr_sum = sum(
            _dice_distance(references[i], references[j])
            for i in range(n_r)
            for j in range(i + 1, n_r)
        )
        rr_term = 2.0 * rr_sum / (n_r * (n_r - 1))
    else:
        rr_term = 0.0

    return max(0.0, 2.0 * cross_term - ss_term - rr_term)


def q_dice(
    prob_map: NDArray,
    reference: NDArray,
    thresholds: list[float] | None = None,
) -> float:
    """Compute Q-Dice (Quantized Dice) from QUBIQ benchmark.

    Averages Dice scores across multiple probability thresholds,
    providing a threshold-invariant segmentation quality measure.

    Parameters
    ----------
    prob_map:
        Probability map (float, 0-1).
    reference:
        Binary reference mask.
    thresholds:
        List of thresholds to evaluate. Default: [0.1, 0.3, 0.5, 0.7, 0.9].
    """
    if thresholds is None:
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

    ref_bool = reference.astype(bool)
    dice_scores: list[float] = []

    for t in thresholds:
        pred_bool = prob_map >= t
        intersection = np.sum(pred_bool & ref_bool)
        total = np.sum(pred_bool) + np.sum(ref_bool)
        if total == 0:
            dice_scores.append(1.0)
        else:
            dice_scores.append(float(2.0 * intersection / total))

    return float(np.mean(dice_scores))


class GenerativeUQEvaluator:
    """Evaluates generative UQ methods against multi-rater ground truth.

    Computes GED and Q-Dice metrics for each volume, supporting
    comparison across different generative UQ methods.
    """

    def __init__(self) -> None:
        self.volumes: dict[str, dict[str, Any]] = {}

    def add_volume(
        self,
        volume_id: str,
        prediction_samples: list[NDArray],
        rater_annotations: list[NDArray],
    ) -> None:
        """Register prediction samples and rater annotations for a volume.

        Parameters
        ----------
        volume_id:
            Volume identifier.
        prediction_samples:
            Model-generated segmentation samples.
        rater_annotations:
            Ground-truth rater annotations.
        """
        self.volumes[volume_id] = {
            "samples": prediction_samples,
            "raters": rater_annotations,
        }

    def compute_metrics(self, volume_id: str) -> dict[str, float]:
        """Compute GED and Q-Dice for a volume.

        Parameters
        ----------
        volume_id:
            Volume identifier.
        """
        data = self.volumes[volume_id]
        samples = data["samples"]
        raters = data["raters"]

        ged = generalized_energy_distance(samples, raters)

        # Q-Dice: average probability map from samples vs first rater
        prob_map = np.mean(
            np.stack([s.astype(np.float32) for s in samples], axis=0),
            axis=0,
        )
        qd = q_dice(prob_map, raters[0])

        return {"ged": ged, "q_dice": qd}

    def to_markdown(self) -> str:
        """Generate an evaluation report."""
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        sections = [
            "# Generative UQ Evaluation Report",
            "",
            f"**Generated:** {now}",
            f"**Volumes:** {len(self.volumes)}",
            "",
        ]

        if not self.volumes:
            sections.append("No volumes registered.")
            sections.append("")
            return "\n".join(sections)

        sections.extend([
            "## Per-Volume Metrics",
            "",
            "| Volume | Samples | Raters | GED | Q-Dice |",
            "|--------|---------|--------|-----|--------|",
        ])

        for vid in sorted(self.volumes):
            data = self.volumes[vid]
            metrics = self.compute_metrics(vid)
            sections.append(
                f"| {vid} | {len(data['samples'])} "
                f"| {len(data['raters'])} "
                f"| {metrics['ged']:.4f} | {metrics['q_dice']:.4f} |"
            )

        sections.append("")
        return "\n".join(sections)
