"""Calibration-under-shift framework (Moreo et al., 2025).

Evaluates calibration degradation under synthetic domain shifts and
provides cross-site calibration transfer analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np

from minivess.ensemble.calibration import expected_calibration_error

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ShiftType(StrEnum):
    """Types of distribution shift."""

    COVARIATE = "covariate"
    LABEL = "label"
    CONCEPT = "concept"


@dataclass
class ShiftedCalibrationResult:
    """Calibration transfer analysis result.

    Parameters
    ----------
    source_domain:
        Name of the source domain.
    target_domain:
        Name of the target domain.
    source_ece:
        ECE on the source domain.
    target_ece:
        ECE on the target domain.
    degradation:
        Increase in ECE from source to target.
    """

    source_domain: str
    target_domain: str
    source_ece: float
    target_ece: float
    degradation: float


def apply_synthetic_shift(
    data: NDArray,
    *,
    shift_type: str,
    magnitude: float,
    seed: int | None = None,
) -> NDArray:
    """Apply a synthetic domain shift to confidence data.

    Parameters
    ----------
    data:
        Input array to shift.
    shift_type:
        Type of shift: "intensity" (additive), "noise" (Gaussian).
    magnitude:
        Magnitude of the shift.
    seed:
        Random seed for reproducibility.
    """
    if magnitude == 0.0:
        return data.copy()

    rng = np.random.default_rng(seed)

    if shift_type == "intensity":
        return data + magnitude
    if shift_type == "noise":
        noise = rng.normal(0, magnitude, size=data.shape)
        return data + noise

    return data.copy()


def evaluate_calibration_transfer(
    *,
    source_confidences: NDArray,
    source_accuracies: NDArray,
    target_confidences: NDArray,
    target_accuracies: NDArray,
    n_bins: int = 15,
    source_domain: str = "source",
    target_domain: str = "target",
) -> ShiftedCalibrationResult:
    """Evaluate calibration degradation from source to target domain.

    Parameters
    ----------
    source_confidences:
        Confidence scores on source domain.
    source_accuracies:
        Binary correctness on source domain.
    target_confidences:
        Confidence scores on target domain.
    target_accuracies:
        Binary correctness on target domain.
    """
    source_ece, _ = expected_calibration_error(
        source_confidences,
        source_accuracies,
        n_bins=n_bins,
    )
    target_ece, _ = expected_calibration_error(
        target_confidences,
        target_accuracies,
        n_bins=n_bins,
    )
    return ShiftedCalibrationResult(
        source_domain=source_domain,
        target_domain=target_domain,
        source_ece=source_ece,
        target_ece=target_ece,
        degradation=target_ece - source_ece,
    )


class CalibrationShiftAnalyzer:
    """Cross-domain calibration shift analyzer.

    Registers multiple domains and computes pairwise calibration
    transfer degradation.
    """

    def __init__(self) -> None:
        self.domains: dict[str, dict[str, Any]] = {}

    def add_domain(
        self,
        name: str,
        confidences: NDArray,
        accuracies: NDArray,
    ) -> None:
        """Register a domain's predictions.

        Parameters
        ----------
        name:
            Domain identifier.
        confidences:
            Predicted confidence scores.
        accuracies:
            Binary correctness indicators.
        """
        self.domains[name] = {
            "confidences": confidences,
            "accuracies": accuracies,
        }

    def analyze_transfer(self) -> list[ShiftedCalibrationResult]:
        """Compute pairwise calibration transfer analysis.

        Returns
        -------
        List of ShiftedCalibrationResult for all domain pairs.
        """
        results: list[ShiftedCalibrationResult] = []
        for src_name, tgt_name in combinations(sorted(self.domains), 2):
            src = self.domains[src_name]
            tgt = self.domains[tgt_name]
            result = evaluate_calibration_transfer(
                source_confidences=src["confidences"],
                source_accuracies=src["accuracies"],
                target_confidences=tgt["confidences"],
                target_accuracies=tgt["accuracies"],
                source_domain=src_name,
                target_domain=tgt_name,
            )
            results.append(result)
        return results

    def to_markdown(self) -> str:
        """Generate a calibration transfer report."""
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        sections = [
            "# Calibration-Under-Shift Analysis",
            "",
            f"**Generated:** {now}",
            f"**Domains:** {len(self.domains)}",
            "",
        ]

        if len(self.domains) < 2:  # noqa: PLR2004
            sections.append("Need at least 2 domains for transfer analysis.")
            sections.append("")
            return "\n".join(sections)

        # Per-domain ECE
        sections.extend(
            [
                "## Per-Domain Calibration",
                "",
                "| Domain | ECE |",
                "|--------|-----|",
            ]
        )
        for name in sorted(self.domains):
            d = self.domains[name]
            ece, _ = expected_calibration_error(
                d["confidences"],
                d["accuracies"],
            )
            sections.append(f"| {name} | {ece:.4f} |")

        # Pairwise transfer
        results = self.analyze_transfer()
        sections.extend(
            [
                "",
                "## Pairwise Transfer Degradation",
                "",
                "| Source | Target | Source ECE | Target ECE | Degradation |",
                "|--------|--------|-----------|-----------|-------------|",
            ]
        )
        for r in results:
            sections.append(
                f"| {r.source_domain} | {r.target_domain} "
                f"| {r.source_ece:.4f} | {r.target_ece:.4f} "
                f"| {r.degradation:+.4f} |"
            )

        sections.append("")
        return "\n".join(sections)
