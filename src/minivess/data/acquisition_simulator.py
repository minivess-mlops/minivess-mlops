"""Synthetic acquisition simulator for temporal drift streams.

Generates controlled distribution shifts over time for drift monitoring
validation. Uses :func:`~minivess.data.drift_synthetic.apply_drift` as
the underlying drift engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import torch

from minivess.data.drift_synthetic import DriftType, apply_drift

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DriftSchedule:
    """Schedule for a single drift type over a timepoint range.

    Parameters
    ----------
    drift_type:
        Type of drift to apply.
    start_timepoint:
        First timepoint where this drift becomes active.
    end_timepoint:
        Last timepoint (exclusive) for this drift.
    severity_curve:
        How severity increases: ``"linear"`` (gradual), ``"exponential"``
        (slow start, rapid end), or ``"step"`` (instant at midpoint).
    max_severity:
        Maximum severity at end of range (default 1.0).
    """

    drift_type: DriftType
    start_timepoint: int
    end_timepoint: int
    severity_curve: Literal["linear", "exponential", "step"] = "linear"
    max_severity: float = 1.0


@dataclass
class AcquisitionSimulatorConfig:
    """Configuration for the synthetic acquisition simulator.

    Parameters
    ----------
    n_timepoints:
        Number of temporal acquisition points to generate.
    base_volume_shape:
        Shape of the base volume ``(C, D, H, W)``.
    schedules:
        List of drift schedules to apply.
    seed:
        Random seed for reproducibility.
    """

    n_timepoints: int
    base_volume_shape: tuple[int, ...]
    schedules: list[DriftSchedule] = field(default_factory=list)
    seed: int = 42


def _compute_severity(
    schedule: DriftSchedule,
    timepoint: int,
) -> float:
    """Compute drift severity at a given timepoint."""
    if timepoint < schedule.start_timepoint or timepoint >= schedule.end_timepoint:
        return 0.0

    span = schedule.end_timepoint - schedule.start_timepoint
    if span <= 0:
        return 0.0

    progress = (timepoint - schedule.start_timepoint) / span

    if schedule.severity_curve == "linear":
        return progress * schedule.max_severity
    elif schedule.severity_curve == "exponential":
        return (progress**2) * schedule.max_severity
    elif schedule.severity_curve == "step":
        return schedule.max_severity if progress >= 0.5 else 0.0
    else:
        return progress * schedule.max_severity


class SyntheticAcquisitionSimulator:
    """Generates temporal streams of volumes with controlled drift.

    Parameters
    ----------
    config:
        Simulator configuration.
    base_volume:
        Optional pre-existing base volume. If ``None``, generates a
        random volume matching ``config.base_volume_shape``.
    """

    def __init__(
        self,
        config: AcquisitionSimulatorConfig,
        base_volume: torch.Tensor | None = None,
    ) -> None:
        self.config = config
        self._rng = torch.Generator().manual_seed(config.seed)

        if base_volume is not None:
            self.base_volume = base_volume.clone()
        else:
            self.base_volume = torch.rand(
                config.base_volume_shape,
                generator=self._rng,
            )

        self._start_time = datetime.now(UTC)

    def generate_timepoint(self, t: int) -> dict[str, Any]:
        """Generate a single timepoint with accumulated drift.

        Parameters
        ----------
        t:
            Timepoint index (0-based).

        Returns
        -------
        Dict with ``"volume"`` (Tensor) and ``"metadata"`` (dict).
        """
        volume = self.base_volume.clone()

        active_drifts: list[dict[str, Any]] = []
        for schedule in self.config.schedules:
            severity = _compute_severity(schedule, t)
            if severity > 1e-9:
                volume = apply_drift(
                    volume,
                    drift_type=schedule.drift_type,
                    severity=severity,
                    seed=self.config.seed + t,
                )
                active_drifts.append(
                    {
                        "drift_type": str(schedule.drift_type),
                        "severity": severity,
                    }
                )

        metadata: dict[str, Any] = {
            "timepoint": t,
            "timestamp": (self._start_time + timedelta(seconds=t * 60)).isoformat(),
            "active_drifts": active_drifts,
            "n_schedules_active": len(active_drifts),
        }

        return {"volume": volume, "metadata": metadata}

    def generate_batch(self) -> dict[str, Any]:
        """Generate all timepoints as a batch.

        Returns
        -------
        Dict with ``"volumes"`` (list of Tensors) and ``"metadata"``
        (list of dicts).
        """
        volumes: list[torch.Tensor] = []
        metadata_list: list[dict[str, Any]] = []

        for t in range(self.config.n_timepoints):
            result = self.generate_timepoint(t)
            volumes.append(result["volume"])
            metadata_list.append(result["metadata"])

        logger.info(
            "Generated %d timepoints with %d drift schedules",
            self.config.n_timepoints,
            len(self.config.schedules),
        )

        return {"volumes": volumes, "metadata": metadata_list}
