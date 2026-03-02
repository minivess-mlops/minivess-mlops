"""Tests for SyntheticAcquisitionSimulator.

Covers Task 2.1 of data-engineering-improvement-plan.xml.
Closes #177.
"""

from __future__ import annotations

import torch

from minivess.data.acquisition_simulator import (
    AcquisitionSimulatorConfig,
    DriftSchedule,
    SyntheticAcquisitionSimulator,
)
from minivess.data.drift_synthetic import DriftType


class TestDriftSchedule:
    """DriftSchedule dataclass validation."""

    def test_drift_schedule_has_required_fields(self) -> None:
        sched = DriftSchedule(
            drift_type=DriftType.NOISE_INJECTION,
            start_timepoint=0,
            end_timepoint=10,
            severity_curve="linear",
        )
        assert sched.drift_type == DriftType.NOISE_INJECTION
        assert sched.start_timepoint == 0
        assert sched.end_timepoint == 10
        assert sched.severity_curve == "linear"


class TestSyntheticAcquisitionSimulator:
    """Temporal drift stream generation."""

    def _make_config(self, n_timepoints: int = 5) -> AcquisitionSimulatorConfig:
        return AcquisitionSimulatorConfig(
            n_timepoints=n_timepoints,
            base_volume_shape=(1, 8, 8, 8),
            schedules=[
                DriftSchedule(
                    drift_type=DriftType.NOISE_INJECTION,
                    start_timepoint=0,
                    end_timepoint=n_timepoints,
                    severity_curve="linear",
                ),
            ],
            seed=42,
        )

    def test_simulator_produces_n_timepoints(self) -> None:
        config = self._make_config(n_timepoints=5)
        sim = SyntheticAcquisitionSimulator(config)
        batch = sim.generate_batch()
        assert len(batch["volumes"]) == 5

    def test_simulator_gradual_noise_increases(self) -> None:
        config = self._make_config(n_timepoints=10)
        sim = SyntheticAcquisitionSimulator(config)
        batch = sim.generate_batch()
        # Later timepoints should differ more from base
        base = sim.base_volume
        early_diff = (batch["volumes"][1] - base).abs().mean().item()
        late_diff = (batch["volumes"][8] - base).abs().mean().item()
        assert late_diff > early_diff

    def test_simulator_seed_reproducibility(self) -> None:
        config = self._make_config()
        sim1 = SyntheticAcquisitionSimulator(config)
        sim2 = SyntheticAcquisitionSimulator(config)
        b1 = sim1.generate_batch()
        b2 = sim2.generate_batch()
        for v1, v2 in zip(b1["volumes"], b2["volumes"], strict=True):
            assert torch.allclose(v1, v2)

    def test_simulator_metadata_has_timestamps(self) -> None:
        config = self._make_config()
        sim = SyntheticAcquisitionSimulator(config)
        batch = sim.generate_batch()
        for meta in batch["metadata"]:
            assert "timestamp" in meta
            assert "timepoint" in meta

    def test_simulator_generate_timepoint(self) -> None:
        config = self._make_config()
        sim = SyntheticAcquisitionSimulator(config)
        result = sim.generate_timepoint(0)
        assert "volume" in result
        assert "metadata" in result
        assert isinstance(result["volume"], torch.Tensor)

    def test_simulator_generate_batch_returns_dict(self) -> None:
        config = self._make_config()
        sim = SyntheticAcquisitionSimulator(config)
        batch = sim.generate_batch()
        assert "volumes" in batch
        assert "metadata" in batch

    def test_simulator_with_multiple_schedules(self) -> None:
        config = AcquisitionSimulatorConfig(
            n_timepoints=5,
            base_volume_shape=(1, 8, 8, 8),
            schedules=[
                DriftSchedule(
                    drift_type=DriftType.NOISE_INJECTION,
                    start_timepoint=0,
                    end_timepoint=5,
                    severity_curve="linear",
                ),
                DriftSchedule(
                    drift_type=DriftType.INTENSITY_SHIFT,
                    start_timepoint=2,
                    end_timepoint=5,
                    severity_curve="step",
                ),
            ],
            seed=42,
        )
        sim = SyntheticAcquisitionSimulator(config)
        batch = sim.generate_batch()
        assert len(batch["volumes"]) == 5
