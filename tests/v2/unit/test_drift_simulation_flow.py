"""Tests for drift simulation Prefect flow (T-E1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def reference_volumes() -> list[np.ndarray]:
    """20 reference volumes for drift baseline."""
    rng = np.random.default_rng(42)
    return [rng.random((32, 32, 8), dtype=np.float32) for _ in range(20)]


class TestDriftSimulationFlow:
    """Test drift simulation flow tasks."""

    def test_extract_reference_features_task(
        self, reference_volumes: list[np.ndarray]
    ) -> None:
        """Should extract features from reference volumes."""
        from minivess.orchestration.flows.drift_simulation_flow import (
            extract_reference_features_task,
        )

        result = extract_reference_features_task.fn(reference_volumes)
        assert "features" in result
        assert result["n_volumes"] == 20

    def test_run_batch_drift_task(self, reference_volumes: list[np.ndarray]) -> None:
        """Should run drift detection on a batch."""
        from minivess.orchestration.flows.drift_simulation_flow import (
            extract_reference_features_task,
            run_batch_drift_task,
        )

        ref_result = extract_reference_features_task.fn(reference_volumes)

        # Create a drifted batch
        batch = [(v * 2.5 + 0.3).astype(np.float32) for v in reference_volumes[:2]]

        drift_result = run_batch_drift_task.fn(
            batch_volumes=batch,
            reference_features=ref_result["features"],
            batch_id=0,
        )
        assert "drift_detected" in drift_result
        assert drift_result["batch_id"] == 0

    def test_flow_definition_exists(self) -> None:
        """The flow function should be importable."""
        from minivess.orchestration.flows.drift_simulation_flow import (
            drift_simulation_flow,
        )

        assert callable(drift_simulation_flow)

    def test_flow_runs_with_synthetic_data(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        reference_volumes: list[np.ndarray],
    ) -> None:
        """Flow should complete on synthetic data."""
        monkeypatch.setenv("MINIVESS_ALLOW_HOST", "1")
        monkeypatch.setenv("PREFECT_DISABLED", "1")

        from minivess.orchestration.flows.drift_simulation_flow import (
            drift_simulation_flow,
        )

        # Create 6 batches of 2 volumes each
        rng = np.random.default_rng(42)
        batches = []
        for i in range(6):
            shift = i * 0.3  # progressive drift
            batch = [
                (rng.random((32, 32, 8), dtype=np.float32) + shift).astype(np.float32)
                for _ in range(2)
            ]
            batches.append(batch)

        result = drift_simulation_flow.fn(
            reference_volumes=reference_volumes,
            batches=batches,
            output_dir=str(tmp_path / "drift_reports"),
        )
        assert result["status"] == "completed"
        assert result["n_batches"] == 6
        assert len(result["batch_results"]) == 6

    def test_progressive_drift_detected(
        self, reference_volumes: list[np.ndarray]
    ) -> None:
        """Later batches with more drift should have more detections."""
        from minivess.orchestration.flows.drift_simulation_flow import (
            extract_reference_features_task,
            run_batch_drift_task,
        )

        ref_result = extract_reference_features_task.fn(reference_volumes)

        # No drift
        rng = np.random.default_rng(99)
        clean_batch = [rng.random((32, 32, 8), dtype=np.float32) for _ in range(2)]
        clean_result = run_batch_drift_task.fn(
            batch_volumes=clean_batch,
            reference_features=ref_result["features"],
            batch_id=0,
        )

        # Heavy drift
        heavy_batch = [
            (v * 3.0 + 1.0).astype(np.float32) for v in reference_volumes[:2]
        ]
        heavy_result = run_batch_drift_task.fn(
            batch_volumes=heavy_batch,
            reference_features=ref_result["features"],
            batch_id=5,
        )

        # Heavy drift should have higher drift score
        assert (
            heavy_result["dataset_drift_score"] >= clean_result["dataset_drift_score"]
        )
