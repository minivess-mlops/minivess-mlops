"""Tests for quality gate tasks wired into data_flow (T3, T4, T6).

Covers pandera_gate_task, ge_gate_task, datacare_gate_task, deepchecks_gate_task.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_valid_metadata_df(n: int = 5) -> pd.DataFrame:
    """Create a valid NIfTI metadata DataFrame for testing."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "file_path": [f"/data/vol_{i:03d}.nii.gz" for i in range(n)],
            "shape_x": [64] * n,
            "shape_y": [64] * n,
            "shape_z": [32] * n,
            "voxel_spacing_x": [0.5] * n,
            "voxel_spacing_y": [0.5] * n,
            "voxel_spacing_z": [1.0] * n,
            "intensity_min": rng.uniform(0.0, 10.0, n).tolist(),
            "intensity_max": rng.uniform(500.0, 1000.0, n).tolist(),
            "num_foreground_voxels": [4000] * n,
            "has_valid_affine": [True] * n,
        }
    )


def _make_invalid_metadata_df() -> pd.DataFrame:
    """Create an invalid NIfTI metadata DataFrame (out-of-range spacing)."""
    return pd.DataFrame(
        {
            "file_path": ["/data/bad_vol.nii.gz"],
            "shape_x": [64],
            "shape_y": [64],
            "shape_z": [32],
            "voxel_spacing_x": [99.0],  # Out of range (max 10.0)
            "voxel_spacing_y": [0.5],
            "voxel_spacing_z": [1.0],
            "intensity_min": [0.0],
            "intensity_max": [1000.0],
            "num_foreground_voxels": [4000],
            "has_valid_affine": [True],
        }
    )


# ---------------------------------------------------------------------------
# T3: Pandera gate task
# ---------------------------------------------------------------------------


class TestPanderaGateTask:
    """pandera_gate_task validates metadata via Pandera schema."""

    def test_returns_gate_result(self) -> None:
        """T3-R1: Returns a GateResult."""
        from minivess.orchestration.flows.data_flow import pandera_gate_task
        from minivess.validation.gates import GateResult

        df = _make_valid_metadata_df()
        result = pandera_gate_task(df)
        assert isinstance(result, GateResult)

    def test_valid_data_passes(self) -> None:
        """T3-R2: Valid data returns passed=True."""
        from minivess.orchestration.flows.data_flow import pandera_gate_task

        df = _make_valid_metadata_df()
        result = pandera_gate_task(df)
        assert result.passed is True

    def test_invalid_data_fails(self) -> None:
        """T3-R3: Invalid data returns passed=False."""
        from minivess.orchestration.flows.data_flow import pandera_gate_task

        df = _make_invalid_metadata_df()
        result = pandera_gate_task(df)
        assert result.passed is False
        assert len(result.errors) > 0


# ---------------------------------------------------------------------------
# T3: GE gate task
# ---------------------------------------------------------------------------


class TestGEGateTask:
    """ge_gate_task validates metadata via Great Expectations."""

    def test_returns_gate_result(self) -> None:
        """T3-R4: Returns a GateResult."""
        from minivess.orchestration.flows.data_flow import ge_gate_task
        from minivess.validation.gates import GateResult

        df = _make_valid_metadata_df()
        result = ge_gate_task(df)
        assert isinstance(result, GateResult)

    def test_valid_data_passes(self) -> None:
        """T3-R5: Valid data returns passed=True."""
        from minivess.orchestration.flows.data_flow import ge_gate_task

        df = _make_valid_metadata_df()
        result = ge_gate_task(df)
        assert result.passed is True

    def test_invalid_spacing_fails(self) -> None:
        """T3-R6: Out-of-range spacing returns passed=False."""
        from minivess.orchestration.flows.data_flow import ge_gate_task

        df = _make_invalid_metadata_df()
        result = ge_gate_task(df)
        assert result.passed is False
        assert len(result.errors) > 0
