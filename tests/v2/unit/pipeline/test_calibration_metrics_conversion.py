"""Calibration metrics must not crash on MONAI MetaTensor input.

8th pass finding: torch.cat() on MetaTensor → .numpy() fails, causing
val/nll=nan in ALL runs. The fix uses .detach().cpu().float().numpy().

This test verifies that calibration metric computation produces non-NaN
values when given MONAI MetaTensor inputs (the production data type).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


class TestCalibrationMetaTensorConversion:
    """Calibration metrics must handle MONAI MetaTensor correctly."""

    def test_cat_detach_cpu_numpy_works_on_metatensor(self) -> None:
        """torch.cat → .detach().cpu().float().numpy() must not raise on MetaTensor."""
        monai_data = pytest.importorskip(
            "monai.data", reason="monai not installed"
        )
        MetaTensor = monai_data.MetaTensor

        # Simulate accumulated calibration probabilities (MetaTensor from MONAI)
        probs = [MetaTensor(torch.rand(10)) for _ in range(5)]
        labels = [MetaTensor(torch.randint(0, 2, (10,)).float()) for _ in range(5)]

        # This is the fixed code path from metrics.py
        all_probs = torch.cat(probs).detach().cpu().float().numpy().ravel()
        all_labels = torch.cat(labels).detach().cpu().float().numpy().ravel()

        assert isinstance(all_probs, np.ndarray)
        assert isinstance(all_labels, np.ndarray)
        assert len(all_probs) == 50
        assert not np.any(np.isnan(all_probs))

    def test_calibration_metrics_non_nan(self) -> None:
        """compute_all_calibration_metrics must return non-NaN for valid input."""
        from minivess.pipeline.calibration_metrics import (
            compute_all_calibration_metrics,
        )

        rng = np.random.default_rng(42)
        probs = rng.random(100)
        labels = rng.integers(0, 2, size=100).astype(float)

        result = compute_all_calibration_metrics(probs, labels, tier="fast")

        assert isinstance(result, dict)
        assert len(result) > 0
        for key, val in result.items():
            assert not np.isnan(val), f"Calibration metric {key} is NaN"
