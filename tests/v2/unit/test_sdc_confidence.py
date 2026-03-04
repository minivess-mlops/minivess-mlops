"""Tests for Soft Dice Confidence (SDC) quality gate.

SDC (Borges et al. 2025, arXiv:2402.10665v4) is a near-optimal confidence
estimator for selective prediction: SDC = 2*sum(p*y_hat) / sum(p + y_hat),
where p = softmax probabilities and y_hat = hard predictions.

Issue: #306 | Phase 0 | Plan: T0.1 (RED)
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch


class TestSDCMetric:
    """Unit tests for compute_sdc() function."""

    def test_sdc_perfect_prediction_returns_one(self) -> None:
        """When softmax probability is 1.0 everywhere the prediction is 1,
        SDC should be 1.0 (perfect confidence)."""
        from minivess.pipeline.validation_metrics import compute_sdc

        # Binary: 2 classes, batch=1, small volume 4x4x4
        probs = torch.zeros(1, 2, 4, 4, 4)
        probs[:, 1, :, :, :] = 1.0  # 100% confident class 1
        preds = torch.ones(1, 1, 4, 4, 4, dtype=torch.long)  # all class 1

        sdc = compute_sdc(probs, preds)
        assert sdc == pytest.approx(1.0, abs=1e-6)

    def test_sdc_empty_prediction_returns_zero(self) -> None:
        """When prediction is all background (class 0) with high confidence,
        SDC for the foreground should be 0.0."""
        from minivess.pipeline.validation_metrics import compute_sdc

        probs = torch.zeros(1, 2, 4, 4, 4)
        probs[:, 0, :, :, :] = 1.0  # 100% confident background
        preds = torch.zeros(1, 1, 4, 4, 4, dtype=torch.long)  # all background

        sdc = compute_sdc(probs, preds)
        assert sdc == pytest.approx(0.0, abs=1e-6)

    def test_sdc_bounded_zero_one(self) -> None:
        """SDC must always be in [0, 1] for any valid input."""
        from minivess.pipeline.validation_metrics import compute_sdc

        rng = torch.Generator().manual_seed(42)
        # Random softmax probabilities
        logits = torch.randn(2, 2, 8, 8, 8, generator=rng)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1, keepdim=True)

        sdc = compute_sdc(probs, preds)
        assert 0.0 <= sdc <= 1.0, f"SDC={sdc} out of bounds"

    def test_sdc_nan_safe(self) -> None:
        """SDC should return 0.0 when both numerator and denominator are zero
        (no foreground predicted and no foreground probability)."""
        from minivess.pipeline.validation_metrics import compute_sdc

        probs = torch.zeros(1, 2, 4, 4, 4)
        probs[:, 0, :, :, :] = 1.0  # all background probability
        preds = torch.zeros(1, 1, 4, 4, 4, dtype=torch.long)  # all background

        sdc = compute_sdc(probs, preds)
        assert not math.isnan(sdc), "SDC should not return NaN"
        assert isinstance(sdc, float)

    def test_sdc_batch_computation(self) -> None:
        """SDC should handle batch dimension correctly, returning a single
        scalar averaged over the batch."""
        from minivess.pipeline.validation_metrics import compute_sdc

        batch_size = 4
        probs = torch.zeros(batch_size, 2, 4, 4, 4)
        probs[:, 1, :, :, :] = 0.8  # high foreground confidence
        probs[:, 0, :, :, :] = 0.2
        preds = torch.ones(batch_size, 1, 4, 4, 4, dtype=torch.long)

        sdc = compute_sdc(probs, preds)
        assert isinstance(sdc, float)
        assert 0.0 < sdc <= 1.0

    def test_sdc_registered_in_metric_registry(self) -> None:
        """SDC must be registered in configs/metric_registry.yaml."""
        from pathlib import Path

        import yaml

        registry_path = Path("configs/metric_registry.yaml")
        assert registry_path.exists(), "Metric registry not found"

        with registry_path.open(encoding="utf-8") as f:
            registry = yaml.safe_load(f)

        # Registry is a flat list under "metrics" key
        all_metrics = registry.get("metrics", [])
        metric_names = [
            m.get("name", "") if isinstance(m, dict) else str(m) for m in all_metrics
        ]
        assert any("sdc" in name.lower() for name in metric_names), (
            f"SDC not found in metric registry. Available: {metric_names[:10]}"
        )

    def test_sdc_half_confidence(self) -> None:
        """When softmax gives 0.5 probability to foreground and prediction
        is foreground, SDC should be approximately 2*0.5*1/(0.5+1) = 2/3."""
        from minivess.pipeline.validation_metrics import compute_sdc

        probs = torch.zeros(1, 2, 4, 4, 4)
        probs[:, 1, :, :, :] = 0.5  # 50% foreground confidence
        probs[:, 0, :, :, :] = 0.5
        preds = torch.ones(1, 1, 4, 4, 4, dtype=torch.long)  # all foreground

        sdc = compute_sdc(probs, preds)
        # SDC = 2 * sum(0.5 * 1) / sum(0.5 + 1) = 2 * 0.5 / 1.5 = 2/3
        expected = 2.0 / 3.0
        assert sdc == pytest.approx(expected, abs=1e-5)

    def test_sdc_numpy_input(self) -> None:
        """compute_sdc should also accept numpy arrays."""
        from minivess.pipeline.validation_metrics import compute_sdc

        probs = np.zeros((1, 2, 4, 4, 4), dtype=np.float32)
        probs[:, 1, :, :, :] = 1.0
        preds = np.ones((1, 1, 4, 4, 4), dtype=np.int64)

        sdc = compute_sdc(probs, preds)
        assert sdc == pytest.approx(1.0, abs=1e-6)
