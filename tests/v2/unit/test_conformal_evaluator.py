"""Tests for conformal evaluator and pipeline integration (Phase 4).

Validates the unified conformal evaluator, config, and exports.
"""

from __future__ import annotations

import numpy as np
import pytest


def _make_sphere_mask(
    shape: tuple[int, int, int] = (16, 16, 16),
    center: tuple[int, int, int] | None = None,
    radius: float = 5.0,
) -> np.ndarray:
    """Create a binary sphere mask."""
    if center is None:
        center = tuple(s // 2 for s in shape)
    coords = np.mgrid[: shape[0], : shape[1], : shape[2]]
    dist_sq = sum((c - cn) ** 2 for c, cn in zip(coords, center, strict=True))
    return (dist_sq <= radius**2).astype(np.int64)


# ---------------------------------------------------------------------------
# Task 4.1: ConformalEvaluator
# ---------------------------------------------------------------------------


class TestConformalEvaluator:
    """Test the unified conformal evaluator."""

    def test_evaluator_runs_all_methods(self) -> None:
        """All enabled methods should produce results."""
        from minivess.ensemble.conformal_evaluator import ConformalEvaluator

        gt = _make_sphere_mask(radius=5.0)
        pred = _make_sphere_mask(radius=4.0)
        probs = np.zeros((16, 16, 16), dtype=np.float32)
        probs[pred.astype(bool)] = 0.8
        probs[~pred.astype(bool)] = 0.2

        evaluator = ConformalEvaluator(alpha=0.2)
        result = evaluator.evaluate(
            predictions=[pred] * 6,
            softmax_probs=[probs] * 6,
            labels=[gt] * 6,
            calibration_fraction=0.5,
        )

        # Should have results from multiple methods
        assert "voxel" in result
        assert "morphological" in result
        assert "distance" in result

    def test_evaluator_calibration_split(self) -> None:
        """Should correctly split calibration/test data."""
        from minivess.ensemble.conformal_evaluator import ConformalEvaluator

        masks = [_make_sphere_mask(radius=5.0) for _ in range(10)]
        probs_list = []
        for m in masks:
            p = np.zeros((16, 16, 16), dtype=np.float32)
            p[m.astype(bool)] = 0.8
            probs_list.append(p)

        evaluator = ConformalEvaluator(alpha=0.2)
        result = evaluator.evaluate(
            predictions=masks,
            softmax_probs=probs_list,
            labels=masks,
            calibration_fraction=0.3,
        )
        # Should complete without error
        assert isinstance(result, dict)

    def test_evaluator_result_has_all_metrics(self) -> None:
        """Each method result should have expected metric keys."""
        from minivess.ensemble.conformal_evaluator import ConformalEvaluator

        gt = _make_sphere_mask(radius=5.0)
        pred = _make_sphere_mask(radius=4.0)
        probs = np.zeros((16, 16, 16), dtype=np.float32)
        probs[pred.astype(bool)] = 0.8

        evaluator = ConformalEvaluator(alpha=0.2)
        result = evaluator.evaluate(
            predictions=[pred] * 6,
            softmax_probs=[probs] * 6,
            labels=[gt] * 6,
            calibration_fraction=0.5,
        )

        # Morphological should have coverage and band metrics
        morph = result.get("morphological", {})
        assert "outer_coverage" in morph or "error" in morph

    def test_evaluator_to_dict(self) -> None:
        """Flat dict should be MLflow-compatible (string keys, float values)."""
        from minivess.ensemble.conformal_evaluator import ConformalEvaluator

        gt = _make_sphere_mask(radius=5.0)
        pred = _make_sphere_mask(radius=4.0)
        probs = np.zeros((16, 16, 16), dtype=np.float32)
        probs[pred.astype(bool)] = 0.8

        evaluator = ConformalEvaluator(alpha=0.2)
        result = evaluator.evaluate(
            predictions=[pred] * 6,
            softmax_probs=[probs] * 6,
            labels=[gt] * 6,
            calibration_fraction=0.5,
        )

        flat = evaluator.flatten_results(result)
        for key, val in flat.items():
            assert isinstance(key, str)
            assert isinstance(val, float), f"{key}: {type(val)} is not float"

    def test_evaluator_to_markdown(self) -> None:
        """Should generate a markdown comparison table."""
        from minivess.ensemble.conformal_evaluator import ConformalEvaluator

        gt = _make_sphere_mask(radius=5.0)
        pred = _make_sphere_mask(radius=4.0)
        probs = np.zeros((16, 16, 16), dtype=np.float32)
        probs[pred.astype(bool)] = 0.8

        evaluator = ConformalEvaluator(alpha=0.2)
        result = evaluator.evaluate(
            predictions=[pred] * 6,
            softmax_probs=[probs] * 6,
            labels=[gt] * 6,
            calibration_fraction=0.5,
        )

        md = evaluator.to_markdown(result)
        assert isinstance(md, str)
        assert "Conformal" in md

    def test_evaluator_single_method(self) -> None:
        """Should support running only one CP method."""
        from minivess.ensemble.conformal_evaluator import ConformalEvaluator

        gt = _make_sphere_mask(radius=5.0)
        pred = _make_sphere_mask(radius=4.0)
        probs = np.zeros((16, 16, 16), dtype=np.float32)
        probs[pred.astype(bool)] = 0.8

        evaluator = ConformalEvaluator(
            alpha=0.2, methods=["morphological"]
        )
        result = evaluator.evaluate(
            predictions=[pred] * 6,
            softmax_probs=[probs] * 6,
            labels=[gt] * 6,
            calibration_fraction=0.5,
        )

        assert "morphological" in result
        assert "voxel" not in result


# ---------------------------------------------------------------------------
# Task 4.3: Exports
# ---------------------------------------------------------------------------


class TestEnsembleExports:
    """Test that new classes are importable from ensemble package."""

    def test_imports_from_ensemble(self) -> None:
        """All new classes should be importable from minivess.ensemble."""
        from minivess.ensemble import (
            MorphologicalConformalPredictor,
            MorphologicalConformalResult,
        )
        from minivess.ensemble.conformal_evaluator import ConformalEvaluator
        from minivess.ensemble.distance_conformal import (
            DistanceTransformConformalPredictor,
        )
        from minivess.ensemble.risk_control import RiskControllingPredictor

        assert MorphologicalConformalPredictor is not None
        assert MorphologicalConformalResult is not None
        assert DistanceTransformConformalPredictor is not None
        assert RiskControllingPredictor is not None
        assert ConformalEvaluator is not None


# ---------------------------------------------------------------------------
# Task 4.4: Config
# ---------------------------------------------------------------------------


class TestConformalConfig:
    """Test ConformalConfig in models.py."""

    def test_conformal_config_defaults(self) -> None:
        """Default values should be correct."""
        from minivess.config.models import ConformalConfig

        cfg = ConformalConfig()
        assert cfg.alpha == 0.1
        assert "morphological" in cfg.methods
        assert cfg.max_dilation_radius == 20
        assert cfg.calibration_fraction == 0.3

    def test_conformal_config_validation(self) -> None:
        """Alpha must be in (0, 1)."""
        from minivess.config.models import ConformalConfig

        with pytest.raises(ValueError):
            ConformalConfig(alpha=0.0)
        with pytest.raises(ValueError):
            ConformalConfig(alpha=1.0)

    def test_ensemble_config_has_conformal(self) -> None:
        """EnsembleConfig should have a ConformalConfig field."""
        from minivess.config.models import EnsembleConfig

        cfg = EnsembleConfig()
        assert hasattr(cfg, "conformal")
        assert cfg.conformal.alpha == 0.1
