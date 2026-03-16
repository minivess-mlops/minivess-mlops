"""Tests for dual-mode champion evaluation.

TDD RED phase for Task T-C2 (Issue #766).
Supervised (Dice/clDice with masks) + unsupervised (uncertainty only).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest


@pytest.fixture
def mock_predictions() -> list[np.ndarray]:
    """Simulated segmentation predictions."""
    rng = np.random.default_rng(42)
    return [rng.integers(0, 2, size=(32, 32, 32), dtype=np.uint8) for _ in range(4)]


@pytest.fixture
def mock_masks() -> list[np.ndarray]:
    """Simulated ground truth masks."""
    rng = np.random.default_rng(99)
    return [rng.integers(0, 2, size=(32, 32, 32), dtype=np.uint8) for _ in range(4)]


@pytest.fixture
def mock_uncertainty_maps() -> list[np.ndarray]:
    """Simulated per-voxel uncertainty maps (MC Dropout variance)."""
    rng = np.random.default_rng(7)
    return [rng.random((32, 32, 32), dtype=np.float32) * 0.5 for _ in range(4)]


class TestEvaluationResult:
    """Test the EvaluationResult dataclass."""

    def test_evaluation_result_is_dataclass(self) -> None:
        from minivess.serving.champion_evaluator import EvaluationResult

        assert dataclasses.is_dataclass(EvaluationResult)

    def test_evaluation_result_supervised_fields(self) -> None:
        from minivess.serving.champion_evaluator import EvaluationResult

        result = EvaluationResult(
            mode="supervised",
            batch_id="drift-batch-1",
            n_volumes=4,
            dice_scores=[0.85, 0.82, 0.79, 0.88],
            mean_dice=0.835,
        )
        assert result.mode == "supervised"
        assert result.mean_dice == 0.835
        assert len(result.dice_scores) == 4

    def test_evaluation_result_unsupervised_fields(self) -> None:
        from minivess.serving.champion_evaluator import EvaluationResult

        result = EvaluationResult(
            mode="unsupervised",
            batch_id="drift-batch-2",
            n_volumes=4,
            mean_uncertainty=0.15,
        )
        assert result.mode == "unsupervised"
        assert result.mean_uncertainty == 0.15
        assert result.dice_scores is None


class TestChampionEvaluator:
    """Test the ChampionEvaluator class."""

    def test_evaluator_instantiation(self) -> None:
        from minivess.serving.champion_evaluator import ChampionEvaluator

        evaluator = ChampionEvaluator(mode="both")
        assert evaluator is not None

    def test_compute_dice(self) -> None:
        from minivess.serving.champion_evaluator import compute_dice

        pred = np.ones((10, 10, 10), dtype=np.uint8)
        mask = np.ones((10, 10, 10), dtype=np.uint8)
        dice = compute_dice(pred, mask)
        assert abs(dice - 1.0) < 1e-6

    def test_compute_dice_no_overlap(self) -> None:
        from minivess.serving.champion_evaluator import compute_dice

        pred = np.ones((10, 10, 10), dtype=np.uint8)
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        dice = compute_dice(pred, mask)
        assert dice == 0.0

    def test_supervised_evaluation(
        self,
        mock_predictions: list[np.ndarray],
        mock_masks: list[np.ndarray],
    ) -> None:
        from minivess.serving.champion_evaluator import ChampionEvaluator

        evaluator = ChampionEvaluator(mode="supervised")
        result = evaluator.evaluate(
            predictions=mock_predictions,
            masks=mock_masks,
            batch_id="drift-batch-1",
        )
        assert result.mode == "supervised"
        assert result.dice_scores is not None
        assert len(result.dice_scores) == 4
        assert result.mean_dice is not None

    def test_unsupervised_evaluation(
        self,
        mock_predictions: list[np.ndarray],
        mock_uncertainty_maps: list[np.ndarray],
    ) -> None:
        from minivess.serving.champion_evaluator import ChampionEvaluator

        evaluator = ChampionEvaluator(mode="unsupervised")
        result = evaluator.evaluate(
            predictions=mock_predictions,
            uncertainty_maps=mock_uncertainty_maps,
            batch_id="drift-batch-2",
        )
        assert result.mode == "unsupervised"
        assert result.mean_uncertainty is not None
        assert result.dice_scores is None

    def test_both_modes_evaluation(
        self,
        mock_predictions: list[np.ndarray],
        mock_masks: list[np.ndarray],
        mock_uncertainty_maps: list[np.ndarray],
    ) -> None:
        from minivess.serving.champion_evaluator import ChampionEvaluator

        evaluator = ChampionEvaluator(mode="both")
        result = evaluator.evaluate(
            predictions=mock_predictions,
            masks=mock_masks,
            uncertainty_maps=mock_uncertainty_maps,
            batch_id="drift-batch-3",
        )
        assert result.mode == "both"
        assert result.dice_scores is not None
        assert result.mean_uncertainty is not None

    def test_invalid_mode_raises(self) -> None:
        from minivess.serving.champion_evaluator import ChampionEvaluator

        with pytest.raises(ValueError, match="invalid"):
            ChampionEvaluator(mode="invalid")
