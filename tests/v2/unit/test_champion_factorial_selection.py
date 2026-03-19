"""Tests for champion model selection from factorial experiment results.

PR-D T1 (Issue #825): Select champion model by compound metric from
factorial evaluation runs. Tags champion with all factorial factors.

TDD Phase: RED — tests written before implementation.
"""

from __future__ import annotations

from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Synthetic factorial evaluation data
# ---------------------------------------------------------------------------


def _make_factorial_runs() -> list[dict[str, Any]]:
    """Create synthetic factorial evaluation results.

    Returns 6 runs: 2 models x 3 losses, each with metrics.
    The best run has model=dynunet, loss=cbdice_cldice with
    clDice=0.85 and MASD=0.5.
    """
    runs = [
        {
            "run_id": "run_dynunet_cbdice",
            "experiment_id": "1",
            "model": "dynunet",
            "loss": "cbdice_cldice",
            "aux_calib": False,
            "fold_strategy": "cv_average",
            "ensemble": "none",
            "cldice": 0.85,
            "masd": 0.5,
            "dsc": 0.88,
        },
        {
            "run_id": "run_dynunet_dice_ce",
            "experiment_id": "1",
            "model": "dynunet",
            "loss": "dice_ce",
            "aux_calib": False,
            "fold_strategy": "cv_average",
            "ensemble": "none",
            "cldice": 0.78,
            "masd": 1.2,
            "dsc": 0.82,
        },
        {
            "run_id": "run_dynunet_focal",
            "experiment_id": "1",
            "model": "dynunet",
            "loss": "focal",
            "aux_calib": False,
            "fold_strategy": "cv_average",
            "ensemble": "none",
            "cldice": 0.72,
            "masd": 1.8,
            "dsc": 0.79,
        },
        {
            "run_id": "run_vesselfm_cbdice",
            "experiment_id": "1",
            "model": "vesselfm",
            "loss": "cbdice_cldice",
            "aux_calib": True,
            "fold_strategy": "cv_average",
            "ensemble": "none",
            "cldice": 0.83,
            "masd": 0.6,
            "dsc": 0.86,
        },
        {
            "run_id": "run_vesselfm_dice_ce",
            "experiment_id": "1",
            "model": "vesselfm",
            "loss": "dice_ce",
            "aux_calib": True,
            "fold_strategy": "cv_average",
            "ensemble": "none",
            "cldice": 0.76,
            "masd": 1.3,
            "dsc": 0.81,
        },
        {
            "run_id": "run_vesselfm_focal",
            "experiment_id": "1",
            "model": "vesselfm",
            "loss": "focal",
            "aux_calib": False,
            "fold_strategy": "cv_average",
            "ensemble": "none",
            "cldice": 0.70,
            "masd": 2.0,
            "dsc": 0.77,
        },
    ]
    return runs


class TestChampionSelectionFromFactorial:
    """Select the best model from factorial evaluation results."""

    def test_champion_selection_from_factorial(self) -> None:
        """Best compound metric run is selected as champion."""
        from minivess.serving.champion_factorial_selection import (
            select_factorial_champion,
        )

        runs = _make_factorial_runs()
        champion = select_factorial_champion(runs)

        assert champion is not None
        # dynunet_cbdice has highest clDice (0.85) and lowest MASD (0.5)
        assert champion["run_id"] == "run_dynunet_cbdice"

    def test_champion_selection_compound_metric(self) -> None:
        """Compound metric = 0.5*clDice + 0.5*normalize(MASD)."""
        from minivess.serving.champion_factorial_selection import (
            compute_compound_metric,
        )

        # Perfect MASD (0.0) → normalized to 1.0
        # clDice = 0.85, MASD normalized ~ 1.0 for best
        score = compute_compound_metric(cldice=0.85, masd=0.5, masd_max=2.0)
        # 0.5*0.85 + 0.5*(1 - 0.5/2.0) = 0.425 + 0.375 = 0.8
        assert score == pytest.approx(0.8, abs=0.01)

        # Worst run: clDice=0.70, MASD=2.0 → normalized to 0.0
        worst = compute_compound_metric(cldice=0.70, masd=2.0, masd_max=2.0)
        # 0.5*0.70 + 0.5*(1 - 2.0/2.0) = 0.35 + 0.0 = 0.35
        assert worst == pytest.approx(0.35, abs=0.01)

    def test_champion_selection_cv_average_preferred(self) -> None:
        """CV-average fold strategy is preferred over single fold."""
        from minivess.serving.champion_factorial_selection import (
            select_factorial_champion,
        )

        runs = _make_factorial_runs()
        # Add a single-fold run with slightly better metrics
        runs.append(
            {
                "run_id": "run_single_fold_great",
                "experiment_id": "1",
                "model": "dynunet",
                "loss": "cbdice_cldice",
                "aux_calib": False,
                "fold_strategy": "single_fold",
                "ensemble": "none",
                "cldice": 0.87,  # higher than cv_average
                "masd": 0.4,
                "dsc": 0.90,
            }
        )
        champion = select_factorial_champion(runs)
        # cv_average is preferred even with slightly lower metric
        assert champion["fold_strategy"] == "cv_average"

    def test_champion_metadata_tags(self) -> None:
        """Champion has all required factorial factor tags."""
        from minivess.serving.champion_factorial_selection import (
            build_champion_tags,
            select_factorial_champion,
        )

        runs = _make_factorial_runs()
        champion = select_factorial_champion(runs)
        tags = build_champion_tags(champion)

        required_tags = {
            "champion/model",
            "champion/loss",
            "champion/aux_calib",
            "champion/fold_strategy",
            "champion/ensemble",
            "champion/cldice",
            "champion/masd",
            "champion/dsc",
            "champion/compound_metric",
        }
        assert required_tags.issubset(set(tags.keys()))
        assert tags["champion/model"] == "dynunet"
        assert tags["champion/loss"] == "cbdice_cldice"

    def test_champion_registry_promotion(self) -> None:
        """Champion can be promoted to MLflow Model Registry."""
        from minivess.serving.champion_factorial_selection import (
            prepare_registry_promotion,
            select_factorial_champion,
        )

        runs = _make_factorial_runs()
        champion = select_factorial_champion(runs)
        promotion = prepare_registry_promotion(champion)

        assert promotion["model_name"] == "minivess-champion"
        assert promotion["run_id"] == champion["run_id"]
        assert "tags" in promotion
        assert promotion["tags"]["champion/model"] == "dynunet"

    def test_empty_runs_returns_none(self) -> None:
        """Empty runs list returns None."""
        from minivess.serving.champion_factorial_selection import (
            select_factorial_champion,
        )

        assert select_factorial_champion([]) is None

    def test_all_same_metric_selects_first_cv_average(self) -> None:
        """When metrics are identical, selects first cv_average run."""
        from minivess.serving.champion_factorial_selection import (
            select_factorial_champion,
        )

        runs = [
            {
                "run_id": f"run_{i}",
                "experiment_id": "1",
                "model": "dynunet",
                "loss": "dice_ce",
                "aux_calib": False,
                "fold_strategy": "cv_average" if i == 1 else "single_fold",
                "ensemble": "none",
                "cldice": 0.80,
                "masd": 1.0,
                "dsc": 0.85,
            }
            for i in range(3)
        ]
        champion = select_factorial_champion(runs)
        assert champion["fold_strategy"] == "cv_average"
