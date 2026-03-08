"""Integration tests for auto-resume discovery using mocked MLflow.

Tests verify that find_completed_config() and load_fold_result_from_mlflow()
behave correctly without a running MLflow server — all MLflow calls are mocked.

CLAUDE.md Rule #16: No regex. Uses unittest.mock.patch for isolation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# T-04.2: find_completed_config tests
# ---------------------------------------------------------------------------


def test_find_completed_config_returns_none_when_no_match() -> None:
    """When MLflow search_runs returns empty list, find_completed_config returns None."""
    from minivess.pipeline.resume_discovery import find_completed_config

    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "exp-001"

    with (
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.get_experiment_by_name", return_value=mock_experiment),
        patch("mlflow.search_runs", return_value=[]),
    ):
        result = find_completed_config(
            tracking_uri="mlruns",
            experiment_name="test_experiment",
            config_fingerprint="abcd1234abcd1234",
        )

    assert result is None


def test_find_completed_config_returns_run_id_when_match() -> None:
    """When MLflow search_runs returns a run, find_completed_config returns its run_id."""
    from minivess.pipeline.resume_discovery import find_completed_config

    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "exp-001"

    mock_run = MagicMock()
    mock_run.info.run_id = "run-abc123"

    with (
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.get_experiment_by_name", return_value=mock_experiment),
        patch("mlflow.search_runs", return_value=[mock_run]),
    ):
        result = find_completed_config(
            tracking_uri="mlruns",
            experiment_name="test_experiment",
            config_fingerprint="abcd1234abcd1234",
        )

    assert result == "run-abc123"


def test_find_completed_config_returns_none_when_experiment_not_found() -> None:
    """When the experiment does not exist, find_completed_config returns None."""
    from minivess.pipeline.resume_discovery import find_completed_config

    with (
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.get_experiment_by_name", return_value=None),
    ):
        result = find_completed_config(
            tracking_uri="mlruns",
            experiment_name="nonexistent_experiment",
            config_fingerprint="abcd1234abcd1234",
        )

    assert result is None


def test_find_completed_config_returns_none_on_mlflow_error() -> None:
    """When MLflow raises an exception, find_completed_config returns None (graceful)."""
    from minivess.pipeline.resume_discovery import find_completed_config

    with (
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.get_experiment_by_name", side_effect=RuntimeError("MLflow down")),
    ):
        result = find_completed_config(
            tracking_uri="mlruns",
            experiment_name="test_experiment",
            config_fingerprint="abcd1234abcd1234",
        )

    assert result is None


# ---------------------------------------------------------------------------
# T-04.2: compute_config_fingerprint stability
# ---------------------------------------------------------------------------


def test_compute_config_fingerprint_stable() -> None:
    """Same inputs produce same fingerprint across multiple calls."""
    from minivess.pipeline.resume_discovery import compute_config_fingerprint

    kwargs = {
        "loss_name": "cbdice_cldice",
        "model_family": "dynunet",
        "fold_id": 2,
        "max_epochs": 50,
        "batch_size": 1,
        "patch_size": (64, 64, 16),
    }

    fp_a = compute_config_fingerprint(**kwargs)
    fp_b = compute_config_fingerprint(**kwargs)
    fp_c = compute_config_fingerprint(**kwargs)

    assert fp_a == fp_b == fp_c, (
        "Fingerprint must be stable across calls with identical inputs"
    )


def test_compute_config_fingerprint_patch_size_affects_hash() -> None:
    """Different patch_size values produce different fingerprints."""
    from minivess.pipeline.resume_discovery import compute_config_fingerprint

    fp_small = compute_config_fingerprint(
        loss_name="dice_ce",
        model_family="dynunet",
        fold_id=0,
        max_epochs=100,
        batch_size=2,
        patch_size=(64, 64, 16),
    )
    fp_large = compute_config_fingerprint(
        loss_name="dice_ce",
        model_family="dynunet",
        fold_id=0,
        max_epochs=100,
        batch_size=2,
        patch_size=(128, 128, 32),
    )
    assert fp_small != fp_large


def test_compute_config_fingerprint_no_patch_size() -> None:
    """compute_config_fingerprint works when patch_size is None."""
    from minivess.pipeline.resume_discovery import compute_config_fingerprint

    fp = compute_config_fingerprint(
        loss_name="dice_ce",
        model_family="dynunet",
        fold_id=0,
        max_epochs=100,
        batch_size=2,
        patch_size=None,
    )
    assert isinstance(fp, str)
    assert len(fp) == 16


# ---------------------------------------------------------------------------
# T-04.2: load_fold_result_from_mlflow tests
# ---------------------------------------------------------------------------


def test_load_fold_result_from_mlflow() -> None:
    """Mock MLflow get_run → load_fold_result returns correct metrics."""
    from minivess.pipeline.resume_discovery import load_fold_result_from_mlflow

    mock_run = MagicMock()
    mock_run.data.metrics = {"best_val_loss": 0.321, "val_dice": 0.87}
    mock_run.data.params = {"loss_name": "dice_ce", "fold_id": "0"}

    with (
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.get_run", return_value=mock_run),
    ):
        result = load_fold_result_from_mlflow(
            tracking_uri="mlruns",
            run_id="run-test-001",
        )

    assert result["run_id"] == "run-test-001"
    assert result["status"] == "resumed"
    assert pytest.approx(result["best_val_loss"], abs=1e-6) == 0.321
    assert "best_val_loss" in result["metrics"]
    assert "loss_name" in result["params"]


def test_load_fold_result_from_mlflow_missing_metric() -> None:
    """When best_val_loss is missing, it defaults to inf."""
    from minivess.pipeline.resume_discovery import load_fold_result_from_mlflow

    mock_run = MagicMock()
    mock_run.data.metrics = {}
    mock_run.data.params = {}

    with (
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.get_run", return_value=mock_run),
    ):
        result = load_fold_result_from_mlflow(
            tracking_uri="mlruns",
            run_id="run-no-metrics",
        )

    assert result["best_val_loss"] == float("inf")


def test_load_fold_result_from_mlflow_error_returns_fallback() -> None:
    """When MLflow raises, load_fold_result returns a resume_failed status dict."""
    from minivess.pipeline.resume_discovery import load_fold_result_from_mlflow

    with (
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.get_run", side_effect=RuntimeError("MLflow unavailable")),
    ):
        result = load_fold_result_from_mlflow(
            tracking_uri="mlruns",
            run_id="run-broken",
        )

    assert result["status"] == "resume_failed"
    assert result["best_val_loss"] == float("inf")
    assert result["run_id"] == "run-broken"
