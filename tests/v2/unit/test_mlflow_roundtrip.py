"""Tests for MLflow pyfunc round-trip and ExperimentTracker integration.

Phase 4 of MLflow serving integration (#81, #84): Verifies that models
can be logged, loaded, and used for prediction end-to-end, and that
ExperimentTracker has pyfunc model logging support.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockNet(nn.Module):
    """Minimal network for checkpoint creation."""

    def __init__(self, fg_prob: float = 0.8) -> None:
        super().__init__()
        self._dummy = nn.Parameter(torch.tensor(fg_prob))

    def forward(self, x: Tensor) -> Tensor:
        b, _c, d, h, w = x.shape
        fg = torch.sigmoid(self._dummy).expand(b, 1, d, h, w)
        bg = 1.0 - fg
        return torch.cat([bg, fg], dim=1)


def _make_checkpoint(tmp_path: Path, name: str = "ckpt.pth") -> Path:
    """Save a mock checkpoint."""
    net = _MockNet()
    ckpt_path = tmp_path / name
    torch.save({"model_state_dict": net.state_dict()}, ckpt_path)
    return ckpt_path


def _make_model_config(tmp_path: Path) -> Path:
    """Save a model config JSON."""
    config_path = tmp_path / "model_config.json"
    config_path.write_text(
        json.dumps({"family": "test", "out_channels": 2}),
        encoding="utf-8",
    )
    return config_path


def _make_model_config_dict() -> dict:
    """Return a minimal model config dict."""
    return {"family": "test", "out_channels": 2}


def _make_ensemble_manifest(
    tmp_path: Path, member_paths: list[Path]
) -> Path:
    """Save an ensemble manifest JSON."""
    manifest_path = tmp_path / "ensemble_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "test_ensemble",
                "strategy": "all_loss_single_best",
                "n_members": len(member_paths),
                "members": [
                    {
                        "checkpoint_path": str(p),
                        "run_id": f"run_{i}",
                        "loss_type": "dice_ce",
                        "fold_id": i,
                        "metric_name": "val_dice",
                    }
                    for i, p in enumerate(member_paths)
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


# ---------------------------------------------------------------------------
# Tests: Pyfunc round-trip (log -> load -> predict)
# ---------------------------------------------------------------------------


class TestPyfuncRoundTrip:
    """Tests for single model pyfunc log/load/predict round-trip."""

    def test_log_and_load_single_model(self, tmp_path: Path) -> None:
        """A single model can be logged and loaded via local file backend."""
        import mlflow
        import mlflow.pyfunc

        from minivess.serving.mlflow_wrapper import (
            MiniVessSegModel,
            get_model_signature,
        )

        ckpt_path = _make_checkpoint(tmp_path)
        config_path = _make_model_config(tmp_path)

        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        mlflow.set_experiment("test_roundtrip")

        with mlflow.start_run():
            model_info = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=MiniVessSegModel(),
                artifacts={
                    "checkpoint": str(ckpt_path),
                    "model_config": str(config_path),
                },
                signature=get_model_signature(),
            )

            # Load back
            loaded = mlflow.pyfunc.load_model(model_info.model_uri)
            assert loaded is not None

    def test_loaded_model_predict_shape(self, tmp_path: Path) -> None:
        """Loaded pyfunc model returns correct output shape."""
        import mlflow
        import mlflow.pyfunc

        from minivess.serving.mlflow_wrapper import (
            MiniVessSegModel,
            get_model_signature,
        )

        ckpt_path = _make_checkpoint(tmp_path)
        config_path = _make_model_config(tmp_path)

        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        mlflow.set_experiment("test_roundtrip")

        with mlflow.start_run():
            model_info = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=MiniVessSegModel(),
                artifacts={
                    "checkpoint": str(ckpt_path),
                    "model_config": str(config_path),
                },
                signature=get_model_signature(),
            )

            loaded = mlflow.pyfunc.load_model(model_info.model_uri)
            input_data = np.random.default_rng(42).random(
                (1, 1, 8, 8, 4), dtype=np.float32
            )
            result = loaded.predict(input_data)

            assert isinstance(result, np.ndarray)
            assert result.shape == (1, 2, 8, 8, 4)

    def test_loaded_model_predict_probabilities(self, tmp_path: Path) -> None:
        """Loaded model output sums to ~1.0 along class dimension."""
        import mlflow
        import mlflow.pyfunc

        from minivess.serving.mlflow_wrapper import (
            MiniVessSegModel,
            get_model_signature,
        )

        ckpt_path = _make_checkpoint(tmp_path)
        config_path = _make_model_config(tmp_path)

        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        mlflow.set_experiment("test_roundtrip")

        with mlflow.start_run():
            model_info = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=MiniVessSegModel(),
                artifacts={
                    "checkpoint": str(ckpt_path),
                    "model_config": str(config_path),
                },
                signature=get_model_signature(),
            )

            loaded = mlflow.pyfunc.load_model(model_info.model_uri)
            input_data = np.random.default_rng(42).random(
                (1, 1, 8, 8, 4), dtype=np.float32
            )
            result = loaded.predict(input_data)

            # Sum across class dim should be ~1.0
            np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-5)


class TestEnsemblePyfuncRoundTrip:
    """Tests for ensemble model pyfunc log/load/predict round-trip."""

    def test_log_and_load_ensemble_model(self, tmp_path: Path) -> None:
        """An ensemble model can be logged and loaded."""
        import mlflow
        import mlflow.pyfunc

        from minivess.serving.mlflow_wrapper import (
            MiniVessEnsembleModel,
            get_model_signature,
        )

        # Create 2 member checkpoints
        member_paths = []
        for i in range(2):
            ckpt = _make_checkpoint(tmp_path, name=f"member_{i}.pth")
            member_paths.append(ckpt)

        config_path = _make_model_config(tmp_path)
        manifest_path = _make_ensemble_manifest(tmp_path, member_paths)

        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        mlflow.set_experiment("test_roundtrip")

        with mlflow.start_run():
            model_info = mlflow.pyfunc.log_model(
                artifact_path="ensemble_model",
                python_model=MiniVessEnsembleModel(),
                artifacts={
                    "ensemble_manifest": str(manifest_path),
                    "model_config": str(config_path),
                },
                signature=get_model_signature(),
            )

            loaded = mlflow.pyfunc.load_model(model_info.model_uri)
            assert loaded is not None

    def test_ensemble_predict_mean_probabilities(self, tmp_path: Path) -> None:
        """Loaded ensemble returns valid mean probabilities."""
        import mlflow
        import mlflow.pyfunc

        from minivess.serving.mlflow_wrapper import (
            MiniVessEnsembleModel,
            get_model_signature,
        )

        member_paths = []
        for i in range(2):
            ckpt = _make_checkpoint(tmp_path, name=f"member_{i}.pth")
            member_paths.append(ckpt)

        config_path = _make_model_config(tmp_path)
        manifest_path = _make_ensemble_manifest(tmp_path, member_paths)

        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        mlflow.set_experiment("test_roundtrip")

        with mlflow.start_run():
            model_info = mlflow.pyfunc.log_model(
                artifact_path="ensemble_model",
                python_model=MiniVessEnsembleModel(),
                artifacts={
                    "ensemble_manifest": str(manifest_path),
                    "model_config": str(config_path),
                },
                signature=get_model_signature(),
            )

            loaded = mlflow.pyfunc.load_model(model_info.model_uri)
            input_data = np.random.default_rng(42).random(
                (1, 1, 8, 8, 4), dtype=np.float32
            )
            result = loaded.predict(input_data)

            assert result.shape == (1, 2, 8, 8, 4)
            np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: ExperimentTracker.log_pyfunc_model
# ---------------------------------------------------------------------------


class TestExperimentTrackerLogPyfunc:
    """Tests for the log_pyfunc_model method on ExperimentTracker."""

    @patch("minivess.observability.tracking.mlflow")
    def test_log_pyfunc_model_within_run(
        self, mock_mlflow: MagicMock, tmp_path: Path
    ) -> None:
        """log_pyfunc_model delegates to model_logger.log_single_model."""
        # Create minimal config
        from minivess.config.models import (
            DataConfig,
            ExperimentConfig,
            ModelConfig,
            ModelFamily,
            TrainingConfig,
        )
        from minivess.observability.tracking import ExperimentTracker

        model_cfg = ModelConfig(
            family=ModelFamily.MONAI_DYNUNET,
            name="test",
            in_channels=1,
            out_channels=2,
        )
        train_cfg = TrainingConfig(max_epochs=1)
        data_cfg = DataConfig(dataset_name="test")
        exp_cfg = ExperimentConfig(
            experiment_name="test_exp",
            data=data_cfg,
            model=model_cfg,
            training=train_cfg,
        )

        tracker = ExperimentTracker(exp_cfg, tracking_uri=str(tmp_path))

        # Simulate being in a run
        tracker._run_id = "fake_run_id"

        ckpt_path = _make_checkpoint(tmp_path)
        config_dict = _make_model_config_dict()

        with patch(
            "minivess.observability.tracking.log_single_model"
        ) as mock_log:
            mock_log.return_value = MagicMock()
            tracker.log_pyfunc_model(ckpt_path, config_dict)

        mock_log.assert_called_once()

    def test_log_pyfunc_model_outside_run_raises(self, tmp_path: Path) -> None:
        """log_pyfunc_model raises RuntimeError outside a run context."""
        from minivess.config.models import (
            DataConfig,
            ExperimentConfig,
            ModelConfig,
            ModelFamily,
            TrainingConfig,
        )
        from minivess.observability.tracking import ExperimentTracker

        model_cfg = ModelConfig(
            family=ModelFamily.MONAI_DYNUNET,
            name="test",
            in_channels=1,
            out_channels=2,
        )
        train_cfg = TrainingConfig(max_epochs=1)
        data_cfg = DataConfig(dataset_name="test")
        exp_cfg = ExperimentConfig(
            experiment_name="test_exp",
            data=data_cfg,
            model=model_cfg,
            training=train_cfg,
        )

        tracker = ExperimentTracker(exp_cfg, tracking_uri=str(tmp_path))

        ckpt_path = _make_checkpoint(tmp_path)
        config_dict = _make_model_config_dict()

        import pytest

        with pytest.raises(RuntimeError, match="outside of a run"):
            tracker.log_pyfunc_model(ckpt_path, config_dict)
