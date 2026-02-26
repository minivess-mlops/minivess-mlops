"""Tests for MLflow model artifact logger.

Phase 1 of MLflow serving integration (#84): Tests for logging single models
and ensembles as MLflow pyfunc artifacts via mlflow.pyfunc.log_model().
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Mock helpers
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
    """Save a mock checkpoint and return its path."""
    net = _MockNet()
    ckpt_path = tmp_path / name
    torch.save({"model_state_dict": net.state_dict()}, ckpt_path)
    return ckpt_path


def _make_model_config() -> dict:
    """Return a minimal model config dict."""
    return {
        "family": "test",
        "name": "test-model",
        "in_channels": 1,
        "out_channels": 2,
    }


def _make_ensemble_spec(tmp_path: Path, n_members: int = 3):  # noqa: ANN202
    """Create an EnsembleSpec with mock members."""
    from minivess.config.evaluation_config import EnsembleStrategyName
    from minivess.ensemble.builder import EnsembleMember, EnsembleSpec

    members = []
    for i in range(n_members):
        ckpt_path = _make_checkpoint(tmp_path, name=f"member_{i}.pth")
        net = _MockNet(fg_prob=0.7 + i * 0.05)
        members.append(
            EnsembleMember(
                checkpoint_path=ckpt_path,
                run_id=f"run_{i:03d}",
                loss_type="dice_ce",
                fold_id=i,
                metric_name="val_compound_masd_cldice",
                net=net,
            )
        )

    return EnsembleSpec(
        name="test_ensemble",
        strategy=EnsembleStrategyName.ALL_LOSS_SINGLE_BEST,
        members=members,
        description=f"{n_members}-member test ensemble",
    )


# ---------------------------------------------------------------------------
# Tests: create_ensemble_manifest
# ---------------------------------------------------------------------------


class TestCreateEnsembleManifest:
    """Tests for EnsembleSpec → JSON manifest conversion."""

    def test_manifest_from_spec_basic(self, tmp_path: Path) -> None:
        """create_ensemble_manifest returns a dict with required top-level keys."""
        from minivess.serving.model_logger import create_ensemble_manifest

        spec = _make_ensemble_spec(tmp_path)
        manifest = create_ensemble_manifest(spec)

        assert "name" in manifest
        assert "strategy" in manifest
        assert "n_members" in manifest
        assert "members" in manifest

    def test_manifest_member_count(self, tmp_path: Path) -> None:
        """Manifest n_members matches actual member list length."""
        from minivess.serving.model_logger import create_ensemble_manifest

        spec = _make_ensemble_spec(tmp_path, n_members=5)
        manifest = create_ensemble_manifest(spec)

        assert manifest["n_members"] == 5
        assert len(manifest["members"]) == 5

    def test_manifest_member_fields(self, tmp_path: Path) -> None:
        """Each member entry has checkpoint_path, run_id, loss_type, fold_id, metric_name."""
        from minivess.serving.model_logger import create_ensemble_manifest

        spec = _make_ensemble_spec(tmp_path)
        manifest = create_ensemble_manifest(spec)

        for member in manifest["members"]:
            assert "checkpoint_path" in member
            assert "run_id" in member
            assert "loss_type" in member
            assert "fold_id" in member
            assert "metric_name" in member

    def test_manifest_json_serializable(self, tmp_path: Path) -> None:
        """Manifest dict is fully JSON-serializable."""
        from minivess.serving.model_logger import create_ensemble_manifest

        spec = _make_ensemble_spec(tmp_path)
        manifest = create_ensemble_manifest(spec)

        # Should not raise
        json_str = json.dumps(manifest)
        assert isinstance(json_str, str)

    def test_manifest_empty_members(self, tmp_path: Path) -> None:
        """Manifest handles an ensemble with no members."""
        from minivess.config.evaluation_config import EnsembleStrategyName
        from minivess.ensemble.builder import EnsembleSpec
        from minivess.serving.model_logger import create_ensemble_manifest

        spec = EnsembleSpec(
            name="empty",
            strategy=EnsembleStrategyName.ALL_LOSS_SINGLE_BEST,
            members=[],
            description="empty ensemble",
        )
        manifest = create_ensemble_manifest(spec)

        assert manifest["n_members"] == 0
        assert manifest["members"] == []


# ---------------------------------------------------------------------------
# Tests: log_single_model
# ---------------------------------------------------------------------------


class TestLogSingleModel:
    """Tests for logging a single model as MLflow pyfunc artifact."""

    @patch("minivess.serving.model_logger.mlflow")
    def test_calls_log_model(self, mock_mlflow: MagicMock, tmp_path: Path) -> None:
        """log_single_model calls mlflow.pyfunc.log_model."""
        from minivess.serving.model_logger import log_single_model

        ckpt_path = _make_checkpoint(tmp_path)
        config = _make_model_config()

        mock_mlflow.pyfunc.log_model.return_value = MagicMock()
        log_single_model(
            checkpoint_path=ckpt_path,
            model_config_dict=config,
        )

        mock_mlflow.pyfunc.log_model.assert_called_once()

    @patch("minivess.serving.model_logger.mlflow")
    def test_passes_signature(self, mock_mlflow: MagicMock, tmp_path: Path) -> None:
        """log_single_model passes the model signature."""
        from minivess.serving.model_logger import log_single_model

        ckpt_path = _make_checkpoint(tmp_path)
        config = _make_model_config()

        mock_mlflow.pyfunc.log_model.return_value = MagicMock()
        log_single_model(
            checkpoint_path=ckpt_path,
            model_config_dict=config,
        )

        call_kwargs = mock_mlflow.pyfunc.log_model.call_args
        assert call_kwargs.kwargs.get("signature") is not None or (
            len(call_kwargs.args) > 3 and call_kwargs.args[3] is not None
        )

    @patch("minivess.serving.model_logger.mlflow")
    def test_passes_artifacts(self, mock_mlflow: MagicMock, tmp_path: Path) -> None:
        """log_single_model passes artifacts dict with checkpoint and model_config."""
        from minivess.serving.model_logger import log_single_model

        ckpt_path = _make_checkpoint(tmp_path)
        config = _make_model_config()

        mock_mlflow.pyfunc.log_model.return_value = MagicMock()
        log_single_model(
            checkpoint_path=ckpt_path,
            model_config_dict=config,
        )

        call_kwargs = mock_mlflow.pyfunc.log_model.call_args.kwargs
        artifacts = call_kwargs.get("artifacts", {})
        assert "checkpoint" in artifacts
        assert "model_config" in artifacts

    @patch("minivess.serving.model_logger.mlflow")
    def test_creates_config_json(self, mock_mlflow: MagicMock, tmp_path: Path) -> None:
        """log_single_model writes model config as JSON file for artifact."""
        from minivess.serving.model_logger import log_single_model

        ckpt_path = _make_checkpoint(tmp_path)
        config = _make_model_config()

        mock_mlflow.pyfunc.log_model.return_value = MagicMock()
        log_single_model(
            checkpoint_path=ckpt_path,
            model_config_dict=config,
        )

        call_kwargs = mock_mlflow.pyfunc.log_model.call_args.kwargs
        config_path = Path(call_kwargs["artifacts"]["model_config"])
        assert config_path.exists()
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        assert loaded["family"] == "test"

    @patch("minivess.serving.model_logger.mlflow")
    def test_uses_correct_python_model_class(
        self, mock_mlflow: MagicMock, tmp_path: Path
    ) -> None:
        """log_single_model uses MiniVessSegModel as the python_model."""
        from minivess.serving.mlflow_wrapper import MiniVessSegModel
        from minivess.serving.model_logger import log_single_model

        ckpt_path = _make_checkpoint(tmp_path)
        config = _make_model_config()

        mock_mlflow.pyfunc.log_model.return_value = MagicMock()
        log_single_model(
            checkpoint_path=ckpt_path,
            model_config_dict=config,
        )

        call_kwargs = mock_mlflow.pyfunc.log_model.call_args.kwargs
        python_model = call_kwargs.get("python_model")
        assert isinstance(python_model, MiniVessSegModel)

    @patch("minivess.serving.model_logger.mlflow")
    def test_custom_artifact_path(self, mock_mlflow: MagicMock, tmp_path: Path) -> None:
        """log_single_model respects custom artifact_path."""
        from minivess.serving.model_logger import log_single_model

        ckpt_path = _make_checkpoint(tmp_path)
        config = _make_model_config()

        mock_mlflow.pyfunc.log_model.return_value = MagicMock()
        log_single_model(
            checkpoint_path=ckpt_path,
            model_config_dict=config,
            artifact_path="custom_model",
        )

        call_kwargs = mock_mlflow.pyfunc.log_model.call_args.kwargs
        assert call_kwargs.get("artifact_path") == "custom_model"


# ---------------------------------------------------------------------------
# Tests: log_ensemble_model
# ---------------------------------------------------------------------------


class TestLogEnsembleModel:
    """Tests for logging an ensemble model as MLflow pyfunc artifact."""

    @patch("minivess.serving.model_logger.mlflow")
    def test_calls_log_model(self, mock_mlflow: MagicMock, tmp_path: Path) -> None:
        """log_ensemble_model calls mlflow.pyfunc.log_model."""
        from minivess.serving.model_logger import log_ensemble_model

        spec = _make_ensemble_spec(tmp_path)
        config = _make_model_config()

        mock_mlflow.pyfunc.log_model.return_value = MagicMock()
        log_ensemble_model(
            ensemble_spec=spec,
            model_config_dict=config,
        )

        mock_mlflow.pyfunc.log_model.assert_called_once()

    @patch("minivess.serving.model_logger.mlflow")
    def test_creates_manifest(self, mock_mlflow: MagicMock, tmp_path: Path) -> None:
        """log_ensemble_model creates a manifest JSON artifact."""
        from minivess.serving.model_logger import log_ensemble_model

        spec = _make_ensemble_spec(tmp_path)
        config = _make_model_config()

        mock_mlflow.pyfunc.log_model.return_value = MagicMock()
        log_ensemble_model(
            ensemble_spec=spec,
            model_config_dict=config,
        )

        call_kwargs = mock_mlflow.pyfunc.log_model.call_args.kwargs
        artifacts = call_kwargs.get("artifacts", {})
        assert "ensemble_manifest" in artifacts

        manifest_path = Path(artifacts["ensemble_manifest"])
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["n_members"] == 3

    @patch("minivess.serving.model_logger.mlflow")
    def test_passes_signature(self, mock_mlflow: MagicMock, tmp_path: Path) -> None:
        """log_ensemble_model passes the model signature."""
        from minivess.serving.model_logger import log_ensemble_model

        spec = _make_ensemble_spec(tmp_path)
        config = _make_model_config()

        mock_mlflow.pyfunc.log_model.return_value = MagicMock()
        log_ensemble_model(
            ensemble_spec=spec,
            model_config_dict=config,
        )

        call_kwargs = mock_mlflow.pyfunc.log_model.call_args.kwargs
        assert call_kwargs.get("signature") is not None

    @patch("minivess.serving.model_logger.mlflow")
    def test_manifest_has_all_members(
        self, mock_mlflow: MagicMock, tmp_path: Path
    ) -> None:
        """Manifest includes entries for all ensemble members."""
        from minivess.serving.model_logger import log_ensemble_model

        spec = _make_ensemble_spec(tmp_path, n_members=4)
        config = _make_model_config()

        mock_mlflow.pyfunc.log_model.return_value = MagicMock()
        log_ensemble_model(
            ensemble_spec=spec,
            model_config_dict=config,
        )

        call_kwargs = mock_mlflow.pyfunc.log_model.call_args.kwargs
        manifest_path = Path(call_kwargs["artifacts"]["ensemble_manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert len(manifest["members"]) == 4

    @patch("minivess.serving.model_logger.mlflow")
    def test_uses_ensemble_model_class(
        self, mock_mlflow: MagicMock, tmp_path: Path
    ) -> None:
        """log_ensemble_model uses MiniVessEnsembleModel as python_model."""
        from minivess.serving.mlflow_wrapper import MiniVessEnsembleModel
        from minivess.serving.model_logger import log_ensemble_model

        spec = _make_ensemble_spec(tmp_path)
        config = _make_model_config()

        mock_mlflow.pyfunc.log_model.return_value = MagicMock()
        log_ensemble_model(
            ensemble_spec=spec,
            model_config_dict=config,
        )

        call_kwargs = mock_mlflow.pyfunc.log_model.call_args.kwargs
        python_model = call_kwargs.get("python_model")
        assert isinstance(python_model, MiniVessEnsembleModel)
