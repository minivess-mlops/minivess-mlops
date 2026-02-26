"""Tests for MLflow pyfunc serving wrappers.

Phase 1 of the evaluation plan: MLflow pyfunc wrapper (#76).
Tests MiniVessSegModel (single model) and MiniVessEnsembleModel (ensemble).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Mock helpers for tests
# ---------------------------------------------------------------------------


class _MockNet(nn.Module):
    """Minimal network that produces predictable 2-class softmax output."""

    def __init__(self, fg_prob: float = 0.8) -> None:
        super().__init__()
        self._fg_prob = fg_prob
        # Need at least one parameter so state_dict is non-empty
        self._dummy = nn.Parameter(torch.tensor(fg_prob))

    def forward(self, x: Tensor) -> Tensor:
        b, _c, d, h, w = x.shape
        fg = torch.full((b, 1, d, h, w), self._fg_prob)
        bg = torch.full((b, 1, d, h, w), 1.0 - self._fg_prob)
        return torch.cat([bg, fg], dim=1)


def _save_single_checkpoint(tmp_path: Path, *, fg_prob: float = 0.8) -> Path:
    """Save a mock checkpoint in the new format."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    net = _MockNet(fg_prob)
    ckpt_path = tmp_path / "best_val_compound_masd_cldice.pth"
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "checkpoint_metadata": {"epoch": 50, "metric_name": "val_compound_masd_cldice"},
        },
        ckpt_path,
    )
    return ckpt_path


def _save_model_config(tmp_path: Path) -> Path:
    """Save a minimal model config JSON using 'test' family for _SimpleNet."""
    config_path = tmp_path / "model_config.json"
    config_path.write_text(
        json.dumps(
            {
                "family": "test",
                "name": "test-model",
                "in_channels": 1,
                "out_channels": 2,
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _save_ensemble_manifest(
    tmp_path: Path,
    member_paths: list[Path],
) -> Path:
    """Save an ensemble manifest JSON."""
    manifest_path = tmp_path / "ensemble_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "strategy": "mean",
                "members": [
                    {
                        "checkpoint_path": str(p),
                        "run_id": f"run_{i}",
                        "loss_type": "dice_ce",
                        "fold_id": i,
                    }
                    for i, p in enumerate(member_paths)
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


# ---------------------------------------------------------------------------
# Tests: MiniVessSegModel (single model)
# ---------------------------------------------------------------------------


class TestMiniVessSegModel:
    """Tests for single-model MLflow pyfunc wrapper."""

    def test_import(self) -> None:
        """MiniVessSegModel is importable from the serving module."""
        from minivess.serving.mlflow_wrapper import MiniVessSegModel

        assert MiniVessSegModel is not None

    def test_is_pyfunc_python_model(self) -> None:
        """MiniVessSegModel extends mlflow.pyfunc.PythonModel."""
        import mlflow.pyfunc

        from minivess.serving.mlflow_wrapper import MiniVessSegModel

        model = MiniVessSegModel()
        assert isinstance(model, mlflow.pyfunc.PythonModel)

    def test_load_context_from_checkpoint(self, tmp_path: Path) -> None:
        """load_context() loads a model from a checkpoint file."""
        from minivess.serving.mlflow_wrapper import MiniVessSegModel

        ckpt_path = _save_single_checkpoint(tmp_path)
        config_path = _save_model_config(tmp_path)

        context = MagicMock()
        context.artifacts = {
            "checkpoint": str(ckpt_path),
            "model_config": str(config_path),
        }

        model = MiniVessSegModel()
        model.load_context(context)

        assert model._net is not None

    def test_predict_returns_numpy(self, tmp_path: Path) -> None:
        """predict() returns numpy array with correct shape."""
        from minivess.serving.mlflow_wrapper import MiniVessSegModel

        ckpt_path = _save_single_checkpoint(tmp_path)
        config_path = _save_model_config(tmp_path)

        context = MagicMock()
        context.artifacts = {
            "checkpoint": str(ckpt_path),
            "model_config": str(config_path),
        }

        model = MiniVessSegModel()
        model.load_context(context)

        # Input: (B, C_in, D, H, W) = (1, 1, 8, 8, 4)
        input_data = np.random.default_rng(42).random((1, 1, 8, 8, 4), dtype=np.float32)
        result = model.predict(context, input_data)

        assert isinstance(result, np.ndarray)
        # Output: (B, C_out, D, H, W) = (1, 2, 8, 8, 4)
        assert result.shape == (1, 2, 8, 8, 4)

    def test_predict_output_is_probability(self, tmp_path: Path) -> None:
        """predict() output sums to ~1.0 along class dimension."""
        from minivess.serving.mlflow_wrapper import MiniVessSegModel

        ckpt_path = _save_single_checkpoint(tmp_path)
        config_path = _save_model_config(tmp_path)

        context = MagicMock()
        context.artifacts = {
            "checkpoint": str(ckpt_path),
            "model_config": str(config_path),
        }

        model = MiniVessSegModel()
        model.load_context(context)

        input_data = np.random.default_rng(42).random((1, 1, 8, 8, 4), dtype=np.float32)
        result = model.predict(context, input_data)

        # Sum across class dim (axis=1) should be ~1.0
        class_sum = result.sum(axis=1)
        np.testing.assert_allclose(class_sum, 1.0, atol=1e-5)

    def test_predict_with_uncertainty(self, tmp_path: Path) -> None:
        """predict_with_uncertainty() returns dict with required keys."""
        from minivess.serving.mlflow_wrapper import MiniVessSegModel

        ckpt_path = _save_single_checkpoint(tmp_path)
        config_path = _save_model_config(tmp_path)

        context = MagicMock()
        context.artifacts = {
            "checkpoint": str(ckpt_path),
            "model_config": str(config_path),
        }

        model = MiniVessSegModel()
        model.load_context(context)

        input_data = np.random.default_rng(42).random((1, 1, 8, 8, 4), dtype=np.float32)
        result = model.predict_with_uncertainty(input_data)

        assert "prediction" in result
        assert "uncertainty_map" in result
        assert result["prediction"].shape == (1, 2, 8, 8, 4)

    def test_model_signature_spec(self) -> None:
        """get_model_signature() returns correct tensor specs."""
        from minivess.serving.mlflow_wrapper import get_model_signature

        sig = get_model_signature()
        assert sig is not None


# ---------------------------------------------------------------------------
# Tests: MiniVessEnsembleModel
# ---------------------------------------------------------------------------


class TestMiniVessEnsembleModel:
    """Tests for ensemble MLflow pyfunc wrapper."""

    def test_import(self) -> None:
        """MiniVessEnsembleModel is importable."""
        from minivess.serving.mlflow_wrapper import MiniVessEnsembleModel

        assert MiniVessEnsembleModel is not None

    def test_is_pyfunc_python_model(self) -> None:
        """MiniVessEnsembleModel extends mlflow.pyfunc.PythonModel."""
        import mlflow.pyfunc

        from minivess.serving.mlflow_wrapper import MiniVessEnsembleModel

        model = MiniVessEnsembleModel()
        assert isinstance(model, mlflow.pyfunc.PythonModel)

    def test_load_context_from_manifest(self, tmp_path: Path) -> None:
        """load_context() loads multiple models from manifest."""
        from minivess.serving.mlflow_wrapper import MiniVessEnsembleModel

        # Create 3 member checkpoints
        member_paths = []
        for i in range(3):
            p = _save_single_checkpoint(tmp_path / f"member_{i}", fg_prob=0.7 + i * 0.05)
            member_paths.append(p)

        config_path = _save_model_config(tmp_path)
        manifest_path = _save_ensemble_manifest(tmp_path, member_paths)

        context = MagicMock()
        context.artifacts = {
            "ensemble_manifest": str(manifest_path),
            "model_config": str(config_path),
        }

        model = MiniVessEnsembleModel()
        model.load_context(context)

        assert model._members is not None
        assert len(model._members) == 3

    def test_predict_returns_mean_probabilities(self, tmp_path: Path) -> None:
        """predict() returns averaged probabilities across members."""
        from minivess.serving.mlflow_wrapper import MiniVessEnsembleModel

        member_paths = []
        for i in range(3):
            p = _save_single_checkpoint(tmp_path / f"member_{i}", fg_prob=0.7 + i * 0.05)
            member_paths.append(p)

        config_path = _save_model_config(tmp_path)
        manifest_path = _save_ensemble_manifest(tmp_path, member_paths)

        context = MagicMock()
        context.artifacts = {
            "ensemble_manifest": str(manifest_path),
            "model_config": str(config_path),
        }

        model = MiniVessEnsembleModel()
        model.load_context(context)

        input_data = np.random.default_rng(42).random((1, 1, 8, 8, 4), dtype=np.float32)
        result = model.predict(context, input_data)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 2, 8, 8, 4)
        # Probabilities sum to 1
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_with_uncertainty_decomposition(self, tmp_path: Path) -> None:
        """predict_with_uncertainty() returns total, aleatoric, epistemic maps."""
        from minivess.serving.mlflow_wrapper import MiniVessEnsembleModel

        member_paths = []
        for i in range(3):
            p = _save_single_checkpoint(tmp_path / f"member_{i}", fg_prob=0.6 + i * 0.1)
            member_paths.append(p)

        config_path = _save_model_config(tmp_path)
        manifest_path = _save_ensemble_manifest(tmp_path, member_paths)

        context = MagicMock()
        context.artifacts = {
            "ensemble_manifest": str(manifest_path),
            "model_config": str(config_path),
        }

        model = MiniVessEnsembleModel()
        model.load_context(context)

        input_data = np.random.default_rng(42).random((1, 1, 8, 8, 4), dtype=np.float32)
        result = model.predict_with_uncertainty(input_data)

        assert "prediction" in result
        assert "total_uncertainty" in result
        assert "aleatoric_uncertainty" in result
        assert "epistemic_uncertainty" in result
        assert result["prediction"].shape == (1, 2, 8, 8, 4)
        assert result["total_uncertainty"].shape == (1, 1, 8, 8, 4)

    def test_ensemble_uncertainty_epistemic_nonnegative(self, tmp_path: Path) -> None:
        """Epistemic uncertainty should be non-negative."""
        from minivess.serving.mlflow_wrapper import MiniVessEnsembleModel

        member_paths = []
        for i in range(3):
            p = _save_single_checkpoint(tmp_path / f"member_{i}", fg_prob=0.5 + i * 0.15)
            member_paths.append(p)

        config_path = _save_model_config(tmp_path)
        manifest_path = _save_ensemble_manifest(tmp_path, member_paths)

        context = MagicMock()
        context.artifacts = {
            "ensemble_manifest": str(manifest_path),
            "model_config": str(config_path),
        }

        model = MiniVessEnsembleModel()
        model.load_context(context)

        input_data = np.random.default_rng(42).random((1, 1, 8, 8, 4), dtype=np.float32)
        result = model.predict_with_uncertainty(input_data)

        # Epistemic = total - aleatoric >= 0 (by Jensen's inequality)
        np.testing.assert_array_less(-1e-6, result["epistemic_uncertainty"])
