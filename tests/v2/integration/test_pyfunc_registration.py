"""Integration tests for post-hoc pyfunc model registration.

Task B1: Register existing training checkpoints as MLflow pyfunc models
so they can be loaded via ``mlflow.pyfunc.load_model()``.

Tests verify:
1. A single checkpoint can be registered as a pyfunc model in a tmp MLflow backend
2. The registered model is loadable via ``mlflow.pyfunc.load_model()``
3. The loaded model produces valid output (correct shape, valid probabilities)
4. All production checkpoints can be registered in bulk
5. The MLflow model registry is populated after registration

Tests use ``tmp_path`` for the MLflow tracking URI to avoid modifying the
production mlruns/ directory.  Integration tests against real checkpoints
are skipped when mlruns/ is absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pytest
import torch
from torch import Tensor, nn

from minivess.pipeline.mlruns_inspector import get_production_runs, get_run_tags

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MLRUNS_DIR: Path = Path(__file__).resolve().parents[3] / "mlruns"
V2_EXPERIMENT_ID: str = "843896622863223169"

_mlruns_missing: bool = not (MLRUNS_DIR / V2_EXPERIMENT_ID).is_dir()
_skip_reason: str = (
    f"mlruns experiment directory not found: {MLRUNS_DIR / V2_EXPERIMENT_ID}. "
    "Run training first or verify the MLRUNS_DIR path."
)

# Default model config for DynUNet used in production runs
DYNUNET_MODEL_CONFIG: dict[str, Any] = {
    "family": "dynunet",
    "name": "dynunet",
    "in_channels": 1,
    "out_channels": 2,
    "architecture_params": {
        "filters": [32, 64, 128, 256],
    },
}

# Input/output shapes for inference tests
_INPUT_SHAPE: tuple[int, ...] = (1, 1, 32, 32, 16)
_OUTPUT_CLASSES: int = 2


# ---------------------------------------------------------------------------
# Mock helpers (for unit-level tests that don't need real checkpoints)
# ---------------------------------------------------------------------------


class _MockNet(nn.Module):  # type: ignore[misc]
    """Minimal network for checkpoint creation."""

    def __init__(self, fg_prob: float = 0.8) -> None:
        super().__init__()
        self._dummy = nn.Parameter(torch.tensor(fg_prob))

    def forward(self, x: Tensor) -> Tensor:
        b, _c, d, h, w = x.shape
        fg = torch.sigmoid(self._dummy).expand(b, 1, d, h, w)
        bg = 1.0 - fg
        return torch.cat([bg, fg], dim=1)


def _make_mock_checkpoint(tmp_path: Path, name: str = "ckpt.pth") -> Path:
    """Save a mock checkpoint and return its path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    net = _MockNet()
    ckpt_path = tmp_path / name
    torch.save({"model_state_dict": net.state_dict()}, ckpt_path)
    return ckpt_path


def _make_mock_model_config() -> dict[str, Any]:
    """Return a minimal model config dict (uses _SimpleNet fallback)."""
    return {
        "family": "test",
        "name": "test-model",
        "in_channels": 1,
        "out_channels": 2,
    }


# ---------------------------------------------------------------------------
# Tests: Unit-level (mock checkpoints, tmp_path MLflow)
# ---------------------------------------------------------------------------


class TestRegisterSingleCheckpointAsPyfunc:
    """Test that a single checkpoint can be registered as a pyfunc model."""

    def test_register_single_checkpoint_as_pyfunc(self, tmp_path: Path) -> None:
        """register_checkpoint_as_pyfunc creates a pyfunc model in MLflow.

        Uses a mock checkpoint and temporary MLflow tracking URI.
        Verifies the model_info is returned and has a valid model_uri.
        """
        from scripts.register_models import register_checkpoint_as_pyfunc

        ckpt_path = _make_mock_checkpoint(tmp_path / "checkpoints")
        tracking_uri = str(tmp_path / "mlruns")

        model_info = register_checkpoint_as_pyfunc(
            checkpoint_path=ckpt_path,
            model_config=_make_mock_model_config(),
            tracking_uri=tracking_uri,
            experiment_name="test_registration",
            run_name="test_run",
            loss_type="dice_ce",
            checkpoint_name="best_val_dice",
        )

        assert model_info is not None
        assert hasattr(model_info, "model_uri")
        assert "model" in model_info.model_uri


class TestRegisteredModelLoadable:
    """Test that a registered pyfunc model can be loaded back."""

    def test_registered_model_loadable(self, tmp_path: Path) -> None:
        """A pyfunc model registered by register_checkpoint_as_pyfunc is loadable.

        Calls mlflow.pyfunc.load_model() on the returned model_uri and
        verifies the loaded model is not None.
        """
        from scripts.register_models import register_checkpoint_as_pyfunc

        ckpt_path = _make_mock_checkpoint(tmp_path / "checkpoints")
        tracking_uri = str(tmp_path / "mlruns")

        model_info = register_checkpoint_as_pyfunc(
            checkpoint_path=ckpt_path,
            model_config=_make_mock_model_config(),
            tracking_uri=tracking_uri,
            experiment_name="test_registration",
            run_name="test_loadable",
            loss_type="dice_ce",
            checkpoint_name="best_val_dice",
        )

        mlflow.set_tracking_uri(tracking_uri)
        loaded = mlflow.pyfunc.load_model(model_info.model_uri)
        assert loaded is not None


class TestRegisteredModelProducesValidOutput:
    """Test that a loaded registered model produces valid inference output."""

    def test_registered_model_produces_valid_output(self, tmp_path: Path) -> None:
        """Loaded pyfunc model produces correct shape and valid probabilities.

        Input: (1, 1, 8, 8, 4) float32
        Expected output: (1, 2, 8, 8, 4) float32 with class-sum ~1.0
        """
        from scripts.register_models import register_checkpoint_as_pyfunc

        ckpt_path = _make_mock_checkpoint(tmp_path / "checkpoints")
        tracking_uri = str(tmp_path / "mlruns")

        model_info = register_checkpoint_as_pyfunc(
            checkpoint_path=ckpt_path,
            model_config=_make_mock_model_config(),
            tracking_uri=tracking_uri,
            experiment_name="test_registration",
            run_name="test_output",
            loss_type="dice_ce",
            checkpoint_name="best_val_dice",
        )

        mlflow.set_tracking_uri(tracking_uri)
        loaded = mlflow.pyfunc.load_model(model_info.model_uri)

        input_data = np.random.default_rng(42).random((1, 1, 8, 8, 4), dtype=np.float32)
        result = loaded.predict(input_data)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 2, 8, 8, 4)
        # Probabilities sum to ~1.0 along class dimension
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-5)


class TestRegisterAllProductionCheckpoints:
    """Test bulk registration of all production checkpoints."""

    def test_register_all_production_checkpoints_mock(self, tmp_path: Path) -> None:
        """register_all_production_checkpoints handles multiple checkpoints.

        Creates a mock mlruns directory structure matching production layout,
        then verifies all checkpoints are registered.
        """
        from scripts.register_models import register_all_production_checkpoints

        # Build mock mlruns structure: 2 runs x 2 checkpoints
        mock_mlruns = tmp_path / "source_mlruns"
        exp_dir = mock_mlruns / "test_experiment"
        exp_dir.mkdir(parents=True)

        checkpoint_names = ["best_val_dice.pth", "last.pth"]
        run_ids = []
        for i, loss_name in enumerate(["dice_ce", "cbdice"]):
            run_id = f"run_{i:032d}"
            run_ids.append(run_id)
            run_dir = exp_dir / run_id

            # Create checkpoints
            ckpt_dir = run_dir / "artifacts" / "checkpoints"
            ckpt_dir.mkdir(parents=True)
            for ckpt_name in checkpoint_names:
                _make_mock_checkpoint(ckpt_dir, name=ckpt_name)

            # Create tags
            tags_dir = run_dir / "tags"
            tags_dir.mkdir(parents=True)
            (tags_dir / "loss_function").write_text(loss_name, encoding="utf-8")
            (tags_dir / "num_folds").write_text("3", encoding="utf-8")
            (tags_dir / "model_family").write_text("dynunet", encoding="utf-8")

            # Create metrics with eval_fold2 marker (production discriminator)
            metrics_dir = run_dir / "metrics"
            metrics_dir.mkdir(parents=True)
            (metrics_dir / "eval_fold2_dsc").write_text(
                "1234567890 0.85 0", encoding="utf-8"
            )

        tracking_uri = str(tmp_path / "dest_mlruns")
        results = register_all_production_checkpoints(
            source_mlruns_dir=mock_mlruns,
            experiment_id="test_experiment",
            model_config=_make_mock_model_config(),
            tracking_uri=tracking_uri,
            target_experiment_name="test_registered",
            checkpoint_names=checkpoint_names,
        )

        # Should register 2 runs x 2 checkpoints = 4 total
        assert len(results) == 4  # noqa: PLR2004
        for info in results:
            assert info is not None
            assert hasattr(info, "model_uri")


class TestRegistryPopulatedAfterRegistration:
    """Test that MLflow experiment has runs after registration."""

    def test_registry_populated_after_registration(self, tmp_path: Path) -> None:
        """After registration, the target experiment has the expected runs.

        Verifies by querying the MLflow tracking backend for runs in the
        target experiment.
        """
        from scripts.register_models import register_checkpoint_as_pyfunc

        tracking_uri = str(tmp_path / "mlruns")
        experiment_name = "test_registry_populated"

        # Register 3 models
        for i in range(3):
            ckpt_path = _make_mock_checkpoint(
                tmp_path / f"checkpoints_{i}", name=f"ckpt_{i}.pth"
            )
            register_checkpoint_as_pyfunc(
                checkpoint_path=ckpt_path,
                model_config=_make_mock_model_config(),
                tracking_uri=tracking_uri,
                experiment_name=experiment_name,
                run_name=f"run_{i}",
                loss_type="dice_ce",
                checkpoint_name=f"best_val_dice_{i}",
            )

        # Query MLflow for runs in the experiment
        mlflow.set_tracking_uri(tracking_uri)
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(experiment_name)
        assert experiment is not None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
        )
        assert len(runs) == 3  # noqa: PLR2004


# ---------------------------------------------------------------------------
# Integration tests: Real production checkpoints
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(_mlruns_missing, reason=_skip_reason)
class TestRealCheckpointRegistration:
    """Integration tests using real production checkpoints.

    These tests register real DynUNet checkpoints from the v2 experiment
    into a temporary MLflow backend (not modifying the production mlruns/).
    """

    def test_register_real_checkpoint_loadable(self, tmp_path: Path) -> None:
        """A real production checkpoint can be registered and loaded back.

        Uses the first production run's best_val_compound_masd_cldice.pth.
        """
        from scripts.register_models import register_checkpoint_as_pyfunc

        production_runs = get_production_runs(MLRUNS_DIR, V2_EXPERIMENT_ID)
        assert len(production_runs) > 0, "No production runs found"

        run_id = production_runs[0]
        tags = get_run_tags(MLRUNS_DIR, V2_EXPERIMENT_ID, run_id)
        loss_type = tags.get("loss_function", "unknown")

        ckpt_path = (
            MLRUNS_DIR
            / V2_EXPERIMENT_ID
            / run_id
            / "artifacts"
            / "checkpoints"
            / "best_val_compound_masd_cldice.pth"
        )
        assert ckpt_path.is_file(), f"Checkpoint not found: {ckpt_path}"

        tracking_uri = str(tmp_path / "mlruns")
        model_info = register_checkpoint_as_pyfunc(
            checkpoint_path=ckpt_path,
            model_config=DYNUNET_MODEL_CONFIG,
            tracking_uri=tracking_uri,
            experiment_name="test_real_registration",
            run_name=f"{loss_type}_best_compound",
            loss_type=loss_type,
            checkpoint_name="best_val_compound_masd_cldice",
        )

        mlflow.set_tracking_uri(tracking_uri)
        loaded = mlflow.pyfunc.load_model(model_info.model_uri)
        assert loaded is not None

    def test_real_checkpoint_produces_valid_output(self, tmp_path: Path) -> None:
        """A real registered checkpoint produces valid softmax output.

        Input: (1, 1, 32, 32, 16) float32
        Output: (1, 2, 32, 32, 16) float32 with valid probabilities.
        """
        from scripts.register_models import register_checkpoint_as_pyfunc

        production_runs = get_production_runs(MLRUNS_DIR, V2_EXPERIMENT_ID)
        run_id = production_runs[0]
        tags = get_run_tags(MLRUNS_DIR, V2_EXPERIMENT_ID, run_id)
        loss_type = tags.get("loss_function", "unknown")

        ckpt_path = (
            MLRUNS_DIR
            / V2_EXPERIMENT_ID
            / run_id
            / "artifacts"
            / "checkpoints"
            / "best_val_compound_masd_cldice.pth"
        )

        tracking_uri = str(tmp_path / "mlruns")
        model_info = register_checkpoint_as_pyfunc(
            checkpoint_path=ckpt_path,
            model_config=DYNUNET_MODEL_CONFIG,
            tracking_uri=tracking_uri,
            experiment_name="test_real_output",
            run_name=f"{loss_type}_output_test",
            loss_type=loss_type,
            checkpoint_name="best_val_compound_masd_cldice",
        )

        mlflow.set_tracking_uri(tracking_uri)
        loaded = mlflow.pyfunc.load_model(model_info.model_uri)

        input_data = np.random.default_rng(42).random(_INPUT_SHAPE, dtype=np.float32)
        result = loaded.predict(input_data)

        assert isinstance(result, np.ndarray)
        expected_shape = (
            _INPUT_SHAPE[0],
            _OUTPUT_CLASSES,
            *_INPUT_SHAPE[2:],
        )
        assert result.shape == expected_shape
        # Valid probabilities
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_register_all_real_production_checkpoints(self, tmp_path: Path) -> None:
        """All production best_val_compound_masd_cldice checkpoints register.

        Registers one checkpoint per production run (4 runs total) and
        verifies the target experiment has 4 runs afterward.
        """
        from scripts.register_models import register_all_production_checkpoints

        tracking_uri = str(tmp_path / "mlruns")
        results = register_all_production_checkpoints(
            source_mlruns_dir=MLRUNS_DIR,
            experiment_id=V2_EXPERIMENT_ID,
            model_config=DYNUNET_MODEL_CONFIG,
            tracking_uri=tracking_uri,
            target_experiment_name="test_all_real",
            checkpoint_names=["best_val_compound_masd_cldice.pth"],
        )

        # 4 production runs x 1 checkpoint each = 4
        assert len(results) == 4  # noqa: PLR2004

        # Verify MLflow experiment populated
        mlflow.set_tracking_uri(tracking_uri)
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name("test_all_real")
        assert experiment is not None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
        )
        assert len(runs) == 4  # noqa: PLR2004
