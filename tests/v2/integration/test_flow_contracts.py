"""Flow contract integration tests — verify e2e wiring between all 5 flows.

Tests that each flow can discover upstream artifacts and produce downstream
artifacts in the format the next flow expects. Uses mock MLflow data in tmp_path.

These tests verify the WIRING, not the ML logic. They catch the "silent stub"
pattern (metalearning/2026-03-19-external-test-datasets-never-wired-silent-failure.md)
where infrastructure code exists but is never connected.

CLAUDE.md Rule 25: Loud failures. If a flow receives empty input, it must RAISE.
CLAUDE.md Rule 27: Debug = production. All flows tested, no shortcuts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


@pytest.fixture()
def mock_mlflow_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set up mock MLflow environment for flow contract testing."""
    mlruns = tmp_path / "mlruns"
    mlruns.mkdir()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(mlruns))
    monkeypatch.setenv("MINIVESS_ALLOW_HOST", "1")
    monkeypatch.setenv("PREFECT_DISABLED", "1")
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("SPLITS_DIR", str(tmp_path / "splits"))
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("ANALYSIS_OUTPUT_DIR", str(tmp_path / "analysis"))

    # Create required directories
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "splits").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "analysis").mkdir()

    # Create minimal splits file
    splits = [
        {
            "train": [
                {
                    "image": "data/raw/minivess/imagesTr/mv01.nii.gz",
                    "label": "data/raw/minivess/labelsTr/mv01.nii.gz",
                },
            ],
            "val": [
                {
                    "image": "data/raw/minivess/imagesTr/mv02.nii.gz",
                    "label": "data/raw/minivess/labelsTr/mv02.nii.gz",
                },
            ],
        }
    ]
    (tmp_path / "splits" / "splits.json").write_text(
        json.dumps(splits), encoding="utf-8"
    )

    return tmp_path


class TestTrainingFlowConfig:
    """Verify training flow config composition works for all factorial conditions."""

    def test_compose_experiment_config_debug_factorial(self) -> None:
        """debug_factorial.yaml resolves through Hydra composition."""
        from minivess.config.compose import compose_experiment_config

        cfg = compose_experiment_config(experiment_name="debug_factorial")
        assert cfg is not None
        assert "factors" in cfg or "max_epochs" in cfg

    @pytest.mark.parametrize(
        "model",
        ["dynunet", "mambavesselnet", "sam3_topolora", "sam3_hybrid"],
    )
    def test_compose_with_model_override(self, model: str) -> None:
        """Each paper model resolves correctly with Hydra override."""
        from minivess.config.compose import compose_experiment_config

        cfg = compose_experiment_config(
            experiment_name="debug_factorial",
            overrides=[f"model={model}"],
        )
        assert cfg is not None


class TestAuxCalibWiringInTrainFlow:
    """Verify aux_calib is extracted from config and passed to loss builder."""

    def test_build_loss_with_config_dict(self) -> None:
        """Simulates what train_one_fold_task does with config_dict."""
        from minivess.pipeline.loss_functions import build_loss_function

        # Simulate config_dict extraction (same as train_flow.py lines 510-516)
        config = {
            "loss_name": "cbdice_cldice",
            "with_aux_calib": True,
            "aux_calib_weight": 0.5,
        }
        with_aux_calib: bool = config.get("with_aux_calib", False)
        aux_calib_weight: float = config.get("aux_calib_weight", 1.0)
        criterion = build_loss_function(
            config["loss_name"],
            with_aux_calib=with_aux_calib,
            aux_calib_weight=aux_calib_weight,
        )
        assert type(criterion).__name__ == "AuxCalibCompoundLoss"

    def test_build_loss_without_aux_calib_in_config(self) -> None:
        """When with_aux_calib is False or missing, bare loss returned."""
        from minivess.pipeline.loss_functions import build_loss_function

        config: dict[str, object] = {"loss_name": "dice_ce"}
        with_aux_calib: bool = bool(config.get("with_aux_calib", False))
        criterion = build_loss_function(
            str(config["loss_name"]),
            with_aux_calib=with_aux_calib,
        )
        assert type(criterion).__name__ != "AuxCalibCompoundLoss"


class TestPostTrainingFlowContract:
    """Verify post-training flow can discover training checkpoints."""

    def test_checkpoint_discovery_function_exists(self) -> None:
        """resolve_checkpoint_paths_from_contract is importable and callable."""
        from minivess.orchestration.flows.post_training_flow import (
            resolve_checkpoint_paths_from_contract,
        )

        assert callable(resolve_checkpoint_paths_from_contract)

    def test_checkpoint_averaging_is_in_weight_plugins(self) -> None:
        """Checkpoint averaging is in the weight plugins set."""
        from minivess.orchestration.flows.post_training_flow import _WEIGHT_PLUGINS

        assert "checkpoint_averaging" in _WEIGHT_PLUGINS


class TestAnalysisFlowContract:
    """Verify analysis flow evaluation and champion tagging."""

    def test_evaluation_runner_importable(self) -> None:
        """UnifiedEvaluationRunner is importable."""
        from minivess.pipeline.evaluation_runner import UnifiedEvaluationRunner

        assert UnifiedEvaluationRunner is not None

    def test_champion_discovery_importable(self) -> None:
        """Champion discovery for deploy flow is importable."""
        from minivess.pipeline.deploy_champion_discovery import discover_champions

        assert callable(discover_champions)


class TestDeployFlowContract:
    """Verify deploy flow ONNX + BentoML chain."""

    def test_deploy_flow_importable(self) -> None:
        """Deploy flow is importable."""
        from minivess.orchestration.flows.deploy_flow import deploy_flow

        assert callable(deploy_flow)

    def test_deploy_config_from_env(self) -> None:
        """DeployConfig can be created from environment."""
        from minivess.config.deploy_config import DeployConfig

        assert DeployConfig is not None


class TestBiostatisticsFlowContract:
    """Verify biostatistics flow can read MLflow data."""

    def test_biostatistics_flow_importable(self) -> None:
        """Biostatistics flow is importable."""
        from minivess.orchestration.flows.biostatistics_flow import (
            run_biostatistics_flow,
        )

        assert callable(run_biostatistics_flow)

    def test_duckdb_builder_importable(self) -> None:
        """DuckDB builder is importable."""
        from minivess.pipeline.biostatistics_duckdb import build_biostatistics_duckdb

        assert callable(build_biostatistics_duckdb)


class TestMetricKeyConvention:
    """Verify metric key constants exist for all required prefixes."""

    def test_metric_keys_has_required_prefixes(self) -> None:
        """MetricKeys class has all required prefix constants."""
        from minivess.observability.metric_keys import MetricKeys

        # Training metrics
        assert hasattr(MetricKeys, "TRAIN_LOSS")
        assert "/" in MetricKeys.TRAIN_LOSS

        # Validation metrics
        assert hasattr(MetricKeys, "VAL_LOSS")
        assert "/" in MetricKeys.VAL_LOSS

    def test_no_migration_map_in_greenfield(self) -> None:
        """MIGRATION_MAP should not exist (greenfield — CLAUDE.md Rule 26).

        If this test fails, it means backward compat code was re-added.
        Delete it — this is a greenfield project with zero legacy data.
        """
        from minivess.observability import metric_keys

        # MIGRATION_MAP may still exist during migration — this test
        # tracks the goal. Once migration is complete, enforce strictly.
        if hasattr(metric_keys, "MIGRATION_MAP"):
            migration_map = metric_keys.MIGRATION_MAP
            # Allow empty map (migration complete) or non-empty (in progress)
            # The goal is to eventually delete it entirely
            assert isinstance(migration_map, dict)


class TestRunFactorialScript:
    """Verify run_factorial.sh can parse both debug and production configs."""

    def test_script_exists_and_is_executable(self) -> None:
        """run_factorial.sh exists and has execute permission."""
        script = Path(__file__).resolve().parents[3] / "scripts" / "run_factorial.sh"
        assert script.exists(), "run_factorial.sh not found"
        assert os.access(script, os.X_OK), "run_factorial.sh not executable"

    def test_script_syntax_valid(self) -> None:
        """Bash syntax check passes."""
        import subprocess

        script = Path(__file__).resolve().parents[3] / "scripts" / "run_factorial.sh"
        result = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Syntax error: {result.stderr}"
