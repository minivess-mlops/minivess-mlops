"""Integration smoke tests for the Deploy Flow (Flow 4).

End-to-end tests with mock mlruns directories verifying the full
pipeline: champion discovery -> ONNX export -> BentoML import ->
artifact generation -> promotion.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003 — used by pytest fixtures at runtime
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_mock_mlruns(
    tmp_path: Path,
    *,
    experiment_id: str = "1",
    champions: list[dict[str, Any]] | None = None,
) -> Path:
    """Create a comprehensive mock MLflow directory structure."""
    mlruns_dir = tmp_path / "mlruns"

    if champions is None:
        champions = [
            {
                "run_id": "balanced_run_001",
                "tags": {
                    "champion_best_cv_mean": "true",
                    "champion_metric_name": "val_compound_nsd_cldice",
                    "champion_metric_value": "0.88",
                },
                "metrics": {"val_dice": "0.85", "val_cldice": "0.90"},
            },
        ]

    for champ in champions:
        run_id = champ["run_id"]
        run_dir = mlruns_dir / experiment_id / run_id

        # Tags
        tags_dir = run_dir / "tags"
        tags_dir.mkdir(parents=True)
        (tags_dir / "mlflow.runName").write_text(run_id, encoding="utf-8")
        for key, value in champ.get("tags", {}).items():
            (tags_dir / key).write_text(value, encoding="utf-8")

        # Metrics
        metrics_dir = run_dir / "metrics"
        metrics_dir.mkdir(parents=True)
        for name, value in champ.get("metrics", {}).items():
            ts = datetime.now(UTC).timestamp()
            (metrics_dir / name).write_text(f"{ts} {value} 0", encoding="utf-8")

        # Params
        params_dir = run_dir / "params"
        params_dir.mkdir(parents=True)
        (params_dir / "loss_name").write_text("dice_ce", encoding="utf-8")

        # meta.yaml
        meta = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "status": "FINISHED",
        }
        (run_dir / "meta.yaml").write_text(
            "\n".join(f"{k}: {v}" for k, v in meta.items()),
            encoding="utf-8",
        )

        # Checkpoint artifact
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True)
        _save_tiny_checkpoint(artifacts_dir / "best_checkpoint.pt")

    return mlruns_dir


def _save_tiny_checkpoint(path: Path) -> None:
    """Save a tiny Conv3d checkpoint for testing."""

    class TinyModel(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv3d(1, 2, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    model = TinyModel()
    torch.save({"model_state_dict": model.state_dict()}, path)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestDeployFlowIntegration:
    """End-to-end integration tests for the deploy flow."""

    def test_full_pipeline_single_champion(self, tmp_path: Path) -> None:
        """Full pipeline with one champion: discover -> export -> import -> artifacts."""
        from minivess.config.deploy_config import DeployConfig
        from minivess.orchestration.deploy_flow import deploy_flow

        mlruns_dir = _create_mock_mlruns(tmp_path)
        config = DeployConfig(
            mlruns_dir=mlruns_dir,
            output_dir=tmp_path / "deploy_output",
        )

        result = deploy_flow(config, experiment_id="1")

        assert len(result.champions) == 1
        assert result.champions[0].run_id == "balanced_run_001"
        assert len(result.onnx_paths) == 1
        # Artifacts dir should be created
        assert result.artifacts_dir.exists()

    def test_full_pipeline_no_champions(self, tmp_path: Path) -> None:
        """Flow completes gracefully when no champions are found."""
        from minivess.config.deploy_config import DeployConfig
        from minivess.orchestration.deploy_flow import deploy_flow

        mlruns_dir = _create_mock_mlruns(tmp_path, champions=[])
        config = DeployConfig(
            mlruns_dir=mlruns_dir,
            output_dir=tmp_path / "deploy_output",
        )

        result = deploy_flow(config, experiment_id="1")

        assert len(result.champions) == 0
        assert len(result.onnx_paths) == 0
        assert len(result.bento_tags) == 0

    def test_onnx_export_produces_valid_model(self, tmp_path: Path) -> None:
        """Exported ONNX model can be loaded and run inference."""
        from minivess.config.deploy_config import DeployConfig
        from minivess.orchestration.deploy_flow import deploy_flow
        from minivess.pipeline.deploy_onnx_export import validate_onnx_model

        mlruns_dir = _create_mock_mlruns(tmp_path)
        config = DeployConfig(
            mlruns_dir=mlruns_dir,
            output_dir=tmp_path / "deploy_output",
        )

        result = deploy_flow(config, experiment_id="1")

        for onnx_path in result.onnx_paths.values():
            assert onnx_path.exists()
            assert validate_onnx_model(onnx_path)

    def test_artifacts_generated(self, tmp_path: Path) -> None:
        """Deployment artifacts are generated correctly."""
        from minivess.config.deploy_config import DeployConfig
        from minivess.orchestration.deploy_flow import deploy_flow

        mlruns_dir = _create_mock_mlruns(tmp_path)
        config = DeployConfig(
            mlruns_dir=mlruns_dir,
            output_dir=tmp_path / "deploy_output",
        )

        result = deploy_flow(config, experiment_id="1")

        artifacts_dir = result.artifacts_dir
        assert (artifacts_dir / "bentofile.yaml").exists()
        assert (artifacts_dir / "docker-compose.yaml").exists()
        assert (artifacts_dir / "DEPLOY_README.md").exists()

    def test_deploy_result_summary(self, tmp_path: Path) -> None:
        """DeployResult.to_summary() returns valid summary dict."""
        from minivess.config.deploy_config import DeployConfig
        from minivess.orchestration.deploy_flow import deploy_flow

        mlruns_dir = _create_mock_mlruns(tmp_path)
        config = DeployConfig(
            mlruns_dir=mlruns_dir,
            output_dir=tmp_path / "deploy_output",
        )

        result = deploy_flow(config, experiment_id="1")
        summary = result.to_summary()

        assert isinstance(summary, dict)
        assert summary["num_champions"] == 1
        assert "onnx_models" in summary
        assert "bento_models" in summary

    def test_promotion_audit_trail(self, tmp_path: Path) -> None:
        """Promotion creates audit trail entries."""
        from minivess.config.deploy_config import DeployConfig
        from minivess.orchestration.deploy_flow import deploy_flow

        mlruns_dir = _create_mock_mlruns(tmp_path)
        config = DeployConfig(
            mlruns_dir=mlruns_dir,
            output_dir=tmp_path / "deploy_output",
        )

        result = deploy_flow(config, experiment_id="1")

        assert len(result.audit_trails) >= 1
        trail = result.audit_trails[0]
        assert "timestamp" in trail
        assert "run_id" in trail
        assert "promotion_approved" in trail

    def test_multiple_champions(self, tmp_path: Path) -> None:
        """Pipeline handles multiple champions from different categories."""
        from minivess.config.deploy_config import DeployConfig
        from minivess.orchestration.deploy_flow import deploy_flow

        champions = [
            {
                "run_id": "balanced_001",
                "tags": {
                    "champion_best_cv_mean": "true",
                    "champion_metric_name": "dsc",
                    "champion_metric_value": "0.85",
                },
                "metrics": {"val_dice": "0.85"},
            },
            {
                "run_id": "overlap_001",
                "tags": {
                    "champion_best_single_fold": "true",
                    "champion_metric_name": "dsc",
                    "champion_metric_value": "0.88",
                },
                "metrics": {"val_dice": "0.88"},
            },
        ]
        mlruns_dir = _create_mock_mlruns(tmp_path, champions=champions)
        config = DeployConfig(
            mlruns_dir=mlruns_dir,
            output_dir=tmp_path / "deploy_output",
        )

        result = deploy_flow(config, experiment_id="1")

        assert len(result.champions) == 2
        categories = {c.category for c in result.champions}
        assert len(categories) >= 1

    def test_monai_deploy_operator(self, tmp_path: Path) -> None:
        """MONAI Deploy inference operator processes volumes correctly."""
        import onnx
        from onnx import TensorProto, helper

        from minivess.serving.monai_deploy_app import MiniVessInferenceOperator

        # Create tiny ONNX model
        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4, 4])
        Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 4, 4, 4])
        weight = np.ones([2, 1, 1, 1, 1], dtype=np.float32)
        weight_init = helper.make_tensor(
            "w", TensorProto.FLOAT, [2, 1, 1, 1, 1], weight.flatten().tolist()
        )
        node = helper.make_node(
            "Conv", ["input", "w"], ["output"], kernel_shape=[1, 1, 1]
        )
        graph = helper.make_graph([node], "test", [X], [Y], initializer=[weight_init])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        onnx_path = tmp_path / "model.onnx"
        onnx.save(model, str(onnx_path))

        operator = MiniVessInferenceOperator(model_path=onnx_path)
        volume = np.random.rand(1, 1, 4, 4, 4).astype(np.float32)
        result = operator.process(volume)

        assert "segmentation" in result
        assert result["segmentation"].shape == (1, 4, 4, 4)
