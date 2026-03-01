"""Tests for the Deploy Flow (Flow 4): BentoML + MLflow + MONAI Deploy.

Tests are organized by task in the implementation plan.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ============================================================================
# Task 1: DeployConfig + ChampionCategory enum (~12 tests)
# ============================================================================


class TestChampionCategory:
    """Test ChampionCategory enum."""

    def test_champion_category_has_balanced(self) -> None:
        from minivess.config.deploy_config import ChampionCategory

        assert ChampionCategory.BALANCED.value == "balanced"

    def test_champion_category_has_topology(self) -> None:
        from minivess.config.deploy_config import ChampionCategory

        assert ChampionCategory.TOPOLOGY.value == "topology"

    def test_champion_category_has_overlap(self) -> None:
        from minivess.config.deploy_config import ChampionCategory

        assert ChampionCategory.OVERLAP.value == "overlap"

    def test_champion_category_is_strenum(self) -> None:
        from minivess.config.deploy_config import ChampionCategory

        assert isinstance(ChampionCategory.BALANCED, str)

    def test_champion_category_all_values(self) -> None:
        from minivess.config.deploy_config import ChampionCategory

        values = {c.value for c in ChampionCategory}
        assert values == {"balanced", "topology", "overlap"}


class TestDeployConfig:
    """Test DeployConfig Pydantic model."""

    def test_deploy_config_defaults(self, tmp_path: Path) -> None:
        from minivess.config.deploy_config import DeployConfig

        config = DeployConfig(
            mlruns_dir=tmp_path / "mlruns", output_dir=tmp_path / "out"
        )
        assert config.onnx_opset == 17
        assert config.bento_service_name == "minivess-segmentation"
        assert config.monai_deploy_enabled is False
        assert config.docker_registry is None

    def test_deploy_config_champion_categories_default(self, tmp_path: Path) -> None:
        from minivess.config.deploy_config import ChampionCategory, DeployConfig

        config = DeployConfig(
            mlruns_dir=tmp_path / "mlruns", output_dir=tmp_path / "out"
        )
        assert ChampionCategory.BALANCED in config.champion_categories

    def test_deploy_config_custom_categories(self, tmp_path: Path) -> None:
        from minivess.config.deploy_config import ChampionCategory, DeployConfig

        config = DeployConfig(
            mlruns_dir=tmp_path / "mlruns",
            output_dir=tmp_path / "out",
            champion_categories=[ChampionCategory.TOPOLOGY],
        )
        assert config.champion_categories == [ChampionCategory.TOPOLOGY]

    def test_deploy_config_custom_onnx_opset(self, tmp_path: Path) -> None:
        from minivess.config.deploy_config import DeployConfig

        config = DeployConfig(
            mlruns_dir=tmp_path / "mlruns",
            output_dir=tmp_path / "out",
            onnx_opset=15,
        )
        assert config.onnx_opset == 15

    def test_deploy_config_onnx_opset_min_validation(self, tmp_path: Path) -> None:
        from minivess.config.deploy_config import DeployConfig

        with pytest.raises(ValueError, match="greater than or equal to 11"):
            DeployConfig(
                mlruns_dir=tmp_path / "mlruns",
                output_dir=tmp_path / "out",
                onnx_opset=5,
            )

    def test_deploy_config_docker_registry(self, tmp_path: Path) -> None:
        from minivess.config.deploy_config import DeployConfig

        config = DeployConfig(
            mlruns_dir=tmp_path / "mlruns",
            output_dir=tmp_path / "out",
            docker_registry="ghcr.io/minivess",
        )
        assert config.docker_registry == "ghcr.io/minivess"

    def test_deploy_config_paths_are_pathlib(self, tmp_path: Path) -> None:
        from minivess.config.deploy_config import DeployConfig

        config = DeployConfig(
            mlruns_dir=tmp_path / "mlruns", output_dir=tmp_path / "out"
        )
        assert isinstance(config.mlruns_dir, Path)
        assert isinstance(config.output_dir, Path)


# ============================================================================
# Task 2: Champion model discovery (~10 tests)
# ============================================================================


def _make_mock_mlruns(
    tmp_path: Path,
    *,
    experiment_id: str = "1",
    run_id: str = "abc123",
    champion_tags: dict[str, str] | None = None,
    metrics: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
) -> Path:
    """Create a mock MLflow filesystem structure for testing."""
    mlruns_dir = tmp_path / "mlruns"
    run_dir = mlruns_dir / experiment_id / run_id
    tags_dir = run_dir / "tags"
    tags_dir.mkdir(parents=True)

    # Write mlflow.runName
    (tags_dir / "mlflow.runName").write_text("test_run", encoding="utf-8")

    if champion_tags:
        for key, value in champion_tags.items():
            (tags_dir / key).write_text(value, encoding="utf-8")

    # Write metrics
    if metrics:
        metrics_dir = run_dir / "metrics"
        metrics_dir.mkdir(parents=True)
        for name, value in metrics.items():
            (metrics_dir / name).write_text(
                f"{datetime.now(UTC).timestamp()} {value} 0", encoding="utf-8"
            )

    # Write params
    if params:
        params_dir = run_dir / "params"
        params_dir.mkdir(parents=True)
        for name, value in params.items():
            (params_dir / name).write_text(value, encoding="utf-8")

    # Write meta.yaml for the run
    meta = {"run_id": run_id, "experiment_id": experiment_id, "status": "FINISHED"}
    (run_dir / "meta.yaml").write_text(
        "\n".join(f"{k}: {v}" for k, v in meta.items()), encoding="utf-8"
    )

    return mlruns_dir


class TestChampionModelDataclass:
    """Test ChampionModel dataclass."""

    def test_champion_model_creation(self) -> None:
        from minivess.pipeline.deploy_champion_discovery import ChampionModel

        model = ChampionModel(
            run_id="abc123",
            experiment_id="1",
            category="balanced",
            metrics={"dsc": 0.85, "cldice": 0.90},
            checkpoint_path=Path("/tmp/checkpoint.pt"),
        )
        assert model.run_id == "abc123"
        assert model.category == "balanced"
        assert model.metrics["dsc"] == 0.85

    def test_champion_model_optional_fields(self) -> None:
        from minivess.pipeline.deploy_champion_discovery import ChampionModel

        model = ChampionModel(
            run_id="abc123",
            experiment_id="1",
            category="topology",
            metrics={},
        )
        assert model.checkpoint_path is None
        assert model.model_config is None


class TestDiscoverChampions:
    """Test discover_champions function."""

    def test_discover_champions_finds_tagged_runs(self, tmp_path: Path) -> None:
        from minivess.pipeline.deploy_champion_discovery import discover_champions

        mlruns_dir = _make_mock_mlruns(
            tmp_path,
            champion_tags={
                "champion_best_single_fold": "true",
                "champion_metric_name": "val_compound_nsd_cldice",
                "champion_metric_value": "0.85",
            },
        )
        champions = discover_champions(mlruns_dir, experiment_id="1")
        assert len(champions) >= 1
        assert champions[0].run_id == "abc123"

    def test_discover_champions_empty_when_no_tags(self, tmp_path: Path) -> None:
        from minivess.pipeline.deploy_champion_discovery import discover_champions

        mlruns_dir = _make_mock_mlruns(tmp_path, champion_tags={})
        champions = discover_champions(mlruns_dir, experiment_id="1")
        assert len(champions) == 0

    def test_discover_champions_returns_empty_for_missing_experiment(
        self, tmp_path: Path
    ) -> None:
        from minivess.pipeline.deploy_champion_discovery import discover_champions

        mlruns_dir = tmp_path / "mlruns"
        mlruns_dir.mkdir()
        champions = discover_champions(mlruns_dir, experiment_id="999")
        assert champions == []

    def test_discover_champions_extracts_metrics(self, tmp_path: Path) -> None:
        from minivess.pipeline.deploy_champion_discovery import discover_champions

        mlruns_dir = _make_mock_mlruns(
            tmp_path,
            champion_tags={
                "champion_best_cv_mean": "true",
                "champion_metric_name": "dsc",
                "champion_metric_value": "0.82",
            },
            metrics={"val_dice": "0.82", "val_cldice": "0.90"},
        )
        champions = discover_champions(mlruns_dir, experiment_id="1")
        assert len(champions) == 1
        assert "val_dice" in champions[0].metrics

    def test_discover_champions_filters_by_category(self, tmp_path: Path) -> None:
        from minivess.config.deploy_config import ChampionCategory
        from minivess.pipeline.deploy_champion_discovery import discover_champions

        mlruns_dir = _make_mock_mlruns(
            tmp_path,
            champion_tags={
                "champion_best_single_fold": "true",
                "champion_metric_name": "dsc",
                "champion_metric_value": "0.85",
            },
        )
        # Filter by topology — should find nothing since run is tagged as single_fold
        champions = discover_champions(
            mlruns_dir,
            experiment_id="1",
            categories=[ChampionCategory.TOPOLOGY],
        )
        # single_fold maps to "overlap" category, not topology
        assert len(champions) == 0

    def test_discover_champions_multiple_runs(self, tmp_path: Path) -> None:
        from minivess.pipeline.deploy_champion_discovery import discover_champions

        # Create first run
        mlruns_dir = _make_mock_mlruns(
            tmp_path,
            run_id="run1",
            champion_tags={
                "champion_best_single_fold": "true",
                "champion_metric_name": "dsc",
                "champion_metric_value": "0.80",
            },
        )
        # Create second run in same experiment
        run2_dir = mlruns_dir / "1" / "run2" / "tags"
        run2_dir.mkdir(parents=True)
        (run2_dir / "champion_best_cv_mean").write_text("true", encoding="utf-8")
        (run2_dir / "champion_metric_name").write_text("dsc", encoding="utf-8")
        (run2_dir / "champion_metric_value").write_text("0.85", encoding="utf-8")
        meta2 = {"run_id": "run2", "experiment_id": "1", "status": "FINISHED"}
        (mlruns_dir / "1" / "run2" / "meta.yaml").write_text(
            "\n".join(f"{k}: {v}" for k, v in meta2.items()), encoding="utf-8"
        )

        champions = discover_champions(mlruns_dir, experiment_id="1")
        assert len(champions) == 2

    def test_discover_champions_finds_checkpoint_path(self, tmp_path: Path) -> None:
        from minivess.pipeline.deploy_champion_discovery import discover_champions

        mlruns_dir = _make_mock_mlruns(
            tmp_path,
            champion_tags={
                "champion_best_single_fold": "true",
                "champion_metric_name": "dsc",
                "champion_metric_value": "0.85",
            },
        )
        # Create artifacts/checkpoint directory
        artifacts_dir = mlruns_dir / "1" / "abc123" / "artifacts"
        artifacts_dir.mkdir(parents=True)
        ckpt_path = artifacts_dir / "best_checkpoint.pt"
        ckpt_path.write_bytes(b"fake_checkpoint_data")

        champions = discover_champions(mlruns_dir, experiment_id="1")
        assert len(champions) == 1
        assert champions[0].checkpoint_path is not None


# ============================================================================
# Task 3: ONNX export + validation (~8 tests)
# ============================================================================


class TestOnnxExportValidation:
    """Test ONNX export and validation utilities."""

    def test_validate_onnx_model_with_valid_model(self, tmp_path: Path) -> None:
        """Create a tiny ONNX model and validate it."""
        from minivess.pipeline.deploy_onnx_export import validate_onnx_model

        onnx_path = _create_tiny_onnx_model(tmp_path)
        result = validate_onnx_model(onnx_path, input_shape=(1, 1, 4, 4, 4))
        assert result is True

    def test_validate_onnx_model_missing_file(self, tmp_path: Path) -> None:
        from minivess.pipeline.deploy_onnx_export import validate_onnx_model

        result = validate_onnx_model(
            tmp_path / "nonexistent.onnx", input_shape=(1, 1, 4, 4, 4)
        )
        assert result is False

    def test_export_champion_to_onnx_creates_file(self, tmp_path: Path) -> None:
        from minivess.pipeline.deploy_champion_discovery import ChampionModel
        from minivess.pipeline.deploy_onnx_export import export_champion_to_onnx

        # Create mock champion with a simple torch model checkpoint
        ckpt_path = _create_mock_checkpoint(tmp_path / "checkpoints")
        champion = ChampionModel(
            run_id="test_run",
            experiment_id="1",
            category="balanced",
            metrics={"dsc": 0.85},
            checkpoint_path=ckpt_path,
            model_config={"family": "dynunet", "in_channels": 1, "out_channels": 2},
        )
        output_dir = tmp_path / "onnx_output"
        output_dir.mkdir()

        onnx_path = export_champion_to_onnx(
            champion, output_dir, opset_version=17, input_shape=(1, 1, 4, 4, 4)
        )
        assert onnx_path.exists()
        assert onnx_path.suffix == ".onnx"

    def test_export_champion_to_onnx_validates_output(self, tmp_path: Path) -> None:
        from minivess.pipeline.deploy_champion_discovery import ChampionModel
        from minivess.pipeline.deploy_onnx_export import (
            export_champion_to_onnx,
            validate_onnx_model,
        )

        ckpt_path = _create_mock_checkpoint(tmp_path / "checkpoints")
        champion = ChampionModel(
            run_id="test_run",
            experiment_id="1",
            category="balanced",
            metrics={"dsc": 0.85},
            checkpoint_path=ckpt_path,
            model_config={"family": "dynunet", "in_channels": 1, "out_channels": 2},
        )
        output_dir = tmp_path / "onnx_output"
        output_dir.mkdir()

        onnx_path = export_champion_to_onnx(
            champion, output_dir, opset_version=17, input_shape=(1, 1, 4, 4, 4)
        )
        assert validate_onnx_model(onnx_path, input_shape=(1, 1, 4, 4, 4))

    def test_export_champion_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        from minivess.pipeline.deploy_champion_discovery import ChampionModel
        from minivess.pipeline.deploy_onnx_export import export_champion_to_onnx

        champion = ChampionModel(
            run_id="test_run",
            experiment_id="1",
            category="balanced",
            metrics={},
            checkpoint_path=None,
        )
        with pytest.raises(ValueError, match="checkpoint"):
            export_champion_to_onnx(champion, tmp_path, opset_version=17)

    def test_export_uses_champion_run_id_in_filename(self, tmp_path: Path) -> None:
        from minivess.pipeline.deploy_champion_discovery import ChampionModel
        from minivess.pipeline.deploy_onnx_export import export_champion_to_onnx

        ckpt_path = _create_mock_checkpoint(tmp_path / "checkpoints")
        champion = ChampionModel(
            run_id="my_run_42",
            experiment_id="1",
            category="balanced",
            metrics={},
            checkpoint_path=ckpt_path,
            model_config={"family": "dynunet", "in_channels": 1, "out_channels": 2},
        )
        output_dir = tmp_path / "onnx_output"
        output_dir.mkdir()

        onnx_path = export_champion_to_onnx(
            champion, output_dir, opset_version=17, input_shape=(1, 1, 4, 4, 4)
        )
        assert "my_run_42" in onnx_path.name


# ============================================================================
# Task 4: BentoML model import (~10 tests)
# ============================================================================


class TestBentoModelImport:
    """Test BentoML model import from ONNX."""

    def test_get_bento_model_tag_format(self) -> None:
        from minivess.pipeline.deploy_champion_discovery import ChampionModel
        from minivess.serving.bento_model_import import get_bento_model_tag

        champion = ChampionModel(
            run_id="abc123",
            experiment_id="1",
            category="balanced",
            metrics={},
        )
        tag = get_bento_model_tag(champion)
        assert "minivess" in tag.lower()
        assert "abc123" in tag

    def test_get_bento_model_tag_includes_category(self) -> None:
        from minivess.pipeline.deploy_champion_discovery import ChampionModel
        from minivess.serving.bento_model_import import get_bento_model_tag

        champion = ChampionModel(
            run_id="abc123",
            experiment_id="1",
            category="topology",
            metrics={},
        )
        tag = get_bento_model_tag(champion)
        assert "topology" in tag

    def test_import_champion_to_bento_returns_model_info(self, tmp_path: Path) -> None:
        from minivess.pipeline.deploy_champion_discovery import ChampionModel
        from minivess.serving.bento_model_import import import_champion_to_bento

        onnx_path = _create_tiny_onnx_model(tmp_path)
        champion = ChampionModel(
            run_id="test_run",
            experiment_id="1",
            category="balanced",
            metrics={"dsc": 0.85},
        )
        result = import_champion_to_bento(champion, onnx_path)
        assert result is not None
        assert result.tag is not None

    def test_import_champion_stores_metadata(self, tmp_path: Path) -> None:
        from minivess.pipeline.deploy_champion_discovery import ChampionModel
        from minivess.serving.bento_model_import import import_champion_to_bento

        onnx_path = _create_tiny_onnx_model(tmp_path)
        champion = ChampionModel(
            run_id="test_run",
            experiment_id="1",
            category="balanced",
            metrics={"dsc": 0.85, "cldice": 0.90},
        )
        result = import_champion_to_bento(champion, onnx_path)
        assert result.metadata is not None
        assert result.metadata["champion_category"] == "balanced"
        assert result.metadata["run_id"] == "test_run"

    def test_import_missing_onnx_raises(self, tmp_path: Path) -> None:
        from minivess.pipeline.deploy_champion_discovery import ChampionModel
        from minivess.serving.bento_model_import import import_champion_to_bento

        champion = ChampionModel(
            run_id="test_run",
            experiment_id="1",
            category="balanced",
            metrics={},
        )
        with pytest.raises(FileNotFoundError):
            import_champion_to_bento(champion, tmp_path / "nonexistent.onnx")


# ============================================================================
# Task 5: BentoML service v2 (ONNX-backed) (~12 tests)
# ============================================================================


class TestBentoServiceV2:
    """Test ONNX-backed BentoML service."""

    def test_service_class_exists(self) -> None:
        from minivess.serving.bento_service import OnnxSegmentationService

        assert OnnxSegmentationService is not None

    def test_service_has_predict_method(self) -> None:
        from minivess.serving.bento_service import OnnxSegmentationService

        assert hasattr(OnnxSegmentationService, "predict")

    def test_service_has_health_method(self) -> None:
        from minivess.serving.bento_service import OnnxSegmentationService

        assert hasattr(OnnxSegmentationService, "health")

    def test_predict_returns_segmentation_and_probs(self, tmp_path: Path) -> None:
        """Test predict with a mocked ONNX session."""
        from minivess.serving.bento_service import OnnxSegmentationService

        _create_tiny_onnx_model(tmp_path)
        service = OnnxSegmentationService.__new__(OnnxSegmentationService)
        # Mock the inference engine
        service._engine = _MockOnnxEngine()
        service._model_tag = "test-model"

        volume = np.random.rand(1, 1, 4, 4, 4).astype(np.float32)
        result = service.predict(volume)
        assert "segmentation" in result
        assert "probabilities" in result
        assert "shape" in result

    def test_health_returns_status(self) -> None:
        from minivess.serving.bento_service import OnnxSegmentationService

        service = OnnxSegmentationService.__new__(OnnxSegmentationService)
        service._model_tag = "test-model"
        service._engine = None
        result = service.health()
        assert result["status"] == "healthy"
        assert "model" in result

    def test_predict_validates_input_dimensions(self) -> None:
        from minivess.serving.bento_service import OnnxSegmentationService

        service = OnnxSegmentationService.__new__(OnnxSegmentationService)
        service._engine = _MockOnnxEngine()
        service._model_tag = "test-model"

        # 2D input should be rejected
        with pytest.raises(ValueError, match="shape"):
            service.predict(np.zeros((4, 4), dtype=np.float32))


# ============================================================================
# Task 6: Deployment artifact generation (~10 tests)
# ============================================================================


class TestDeployArtifacts:
    """Test deployment artifact generation."""

    def test_generate_bentofile(self, tmp_path: Path) -> None:
        from minivess.serving.deploy_artifacts import generate_bentofile

        path = generate_bentofile(
            service_name="minivess-segmentation",
            models=["minivess-balanced:latest"],
            output_dir=tmp_path,
        )
        assert path.exists()
        assert path.name == "bentofile.yaml"
        content = path.read_text(encoding="utf-8")
        assert "minivess-segmentation" in content

    def test_generate_docker_compose(self, tmp_path: Path) -> None:
        from minivess.serving.deploy_artifacts import generate_docker_compose

        services = [
            {"name": "segmentation", "port": 3000, "model_tag": "balanced:latest"}
        ]
        path = generate_docker_compose(services, tmp_path)
        assert path.exists()
        assert path.name == "docker-compose.yaml"
        content = path.read_text(encoding="utf-8")
        assert "healthcheck" in content

    def test_generate_deployment_readme(self, tmp_path: Path) -> None:
        from minivess.pipeline.deploy_champion_discovery import ChampionModel
        from minivess.serving.deploy_artifacts import generate_deployment_readme

        champions = [
            ChampionModel(
                run_id="abc123",
                experiment_id="1",
                category="balanced",
                metrics={"dsc": 0.85},
            )
        ]
        path = generate_deployment_readme(champions, tmp_path)
        assert path.exists()
        assert path.name == "DEPLOY_README.md"
        content = path.read_text(encoding="utf-8")
        assert "balanced" in content

    def test_docker_compose_has_health_checks(self, tmp_path: Path) -> None:
        from minivess.serving.deploy_artifacts import generate_docker_compose

        services = [
            {"name": "segmentation", "port": 3000, "model_tag": "balanced:latest"}
        ]
        path = generate_docker_compose(services, tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "healthcheck" in content
        assert "interval" in content

    def test_bentofile_includes_models(self, tmp_path: Path) -> None:
        from minivess.serving.deploy_artifacts import generate_bentofile

        models = ["model-a:v1", "model-b:v2"]
        path = generate_bentofile("svc", models, tmp_path)
        content = path.read_text(encoding="utf-8")
        for model in models:
            assert model in content

    def test_generate_bentofile_includes_python_packages(self, tmp_path: Path) -> None:
        from minivess.serving.deploy_artifacts import generate_bentofile

        path = generate_bentofile("svc", ["m:v1"], tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "onnxruntime" in content


# ============================================================================
# Task 7: Registry promotion integration (~8 tests)
# ============================================================================


class TestDeployPromotion:
    """Test deployment promotion integration."""

    def test_promote_champion_for_deploy_approved(self) -> None:
        from minivess.observability.model_registry import (
            ModelRegistry,
            PromotionCriteria,
        )
        from minivess.pipeline.deploy_champion_discovery import ChampionModel
        from minivess.pipeline.deploy_promotion import promote_champion_for_deploy

        registry = ModelRegistry()
        registry.register_version(
            "minivess-balanced", "1.0.0", {"dsc": 0.85, "cldice": 0.90}
        )
        champion = ChampionModel(
            run_id="abc123",
            experiment_id="1",
            category="balanced",
            metrics={"dsc": 0.85, "cldice": 0.90},
        )
        criteria = PromotionCriteria(min_thresholds={"dsc": 0.80})
        result = promote_champion_for_deploy(champion, registry, criteria)
        assert result.approved is True

    def test_promote_champion_for_deploy_rejected(self) -> None:
        from minivess.observability.model_registry import (
            ModelRegistry,
            PromotionCriteria,
        )
        from minivess.pipeline.deploy_champion_discovery import ChampionModel
        from minivess.pipeline.deploy_promotion import promote_champion_for_deploy

        registry = ModelRegistry()
        registry.register_version(
            "minivess-balanced", "1.0.0", {"dsc": 0.50, "cldice": 0.40}
        )
        champion = ChampionModel(
            run_id="abc123",
            experiment_id="1",
            category="balanced",
            metrics={"dsc": 0.50, "cldice": 0.40},
        )
        criteria = PromotionCriteria(min_thresholds={"dsc": 0.80})
        result = promote_champion_for_deploy(champion, registry, criteria)
        assert result.approved is False

    def test_create_deployment_audit_trail(self, tmp_path: Path) -> None:
        from minivess.pipeline.deploy_champion_discovery import ChampionModel
        from minivess.pipeline.deploy_promotion import create_deployment_audit_trail

        champion = ChampionModel(
            run_id="abc123",
            experiment_id="1",
            category="balanced",
            metrics={"dsc": 0.85},
        )
        trail = create_deployment_audit_trail(
            champion=champion,
            onnx_path=tmp_path / "model.onnx",
            bento_tag="minivess-balanced:abc123",
            promotion_approved=True,
        )
        assert "run_id" in trail
        assert "timestamp" in trail
        assert trail["promotion_approved"] is True

    def test_audit_trail_has_utc_timestamp(self, tmp_path: Path) -> None:
        from minivess.pipeline.deploy_champion_discovery import ChampionModel
        from minivess.pipeline.deploy_promotion import create_deployment_audit_trail

        champion = ChampionModel(
            run_id="abc123",
            experiment_id="1",
            category="balanced",
            metrics={},
        )
        trail = create_deployment_audit_trail(
            champion=champion,
            onnx_path=tmp_path / "model.onnx",
            bento_tag="tag",
            promotion_approved=True,
        )
        # ISO format timestamp
        assert "T" in trail["timestamp"]


# ============================================================================
# Task 8: MONAI Deploy MAP application (~12 tests)
# ============================================================================


class TestMonaiDeployApp:
    """Test MONAI Deploy MAP application."""

    def test_monai_deploy_app_class_exists(self) -> None:
        from minivess.serving.monai_deploy_app import MiniVessSegApp

        assert MiniVessSegApp is not None

    def test_monai_deploy_inference_operator_exists(self) -> None:
        from minivess.serving.monai_deploy_app import MiniVessInferenceOperator

        assert MiniVessInferenceOperator is not None

    def test_inference_operator_init(self, tmp_path: Path) -> None:
        from minivess.serving.monai_deploy_app import MiniVessInferenceOperator

        onnx_path = _create_tiny_onnx_model(tmp_path)
        operator = MiniVessInferenceOperator(model_path=onnx_path)
        assert operator.model_path == onnx_path

    def test_inference_operator_process(self, tmp_path: Path) -> None:
        from minivess.serving.monai_deploy_app import MiniVessInferenceOperator

        onnx_path = _create_tiny_onnx_model(tmp_path)
        operator = MiniVessInferenceOperator(model_path=onnx_path)

        # Simulate input volume
        volume = np.random.rand(1, 1, 4, 4, 4).astype(np.float32)
        result = operator.process(volume)
        assert "segmentation" in result
        assert isinstance(result["segmentation"], np.ndarray)

    def test_generate_map_manifest(self, tmp_path: Path) -> None:
        from minivess.serving.monai_deploy_app import generate_map_manifest

        manifest = generate_map_manifest(
            app_name="MiniVessSegApp",
            version="1.0.0",
            model_name="minivess-balanced",
        )
        assert manifest["api-version"] == "1.0"
        assert manifest["application"]["name"] == "MiniVessSegApp"

    def test_generate_map_main_py(self, tmp_path: Path) -> None:
        from minivess.serving.monai_deploy_app import generate_map_main_py

        content = generate_map_main_py()
        assert "if __name__" in content
        assert "MiniVessSegApp" in content

    def test_app_compose_returns_operators(self) -> None:
        from minivess.serving.monai_deploy_app import MiniVessSegApp

        app = MiniVessSegApp(model_path=Path("/tmp/model.onnx"))
        operators = app.compose()
        assert len(operators) >= 1

    def test_map_manifest_has_required_fields(self) -> None:
        from minivess.serving.monai_deploy_app import generate_map_manifest

        manifest = generate_map_manifest("App", "1.0.0", "model")
        required = {"api-version", "application", "resources"}
        assert required.issubset(manifest.keys())


# ============================================================================
# Task 9: Deploy Prefect Flow assembly (~20 tests)
# ============================================================================


class TestDeployResult:
    """Test DeployResult dataclass."""

    def test_deploy_result_creation(self) -> None:
        from minivess.orchestration.deploy_flow import DeployResult

        result = DeployResult(
            champions=[],
            onnx_paths={},
            bento_tags={},
            artifacts_dir=Path("/tmp/artifacts"),
            promotion_results={},
        )
        assert result.artifacts_dir == Path("/tmp/artifacts")

    def test_deploy_result_to_summary(self) -> None:
        from minivess.orchestration.deploy_flow import DeployResult

        result = DeployResult(
            champions=[],
            onnx_paths={"balanced": Path("/tmp/a.onnx")},
            bento_tags={"balanced": "tag:v1"},
            artifacts_dir=Path("/tmp/artifacts"),
            promotion_results={"balanced": True},
        )
        summary = result.to_summary()
        assert isinstance(summary, dict)
        assert "onnx_models" in summary
        assert "bento_models" in summary


class TestDeployFlowTasks:
    """Test individual deploy flow tasks."""

    def test_discover_task_callable(self) -> None:
        from minivess.orchestration.deploy_flow import discover_task

        assert callable(discover_task)

    def test_export_task_callable(self) -> None:
        from minivess.orchestration.deploy_flow import export_task

        assert callable(export_task)

    def test_import_task_callable(self) -> None:
        from minivess.orchestration.deploy_flow import import_task

        assert callable(import_task)

    def test_generate_artifacts_task_callable(self) -> None:
        from minivess.orchestration.deploy_flow import generate_artifacts_task

        assert callable(generate_artifacts_task)

    def test_promote_task_callable(self) -> None:
        from minivess.orchestration.deploy_flow import promote_task

        assert callable(promote_task)

    def test_deploy_flow_callable(self) -> None:
        from minivess.orchestration.deploy_flow import deploy_flow

        assert callable(deploy_flow)


class TestDeployFlowExecution:
    """Test deploy flow execution with mocks."""

    def test_discover_task_returns_champions(self, tmp_path: Path) -> None:
        from minivess.config.deploy_config import DeployConfig
        from minivess.orchestration.deploy_flow import discover_task

        mlruns_dir = _make_mock_mlruns(
            tmp_path,
            champion_tags={
                "champion_best_single_fold": "true",
                "champion_metric_name": "dsc",
                "champion_metric_value": "0.85",
            },
        )
        config = DeployConfig(
            mlruns_dir=mlruns_dir,
            output_dir=tmp_path / "out",
        )
        champions = discover_task(config, experiment_id="1")
        assert isinstance(champions, list)

    def test_export_task_produces_onnx(self, tmp_path: Path) -> None:
        from minivess.orchestration.deploy_flow import export_task
        from minivess.pipeline.deploy_champion_discovery import ChampionModel

        ckpt_path = _create_mock_checkpoint(tmp_path / "ckpt")
        champion = ChampionModel(
            run_id="run1",
            experiment_id="1",
            category="balanced",
            metrics={},
            checkpoint_path=ckpt_path,
            model_config={"family": "dynunet", "in_channels": 1, "out_channels": 2},
        )
        output_dir = tmp_path / "onnx"
        output_dir.mkdir()
        result = export_task(champion, output_dir, opset_version=17)
        assert isinstance(result, Path)
        assert result.exists()

    def test_deploy_flow_handles_no_champions(self, tmp_path: Path) -> None:
        """Flow should complete gracefully when no champions found."""
        from minivess.config.deploy_config import DeployConfig
        from minivess.orchestration.deploy_flow import deploy_flow

        mlruns_dir = _make_mock_mlruns(tmp_path, champion_tags={})
        config = DeployConfig(
            mlruns_dir=mlruns_dir,
            output_dir=tmp_path / "out",
        )
        result = deploy_flow(config, experiment_id="1")
        assert result is not None
        assert len(result.champions) == 0

    def test_deploy_flow_e2e_with_mock_champion(self, tmp_path: Path) -> None:
        """Full flow with a mock champion."""
        from minivess.config.deploy_config import DeployConfig
        from minivess.orchestration.deploy_flow import deploy_flow

        ckpt_path = _create_mock_checkpoint(tmp_path / "ckpt")
        mlruns_dir = _make_mock_mlruns(
            tmp_path,
            champion_tags={
                "champion_best_single_fold": "true",
                "champion_metric_name": "dsc",
                "champion_metric_value": "0.85",
            },
        )
        # Place checkpoint in the expected artifacts dir
        artifacts_dir = mlruns_dir / "1" / "abc123" / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy(ckpt_path, artifacts_dir / "best_checkpoint.pt")

        config = DeployConfig(
            mlruns_dir=mlruns_dir,
            output_dir=tmp_path / "out",
        )
        result = deploy_flow(config, experiment_id="1")
        assert result is not None
        assert len(result.champions) >= 1


# ============================================================================
# Task 10: SDD deployment decision nodes (~8 tests)
# ============================================================================


class TestSddDecisionNodes:
    """Test SDD deployment decision YAML files."""

    @pytest.fixture()
    def decisions_dir(self) -> Path:
        # Discover project root from this test file's location
        repo_root = Path(__file__).resolve().parents[3]
        return repo_root / "docs" / "prd" / "decisions" / "L4-deployment"

    EXPECTED_DECISIONS = [
        "serving-framework.decision.yaml",
        "model-format.decision.yaml",
        "container-strategy.decision.yaml",
        "clinical-packaging.decision.yaml",
        "orchestration-deploy.decision.yaml",
        "model-promotion.decision.yaml",
    ]

    def test_decision_files_exist(self, decisions_dir: Path) -> None:
        for filename in self.EXPECTED_DECISIONS:
            assert (decisions_dir / filename).exists(), f"Missing: {filename}"

    def test_decision_files_are_valid_yaml(self, decisions_dir: Path) -> None:
        import yaml

        for filename in self.EXPECTED_DECISIONS:
            path = decisions_dir / filename
            with path.open(encoding="utf-8") as f:
                data = yaml.safe_load(f)
            assert isinstance(data, dict), f"{filename} is not a valid YAML dict"

    def test_decision_files_have_required_fields(self, decisions_dir: Path) -> None:
        import yaml

        required = {"decision_id", "title", "description", "decision_level", "options"}
        for filename in self.EXPECTED_DECISIONS:
            path = decisions_dir / filename
            with path.open(encoding="utf-8") as f:
                data = yaml.safe_load(f)
            missing = required - data.keys()
            assert not missing, f"{filename} missing fields: {missing}"

    def test_decision_files_have_options_with_probabilities(
        self, decisions_dir: Path
    ) -> None:
        import yaml

        for filename in self.EXPECTED_DECISIONS:
            path = decisions_dir / filename
            with path.open(encoding="utf-8") as f:
                data = yaml.safe_load(f)
            options = data["options"]
            assert len(options) >= 2, f"{filename} needs at least 2 options"
            total = sum(opt.get("prior_probability", 0) for opt in options)
            assert abs(total - 1.0) < 0.05, (
                f"{filename} probabilities sum to {total:.2f}, expected ~1.0"
            )

    def test_decision_level_is_l4(self, decisions_dir: Path) -> None:
        import yaml

        for filename in self.EXPECTED_DECISIONS:
            path = decisions_dir / filename
            with path.open(encoding="utf-8") as f:
                data = yaml.safe_load(f)
            assert "L4" in data["decision_level"], f"{filename} should be L4 level"

    def test_decision_files_have_references(self, decisions_dir: Path) -> None:
        import yaml

        for filename in self.EXPECTED_DECISIONS:
            path = decisions_dir / filename
            with path.open(encoding="utf-8") as f:
                data = yaml.safe_load(f)
            assert "references" in data, f"{filename} missing references"
            assert len(data["references"]) >= 1, (
                f"{filename} needs at least 1 reference"
            )


# ============================================================================
# Helpers
# ============================================================================


def _create_tiny_onnx_model(output_dir: Path) -> Path:
    """Create a minimal ONNX model for testing (identity-like)."""
    import onnx
    from onnx import TensorProto, helper

    # Simple model: input -> output (identity via Relu for valid graph)
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4, 4])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 4, 4, 4])

    # Use a Conv to go from 1 channel to 2 channels
    weight_shape = [2, 1, 1, 1, 1]
    weight_data = np.ones(weight_shape, dtype=np.float32)
    weight_init = helper.make_tensor(
        "conv_weight", TensorProto.FLOAT, weight_shape, weight_data.flatten().tolist()
    )

    conv_node = helper.make_node(
        "Conv", ["input", "conv_weight"], ["output"], kernel_shape=[1, 1, 1]
    )

    graph = helper.make_graph(
        [conv_node], "test_model", [X], [Y], initializer=[weight_init]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "test_model.onnx"
    onnx.save(model, str(onnx_path))
    return onnx_path


def _create_mock_checkpoint(output_dir: Path) -> Path:
    """Create a mock PyTorch checkpoint with a tiny model."""
    import torch
    import torch.nn as nn

    class TinyModel(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv3d(1, 2, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    output_dir.mkdir(parents=True, exist_ok=True)
    model = TinyModel()
    ckpt_path = output_dir / "best_checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    return ckpt_path


class _MockOnnxEngine:
    """Mock ONNX inference engine for testing."""

    def predict(self, volume: np.ndarray) -> Any:
        from minivess.serving.onnx_inference import OnnxPrediction

        b, c, d, h, w = volume.shape
        probs = np.random.rand(b, 2, d, h, w).astype(np.float32)
        labels = probs.argmax(axis=1)
        return OnnxPrediction(
            segmentation=labels, probabilities=probs, shape=list(labels.shape)
        )

    def get_metadata(self) -> Any:
        from minivess.serving.onnx_inference import OnnxModelMetadata

        return OnnxModelMetadata(inputs=[], outputs=[])
