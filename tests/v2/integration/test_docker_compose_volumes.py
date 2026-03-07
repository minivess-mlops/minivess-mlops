"""Tests for T-04: Docker volume audit — docker-compose.flows.yml.

Uses yaml.safe_load() for all YAML parsing — NO regex (CLAUDE.md Rule #16).
Verifies that every flow service has the required volume mounts declared.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_COMPOSE_PATH = Path("deployment/docker-compose.flows.yml")


def _load_compose() -> dict:
    return yaml.safe_load(_COMPOSE_PATH.read_text(encoding="utf-8"))


def _service_volume_names(services: dict, service_name: str) -> list[str]:
    """Extract volume names (the source part) from a service's volumes list."""
    vols = services.get(service_name, {}).get("volumes", [])
    names = []
    for v in vols:
        # Volume entries are either "name:/container/path" or "name:/container/path:mode"
        source = str(v).split(":")[0]
        names.append(source)
    return names


# ---------------------------------------------------------------------------
# Acquisition service (was MISSING entirely)
# ---------------------------------------------------------------------------


class TestAcquisitionService:
    def test_acquisition_service_exists(self) -> None:
        compose = _load_compose()
        assert "acquisition" in compose["services"], (
            "acquisition service is missing from docker-compose.flows.yml"
        )

    def test_acquisition_has_raw_data_mount(self) -> None:
        compose = _load_compose()
        vol_names = _service_volume_names(compose["services"], "acquisition")
        assert "raw_data" in vol_names, (
            "acquisition service missing raw_data volume mount"
        )

    def test_acquisition_has_logs_mount(self) -> None:
        compose = _load_compose()
        vol_names = _service_volume_names(compose["services"], "acquisition")
        assert "logs_data" in vol_names, (
            "acquisition service missing logs_data volume mount"
        )

    def test_acquisition_image_set(self) -> None:
        compose = _load_compose()
        svc = compose["services"].get("acquisition", {})
        assert svc.get("image") == "minivess-acquisition:latest"

    def test_acquisition_image_correct(self) -> None:
        """acquisition service image must be minivess-acquisition:latest."""
        compose = _load_compose()
        svc = compose["services"].get("acquisition", {})
        assert svc.get("image") == "minivess-acquisition:latest", (
            f"acquisition image is {svc.get('image')!r}, expected 'minivess-acquisition:latest'"
        )

    def test_acquisition_env_has_output_dir(self) -> None:
        """acquisition service must have ACQUISITION_OUTPUT_DIR in environment."""
        compose = _load_compose()
        svc = compose["services"].get("acquisition", {})
        env = svc.get("environment", {})
        assert "ACQUISITION_OUTPUT_DIR" in env, (
            "acquisition service missing ACQUISITION_OUTPUT_DIR environment variable. "
            "Add ACQUISITION_OUTPUT_DIR: /app/data/raw to the environment section."
        )

    def test_raw_data_volume_declared(self) -> None:
        """raw_data named volume must be declared at top level."""
        compose = _load_compose()
        top_volumes = compose.get("volumes", {})
        assert "raw_data" in top_volumes, (
            "raw_data named volume not declared in docker-compose.flows.yml volumes section. "
            "Add raw_data: under the top-level volumes key."
        )


# ---------------------------------------------------------------------------
# Data service
# ---------------------------------------------------------------------------


class TestDataService:
    def test_data_has_mlruns_mount(self) -> None:
        compose = _load_compose()
        vol_names = _service_volume_names(compose["services"], "data")
        assert "mlruns_data" in vol_names, "data service missing mlruns_data volume"

    def test_data_has_configs_splits_mount(self) -> None:
        compose = _load_compose()
        vol_names = _service_volume_names(compose["services"], "data")
        assert "configs_splits" in vol_names, (
            "data service missing configs_splits volume mount"
        )


# ---------------------------------------------------------------------------
# Train service
# ---------------------------------------------------------------------------


class TestTrainService:
    def test_train_has_mlruns_mount(self) -> None:
        compose = _load_compose()
        vol_names = _service_volume_names(compose["services"], "train")
        assert "mlruns_data" in vol_names, "train service missing mlruns_data volume"

    def test_train_has_configs_splits_mount(self) -> None:
        compose = _load_compose()
        vol_names = _service_volume_names(compose["services"], "train")
        assert "configs_splits" in vol_names, (
            "train service missing configs_splits volume"
        )

    def test_train_has_logs_mount(self) -> None:
        compose = _load_compose()
        vol_names = _service_volume_names(compose["services"], "train")
        assert "logs_data" in vol_names, "train service missing logs_data volume"

    def test_train_has_no_tmp_paths(self) -> None:
        compose = _load_compose()
        vols = compose["services"].get("train", {}).get("volumes", [])
        for v in vols:
            container_path = str(v).split(":")[1] if ":" in str(v) else str(v)
            assert not container_path.startswith("/tmp"), (
                f"train service has a volume mount into /tmp: {v}"
            )

    def test_train_has_checkpoint_dir_env(self) -> None:
        compose = _load_compose()
        env = compose["services"].get("train", {}).get("environment", {})
        assert "CHECKPOINT_DIR" in env, (
            "train service missing CHECKPOINT_DIR environment variable"
        )


# ---------------------------------------------------------------------------
# Post-training service
# ---------------------------------------------------------------------------


class TestPostTrainingService:
    def test_post_training_has_mlruns_mount(self) -> None:
        compose = _load_compose()
        vol_names = _service_volume_names(compose["services"], "post_training")
        assert "mlruns_data" in vol_names, (
            "post_training service missing mlruns_data volume"
        )

    def test_post_training_has_output_mount(self) -> None:
        compose = _load_compose()
        vol_names = _service_volume_names(compose["services"], "post_training")
        assert "post_training_out" in vol_names, (
            "post_training service missing post_training_out volume"
        )


# ---------------------------------------------------------------------------
# Analyze service (was ZERO volumes)
# ---------------------------------------------------------------------------


class TestAnalyzeService:
    def test_analyze_has_mlruns_mount(self) -> None:
        compose = _load_compose()
        vol_names = _service_volume_names(compose["services"], "analyze")
        assert "mlruns_data" in vol_names, "analyze service missing mlruns_data volume"

    def test_analyze_has_outputs_mount(self) -> None:
        compose = _load_compose()
        vol_names = _service_volume_names(compose["services"], "analyze")
        assert "outputs_analysis" in vol_names, (
            "analyze service missing outputs_analysis volume"
        )

    def test_analyze_not_zero_volumes(self) -> None:
        compose = _load_compose()
        vols = compose["services"].get("analyze", {}).get("volumes", [])
        assert len(vols) > 0, "analyze service has ZERO volume mounts"


# ---------------------------------------------------------------------------
# Deploy service (was ZERO volumes)
# ---------------------------------------------------------------------------


class TestDeployService:
    def test_deploy_has_mlruns_mount(self) -> None:
        compose = _load_compose()
        vol_names = _service_volume_names(compose["services"], "deploy")
        assert "mlruns_data" in vol_names, "deploy service missing mlruns_data volume"

    def test_deploy_has_outputs_mount(self) -> None:
        compose = _load_compose()
        vol_names = _service_volume_names(compose["services"], "deploy")
        assert "outputs_deploy" in vol_names, (
            "deploy service missing outputs_deploy volume"
        )

    def test_deploy_has_bentoml_mount(self) -> None:
        compose = _load_compose()
        vol_names = _service_volume_names(compose["services"], "deploy")
        assert "bentoml_store" in vol_names, (
            "deploy service missing bentoml_store volume mount"
        )

    def test_deploy_not_zero_volumes(self) -> None:
        compose = _load_compose()
        vols = compose["services"].get("deploy", {}).get("volumes", [])
        assert len(vols) > 0, "deploy service has ZERO volume mounts"


# ---------------------------------------------------------------------------
# Dashboard service (was ZERO volumes)
# ---------------------------------------------------------------------------


class TestDashboardService:
    def test_dashboard_has_mlruns_mount(self) -> None:
        compose = _load_compose()
        vol_names = _service_volume_names(compose["services"], "dashboard")
        assert "mlruns_data" in vol_names, (
            "dashboard service missing mlruns_data volume"
        )

    def test_dashboard_has_outputs_mount(self) -> None:
        compose = _load_compose()
        vol_names = _service_volume_names(compose["services"], "dashboard")
        assert "outputs_dashboard" in vol_names, (
            "dashboard service missing outputs_dashboard volume"
        )

    def test_dashboard_not_zero_volumes(self) -> None:
        compose = _load_compose()
        vols = compose["services"].get("dashboard", {}).get("volumes", [])
        assert len(vols) > 0, "dashboard service has ZERO volume mounts"


# ---------------------------------------------------------------------------
# QA service (was ZERO volumes)
# ---------------------------------------------------------------------------


class TestQaService:
    def test_qa_has_mlruns_mount(self) -> None:
        compose = _load_compose()
        vol_names = _service_volume_names(compose["services"], "qa")
        assert "mlruns_data" in vol_names, "qa service missing mlruns_data volume"

    def test_qa_has_dashboard_output_mount(self) -> None:
        compose = _load_compose()
        vol_names = _service_volume_names(compose["services"], "qa")
        assert "outputs_dashboard" in vol_names, (
            "qa service missing outputs_dashboard volume for QA reports"
        )

    def test_qa_not_zero_volumes(self) -> None:
        compose = _load_compose()
        vols = compose["services"].get("qa", {}).get("volumes", [])
        assert len(vols) > 0, "qa service has ZERO volume mounts"


# ---------------------------------------------------------------------------
# Top-level volumes declaration consistency
# ---------------------------------------------------------------------------


class TestVolumeDeclarations:
    def test_all_referenced_volumes_declared(self) -> None:
        compose = _load_compose()
        declared_volumes = set(compose.get("volumes", {}).keys())
        for svc_name, svc_config in compose["services"].items():
            for vol_entry in svc_config.get("volumes", []):
                vol_name = str(vol_entry).split(":")[0]
                # Skip bind mounts (start with . or /)
                if vol_name.startswith((".", "/")):
                    continue
                assert vol_name in declared_volumes, (
                    f"Service '{svc_name}' references volume '{vol_name}' "
                    f"but it is not declared in the top-level volumes: section"
                )

    def test_no_service_has_zero_volumes(self) -> None:
        compose = _load_compose()
        zero_volume_services = []
        for svc_name, svc_config in compose["services"].items():
            vols = svc_config.get("volumes", [])
            if len(vols) == 0:
                zero_volume_services.append(svc_name)
        assert not zero_volume_services, (
            f"Services with ZERO volume mounts (all outputs will be lost on container exit): "
            f"{zero_volume_services}"
        )


# ---------------------------------------------------------------------------
# T-25: Annotation service Docker infrastructure
# ---------------------------------------------------------------------------


class TestAnnotationDockerInfrastructure:
    def test_annotation_service_exists(self) -> None:
        """'annotation' service must exist in docker-compose.flows.yml."""
        compose = _load_compose()
        assert "annotation" in compose.get("services", {}), (
            "'annotation' service not found in docker-compose.flows.yml. "
            "Add it with bentoml_store, data_cache, and mlruns_data volumes."
        )

    def test_annotation_has_bentoml_mount(self) -> None:
        """annotation service must mount bentoml_store volume."""
        compose = _load_compose()
        vols = _service_volume_names(compose["services"], "annotation")
        assert "bentoml_store" in vols, (
            f"annotation service must mount 'bentoml_store' to read BentoML models. "
            f"Found volumes: {vols}"
        )

    def test_annotation_has_data_cache_mount(self) -> None:
        """annotation service must mount data_cache volume for input volumes."""
        compose = _load_compose()
        vols = _service_volume_names(compose["services"], "annotation")
        assert "data_cache" in vols, (
            f"annotation service must mount 'data_cache' for input data access. "
            f"Found volumes: {vols}"
        )

    def test_annotation_has_mlruns_mount(self) -> None:
        """annotation service must mount mlruns_data to log annotation sessions."""
        compose = _load_compose()
        vols = _service_volume_names(compose["services"], "annotation")
        assert "mlruns_data" in vols, (
            f"annotation service must mount 'mlruns_data' for MLflow tracking. "
            f"Found volumes: {vols}"
        )

    def test_annotation_has_bentoml_env(self) -> None:
        """annotation service must set BENTOML_HOME environment variable."""
        compose = _load_compose()
        svc = compose["services"].get("annotation", {})
        env = svc.get("environment", {})
        # environment can be a dict or a list of KEY=VAL strings
        if isinstance(env, dict):
            env_keys = set(env.keys())
        else:
            env_keys = {str(e).split("=")[0] for e in env}
        assert "BENTOML_HOME" in env_keys, (
            f"annotation service must set BENTOML_HOME env var. "
            f"Found env keys: {env_keys}"
        )

    def test_dockerfile_annotation_exists(self) -> None:
        """deployment/docker/Dockerfile.annotation must exist."""
        dockerfile = Path("deployment/docker/Dockerfile.annotation")
        assert dockerfile.exists(), (
            "deployment/docker/Dockerfile.annotation does not exist. "
            "Create it FROM minivess-base:latest with annotation flow CMD."
        )

    def test_dockerfile_annotation_has_flow_label(self) -> None:
        """Dockerfile.annotation must have LABEL flow=annotation."""
        dockerfile = Path("deployment/docker/Dockerfile.annotation")
        if not dockerfile.exists():
            return
        content = dockerfile.read_text(encoding="utf-8")
        assert "flow" in content and "annotation" in content, (
            "Dockerfile.annotation must include a LABEL identifying the flow. "
            'Add: LABEL flow="annotation"'
        )
