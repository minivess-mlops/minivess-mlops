"""Tests for Prefect Server + Work Pools (#281).

Covers:
- Work pool configuration
- Flow deployment mapping
- Prefect compat work pool awareness
- Docker compose Prefect services
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestWorkPoolConfig:
    """Test work pool configuration."""

    def test_work_pools_yaml_exists(self) -> None:
        path = PROJECT_ROOT / "deployment" / "prefect" / "work-pools.yaml"
        assert path.exists()

    def test_work_pools_yaml_valid(self) -> None:
        import yaml

        path = PROJECT_ROOT / "deployment" / "prefect" / "work-pools.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "work_pools" in data

    def test_cpu_pool_defined(self) -> None:
        import yaml

        path = PROJECT_ROOT / "deployment" / "prefect" / "work-pools.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        pool_names = [p["name"] for p in data["work_pools"]]
        assert "cpu-pool" in pool_names

    def test_gpu_pool_defined(self) -> None:
        import yaml

        path = PROJECT_ROOT / "deployment" / "prefect" / "work-pools.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        pool_names = [p["name"] for p in data["work_pools"]]
        assert "gpu-pool" in pool_names


class TestFlowDeployments:
    """Test flow deployment configuration."""

    def test_deployments_module_exists(self) -> None:
        from minivess.orchestration import deployments

        assert hasattr(deployments, "FLOW_WORK_POOL_MAP")

    def test_flow_pool_mapping(self) -> None:
        from minivess.orchestration.deployments import FLOW_WORK_POOL_MAP

        assert FLOW_WORK_POOL_MAP["train"] == "gpu-pool"
        assert FLOW_WORK_POOL_MAP["data"] == "cpu-pool"
        assert FLOW_WORK_POOL_MAP["analyze"] == "cpu-pool"

    def test_get_flow_deployment_config(self) -> None:
        from minivess.orchestration.deployments import get_flow_deployment_config

        config = get_flow_deployment_config("train")
        assert config["work_pool"] == "gpu-pool"
        assert "flow_name" in config


class TestPrefectCompatWorkPools:
    """Test deployments.py get_work_pool function."""

    def test_get_work_pool_for_flow(self) -> None:
        from minivess.orchestration.deployments import get_work_pool

        pool = get_work_pool("train")
        assert pool == "gpu-pool"

    def test_get_work_pool_default(self) -> None:
        from minivess.orchestration.deployments import get_work_pool

        pool = get_work_pool("unknown_flow")
        assert pool == "cpu-pool"


class TestDockerComposePrefect:
    """Test Prefect services in docker-compose."""

    def test_docker_compose_has_prefect(self) -> None:
        import yaml

        path = PROJECT_ROOT / "deployment" / "docker-compose.yml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        # Should have prefect-server in services
        assert "prefect-server" in data["services"]
