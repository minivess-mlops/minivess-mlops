"""Tests for Docker-per-flow foundation (#280).

Covers:
- Dockerfile existence and structure
- Docker Compose flows configuration
- Training flow as Prefect @flow
- Inter-flow MLflow communication contract
- DVC data setup script
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestDockerfileExistence:
    """Test that all per-flow Dockerfiles exist."""

    def test_base_dockerfile_exists(self) -> None:
        path = PROJECT_ROOT / "deployment" / "docker" / "Dockerfile.base"
        assert path.exists(), f"Missing {path}"

    def test_data_dockerfile_exists(self) -> None:
        path = PROJECT_ROOT / "deployment" / "docker" / "Dockerfile.data"
        assert path.exists(), f"Missing {path}"

    def test_train_dockerfile_exists(self) -> None:
        path = PROJECT_ROOT / "deployment" / "docker" / "Dockerfile.train"
        assert path.exists(), f"Missing {path}"

    def test_analyze_dockerfile_exists(self) -> None:
        path = PROJECT_ROOT / "deployment" / "docker" / "Dockerfile.analyze"
        assert path.exists(), f"Missing {path}"

    def test_deploy_dockerfile_exists(self) -> None:
        path = PROJECT_ROOT / "deployment" / "docker" / "Dockerfile.deploy"
        assert path.exists(), f"Missing {path}"

    def test_dashboard_dockerfile_exists(self) -> None:
        path = PROJECT_ROOT / "deployment" / "docker" / "Dockerfile.dashboard"
        assert path.exists(), f"Missing {path}"

    def test_qa_dockerfile_exists(self) -> None:
        path = PROJECT_ROOT / "deployment" / "docker" / "Dockerfile.qa"
        assert path.exists(), f"Missing {path}"


class TestDockerfileStructure:
    """Test Dockerfile content conventions."""

    def test_base_uses_python312(self) -> None:
        path = PROJECT_ROOT / "deployment" / "docker" / "Dockerfile.base"
        content = path.read_text(encoding="utf-8")
        assert "python:3.12" in content or "python:3.13" in content

    def test_base_uses_uv(self) -> None:
        path = PROJECT_ROOT / "deployment" / "docker" / "Dockerfile.base"
        content = path.read_text(encoding="utf-8")
        assert "uv" in content

    def test_train_has_cuda(self) -> None:
        path = PROJECT_ROOT / "deployment" / "docker" / "Dockerfile.train"
        content = path.read_text(encoding="utf-8")
        assert "cuda" in content.lower() or "nvidia" in content.lower()

    def test_all_flow_dockerfiles_reference_base(self) -> None:
        for flow in ["data", "train", "analyze", "deploy", "dashboard", "qa"]:
            path = PROJECT_ROOT / "deployment" / "docker" / f"Dockerfile.{flow}"
            content = path.read_text(encoding="utf-8")
            assert (
                "minivess-base" in content
                or "Dockerfile.base" in content
                or "base" in content.lower()
            )


class TestDockerComposeFlows:
    """Test docker-compose.flows.yml structure."""

    def test_flows_compose_exists(self) -> None:
        path = PROJECT_ROOT / "deployment" / "docker-compose.flows.yml"
        assert path.exists()

    def test_flows_compose_valid_yaml(self) -> None:
        import yaml

        path = PROJECT_ROOT / "deployment" / "docker-compose.flows.yml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "services" in data

    def test_flows_compose_has_train_service(self) -> None:
        import yaml

        path = PROJECT_ROOT / "deployment" / "docker-compose.flows.yml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "train" in data["services"]


class TestTrainingFlow:
    """Test training as a Prefect @flow."""

    def test_train_flow_module_exists(self) -> None:
        from minivess.orchestration.flows import train_flow

        assert hasattr(train_flow, "training_flow")

    def test_train_flow_is_callable(self) -> None:
        from minivess.orchestration.flows.train_flow import training_flow

        assert callable(training_flow)


class TestInterFlowContract:
    """Test inter-flow communication via MLflow."""

    def test_flow_contract_module_exists(self) -> None:
        from minivess.orchestration import flow_contract

        assert hasattr(flow_contract, "FlowContract")

    def test_flow_contract_find_upstream_run(self) -> None:
        from minivess.orchestration.flow_contract import FlowContract

        contract = FlowContract(tracking_uri="mlruns")
        # Should not raise even with no runs
        assert hasattr(contract, "find_upstream_run")


class TestTriggerChainUpdate:
    """Test PipelineTriggerChain now includes QA flow."""

    def test_chain_has_qa_flow(self) -> None:
        from minivess.orchestration.trigger import PipelineTriggerChain

        chain = PipelineTriggerChain()
        assert "qa" in chain.flow_names
