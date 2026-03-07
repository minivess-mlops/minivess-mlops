"""Tests for T-21: deployment/prefect/deployments.yaml with all 9 flows defined.

Verifies that:
- The YAML is parseable via yaml.safe_load()
- All 9 flow names appear in the deployments
- training-flow uses gpu-pool
- All other flows use cpu-pool

NO regex — uses yaml.safe_load() for parsing.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_DEPLOYMENTS_YAML = Path("deployment/prefect/deployments.yaml")

_ALL_FLOW_NAMES = [
    "minivess-acquisition",
    "minivess-data",
    "training-flow",
    "Post-Training Pipeline",
    "analysis-flow",
    "Deploy Pipeline",
    "minivess-dashboard",
    "qa-flow",
    "minivess-annotation",
]

_CPU_FLOW_NAMES = [n for n in _ALL_FLOW_NAMES if n != "training-flow"]


def _load_deployments() -> dict:
    """Load and return the parsed deployments YAML."""
    return yaml.safe_load(_DEPLOYMENTS_YAML.read_text(encoding="utf-8"))


class TestDeploymentsYamlBasic:
    def test_deployments_yaml_exists(self) -> None:
        """deployment/prefect/deployments.yaml must exist."""
        assert _DEPLOYMENTS_YAML.exists(), (
            "deployment/prefect/deployments.yaml does not exist. "
            "Create it with all 9 flow deployments."
        )

    def test_deployments_yaml_parseable(self) -> None:
        """deployment/prefect/deployments.yaml must be parseable via yaml.safe_load()."""
        if not _DEPLOYMENTS_YAML.exists():
            return
        data = _load_deployments()
        assert isinstance(data, dict), (
            "deployments.yaml must parse to a dict (mapping), not "
            f"{type(data).__name__}"
        )

    def test_deployments_yaml_has_deployments_key(self) -> None:
        """deployments.yaml must have a top-level 'deployments' key."""
        if not _DEPLOYMENTS_YAML.exists():
            return
        data = _load_deployments()
        assert "deployments" in data, (
            "deployments.yaml must have a top-level 'deployments' key."
        )

    def test_deployments_is_list(self) -> None:
        """deployments.yaml['deployments'] must be a list."""
        if not _DEPLOYMENTS_YAML.exists():
            return
        data = _load_deployments()
        if "deployments" not in data:
            return
        assert isinstance(data["deployments"], list), (
            "'deployments' must be a list of deployment entries."
        )


class TestAllFlowsHaveDeployment:
    def _get_flow_names(self) -> list[str]:
        """Extract flow_name values from all deployment entries."""
        if not _DEPLOYMENTS_YAML.exists():
            return []
        data = _load_deployments()
        deployments = data.get("deployments", [])
        return [d.get("flow_name", "") for d in deployments if isinstance(d, dict)]

    def test_all_flows_have_deployment(self) -> None:
        """All 9 flow names must appear in deployments YAML."""
        if not _DEPLOYMENTS_YAML.exists():
            return
        registered = set(self._get_flow_names())
        missing = [name for name in _ALL_FLOW_NAMES if name not in registered]
        assert not missing, (
            f"Missing deployments for flows: {missing}. "
            "Each flow must have an entry with flow_name matching its @flow(name=...) decorator."
        )

    def test_exactly_nine_deployments(self) -> None:
        """There must be exactly 9 deployments (one per flow)."""
        if not _DEPLOYMENTS_YAML.exists():
            return
        data = _load_deployments()
        deployments = data.get("deployments", [])
        assert len(deployments) == 9, (
            f"Expected 9 deployments, found {len(deployments)}. "
            "One deployment per flow."
        )

    def test_each_deployment_has_name_field(self) -> None:
        """Each deployment entry must have a 'name' field (the deployment name)."""
        if not _DEPLOYMENTS_YAML.exists():
            return
        data = _load_deployments()
        deployments = data.get("deployments", [])
        missing_name = [
            d.get("flow_name", "<unknown>")
            for d in deployments
            if isinstance(d, dict) and not d.get("name")
        ]
        assert not missing_name, (
            f"These flow deployments are missing a 'name' field: {missing_name}. "
            "Each deployment must have name: default (or another explicit name)."
        )


class TestWorkPoolAssignment:
    def _get_work_pool_for_flow(self, flow_name: str) -> str | None:
        """Return the work_pool_name for a given flow_name."""
        if not _DEPLOYMENTS_YAML.exists():
            return None
        data = _load_deployments()
        for d in data.get("deployments", []):
            if not isinstance(d, dict):
                continue
            if d.get("flow_name") == flow_name:
                work_pool = d.get("work_pool", {})
                if isinstance(work_pool, dict):
                    return work_pool.get("name")
        return None

    def test_train_flow_on_gpu_pool(self) -> None:
        """training-flow deployment must use work_pool_name == 'gpu-pool'."""
        if not _DEPLOYMENTS_YAML.exists():
            return
        pool = self._get_work_pool_for_flow("training-flow")
        assert pool == "gpu-pool", (
            f"training-flow must use work_pool.name: gpu-pool, got: {pool!r}. "
            "Training requires GPU — assign it to gpu-pool."
        )

    def test_cpu_flows_on_cpu_pool(self) -> None:
        """All non-training flows must use work_pool_name == 'cpu-pool'."""
        if not _DEPLOYMENTS_YAML.exists():
            return
        wrong = []
        for flow_name in _CPU_FLOW_NAMES:
            pool = self._get_work_pool_for_flow(flow_name)
            if pool != "cpu-pool":
                wrong.append(f"{flow_name!r} → {pool!r}")
        assert not wrong, (
            f"These flows must use cpu-pool but don't: {wrong}. "
            "Only training-flow should use gpu-pool."
        )

    def test_acquisition_on_cpu_pool(self) -> None:
        """minivess-acquisition must use cpu-pool."""
        if not _DEPLOYMENTS_YAML.exists():
            return
        pool = self._get_work_pool_for_flow("minivess-acquisition")
        assert pool == "cpu-pool", (
            f"minivess-acquisition must use cpu-pool, got: {pool!r}"
        )

    def test_annotation_on_cpu_pool(self) -> None:
        """minivess-annotation must use cpu-pool."""
        if not _DEPLOYMENTS_YAML.exists():
            return
        pool = self._get_work_pool_for_flow("minivess-annotation")
        assert pool == "cpu-pool", (
            f"minivess-annotation must use cpu-pool, got: {pool!r}"
        )
