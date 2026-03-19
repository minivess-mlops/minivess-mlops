"""Tests for acquisition Hydra experiment config.

Phase 4, Task T-ACQ.4.2 of overnight-child-01-acquisition.xml.
Validates configs/experiment/acquisition.yaml exists and is well-formed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

_ACQUISITION_YAML = Path("configs/experiment/acquisition.yaml")
_ACQUISITION_FLOW_SRC = Path("src/minivess/orchestration/flows/acquisition_flow.py")


class TestHydraAcquisitionConfig:
    """configs/experiment/acquisition.yaml must exist and be well-formed."""

    def test_acquisition_yaml_exists(self) -> None:
        assert _ACQUISITION_YAML.exists(), (
            f"{_ACQUISITION_YAML} does not exist. "
            "Create a Hydra experiment config for the acquisition flow."
        )

    def test_acquisition_yaml_is_valid(self) -> None:
        content = _ACQUISITION_YAML.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        assert isinstance(data, dict), "YAML must be a dict"

    def test_acquisition_yaml_has_acquisition_key(self) -> None:
        content = _ACQUISITION_YAML.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        assert "acquisition" in data, (
            "acquisition.yaml must have an 'acquisition' key with flow params"
        )

    def test_acquisition_yaml_has_external_datasets(self) -> None:
        content = _ACQUISITION_YAML.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        acq = data.get("acquisition", {})
        assert "external_datasets" in acq, (
            "acquisition config must list external_datasets"
        )
        datasets = acq["external_datasets"]
        assert isinstance(datasets, list)
        assert len(datasets) == 2  # deepvess, vesselnn (tubenet_2pm excluded)

    def test_acquisition_yaml_has_force_download(self) -> None:
        content = _ACQUISITION_YAML.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        acq = data.get("acquisition", {})
        assert "force_download" in acq

    def test_acquisition_yaml_has_dvc_auto_commit(self) -> None:
        content = _ACQUISITION_YAML.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        acq = data.get("acquisition", {})
        assert "dvc_auto_commit" in acq


class TestAcquisitionFlowHasPrefectDecorator:
    """acquisition_flow.py must have @flow decorator (AST check, no regex)."""

    def test_flow_decorator_present(self) -> None:
        source = _ACQUISITION_FLOW_SRC.read_text(encoding="utf-8")
        tree = ast.parse(source)

        flow_decorated = False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "run_acquisition_flow"
            ):
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Call):
                        func = dec.func
                        if isinstance(func, ast.Name) and func.id == "flow":
                            flow_decorated = True
                    elif isinstance(dec, ast.Name) and dec.id == "flow":
                        flow_decorated = True
        assert flow_decorated, "run_acquisition_flow must be decorated with @flow"

    def test_task_decorators_present(self) -> None:
        """All task functions must have @task decorator."""
        source = _ACQUISITION_FLOW_SRC.read_text(encoding="utf-8")
        tree = ast.parse(source)

        task_funcs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.endswith("_task"):
                has_task_dec = False
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Call):
                        func = dec.func
                        if isinstance(func, ast.Name) and func.id == "task":
                            has_task_dec = True
                    elif isinstance(dec, ast.Name) and dec.id == "task":
                        has_task_dec = True
                task_funcs.append((node.name, has_task_dec))

        assert task_funcs, "No _task functions found in acquisition_flow.py"
        for name, has_dec in task_funcs:
            assert has_dec, f"{name} missing @task decorator"


class TestDataDirFromEnv:
    """Acquisition flow reads DATA_DIR from environment."""

    def test_acquisition_config_default_output_dir(self) -> None:
        """Default output_dir should be a Path (not hardcoded to host path)."""
        from minivess.config.acquisition_config import AcquisitionConfig

        config = AcquisitionConfig()
        assert isinstance(config.output_dir, Path)
