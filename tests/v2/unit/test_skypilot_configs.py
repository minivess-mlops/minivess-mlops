"""Tests for T-07: SkyPilot YAML configs must use prefect deployment run.

Uses yaml.safe_load() for all YAML parsing — NO regex (CLAUDE.md Rule #16).
Verifies that scripts/train_monitored.py and scripts/run_hpo.py are NOT invoked
directly from SkyPilot run sections.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_TRAIN_GENERIC = Path("deployment/skypilot/train_generic.yaml")
_TRAIN_HPO = Path("deployment/skypilot/train_hpo_sweep.yaml")


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# train_generic.yaml
# ---------------------------------------------------------------------------


class TestTrainGenericYaml:
    def test_train_generic_yaml_is_valid(self) -> None:
        """train_generic.yaml must parse without error."""
        config = _load(_TRAIN_GENERIC)
        assert isinstance(config, dict)

    def test_train_generic_no_python_script_invocation(self) -> None:
        """train_generic.yaml run section must not invoke train_monitored.py directly."""
        config = _load(_TRAIN_GENERIC)
        run_section: str = config.get("run", "")
        assert "scripts/train_monitored.py" not in run_section, (
            "train_generic.yaml still invokes scripts/train_monitored.py directly. "
            "Replace with: prefect deployment run 'training-flow/default' --params ..."
        )

    def test_train_generic_uses_prefect_run(self) -> None:
        """train_generic.yaml run section must use prefect deployment run."""
        config = _load(_TRAIN_GENERIC)
        run_section: str = config.get("run", "")
        assert "prefect deployment run" in run_section, (
            "train_generic.yaml does not use 'prefect deployment run'. "
            "Add: prefect deployment run 'training-flow/default' --params ..."
        )

    def test_train_generic_has_prefect_api_url_env(self) -> None:
        """train_generic.yaml must declare PREFECT_API_URL in envs."""
        config = _load(_TRAIN_GENERIC)
        envs: dict = config.get("envs", {})
        assert "PREFECT_API_URL" in envs, (
            "train_generic.yaml missing PREFECT_API_URL in envs section. "
            "Add PREFECT_API_URL: ${PREFECT_API_URL} so the spot instance can reach "
            "the Prefect server."
        )

    def test_train_generic_has_experiment_name_env(self) -> None:
        """train_generic.yaml must have EXPERIMENT_NAME env var for the flow."""
        config = _load(_TRAIN_GENERIC)
        envs: dict = config.get("envs", {})
        assert "EXPERIMENT_NAME" in envs, (
            "train_generic.yaml missing EXPERIMENT_NAME in envs section."
        )


# ---------------------------------------------------------------------------
# train_hpo_sweep.yaml
# ---------------------------------------------------------------------------


class TestTrainHpoSweepYaml:
    def test_train_hpo_yaml_is_valid(self) -> None:
        """train_hpo_sweep.yaml must parse without error."""
        config = _load(_TRAIN_HPO)
        assert isinstance(config, dict)

    def test_train_hpo_no_python_script(self) -> None:
        """train_hpo_sweep.yaml run section must not invoke run_hpo.py directly."""
        config = _load(_TRAIN_HPO)
        run_section: str = config.get("run", "")
        assert "scripts/run_hpo.py" not in run_section, (
            "train_hpo_sweep.yaml still invokes scripts/run_hpo.py directly. "
            "Replace with: prefect deployment run 'hpo-flow/default' --params ..."
        )
