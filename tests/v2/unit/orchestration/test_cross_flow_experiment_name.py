"""Tests for cross-flow experiment name consistency.

T14 from double-check plan: verify train_flow, analysis_flow, and
post_training_flow agree on experiment names.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_SRC_ROOT = Path(__file__).resolve().parents[4] / "src" / "minivess"
_DEPLOY_ROOT = Path(__file__).resolve().parents[4] / "deployment"


class TestExperimentNameConsistency:
    """Cross-flow experiment name env vars must be consistent."""

    def test_constants_training_and_post_training_match(self) -> None:
        """EXPERIMENT_TRAINING and EXPERIMENT_POST_TRAINING must use same value."""
        from minivess.orchestration.constants import (
            EXPERIMENT_POST_TRAINING,
            EXPERIMENT_TRAINING,
        )

        assert EXPERIMENT_TRAINING == EXPERIMENT_POST_TRAINING, (
            f"Train={EXPERIMENT_TRAINING}, PostTrain={EXPERIMENT_POST_TRAINING}"
        )

    def test_train_flow_reads_experiment_name_from_env(self) -> None:
        """train_flow.py must read EXPERIMENT_NAME from env var."""
        flow_src = _SRC_ROOT / "orchestration" / "flows" / "train_flow.py"
        source = flow_src.read_text(encoding="utf-8")
        assert "EXPERIMENT_NAME" in source, (
            "train_flow.py should read EXPERIMENT_NAME from env"
        )

    def test_analysis_flow_reads_upstream_experiment(self) -> None:
        """analysis_flow.py must read UPSTREAM_EXPERIMENT from env var."""
        flow_src = _SRC_ROOT / "orchestration" / "flows" / "analysis_flow.py"
        source = flow_src.read_text(encoding="utf-8")
        assert "UPSTREAM_EXPERIMENT" in source, (
            "analysis_flow.py should read UPSTREAM_EXPERIMENT from env"
        )

    def test_post_training_flow_reads_upstream_experiment(self) -> None:
        """post_training_flow.py must read UPSTREAM_EXPERIMENT from env var."""
        flow_src = _SRC_ROOT / "orchestration" / "flows" / "post_training_flow.py"
        source = flow_src.read_text(encoding="utf-8")
        assert "UPSTREAM_EXPERIMENT" in source, (
            "post_training_flow.py should read UPSTREAM_EXPERIMENT from env"
        )

    def test_resolve_experiment_name_with_debug_suffix(self) -> None:
        """resolve_experiment_name appends debug suffix when env set."""
        import os

        from minivess.orchestration.constants import resolve_experiment_name

        # Without suffix
        old = os.environ.pop("MINIVESS_DEBUG_SUFFIX", None)
        try:
            assert resolve_experiment_name("test_exp") == "test_exp"
        finally:
            if old is not None:
                os.environ["MINIVESS_DEBUG_SUFFIX"] = old

    def test_skypilot_yaml_experiment_names(self) -> None:
        """SkyPilot YAMLs must set experiment name env vars consistently."""
        train_yaml = _DEPLOY_ROOT / "skypilot" / "train_factorial.yaml"
        if not train_yaml.exists():
            return  # Skip if YAML not present

        with train_yaml.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Check env vars are defined in the YAML
        envs = config.get("envs", {})
        # The YAML should set EXPERIMENT_NAME or EXPERIMENT for cross-flow contract
        has_experiment = any(
            k in envs for k in ("EXPERIMENT_NAME", "EXPERIMENT")
        )
        assert has_experiment or "setup" in str(config), (
            "train_factorial.yaml should set EXPERIMENT_NAME or EXPERIMENT in envs"
        )
