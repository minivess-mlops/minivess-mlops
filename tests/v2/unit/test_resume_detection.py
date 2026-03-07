"""Tests for T-18: check_resume_state_task in training_flow.

Verifies that check_resume_state_task() reads epoch_latest.yaml with
yaml.safe_load() (not regex), returns None when no YAML exists, and
returns state dict when a valid YAML is present.

Uses ast.parse() for source inspection — NO regex (CLAUDE.md Rule #16).
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

_TRAIN_FLOW_SRC = Path("src/minivess/orchestration/flows/train_flow.py")


# ---------------------------------------------------------------------------
# Source-level: check_resume_state_task must exist and use yaml.safe_load
# ---------------------------------------------------------------------------


class TestCheckResumeStateTaskExists:
    def test_check_resume_state_task_in_source(self) -> None:
        """train_flow.py must define check_resume_state_task."""
        source = _TRAIN_FLOW_SRC.read_text(encoding="utf-8")
        assert "check_resume_state_task" in source, (
            "train_flow.py must define check_resume_state_task() for resume detection. "
            "Add: @task def check_resume_state_task(checkpoint_dir: Path) -> dict | None"
        )

    def test_uses_yaml_safe_load_not_regex(self) -> None:
        """check_resume_state_task must use yaml.safe_load (no import re)."""
        source = _TRAIN_FLOW_SRC.read_text(encoding="utf-8")
        assert "yaml.safe_load" in source, (
            "train_flow.py must use yaml.safe_load() to read epoch_latest.yaml. "
            "Never use regex or ast.literal_eval for YAML parsing."
        )
        # Ensure no regex import
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "re", (
                        "train_flow.py must NOT import 're'. Use yaml.safe_load() for YAML parsing."
                    )
            if isinstance(node, ast.ImportFrom) and node.module == "re":
                raise AssertionError(
                    "train_flow.py must NOT import from 're'. Use yaml.safe_load()."
                )


# ---------------------------------------------------------------------------
# Functional: check_resume_state_task behavior
# ---------------------------------------------------------------------------


class TestCheckResumeStateTaskBehavior:
    def test_no_resume_when_no_yaml(self, tmp_path) -> None:
        """check_resume_state_task() must return None if no epoch_latest.yaml."""
        from minivess.orchestration.flows.train_flow import check_resume_state_task

        result = check_resume_state_task(tmp_path)
        assert result is None, (
            f"check_resume_state_task() must return None when no epoch_latest.yaml. Got: {result}"
        )

    def test_resume_detected_when_yaml_exists(self, tmp_path, monkeypatch) -> None:
        """check_resume_state_task() returns state dict when epoch_latest.yaml exists."""
        yaml_path = tmp_path / "epoch_latest.yaml"
        state = {
            "epoch": 5,
            "fold": 0,
            "mlflow_run_id": "test_run_123",
            "best_val_loss": 0.42,
            "timestamp": "2026-03-07T00:00:00+00:00",
        }
        yaml_path.write_text(yaml.dump(state), encoding="utf-8")

        # Mock MLflow run status as RUNNING
        mock_run = MagicMock()
        mock_run.info.status = "RUNNING"

        with patch("mlflow.get_run", return_value=mock_run):
            from minivess.orchestration.flows.train_flow import (
                check_resume_state_task,
            )

            result = check_resume_state_task(tmp_path)

        assert result is not None, (
            "check_resume_state_task() must return state dict when epoch_latest.yaml "
            "exists and MLflow run is RUNNING."
        )
        assert "epoch" in result, (
            f"Returned state dict must have 'epoch' key. Got keys: {list(result.keys())}"
        )
        assert result["epoch"] == 5

    def test_stale_state_ignored(self, tmp_path) -> None:
        """check_resume_state_task() returns None if MLflow run is FINISHED."""
        yaml_path = tmp_path / "epoch_latest.yaml"
        state = {
            "epoch": 10,
            "fold": 0,
            "mlflow_run_id": "finished_run",
            "best_val_loss": 0.3,
            "timestamp": "2026-03-07T00:00:00+00:00",
        }
        yaml_path.write_text(yaml.dump(state), encoding="utf-8")

        mock_run = MagicMock()
        mock_run.info.status = "FINISHED"

        with patch("mlflow.get_run", return_value=mock_run):
            from minivess.orchestration.flows.train_flow import (
                check_resume_state_task,
            )

            result = check_resume_state_task(tmp_path)

        assert result is None, (
            "check_resume_state_task() must return None when MLflow run is FINISHED. "
            f"Got: {result}"
        )

    def test_missing_mlflow_run_id_returns_none(self, tmp_path) -> None:
        """check_resume_state_task() returns None if mlflow_run_id is missing."""
        yaml_path = tmp_path / "epoch_latest.yaml"
        state = {
            "epoch": 3,
            "fold": 0,
            "best_val_loss": 0.5,
            "timestamp": "2026-03-07T00:00:00+00:00",
            # No mlflow_run_id
        }
        yaml_path.write_text(yaml.dump(state), encoding="utf-8")

        from minivess.orchestration.flows.train_flow import check_resume_state_task

        result = check_resume_state_task(tmp_path)
        assert result is None, (
            "check_resume_state_task() must return None if mlflow_run_id is missing "
            f"from epoch_latest.yaml. Got: {result}"
        )
