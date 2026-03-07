"""Tests for preflight env var validation in training and analysis flows (T-12, closes #417).

TDD RED phase: training_flow() must validate SPLITS_DIR and CHECKPOINT_DIR
at flow start. analysis_flow() must validate ANALYSIS_OUTPUT_DIR.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


class TestTrainingFlowPreflight:
    def test_raises_when_splits_dir_missing(self) -> None:
        """training_flow raises RuntimeError when SPLITS_DIR not set."""
        from minivess.orchestration.flows.train_flow import _validate_training_env

        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("SPLITS_DIR", "CHECKPOINT_DIR")
        }
        with (
            patch.dict(os.environ, env, clear=True),
            pytest.raises(RuntimeError, match="SPLITS_DIR"),
        ):
            _validate_training_env()

    def test_raises_when_checkpoint_dir_missing(self) -> None:
        """training_flow raises RuntimeError when CHECKPOINT_DIR not set."""
        from minivess.orchestration.flows.train_flow import _validate_training_env

        env = {k: v for k, v in os.environ.items() if k != "CHECKPOINT_DIR"}
        env["SPLITS_DIR"] = "/tmp/splits"
        with (
            patch.dict(os.environ, env, clear=True),
            pytest.raises(RuntimeError, match="CHECKPOINT_DIR"),
        ):
            _validate_training_env()

    def test_no_error_when_all_vars_set(self) -> None:
        """_validate_training_env() passes when required env vars are set."""
        from minivess.orchestration.flows.train_flow import _validate_training_env

        with patch.dict(
            os.environ,
            {"SPLITS_DIR": "/tmp/splits", "CHECKPOINT_DIR": "/tmp/checkpoints"},
        ):
            _validate_training_env()  # must not raise

    def test_error_message_actionable(self) -> None:
        """RuntimeError message must name the missing variable and suggest fix."""
        from minivess.orchestration.flows.train_flow import _validate_training_env

        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("SPLITS_DIR", "CHECKPOINT_DIR")
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(RuntimeError) as exc_info:
                _validate_training_env()
            msg = str(exc_info.value)
            # Must name missing variable and suggest remedy
            assert "SPLITS_DIR" in msg or "CHECKPOINT_DIR" in msg
            assert "export" in msg.lower() or "set" in msg.lower()


class TestAnalysisFlowPreflight:
    def test_raises_when_analysis_output_dir_missing(self) -> None:
        """analysis_flow raises RuntimeError when ANALYSIS_OUTPUT_DIR not set."""
        from minivess.orchestration.flows.analysis_flow import _validate_analysis_env

        env = {k: v for k, v in os.environ.items() if k != "ANALYSIS_OUTPUT_DIR"}
        with (
            patch.dict(os.environ, env, clear=True),
            pytest.raises(RuntimeError, match="ANALYSIS_OUTPUT_DIR"),
        ):
            _validate_analysis_env()

    def test_no_error_when_analysis_output_dir_set(self) -> None:
        """_validate_analysis_env() passes when ANALYSIS_OUTPUT_DIR is set."""
        from minivess.orchestration.flows.analysis_flow import _validate_analysis_env

        with patch.dict(os.environ, {"ANALYSIS_OUTPUT_DIR": "/tmp/analysis"}):
            _validate_analysis_env()  # must not raise
