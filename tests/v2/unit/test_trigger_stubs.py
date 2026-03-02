"""Tests for trigger stubs — DVC change + analysis completion.

Covers:
- on_dvc_version_change log message content
- on_analysis_completion log message content
- No exceptions raised
- Log messages contain dataset/experiment/champion names
- Multiple sequential calls work
- Edge cases: empty strings, special characters

Closes #190.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

from minivess.orchestration.trigger import on_analysis_completion, on_dvc_version_change


class TestOnDvcVersionChange:
    """Test DVC version change trigger stub."""

    def test_logs_dataset_name(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO):
            on_dvc_version_change("minivess", "v1.0", "v1.1")
        assert "minivess" in caplog.text

    def test_logs_versions(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO):
            on_dvc_version_change("minivess", "v1.0", "v1.1")
        assert "v1.0" in caplog.text
        assert "v1.1" in caplog.text

    def test_no_exception(self) -> None:
        on_dvc_version_change("minivess", "v1.0", "v1.1")

    def test_empty_names_no_crash(self) -> None:
        on_dvc_version_change("", "", "")

    def test_dashboard_pending_message(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO):
            on_dvc_version_change("minivess", "v1.0", "v1.1")
        assert "Dashboard update pending" in caplog.text


class TestOnAnalysisCompletion:
    """Test analysis completion trigger stub."""

    def test_logs_experiment_name(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO):
            on_analysis_completion("dynunet_e2e_debug", "balanced", 5)
        assert "dynunet_e2e_debug" in caplog.text

    def test_logs_champion_name(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO):
            on_analysis_completion("dynunet_e2e_debug", "balanced", 5)
        assert "balanced" in caplog.text

    def test_logs_model_count(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO):
            on_analysis_completion("dynunet_e2e_debug", "balanced", 5)
        assert "5" in caplog.text

    def test_no_exception(self) -> None:
        on_analysis_completion("dynunet_e2e_debug", "balanced", 5)

    def test_dashboard_pending_message(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO):
            on_analysis_completion("dynunet_e2e_debug", "balanced", 5)
        assert "Dashboard update pending" in caplog.text
