"""Preflight env var VALUE validation — placeholder, empty, localhost detection.

Extends check_env_vars() in scripts/preflight_gcp.py to detect:
- Unexpanded ${...} placeholders (shell variable not expanded)
- REPLACE_WITH_... templates (copied from .env.example without editing)
- Empty required values
- localhost URIs (soft WARNING for MLFLOW_TRACKING_URI — valid for local dev)
- Valid cloud URIs (should pass cleanly)

TDD Task 1.4 from cloud robustness plan (#945).
Rule #16: No regex — uses str methods only.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

if TYPE_CHECKING:
    import pytest

PREFLIGHT = Path("scripts/preflight_gcp.py")


def _load_preflight_module() -> Any:
    """Load preflight_gcp.py as a module for testing."""
    spec = importlib.util.spec_from_file_location("preflight", PREFLIGHT)
    assert spec is not None, f"Could not load module spec from {PREFLIGHT}"
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None, "Module spec has no loader"
    spec.loader.exec_module(mod)
    return mod


class TestEnvVarValidation:
    """Validate that check_env_vars detects bad env var VALUES, not just presence."""

    def test_unexpanded_dollar_placeholder_fails(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """HF_TOKEN=${HF_TOKEN} -> (False, msg with 'placeholder')."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "HF_TOKEN=${HF_TOKEN}\nMLFLOW_TRACKING_URI=https://mlflow-xxx.run.app\n",
            encoding="utf-8",
        )
        mod = _load_preflight_module()
        with patch.object(mod, "ENV_FILE", env_file):
            ok, msg = mod.check_env_vars()
        assert ok is False, f"Unexpanded ${{HF_TOKEN}} should fail, got: {msg}"
        assert "placeholder" in msg.lower(), f"Message should mention 'placeholder': {msg}"

    def test_replace_with_placeholder_fails_for_mlflow(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """MLFLOW_TRACKING_URI=REPLACE_WITH_YOUR_URI -> (False, msg)."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "HF_TOKEN=hf_abc123\nMLFLOW_TRACKING_URI=REPLACE_WITH_YOUR_URI\n",
            encoding="utf-8",
        )
        mod = _load_preflight_module()
        with patch.object(mod, "ENV_FILE", env_file):
            ok, msg = mod.check_env_vars()
        assert ok is False, f"REPLACE_WITH_ placeholder should fail, got: {msg}"
        assert "placeholder" in msg.lower() or "replace" in msg.lower(), (
            f"Message should mention placeholder/replace: {msg}"
        )

    def test_replace_with_placeholder_fails_for_hf_token(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """HF_TOKEN=hf_REPLACE_WITH_YOUR_TOKEN -> (False, msg)."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "HF_TOKEN=hf_REPLACE_WITH_YOUR_TOKEN\nMLFLOW_TRACKING_URI=https://mlflow-xxx.run.app\n",
            encoding="utf-8",
        )
        mod = _load_preflight_module()
        with patch.object(mod, "ENV_FILE", env_file):
            ok, msg = mod.check_env_vars()
        assert ok is False, f"REPLACE_WITH_ in HF_TOKEN should fail, got: {msg}"

    def test_partial_placeholder_fails(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """MLFLOW_TRACKING_URI=http://REPLACE_WITH_YOUR_HOST:5000 -> (False, msg)."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "HF_TOKEN=hf_abc123\nMLFLOW_TRACKING_URI=http://REPLACE_WITH_YOUR_HOST:5000\n",
            encoding="utf-8",
        )
        mod = _load_preflight_module()
        with patch.object(mod, "ENV_FILE", env_file):
            ok, msg = mod.check_env_vars()
        assert ok is False, f"Partial REPLACE_WITH_ should fail, got: {msg}"

    def test_empty_required_var_fails(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """MLFLOW_TRACKING_URI= (empty) -> (False, msg)."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "HF_TOKEN=hf_abc123\nMLFLOW_TRACKING_URI=\n",
            encoding="utf-8",
        )
        mod = _load_preflight_module()
        with patch.object(mod, "ENV_FILE", env_file):
            ok, msg = mod.check_env_vars()
        assert ok is False, f"Empty MLFLOW_TRACKING_URI should fail, got: {msg}"

    def test_localhost_uri_returns_warning_not_fail(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """MLFLOW_TRACKING_URI=http://localhost:5000 -> (True, 'localhost' in msg)."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "HF_TOKEN=hf_abc123\nMLFLOW_TRACKING_URI=http://localhost:5000\n",
            encoding="utf-8",
        )
        mod = _load_preflight_module()
        with patch.object(mod, "ENV_FILE", env_file):
            ok, msg = mod.check_env_vars()
        assert ok is True, f"localhost should WARN (True), not fail: {msg}"
        assert "localhost" in msg.lower(), (
            f"Message should mention localhost warning: {msg}"
        )

    def test_valid_cloud_run_uri_passes(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """MLFLOW_TRACKING_URI=https://mlflow-xxx.run.app -> (True, msg)."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "HF_TOKEN=hf_abc123\nMLFLOW_TRACKING_URI=https://mlflow-xxx.run.app\n",
            encoding="utf-8",
        )
        mod = _load_preflight_module()
        with patch.object(mod, "ENV_FILE", env_file):
            ok, msg = mod.check_env_vars()
        assert ok is True, f"Valid cloud URI should pass, got: {msg}"

    def test_missing_env_var_still_fails(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Original behavior preserved: missing var -> (False, 'Missing')."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "HF_TOKEN=hf_abc123\n",
            encoding="utf-8",
        )
        mod = _load_preflight_module()
        with patch.object(mod, "ENV_FILE", env_file):
            ok, msg = mod.check_env_vars()
        assert ok is False, f"Missing MLFLOW_TRACKING_URI should fail, got: {msg}"

    def test_no_env_file_fails(self, tmp_path: Path) -> None:
        """Missing .env file -> (False, msg with '.env')."""
        env_file = tmp_path / ".env_nonexistent"
        mod = _load_preflight_module()
        with patch.object(mod, "ENV_FILE", env_file):
            ok, msg = mod.check_env_vars()
        assert ok is False
        assert ".env" in msg.lower() or "not found" in msg.lower()

    def test_dollar_brace_in_middle_of_value_fails(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """MLFLOW_TRACKING_URI=https://${HOST}:5000 -> (False, msg)."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "HF_TOKEN=hf_abc123\nMLFLOW_TRACKING_URI=https://${HOST}:5000\n",
            encoding="utf-8",
        )
        mod = _load_preflight_module()
        with patch.object(mod, "ENV_FILE", env_file):
            ok, msg = mod.check_env_vars()
        assert ok is False, f"${{HOST}} in value should fail, got: {msg}"
        assert "placeholder" in msg.lower(), f"Should mention placeholder: {msg}"
