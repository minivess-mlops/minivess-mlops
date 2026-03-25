"""Controller health preflight check — detects INIT/STOPPED controller state.

Verifies that check_controller_health() exists in preflight_gcp.py, is wired
into the checks list, and correctly interprets sky status output for various
controller states (UP, STOPPED, INIT, absent, sky unavailable).

Plan: run-debug-factorial-experiment-report-9th-pass (Task 2.C)
Issue #957: Cloud robustness — controller health preflight check.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

PREFLIGHT = Path("scripts/preflight_gcp.py")


def _load_preflight_module():
    """Load preflight_gcp.py as a module for testing."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("preflight", PREFLIGHT)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestControllerHealthCheck:
    """Preflight must detect unhealthy SkyPilot controller states."""

    def test_controller_health_function_exists(self) -> None:
        """check_controller_health must exist in preflight_gcp.py."""
        source = PREFLIGHT.read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        assert "check_controller_health" in names, (
            "preflight_gcp.py must define check_controller_health() — "
            "Issue #957: detect INIT/STOPPED controller before launch"
        )

    def test_controller_health_wired_in_checks(self) -> None:
        """check_controller_health must be in the checks = [...] list."""
        source = PREFLIGHT.read_text(encoding="utf-8")
        lines = source.splitlines()
        in_checks = False
        for line in lines:
            if "checks = [" in line:
                in_checks = True
            if in_checks and "check_controller_health" in line:
                return
            if in_checks and line.strip() == "]":
                break
        pytest.fail("check_controller_health not found in main checks list")

    def test_controller_up_passes(self) -> None:
        """Mock sky status showing UP -> (True, msg)."""
        mod = _load_preflight_module()
        mock_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="sky-jobs-controller-abc  GCP  UP  10m",
            stderr="",
        )
        with patch.object(mod, "_run", return_value=mock_result):
            ok, msg = mod.check_controller_health()
        assert ok is True

    def test_controller_stopped_fails(self) -> None:
        """Mock sky status showing STOPPED -> (False, msg)."""
        mod = _load_preflight_module()
        mock_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="sky-jobs-controller-abc  GCP  STOPPED",
            stderr="",
        )
        with patch.object(mod, "_run", return_value=mock_result):
            ok, msg = mod.check_controller_health()
        assert ok is False

    def test_controller_init_warns(self) -> None:
        """INIT state -> (True, warning) -- may resolve on its own."""
        mod = _load_preflight_module()
        mock_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="sky-jobs-controller-abc  GCP  INIT",
            stderr="",
        )
        with patch.object(mod, "_run", return_value=mock_result):
            ok, msg = mod.check_controller_health()
        assert ok is True
        assert "INIT" in msg

    def test_no_controller_passes(self) -> None:
        """No controller yet -> (True, msg)."""
        mod = _load_preflight_module()
        mock_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr="",
        )
        with patch.object(mod, "_run", return_value=mock_result):
            ok, msg = mod.check_controller_health()
        assert ok is True

    def test_sky_unavailable_skips(self) -> None:
        """sky not installed -> (True, 'skip' in msg)."""
        mod = _load_preflight_module()
        mock_result = subprocess.CompletedProcess(
            args=[],
            returncode=127,
            stdout="",
            stderr="sky: command not found",
        )
        with patch.object(mod, "_run", return_value=mock_result):
            ok, msg = mod.check_controller_health()
        assert ok is True
        assert "skip" in msg.lower()

    def test_controller_error_state_fails(self) -> None:
        """ERROR state -> (False, msg)."""
        mod = _load_preflight_module()
        mock_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="sky-jobs-controller-abc  GCP  ERROR",
            stderr="",
        )
        with patch.object(mod, "_run", return_value=mock_result):
            ok, msg = mod.check_controller_health()
        assert ok is False

    def test_returns_tuple_format(self) -> None:
        """check_controller_health must return (bool, str) tuple."""
        mod = _load_preflight_module()
        mock_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr="",
        )
        with patch.object(mod, "_run", return_value=mock_result):
            result = mod.check_controller_health()
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)
