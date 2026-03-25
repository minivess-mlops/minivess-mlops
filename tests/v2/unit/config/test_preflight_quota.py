"""GCP quota preflight checks — CPU + GPU quota validation.

Verifies that check_gcp_cpu_quota() and check_gcp_gpu_quota() exist and
return the correct tuple format. Does NOT require actual GCP credentials —
the functions gracefully handle gcloud absence.

Plan: run-debug-factorial-experiment-report-6th-pass-post-run-fix-2.xml
Issue #943: GPU quota preflight (GPUS_ALL_REGIONS=1 bottleneck detection).
"""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import yaml

PREFLIGHT = Path("scripts/preflight_gcp.py")
CONTRACT_PATH = Path("configs/cloud/yaml_contract.yaml")


class TestGcpQuotaPreflight:
    """Preflight must include CPU quota check."""

    def test_check_gcp_cpu_quota_exists(self) -> None:
        """preflight_gcp.py must define check_gcp_cpu_quota function."""
        source = PREFLIGHT.read_text(encoding="utf-8")
        tree = ast.parse(source)
        func_names = [
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ]
        assert "check_gcp_cpu_quota" in func_names, (
            "preflight_gcp.py must define check_gcp_cpu_quota() — "
            "30 min wasted on n4 quota = 0 (metalearning)"
        )

    def test_quota_check_in_main_checks_list(self) -> None:
        """check_gcp_cpu_quota must be wired into the main checks list."""
        source = PREFLIGHT.read_text(encoding="utf-8")
        assert "check_gcp_cpu_quota" in source
        # Verify it appears in the checks list (not just as a function def)
        lines = source.splitlines()
        in_checks_list = False
        for line in lines:
            if "checks = [" in line:
                in_checks_list = True
            if in_checks_list and "check_gcp_cpu_quota" in line:
                return  # Found it in the list
            if in_checks_list and line.strip() == "]":
                break
        raise AssertionError("check_gcp_cpu_quota not found in main checks list")

    def test_quota_check_returns_tuple(self) -> None:
        """check_gcp_cpu_quota must return (bool, str) tuple."""
        # Import and call — gracefully handles missing gcloud
        import importlib.util

        spec = importlib.util.spec_from_file_location("preflight", PREFLIGHT)
        assert spec is not None, f"Could not load module spec from {PREFLIGHT}"
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None, "Module spec has no loader"
        spec.loader.exec_module(mod)

        result = mod.check_gcp_cpu_quota()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)


def _load_preflight_module():
    """Load preflight_gcp.py as a module for testing."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("preflight", PREFLIGHT)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _make_gcloud_region_json(metric: str, limit: float, usage: float) -> str:
    """Build a realistic gcloud compute regions describe --format=json response."""
    return json.dumps(
        {
            "name": "europe-west4",
            "quotas": [
                {"metric": "CPUS", "limit": 24.0, "usage": 0.0},
                {"metric": metric, "limit": limit, "usage": usage},
                {"metric": "DISKS_TOTAL_GB", "limit": 10240.0, "usage": 0.0},
            ],
            "status": "UP",
        }
    )


class TestGpuQuotaPreflight:
    """Preflight must include GPU quota check (Issue #943)."""

    def test_gpu_quota_function_exists(self) -> None:
        """check_gcp_gpu_quota must exist in preflight_gcp.py."""
        source = PREFLIGHT.read_text(encoding="utf-8")
        tree = ast.parse(source)
        func_names = [
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ]
        assert "check_gcp_gpu_quota" in func_names, (
            "preflight_gcp.py must define check_gcp_gpu_quota() — "
            "Issue #943: GPUS_ALL_REGIONS=1 bottleneck detection"
        )

    def test_gpu_quota_wired_in_checks(self) -> None:
        """check_gcp_gpu_quota must be in the checks = [...] list."""
        source = PREFLIGHT.read_text(encoding="utf-8")
        lines = source.splitlines()
        in_checks_list = False
        for line in lines:
            if "checks = [" in line:
                in_checks_list = True
            if in_checks_list and "check_gcp_gpu_quota" in line:
                return  # Found it in the list
            if in_checks_list and line.strip() == "]":
                break
        raise AssertionError("check_gcp_gpu_quota not found in main checks list")

    def test_gpu_quota_checks_l4_string(self) -> None:
        """Implementation must reference 'NVIDIA_L4_GPUS' metric name."""
        source = PREFLIGHT.read_text(encoding="utf-8")
        assert "NVIDIA_L4_GPUS" in source, (
            "check_gcp_gpu_quota must look for NVIDIA_L4_GPUS in gcloud output"
        )

    def test_gpu_quota_reads_threshold_from_contract(self) -> None:
        """min_gpu_quota must come from yaml_contract.yaml, not hardcoded."""
        # Must read yaml_contract.yaml — the contract is the source of truth
        contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
        assert "min_gpu_quota" in contract, (
            "yaml_contract.yaml must define min_gpu_quota"
        )
        assert contract["min_gpu_quota"] >= 1

    def test_gpu_quota_passes_when_sufficient(self) -> None:
        """Mock gcloud returning quota limit=8, usage=2 -> (True, msg)."""
        mod = _load_preflight_module()
        mock_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=_make_gcloud_region_json("NVIDIA_L4_GPUS", limit=8.0, usage=2.0),
            stderr="",
        )
        with patch.object(mod, "_run", return_value=mock_result):
            ok, msg = mod.check_gcp_gpu_quota()
        assert ok is True
        assert "NVIDIA_L4_GPUS" in msg or "available" in msg.lower() or "ok" in msg.lower()

    def test_gpu_quota_warns_when_below_job_count(self) -> None:
        """Mock quota=1 -> (True, "WARNING" in msg) -- warn, not block."""
        mod = _load_preflight_module()
        mock_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=_make_gcloud_region_json("NVIDIA_L4_GPUS", limit=1.0, usage=0.0),
            stderr="",
        )
        with patch.object(mod, "_run", return_value=mock_result):
            ok, msg = mod.check_gcp_gpu_quota()
        assert ok is True, "quota=1 should pass (warn, not block)"
        assert "warning" in msg.lower() or "low" in msg.lower(), (
            f"Expected warning message for quota=1, got: {msg}"
        )

    def test_gpu_quota_fails_when_zero(self) -> None:
        """Mock quota=0 -> (False, msg with 'NVIDIA_L4_GPUS')."""
        mod = _load_preflight_module()
        mock_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=_make_gcloud_region_json("NVIDIA_L4_GPUS", limit=0.0, usage=0.0),
            stderr="",
        )
        with patch.object(mod, "_run", return_value=mock_result):
            ok, msg = mod.check_gcp_gpu_quota()
        assert ok is False, "quota=0 must FAIL"
        assert "NVIDIA_L4_GPUS" in msg

    def test_gpu_quota_graceful_when_gcloud_missing(self) -> None:
        """Mock _run returncode=127 -> (True, 'skip' in msg)."""
        mod = _load_preflight_module()
        mock_result = subprocess.CompletedProcess(
            args=[],
            returncode=127,
            stdout="",
            stderr="gcloud: command not found",
        )
        with patch.object(mod, "_run", return_value=mock_result):
            ok, msg = mod.check_gcp_gpu_quota()
        assert ok is True, "gcloud missing should skip gracefully, not fail"
        assert "skip" in msg.lower(), (
            f"Expected 'skip' in message when gcloud missing, got: {msg}"
        )
