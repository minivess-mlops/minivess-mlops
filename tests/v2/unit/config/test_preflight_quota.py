"""GCP quota preflight check — Phase 7 Task 7.3.

Verifies that check_gcp_cpu_quota() exists and returns the correct
tuple format. Does NOT require actual GCP credentials — the function
gracefully handles gcloud absence.

Plan: run-debug-factorial-experiment-report-6th-pass-post-run-fix-2.xml
"""

from __future__ import annotations

import ast
from pathlib import Path

PREFLIGHT = Path("scripts/preflight_gcp.py")


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
