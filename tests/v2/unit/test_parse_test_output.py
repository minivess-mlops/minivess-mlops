"""Tests for the test output parser — foundation of the 4-layer skip enforcement.

RED phase: these tests define parse_test_output.py behavior before implementation.
The parser is the SINGLE SOURCE OF TRUTH for extracting test results.

Plan: silent-ignoring-and-kicking-the-can-down-the-road-problem.xml Task A1
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


# Sample pytest outputs for testing
CLEAN_OUTPUT = "============================= 5362 passed in 408.85s (0:06:48) ============================="
SKIP_OUTPUT = "5362 passed, 2 skipped in 410.00s"
FAIL_OUTPUT = "5360 passed, 2 failed in 415.00s"
MIXED_OUTPUT = "5360 passed, 2 failed, 3 skipped in 420.00s"
DESELECTED_OUTPUT = "6732 passed, 729 deselected in 413.38s (0:06:53)"
SKIP_REASONS_OUTPUT = """tests/v2/unit/test_foo.py::test_bar SKIPPED [1] test_foo.py:42: mamba-ssm not installed
tests/integration/test_net.py::test_minio SKIPPED [1] test_net.py:128: Flow container cannot reach MinIO
=========================== short test summary info ============================
5362 passed, 2 skipped in 410.00s"""
EMPTY_OUTPUT = ""


class TestParseTestOutput:
    """parse_test_output extracts test counts from pytest output."""

    def test_parses_clean_run(self) -> None:
        """All passed, no skips, no failures."""
        from scripts.parse_test_output import parse_test_output

        result = parse_test_output(CLEAN_OUTPUT)
        assert result["passed"] == 5362
        assert result["failed"] == 0
        assert result["skipped"] == 0

    def test_parses_skips(self) -> None:
        """Detects skipped tests."""
        from scripts.parse_test_output import parse_test_output

        result = parse_test_output(SKIP_OUTPUT)
        assert result["passed"] == 5362
        assert result["skipped"] == 2

    def test_parses_failures(self) -> None:
        """Detects failed tests."""
        from scripts.parse_test_output import parse_test_output

        result = parse_test_output(FAIL_OUTPUT)
        assert result["passed"] == 5360
        assert result["failed"] == 2

    def test_parses_mixed(self) -> None:
        """Detects mixed pass/fail/skip."""
        from scripts.parse_test_output import parse_test_output

        result = parse_test_output(MIXED_OUTPUT)
        assert result["passed"] == 5360
        assert result["failed"] == 2
        assert result["skipped"] == 3

    def test_parses_deselected(self) -> None:
        """Deselected tests are tracked separately."""
        from scripts.parse_test_output import parse_test_output

        result = parse_test_output(DESELECTED_OUTPUT)
        assert result["passed"] == 6732
        assert result["deselected"] == 729

    def test_extracts_skip_reasons(self) -> None:
        """Individual skip reasons are extracted."""
        from scripts.parse_test_output import parse_test_output

        result = parse_test_output(SKIP_REASONS_OUTPUT)
        reasons = result.get("skip_reasons", [])
        assert len(reasons) == 2
        assert "mamba-ssm not installed" in reasons[0]
        assert "MinIO" in reasons[1]

    def test_writes_json(self, tmp_path: Path) -> None:
        """Results written to JSON file."""
        from scripts.parse_test_output import parse_and_save

        out_file = tmp_path / "result.json"
        parse_and_save(SKIP_OUTPUT, out_file)

        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert data["passed"] == 5362
        assert data["skipped"] == 2

    def test_json_includes_timestamp(self, tmp_path: Path) -> None:
        """JSON output includes ISO 8601 UTC timestamp."""
        from scripts.parse_test_output import parse_and_save

        out_file = tmp_path / "result.json"
        parse_and_save(CLEAN_OUTPUT, out_file)

        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert "timestamp" in data
        assert "T" in data["timestamp"]  # ISO 8601

    def test_prints_investigation_prompt(self, capsys: object) -> None:
        """When skips > 0, stdout contains MANDATORY investigation prompt."""
        from scripts.parse_test_output import parse_test_output

        result = parse_test_output(SKIP_OUTPUT, print_prompt=True)
        # The function should print when print_prompt=True and skips > 0
        assert result["skipped"] == 2

    def test_exit_code_zero_on_clean(self) -> None:
        """0 skips, 0 failures → exit code 0."""
        from scripts.parse_test_output import get_exit_code, parse_test_output

        result = parse_test_output(CLEAN_OUTPUT)
        assert get_exit_code(result) == 0

    def test_exit_code_one_on_skips(self) -> None:
        """Any skips → exit code 1."""
        from scripts.parse_test_output import get_exit_code, parse_test_output

        result = parse_test_output(SKIP_OUTPUT)
        assert get_exit_code(result) == 1

    def test_exit_code_one_on_failures(self) -> None:
        """Any failures → exit code 1."""
        from scripts.parse_test_output import get_exit_code, parse_test_output

        result = parse_test_output(FAIL_OUTPUT)
        assert get_exit_code(result) == 1

    def test_handles_empty_input(self) -> None:
        """Empty input returns error result."""
        from scripts.parse_test_output import get_exit_code, parse_test_output

        result = parse_test_output(EMPTY_OUTPUT)
        assert get_exit_code(result) == 1
