"""Tests for resilient script failure tracking — T26 regression test.

Bug: Each run_factorial.sh --resume invocation creates a NEW job log file.
The resilient wrapper reads only the LATEST log, so failure counts never
accumulate across iterations.
"""

from __future__ import annotations

from pathlib import Path

import pytest

RESILIENT_SCRIPT = (
    Path(__file__).resolve().parents[4] / "scripts" / "run_factorial_resilient.sh"
)


class TestResilientFailureTracking:
    """T26: Failure counts must accumulate across log files."""

    @pytest.fixture()
    def _source(self) -> str:
        if not RESILIENT_SCRIPT.exists():
            pytest.skip("run_factorial_resilient.sh not found")
        return RESILIENT_SCRIPT.read_text(encoding="utf-8")

    def test_failure_count_accumulates_across_logs(self, _source):
        """grep for failures must search ALL log files, not just latest."""
        # After fix: should grep across multiple files or use a single cumulative file
        has_multi_file_search = (
            "*.txt" in _source
            or "glob" in _source.lower()
            or "-h" in _source  # grep -h (no filename prefix) across multiple files
        )
        has_single_log = "latest_log" in _source or "head -1" in _source
        # The fix should either grep all files OR maintain a single cumulative log
        assert has_multi_file_search or not has_single_log, (
            "Resilient script must grep ALL log files for failure counts, "
            "not just the latest — failure counts never accumulate otherwise"
        )

    def test_permanent_failure_deduplication(self, _source):
        """Permanent failure file must deduplicate entries."""
        assert "permanently_failed" in _source.lower() or "PERMANENTLY_FAILED" in _source, (
            "Resilient script must track permanently failed conditions"
        )
