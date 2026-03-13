"""Tests for CUDA memory stats extraction (T4.1).

Validates:
- extract_memory_stats() returns dict with expected keys
- Graceful CPU fallback (empty dict)
- Fragmentation warning when alloc retries exceed threshold
- Metrics use prof_cuda_ prefix (RC16)
"""

from __future__ import annotations

from unittest.mock import patch


class TestMemoryStatsExtracted:
    """extract_memory_stats() returns dict with expected metric keys."""

    def test_returns_dict_with_expected_keys(self) -> None:
        from minivess.diagnostics.cuda_memory import extract_memory_stats

        mock_stats = {
            "allocated_bytes.all.peak": 3_670_016_000,
            "reserved_bytes.all.peak": 4_194_304_000,
            "num_alloc_retries": 2,
        }

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.memory_stats", return_value=mock_stats),
        ):
            result = extract_memory_stats()

        assert "prof_cuda_peak_allocated_mb" in result
        assert "prof_cuda_peak_reserved_mb" in result
        assert "prof_cuda_alloc_retries" in result
        # Verify MB conversion (3_670_016_000 / 1024 / 1024 ≈ 3500)
        assert abs(result["prof_cuda_peak_allocated_mb"] - 3500.0) < 1.0


class TestCpuFallback:
    """CPU-only execution returns empty dict."""

    def test_memory_stats_skipped_on_cpu(self) -> None:
        from minivess.diagnostics.cuda_memory import extract_memory_stats

        with patch("torch.cuda.is_available", return_value=False):
            result = extract_memory_stats()

        assert result == {}


class TestFragmentationWarning:
    """Warning logged when alloc retries exceed threshold."""

    def test_fragmentation_warning(self) -> None:
        from minivess.diagnostics.cuda_memory import extract_memory_stats

        mock_stats = {
            "allocated_bytes.all.peak": 3_670_016_000,
            "reserved_bytes.all.peak": 4_194_304_000,
            "num_alloc_retries": 15,
        }

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.memory_stats", return_value=mock_stats),
            patch("minivess.diagnostics.cuda_memory.logger") as mock_logger,
        ):
            result = extract_memory_stats(alloc_retry_warning_threshold=10)

        assert result["prof_cuda_alloc_retries"] == 15
        mock_logger.warning.assert_called_once()


class TestMetricPrefix:
    """All metrics use prof_cuda_ prefix (RC16)."""

    def test_all_keys_have_prof_cuda_prefix(self) -> None:
        from minivess.diagnostics.cuda_memory import extract_memory_stats

        mock_stats = {
            "allocated_bytes.all.peak": 1_000_000,
            "reserved_bytes.all.peak": 2_000_000,
            "num_alloc_retries": 0,
        }

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.memory_stats", return_value=mock_stats),
        ):
            result = extract_memory_stats()

        for key in result:
            assert key.startswith("prof_cuda_"), (
                f"Key '{key}' must start with 'prof_cuda_' (RC16)"
            )
