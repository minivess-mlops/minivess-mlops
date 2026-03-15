"""Tests for GPU benchmark cache integration (T3.3).

Validates:
- Valid cache logged to MLflow with sys_bench_ prefix
- Missing cache logs warning (non-blocking)
- Invalid cache ignored gracefully
- All benchmark params use sys_bench_ prefix (NOT prof_gpu_ — RC8)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import yaml


def _write_valid_cache(cache_path: Path) -> None:
    """Write a valid benchmark cache YAML."""
    data = {
        "schema_version": 1,
        "instance_info": {
            "gpu_model": "rtx_2070_super",
            "gpu_name_raw": "NVIDIA GeForce RTX 2070 SUPER",
            "total_vram_mb": 8000.0,
            "driver_version": "550.54",
            "cuda_version": "12.4",
        },
        "capabilities": {
            "dynunet": {"feasible": True},
            "sam3_vanilla": {"feasible": True},
        },
        "benchmarks": {
            "dynunet": {
                "peak_vram_mb": 3500.0,
                "throughput_img_per_sec": 12.5,
                "forward_ms": 80.0,
                "feasible": True,
            },
        },
    }
    cache_path.write_text(yaml.dump(data), encoding="utf-8")


class TestValidCacheLogged:
    """Valid cache logs params to MLflow."""

    def test_valid_cache_logged_to_mlflow(self, tmp_path: Path) -> None:
        from minivess.compute.gpu_profile import log_benchmark_to_mlflow

        cache_path = tmp_path / "gpu_benchmark.yaml"
        _write_valid_cache(cache_path)
        tracker = MagicMock()

        log_benchmark_to_mlflow(cache_path, tracker=tracker)

        tracker.log_params.assert_called_once()
        params = tracker.log_params.call_args[0][0]
        assert "sys_bench_gpu_model" in params
        assert params["sys_bench_gpu_model"] == "rtx_2070_super"


class TestMissingCache:
    """Missing cache logs warning, doesn't crash."""

    def test_missing_cache_logs_warning(self, tmp_path: Path) -> None:
        from minivess.compute.gpu_profile import log_benchmark_to_mlflow

        cache_path = tmp_path / "nonexistent.yaml"
        tracker = MagicMock()

        # Should not raise
        log_benchmark_to_mlflow(cache_path, tracker=tracker)

        # No params logged
        tracker.log_params.assert_not_called()


class TestInvalidCache:
    """Invalid cache is ignored gracefully."""

    def test_invalid_cache_ignored(self, tmp_path: Path) -> None:
        from minivess.compute.gpu_profile import log_benchmark_to_mlflow

        cache_path = tmp_path / "bad.yaml"
        cache_path.write_text("not: valid: yaml: [broken", encoding="utf-8")
        tracker = MagicMock()

        # Should not raise
        log_benchmark_to_mlflow(cache_path, tracker=tracker)

        tracker.log_params.assert_not_called()


class TestSysBenchPrefix:
    """All benchmark params use sys_bench_ prefix (RC8)."""

    def test_benchmark_params_have_sys_bench_prefix(self, tmp_path: Path) -> None:
        from minivess.compute.gpu_profile import load_benchmark_params

        cache_path = tmp_path / "gpu_benchmark.yaml"
        _write_valid_cache(cache_path)

        params = load_benchmark_params(cache_path)

        assert len(params) > 0, "Should return at least one param"
        for key in params:
            assert key.startswith("sys_bench_"), (
                f"Key '{key}' must start with 'sys_bench_' (RC8)"
            )
