"""Tests for GPU benchmark module (T3.1, #650).

Validates:
- Benchmark YAML schema version
- Instance info section in cache
- Capabilities section (binary feasibility)
- Cache invalidation on driver change
- BenchmarkResult dataclass
- Feasibility check logic
- GPU model name normalization (RC11)
"""

from __future__ import annotations

from pathlib import Path


class TestGpuNameNormalization:
    """GPU model name is normalized to canonical form (RC11)."""

    def test_strips_nvidia_geforce_prefix(self) -> None:
        from minivess.compute.gpu_benchmark import normalize_gpu_name

        assert normalize_gpu_name("NVIDIA GeForce RTX 2070 SUPER") == "rtx_2070_super"

    def test_strips_nvidia_prefix(self) -> None:
        from minivess.compute.gpu_benchmark import normalize_gpu_name

        assert normalize_gpu_name("NVIDIA A100-SXM4-80GB") == "a100_sxm4_80gb"

    def test_lowercase_and_underscores(self) -> None:
        from minivess.compute.gpu_benchmark import normalize_gpu_name

        assert normalize_gpu_name("Tesla V100-PCIE-16GB") == "tesla_v100_pcie_16gb"

    def test_handles_empty_string(self) -> None:
        from minivess.compute.gpu_benchmark import normalize_gpu_name

        assert normalize_gpu_name("") == "unknown_gpu"


class TestBenchmarkResult:
    """BenchmarkResult dataclass stores per-model benchmark data."""

    def test_dataclass_fields(self) -> None:
        from minivess.compute.gpu_benchmark import BenchmarkResult

        result = BenchmarkResult(
            model_family="dynunet",
            peak_vram_mb=3500.0,
            throughput_img_per_sec=12.5,
            forward_ms=80.0,
            feasible=True,
        )
        assert result.model_family == "dynunet"
        assert result.peak_vram_mb == 3500.0
        assert result.throughput_img_per_sec == 12.5
        assert result.forward_ms == 80.0
        assert result.feasible is True


class TestFeasibilityCheck:
    """Feasibility logic: model fits if peak_vram < available * threshold."""

    def test_feasible_when_vram_fits(self) -> None:
        from minivess.compute.gpu_benchmark import check_feasibility

        assert check_feasibility(peak_vram_mb=3500.0, total_vram_mb=8000.0) is True

    def test_infeasible_when_vram_exceeds(self) -> None:
        from minivess.compute.gpu_benchmark import check_feasibility

        assert check_feasibility(peak_vram_mb=7500.0, total_vram_mb=8000.0) is False

    def test_infeasible_at_boundary(self) -> None:
        from minivess.compute.gpu_benchmark import check_feasibility

        # 90% threshold: 7200 / 8000 = 0.9 — at boundary, should be infeasible
        assert check_feasibility(peak_vram_mb=7200.0, total_vram_mb=8000.0) is False


class TestBenchmarkYamlSchema:
    """Benchmark YAML cache has required schema fields."""

    def test_schema_version_present(self, tmp_path: Path) -> None:
        from minivess.compute.gpu_benchmark import (
            GpuInfo,
            write_benchmark_yaml,
        )

        info = GpuInfo(
            name="RTX 2070 SUPER",
            normalized_name="rtx_2070_super",
            total_vram_mb=8000.0,
            driver_version="550.54",
            cuda_version="12.4",
        )
        cache_path = tmp_path / "gpu_benchmark.yaml"
        write_benchmark_yaml(cache_path, gpu_info=info, results=[])

        import yaml

        data = yaml.safe_load(cache_path.read_text(encoding="utf-8"))
        assert "schema_version" in data

    def test_instance_section_present(self, tmp_path: Path) -> None:
        from minivess.compute.gpu_benchmark import (
            GpuInfo,
            write_benchmark_yaml,
        )

        info = GpuInfo(
            name="RTX 2070 SUPER",
            normalized_name="rtx_2070_super",
            total_vram_mb=8000.0,
            driver_version="550.54",
            cuda_version="12.4",
        )
        cache_path = tmp_path / "gpu_benchmark.yaml"
        write_benchmark_yaml(cache_path, gpu_info=info, results=[])

        import yaml

        data = yaml.safe_load(cache_path.read_text(encoding="utf-8"))
        assert "instance_info" in data
        inst = data["instance_info"]
        assert inst["gpu_model"] == "rtx_2070_super"
        assert inst["driver_version"] == "550.54"
        assert inst["cuda_version"] == "12.4"

    def test_capabilities_section_present(self, tmp_path: Path) -> None:
        from minivess.compute.gpu_benchmark import (
            BenchmarkResult,
            GpuInfo,
            write_benchmark_yaml,
        )

        info = GpuInfo(
            name="RTX 2070 SUPER",
            normalized_name="rtx_2070_super",
            total_vram_mb=8000.0,
            driver_version="550.54",
            cuda_version="12.4",
        )
        results = [
            BenchmarkResult(
                model_family="dynunet",
                peak_vram_mb=3500.0,
                throughput_img_per_sec=12.5,
                forward_ms=80.0,
                feasible=True,
            ),
        ]
        cache_path = tmp_path / "gpu_benchmark.yaml"
        write_benchmark_yaml(cache_path, gpu_info=info, results=results)

        import yaml

        data = yaml.safe_load(cache_path.read_text(encoding="utf-8"))
        assert "capabilities" in data
        assert "dynunet" in data["capabilities"]
        assert data["capabilities"]["dynunet"]["feasible"] is True


class TestCacheInvalidation:
    """Cache is invalidated when driver/CUDA version changes."""

    def test_valid_cache_accepted(self, tmp_path: Path) -> None:
        from minivess.compute.gpu_benchmark import (
            GpuInfo,
            is_cache_valid,
            write_benchmark_yaml,
        )

        info = GpuInfo(
            name="RTX 2070 SUPER",
            normalized_name="rtx_2070_super",
            total_vram_mb=8000.0,
            driver_version="550.54",
            cuda_version="12.4",
        )
        cache_path = tmp_path / "gpu_benchmark.yaml"
        write_benchmark_yaml(cache_path, gpu_info=info, results=[])

        assert is_cache_valid(cache_path, current_driver="550.54", current_cuda="12.4")

    def test_invalidated_on_driver_change(self, tmp_path: Path) -> None:
        from minivess.compute.gpu_benchmark import (
            GpuInfo,
            is_cache_valid,
            write_benchmark_yaml,
        )

        info = GpuInfo(
            name="RTX 2070 SUPER",
            normalized_name="rtx_2070_super",
            total_vram_mb=8000.0,
            driver_version="550.54",
            cuda_version="12.4",
        )
        cache_path = tmp_path / "gpu_benchmark.yaml"
        write_benchmark_yaml(cache_path, gpu_info=info, results=[])

        assert not is_cache_valid(
            cache_path, current_driver="560.00", current_cuda="12.4"
        )

    def test_missing_cache_is_invalid(self, tmp_path: Path) -> None:
        from minivess.compute.gpu_benchmark import is_cache_valid

        cache_path = tmp_path / "nonexistent.yaml"
        assert not is_cache_valid(
            cache_path, current_driver="550.54", current_cuda="12.4"
        )
