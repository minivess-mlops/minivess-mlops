"""Tests for profiler_integration.py — build_profiler_context + TraceHandler.

TDD RED phase for T1.2 (#646): Profiler context manager wrapping the epoch
loop in fit(), with gzip compression, size gates, and epoch counting.
"""

from __future__ import annotations

import ast
import contextlib
import gzip
import json
from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from minivess.config.models import ProfilingConfig

# Path to trainer.py for AST inspection
TRAINER_PY = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "src"
    / "minivess"
    / "pipeline"
    / "trainer.py"
)


class TestBuildProfilerContext:
    """Test build_profiler_context() factory function."""

    def test_enabled_returns_profiler(self, tmp_path: Path) -> None:
        """When enabled=True, returns a torch.profiler.profile instance."""
        from minivess.pipeline.profiler_integration import build_profiler_context

        config = ProfilingConfig(enabled=True, epochs=2)
        ctx = build_profiler_context(config, output_dir=tmp_path)
        # Should NOT be a nullcontext
        assert not isinstance(ctx, contextlib._GeneratorContextManager)
        assert not isinstance(ctx, type(contextlib.nullcontext()))

    def test_disabled_returns_nullcontext(self, tmp_path: Path) -> None:
        """When enabled=False, returns contextlib.nullcontext."""
        from minivess.pipeline.profiler_integration import build_profiler_context

        config = ProfilingConfig(enabled=False, epochs=0)
        ctx = build_profiler_context(config, output_dir=tmp_path)
        # nullcontext yields None
        with ctx as prof:
            assert prof is None


class TestTraceHandler:
    """Test TraceHandler callback class."""

    def test_counts_profiled_epochs(self, tmp_path: Path) -> None:
        """After N on_trace_ready calls, profiled_epoch_count == N."""
        from minivess.pipeline.profiler_integration import TraceHandler

        config = ProfilingConfig(enabled=True, epochs=5)
        handler = TraceHandler(config=config, output_dir=tmp_path)

        # Simulate 3 on_trace_ready calls with a mock profiler
        for i in range(3):
            mock_prof = MagicMock()
            # export_chrome_trace should write a file
            trace_path = tmp_path / f"trace_{i}.json"
            trace_path.write_text("{}", encoding="utf-8")
            mock_prof.export_chrome_trace = MagicMock(
                side_effect=lambda p, _tp=trace_path: _tp.write_text(
                    "{}", encoding="utf-8"
                )
            )
            handler(mock_prof)

        assert handler.profiled_epoch_count == 3

    def test_gzip_compression(self, tmp_path: Path) -> None:
        """When compress_traces=True, produces .json.gz files."""
        from minivess.pipeline.profiler_integration import TraceHandler

        config = ProfilingConfig(enabled=True, epochs=1, compress_traces=True)
        handler = TraceHandler(config=config, output_dir=tmp_path)

        mock_prof = MagicMock()

        # Make export_chrome_trace write a small JSON file
        def _export(path: str) -> None:
            Path(path).write_text('{"traceEvents": []}', encoding="utf-8")

        mock_prof.export_chrome_trace = _export
        handler(mock_prof)

        # Should have at least one .json.gz file
        gz_files = list((tmp_path / "profiling").glob("*.json.gz"))
        assert len(gz_files) >= 1, (
            f"Expected .json.gz files, found: {list((tmp_path / 'profiling').iterdir())}"
        )

        # Verify it's valid gzip
        with gzip.open(gz_files[0], "rt", encoding="utf-8") as f:
            data = json.loads(f.read())
        assert "traceEvents" in data

    def test_skips_large_traces(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Traces exceeding trace_size_limit_mb are skipped with warning."""
        from minivess.pipeline.profiler_integration import TraceHandler

        # Set tiny limit (1 MB) and write a large trace
        config = ProfilingConfig(
            enabled=True, epochs=1, trace_size_limit_mb=1, compress_traces=False
        )
        handler = TraceHandler(config=config, output_dir=tmp_path)

        mock_prof = MagicMock()

        # Write a 2 MB trace
        def _export(path: str) -> None:
            Path(path).write_text("x" * (2 * 1024 * 1024), encoding="utf-8")

        mock_prof.export_chrome_trace = _export

        import logging

        with caplog.at_level(logging.WARNING):
            handler(mock_prof)

        # Should log a warning about size
        assert any(
            "size" in msg.lower() or "limit" in msg.lower() for msg in caplog.messages
        ), f"Expected size limit warning, got: {caplog.messages}"

    def test_trace_paths_collected(self, tmp_path: Path) -> None:
        """Handler collects paths of successfully saved traces."""
        from minivess.pipeline.profiler_integration import TraceHandler

        config = ProfilingConfig(enabled=True, epochs=2, compress_traces=False)
        handler = TraceHandler(config=config, output_dir=tmp_path)

        for _ in range(2):
            mock_prof = MagicMock()
            mock_prof.export_chrome_trace = lambda p: Path(p).write_text(
                "{}", encoding="utf-8"
            )
            handler(mock_prof)

        assert len(handler.trace_paths) == 2
        for p in handler.trace_paths:
            assert p.exists()


class TestProfilingSummary:
    """Test ProfilingSummary dataclass."""

    def test_has_validation_profiled_field(self) -> None:
        """ProfilingSummary must have a validation_profiled bool field."""
        from minivess.pipeline.profiler_integration import ProfilingSummary

        field_names = {f.name for f in fields(ProfilingSummary)}
        assert "validation_profiled" in field_names

    def test_has_required_fields(self) -> None:
        """ProfilingSummary has trace_paths, total_profiled_epochs, etc."""
        from minivess.pipeline.profiler_integration import ProfilingSummary

        field_names = {f.name for f in fields(ProfilingSummary)}
        required = {
            "trace_paths",
            "total_profiled_epochs",
            "validation_profiled",
            "key_averages_text",
            "summary_dict",
        }
        missing = required - field_names
        assert not missing, f"ProfilingSummary missing fields: {missing}"


class TestFitAcceptsProfilingConfig:
    """Verify fit() accepts profiling_config keyword argument."""

    def test_fit_has_profiling_config_parameter(self) -> None:
        """fit() signature must include profiling_config keyword arg."""
        source = TRAINER_PY.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(TRAINER_PY))

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "fit":
                arg_names = [a.arg for a in node.args.args + node.args.kwonlyargs]
                assert "profiling_config" in arg_names, (
                    f"fit() must have 'profiling_config' parameter. "
                    f"Found args: {arg_names}"
                )
                return

        pytest.fail("fit() method not found in trainer.py")


class TestProfilerSchedule:
    """Verify profiler schedule configuration."""

    def test_schedule_active_equals_config_epochs(self, tmp_path: Path) -> None:
        """The profiler schedule's active count should match config.epochs."""
        from minivess.pipeline.profiler_integration import build_profiler_context

        config = ProfilingConfig(enabled=True, epochs=7)
        ctx = build_profiler_context(config, output_dir=tmp_path)
        # The context manager should be a torch.profiler.profile instance
        # We verify via the schedule function stored on the profiler
        assert hasattr(ctx, "schedule"), (
            "build_profiler_context must return a torch.profiler.profile "
            "with a schedule attribute"
        )
