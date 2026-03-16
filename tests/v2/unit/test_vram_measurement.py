"""T09 VRAM measurement — verify in-process capture via trainer.fit() return dict.

The bug: smoke_test_mamba.yaml used separate `python -c` processes to capture
VRAM after training. Each process had a fresh CUDA context → VRAM_PEAK_MB=0.

The fix: torch.cuda.max_memory_allocated() inside trainer.fit() (same CUDA context),
returned in the result dict, printed as sentinel by train_flow.py.

Tests are AST/content-based — no model instantiation, no CUDA needed.
Runs in staging tier.
"""

from __future__ import annotations

import ast
import pathlib

import yaml

_REPO = pathlib.Path(__file__).resolve().parents[3]
_TRAINER = _REPO / "src" / "minivess" / "pipeline" / "trainer.py"
_TRAIN_FLOW = _REPO / "src" / "minivess" / "orchestration" / "flows" / "train_flow.py"
_MAMBA_YAML = _REPO / "deployment" / "skypilot" / "smoke_test_mamba.yaml"


def _get_all_string_literals(path: pathlib.Path) -> list[str]:
    """Extract all string literal values from a Python file via AST."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    strings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.append(node.value)
    return strings


class TestTrainerFitReturnsVram:
    """trainer.py fit() must return vram_peak_mb and vram_peak_gb in result dict."""

    def test_fit_return_contains_vram_peak_mb(self) -> None:
        strings = _get_all_string_literals(_TRAINER)
        assert "vram_peak_mb" in strings, (
            "trainer.py must include 'vram_peak_mb' string literal (fit() return key)"
        )

    def test_fit_return_contains_vram_peak_gb(self) -> None:
        strings = _get_all_string_literals(_TRAINER)
        assert "vram_peak_gb" in strings, (
            "trainer.py must include 'vram_peak_gb' string literal (fit() return key)"
        )


class TestTrainFlowVramSentinel:
    """train_flow.py must print VRAM_PEAK_MB= sentinel for Ralph Loop parsing."""

    def test_train_flow_prints_vram_sentinel(self) -> None:
        strings = _get_all_string_literals(_TRAIN_FLOW)
        assert any("VRAM_PEAK_MB=" in s for s in strings), (
            "train_flow.py must contain 'VRAM_PEAK_MB=' sentinel string "
            "(printed to stdout for Ralph Loop log parsing)"
        )

    def test_train_flow_logs_vram_metric(self) -> None:
        strings = _get_all_string_literals(_TRAIN_FLOW)
        assert "vram_peak_mb" in strings, (
            "train_flow.py must reference 'vram_peak_mb' for MLflow metric logging"
        )


class TestSkyPilotYamlNoSeparateVram:
    """smoke_test_mamba.yaml must NOT use separate python -c for VRAM capture."""

    def test_no_separate_process_vram_capture(self) -> None:
        data = yaml.safe_load(_MAMBA_YAML.read_text(encoding="utf-8"))
        run_block = data.get("run", "")
        assert "max_memory_allocated" not in run_block, (
            "smoke_test_mamba.yaml run: block must not use separate-process "
            "torch.cuda.max_memory_allocated() — VRAM is captured in-process "
            "by trainer.fit()"
        )

    def test_no_separate_process_vram_reset(self) -> None:
        data = yaml.safe_load(_MAMBA_YAML.read_text(encoding="utf-8"))
        run_block = data.get("run", "")
        assert "reset_peak_memory_stats" not in run_block, (
            "smoke_test_mamba.yaml run: block must not use separate-process "
            "torch.cuda.reset_peak_memory_stats() — reset happens in-process "
            "inside trainer.fit()"
        )
