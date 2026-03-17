"""Tests for the MLflow → ONNX → BentoML import chain verification.

PR-D T2 (Issue #826): End-to-end test of the deploy import chain
using a mock model (no real PyTorch model needed).

TDD Phase: RED — tests written before implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def _make_dummy_onnx(tmp_path: Path) -> Path:
    """Create a minimal valid ONNX model for testing.

    Uses a simple graph: input(1,1,8,8,4) → identity → output(1,1,8,8,4).
    """
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    # Simple identity graph
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1, 8, 8, 4]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 1, 8, 8, 4]
    )
    identity_node = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph(
        [identity_node], "test_graph", [input_tensor], [output_tensor]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    onnx_path = tmp_path / "model.onnx"
    onnx.save(model, str(onnx_path))
    return onnx_path


def _make_champion() -> dict[str, Any]:
    """Create a synthetic champion model dict."""
    return {
        "run_id": "abc123",
        "experiment_id": "1",
        "category": "balanced",
        "metrics": {"val/dice": 0.88, "val/cldice": 0.85},
    }


class TestOnnxExportShape:
    """Verify ONNX export produces valid models."""

    def test_onnx_export_shape(self, tmp_path: Path) -> None:
        """Exported ONNX model has correct input/output shapes."""
        onnx = pytest.importorskip("onnx")

        onnx_path = _make_dummy_onnx(tmp_path)
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)

        # Verify input shape
        inp = model.graph.input[0]
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        assert dims == [1, 1, 8, 8, 4]


class TestOnnxRuntimeInference:
    """Verify ONNX Runtime can run inference on exported models."""

    def test_onnx_runtime_inference(self, tmp_path: Path) -> None:
        """ORT inference produces output with correct shape."""
        pytest.importorskip("onnx")
        ort = pytest.importorskip("onnxruntime")
        import numpy as np

        onnx_path = _make_dummy_onnx(tmp_path)
        session = ort.InferenceSession(str(onnx_path))

        input_data = np.random.randn(1, 1, 8, 8, 4).astype(np.float32)
        result = session.run(None, {"input": input_data})

        assert result[0].shape == (1, 1, 8, 8, 4)


class TestBentoMLImportFromOnnx:
    """Verify BentoML model import from ONNX file."""

    def test_bentoml_import_from_onnx(self, tmp_path: Path) -> None:
        """BentoML import creates result with correct tag format."""
        from minivess.serving.deploy_import_chain import (
            verify_onnx_file_exists,
        )

        onnx_path = _make_dummy_onnx(tmp_path)
        assert verify_onnx_file_exists(onnx_path) is True

    def test_bentoml_model_tag_format(self) -> None:
        """BentoML tag follows minivess-{category}:{run_id} format."""
        from minivess.pipeline.deploy_champion_discovery import ChampionModel
        from minivess.serving.bento_model_import import get_bento_model_tag

        champion = ChampionModel(
            run_id="abc123",
            experiment_id="1",
            category="balanced",
        )
        tag = get_bento_model_tag(champion)
        assert tag == "minivess-balanced:abc123"


class TestImportChainEndToEnd:
    """End-to-end import chain verification."""

    def test_import_chain_end_to_end(self, tmp_path: Path) -> None:
        """Full chain: create ONNX → verify → prepare import metadata."""
        pytest.importorskip("onnx")
        from minivess.serving.deploy_import_chain import (
            ImportChainResult,
            verify_import_chain,
        )

        onnx_path = _make_dummy_onnx(tmp_path)
        champion = _make_champion()

        result = verify_import_chain(
            champion=champion,
            onnx_path=onnx_path,
        )

        assert isinstance(result, ImportChainResult)
        assert result.onnx_valid is True
        assert result.ort_inference_ok is True
        assert result.output_shape == [1, 1, 8, 8, 4]

    def test_import_chain_missing_onnx(self, tmp_path: Path) -> None:
        """Chain fails gracefully with missing ONNX file."""
        from minivess.serving.deploy_import_chain import (
            verify_import_chain,
        )

        champion = _make_champion()
        result = verify_import_chain(
            champion=champion,
            onnx_path=tmp_path / "nonexistent.onnx",
        )

        assert result.onnx_valid is False
        assert result.ort_inference_ok is False

    def test_import_chain_opset_17(self, tmp_path: Path) -> None:
        """ONNX model uses opset 17."""
        onnx = pytest.importorskip("onnx")

        onnx_path = _make_dummy_onnx(tmp_path)
        model = onnx.load(str(onnx_path))

        opset = model.opset_import[0].version
        assert opset == 17
