"""Tests for BentoML model import with SAM3 ONNX models.

Verifies that SAM3 ONNX models can be imported into BentoML model store
and served via BentoML service.

IMPORTANT: SAM3 export/import tests require real pretrained weights (GPU ≥16 GB).
They are skipped in CI where SAM3 is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from minivess.adapters.model_builder import _sam3_package_available

if TYPE_CHECKING:
    from pathlib import Path

_sam3_skip = pytest.mark.skipif(
    not _sam3_package_available(), reason="SAM3 not installed"
)


@_sam3_skip
@pytest.mark.slow
class TestBentoImportSam3:
    """Test BentoML model import with SAM3 ONNX files."""

    def _export_sam3_onnx(self, tmp_path: Path) -> Path:
        """Export a SAM3 vanilla adapter to ONNX for testing."""
        from minivess.adapters.model_builder import build_adapter
        from minivess.config.models import ModelConfig, ModelFamily

        config = ModelConfig(
            family=ModelFamily.SAM3_VANILLA,
            name="sam3_vanilla",
            in_channels=1,
            out_channels=2,
        )
        adapter = build_adapter(config)
        onnx_path = tmp_path / "sam3_test.onnx"
        example_input = torch.randn(1, 1, 4, 32, 32)
        adapter.export_onnx(onnx_path, example_input)
        return onnx_path

    def test_bento_model_tag_from_champion(self) -> None:
        from minivess.pipeline.deploy_champion_discovery import ChampionModel
        from minivess.serving.bento_model_import import get_bento_model_tag

        champion = ChampionModel(
            run_id="sam3_run_123",
            experiment_id="exp1",
            category="balanced",
            metrics={"dsc": 0.35},
            checkpoint_path=None,
            model_config={"model_family": "sam3_vanilla"},
        )
        tag = get_bento_model_tag(champion)
        assert tag == "minivess-balanced:sam3_run_123"

    def test_bento_import_returns_metadata(self, tmp_path: Path) -> None:
        """Verify import_champion_to_bento returns correct metadata."""
        from minivess.pipeline.deploy_champion_discovery import ChampionModel
        from minivess.serving.bento_model_import import import_champion_to_bento

        onnx_path = self._export_sam3_onnx(tmp_path)
        champion = ChampionModel(
            run_id="sam3_run_456",
            experiment_id="exp1",
            category="topology",
            metrics={"dsc": 0.40, "cldice": 0.55},
            checkpoint_path=None,
            model_config={"model_family": "sam3_topolora"},
        )
        # BentoML may not be installed — function handles gracefully
        result = import_champion_to_bento(champion, onnx_path)
        assert result.metadata["champion_category"] == "topology"
        assert result.metadata["run_id"] == "sam3_run_456"

    def test_sam3_onnx_file_valid_for_import(self, tmp_path: Path) -> None:
        """Verify the ONNX file exported from SAM3 is valid for BentoML import."""
        onnx_path = self._export_sam3_onnx(tmp_path)
        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 0
        # BentoML bentoml.onnx.save_model just needs a valid file path
        # We don't call save_model directly (requires BentoML runtime)
        # but validate the file is ready for import.

    def test_bentofile_generator_works(self, tmp_path: Path) -> None:
        """Verify bentofile.yaml generation for SAM3 models."""
        from minivess.serving.deploy_artifacts import generate_bentofile

        result_path = generate_bentofile(
            service_name="minivess-sam3",
            models=["minivess-balanced:sam3_run_123"],
            output_dir=tmp_path,
        )
        assert result_path.exists()
        content = result_path.read_text(encoding="utf-8")
        assert "onnxruntime" in content
