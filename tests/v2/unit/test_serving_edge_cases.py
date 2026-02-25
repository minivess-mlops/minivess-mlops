"""Edge-case and error-path tests for serving modules (Code Review R2.5).

Tests gaps identified in the code review:
- ONNX: corrupted model, metadata structure
- DICOM: empty metadata, SR with empty findings, all-tags-missing
- ClinicalDeploymentPipeline: validation warnings, manifest generation
- Gradio: extract_slice edge cases, predict_slice with None
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# T1: ONNX inference error paths
# ---------------------------------------------------------------------------


class TestOnnxErrorPaths:
    """Test ONNX inference error handling."""

    def test_nonexistent_model_raises(self, tmp_path: Path) -> None:
        """Loading a nonexistent ONNX model should raise."""
        from minivess.serving.onnx_inference import OnnxSegmentationInference

        with pytest.raises(Exception):  # noqa: B017
            OnnxSegmentationInference(tmp_path / "missing.onnx")

    def test_corrupted_model_raises(self, tmp_path: Path) -> None:
        """Loading a corrupted ONNX file should raise."""
        from minivess.serving.onnx_inference import OnnxSegmentationInference

        bad_onnx = tmp_path / "bad.onnx"
        bad_onnx.write_bytes(b"not a valid onnx model")
        with pytest.raises(Exception):  # noqa: B017
            OnnxSegmentationInference(bad_onnx)

    def test_metadata_has_inputs_and_outputs(self) -> None:
        """get_metadata should return dict with 'inputs' and 'outputs' keys."""
        import tempfile

        import torch

        from minivess.adapters.segresnet import SegResNetAdapter
        from minivess.config.models import ModelConfig, ModelFamily
        from minivess.serving.onnx_inference import OnnxSegmentationInference

        config = ModelConfig(
            family=ModelFamily.MONAI_SEGRESNET,
            name="test_meta", in_channels=1, out_channels=2,
        )
        model = SegResNetAdapter(config)
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = Path(f.name)
            example = torch.randn(1, 1, 16, 16, 16)
            model.export_onnx(onnx_path, example)
            engine = OnnxSegmentationInference(onnx_path)

        meta = engine.get_metadata()
        assert meta.inputs is not None
        assert meta.outputs is not None
        assert len(meta.inputs) >= 1
        assert meta.inputs[0].name is not None
        assert meta.inputs[0].shape is not None


# ---------------------------------------------------------------------------
# T2: DICOM handler edge cases
# ---------------------------------------------------------------------------


class TestDICOMEdgeCases:
    """Test DICOM handler edge cases."""

    def test_validate_empty_metadata(self) -> None:
        """Validating empty metadata should flag all required tags."""
        from minivess.serving.clinical_deploy import DICOMHandler

        handler = DICOMHandler()
        result = handler.validate_dicom_metadata({})
        assert result.is_valid is False
        assert len(result.missing_tags) == 4

    def test_validate_partial_metadata(self) -> None:
        """Partial metadata should flag only missing tags."""
        from minivess.serving.clinical_deploy import DICOMHandler

        handler = DICOMHandler()
        result = handler.validate_dicom_metadata({
            "PatientID": "P001",
            "Modality": "MR",
        })
        assert result.is_valid is False
        assert "StudyInstanceUID" in result.missing_tags
        assert "SeriesInstanceUID" in result.missing_tags
        assert "PatientID" not in result.missing_tags

    def test_create_sr_has_required_fields(self) -> None:
        """DICOM SR should contain study_uid, series_uid, findings, created."""
        from minivess.serving.clinical_deploy import DICOMHandler

        handler = DICOMHandler()
        sr = handler.create_dicom_sr(
            "1.2.3.4", "1.2.3.5",
            findings={"vessel_count": 42, "dice": 0.87},
        )
        assert sr["study_uid"] == "1.2.3.4"
        assert sr["series_uid"] == "1.2.3.5"
        assert sr["findings"]["vessel_count"] == 42
        assert "created" in sr
        assert sr["software"] == "minivess-segmentation"

    def test_create_sr_with_empty_findings(self) -> None:
        """SR creation with empty findings should still work."""
        from minivess.serving.clinical_deploy import DICOMHandler

        handler = DICOMHandler()
        sr = handler.create_dicom_sr("1.2.3.4", "1.2.3.5", findings={})
        assert sr["findings"] == {}
        assert sr["sr_type"] == "COMPREHENSIVE"


# ---------------------------------------------------------------------------
# T3: ClinicalDeploymentPipeline edge cases
# ---------------------------------------------------------------------------


class TestClinicalPipelineEdgeCases:
    """Test clinical deployment pipeline edge cases."""

    def test_validate_empty_config_fails(self) -> None:
        """Default config (empty model_name/path) should fail validation."""
        from minivess.serving.clinical_deploy import (
            ClinicalDeployConfig,
            ClinicalDeploymentPipeline,
        )

        pipeline = ClinicalDeploymentPipeline(ClinicalDeployConfig())
        result = pipeline.validate()
        assert result.is_valid is False
        assert "model_name" in result.missing_tags
        assert "model_path" in result.missing_tags

    def test_validate_default_version_warning(self) -> None:
        """Default version 0.0.0 should produce a warning."""
        from minivess.serving.clinical_deploy import (
            ClinicalDeployConfig,
            ClinicalDeploymentPipeline,
        )

        config = ClinicalDeployConfig(
            model_name="test", model_path="/some/path",
        )
        pipeline = ClinicalDeploymentPipeline(config)
        result = pipeline.validate()
        assert result.is_valid is True
        assert any("0.0.0" in w for w in result.warnings)

    def test_validate_clinical_default_ae_title_warning(self) -> None:
        """Clinical target with default AE title should warn."""
        from minivess.serving.clinical_deploy import (
            ClinicalDeployConfig,
            ClinicalDeploymentPipeline,
        )

        config = ClinicalDeployConfig(
            model_name="test", model_path="/path",
            deployment_target="clinical",
        )
        pipeline = ClinicalDeploymentPipeline(config)
        result = pipeline.validate()
        assert any("AE title" in w for w in result.warnings)

    def test_generate_manifest(self) -> None:
        """generate_manifest should return MonaiDeployManifest."""
        from minivess.serving.clinical_deploy import (
            ClinicalDeployConfig,
            ClinicalDeploymentPipeline,
            MonaiDeployManifest,
        )

        config = ClinicalDeployConfig(
            model_name="segresnet", model_path="/model.onnx",
            version="1.2.0",
        )
        pipeline = ClinicalDeploymentPipeline(config)
        manifest = pipeline.generate_manifest()
        assert isinstance(manifest, MonaiDeployManifest)
        assert manifest.model_name == "segresnet"
        assert manifest.version == "1.2.0"

    def test_to_markdown_contains_validation(self) -> None:
        """to_markdown should include validation status."""
        from minivess.serving.clinical_deploy import (
            ClinicalDeployConfig,
            ClinicalDeploymentPipeline,
        )

        config = ClinicalDeployConfig(model_name="test", model_path="/p")
        pipeline = ClinicalDeploymentPipeline(config)
        md = pipeline.to_markdown()
        assert "PASS" in md
        assert "test" in md


# ---------------------------------------------------------------------------
# T4: Gradio slice extraction edge cases
# ---------------------------------------------------------------------------


class TestGradioEdgeCases:
    """Test Gradio demo helper function edge cases."""

    def test_extract_slice_clamps_negative_index(self) -> None:
        """Negative index should be clamped to 0."""
        from minivess.serving.gradio_demo import extract_slice

        vol = np.random.default_rng(42).random((10, 20, 30), dtype=np.float32)
        s = extract_slice(vol, axis=0, index=-5)
        expected = vol[0]
        np.testing.assert_array_equal(s, expected.astype(np.float32))

    def test_extract_slice_clamps_large_index(self) -> None:
        """Index beyond volume size should be clamped to max."""
        from minivess.serving.gradio_demo import extract_slice

        vol = np.random.default_rng(42).random((10, 20, 30), dtype=np.float32)
        s = extract_slice(vol, axis=0, index=999)
        expected = vol[9]
        np.testing.assert_array_equal(s, expected.astype(np.float32))

    def test_extract_slice_all_axes(self) -> None:
        """extract_slice should work for all three axes."""
        from minivess.serving.gradio_demo import extract_slice

        vol = np.random.default_rng(42).random((10, 20, 30), dtype=np.float32)
        for axis, expected_shape in [(0, (20, 30)), (1, (10, 30)), (2, (10, 20))]:
            s = extract_slice(vol, axis=axis, index=0)
            assert s.shape == expected_shape

    def test_extract_slice_returns_float32(self) -> None:
        """extract_slice should always return float32."""
        from minivess.serving.gradio_demo import extract_slice

        vol = np.ones((5, 5, 5), dtype=np.float64)
        s = extract_slice(vol, axis=0, index=0)
        assert s.dtype == np.float32
