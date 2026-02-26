"""Unit tests for serving components (ONNX, BentoML, Gradio)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path
import pytest
import torch


class TestOnnxInferenceMetadata:
    """Test ONNX inference module structure."""

    def test_module_importable(self) -> None:
        pytest.importorskip("onnxruntime")
        from minivess.serving import onnx_inference

        assert hasattr(onnx_inference, "OnnxSegmentationInference")

    def test_class_has_predict_method(self) -> None:
        pytest.importorskip("onnxruntime")
        from minivess.serving.onnx_inference import OnnxSegmentationInference

        assert hasattr(OnnxSegmentationInference, "predict")
        assert hasattr(OnnxSegmentationInference, "get_metadata")

    def test_class_has_init_signature(self) -> None:
        """Constructor should accept model_path and use_gpu kwargs."""
        pytest.importorskip("onnxruntime")
        import inspect

        from minivess.serving.onnx_inference import OnnxSegmentationInference

        sig = inspect.signature(OnnxSegmentationInference.__init__)
        params = list(sig.parameters.keys())
        assert "model_path" in params
        assert "use_gpu" in params

    def test_predict_returns_dict_keys(self) -> None:
        """Predict method signature should match expected contract."""
        pytest.importorskip("onnxruntime")
        import inspect

        from minivess.serving.onnx_inference import OnnxSegmentationInference

        sig = inspect.signature(OnnxSegmentationInference.predict)
        params = list(sig.parameters.keys())
        assert "volume" in params


class TestOnnxExportRoundtrip:
    """Export SegResNet to ONNX and run inference."""

    def test_onnx_export_creates_file(self, tmp_path: Path) -> None:
        from minivess.adapters.segresnet import SegResNetAdapter
        from minivess.config.models import ModelConfig, ModelFamily

        config = ModelConfig(
            family=ModelFamily.MONAI_SEGRESNET,
            name="onnx-test",
            in_channels=1,
            out_channels=2,
        )
        adapter = SegResNetAdapter(config)
        onnx_path = tmp_path / "model.onnx"
        example = torch.randn(1, 1, 16, 16, 8)
        adapter.export_onnx(onnx_path, example)

        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 0

    def test_onnx_roundtrip_inference(self, tmp_path: Path) -> None:
        """Export → load → predict should produce valid output."""
        pytest.importorskip("onnxruntime")
        from minivess.adapters.segresnet import SegResNetAdapter
        from minivess.config.models import ModelConfig, ModelFamily
        from minivess.serving.onnx_inference import OnnxSegmentationInference

        config = ModelConfig(
            family=ModelFamily.MONAI_SEGRESNET,
            name="roundtrip",
            in_channels=1,
            out_channels=2,
        )
        adapter = SegResNetAdapter(config)
        onnx_path = tmp_path / "model.onnx"
        example = torch.randn(1, 1, 16, 16, 8)
        adapter.export_onnx(onnx_path, example)

        engine = OnnxSegmentationInference(onnx_path)
        volume = np.random.randn(1, 1, 16, 16, 8).astype(np.float32)
        result = engine.predict(volume)

        assert result.segmentation is not None
        assert result.probabilities is not None
        assert result.shape is not None

    def test_onnx_output_shape_matches_pytorch(self, tmp_path: Path) -> None:
        """ONNX output shape should match PyTorch model output."""
        pytest.importorskip("onnxruntime")
        from minivess.adapters.segresnet import SegResNetAdapter
        from minivess.config.models import ModelConfig, ModelFamily
        from minivess.serving.onnx_inference import OnnxSegmentationInference

        config = ModelConfig(
            family=ModelFamily.MONAI_SEGRESNET,
            name="shape-check",
            in_channels=1,
            out_channels=2,
        )
        adapter = SegResNetAdapter(config)
        onnx_path = tmp_path / "model.onnx"
        example = torch.randn(1, 1, 16, 16, 8)
        adapter.export_onnx(onnx_path, example)

        # PyTorch output
        adapter.net.eval()
        with torch.no_grad():
            pt_out = adapter.net(example)
        pt_shape = pt_out.shape  # (1, 2, 16, 16, 8)

        # ONNX output
        engine = OnnxSegmentationInference(onnx_path)
        onnx_result = engine.predict(example.numpy())
        onnx_seg_shape = onnx_result.segmentation.shape  # (1, 16, 16, 8)
        onnx_prob_shape = onnx_result.probabilities.shape  # (1, 2, 16, 16, 8)

        assert onnx_prob_shape == tuple(pt_shape), (
            f"ONNX probs shape {onnx_prob_shape} != PyTorch {tuple(pt_shape)}"
        )
        assert onnx_seg_shape == (1, 16, 16, 8)  # B, D, H, W (no class dim)

    def test_onnx_metadata(self, tmp_path: Path) -> None:
        """ONNX engine should report model metadata."""
        pytest.importorskip("onnxruntime")
        from minivess.adapters.segresnet import SegResNetAdapter
        from minivess.config.models import ModelConfig, ModelFamily
        from minivess.serving.onnx_inference import OnnxSegmentationInference

        config = ModelConfig(
            family=ModelFamily.MONAI_SEGRESNET,
            name="meta",
            in_channels=1,
            out_channels=2,
        )
        adapter = SegResNetAdapter(config)
        onnx_path = tmp_path / "model.onnx"
        adapter.export_onnx(onnx_path, torch.randn(1, 1, 16, 16, 8))

        engine = OnnxSegmentationInference(onnx_path)
        meta = engine.get_metadata()
        assert meta.inputs is not None
        assert meta.outputs is not None
        assert len(meta.inputs) > 0


class TestBentoServiceStructure:
    """Test BentoML service structure."""

    def test_module_importable(self) -> None:
        from minivess.serving import bento_service

        assert hasattr(bento_service, "SegmentationService")
        assert hasattr(bento_service, "BENTO_MODEL_TAG")

    def test_model_tag(self) -> None:
        from minivess.serving.bento_service import BENTO_MODEL_TAG

        assert isinstance(BENTO_MODEL_TAG, str)
        assert len(BENTO_MODEL_TAG) > 0

    def test_model_tag_value(self) -> None:
        from minivess.serving.bento_service import BENTO_MODEL_TAG

        assert BENTO_MODEL_TAG == "minivess-segmentor"

    def test_service_has_predict_api(self) -> None:
        """BentoML wraps the class; 'predict' should be a registered API."""
        from minivess.serving.bento_service import SegmentationService

        assert "predict" in SegmentationService.apis

    def test_service_has_health_api(self) -> None:
        """BentoML wraps the class; 'health' should be a registered API."""
        from minivess.serving.bento_service import SegmentationService

        assert "health" in SegmentationService.apis

    def test_inner_class_has_predict(self) -> None:
        """The unwrapped inner class should expose predict."""
        from minivess.serving.bento_service import SegmentationService

        assert hasattr(SegmentationService.inner, "predict")

    def test_predict_accepts_numpy_volume(self) -> None:
        """Predict method should accept a volume parameter."""
        import inspect

        from minivess.serving.bento_service import SegmentationService

        # BentoML wraps methods into APIMethod objects; unwrap via .func
        api_method = SegmentationService.inner.predict
        func = getattr(api_method, "func", api_method)
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        assert "volume" in params


class TestGradioDemo:
    """Test Gradio demo module."""

    def test_module_importable(self) -> None:
        pytest.importorskip("gradio")
        from minivess.serving import gradio_demo

        assert hasattr(gradio_demo, "build_demo")
        assert hasattr(gradio_demo, "main")

    def test_build_demo_returns_blocks(self) -> None:
        gr = pytest.importorskip("gradio")
        from minivess.serving.gradio_demo import build_demo

        demo = build_demo(model_path=None)
        assert isinstance(demo, gr.Blocks)

    def test_build_demo_accepts_model_path(self) -> None:
        """build_demo should accept a model_path keyword argument."""
        pytest.importorskip("gradio")
        import inspect

        from minivess.serving.gradio_demo import build_demo

        sig = inspect.signature(build_demo)
        params = list(sig.parameters.keys())
        assert "model_path" in params

    def test_main_callable(self) -> None:
        pytest.importorskip("gradio")
        from minivess.serving.gradio_demo import main

        assert callable(main)

    def test_predict_slice_dummy_mode(self) -> None:
        """Dummy mode should produce a thresholded mask."""
        pytest.importorskip("gradio")
        from minivess.serving.gradio_demo import build_demo

        demo = build_demo(model_path=None)
        # Access the predict function via Gradio's API
        # The function is the first registered fn
        predict_fn = demo.fns[0].fn

        # Create a simple test image
        test_input = np.random.rand(64, 64).astype(np.float32)
        mask, info = predict_fn(test_input)

        assert mask is not None
        assert mask.shape == test_input.shape
        assert "Demo mode" in info

    def test_predict_slice_none_input(self) -> None:
        """None input should return None mask."""
        pytest.importorskip("gradio")
        from minivess.serving.gradio_demo import build_demo

        demo = build_demo(model_path=None)
        predict_fn = demo.fns[0].fn

        mask, info = predict_fn(None)
        assert mask is None
        assert "No input" in info


class TestGradioNiftiHandling:
    """Test NIfTI volume handling in Gradio demo."""

    def test_load_nifti_volume(self, tmp_path: Path) -> None:
        """Should load a NIfTI file and return numpy array."""
        # Create a synthetic NIfTI file
        import nibabel as nib

        from minivess.serving.gradio_demo import load_nifti_volume

        data = np.random.rand(32, 32, 16).astype(np.float32)
        nii = nib.Nifti1Image(data, affine=np.eye(4))
        nifti_path = tmp_path / "test.nii.gz"
        nib.save(nii, nifti_path)

        volume = load_nifti_volume(str(nifti_path))
        assert volume.shape == (32, 32, 16)
        assert volume.dtype == np.float32

    def test_extract_slice(self) -> None:
        """Should extract 2D slice from 3D volume."""
        from minivess.serving.gradio_demo import extract_slice

        volume = np.random.rand(32, 32, 16).astype(np.float32)
        axial = extract_slice(volume, axis=2, index=8)
        assert axial.shape == (32, 32)

        sagittal = extract_slice(volume, axis=0, index=16)
        assert sagittal.shape == (32, 16)

    def test_extract_slice_clamps_index(self) -> None:
        """Out-of-range index should be clamped."""
        from minivess.serving.gradio_demo import extract_slice

        volume = np.random.rand(32, 32, 16).astype(np.float32)
        # Index beyond range should clamp
        s = extract_slice(volume, axis=2, index=999)
        assert s.shape == (32, 32)


class TestOnnxServingIntegration:
    """End-to-end: export → load → predict → valid output."""

    def test_export_serve_predict_roundtrip(self, tmp_path: Path) -> None:
        """Full serving pipeline: adapter → ONNX → engine → prediction."""
        pytest.importorskip("onnxruntime")
        from minivess.adapters.segresnet import SegResNetAdapter
        from minivess.config.models import ModelConfig, ModelFamily
        from minivess.serving.onnx_inference import OnnxSegmentationInference

        # Create and export model
        config = ModelConfig(
            family=ModelFamily.MONAI_SEGRESNET,
            name="e2e-test",
            in_channels=1,
            out_channels=2,
        )
        adapter = SegResNetAdapter(config)
        onnx_path = tmp_path / "model.onnx"
        adapter.export_onnx(onnx_path, torch.randn(1, 1, 16, 16, 8))

        # Load and predict
        engine = OnnxSegmentationInference(onnx_path)
        volume = np.random.randn(1, 1, 16, 16, 8).astype(np.float32)
        result = engine.predict(volume)

        # Validate output
        seg = result.segmentation
        probs = result.probabilities
        assert seg.shape == (1, 16, 16, 8)
        assert probs.shape == (1, 2, 16, 16, 8)
        assert seg.min() >= 0
        assert seg.max() <= 1  # Binary segmentation (2 classes)
        # Probabilities should sum to ~1 along class axis
        prob_sums = probs.sum(axis=1)
        np.testing.assert_allclose(prob_sums, 1.0, atol=1e-5)
