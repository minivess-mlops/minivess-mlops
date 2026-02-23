"""Unit tests for serving components (ONNX, BentoML, Gradio)."""

from __future__ import annotations

import pytest


class TestOnnxInferenceMetadata:
    """Test ONNX inference module structure."""

    def test_module_importable(self) -> None:
        ort = pytest.importorskip("onnxruntime")  # noqa: F841
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

        assert hasattr(gradio_demo, "create_demo")
        assert hasattr(gradio_demo, "main")

    def test_create_demo_returns_blocks(self) -> None:
        gr = pytest.importorskip("gradio")
        from minivess.serving.gradio_demo import create_demo

        demo = create_demo(model_path=None)
        assert isinstance(demo, gr.Blocks)

    def test_create_demo_accepts_model_path(self) -> None:
        """create_demo should accept a model_path keyword argument."""
        pytest.importorskip("gradio")
        import inspect

        from minivess.serving.gradio_demo import create_demo

        sig = inspect.signature(create_demo)
        params = list(sig.parameters.keys())
        assert "model_path" in params

    def test_main_callable(self) -> None:
        pytest.importorskip("gradio")
        from minivess.serving.gradio_demo import main

        assert callable(main)
