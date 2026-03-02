"""Tests for enhanced deploy server API.

Covers:
- SegmentationRequest validation
- SegmentationResponse serialization
- OutputMode / ModelName / UQMethod enums
- ModelRegistryServer model routing and lazy load
- InferenceClient protocol
- LocalInferenceClient predict
- Error handling for unknown models

Closes #185.
"""

from __future__ import annotations

import numpy as np
import pytest

from minivess.serving.api_models import (
    ModelName,
    OutputMode,
    SegmentationRequest,
    SegmentationResponse,
)


class TestOutputMode:
    """Test OutputMode enum."""

    def test_binary_value(self) -> None:
        assert OutputMode.BINARY.value == "binary"

    def test_probabilities_value(self) -> None:
        assert OutputMode.PROBABILITIES.value == "probabilities"

    def test_full_value(self) -> None:
        assert OutputMode.FULL.value == "full"

    def test_uq_value(self) -> None:
        assert OutputMode.UQ.value == "uq"


class TestModelName:
    """Test ModelName enum."""

    def test_balanced(self) -> None:
        assert ModelName.BALANCED.value == "balanced"

    def test_topology(self) -> None:
        assert ModelName.TOPOLOGY.value == "topology"

    def test_overlap(self) -> None:
        assert ModelName.OVERLAP.value == "overlap"


class TestSegmentationRequest:
    """Test SegmentationRequest validation."""

    def test_defaults(self) -> None:
        volume = np.zeros((8, 8, 8), dtype=np.float32)
        req = SegmentationRequest(volume=volume)
        assert req.model_name == "balanced"
        assert req.output_mode == "binary"
        assert req.confidence_level == pytest.approx(0.95)

    def test_validate_valid(self) -> None:
        volume = np.zeros((8, 8, 8), dtype=np.float32)
        req = SegmentationRequest(volume=volume)
        errors = req.validate()
        assert errors == []

    def test_validate_invalid_output_mode(self) -> None:
        volume = np.zeros((8, 8, 8), dtype=np.float32)
        req = SegmentationRequest(volume=volume, output_mode="invalid")
        errors = req.validate()
        assert any("output_mode" in e for e in errors)

    def test_validate_2d_volume(self) -> None:
        volume = np.zeros((8, 8), dtype=np.float32)
        req = SegmentationRequest(volume=volume)
        errors = req.validate()
        assert any("3D" in e or "dimension" in e.lower() for e in errors)

    def test_validate_confidence_out_of_range(self) -> None:
        volume = np.zeros((8, 8, 8), dtype=np.float32)
        req = SegmentationRequest(volume=volume, confidence_level=1.5)
        errors = req.validate()
        assert any("confidence" in e.lower() for e in errors)


class TestSegmentationResponse:
    """Test SegmentationResponse serialization."""

    def test_to_dict_keys(self) -> None:
        resp = SegmentationResponse(
            segmentation=np.ones((8, 8, 8), dtype=np.int64),
            shape=[8, 8, 8],
            model_name="balanced",
            inference_time_ms=42.0,
        )
        d = resp.to_dict()
        assert "segmentation" in d
        assert "shape" in d
        assert "model_name" in d
        assert "inference_time_ms" in d

    def test_to_dict_segmentation_is_list(self) -> None:
        resp = SegmentationResponse(
            segmentation=np.ones((4, 4, 4), dtype=np.int64),
            shape=[4, 4, 4],
            model_name="balanced",
            inference_time_ms=10.0,
        )
        d = resp.to_dict()
        assert isinstance(d["segmentation"], list)

    def test_to_dict_optional_probabilities(self) -> None:
        resp = SegmentationResponse(
            segmentation=np.ones((4, 4, 4), dtype=np.int64),
            shape=[4, 4, 4],
            model_name="balanced",
            inference_time_ms=10.0,
            probabilities=np.ones((2, 4, 4, 4), dtype=np.float32),
        )
        d = resp.to_dict()
        assert d["probabilities"] is not None

    def test_to_dict_no_probabilities_by_default(self) -> None:
        resp = SegmentationResponse(
            segmentation=np.ones((4, 4, 4), dtype=np.int64),
            shape=[4, 4, 4],
            model_name="balanced",
            inference_time_ms=10.0,
        )
        d = resp.to_dict()
        assert d["probabilities"] is None


class TestModelRegistryServer:
    """Test ModelRegistryServer routing and lazy loading."""

    def test_health_empty_registry(self) -> None:
        from minivess.serving.model_registry_server import ModelRegistryServer

        server = ModelRegistryServer(model_paths={})
        health = server.health()
        assert health["status"] == "healthy"
        assert health["available_models"] == []

    def test_predict_unknown_model_raises(self) -> None:
        from minivess.serving.model_registry_server import ModelRegistryServer

        server = ModelRegistryServer(model_paths={})
        req = SegmentationRequest(
            volume=np.zeros((8, 8, 8), dtype=np.float32),
            model_name="nonexistent",
        )
        with pytest.raises(KeyError, match="nonexistent"):
            server.predict(req)


class TestInferenceClient:
    """Test InferenceClient protocol and LocalInferenceClient."""

    def test_local_client_health(self) -> None:
        from minivess.serving.inference_client import LocalInferenceClient
        from minivess.serving.model_registry_server import ModelRegistryServer

        server = ModelRegistryServer(model_paths={})
        client = LocalInferenceClient(server)
        health = client.health()
        assert health["status"] == "healthy"

    def test_local_client_predict_delegates(self) -> None:
        """LocalInferenceClient.predict should delegate to server."""
        from minivess.serving.inference_client import LocalInferenceClient
        from minivess.serving.model_registry_server import ModelRegistryServer

        server = ModelRegistryServer(model_paths={})
        client = LocalInferenceClient(server)
        req = SegmentationRequest(
            volume=np.zeros((8, 8, 8), dtype=np.float32),
            model_name="nonexistent",
        )
        with pytest.raises(KeyError):
            client.predict(req)
