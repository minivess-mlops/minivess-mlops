"""Tests for active learning ABCs and dataclasses.

TDD RED phase for Task T-F1 (Issue #776).
Architecture-only: interfaces and stubs, no full implementation.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestUncertaintySamplerABC:
    """Test the UncertaintySampler abstract base class."""

    def test_abc_cannot_be_instantiated(self) -> None:
        from minivess.active_learning.base import UncertaintySampler

        with pytest.raises(TypeError, match="abstract"):
            UncertaintySampler()  # type: ignore[abstract]

    def test_abc_requires_select_samples(self) -> None:
        from minivess.active_learning.base import UncertaintySampler

        class BadSampler(UncertaintySampler):
            @property
            def strategy_name(self) -> str:
                return "bad"

        with pytest.raises(TypeError, match="abstract"):
            BadSampler()  # type: ignore[abstract]

    def test_concrete_subclass_instantiable(self) -> None:
        from minivess.active_learning.base import UncertaintySampler

        class MaxEntropySampler(UncertaintySampler):
            def select_samples(
                self,
                uncertainty_scores: np.ndarray,
                n: int,
            ) -> np.ndarray:
                return np.argsort(uncertainty_scores)[-n:][::-1]

            @property
            def strategy_name(self) -> str:
                return "max_entropy"

        sampler = MaxEntropySampler()
        assert sampler.strategy_name == "max_entropy"

    def test_select_samples_returns_indices(self) -> None:
        from minivess.active_learning.base import UncertaintySampler

        class SimpleSampler(UncertaintySampler):
            def select_samples(
                self,
                uncertainty_scores: np.ndarray,
                n: int,
            ) -> np.ndarray:
                return np.argsort(uncertainty_scores)[-n:][::-1]

            @property
            def strategy_name(self) -> str:
                return "simple"

        sampler = SimpleSampler()
        scores = np.array([0.1, 0.9, 0.5, 0.3, 0.7])
        indices = sampler.select_samples(scores, n=2)
        assert len(indices) == 2
        assert indices[0] == 1  # highest uncertainty


class TestAnnotationRequest:
    """Test the AnnotationRequest dataclass."""

    def test_annotation_request_fields(self) -> None:
        from minivess.active_learning.base import AnnotationRequest

        req = AnnotationRequest(
            volume_id="vol_001",
            uncertainty_score=0.85,
            source="mc_dropout",
            priority=1,
        )
        assert req.volume_id == "vol_001"
        assert req.uncertainty_score == 0.85
        assert req.source == "mc_dropout"
        assert req.priority == 1

    def test_annotation_request_is_dataclass(self) -> None:
        import dataclasses

        from minivess.active_learning.base import AnnotationRequest

        assert dataclasses.is_dataclass(AnnotationRequest)

    def test_annotation_request_optional_metadata(self) -> None:
        from minivess.active_learning.base import AnnotationRequest

        req = AnnotationRequest(
            volume_id="vol_002",
            uncertainty_score=0.5,
            source="mahalanobis",
            priority=2,
            metadata={"embedding_distance": 3.14},
        )
        assert req.metadata is not None
        assert req.metadata["embedding_distance"] == 3.14

    def test_annotation_request_default_metadata_is_none(self) -> None:
        from minivess.active_learning.base import AnnotationRequest

        req = AnnotationRequest(
            volume_id="vol_003",
            uncertainty_score=0.1,
            source="bald",
            priority=3,
        )
        assert req.metadata is None


class TestStrategyRegistry:
    """Test the available strategy types."""

    def test_available_strategies_exist(self) -> None:
        from minivess.active_learning import SAMPLING_STRATEGIES

        assert isinstance(SAMPLING_STRATEGIES, list)
        assert "max_entropy" in SAMPLING_STRATEGIES
        assert "max_mc_variance" in SAMPLING_STRATEGIES
        assert "bald" in SAMPLING_STRATEGIES
        assert "max_mahalanobis" in SAMPLING_STRATEGIES

    def test_module_exports(self) -> None:
        from minivess.active_learning import (
            AnnotationRequest,
            UncertaintySampler,
        )

        assert UncertaintySampler is not None
        assert AnnotationRequest is not None


class TestMONAILabelAdapterStub:
    """Test the MONAI Label adapter interface."""

    def test_monai_label_adapter_abc_exists(self) -> None:
        from minivess.active_learning.base import MONAILabelAdapter

        with pytest.raises(TypeError, match="abstract"):
            MONAILabelAdapter()  # type: ignore[abstract]

    def test_monai_label_adapter_has_submit_method(self) -> None:
        from minivess.active_learning.base import MONAILabelAdapter

        assert hasattr(MONAILabelAdapter, "submit_for_annotation")

    def test_monai_label_adapter_has_fetch_method(self) -> None:
        from minivess.active_learning.base import MONAILabelAdapter

        assert hasattr(MONAILabelAdapter, "fetch_annotations")
