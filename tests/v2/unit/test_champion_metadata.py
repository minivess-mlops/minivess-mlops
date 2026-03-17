"""Tests for champion metadata logging for manuscript.

PR-D T4 (Issue #828): Log comprehensive champion metadata to MLflow
for the manuscript, including architecture, performance, training,
post-training, and deployment details.

TDD Phase: RED — tests written before implementation.
"""

from __future__ import annotations

from typing import Any


def _make_champion_with_full_metadata() -> dict[str, Any]:
    """Create a champion dict with all metadata fields populated."""
    return {
        "run_id": "abc123",
        "experiment_id": "1",
        "model": "dynunet",
        "loss": "cbdice_cldice",
        "aux_calib": False,
        "fold_strategy": "cv_average",
        "ensemble": "none",
        "cldice": 0.85,
        "masd": 0.5,
        "dsc": 0.88,
        "compound_metric": 0.80,
        # Architecture details
        "param_count": 3_200_000,
        "vram_gb": 2.9,
        "flops_estimate": 45_000_000_000,
        # Training details
        "max_epochs": 100,
        "batch_size": 2,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingWarmRestarts",
        # Post-training details
        "swa_method": "none",
        "recalibration": "none",
        # Performance CIs
        "dsc_ci95_lo": 0.84,
        "dsc_ci95_hi": 0.92,
        "cldice_ci95_lo": 0.81,
        "cldice_ci95_hi": 0.89,
        "masd_ci95_lo": 0.3,
        "masd_ci95_hi": 0.7,
        # Deployment details
        "onnx_opset": 17,
        "bento_tag": "minivess-balanced:abc123",
    }


class TestChampionMetadataComplete:
    """Champion metadata contains all required fields."""

    def test_champion_metadata_complete(self) -> None:
        """Build metadata produces all required field groups."""
        from minivess.serving.champion_metadata import (
            build_champion_metadata,
        )

        champion = _make_champion_with_full_metadata()
        metadata = build_champion_metadata(champion)

        # Must have all 6 field groups
        assert "architecture" in metadata
        assert "training" in metadata
        assert "performance" in metadata
        assert "post_training" in metadata
        assert "deployment" in metadata
        assert "factorial" in metadata


class TestChampionMetadataArchitectureFields:
    """Architecture fields include param count and VRAM."""

    def test_champion_metadata_architecture_fields(self) -> None:
        """Architecture section includes param_count and vram_gb."""
        from minivess.serving.champion_metadata import (
            build_champion_metadata,
        )

        champion = _make_champion_with_full_metadata()
        metadata = build_champion_metadata(champion)
        arch = metadata["architecture"]

        assert "param_count" in arch
        assert arch["param_count"] == 3_200_000
        assert "vram_gb" in arch
        assert arch["vram_gb"] == 2.9
        assert "model" in arch
        assert arch["model"] == "dynunet"


class TestChampionMetadataPerformanceFields:
    """Performance fields include primary metrics with CIs."""

    def test_champion_metadata_performance_fields(self) -> None:
        """Performance section includes DSC, clDice, MASD with CIs."""
        from minivess.serving.champion_metadata import (
            build_champion_metadata,
        )

        champion = _make_champion_with_full_metadata()
        metadata = build_champion_metadata(champion)
        perf = metadata["performance"]

        assert perf["dsc"] == 0.88
        assert perf["cldice"] == 0.85
        assert perf["masd"] == 0.5
        assert perf["dsc_ci95_lo"] == 0.84
        assert perf["dsc_ci95_hi"] == 0.92
        assert perf["compound_metric"] == 0.80


class TestChampionMetadataTripodLink:
    """TRIPOD compliance link in metadata."""

    def test_champion_metadata_tripod_link(self) -> None:
        """Metadata includes TRIPOD compliance matrix link."""
        from minivess.serving.champion_metadata import (
            build_champion_metadata,
        )

        champion = _make_champion_with_full_metadata()
        metadata = build_champion_metadata(champion)

        assert "tripod_compliance" in metadata
        assert metadata["tripod_compliance"]["model_specification"] == "TRIPOD-22"

    def test_champion_metadata_to_mlflow_params(self) -> None:
        """Metadata can be flattened to MLflow param dict."""
        from minivess.serving.champion_metadata import (
            build_champion_metadata,
            flatten_metadata_for_mlflow,
        )

        champion = _make_champion_with_full_metadata()
        metadata = build_champion_metadata(champion)
        params = flatten_metadata_for_mlflow(metadata)

        # All flattened keys should use slash prefix
        for key in params:
            assert "/" in key, f"Key {key} missing slash prefix"

        assert params["champion/arch/param_count"] == "3200000"
        assert params["champion/perf/dsc"] == "0.88"
        assert params["champion/deploy/onnx_opset"] == "17"
