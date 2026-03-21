"""Tests for uncertainty storage policy — debug vs production (#887).

Debug mode: summary stats only. Production mode: full 5D maps + summaries.
"""

from __future__ import annotations

from pathlib import Path

import torch


def _make_uncertainty_maps(
    b: int = 1, d: int = 4, h: int = 4, w: int = 4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create synthetic uncertainty tensors for testing."""
    total = torch.rand(b, 1, d, h, w)
    aleatoric = torch.rand(b, 1, d, h, w) * 0.5
    epistemic = total - aleatoric
    return total, aleatoric, epistemic


class TestUncertaintyStoragePolicyDebug:
    """Debug mode: store only scalar summaries, no maps."""

    def test_debug_returns_summaries(self, tmp_path: Path) -> None:
        from minivess.ensemble.storage_policy import UncertaintyStoragePolicy

        policy = UncertaintyStoragePolicy(debug=True, output_dir=tmp_path)
        total, aleatoric, epistemic = _make_uncertainty_maps()

        result = policy.store(
            volume_id="vol_001",
            total_uncertainty=total,
            aleatoric_uncertainty=aleatoric,
            epistemic_uncertainty=epistemic,
        )

        assert "summaries" in result
        assert result["summaries"]["total_mean"] > 0
        assert result["summaries"]["aleatoric_mean"] >= 0

    def test_debug_does_not_save_maps(self, tmp_path: Path) -> None:
        from minivess.ensemble.storage_policy import UncertaintyStoragePolicy

        policy = UncertaintyStoragePolicy(debug=True, output_dir=tmp_path)
        total, aleatoric, epistemic = _make_uncertainty_maps()

        result = policy.store(
            volume_id="vol_001",
            total_uncertainty=total,
            aleatoric_uncertainty=aleatoric,
            epistemic_uncertainty=epistemic,
        )

        assert result["maps_saved"] is False
        assert "map_path" not in result
        # No .pt files should exist
        pt_files = list(tmp_path.rglob("*.pt"))
        assert len(pt_files) == 0

    def test_debug_has_all_summary_keys(self, tmp_path: Path) -> None:
        from minivess.ensemble.storage_policy import UncertaintyStoragePolicy

        policy = UncertaintyStoragePolicy(debug=True, output_dir=tmp_path)
        total, aleatoric, epistemic = _make_uncertainty_maps()

        result = policy.store(
            volume_id="vol_001",
            total_uncertainty=total,
            aleatoric_uncertainty=aleatoric,
            epistemic_uncertainty=epistemic,
        )

        expected_keys = {
            "total_mean",
            "total_max",
            "aleatoric_mean",
            "aleatoric_max",
            "epistemic_mean",
            "epistemic_max",
        }
        assert expected_keys == set(result["summaries"].keys())


class TestUncertaintyStoragePolicyProduction:
    """Production mode: save full 5D maps + summaries."""

    def test_production_saves_maps(self, tmp_path: Path) -> None:
        from minivess.ensemble.storage_policy import UncertaintyStoragePolicy

        policy = UncertaintyStoragePolicy(debug=False, output_dir=tmp_path)
        total, aleatoric, epistemic = _make_uncertainty_maps()

        result = policy.store(
            volume_id="vol_042",
            total_uncertainty=total,
            aleatoric_uncertainty=aleatoric,
            epistemic_uncertainty=epistemic,
        )

        assert result["maps_saved"] is True
        assert "map_path" in result
        assert Path(result["map_path"]).exists()

    def test_production_map_contains_all_components(self, tmp_path: Path) -> None:
        from minivess.ensemble.storage_policy import UncertaintyStoragePolicy

        policy = UncertaintyStoragePolicy(debug=False, output_dir=tmp_path)
        total, aleatoric, epistemic = _make_uncertainty_maps()

        result = policy.store(
            volume_id="vol_042",
            total_uncertainty=total,
            aleatoric_uncertainty=aleatoric,
            epistemic_uncertainty=epistemic,
        )

        loaded = torch.load(result["map_path"], weights_only=True)
        assert "total" in loaded
        assert "aleatoric" in loaded
        assert "epistemic" in loaded

    def test_production_also_has_summaries(self, tmp_path: Path) -> None:
        from minivess.ensemble.storage_policy import UncertaintyStoragePolicy

        policy = UncertaintyStoragePolicy(debug=False, output_dir=tmp_path)
        total, aleatoric, epistemic = _make_uncertainty_maps()

        result = policy.store(
            volume_id="vol_042",
            total_uncertainty=total,
            aleatoric_uncertainty=aleatoric,
            epistemic_uncertainty=epistemic,
        )

        assert "summaries" in result
        assert result["summaries"]["total_mean"] > 0

    def test_multiple_volumes_create_separate_files(self, tmp_path: Path) -> None:
        from minivess.ensemble.storage_policy import UncertaintyStoragePolicy

        policy = UncertaintyStoragePolicy(debug=False, output_dir=tmp_path)

        for vol_id in ["vol_001", "vol_002", "vol_003"]:
            total, aleatoric, epistemic = _make_uncertainty_maps()
            policy.store(
                volume_id=vol_id,
                total_uncertainty=total,
                aleatoric_uncertainty=aleatoric,
                epistemic_uncertainty=epistemic,
            )

        pt_files = list(tmp_path.rglob("*.pt"))
        assert len(pt_files) == 3
