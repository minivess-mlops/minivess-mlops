"""Tests for DVC drift simulation setup (T-A2).

Tests VesselNN batch partitioning into 6 batches of 2 volumes
for drift simulation, config generation, and batch validation.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# T-A2.1: Batch partitioning
# ---------------------------------------------------------------------------


class TestBatchPartitioning:
    """Test VesselNN volume partitioning into drift simulation batches."""

    def test_partition_12_into_6_batches(self) -> None:
        """12 volumes should partition into 6 batches of 2."""
        from minivess.data.drift_simulation_setup import partition_vesselnn_batches

        volume_ids = [f"vol_{i:03d}" for i in range(12)]
        batches = partition_vesselnn_batches(volume_ids, seed=42)

        assert len(batches) == 6
        for batch in batches:
            assert len(batch) == 2

    def test_partition_deterministic(self) -> None:
        """Same input should produce same partitioning."""
        from minivess.data.drift_simulation_setup import partition_vesselnn_batches

        volume_ids = [f"vol_{i:03d}" for i in range(12)]
        b1 = partition_vesselnn_batches(volume_ids, seed=42)
        b2 = partition_vesselnn_batches(volume_ids, seed=42)

        assert b1 == b2

    def test_partition_different_seed_different_result(self) -> None:
        """Different seeds should produce different partitions."""
        from minivess.data.drift_simulation_setup import partition_vesselnn_batches

        volume_ids = [f"vol_{i:03d}" for i in range(12)]
        b1 = partition_vesselnn_batches(volume_ids, seed=42)
        b2 = partition_vesselnn_batches(volume_ids, seed=99)

        assert b1 != b2

    def test_partition_all_volumes_present(self) -> None:
        """All volumes should appear exactly once across batches."""
        from minivess.data.drift_simulation_setup import partition_vesselnn_batches

        volume_ids = [f"vol_{i:03d}" for i in range(12)]
        batches = partition_vesselnn_batches(volume_ids, seed=42)

        all_vols = [v for batch in batches for v in batch]
        assert sorted(all_vols) == sorted(volume_ids)

    def test_partition_wrong_count_raises(self) -> None:
        """Non-12 volume count should raise ValueError."""
        from minivess.data.drift_simulation_setup import partition_vesselnn_batches

        with pytest.raises(ValueError, match="12"):
            partition_vesselnn_batches([f"vol_{i}" for i in range(10)], seed=42)


# ---------------------------------------------------------------------------
# T-A2.2: Config file generation
# ---------------------------------------------------------------------------


class TestDriftSimulationConfig:
    """Test drift simulation config generation."""

    def test_generate_config(self) -> None:
        """Should generate a valid config dict."""
        from minivess.data.drift_simulation_setup import (
            generate_drift_simulation_config,
        )

        config = generate_drift_simulation_config(seed=42)

        assert "batches" in config
        assert "n_batches" in config
        assert "seed" in config
        assert config["n_batches"] == 6
        assert config["seed"] == 42

    def test_config_batch_structure(self) -> None:
        """Each batch in config should have id and volume_ids."""
        from minivess.data.drift_simulation_setup import (
            generate_drift_simulation_config,
        )

        config = generate_drift_simulation_config(seed=42)

        for batch in config["batches"]:
            assert "batch_id" in batch
            assert "volume_ids" in batch
            assert isinstance(batch["batch_id"], int)
            assert len(batch["volume_ids"]) == 2

    def test_save_config(self, tmp_path: Path) -> None:
        """Config should save to JSON."""
        from minivess.data.drift_simulation_setup import (
            generate_drift_simulation_config,
            save_drift_simulation_config,
        )

        config = generate_drift_simulation_config(seed=42)
        out_path = tmp_path / "vesselnn_drift_simulation.json"
        save_drift_simulation_config(config, out_path)

        assert out_path.exists()
        loaded = json.loads(out_path.read_text(encoding="utf-8"))
        assert loaded["n_batches"] == 6

    def test_load_config(self, tmp_path: Path) -> None:
        """Saved config should load correctly."""
        from minivess.data.drift_simulation_setup import (
            generate_drift_simulation_config,
            load_drift_simulation_config,
            save_drift_simulation_config,
        )

        config = generate_drift_simulation_config(seed=42)
        out_path = tmp_path / "vesselnn_drift_simulation.json"
        save_drift_simulation_config(config, out_path)

        loaded = load_drift_simulation_config(out_path)
        assert loaded == config


# ---------------------------------------------------------------------------
# T-A2.3: Git tag generation
# ---------------------------------------------------------------------------


class TestGitTagGeneration:
    """Test batch git tag name generation."""

    def test_generate_batch_tags(self) -> None:
        """Should produce 6 git tag names."""
        from minivess.data.drift_simulation_setup import generate_batch_tags

        tags = generate_batch_tags()
        assert len(tags) == 6
        assert tags[0] == "data/vesselnn/batch-1"
        assert tags[5] == "data/vesselnn/batch-6"

    def test_tag_format(self) -> None:
        """Tags should follow the DVC convention."""
        from minivess.data.drift_simulation_setup import generate_batch_tags

        for tag in generate_batch_tags():
            assert tag.startswith("data/vesselnn/batch-")
