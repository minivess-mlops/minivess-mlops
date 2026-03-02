"""Tests for Disconnect-to-Connect 3D augmentation (T3 — #228, T4 — #229, T5 — #230)."""

from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np

from minivess.data.disconnect_augmentation import DisconnectToConnectd


def _make_y_junction_mask(shape: tuple[int, int, int] = (32, 64, 64)) -> np.ndarray:
    """Create a synthetic Y-shaped vessel for testing junction detection.

    Three tubes meeting at the center: one along +Z, one along +Y, one along +X.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    cz, cy, cx = shape[0] // 2, shape[1] // 2, shape[2] // 2
    radius = 2

    # Tube 1: along Z axis through center
    for z in range(shape[0]):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy * dy + dx * dx <= radius * radius:
                    y, x = cy + dy, cx + dx
                    if 0 <= y < shape[1] and 0 <= x < shape[2]:
                        mask[z, y, x] = 1

    # Tube 2: along Y axis from center outward
    for y in range(cy, shape[1]):
        for dz in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dz * dz + dx * dx <= radius * radius:
                    z, x = cz + dz, cx + dx
                    if 0 <= z < shape[0] and 0 <= x < shape[2]:
                        mask[z, y, x] = 1

    # Tube 3: along X axis from center outward
    for x in range(cx, shape[2]):
        for dz in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dz * dz + dy * dy <= radius * radius:
                    z, y = cz + dz, cy + dy
                    if 0 <= z < shape[0] and 0 <= y < shape[1]:
                        mask[z, y, x] = 1

    return mask


class TestDisconnectToConnectd:
    """Tests for DisconnectToConnectd MONAI MapTransform."""

    def test_d2c_no_op_when_probability_zero(self) -> None:
        """Image unchanged when p=0."""
        mask = _make_y_junction_mask()
        image = mask.astype(np.float32)
        data = {"image": image.copy(), "label": mask.copy()}
        transform = DisconnectToConnectd(keys=["image"], label_key="label", prob=0.0)
        result = transform(data)
        np.testing.assert_array_equal(result["image"], image)

    def test_d2c_always_applies_when_probability_one(self) -> None:
        """Image modified when p=1 and junctions exist."""
        mask = _make_y_junction_mask()
        image = mask.astype(np.float32)
        data = {"image": image.copy(), "label": mask.copy()}
        transform = DisconnectToConnectd(
            keys=["image"], label_key="label", prob=1.0, seed=42
        )
        result = transform(data)
        # Image should be different from original (some voxels zeroed)
        assert not np.array_equal(result["image"], image), (
            "D2C with p=1 should modify the image"
        )

    def test_d2c_label_never_modified(self) -> None:
        """GT mask identical before and after."""
        mask = _make_y_junction_mask()
        image = mask.astype(np.float32)
        label_copy = mask.copy()
        data = {"image": image.copy(), "label": mask.copy()}
        transform = DisconnectToConnectd(
            keys=["image"], label_key="label", prob=1.0, seed=42
        )
        result = transform(data)
        np.testing.assert_array_equal(result["label"], label_copy)

    def test_d2c_sdf_never_modified(self) -> None:
        """SDF key (if present) unchanged."""
        mask = _make_y_junction_mask()
        image = mask.astype(np.float32)
        sdf = np.random.default_rng(0).standard_normal(mask.shape).astype(np.float32)
        data = {"image": image.copy(), "label": mask.copy(), "sdf": sdf.copy()}
        transform = DisconnectToConnectd(
            keys=["image"], label_key="label", prob=1.0, seed=42
        )
        result = transform(data)
        np.testing.assert_array_equal(result["sdf"], sdf)

    def test_d2c_input_modified_at_junction(self) -> None:
        """Input zeroed near junction when using zero mode."""
        mask = _make_y_junction_mask()
        image = mask.astype(np.float32) * 255.0
        data = {"image": image.copy(), "label": mask.copy()}
        transform = DisconnectToConnectd(
            keys=["image"], label_key="label", prob=1.0, mode="zero", seed=42
        )
        result = transform(data)
        # Some vessel voxels should now be zero that weren't before
        was_vessel = mask > 0
        now_zero = result["image"] == 0
        disconnected = was_vessel & now_zero
        assert disconnected.sum() > 0, "Should have some disconnected vessel voxels"

    def test_d2c_empty_mask_no_crash(self) -> None:
        """Graceful skip on empty mask."""
        mask = np.zeros((16, 32, 32), dtype=np.uint8)
        image = np.zeros((16, 32, 32), dtype=np.float32)
        data = {"image": image.copy(), "label": mask.copy()}
        transform = DisconnectToConnectd(keys=["image"], label_key="label", prob=1.0)
        result = transform(data)
        np.testing.assert_array_equal(result["image"], image)

    def test_d2c_no_junctions_no_crash(self) -> None:
        """Graceful skip on straight tube (no junctions)."""
        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        # Simple straight tube along Z axis
        mask[:, 14:18, 14:18] = 1
        image = mask.astype(np.float32)
        data = {"image": image.copy(), "label": mask.copy()}
        transform = DisconnectToConnectd(keys=["image"], label_key="label", prob=1.0)
        result = transform(data)
        # No junctions found, so image should be unchanged
        np.testing.assert_array_equal(result["image"], image)

    def test_d2c_3d_volume(self) -> None:
        """Works with 3D volume (32,64,64)."""
        mask = _make_y_junction_mask((32, 64, 64))
        image = mask.astype(np.float32)
        data = {"image": image.copy(), "label": mask.copy()}
        transform = DisconnectToConnectd(
            keys=["image"], label_key="label", prob=1.0, seed=42
        )
        result = transform(data)
        assert result["image"].shape == (32, 64, 64)

    def test_d2c_uses_centreline_extraction(self) -> None:
        """Calls extract_centreline() from centreline_extraction module."""
        mask = _make_y_junction_mask()
        image = mask.astype(np.float32)
        data = {"image": image.copy(), "label": mask.copy()}

        with patch(
            "minivess.data.disconnect_augmentation.extract_centreline"
        ) as mock_extract:
            from minivess.pipeline.centreline_extraction import CentrelineGraph

            mock_extract.return_value = CentrelineGraph(nodes=[], edges=[])
            transform = DisconnectToConnectd(
                keys=["image"], label_key="label", prob=1.0
            )
            transform(data)
            mock_extract.assert_called_once()

    def test_d2c_max_segment_length_respected(self) -> None:
        """Disconnection limited to max_segment_length voxels along skeleton."""
        mask = _make_y_junction_mask((32, 64, 64))
        image = mask.astype(np.float32) * 255.0
        data = {"image": image.copy(), "label": mask.copy()}
        transform = DisconnectToConnectd(
            keys=["image"],
            label_key="label",
            prob=1.0,
            max_segment_length=5,
            seed=42,
        )
        result = transform(data)
        # The disconnected region should be limited in extent
        was_vessel = mask > 0
        now_zero = result["image"] == 0
        disconnected = was_vessel & now_zero
        # With max_segment_length=5 and dilation_radius=2, the total affected
        # region should be bounded
        assert disconnected.sum() < 2000, (
            f"Disconnection too large: {disconnected.sum()} voxels"
        )

    def test_d2c_synthetic_y_junction(self) -> None:
        """Y-shaped tube junction found and disconnected."""
        mask = _make_y_junction_mask()
        image = mask.astype(np.float32) * 255.0
        data = {"image": image.copy(), "label": mask.copy()}
        transform = DisconnectToConnectd(
            keys=["image"], label_key="label", prob=1.0, seed=42
        )
        result = transform(data)
        # Should have modified some voxels
        diff = np.abs(result["image"] - image)
        assert diff.sum() > 0, "Y-junction should produce disconnection"

    def test_d2c_reproducible_with_seed(self) -> None:
        """Same seed gives same disconnection."""
        mask = _make_y_junction_mask()
        image = mask.astype(np.float32)

        data1 = {"image": image.copy(), "label": mask.copy()}
        data2 = {"image": image.copy(), "label": mask.copy()}

        t1 = DisconnectToConnectd(keys=["image"], label_key="label", prob=1.0, seed=123)
        t2 = DisconnectToConnectd(keys=["image"], label_key="label", prob=1.0, seed=123)

        r1 = t1(data1)
        r2 = t2(data2)
        np.testing.assert_array_equal(r1["image"], r2["image"])

    def test_d2c_full_volume_performance(self) -> None:
        """128x128x64 completes within 10 seconds."""
        mask = _make_y_junction_mask((64, 128, 128))
        image = mask.astype(np.float32)
        data = {"image": image.copy(), "label": mask.copy()}
        transform = DisconnectToConnectd(
            keys=["image"], label_key="label", prob=1.0, seed=42
        )
        start = time.monotonic()
        transform(data)
        elapsed = time.monotonic() - start
        assert elapsed < 10.0, f"D2C on 64x128x128 took {elapsed:.1f}s (>10s limit)"

    def test_d2c_junction_frequency_on_realistic_mask(self) -> None:
        """Verify junctions exist in Y-junction synthetic volumes."""
        from minivess.pipeline.centreline_extraction import extract_centreline

        mask = _make_y_junction_mask()
        graph = extract_centreline(mask)
        junctions = graph.junction_coords
        assert len(junctions) > 0, (
            "Y-junction mask should have at least one junction node"
        )
