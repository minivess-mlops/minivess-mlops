"""Tests for SDF ground truth generation utility (T1 — #226)."""

from __future__ import annotations

import numpy as np
import pytest

from minivess.pipeline.sdf_generation import compute_sdf_from_mask, normalize_sdf


class TestComputeSdfFromMask:
    """Tests for compute_sdf_from_mask()."""

    def test_sdf_empty_mask_returns_positive(self) -> None:
        """All-zero mask gives all-positive distances (everything is outside)."""
        mask = np.zeros((16, 32, 32), dtype=np.uint8)
        sdf = compute_sdf_from_mask(mask)
        assert np.all(sdf > 0), "Empty mask should have all positive SDF values"

    def test_sdf_full_mask_returns_negative(self) -> None:
        """All-one mask gives all-negative distances (everything is inside)."""
        mask = np.ones((16, 32, 32), dtype=np.uint8)
        sdf = compute_sdf_from_mask(mask)
        assert np.all(sdf < 0), "Full mask should have all negative SDF values"

    def test_sdf_sign_convention(self) -> None:
        """Negative inside vessel, positive outside."""
        mask = np.zeros((16, 32, 32), dtype=np.uint8)
        # Place a sphere of radius 5 in the center
        center = np.array([8, 16, 16])
        for z in range(16):
            for y in range(32):
                for x in range(32):
                    if (
                        np.sqrt(
                            (z - center[0]) ** 2
                            + (y - center[1]) ** 2
                            + (x - center[2]) ** 2
                        )
                        <= 5
                    ):
                        mask[z, y, x] = 1

        sdf = compute_sdf_from_mask(mask)
        # Interior voxels (at center) should be negative
        assert sdf[8, 16, 16] < 0, "Center of sphere should be negative"
        # Exterior voxels (far from sphere) should be positive
        assert sdf[0, 0, 0] > 0, "Corner should be positive"

    def test_sdf_zero_at_boundary(self) -> None:
        """SDF crosses zero at vessel boundary voxels."""
        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        # Place a cube in the center
        mask[10:22, 10:22, 10:22] = 1

        sdf = compute_sdf_from_mask(mask)
        # Boundary voxels should have SDF magnitude close to zero (< 1 voxel)
        # Just inside boundary
        assert abs(sdf[10, 16, 16]) < 1.5, (
            "Just-inside boundary SDF should be near zero"
        )
        # Just outside boundary
        assert abs(sdf[9, 16, 16]) < 1.5, (
            "Just-outside boundary SDF should be near zero"
        )

    def test_sdf_3d_sphere(self) -> None:
        """Spherical mask gives radially symmetric SDF."""
        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        center = np.array([16, 16, 16])
        radius = 8
        zz, yy, xx = np.mgrid[:32, :32, :32]
        dist_from_center = np.sqrt(
            (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
        )
        mask[dist_from_center <= radius] = 1

        sdf = compute_sdf_from_mask(mask)
        # At center, SDF should be approximately -radius (negative, deep inside)
        assert sdf[16, 16, 16] < -5, (
            f"Center SDF should be deeply negative, got {sdf[16, 16, 16]}"
        )
        # At distance ~2*radius from center, SDF should be positive
        assert sdf[16, 16, 0] > 0, "Far from sphere should be positive"

    def test_sdf_dtype_float32(self) -> None:
        """Output is float32."""
        mask = np.zeros((8, 8, 8), dtype=np.uint8)
        mask[2:6, 2:6, 2:6] = 1
        sdf = compute_sdf_from_mask(mask)
        assert sdf.dtype == np.float32, f"Expected float32, got {sdf.dtype}"

    def test_sdf_matches_scipy_edt(self) -> None:
        """Verify SDF magnitude matches scipy EDT output."""
        from scipy.ndimage import distance_transform_edt

        mask = np.zeros((16, 16, 16), dtype=np.uint8)
        mask[4:12, 4:12, 4:12] = 1

        sdf = compute_sdf_from_mask(mask)

        # Distance outside vessel (positive region)
        edt_outside = distance_transform_edt(1 - mask)
        # Distance inside vessel (negative region)
        edt_inside = distance_transform_edt(mask)

        # At an exterior point, SDF should match the outside EDT
        ext_point = (0, 0, 0)
        assert abs(sdf[ext_point] - edt_outside[ext_point]) < 0.01

        # At an interior point, SDF should be negative with magnitude matching inside EDT
        int_point = (8, 8, 8)
        assert abs(abs(sdf[int_point]) - edt_inside[int_point]) < 0.01


class TestNormalizeSdf:
    """Tests for normalize_sdf()."""

    def test_normalize_sdf_range(self) -> None:
        """Output clipped to [-1, 1]."""
        sdf = np.array([-20.0, -5.0, 0.0, 5.0, 20.0], dtype=np.float32)
        normalized = normalize_sdf(sdf, max_dist=10.0)
        assert normalized.min() >= -1.0
        assert normalized.max() <= 1.0

    def test_normalize_sdf_preserves_sign(self) -> None:
        """Sign preserved after normalization."""
        sdf = np.array([-5.0, -1.0, 0.0, 1.0, 5.0], dtype=np.float32)
        normalized = normalize_sdf(sdf, max_dist=10.0)
        # Negative values stay negative
        assert normalized[0] < 0
        assert normalized[1] < 0
        # Positive values stay positive
        assert normalized[3] > 0
        assert normalized[4] > 0
        # Zero stays zero
        assert normalized[2] == pytest.approx(0.0, abs=1e-6)
