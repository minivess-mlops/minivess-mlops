"""Tests for SynthICL domain randomization augmentation (Issue #17)."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# T1: RandomizationParam enum
# ---------------------------------------------------------------------------


class TestRandomizationParam:
    """Test randomization parameter enum."""

    def test_enum_values(self) -> None:
        """RandomizationParam should have five parameters."""
        from minivess.data.domain_randomization import RandomizationParam

        assert RandomizationParam.INTENSITY == "intensity"
        assert RandomizationParam.CONTRAST == "contrast"
        assert RandomizationParam.NOISE == "noise"
        assert RandomizationParam.BLUR == "blur"
        assert RandomizationParam.SPACING == "spacing"


# ---------------------------------------------------------------------------
# T2: DomainRandomizationConfig
# ---------------------------------------------------------------------------


class TestDomainRandomizationConfig:
    """Test domain randomization configuration."""

    def test_construction(self) -> None:
        """DomainRandomizationConfig should capture settings."""
        from minivess.data.domain_randomization import DomainRandomizationConfig

        config = DomainRandomizationConfig(
            intensity_range=(0.5, 1.5),
            noise_std_range=(0.0, 0.1),
            seed=42,
        )
        assert config.intensity_range == (0.5, 1.5)
        assert config.seed == 42

    def test_defaults(self) -> None:
        """DomainRandomizationConfig should have sensible defaults."""
        from minivess.data.domain_randomization import DomainRandomizationConfig

        config = DomainRandomizationConfig()
        assert config.intensity_range == (0.7, 1.3)
        assert config.noise_std_range == (0.0, 0.05)
        assert config.contrast_range == (0.5, 2.0)
        assert config.blur_sigma_range == (0.0, 1.0)


# ---------------------------------------------------------------------------
# T3: SyntheticVesselGenerator
# ---------------------------------------------------------------------------


class TestSyntheticVesselGenerator:
    """Test synthetic vessel-like structure generation."""

    def test_random_tubular_mask(self) -> None:
        """random_tubular_mask should produce a binary 3D volume."""
        from minivess.data.domain_randomization import SyntheticVesselGenerator

        gen = SyntheticVesselGenerator(seed=42)
        mask = gen.random_tubular_mask(shape=(32, 32, 16))
        assert mask.shape == (32, 32, 16)
        assert mask.dtype == np.uint8
        assert np.unique(mask).tolist() in [[0, 1], [0], [1]]
        assert mask.sum() > 0  # should have some foreground

    def test_randomize_intensity(self) -> None:
        """randomize_intensity should scale the volume intensity."""
        from minivess.data.domain_randomization import SyntheticVesselGenerator

        gen = SyntheticVesselGenerator(seed=42)
        vol = np.ones((8, 8, 8), dtype=np.float32) * 0.5
        result = gen.randomize_intensity(vol, scale_range=(2.0, 2.0))
        np.testing.assert_allclose(result, 1.0, atol=1e-5)

    def test_randomize_contrast(self) -> None:
        """randomize_contrast should change the contrast."""
        from minivess.data.domain_randomization import SyntheticVesselGenerator

        gen = SyntheticVesselGenerator(seed=42)
        vol = np.linspace(0.0, 1.0, 64).reshape(4, 4, 4).astype(np.float32)
        result = gen.randomize_contrast(vol, gamma_range=(2.0, 2.0))
        # Gamma > 1 should darken mid-tones
        assert np.mean(result) < np.mean(vol)

    def test_add_noise(self) -> None:
        """add_noise should increase variance."""
        from minivess.data.domain_randomization import SyntheticVesselGenerator

        gen = SyntheticVesselGenerator(seed=42)
        vol = np.ones((16, 16, 16), dtype=np.float32) * 0.5
        noisy = gen.add_noise(vol, std=0.1)
        assert np.std(noisy) > 0.01


# ---------------------------------------------------------------------------
# T4: DomainRandomizationPipeline
# ---------------------------------------------------------------------------


class TestDomainRandomizationPipeline:
    """Test domain randomization pipeline."""

    def test_apply_single(self) -> None:
        """apply should produce a randomized volume with same shape."""
        from minivess.data.domain_randomization import (
            DomainRandomizationConfig,
            DomainRandomizationPipeline,
        )

        config = DomainRandomizationConfig(seed=42)
        pipeline = DomainRandomizationPipeline(config)

        vol = np.random.default_rng(42).normal(0.5, 0.1, (16, 16, 8)).astype(np.float32)
        mask = np.zeros((16, 16, 8), dtype=np.uint8)
        mask[4:12, 4:12, 2:6] = 1

        result_vol, result_mask = pipeline.apply(vol, mask)
        assert result_vol.shape == vol.shape
        assert result_mask.shape == mask.shape
        # Volume should be modified
        assert not np.allclose(result_vol, vol)
        # Mask should be unchanged (domain randomization affects intensity only)
        np.testing.assert_array_equal(result_mask, mask)

    def test_generate_batch(self) -> None:
        """generate_batch should produce N randomized samples."""
        from minivess.data.domain_randomization import (
            DomainRandomizationConfig,
            DomainRandomizationPipeline,
        )

        config = DomainRandomizationConfig(seed=42)
        pipeline = DomainRandomizationPipeline(config)

        vol = np.ones((8, 8, 8), dtype=np.float32) * 0.5
        mask = np.zeros((8, 8, 8), dtype=np.uint8)
        mask[2:6, 2:6, 2:6] = 1

        batch = pipeline.generate_batch(vol, mask, n=5)
        assert len(batch) == 5
        assert all(v.shape == vol.shape for v, _ in batch)
        # Each sample should be different
        assert not np.allclose(batch[0][0], batch[1][0])

    def test_to_markdown(self) -> None:
        """to_markdown should produce a readable report."""
        from minivess.data.domain_randomization import (
            DomainRandomizationConfig,
            DomainRandomizationPipeline,
        )

        config = DomainRandomizationConfig(seed=42)
        pipeline = DomainRandomizationPipeline(config)
        md = pipeline.to_markdown()
        assert "Domain Randomization" in md
        assert "intensity" in md.lower()
        assert "noise" in md.lower()
