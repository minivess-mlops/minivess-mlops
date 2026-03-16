"""Tests for synthetic generator adapters (T-D2 through T-D5).

T-D2: VesselFM d_drand adapter
T-D3: MONAI VQ-VAE adapter
T-D4: VaMos procedural adapter
T-D5: VascuSynth C++ wrapper
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# T-D2: VesselFM d_drand adapter
# ---------------------------------------------------------------------------


class TestVesselFMDrandAdapter:
    """Test vesselFM domain randomization adapter."""

    def test_adapter_instantiation(self) -> None:
        from minivess.data.synthetic.vesselfm_drand import VesselFMDrandGenerator

        gen = VesselFMDrandGenerator()
        assert gen.name == "vesselFM_drand"
        assert gen.requires_training is False

    def test_adapter_is_registered(self) -> None:
        from minivess.data.synthetic import SYNTHETIC_GENERATORS

        assert "vesselFM_drand" in SYNTHETIC_GENERATORS

    def test_adapter_implements_abc(self) -> None:
        from minivess.data.synthetic.base import SyntheticGeneratorAdapter
        from minivess.data.synthetic.vesselfm_drand import VesselFMDrandGenerator

        assert issubclass(VesselFMDrandGenerator, SyntheticGeneratorAdapter)

    def test_generate_stack_returns_pairs(self) -> None:
        """Should generate (image, mask) pairs even in stub mode."""
        from minivess.data.synthetic.vesselfm_drand import VesselFMDrandGenerator

        gen = VesselFMDrandGenerator()
        pairs = gen.generate_stack(n_volumes=2)
        assert len(pairs) == 2
        for img, mask in pairs:
            assert isinstance(img, np.ndarray)
            assert isinstance(mask, np.ndarray)
            assert img.ndim == 3
            assert mask.ndim == 3

    def test_generate_stack_config_patch_size(self) -> None:
        """Config should allow setting patch size."""
        from minivess.data.synthetic.vesselfm_drand import VesselFMDrandGenerator

        gen = VesselFMDrandGenerator()
        pairs = gen.generate_stack(
            n_volumes=1,
            config={"patch_size": (64, 64, 64)},
        )
        img, mask = pairs[0]
        assert img.shape == (64, 64, 64)

    def test_license_attribute(self) -> None:
        from minivess.data.synthetic.vesselfm_drand import VesselFMDrandGenerator

        gen = VesselFMDrandGenerator()
        assert gen.license == "GPL-3.0"


# ---------------------------------------------------------------------------
# T-D3: MONAI VQ-VAE adapter
# ---------------------------------------------------------------------------


class TestMONAIVQVAEAdapter:
    """Test MONAI VQ-VAE synthetic adapter."""

    def test_adapter_instantiation(self) -> None:
        from minivess.data.synthetic.monai_vqvae import MONAIVQVAEGenerator

        gen = MONAIVQVAEGenerator()
        assert gen.name == "monai_vqvae"
        assert gen.requires_training is True

    def test_adapter_is_registered(self) -> None:
        from minivess.data.synthetic import SYNTHETIC_GENERATORS

        assert "monai_vqvae" in SYNTHETIC_GENERATORS

    def test_adapter_implements_abc(self) -> None:
        from minivess.data.synthetic.base import SyntheticGeneratorAdapter
        from minivess.data.synthetic.monai_vqvae import MONAIVQVAEGenerator

        assert issubclass(MONAIVQVAEGenerator, SyntheticGeneratorAdapter)

    def test_generate_stack_returns_pairs(self) -> None:
        from minivess.data.synthetic.monai_vqvae import MONAIVQVAEGenerator

        gen = MONAIVQVAEGenerator()
        pairs = gen.generate_stack(n_volumes=2)
        assert len(pairs) == 2
        for img, mask in pairs:
            assert img.ndim == 3
            assert mask.ndim == 3

    def test_generate_stack_config_codebook(self) -> None:
        """Config should allow codebook size."""
        from minivess.data.synthetic.monai_vqvae import MONAIVQVAEGenerator

        gen = MONAIVQVAEGenerator()
        pairs = gen.generate_stack(
            n_volumes=1,
            config={"codebook_size": 512, "patch_size": (32, 32, 32)},
        )
        assert len(pairs) == 1

    def test_license_attribute(self) -> None:
        from minivess.data.synthetic.monai_vqvae import MONAIVQVAEGenerator

        gen = MONAIVQVAEGenerator()
        assert gen.license == "Apache-2.0"


# ---------------------------------------------------------------------------
# T-D4: VaMos procedural adapter
# ---------------------------------------------------------------------------


class TestVaMosAdapter:
    """Test VaMos procedural vascular adapter."""

    def test_adapter_instantiation(self) -> None:
        from minivess.data.synthetic.vamos import VaMosGenerator

        gen = VaMosGenerator()
        assert gen.name == "vamos"
        assert gen.requires_training is False

    def test_adapter_is_registered(self) -> None:
        from minivess.data.synthetic import SYNTHETIC_GENERATORS

        assert "vamos" in SYNTHETIC_GENERATORS

    def test_adapter_implements_abc(self) -> None:
        from minivess.data.synthetic.base import SyntheticGeneratorAdapter
        from minivess.data.synthetic.vamos import VaMosGenerator

        assert issubclass(VaMosGenerator, SyntheticGeneratorAdapter)

    def test_generate_stack_returns_pairs(self) -> None:
        from minivess.data.synthetic.vamos import VaMosGenerator

        gen = VaMosGenerator()
        pairs = gen.generate_stack(n_volumes=2)
        assert len(pairs) == 2
        for img, mask in pairs:
            assert img.ndim == 3
            assert mask.ndim == 3

    def test_cpu_only(self) -> None:
        """VaMos is CPU-only procedural generation."""
        from minivess.data.synthetic.vamos import VaMosGenerator

        gen = VaMosGenerator()
        assert gen.requires_gpu is False

    def test_configurable_morphology(self) -> None:
        """Should accept morphological parameters."""
        from minivess.data.synthetic.vamos import VaMosGenerator

        gen = VaMosGenerator()
        pairs = gen.generate_stack(
            n_volumes=1,
            config={"vessel_diameter": 3.0, "branching_angle": 45.0},
        )
        assert len(pairs) == 1


# ---------------------------------------------------------------------------
# T-D5: VascuSynth C++ wrapper
# ---------------------------------------------------------------------------


class TestVascuSynthAdapter:
    """Test VascuSynth C++ subprocess wrapper."""

    def test_adapter_instantiation(self) -> None:
        from minivess.data.synthetic.vascusynth import VascuSynthGenerator

        gen = VascuSynthGenerator()
        assert gen.name == "vascusynth"
        assert gen.requires_training is False

    def test_adapter_is_registered(self) -> None:
        from minivess.data.synthetic import SYNTHETIC_GENERATORS

        assert "vascusynth" in SYNTHETIC_GENERATORS

    def test_adapter_implements_abc(self) -> None:
        from minivess.data.synthetic.base import SyntheticGeneratorAdapter
        from minivess.data.synthetic.vascusynth import VascuSynthGenerator

        assert issubclass(VascuSynthGenerator, SyntheticGeneratorAdapter)

    def test_generate_stack_returns_pairs(self) -> None:
        from minivess.data.synthetic.vascusynth import VascuSynthGenerator

        gen = VascuSynthGenerator()
        pairs = gen.generate_stack(n_volumes=2)
        assert len(pairs) == 2
        for img, mask in pairs:
            assert img.ndim == 3
            assert mask.ndim == 3

    def test_license_attribute(self) -> None:
        from minivess.data.synthetic.vascusynth import VascuSynthGenerator

        gen = VascuSynthGenerator()
        assert gen.license == "Apache-2.0"

    def test_binary_availability_check(self) -> None:
        """Should have a method to check if binary is available."""
        from minivess.data.synthetic.vascusynth import VascuSynthGenerator

        gen = VascuSynthGenerator()
        assert isinstance(gen.is_binary_available(), bool)


# ---------------------------------------------------------------------------
# Cross-adapter: registry integration
# ---------------------------------------------------------------------------


class TestAllAdaptersRegistered:
    """Test that all 5 generators are in the registry."""

    def test_five_generators_registered(self) -> None:
        from minivess.data.synthetic import list_generators

        methods = list_generators()
        expected = {"debug", "vesselFM_drand", "monai_vqvae", "vamos", "vascusynth"}
        assert expected.issubset(set(methods))

    def test_generate_stack_api_all_methods(self) -> None:
        """Top-level API should work for all registered methods."""
        from minivess.data.synthetic import generate_stack, list_generators

        for method in list_generators():
            pairs = generate_stack(method=method, n_volumes=1)
            assert len(pairs) == 1
            img, mask = pairs[0]
            assert img.ndim == 3
            assert mask.ndim == 3
