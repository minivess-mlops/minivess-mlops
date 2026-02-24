"""Tests for generative UQ methods (Issue #51)."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# T1: GenerativeUQMethod enum
# ---------------------------------------------------------------------------


class TestGenerativeUQMethod:
    """Test generative UQ method enum."""

    def test_enum_values(self) -> None:
        """GenerativeUQMethod should have three methods."""
        from minivess.ensemble.generative_uq import GenerativeUQMethod

        assert GenerativeUQMethod.PROB_UNET == "prob_unet"
        assert GenerativeUQMethod.PHISEG == "phiseg"
        assert GenerativeUQMethod.SSN == "ssn"


# ---------------------------------------------------------------------------
# T2: MultiRaterData
# ---------------------------------------------------------------------------


class TestMultiRaterData:
    """Test multi-rater data container."""

    def test_construction(self) -> None:
        """MultiRaterData should hold multiple annotations."""
        from minivess.ensemble.generative_uq import MultiRaterData

        masks = [
            np.zeros((8, 8, 8), dtype=np.uint8),
            np.ones((8, 8, 8), dtype=np.uint8),
        ]
        data = MultiRaterData(volume_id="vol_001", rater_masks=masks)
        assert data.volume_id == "vol_001"
        assert data.num_raters == 2

    def test_single_rater(self) -> None:
        """Single rater should work."""
        from minivess.ensemble.generative_uq import MultiRaterData

        masks = [np.zeros((4, 4, 4), dtype=np.uint8)]
        data = MultiRaterData(volume_id="vol_002", rater_masks=masks)
        assert data.num_raters == 1


# ---------------------------------------------------------------------------
# T3: GenerativeUQConfig
# ---------------------------------------------------------------------------


class TestGenerativeUQConfig:
    """Test generative UQ configuration."""

    def test_construction(self) -> None:
        """GenerativeUQConfig should capture method settings."""
        from minivess.ensemble.generative_uq import GenerativeUQConfig

        config = GenerativeUQConfig(
            method="prob_unet",
            latent_dim=6,
            num_samples=16,
        )
        assert config.method == "prob_unet"
        assert config.latent_dim == 6
        assert config.num_samples == 16

    def test_defaults(self) -> None:
        """GenerativeUQConfig should have sensible defaults."""
        from minivess.ensemble.generative_uq import GenerativeUQConfig

        config = GenerativeUQConfig()
        assert config.method == "prob_unet"
        assert config.latent_dim == 6
        assert config.num_samples == 16


# ---------------------------------------------------------------------------
# T4: GED metric
# ---------------------------------------------------------------------------


class TestGED:
    """Test Generalized Energy Distance metric."""

    def test_identical_samples(self) -> None:
        """GED of identical distributions should be zero."""
        from minivess.ensemble.generative_uq import generalized_energy_distance

        mask = np.ones((8, 8, 8), dtype=np.uint8)
        samples = [mask.copy() for _ in range(5)]
        references = [mask.copy() for _ in range(3)]
        ged = generalized_energy_distance(samples, references)
        assert abs(ged) < 1e-6

    def test_different_samples(self) -> None:
        """GED of non-overlapping distributions should be positive."""
        from minivess.ensemble.generative_uq import generalized_energy_distance

        zeros = [np.zeros((8, 8, 8), dtype=np.uint8) for _ in range(5)]
        ones = [np.ones((8, 8, 8), dtype=np.uint8) for _ in range(3)]
        ged = generalized_energy_distance(zeros, ones)
        assert ged > 0

    def test_symmetric(self) -> None:
        """GED should be approximately symmetric."""
        from minivess.ensemble.generative_uq import generalized_energy_distance

        rng = np.random.default_rng(42)
        a = [(rng.random((4, 4, 4)) > 0.5).astype(np.uint8) for _ in range(5)]
        b = [(rng.random((4, 4, 4)) > 0.3).astype(np.uint8) for _ in range(5)]
        ged_ab = generalized_energy_distance(a, b)
        ged_ba = generalized_energy_distance(b, a)
        assert abs(ged_ab - ged_ba) < 1e-6


# ---------------------------------------------------------------------------
# T5: Q-Dice metric
# ---------------------------------------------------------------------------


class TestQDice:
    """Test Q-Dice (Quantized Dice) metric."""

    def test_perfect_overlap(self) -> None:
        """Q-Dice of identical probability maps should be ~1.0."""
        from minivess.ensemble.generative_uq import q_dice

        prob_map = np.ones((8, 8, 8), dtype=np.float32)
        reference = np.ones((8, 8, 8), dtype=np.uint8)
        score = q_dice(prob_map, reference)
        assert score > 0.9

    def test_no_overlap(self) -> None:
        """Q-Dice with no overlap should be ~0.0."""
        from minivess.ensemble.generative_uq import q_dice

        prob_map = np.ones((8, 8, 8), dtype=np.float32)
        reference = np.zeros((8, 8, 8), dtype=np.uint8)
        score = q_dice(prob_map, reference)
        assert score < 0.1

    def test_partial_overlap(self) -> None:
        """Q-Dice with partial overlap should be between 0 and 1."""
        from minivess.ensemble.generative_uq import q_dice

        prob_map = np.zeros((8, 8, 8), dtype=np.float32)
        prob_map[2:6, 2:6, 2:6] = 0.8
        reference = np.zeros((8, 8, 8), dtype=np.uint8)
        reference[3:7, 3:7, 3:7] = 1
        score = q_dice(prob_map, reference)
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# T6: GenerativeUQEvaluator
# ---------------------------------------------------------------------------


class TestGenerativeUQEvaluator:
    """Test generative UQ evaluator."""

    def test_add_data(self) -> None:
        """Should accept prediction samples and rater annotations."""
        from minivess.ensemble.generative_uq import GenerativeUQEvaluator

        evaluator = GenerativeUQEvaluator()
        samples = [np.zeros((4, 4, 4), dtype=np.uint8) for _ in range(3)]
        raters = [np.ones((4, 4, 4), dtype=np.uint8) for _ in range(2)]
        evaluator.add_volume("vol_001", samples, raters)
        assert "vol_001" in evaluator.volumes

    def test_compute_metrics(self) -> None:
        """compute_metrics should return GED and Q-Dice."""
        from minivess.ensemble.generative_uq import GenerativeUQEvaluator

        evaluator = GenerativeUQEvaluator()
        rng = np.random.default_rng(42)
        samples = [(rng.random((4, 4, 4)) > 0.5).astype(np.uint8) for _ in range(5)]
        raters = [(rng.random((4, 4, 4)) > 0.4).astype(np.uint8) for _ in range(3)]
        evaluator.add_volume("vol_001", samples, raters)
        metrics = evaluator.compute_metrics("vol_001")
        assert "ged" in metrics
        assert "q_dice" in metrics

    def test_to_markdown(self) -> None:
        """to_markdown should produce an evaluation report."""
        from minivess.ensemble.generative_uq import GenerativeUQEvaluator

        evaluator = GenerativeUQEvaluator()
        rng = np.random.default_rng(42)
        samples = [(rng.random((4, 4, 4)) > 0.5).astype(np.uint8) for _ in range(5)]
        raters = [(rng.random((4, 4, 4)) > 0.4).astype(np.uint8) for _ in range(3)]
        evaluator.add_volume("vol_001", samples, raters)
        md = evaluator.to_markdown()
        assert "Generative" in md
        assert "vol_001" in md
