"""Tests for MedSAM3 interactive annotation adapter (Issue #22)."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# T1: PromptType and MedicalConcept enums
# ---------------------------------------------------------------------------


class TestPromptType:
    """Test prompt type enum."""

    def test_enum_values(self) -> None:
        """PromptType should have four prompt types."""
        from minivess.adapters.medsam3 import PromptType

        assert PromptType.POINT == "point"
        assert PromptType.BOX == "box"
        assert PromptType.MASK == "mask"
        assert PromptType.CONCEPT == "concept"


class TestMedicalConcept:
    """Test medical concept enum."""

    def test_enum_values(self) -> None:
        """MedicalConcept should have five anatomical concepts."""
        from minivess.adapters.medsam3 import MedicalConcept

        assert MedicalConcept.VESSEL == "vessel"
        assert MedicalConcept.TUMOR == "tumor"
        assert MedicalConcept.ORGAN == "organ"
        assert MedicalConcept.TISSUE == "tissue"
        assert MedicalConcept.LESION == "lesion"

    def test_vessel_in_concepts(self) -> None:
        """Vessel should be a valid medical concept."""
        from minivess.adapters.medsam3 import MedicalConcept

        assert "vessel" in [c.value for c in MedicalConcept]


# ---------------------------------------------------------------------------
# T2: AnnotationPrompt
# ---------------------------------------------------------------------------


class TestAnnotationPrompt:
    """Test annotation prompt dataclass."""

    def test_point_prompt(self) -> None:
        """Point prompt should capture 3D coordinates."""
        from minivess.adapters.medsam3 import AnnotationPrompt

        prompt = AnnotationPrompt(
            prompt_type="point",
            coordinates=(10, 20, 5),
            label=1,
        )
        assert prompt.prompt_type == "point"
        assert prompt.coordinates == (10, 20, 5)
        assert prompt.label == 1

    def test_box_prompt(self) -> None:
        """Box prompt should capture bounding box corners."""
        from minivess.adapters.medsam3 import AnnotationPrompt

        prompt = AnnotationPrompt(
            prompt_type="box",
            coordinates=(5, 5, 2, 20, 20, 10),
            label=1,
        )
        assert prompt.prompt_type == "box"
        assert len(prompt.coordinates) == 6

    def test_concept_prompt(self) -> None:
        """Concept prompt should carry a medical concept."""
        from minivess.adapters.medsam3 import AnnotationPrompt

        prompt = AnnotationPrompt(
            prompt_type="concept",
            concept="vessel",
        )
        assert prompt.concept == "vessel"


# ---------------------------------------------------------------------------
# T3: MedSAM3Config and Predictor
# ---------------------------------------------------------------------------


class TestMedSAM3Config:
    """Test MedSAM3 configuration."""

    def test_construction(self) -> None:
        """MedSAM3Config should capture model settings."""
        from minivess.adapters.medsam3 import MedSAM3Config

        config = MedSAM3Config(
            model_name="medsam3_base",
            concept_vocabulary=["vessel", "tumor"],
        )
        assert config.model_name == "medsam3_base"
        assert len(config.concept_vocabulary) == 2

    def test_defaults(self) -> None:
        """MedSAM3Config should have sensible defaults."""
        from minivess.adapters.medsam3 import MedSAM3Config

        config = MedSAM3Config()
        assert config.model_name == "medsam3_base"
        assert len(config.concept_vocabulary) > 0


class TestMedSAM3Predictor:
    """Test MedSAM3 interactive predictor."""

    def test_add_prompt(self) -> None:
        """add_prompt should register prompts."""
        from minivess.adapters.medsam3 import (
            AnnotationPrompt,
            MedSAM3Config,
            MedSAM3Predictor,
        )

        predictor = MedSAM3Predictor(MedSAM3Config())
        prompt = AnnotationPrompt(prompt_type="point", coordinates=(5, 5, 5))
        predictor.add_prompt(prompt)
        assert len(predictor.prompts) == 1

    def test_reset_clears_prompts(self) -> None:
        """reset should clear accumulated prompts."""
        from minivess.adapters.medsam3 import (
            AnnotationPrompt,
            MedSAM3Config,
            MedSAM3Predictor,
        )

        predictor = MedSAM3Predictor(MedSAM3Config())
        predictor.add_prompt(
            AnnotationPrompt(prompt_type="point", coordinates=(5, 5, 5))
        )
        predictor.reset()
        assert len(predictor.prompts) == 0

    def test_annotation_record(self) -> None:
        """to_annotation_record should export metadata."""
        from minivess.adapters.medsam3 import (
            AnnotationPrompt,
            MedSAM3Config,
            MedSAM3Predictor,
        )

        predictor = MedSAM3Predictor(MedSAM3Config())
        predictor.add_prompt(AnnotationPrompt(prompt_type="concept", concept="vessel"))
        record = predictor.to_annotation_record()
        assert "prompts" in record
        assert record["model"] == "medsam3_base"


# ---------------------------------------------------------------------------
# T4: AnnotationSession
# ---------------------------------------------------------------------------


class TestAnnotationSession:
    """Test annotation session management."""

    def test_add_interaction(self) -> None:
        """add_interaction should record prompt-result pairs."""
        from minivess.data.annotation_session import AnnotationSession

        session = AnnotationSession(volume_id="vol_001")
        mask = np.zeros((8, 8, 8), dtype=np.uint8)
        session.add_interaction(
            prompt_description="point at (4,4,4)",
            predicted_mask=mask,
        )
        assert len(session.interactions) == 1

    def test_compute_agreement(self) -> None:
        """compute_agreement should return Dice against reference."""
        from minivess.data.annotation_session import AnnotationSession

        session = AnnotationSession(volume_id="vol_001")
        pred = np.zeros((8, 8, 8), dtype=np.uint8)
        pred[2:6, 2:6, 2:6] = 1
        session.add_interaction("point prompt", pred)

        ref = np.zeros((8, 8, 8), dtype=np.uint8)
        ref[2:6, 2:6, 2:6] = 1

        dice = session.compute_agreement(ref)
        assert abs(dice - 1.0) < 1e-6  # Perfect agreement

    def test_to_markdown(self) -> None:
        """to_markdown should produce a session report."""
        from minivess.data.annotation_session import AnnotationSession

        session = AnnotationSession(volume_id="vol_001")
        mask = np.ones((4, 4, 4), dtype=np.uint8)
        session.add_interaction("box prompt", mask)
        md = session.to_markdown()
        assert "Annotation" in md
        assert "vol_001" in md
