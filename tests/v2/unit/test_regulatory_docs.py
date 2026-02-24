"""Tests for automated regulatory documentation generation (Issue #8)."""

from __future__ import annotations

from minivess.compliance.audit import AuditTrail


def _make_sample_audit_trail() -> AuditTrail:
    """Create a sample audit trail with typical lifecycle events."""
    trail = AuditTrail()
    trail.log_data_access(
        dataset_name="minivess",
        file_paths=["/data/raw/vol001.nii.gz", "/data/raw/vol002.nii.gz"],
    )
    trail.log_model_training(
        model_name="segresnet_v1",
        config={"learning_rate": 1e-4, "batch_size": 2, "max_epochs": 100},
    )
    trail.log_test_evaluation(
        model_name="segresnet_v1",
        metrics={"val_dice": 0.85, "val_cldice": 0.72, "hausdorff95": 3.2},
    )
    trail.log_model_deployment(
        model_name="segresnet_v1",
        version="1.0.0",
    )
    return trail


# ---------------------------------------------------------------------------
# T1: SaMD risk classification
# ---------------------------------------------------------------------------


class TestSaMDRiskClass:
    """Test SaMD risk classification enum."""

    def test_risk_classes_exist(self) -> None:
        """All four SaMD risk classes should be defined."""
        from minivess.compliance.regulatory_docs import SaMDRiskClass

        assert hasattr(SaMDRiskClass, "CLASS_I")
        assert hasattr(SaMDRiskClass, "CLASS_IIA")
        assert hasattr(SaMDRiskClass, "CLASS_IIB")
        assert hasattr(SaMDRiskClass, "CLASS_III")

    def test_risk_class_values(self) -> None:
        """Risk class values should be lowercase strings."""
        from minivess.compliance.regulatory_docs import SaMDRiskClass

        assert SaMDRiskClass.CLASS_I.value == "class_i"
        assert SaMDRiskClass.CLASS_III.value == "class_iii"


# ---------------------------------------------------------------------------
# T2: RegulatoryDocGenerator creation
# ---------------------------------------------------------------------------


class TestRegulatoryDocGenerator:
    """Test regulatory document generator."""

    def test_creates_with_audit_trail(self) -> None:
        """Should create from an AuditTrail."""
        from minivess.compliance.regulatory_docs import RegulatoryDocGenerator

        trail = _make_sample_audit_trail()
        gen = RegulatoryDocGenerator(
            audit_trail=trail,
            product_name="MiniVess Segmentation",
            product_version="1.0.0",
        )
        assert gen.product_name == "MiniVess Segmentation"

    def test_creates_with_risk_class(self) -> None:
        """Should accept a SaMD risk classification."""
        from minivess.compliance.regulatory_docs import (
            RegulatoryDocGenerator,
            SaMDRiskClass,
        )

        trail = _make_sample_audit_trail()
        gen = RegulatoryDocGenerator(
            audit_trail=trail,
            product_name="MiniVess",
            product_version="1.0.0",
            risk_class=SaMDRiskClass.CLASS_IIA,
        )
        assert gen.risk_class == SaMDRiskClass.CLASS_IIA


# ---------------------------------------------------------------------------
# T3: Design History File generation
# ---------------------------------------------------------------------------


class TestDesignHistoryFile:
    """Test IEC 62304 Design History File generation."""

    def test_dhf_is_markdown(self) -> None:
        """DHF should be a markdown string."""
        from minivess.compliance.regulatory_docs import RegulatoryDocGenerator

        trail = _make_sample_audit_trail()
        gen = RegulatoryDocGenerator(
            audit_trail=trail,
            product_name="MiniVess",
            product_version="1.0.0",
        )
        dhf = gen.generate_design_history()
        assert isinstance(dhf, str)
        assert "# Design History File" in dhf

    def test_dhf_contains_audit_events(self) -> None:
        """DHF should include audit trail events."""
        from minivess.compliance.regulatory_docs import RegulatoryDocGenerator

        trail = _make_sample_audit_trail()
        gen = RegulatoryDocGenerator(
            audit_trail=trail,
            product_name="MiniVess",
            product_version="1.0.0",
        )
        dhf = gen.generate_design_history()
        assert "DATA_ACCESS" in dhf
        assert "MODEL_TRAINING" in dhf
        assert "MODEL_DEPLOYMENT" in dhf

    def test_dhf_contains_product_info(self) -> None:
        """DHF should include product name and version."""
        from minivess.compliance.regulatory_docs import RegulatoryDocGenerator

        trail = _make_sample_audit_trail()
        gen = RegulatoryDocGenerator(
            audit_trail=trail,
            product_name="MiniVess Segmentation",
            product_version="2.0.0",
        )
        dhf = gen.generate_design_history()
        assert "MiniVess Segmentation" in dhf
        assert "2.0.0" in dhf


# ---------------------------------------------------------------------------
# T4: Risk Analysis document
# ---------------------------------------------------------------------------


class TestRiskAnalysis:
    """Test SaMD risk analysis document generation."""

    def test_risk_analysis_is_markdown(self) -> None:
        """Risk analysis should be a markdown string."""
        from minivess.compliance.regulatory_docs import RegulatoryDocGenerator

        trail = _make_sample_audit_trail()
        gen = RegulatoryDocGenerator(
            audit_trail=trail,
            product_name="MiniVess",
            product_version="1.0.0",
        )
        doc = gen.generate_risk_analysis()
        assert isinstance(doc, str)
        assert "# Risk Analysis" in doc

    def test_risk_analysis_includes_classification(self) -> None:
        """Risk analysis should include the SaMD risk class."""
        from minivess.compliance.regulatory_docs import (
            RegulatoryDocGenerator,
            SaMDRiskClass,
        )

        trail = _make_sample_audit_trail()
        gen = RegulatoryDocGenerator(
            audit_trail=trail,
            product_name="MiniVess",
            product_version="1.0.0",
            risk_class=SaMDRiskClass.CLASS_IIA,
        )
        doc = gen.generate_risk_analysis()
        assert "Class IIa" in doc or "class_iia" in doc

    def test_risk_analysis_includes_mitigations(self) -> None:
        """Risk analysis should include mitigation strategies section."""
        from minivess.compliance.regulatory_docs import RegulatoryDocGenerator

        trail = _make_sample_audit_trail()
        gen = RegulatoryDocGenerator(
            audit_trail=trail,
            product_name="MiniVess",
            product_version="1.0.0",
        )
        doc = gen.generate_risk_analysis()
        assert "Mitigation" in doc or "mitigation" in doc


# ---------------------------------------------------------------------------
# T5: Software Requirements Specification
# ---------------------------------------------------------------------------


class TestSRS:
    """Test IEC 62304 SRS template generation."""

    def test_srs_is_markdown(self) -> None:
        """SRS should be a markdown string."""
        from minivess.compliance.regulatory_docs import RegulatoryDocGenerator

        trail = _make_sample_audit_trail()
        gen = RegulatoryDocGenerator(
            audit_trail=trail,
            product_name="MiniVess",
            product_version="1.0.0",
        )
        srs = gen.generate_srs()
        assert isinstance(srs, str)
        assert "# Software Requirements Specification" in srs

    def test_srs_includes_scope(self) -> None:
        """SRS should include scope with product info."""
        from minivess.compliance.regulatory_docs import RegulatoryDocGenerator

        trail = _make_sample_audit_trail()
        gen = RegulatoryDocGenerator(
            audit_trail=trail,
            product_name="MiniVess",
            product_version="1.0.0",
        )
        srs = gen.generate_srs()
        assert "MiniVess" in srs


# ---------------------------------------------------------------------------
# T6: Validation Summary
# ---------------------------------------------------------------------------


class TestValidationSummary:
    """Test validation summary generation from test evaluation events."""

    def test_validation_summary_includes_metrics(self) -> None:
        """Validation summary should include test evaluation metrics."""
        from minivess.compliance.regulatory_docs import RegulatoryDocGenerator

        trail = _make_sample_audit_trail()
        gen = RegulatoryDocGenerator(
            audit_trail=trail,
            product_name="MiniVess",
            product_version="1.0.0",
        )
        summary = gen.generate_validation_summary()
        assert "val_dice" in summary
        assert "# Validation Summary" in summary

    def test_validation_summary_empty_trail(self) -> None:
        """Validation summary should handle trail with no test events."""
        from minivess.compliance.regulatory_docs import RegulatoryDocGenerator

        trail = AuditTrail()
        gen = RegulatoryDocGenerator(
            audit_trail=trail,
            product_name="MiniVess",
            product_version="1.0.0",
        )
        summary = gen.generate_validation_summary()
        assert "No test evaluation" in summary or "no evaluation" in summary.lower()
