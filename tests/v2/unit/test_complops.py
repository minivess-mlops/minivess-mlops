"""Tests for ComplOps regulatory automation tooling (Issue #49)."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# T1: RegulatoryTemplate enum
# ---------------------------------------------------------------------------


class TestRegulatoryTemplate:
    """Test regulatory template types."""

    def test_enum_values(self) -> None:
        """RegulatoryTemplate should have three framework types."""
        from minivess.compliance.complops import RegulatoryTemplate

        assert RegulatoryTemplate.FDA_510K == "fda_510k"
        assert RegulatoryTemplate.EU_MDR_TECH_FILE == "eu_mdr_tech_file"
        assert RegulatoryTemplate.IEC_62304_FULL == "iec_62304_full"

    def test_all_templates(self) -> None:
        """All template types should be present."""
        from minivess.compliance.complops import RegulatoryTemplate

        assert len(RegulatoryTemplate) == 3


# ---------------------------------------------------------------------------
# T2: ComplianceCheckResult
# ---------------------------------------------------------------------------


class TestComplianceCheckResult:
    """Test compliance check result."""

    def test_construction(self) -> None:
        """ComplianceCheckResult should capture gap analysis."""
        from minivess.compliance.complops import ComplianceCheckResult

        result = ComplianceCheckResult(
            template="fda_510k",
            total_requirements=10,
            satisfied=7,
            gaps=["Clinical evidence", "Predicate comparison", "Biocompatibility"],
        )
        assert result.total_requirements == 10
        assert result.satisfied == 7
        assert len(result.gaps) == 3

    def test_compliance_score(self) -> None:
        """score property should compute percentage."""
        from minivess.compliance.complops import ComplianceCheckResult

        result = ComplianceCheckResult(
            template="eu_mdr",
            total_requirements=10,
            satisfied=8,
            gaps=["Gap1", "Gap2"],
        )
        assert result.score == 0.8


# ---------------------------------------------------------------------------
# T3: 510(k) Summary
# ---------------------------------------------------------------------------


class TestFDA510kSummary:
    """Test FDA 510(k) summary generation."""

    def test_generation(self) -> None:
        """generate_510k_summary should produce structured markdown."""
        from minivess.compliance.complops import generate_510k_summary

        md = generate_510k_summary(
            device_name="MiniVess Segmentation",
            predicate_device="Manual vessel tracing by radiologist",
            intended_use="Automated 3D vessel segmentation",
            technological_characteristics="3D U-Net deep learning model",
        )
        assert "510(k)" in md
        assert "MiniVess" in md
        assert "Predicate" in md or "predicate" in md

    def test_sections_present(self) -> None:
        """510(k) summary should contain required sections."""
        from minivess.compliance.complops import generate_510k_summary

        md = generate_510k_summary(
            device_name="Test Device",
            predicate_device="Predicate",
            intended_use="Testing",
            technological_characteristics="DL model",
            performance_data="Dice=0.85",
        )
        assert "Intended Use" in md
        assert "Performance" in md


# ---------------------------------------------------------------------------
# T4: EU MDR Technical File
# ---------------------------------------------------------------------------


class TestEUMDRTechFile:
    """Test EU MDR technical file generation."""

    def test_generation(self) -> None:
        """generate_eu_mdr_technical_file should produce structured markdown."""
        from minivess.compliance.complops import generate_eu_mdr_technical_file

        md = generate_eu_mdr_technical_file(
            device_name="MiniVess Segmentation",
            manufacturer="MiniVess Research Group",
            device_class="IIa",
            intended_purpose="3D vessel segmentation for research",
        )
        assert "EU MDR" in md or "Technical" in md
        assert "MiniVess" in md

    def test_annex_sections(self) -> None:
        """Technical file should reference MDR Annex II/III requirements."""
        from minivess.compliance.complops import generate_eu_mdr_technical_file

        md = generate_eu_mdr_technical_file(
            device_name="Test",
            manufacturer="Test Corp",
            device_class="IIb",
            intended_purpose="Testing",
            clinical_evaluation="Literature review",
        )
        assert "Annex" in md or "Clinical" in md


# ---------------------------------------------------------------------------
# T5: Gap Assessment
# ---------------------------------------------------------------------------


class TestComplianceGapAssessment:
    """Test automated compliance gap analysis."""

    def test_gap_detection(self) -> None:
        """assess_compliance_gaps should identify missing items."""
        from minivess.compliance.complops import assess_compliance_gaps

        result = assess_compliance_gaps(
            template="fda_510k",
            provided_items={
                "device_description": True,
                "intended_use": True,
                "predicate_comparison": False,
                "performance_testing": False,
                "biocompatibility": True,
            },
        )
        assert result.satisfied >= 3
        assert len(result.gaps) >= 2

    def test_full_compliance(self) -> None:
        """All items provided should yield score of 1.0."""
        from minivess.compliance.complops import assess_compliance_gaps

        result = assess_compliance_gaps(
            template="fda_510k",
            provided_items={
                "device_description": True,
                "intended_use": True,
                "predicate_comparison": True,
            },
        )
        assert result.score == 1.0
        assert len(result.gaps) == 0
