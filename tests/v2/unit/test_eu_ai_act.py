"""Tests for EU AI Act compliance checklist (Issue #20)."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# T1: EUAIActRiskLevel enum
# ---------------------------------------------------------------------------


class TestEUAIActRiskLevel:
    """Test EU AI Act risk level classification enum."""

    def test_enum_values(self) -> None:
        """EUAIActRiskLevel should have four risk tiers."""
        from minivess.compliance.eu_ai_act import EUAIActRiskLevel

        assert EUAIActRiskLevel.UNACCEPTABLE == "unacceptable"
        assert EUAIActRiskLevel.HIGH == "high"
        assert EUAIActRiskLevel.LIMITED == "limited"
        assert EUAIActRiskLevel.MINIMAL == "minimal"

    def test_enum_membership(self) -> None:
        """All four risk levels should be present."""
        from minivess.compliance.eu_ai_act import EUAIActRiskLevel

        names = {e.name for e in EUAIActRiskLevel}
        assert names == {"UNACCEPTABLE", "HIGH", "LIMITED", "MINIMAL"}

    def test_medical_ai_classified_as_high(self) -> None:
        """Medical AI (SaMD) should be classified as HIGH risk."""
        from minivess.compliance.eu_ai_act import classify_risk_level

        level = classify_risk_level(is_medical_device=True)
        assert level.value == "high"


# ---------------------------------------------------------------------------
# T2: EUAIActChecklist
# ---------------------------------------------------------------------------


class TestEUAIActChecklist:
    """Test EU AI Act compliance checklist dataclass."""

    def test_construction(self) -> None:
        """EUAIActChecklist should be constructible with required fields."""
        from minivess.compliance.eu_ai_act import EUAIActChecklist

        checklist = EUAIActChecklist(
            system_name="MiniVess Segmentation",
            risk_level="high",
            intended_purpose="3D vessel segmentation for neurovascular research",
        )
        assert checklist.system_name == "MiniVess Segmentation"
        assert checklist.risk_level == "high"

    def test_to_markdown(self) -> None:
        """to_markdown should produce a structured compliance report."""
        from minivess.compliance.eu_ai_act import EUAIActChecklist

        checklist = EUAIActChecklist(
            system_name="MiniVess",
            risk_level="high",
            intended_purpose="Vessel segmentation",
        )
        md = checklist.to_markdown()
        assert "EU AI Act" in md
        assert "MiniVess" in md
        assert "high" in md.lower()

    def test_required_fields_in_markdown(self) -> None:
        """Markdown should contain all EU AI Act required sections."""
        from minivess.compliance.eu_ai_act import EUAIActChecklist

        checklist = EUAIActChecklist(
            system_name="MiniVess",
            risk_level="high",
            intended_purpose="Vessel segmentation",
            data_governance="GDPR-compliant anonymized datasets",
            transparency="Model card and audit trail provided",
            human_oversight="Clinician reviews all predictions",
            robustness="Cross-validation and drift monitoring",
            risk_management="IEC 62304 lifecycle + PPRM monitoring",
        )
        md = checklist.to_markdown()
        assert "Data Governance" in md
        assert "Transparency" in md
        assert "Human Oversight" in md
        assert "Robustness" in md
        assert "Risk Management" in md

    def test_optional_fields_default_empty(self) -> None:
        """Optional fields should default to empty strings."""
        from minivess.compliance.eu_ai_act import EUAIActChecklist

        checklist = EUAIActChecklist(
            system_name="Test",
            risk_level="minimal",
            intended_purpose="Testing",
        )
        assert checklist.data_governance == ""
        assert checklist.transparency == ""
        assert checklist.human_oversight == ""


# ---------------------------------------------------------------------------
# T3: Compliance report generation
# ---------------------------------------------------------------------------


class TestComplianceReport:
    """Test EU AI Act compliance report generation."""

    def test_generate_compliance_report(self) -> None:
        """generate_compliance_report should produce markdown from params."""
        from minivess.compliance.eu_ai_act import generate_compliance_report

        report = generate_compliance_report(
            system_name="MiniVess Segmentation Pipeline",
            intended_purpose="3D vessel segmentation",
            is_medical_device=True,
        )
        assert "EU AI Act" in report
        assert "MiniVess" in report
        assert "HIGH" in report or "high" in report.lower()

    def test_report_includes_date(self) -> None:
        """Generated report should include generation date."""
        from minivess.compliance.eu_ai_act import generate_compliance_report

        report = generate_compliance_report(
            system_name="Test",
            intended_purpose="Testing",
            is_medical_device=False,
        )
        assert "Generated:" in report

    def test_report_gap_analysis(self) -> None:
        """Report should flag missing compliance areas as gaps."""
        from minivess.compliance.eu_ai_act import generate_compliance_report

        report = generate_compliance_report(
            system_name="Test System",
            intended_purpose="Testing",
            is_medical_device=True,
        )
        # With no optional fields provided, gaps should be identified
        assert "Gap" in report or "gap" in report.lower() or "Not provided" in report
