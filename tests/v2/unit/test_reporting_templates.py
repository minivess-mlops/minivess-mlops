"""Tests for CONSORT-AI and MI-CLEAR-LLM reporting templates (Issue #14)."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# T1: ConsortAIChecklist
# ---------------------------------------------------------------------------


class TestConsortAIChecklist:
    """Test CONSORT-AI reporting checklist."""

    def test_construction(self) -> None:
        """ConsortAIChecklist should be constructible with required fields."""
        from minivess.compliance.reporting_templates import ConsortAIChecklist

        checklist = ConsortAIChecklist(
            title="MiniVess Segmentation Study",
            ai_intervention="3D U-Net for vessel segmentation",
            training_data="MiniVess dataset (20 volumes, 5-fold CV)",
            performance_metrics={"dice": 0.85, "hd95": 3.2},
        )
        assert checklist.title == "MiniVess Segmentation Study"
        assert checklist.performance_metrics["dice"] == 0.85

    def test_to_markdown(self) -> None:
        """to_markdown should produce structured report."""
        from minivess.compliance.reporting_templates import ConsortAIChecklist

        checklist = ConsortAIChecklist(
            title="Test Study",
            ai_intervention="Segmentation model",
            training_data="Test data",
            performance_metrics={"dice": 0.85},
        )
        md = checklist.to_markdown()
        assert "CONSORT-AI" in md
        assert "Test Study" in md
        assert "dice" in md

    def test_required_sections_present(self) -> None:
        """Markdown should contain all CONSORT-AI required sections."""
        from minivess.compliance.reporting_templates import ConsortAIChecklist

        checklist = ConsortAIChecklist(
            title="Study",
            ai_intervention="Model",
            training_data="Data",
            performance_metrics={"dice": 0.8},
            input_data_description="3D volumes",
            output_description="Binary masks",
            preprocessing_steps="Intensity normalization",
            evaluation_protocol="5-fold cross-validation",
        )
        md = checklist.to_markdown()
        assert "AI Intervention" in md
        assert "Training Data" in md
        assert "Performance" in md
        assert "Input" in md
        assert "Output" in md
        assert "Preprocessing" in md
        assert "Evaluation" in md

    def test_optional_fields_default(self) -> None:
        """Optional fields should have sensible defaults."""
        from minivess.compliance.reporting_templates import ConsortAIChecklist

        checklist = ConsortAIChecklist(
            title="Study",
            ai_intervention="Model",
            training_data="Data",
            performance_metrics={},
        )
        assert checklist.input_data_description == ""
        assert checklist.limitations == ""


# ---------------------------------------------------------------------------
# T2: MiClearLLMChecklist
# ---------------------------------------------------------------------------


class TestMiClearLLMChecklist:
    """Test MI-CLEAR-LLM reporting checklist."""

    def test_construction(self) -> None:
        """MiClearLLMChecklist should be constructible."""
        from minivess.compliance.reporting_templates import MiClearLLMChecklist

        checklist = MiClearLLMChecklist(
            task_description="Automated report generation",
            llm_model="Claude 3.5 Sonnet",
            prompt_strategy="System + user prompts with examples",
            output_validation="Human review of all outputs",
        )
        assert checklist.llm_model == "Claude 3.5 Sonnet"

    def test_to_markdown(self) -> None:
        """to_markdown should produce structured report."""
        from minivess.compliance.reporting_templates import MiClearLLMChecklist

        checklist = MiClearLLMChecklist(
            task_description="Report generation",
            llm_model="Claude",
            prompt_strategy="Chain of thought",
            output_validation="Human review",
        )
        md = checklist.to_markdown()
        assert "MI-CLEAR-LLM" in md
        assert "Claude" in md

    def test_required_sections_present(self) -> None:
        """Markdown should contain MI-CLEAR-LLM required sections."""
        from minivess.compliance.reporting_templates import MiClearLLMChecklist

        checklist = MiClearLLMChecklist(
            task_description="Generate reports",
            llm_model="GPT-4",
            prompt_strategy="Few-shot",
            output_validation="Human review",
            hallucination_mitigation="RAG + citation verification",
            temperature=0.3,
            api_version="2024-01",
        )
        md = checklist.to_markdown()
        assert "Task" in md
        assert "Model" in md
        assert "Prompt" in md
        assert "Validation" in md
        assert "Hallucination" in md


# ---------------------------------------------------------------------------
# T3: Integration with model card
# ---------------------------------------------------------------------------


class TestReportingIntegration:
    """Test integration of reporting templates."""

    def test_generate_consort_from_metrics(self) -> None:
        """generate_consort_ai_report should create report from metrics dict."""
        from minivess.compliance.reporting_templates import generate_consort_ai_report

        report = generate_consort_ai_report(
            title="Vessel Segmentation",
            model_name="DynUNet-full",
            metrics={"dice": 0.85, "hd95": 3.2},
            dataset_description="MiniVess (20 volumes)",
        )
        assert "CONSORT-AI" in report
        assert "DynUNet-full" in report
        assert "dice" in report

    def test_generate_miclear_from_config(self) -> None:
        """generate_miclear_llm_report should create report from LLM config."""
        from minivess.compliance.reporting_templates import (
            generate_miclear_llm_report,
        )

        report = generate_miclear_llm_report(
            task="Regulatory document generation",
            model="Claude 3.5 Sonnet",
            prompt_strategy="System prompt with role definition",
        )
        assert "MI-CLEAR-LLM" in report
        assert "Claude" in report

    def test_reports_include_date(self) -> None:
        """Generated reports should include generation date."""
        from minivess.compliance.reporting_templates import generate_consort_ai_report

        report = generate_consort_ai_report(
            title="Test",
            model_name="Test",
            metrics={},
            dataset_description="Test",
        )
        assert "Generated:" in report
