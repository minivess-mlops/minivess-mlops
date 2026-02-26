"""Clinical AI reporting standard templates.

CONSORT-AI (Kwong et al., 2025) — Minimum reporting for AI clinical trials.
MI-CLEAR-LLM (Park et al., 2025) — Minimum reporting for LLM clinical research.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass
class ConsortAIChecklist:
    """CONSORT-AI minimum reporting checklist for AI model documentation.

    Parameters
    ----------
    title:
        Study or experiment title.
    ai_intervention:
        Description of the AI system used.
    training_data:
        Training dataset description.
    performance_metrics:
        Metric name → value pairs.
    """

    title: str
    ai_intervention: str
    training_data: str
    performance_metrics: dict[str, float]
    input_data_description: str = ""
    output_description: str = ""
    preprocessing_steps: str = ""
    evaluation_protocol: str = ""
    confidence_intervals: str = ""
    failure_analysis: str = ""
    limitations: str = ""
    intended_use: str = ""

    def to_markdown(self) -> str:
        """Generate CONSORT-AI compliant markdown report."""
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        sections = [
            "# CONSORT-AI Reporting Checklist",
            "",
            f"**Title:** {self.title}",
            f"**Generated:** {now}",
            "",
            "## 1. AI Intervention",
            "",
            f"{self.ai_intervention}",
            "",
            "## 2. Training Data",
            "",
            f"{self.training_data}",
        ]

        if self.input_data_description:
            sections.extend(
                [
                    "",
                    "## 3. Input Data Description",
                    "",
                    f"{self.input_data_description}",
                ]
            )

        if self.output_description:
            sections.extend(
                [
                    "",
                    "## 4. Output Description",
                    "",
                    f"{self.output_description}",
                ]
            )

        if self.preprocessing_steps:
            sections.extend(
                [
                    "",
                    "## 5. Preprocessing",
                    "",
                    f"{self.preprocessing_steps}",
                ]
            )

        if self.evaluation_protocol:
            sections.extend(
                [
                    "",
                    "## 6. Evaluation Protocol",
                    "",
                    f"{self.evaluation_protocol}",
                ]
            )

        if self.performance_metrics:
            metrics_lines = "\n".join(
                f"| {k} | {v:.4f} |" for k, v in self.performance_metrics.items()
            )
            sections.extend(
                [
                    "",
                    "## 7. Performance Metrics",
                    "",
                    "| Metric | Value |",
                    "|--------|-------|",
                    metrics_lines,
                ]
            )

        if self.confidence_intervals:
            sections.extend(
                [
                    "",
                    "## 8. Confidence Intervals",
                    "",
                    f"{self.confidence_intervals}",
                ]
            )

        if self.failure_analysis:
            sections.extend(
                [
                    "",
                    "## 9. Failure Analysis",
                    "",
                    f"{self.failure_analysis}",
                ]
            )

        if self.limitations:
            sections.extend(
                [
                    "",
                    "## 10. Limitations",
                    "",
                    f"{self.limitations}",
                ]
            )

        if self.intended_use:
            sections.extend(
                [
                    "",
                    "## 11. Intended Use",
                    "",
                    f"{self.intended_use}",
                ]
            )

        sections.append("")
        return "\n".join(sections)


@dataclass
class MiClearLLMChecklist:
    """MI-CLEAR-LLM minimum reporting checklist for LLM components.

    Parameters
    ----------
    task_description:
        What the LLM is used for.
    llm_model:
        Model name and version.
    prompt_strategy:
        How prompts are constructed.
    output_validation:
        How outputs are verified.
    """

    task_description: str
    llm_model: str
    prompt_strategy: str
    output_validation: str
    hallucination_mitigation: str = ""
    temperature: float | None = None
    api_version: str = ""
    human_oversight: str = ""
    cost_reporting: str = ""

    def to_markdown(self) -> str:
        """Generate MI-CLEAR-LLM compliant markdown report."""
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        sections = [
            "# MI-CLEAR-LLM Reporting Checklist",
            "",
            f"**Generated:** {now}",
            "",
            "## 1. Task Description",
            "",
            f"{self.task_description}",
            "",
            "## 2. LLM Model",
            "",
            f"**Model:** {self.llm_model}",
        ]

        if self.api_version:
            sections.append(f"**API Version:** {self.api_version}")

        if self.temperature is not None:
            sections.append(f"**Temperature:** {self.temperature}")

        sections.extend(
            [
                "",
                "## 3. Prompt Strategy",
                "",
                f"{self.prompt_strategy}",
                "",
                "## 4. Output Validation",
                "",
                f"{self.output_validation}",
            ]
        )

        if self.hallucination_mitigation:
            sections.extend(
                [
                    "",
                    "## 5. Hallucination Mitigation",
                    "",
                    f"{self.hallucination_mitigation}",
                ]
            )

        if self.human_oversight:
            sections.extend(
                [
                    "",
                    "## 6. Human Oversight",
                    "",
                    f"{self.human_oversight}",
                ]
            )

        if self.cost_reporting:
            sections.extend(
                [
                    "",
                    "## 7. Cost Reporting",
                    "",
                    f"{self.cost_reporting}",
                ]
            )

        sections.append("")
        return "\n".join(sections)


def generate_consort_ai_report(
    *,
    title: str,
    model_name: str,
    metrics: dict[str, float],
    dataset_description: str,
    evaluation_protocol: str = "",
    confidence_intervals: str = "",
) -> str:
    """Generate a CONSORT-AI report from experiment data.

    Parameters
    ----------
    title:
        Study title.
    model_name:
        Model architecture name.
    metrics:
        Performance metric values.
    dataset_description:
        Dataset description.
    """
    checklist = ConsortAIChecklist(
        title=title,
        ai_intervention=f"Model: {model_name}",
        training_data=dataset_description,
        performance_metrics=metrics,
        evaluation_protocol=evaluation_protocol,
        confidence_intervals=confidence_intervals,
    )
    return checklist.to_markdown()


def generate_miclear_llm_report(
    *,
    task: str,
    model: str,
    prompt_strategy: str,
    output_validation: str = "Human review of all outputs",
    hallucination_mitigation: str = "",
) -> str:
    """Generate a MI-CLEAR-LLM report from LLM configuration.

    Parameters
    ----------
    task:
        What the LLM is used for.
    model:
        LLM model name.
    prompt_strategy:
        Prompting approach used.
    """
    checklist = MiClearLLMChecklist(
        task_description=task,
        llm_model=model,
        prompt_strategy=prompt_strategy,
        output_validation=output_validation,
        hallucination_mitigation=hallucination_mitigation,
    )
    return checklist.to_markdown()
