"""Pydantic output models for agent decision points.

These models define the structured output schemas that Pydantic AI agents
return. They provide type safety and validation for LLM-generated decisions.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ExperimentSummary(BaseModel):
    """Structured experiment summary from LLM analysis.

    Used by the experiment summarizer agent in analysis_flow.
    """

    narrative: str = Field(description="2-3 sentence natural-language summary")
    best_model: str = Field(description="Best performing model identifier")
    best_metric_value: float = Field(description="Best metric value achieved")
    key_findings: list[str] = Field(
        description="3-5 key findings",
        min_length=1,
        max_length=5,
    )
    recommendations: list[str] = Field(
        description="0-3 actionable recommendations",
        default_factory=list,
        max_length=3,
    )


class DriftTriageResult(BaseModel):
    """Structured drift triage decision.

    Used by the drift triage agent in data_flow.
    """

    action: Literal["monitor", "retrain", "investigate"] = Field(
        description="Recommended action based on drift analysis"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Decision confidence (0=uncertain, 1=certain)",
    )
    reasoning: str = Field(description="Explanation of the triage decision")
    affected_features: list[str] = Field(
        default_factory=list,
        description="Features affected by drift",
    )
    severity: Literal["low", "medium", "high"] = Field(
        description="Overall drift severity"
    )


class FigureCaption(BaseModel):
    """Paper-quality figure caption.

    Used by the figure narration agent in biostatistics_flow.
    """

    caption: str = Field(description="Publication-ready caption (1-3 sentences)")
    alt_text: str = Field(description="Accessibility alt text for the figure")
    statistical_note: str | None = Field(
        default=None,
        description="Statistical test details if applicable",
    )


class AcquisitionDecision(BaseModel):
    """Structured acquisition decision from the conformal bandit agent.

    Used by the acquisition agent in acquisition_flow.
    """

    selected_volumes: list[str] = Field(
        description="Volume IDs selected for acquisition",
        default_factory=list,
    )
    reasoning: str = Field(description="Justification for the selection")
    uncertainty_summary: str = Field(
        description="Summary of uncertainty landscape",
        default="",
    )
    budget_used: float = Field(
        ge=0.0,
        description="Budget consumed by this acquisition round",
        default=0.0,
    )


class AnnotationPriority(BaseModel):
    """Structured annotation priority recommendation.

    Used by the active learning annotation agent.
    """

    recommended_volumes: list[str] = Field(
        description="Volume IDs recommended for annotation (priority order)",
        default_factory=list,
    )
    reasoning: str = Field(description="Justification for the ranking")
    expected_improvement: str = Field(
        description="Estimated impact on model performance",
        default="",
    )
    top_disagreement_score: float = Field(
        ge=0.0,
        description="Highest disagreement score in the ranking",
        default=0.0,
    )
