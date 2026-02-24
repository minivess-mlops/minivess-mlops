"""Adaptation method comparison for foundation model customization.

Provides infrastructure to compare full fine-tuning, LoRA, atlas-guided
one-shot, and zero-shot approaches for adapting foundation models to
new anatomical targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class AdaptationMethod(StrEnum):
    """Foundation model adaptation strategies."""

    FULL_FINETUNE = "full_finetune"
    LORA = "lora"
    ATLAS_ONESHOT = "atlas_oneshot"
    ZERO_SHOT = "zero_shot"


@dataclass
class AdaptationResult:
    """Result of a single adaptation method evaluation.

    Parameters
    ----------
    method:
        Adaptation strategy used.
    dice_score:
        Dice similarity coefficient on the target anatomy.
    trainable_params:
        Number of trainable parameters.
    total_params:
        Total model parameters.
    training_time_s:
        Wall-clock training time in seconds.
    notes:
        Free-text observations.
    """

    method: str
    dice_score: float
    trainable_params: int
    total_params: int
    training_time_s: float = 0.0
    notes: str = ""

    @property
    def parameter_efficiency(self) -> float:
        """Fraction of parameters that are trainable."""
        if self.total_params == 0:
            return 0.0
        return self.trainable_params / self.total_params


def compare_adaptation_methods(
    results: list[AdaptationResult],
) -> list[AdaptationResult]:
    """Sort adaptation results by Dice score (descending).

    Parameters
    ----------
    results:
        List of adaptation method evaluations.

    Returns
    -------
    Sorted list (best Dice first).
    """
    return sorted(results, key=lambda r: r.dice_score, reverse=True)


@dataclass
class FeasibilityReport:
    """Feasibility analysis report for foundation model adaptation.

    Parameters
    ----------
    model_name:
        Name of the base segmentation model.
    target_anatomy:
        Anatomical target for adaptation.
    results:
        Adaptation method evaluation results.
    """

    model_name: str
    target_anatomy: str
    results: list[AdaptationResult] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Generate a feasibility analysis markdown report."""
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        sections = [
            "# AtlasSegFM Feasibility Analysis",
            "",
            f"**Generated:** {now}",
            f"**Base Model:** {self.model_name}",
            f"**Target Anatomy:** {self.target_anatomy}",
            "",
        ]

        if not self.results:
            sections.append("No adaptation results available.")
            sections.append("")
            return "\n".join(sections)

        ranked = compare_adaptation_methods(self.results)

        sections.extend([
            "## Adaptation Method Comparison",
            "",
            "| Rank | Method | Dice | Trainable Params | Efficiency | Time (s) |",
            "|------|--------|------|-----------------|------------|----------|",
        ])

        for i, r in enumerate(ranked, 1):
            eff = f"{r.parameter_efficiency:.4f}" if r.total_params > 0 else "N/A"
            sections.append(
                f"| {i} | {r.method} | {r.dice_score:.4f} "
                f"| {r.trainable_params:,} | {eff} "
                f"| {r.training_time_s:.1f} |"
            )

        # Recommendation
        best = ranked[0]
        sections.extend([
            "",
            "## Recommendation",
            "",
            f"**Best method:** {best.method} (Dice = {best.dice_score:.4f})",
            "",
        ])

        # Atlas-specific note
        atlas_results = [r for r in ranked if r.method == "atlas_oneshot"]
        if atlas_results:
            ar = atlas_results[0]
            sections.extend([
                "## Atlas One-Shot Assessment",
                "",
                f"- Dice: {ar.dice_score:.4f}",
                f"- Trainable parameters: {ar.trainable_params:,} "
                f"(zero additional training required)",
                f"- Training time: {ar.training_time_s:.1f}s "
                "(registration only)",
                "",
            ])

            # Viability assessment
            if ar.dice_score >= 0.75:
                sections.append(
                    "**Verdict:** Atlas one-shot is *viable* for rapid "
                    "prototyping. Consider for data-scarce scenarios."
                )
            else:
                sections.append(
                    "**Verdict:** Atlas one-shot performance is below "
                    "threshold (0.75 Dice). Recommend LoRA or full "
                    "fine-tuning for production use."
                )

        sections.append("")
        return "\n".join(sections)
