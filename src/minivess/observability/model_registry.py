"""Model registry with promotion stages and governance.

Implements structured model lifecycle management with semantic versioning
(Matthew, 2025), challenger-champion promotion workflow, and audit-trail
integration for IEC 62304 and EU AI Act Article 12 compliance.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class ModelStage(StrEnum):
    """Model lifecycle stages for registry governance."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelVersion:
    """A versioned model entry in the registry.

    Parameters
    ----------
    model_name:
        Model identifier.
    version:
        Semantic version string (major.minor.patch).
        - major: architecture change
        - minor: retrain with new data
        - patch: hyperparameter tuning
    stage:
        Current lifecycle stage.
    metrics:
        Evaluation metrics from locked test set.
    metadata:
        Arbitrary key-value metadata.
    """

    model_name: str
    version: str
    stage: ModelStage
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def major(self) -> int:
        """Major version (architecture change)."""
        return int(self.version.split(".")[0])

    @property
    def minor(self) -> int:
        """Minor version (retrain)."""
        return int(self.version.split(".")[1])

    @property
    def patch(self) -> int:
        """Patch version (hyperparameter tuning)."""
        return int(self.version.split(".")[2])


@dataclass
class PromotionResult:
    """Result of a promotion evaluation.

    Parameters
    ----------
    approved:
        Whether the promotion was approved.
    reason:
        Human-readable explanation.
    metrics:
        Metrics that were evaluated.
    """

    approved: bool
    reason: str
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class PromotionCriteria:
    """Criteria for model promotion between stages.

    Parameters
    ----------
    min_thresholds:
        Metric name → minimum required value (e.g., dice >= 0.80).
    max_thresholds:
        Metric name → maximum allowed value (e.g., hd95 <= 5.0).
    """

    min_thresholds: dict[str, float] = field(default_factory=dict)
    max_thresholds: dict[str, float] = field(default_factory=dict)

    def check(self, metrics: dict[str, float]) -> PromotionResult:
        """Evaluate metrics against criteria.

        Parameters
        ----------
        metrics:
            Actual metric values to check.
        """
        failures: list[str] = []

        for name, threshold in self.min_thresholds.items():
            value = metrics.get(name, 0.0)
            if value < threshold:
                failures.append(f"{name}={value:.4f} < min {threshold:.4f}")

        for name, threshold in self.max_thresholds.items():
            value = metrics.get(name, float("inf"))
            if value > threshold:
                failures.append(f"{name}={value:.4f} > max {threshold:.4f}")

        if failures:
            return PromotionResult(
                approved=False,
                reason="Criteria not met: " + "; ".join(failures),
                metrics=dict(metrics),
            )
        return PromotionResult(
            approved=True,
            reason="All criteria met",
            metrics=dict(metrics),
        )


class ModelRegistry:
    """Model registry with structured promotion stages.

    Manages model versions through the lifecycle:
    DEVELOPMENT → STAGING → PRODUCTION → ARCHIVED
    """

    def __init__(self) -> None:
        self._versions: dict[str, dict[str, ModelVersion]] = defaultdict(dict)
        self._history: list[dict[str, Any]] = []

    def register_version(
        self,
        model_name: str,
        version: str,
        metrics: dict[str, float],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ModelVersion:
        """Register a new model version in DEVELOPMENT stage.

        Parameters
        ----------
        model_name:
            Model identifier.
        version:
            Semantic version string.
        metrics:
            Evaluation metrics.
        """
        mv = ModelVersion(
            model_name=model_name,
            version=version,
            stage=ModelStage.DEVELOPMENT,
            metrics=dict(metrics),
            metadata=metadata or {},
        )
        self._versions[model_name][version] = mv
        self._log_event("REGISTER", model_name, version, ModelStage.DEVELOPMENT)
        return mv

    def get_version(self, model_name: str, version: str) -> ModelVersion:
        """Retrieve a specific model version."""
        return self._versions[model_name][version]

    def promote(
        self,
        model_name: str,
        version: str,
        target_stage: ModelStage,
        criteria: PromotionCriteria,
    ) -> PromotionResult:
        """Attempt to promote a model version to a new stage.

        Parameters
        ----------
        model_name:
            Model identifier.
        version:
            Version to promote.
        target_stage:
            Target lifecycle stage.
        criteria:
            Promotion criteria to evaluate.
        """
        mv = self._versions[model_name][version]
        result = criteria.check(mv.metrics)

        if result.approved:
            mv.stage = target_stage
            self._log_event("PROMOTE", model_name, version, target_stage)
        else:
            self._log_event(
                "REJECT", model_name, version, target_stage,
                reason=result.reason,
            )

        return result

    def get_production_model(self, model_name: str) -> ModelVersion | None:
        """Get the current production version for a model.

        Returns None if no version is in PRODUCTION stage.
        """
        for mv in self._versions.get(model_name, {}).values():
            if mv.stage == ModelStage.PRODUCTION:
                return mv
        return None

    def list_versions(self, model_name: str) -> list[ModelVersion]:
        """List all versions of a model."""
        return list(self._versions.get(model_name, {}).values())

    def _log_event(
        self,
        action: str,
        model_name: str,
        version: str,
        stage: ModelStage,
        *,
        reason: str = "",
    ) -> None:
        """Record a registry event for audit trail."""
        self._history.append({
            "timestamp": datetime.now(UTC).isoformat(),
            "action": action,
            "model_name": model_name,
            "version": version,
            "stage": stage.value,
            "reason": reason,
        })

    def to_markdown(self) -> str:
        """Generate a model registry report."""
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        sections = [
            "# Model Registry Report",
            "",
            f"**Generated:** {now}",
            "",
        ]

        if not self._versions:
            sections.append("No models registered.")
            sections.append("")
            return "\n".join(sections)

        for model_name in sorted(self._versions):
            versions = self._versions[model_name]
            sections.extend([
                f"## {model_name}",
                "",
                "| Version | Stage | Metrics |",
                "|---------|-------|---------|",
            ])
            for ver in sorted(versions.values(), key=lambda v: v.version):
                metrics_str = ", ".join(
                    f"{k}={v:.4f}" for k, v in ver.metrics.items()
                )
                sections.append(
                    f"| {ver.version} | {ver.stage.value} | {metrics_str} |"
                )
            sections.append("")

        # History
        if self._history:
            sections.extend([
                "## Promotion History",
                "",
                "| Timestamp | Action | Model | Version | Stage |",
                "|-----------|--------|-------|---------|-------|",
            ])
            for event in self._history:
                sections.append(
                    f"| {event['timestamp'][:19]} | {event['action']} "
                    f"| {event['model_name']} | {event['version']} "
                    f"| {event['stage']} |"
                )
            sections.append("")

        return "\n".join(sections)
