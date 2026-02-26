"""Model promotion: champion/challenger tagging for MLflow Model Registry.

Compares evaluation results across all models (single and ensemble) and
promotes the best model based on the configured primary metric.  Supports
environment-specific aliases (``staging-champion``, ``prod-champion``) and
per-loss best aliases.

References
----------
* MLflow Model Registry aliases (2.9+): aliases replace deprecated stages.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import mlflow
from mlflow.tracking import MlflowClient

from minivess.observability.tracking import resolve_tracking_uri
from minivess.pipeline.validation_metrics import compute_compound_masd_cldice

if TYPE_CHECKING:
    from minivess.config.evaluation_config import EvaluationConfig
    from minivess.pipeline.evaluation_runner import EvaluationResult

logger = logging.getLogger(__name__)


class ModelPromoter:
    """Promote the best model to champion in MLflow Model Registry.

    Compares evaluation results across all models (single and ensemble)
    and promotes the best one based on the configured primary metric.

    Parameters
    ----------
    eval_config:
        :class:`EvaluationConfig` with primary metric and registry settings.
    """

    # Metrics where we can compute a compound score from raw components
    _COMPOUND_METRIC = "val_compound_masd_cldice"

    # Mapping from config metric names to MetricsReloaded field names
    _METRIC_ALIASES: dict[str, str] = {
        "val_compound_masd_cldice": "__compound__",
        "val_dice": "dsc",
        "val_cldice": "centreline_dsc",
        "val_masd": "measured_masd",
        "val_f1_foreground": "dsc",
        "dsc": "dsc",
        "centreline_dsc": "centreline_dsc",
        "measured_masd": "measured_masd",
    }

    def __init__(self, eval_config: EvaluationConfig) -> None:
        self.eval_config = eval_config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_best_model(
        self,
        results: dict[str, dict[str, dict[str, EvaluationResult]]],
    ) -> tuple[str, float]:
        """Find the best model by primary metric across all results.

        Parameters
        ----------
        results:
            ``{model_name: {dataset: {subset: EvaluationResult}}}``.

        Returns
        -------
        ``(model_name, best_metric_value)``

        Raises
        ------
        ValueError
            If *results* is empty.
        """
        rankings = self.rank_models(results)
        return rankings[0]

    def rank_models(
        self,
        results: dict[str, dict[str, dict[str, EvaluationResult]]],
    ) -> list[tuple[str, float]]:
        """Rank all models by primary metric (best first).

        For each model the metric is averaged across all dataset/subset
        combinations.  Ranking respects
        :attr:`EvaluationConfig.primary_metric_direction`.

        Parameters
        ----------
        results:
            ``{model_name: {dataset: {subset: EvaluationResult}}}``.

        Returns
        -------
        ``[(model_name, avg_metric), ...]`` sorted best-first.

        Raises
        ------
        ValueError
            If *results* is empty.
        """
        if not results:
            msg = "No results to rank: results dict is empty"
            raise ValueError(msg)

        model_scores: list[tuple[str, float]] = []
        for model_name, ds_dict in results.items():
            values: list[float] = []
            for _ds_name, subset_dict in ds_dict.items():
                for _subset_name, eval_result in subset_dict.items():
                    metric_value = self._extract_metric(eval_result)
                    if not math.isnan(metric_value):
                        values.append(metric_value)
            avg = float(sum(values) / len(values)) if values else float("nan")
            model_scores.append((model_name, avg))

        from minivess.config.evaluation_config import MetricDirection

        reverse = self.eval_config.primary_metric_direction == MetricDirection.MAXIMIZE
        model_scores.sort(key=lambda t: t[1], reverse=reverse)
        return model_scores

    def generate_promotion_report(
        self,
        rankings: list[tuple[str, float]],
        *,
        champion_name: str,
    ) -> str:
        """Generate a markdown promotion report.

        Parameters
        ----------
        rankings:
            ``[(model_name, metric_value), ...]`` sorted best-first.
        champion_name:
            Name of the model promoted to champion.

        Returns
        -------
        Markdown-formatted report string.
        """
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        metric_name = self.eval_config.primary_metric
        direction = self.eval_config.primary_metric_direction.value

        lines: list[str] = [
            "# Model Promotion Report",
            "",
            f"**Generated:** {now}",
            f"**Primary metric:** `{metric_name}` ({direction})",
            "",
            "| Rank | Model | Primary Metric | Status |",
            "|------|-------|---------------|--------|",
        ]

        for rank_idx, (name, value) in enumerate(rankings, start=1):
            status = "Champion" if name == champion_name else "Challenger"
            lines.append(f"| {rank_idx} | {name} | {value:.4f} | {status} |")

        lines.append("")
        return "\n".join(lines)

    def register_champion(
        self,
        model_name: str,
        *,
        run_id: str,
        registry_name: str | None = None,
        environment: str = "staging",
        tracking_uri: str | None = None,
    ) -> dict[str, str]:
        """Register model in MLflow and set champion alias.

        Sets aliases:
          - ``"{environment}-champion"`` (e.g., ``"staging-champion"``)
          - ``"champion"`` (default/latest)

        Parameters
        ----------
        model_name:
            Descriptive name of the champion model.
        run_id:
            MLflow run_id containing the model artifact.
        registry_name:
            MLflow Model Registry name.  Falls back to
            :attr:`EvaluationConfig.model_registry_name`.
        environment:
            Environment label (``"staging"``, ``"production"``).
        tracking_uri:
            MLflow tracking URI override.

        Returns
        -------
        Dict with ``model_name``, ``version``, ``aliases``, ``run_id``.
        """
        effective_registry = registry_name or self.eval_config.model_registry_name
        resolved_uri = resolve_tracking_uri(tracking_uri=tracking_uri)
        mlflow.set_tracking_uri(resolved_uri)
        client = MlflowClient(tracking_uri=resolved_uri)

        model_uri = f"runs:/{run_id}/model"
        mv = mlflow.register_model(model_uri, effective_registry)

        env_alias = f"{environment}-champion"
        aliases = [env_alias, "champion"]

        for alias in aliases:
            client.set_registered_model_alias(effective_registry, alias, mv.version)

        logger.info(
            "Registered champion model '%s' as %s v%s with aliases %s",
            model_name,
            effective_registry,
            mv.version,
            aliases,
        )

        return {
            "model_name": model_name,
            "registry_name": effective_registry,
            "version": str(mv.version),
            "aliases": ", ".join(aliases),
            "run_id": run_id,
        }

    def set_per_loss_aliases(
        self,
        loss_rankings: dict[str, list[tuple[str, float]]],
        *,
        tracking_uri: str | None = None,
    ) -> None:
        """Set per-loss best aliases like ``best-dice_ce``.

        For each loss function, creates a ``best-{loss_name}`` alias
        pointing to the top-ranked model version for that loss.

        Parameters
        ----------
        loss_rankings:
            ``{loss_name: [(model_name, metric_value), ...]}``.
        tracking_uri:
            MLflow tracking URI override.
        """
        registry_name = self.eval_config.model_registry_name
        resolved_uri = resolve_tracking_uri(tracking_uri=tracking_uri)
        mlflow.set_tracking_uri(resolved_uri)
        client = MlflowClient(tracking_uri=resolved_uri)

        for loss_name, rankings in loss_rankings.items():
            if not rankings:
                continue

            best_model_name = rankings[0][0]
            alias = f"best-{loss_name}"

            # Look up the latest version for this registry
            try:
                versions = client.search_model_versions(f"name='{registry_name}'")
                if versions:
                    latest_version = max(versions, key=lambda v: int(v.version))
                    client.set_registered_model_alias(
                        registry_name, alias, latest_version.version
                    )
                    logger.info(
                        "Set alias '%s' -> %s v%s (model: %s)",
                        alias,
                        registry_name,
                        latest_version.version,
                        best_model_name,
                    )
            except Exception:
                logger.warning(
                    "Could not set alias '%s' for loss '%s'",
                    alias,
                    loss_name,
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_metric(self, eval_result: EvaluationResult) -> float:
        """Extract the primary metric value from an EvaluationResult.

        Handles the compound metric (val_compound_masd_cldice) by computing
        it from raw MASD and clDice components when available.

        Parameters
        ----------
        eval_result:
            Single evaluation result with a :class:`FoldResult`.

        Returns
        -------
        Metric value (float), or ``nan`` if not computable.
        """
        primary = self.eval_config.primary_metric
        aggregated = eval_result.fold_result.aggregated

        # Special case: compound metric from components
        if primary == self._COMPOUND_METRIC:
            masd_ci = aggregated.get("measured_masd")
            cldice_ci = aggregated.get("centreline_dsc")
            if masd_ci is not None and cldice_ci is not None:
                return compute_compound_masd_cldice(
                    masd=masd_ci.point_estimate,
                    cldice=cldice_ci.point_estimate,
                )
            return float("nan")

        # Direct lookup via alias mapping
        field_name = self._METRIC_ALIASES.get(primary, primary)
        ci = aggregated.get(field_name)
        if ci is not None:
            return ci.point_estimate

        # Fallback: try raw metric name directly
        ci = aggregated.get(primary)
        if ci is not None:
            return ci.point_estimate

        logger.warning(
            "Metric '%s' (resolved to '%s') not found in aggregated results; "
            "available: %s",
            primary,
            field_name,
            list(aggregated.keys()),
        )
        return float("nan")
