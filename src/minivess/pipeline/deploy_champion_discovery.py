"""Champion model discovery for the deploy flow.

Discovers champion-tagged MLflow runs from the filesystem and returns
typed ``ChampionModel`` dataclass instances for downstream deployment
tasks (ONNX export, BentoML import, artifact generation).

Pattern reference: ``champion_tagger.py`` (filesystem tag reading).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from minivess.config.deploy_config import ChampionCategory

logger = logging.getLogger(__name__)

# Champion tag prefix → deploy category mapping
_TAG_TO_CATEGORY: dict[str, str] = {
    "champion_best_single_fold": "overlap",
    "champion_best_cv_mean": "balanced",
    "champion_best_ensemble": "topology",
}


@dataclass
class ChampionModel:
    """A champion model discovered from MLflow filesystem tags.

    Parameters
    ----------
    run_id:
        MLflow run ID.
    experiment_id:
        MLflow experiment ID.
    category:
        Champion category (balanced, topology, overlap).
    metrics:
        Evaluation metrics read from the run.
    checkpoint_path:
        Path to model checkpoint (if found in artifacts).
    model_config:
        Model configuration dict (if available from params).
    """

    run_id: str
    experiment_id: str
    category: str
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_path: Path | None = None
    model_config: dict[str, Any] | None = None


def discover_champions(
    mlruns_dir: Path,
    experiment_id: str,
    *,
    categories: list[ChampionCategory] | None = None,
) -> list[ChampionModel]:
    """Discover champion-tagged runs from the MLflow filesystem.

    Parameters
    ----------
    mlruns_dir:
        Root MLflow tracking directory.
    experiment_id:
        MLflow experiment ID to search.
    categories:
        Optional filter — only return champions matching these categories.

    Returns
    -------
    List of :class:`ChampionModel` instances, one per discovered champion.
    """
    experiment_dir = mlruns_dir / experiment_id
    if not experiment_dir.is_dir():
        logger.warning("Experiment directory not found: %s", experiment_dir)
        return []

    champions: list[ChampionModel] = []
    category_filter = {c.value for c in categories} if categories else None

    for run_dir in experiment_dir.iterdir():
        if not run_dir.is_dir():
            continue
        tags_dir = run_dir / "tags"
        if not tags_dir.is_dir():
            continue

        # Check for champion tags
        champion_tag_type = _find_champion_tag(tags_dir)
        if champion_tag_type is None:
            continue

        category = _TAG_TO_CATEGORY.get(champion_tag_type, "balanced")
        if category_filter and category not in category_filter:
            continue

        run_id = run_dir.name
        metrics = _read_metrics(run_dir)
        checkpoint_path = _find_checkpoint(run_dir)
        model_config = _read_model_config(run_dir)

        champion = ChampionModel(
            run_id=run_id,
            experiment_id=experiment_id,
            category=category,
            metrics=metrics,
            checkpoint_path=checkpoint_path,
            model_config=model_config,
        )
        champions.append(champion)
        logger.info(
            "Discovered champion: run=%s category=%s",
            run_id,
            category,
        )

    return champions


def _find_champion_tag(tags_dir: Path) -> str | None:
    """Find which champion tag is present in a run's tags directory."""
    for tag_name in _TAG_TO_CATEGORY:
        tag_file = tags_dir / tag_name
        if tag_file.is_file():
            content = tag_file.read_text(encoding="utf-8").strip()
            if content == "true":
                return tag_name
    return None


def _read_metrics(run_dir: Path) -> dict[str, float]:
    """Read metrics from the MLflow run directory."""
    metrics_dir = run_dir / "metrics"
    if not metrics_dir.is_dir():
        return {}

    metrics: dict[str, float] = {}
    for metric_file in metrics_dir.iterdir():
        if not metric_file.is_file():
            continue
        try:
            # MLflow format: "timestamp value step" per line, take last line
            lines = metric_file.read_text(encoding="utf-8").strip().splitlines()
            if lines:
                last_line = lines[-1]
                parts = last_line.split()
                if len(parts) >= 2:  # noqa: PLR2004
                    metrics[metric_file.name] = float(parts[1])
        except (ValueError, IndexError):
            continue

    return metrics


def _find_checkpoint(run_dir: Path) -> Path | None:
    """Find the best checkpoint in the run's artifacts directory."""
    artifacts_dir = run_dir / "artifacts"
    if not artifacts_dir.is_dir():
        return None

    # Look for common checkpoint patterns
    for pattern in ("best_checkpoint.pt", "best_*.pt", "checkpoint_*.pt", "model.pt"):
        if "*" in pattern:
            matches = list(artifacts_dir.glob(pattern))
            if matches:
                return matches[0]
        else:
            path = artifacts_dir / pattern
            if path.is_file():
                return path

    # Fall back to any .pt file
    pt_files = list(artifacts_dir.glob("*.pt"))
    return pt_files[0] if pt_files else None


def _read_model_config(run_dir: Path) -> dict[str, Any] | None:
    """Read model configuration from run params."""
    params_dir = run_dir / "params"
    if not params_dir.is_dir():
        return None

    config: dict[str, Any] = {}
    for param_file in params_dir.iterdir():
        if not param_file.is_file():
            continue
        config[param_file.name] = param_file.read_text(encoding="utf-8").strip()

    return config if config else None
