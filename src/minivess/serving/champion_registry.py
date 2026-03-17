"""Multi-family champion model registry.

Manages champion models per model family (CNN, Foundation, Mamba).
Supports data-driven selection from Biostatistics flow output with
human-in-the-loop alerting when no statistically significant
difference between candidates.

Usage:
    registry = ChampionRegistry()
    registry.register(ChampionEntry(
        model_family="cnn",
        model_name="dynunet_quarter",
        checkpoint_path="/checkpoints/dynunet_best.pt",
        dice_score=0.85,
        cldice_score=0.72,
        mlflow_run_id="abc123",
    ))
    champion = registry.get_champion("cnn")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class ChampionEntry:
    """A single champion model entry.

    Represents the best-performing model within a model family,
    selected by the Biostatistics flow.
    """

    model_family: str  # "cnn", "foundation", "mamba"
    model_name: str  # e.g. "dynunet_quarter", "sam3_vanilla"
    checkpoint_path: str  # Path to the .pt checkpoint
    dice_score: float  # Best Dice on validation/test set
    cldice_score: float  # Best clDice on validation/test set
    mlflow_run_id: str  # MLflow run that produced this champion
    onnx_path: str | None = field(default=None)  # Optional ONNX export


class ChampionRegistry:
    """Registry of champion models, one per model family.

    The registry is populated by the Biostatistics flow which
    evaluates all trained models and selects the best per family.
    """

    def __init__(self) -> None:
        self._champions: dict[str, ChampionEntry] = {}

    @property
    def families(self) -> list[str]:
        """Return list of registered model family names."""
        return list(self._champions.keys())

    def register(self, entry: ChampionEntry) -> None:
        """Register or update a champion for a model family.

        If a champion already exists for this family, it is replaced.
        """
        self._champions[entry.model_family] = entry

    def get_champion(self, family: str) -> ChampionEntry:
        """Get the champion model for a given family.

        Raises:
            KeyError: If no champion registered for *family*.
        """
        if family not in self._champions:
            available = ", ".join(sorted(self._champions)) or "(none)"
            msg = (
                f"No champion registered for family '{family}'. Available: {available}"
            )
            raise KeyError(msg)
        return self._champions[family]

    def list_champions(self) -> list[ChampionEntry]:
        """Return all registered champion entries."""
        return list(self._champions.values())

    def to_yaml_path(self, path: Path) -> None:
        """Serialize registry to YAML config file."""
        import yaml

        data = {}
        for family, entry in self._champions.items():
            data[family] = {
                "model_name": entry.model_name,
                "checkpoint_path": entry.checkpoint_path,
                "dice_score": entry.dice_score,
                "cldice_score": entry.cldice_score,
                "mlflow_run_id": entry.mlflow_run_id,
                "onnx_path": entry.onnx_path,
            }
        path.write_text(
            yaml.dump(data, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )


def needs_human_decision(
    scores: dict[str, float],
    epsilon: float = 0.01,
) -> bool:
    """Check if model scores are too close for automated selection.

    When the range of scores is within *epsilon*, the difference is
    not statistically meaningful and a human should decide.

    Args:
        scores: Mapping of model_name → metric score.
        epsilon: Minimum range for automated decision.

    Returns:
        True if human review is needed (scores within epsilon).
    """
    if len(scores) < 2:
        return False
    values = list(scores.values())
    return (max(values) - min(values)) < epsilon


def select_best(scores: dict[str, float]) -> str:
    """Select the model with the highest score.

    Args:
        scores: Mapping of model_name → metric score.

    Returns:
        Name of the best-performing model.
    """
    return max(scores, key=scores.get)  # type: ignore[arg-type]
