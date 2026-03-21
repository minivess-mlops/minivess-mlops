"""Post-training plugin protocol and dataclasses.

Defines the contract for post-hoc processing plugins (checkpoint averaging, merging,
calibration, conformal) and a registry for name-based lookup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003 — used at runtime in dataclass
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class PluginInput:
    """Input data for a post-training plugin.

    Attributes
    ----------
    checkpoint_paths:
        Paths to model checkpoint files.
    config:
        Plugin-specific configuration dict.
    calibration_data:
        Optional calibration data (scores, labels) for data-dependent plugins.
    run_metadata:
        Optional per-checkpoint metadata (run_id, loss_type, fold_id, etc.).
    """

    checkpoint_paths: list[Path]
    config: dict[str, Any]
    calibration_data: dict[str, Any] | None = None
    run_metadata: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PluginOutput:
    """Output from a post-training plugin.

    Attributes
    ----------
    artifacts:
        Mapping of artifact names to paths or values.
    metrics:
        Computed metrics (e.g., ECE before/after, coverage).
    model_paths:
        Paths to produced model files (state dicts, ONNX, etc.).
    """

    artifacts: dict[str, Any]
    metrics: dict[str, float]
    model_paths: list[Path] = field(default_factory=list)


@runtime_checkable
class PostTrainingPlugin(Protocol):
    """Protocol for post-training plugins.

    Each plugin wraps a post-hoc method (checkpoint averaging, merging, calibration, etc.)
    and integrates with the post-training Prefect flow via a uniform interface.
    """

    @property
    def name(self) -> str:
        """Unique plugin name used for registry lookup and MLflow tagging."""
        ...

    @property
    def requires_calibration_data(self) -> bool:
        """Whether this plugin needs calibration data to execute."""
        ...

    def validate_inputs(self, plugin_input: PluginInput) -> list[str]:
        """Validate inputs before execution.

        Returns
        -------
        List of error messages. Empty list means inputs are valid.
        """
        ...

    def execute(self, plugin_input: PluginInput) -> PluginOutput:
        """Execute the post-training method.

        Parameters
        ----------
        plugin_input:
            Checkpoint paths, config, and optional calibration data.

        Returns
        -------
        Plugin output with artifacts, metrics, and model paths.
        """
        ...


class PluginRegistry:
    """Registry mapping plugin names to implementations."""

    def __init__(self) -> None:
        self._plugins: dict[str, PostTrainingPlugin] = {}

    def register(self, plugin: PostTrainingPlugin) -> None:
        """Register a plugin by its name property."""
        self._plugins[plugin.name] = plugin
        logger.debug("Registered post-training plugin: %s", plugin.name)

    def get(self, name: str) -> PostTrainingPlugin:
        """Look up a plugin by name.

        Raises
        ------
        KeyError
            If no plugin is registered with the given name.
        """
        if name not in self._plugins:
            available = sorted(self._plugins.keys())
            msg = f"Unknown plugin: {name!r}. Available: {available}"
            raise KeyError(msg)
        return self._plugins[name]

    def all_names(self) -> list[str]:
        """Return sorted list of all registered plugin names."""
        return sorted(self._plugins.keys())
