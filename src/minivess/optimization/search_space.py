"""Search space definition for HPO experiments.

Loads hyperparameter search spaces from dicts (parsed from YAML configs)
and validates parameter specifications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_VALID_PARAM_TYPES = frozenset({"float", "int", "categorical"})


@dataclass
class SearchSpace:
    """Hyperparameter search space for Optuna studies.

    Attributes
    ----------
    params:
        Dict mapping parameter names to their specs.
        Each spec contains ``type`` and type-specific keys.
    """

    params: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, Any]]) -> SearchSpace:
        """Create a SearchSpace from a dict (e.g. parsed YAML).

        Parameters
        ----------
        data:
            Dict mapping param names to specs with ``type`` key.

        Returns
        -------
        Validated SearchSpace instance.

        Raises
        ------
        ValueError
            If any param spec is missing ``type`` or has unknown type.
        """
        for name, spec in data.items():
            if "type" not in spec:
                msg = f"Param {name!r} missing 'type' key"
                raise ValueError(msg)
            if spec["type"] not in _VALID_PARAM_TYPES:
                msg = (
                    f"Param {name!r} has unknown type {spec['type']!r}. "
                    f"Valid: {sorted(_VALID_PARAM_TYPES)}"
                )
                raise ValueError(msg)
        return cls(params=dict(data))

    @classmethod
    def from_yaml(cls, path: str) -> SearchSpace:
        """Load search space from a YAML file.

        Parameters
        ----------
        path:
            Path to YAML file containing search_space key.

        Returns
        -------
        SearchSpace instance.
        """
        from pathlib import Path as _Path

        import yaml

        yaml_path = _Path(path)
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if "search_space" in data:
            return cls.from_dict(data["search_space"])
        return cls.from_dict(data)
