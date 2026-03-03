#!/usr/bin/env python
"""Precompute auxiliary GT targets (SDF, centreline distance) for topology experiments.

Usage::

    # From experiment name (Hydra composition)
    uv run python scripts/precompute_targets.py \\
        --experiment dynunet_topology_all_approaches_debug

    # Force recompute
    uv run python scripts/precompute_targets.py \\
        --experiment dynunet_topology_all_approaches_debug --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minivess.orchestration.precompute import precompute_auxiliary_targets

logger = logging.getLogger(__name__)

# Registry of known compute functions by string name
_COMPUTE_FN_REGISTRY: dict[str, str] = {
    "compute_sdf_from_mask": "minivess.pipeline.sdf_generation",
    "compute_centreline_distance_map": "minivess.adapters.centreline_head",
}


def discover_volumes(data_dir: Path) -> list[dict[str, str]]:
    """Discover label volumes from a MiniVess-style data directory.

    Looks for ``labelsTr/*.nii.gz`` and extracts volume IDs from filenames.

    Parameters
    ----------
    data_dir:
        Root data directory containing ``labelsTr/`` subdirectory.

    Returns
    -------
    List of dicts with ``label`` (path) and ``volume_id`` keys.
    """
    labels_dir = data_dir / "labelsTr"
    if not labels_dir.exists():
        logger.warning("No labelsTr/ found in %s", data_dir)
        return []

    volumes: list[dict[str, str]] = []
    for label_path in sorted(labels_dir.glob("*.nii.gz")):
        volume_id = label_path.name.replace(".nii.gz", "")
        volumes.append({"label": str(label_path), "volume_id": volume_id})

    logger.info("Discovered %d volumes in %s", len(volumes), labels_dir)
    return volumes


def resolve_target_configs(
    yaml_configs: list[dict[str, Any]],
) -> list[Any]:
    """Resolve YAML target config dicts to AuxTargetConfig objects.

    Maps string function names (e.g., ``"compute_sdf_from_mask"``) to actual
    callables via a registry.

    Parameters
    ----------
    yaml_configs:
        List of dicts with ``name``, ``suffix``, and ``compute_fn`` (string) keys.

    Returns
    -------
    List of ``AuxTargetConfig`` instances.
    """
    import importlib

    from minivess.data.multitask_targets import AuxTargetConfig

    configs: list[AuxTargetConfig] = []
    for cfg in yaml_configs:
        fn_name = cfg["compute_fn"]
        module_path = _COMPUTE_FN_REGISTRY.get(fn_name)
        if module_path is None:
            msg = f"Unknown compute function: {fn_name}. Known: {list(_COMPUTE_FN_REGISTRY)}"
            raise ValueError(msg)

        module = importlib.import_module(module_path)
        fn = getattr(module, fn_name)
        configs.append(
            AuxTargetConfig(name=cfg["name"], suffix=cfg["suffix"], compute_fn=fn)
        )

    return configs


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for precomputing auxiliary targets.

    Parameters
    ----------
    argv:
        CLI argument list. Defaults to ``sys.argv[1:]`` when ``None``.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Precompute auxiliary GT targets for topology experiments",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name for Hydra composition (e.g., 'dynunet_topology_all_approaches_debug')",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompute even if files exist",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: data_dir/precomputed)",
    )
    args = parser.parse_args(argv)

    # Load experiment config via Hydra composition
    from minivess.config.compose import compose_experiment_config

    config: dict[str, Any] = compose_experiment_config(experiment_name=args.experiment)

    # Resolve paths
    data_dir = Path(config.get("data_dir", "data/raw/minivess"))
    output_dir = args.output_dir or data_dir / "precomputed"

    # Get precompute_targets from config
    target_yaml = config.get("precompute_targets", [])
    if not target_yaml:
        logger.warning("No precompute_targets defined in config — nothing to do")
        return

    # Resolve compute functions
    target_configs = resolve_target_configs(target_yaml)
    logger.info(
        "Resolved %d target configs: %s",
        len(target_configs),
        [c.name for c in target_configs],
    )

    # Discover volumes
    volumes = discover_volumes(data_dir)
    logger.info("Found %d volumes to process", len(volumes))

    # Run precomputation
    result = precompute_auxiliary_targets(
        volumes, output_dir, target_configs, force=args.force
    )
    logger.info(
        "Precompute complete: %d computed, %d skipped",
        result["computed"],
        result["skipped"],
    )


if __name__ == "__main__":
    main()
