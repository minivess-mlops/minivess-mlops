"""Generate SkyPilot YAML with ordered: region failover from composable configs.

Reads a base SkyPilot YAML template and a region config file, then produces
a new YAML file with an ``ordered:`` block injected into ``resources:``.
The ``ordered:`` block specifies priority-ordered (region, accelerator) pairs
so SkyPilot tries EU regions first, then US fallback.

When no region config is provided (e.g., local smoke tests), the base YAML
is returned unchanged.

Usage from run_factorial.sh::

    GENERATED_YAML=$(python3 -m minivess.cloud.region_injection \
        --base deployment/skypilot/train_factorial.yaml \
        --region-config configs/cloud/regions/europe_us.yaml \
        --output-dir /tmp/skypilot)
    sky jobs launch "${GENERATED_YAML}" ...

Plan: docs/planning/cold-start-prompt-composable-regions-phase2.md (Task 1)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def generate_skypilot_yaml(
    *,
    base_yaml_path: Path,
    region_config_path: Path | None,
    output_dir: Path,
) -> Path:
    """Generate a SkyPilot YAML with ordered: region block injected.

    Parameters
    ----------
    base_yaml_path:
        Path to the base SkyPilot YAML template (e.g., train_factorial.yaml).
    region_config_path:
        Path to a region config YAML (e.g., configs/cloud/regions/europe_us.yaml).
        If ``None``, returns ``base_yaml_path`` unchanged (bypass mode).
    output_dir:
        Directory where the generated YAML is written.

    Returns
    -------
    Path
        Path to the generated YAML (or ``base_yaml_path`` if no region config).
    """
    if region_config_path is None:
        return base_yaml_path

    base = yaml.safe_load(base_yaml_path.read_text(encoding="utf-8"))
    region_cfg = yaml.safe_load(region_config_path.read_text(encoding="utf-8"))

    # Extract the cloud provider from base YAML resources
    resources = base["resources"]
    cloud = resources.get("cloud", "gcp")

    # Build ordered: list from region config
    # Region config structure: regions.<provider>.<gpu_type>[].region
    provider_regions = region_cfg["regions"][cloud]

    ordered: list[dict[str, str]] = []
    for gpu_type, entries in provider_regions.items():
        accelerator = f"{gpu_type}:1"
        for entry in entries:
            ordered.append({
                "accelerators": accelerator,
                "region": entry["region"],
            })

    # Inject ordered: into resources, remove standalone accelerators:
    resources["ordered"] = ordered
    resources.pop("accelerators", None)

    # Write generated YAML
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"generated_{base_yaml_path.stem}.yaml"
    output_path.write_text(
        yaml.dump(base, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )

    return output_path


def main() -> None:
    """CLI entry point for use from shell scripts."""
    parser = argparse.ArgumentParser(
        description="Generate SkyPilot YAML with ordered: region block"
    )
    parser.add_argument("--base", required=True, help="Base SkyPilot YAML path")
    parser.add_argument("--region-config", default=None, help="Region config YAML path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    region_path = Path(args.region_config) if args.region_config else None
    result = generate_skypilot_yaml(
        base_yaml_path=Path(args.base),
        region_config_path=region_path,
        output_dir=Path(args.output_dir),
    )
    # Print the path so shell scripts can capture it
    print(result)


if __name__ == "__main__":
    main()
