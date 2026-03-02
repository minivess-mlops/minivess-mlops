"""Run Deploy Flow on real champion models — discover + ONNX export.

Discovers champion-tagged runs from mlruns filesystem and exports
the best checkpoint as ONNX. Validates ONNX model loads correctly.

Run:
    uv run python scripts/run_deploy_real.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
EXPERIMENT_ID = "843896622863223169"


def main() -> int:
    """Discover champions and export to ONNX."""
    import argparse

    parser = argparse.ArgumentParser(description="Deploy real champions")
    parser.add_argument(
        "--output-dir", default="outputs/deploy", help="Output directory"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("DEPLOY FLOW: Real Champion Models")
    logger.info("=" * 70)

    # 1. Discover champions
    logger.info("\n[1/3] Discovering champions")
    from minivess.pipeline.deploy_champion_discovery import discover_champions

    champions = discover_champions(MLRUNS_DIR, EXPERIMENT_ID)
    logger.info("  Found %d champion(s)", len(champions))

    if not champions:
        logger.warning("No champions found. Run scripts/tag_champions.py first.")
        return 1

    for champ in champions:
        logger.info(
            "    %s: run=%s, checkpoint=%s",
            champ.category,
            champ.run_id[:8],
            champ.checkpoint_path.name if champ.checkpoint_path else "NONE",
        )

    # 2. Export to ONNX
    logger.info("\n[2/3] Exporting to ONNX")
    from minivess.pipeline.deploy_onnx_export import export_champion_to_onnx

    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    onnx_paths: dict[str, Path] = {}
    for champ in champions:
        if not champ.checkpoint_path:
            logger.warning("  Skipping %s — no checkpoint", champ.category)
            continue

        try:
            onnx_path = export_champion_to_onnx(
                champ,
                onnx_dir,
                opset_version=17,
                input_shape=(1, 1, 32, 32, 16),
            )
            onnx_paths[champ.category] = onnx_path
            size_mb = onnx_path.stat().st_size / 1e6
            logger.info(
                "    %s: %s (%.1f MB)",
                champ.category,
                onnx_path.name,
                size_mb,
            )
        except Exception:
            logger.exception("    FAILED to export %s", champ.category)

    # 3. Validate ONNX models
    logger.info("\n[3/3] Validating ONNX models")
    from minivess.pipeline.deploy_onnx_export import validate_onnx_model

    for category, onnx_path in onnx_paths.items():
        try:
            is_valid = validate_onnx_model(
                onnx_path,
                input_shape=(1, 1, 32, 32, 16),
            )
            status = "VALID" if is_valid else "INVALID"
            logger.info("    %s: %s", category, status)
        except Exception:
            logger.exception("    %s: VALIDATION FAILED", category)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("  Champions discovered: %d", len(champions))
    logger.info("  ONNX exported:       %d", len(onnx_paths))

    all_files = list(output_dir.rglob("*"))
    artifact_files = [f for f in all_files if f.is_file()]
    logger.info("  Total artifacts:     %d", len(artifact_files))
    for f in sorted(artifact_files):
        size_mb = f.stat().st_size / 1e6
        logger.info("    %s (%.1f MB)", f.relative_to(output_dir), size_mb)

    logger.info("\nDEPLOY FLOW COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
