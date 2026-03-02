"""Run Data Flow (Flow 1) on all 70 real MiniVess volumes.

Discovers image/label pairs, validates data quality, generates K-fold splits,
and logs provenance metadata.

Run:
    uv run python scripts/run_data_flow_real.py
    uv run python scripts/run_data_flow_real.py --output-dir outputs/data_flow
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "minivess"


def main() -> int:
    """Run the data engineering flow on real MiniVess data."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Data Flow on MiniVess")
    parser.add_argument(
        "--output-dir",
        default="outputs/data_flow",
        help="Output directory for flow artifacts",
    )
    parser.add_argument("--n-folds", type=int, default=3, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("DATA FLOW: MiniVess (70 volumes)")
    logger.info("=" * 70)
    logger.info("  Data dir:   %s", DATA_DIR)
    logger.info("  Output dir: %s", output_dir)
    logger.info("  N folds:    %d", args.n_folds)
    logger.info("  Seed:       %d", args.seed)

    # Verify data exists
    if not DATA_DIR.exists():
        logger.error("Data directory not found: %s", DATA_DIR)
        return 1

    images_dir = DATA_DIR / "imagesTr"
    labels_dir = DATA_DIR / "labelsTr"
    n_images = len(list(images_dir.glob("*.nii.gz"))) if images_dir.exists() else 0
    n_labels = len(list(labels_dir.glob("*.nii.gz"))) if labels_dir.exists() else 0
    logger.info("  Images:     %d", n_images)
    logger.info("  Labels:     %d", n_labels)

    if n_images == 0:
        logger.error("No images found in %s", images_dir)
        return 1

    # Run the data flow
    logger.info("\n[1/2] Running data flow...")
    from minivess.orchestration.flows.data_flow import run_data_flow

    result = run_data_flow(
        data_dir=DATA_DIR,
        n_folds=args.n_folds,
        seed=args.seed,
    )

    # Report results
    logger.info("\n[2/2] Results")
    logger.info("  Discovered:     %d volumes", len(result.pairs))
    logger.info("  Quality passed: %s", result.quality_passed)

    if result.validation_report:
        report = result.validation_report
        logger.info("  Valid pairs:    %d / %d", report.n_valid, report.n_pairs)
        if hasattr(report, "errors") and report.errors:
            for err in report.errors:
                logger.warning("  Error: %s", err)

    if result.splits:
        logger.info("  Folds:          %d", len(result.splits))
        for i, fold in enumerate(result.splits):
            logger.info(
                "    Fold %d: %d train, %d val",
                i,
                len(fold.train),
                len(fold.val),
            )

    # Save provenance as JSON artifact
    if result.provenance:
        prov_path = output_dir / "data_provenance.json"
        prov_path.write_text(
            json.dumps(result.provenance, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("  Provenance:     %s", prov_path)

    # Save splits summary
    if result.splits:
        splits_summary = []
        for i, fold in enumerate(result.splits):
            splits_summary.append(
                {
                    "fold": i,
                    "n_train": len(fold.train),
                    "n_val": len(fold.val),
                    "train_ids": [
                        Path(p["image"]).stem.replace(".nii", "")
                        if isinstance(p, dict)
                        else str(p)
                        for p in fold.train
                    ],
                    "val_ids": [
                        Path(p["image"]).stem.replace(".nii", "")
                        if isinstance(p, dict)
                        else str(p)
                        for p in fold.val
                    ],
                }
            )
        splits_path = output_dir / "splits_summary.json"
        splits_path.write_text(
            json.dumps(splits_summary, indent=2),
            encoding="utf-8",
        )
        logger.info("  Splits saved:   %s", splits_path)

    # Save full result summary
    summary = {
        "n_volumes": len(result.pairs),
        "quality_passed": result.quality_passed,
        "n_folds": result.n_folds,
        "external_datasets": list(result.external_datasets.keys())
        if result.external_datasets
        else [],
    }
    summary_path = output_dir / "data_flow_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("  Summary saved:  %s", summary_path)

    logger.info("\n" + "=" * 70)
    logger.info("DATA FLOW %s", "COMPLETE" if result.quality_passed else "FAILED")
    logger.info("=" * 70)

    return 0 if result.quality_passed else 1


if __name__ == "__main__":
    sys.exit(main())
