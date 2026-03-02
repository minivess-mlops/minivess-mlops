"""Generic auxiliary GT precomputation for Flow 1 (Data Engineering).

Precomputes auxiliary ground truth targets (e.g., SDF, centerline distance)
for all volumes. Task-agnostic — computes whatever AuxTargetConfig specifies.
Idempotent: skips files that already exist unless force=True.
"""

from __future__ import annotations

import logging
from pathlib import Path

import nibabel as nib
import numpy as np

from minivess.data.multitask_targets import AuxTargetConfig  # noqa: TC001

logger = logging.getLogger(__name__)


def precompute_auxiliary_targets(
    volume_pairs: list[dict[str, str]],
    output_dir: Path,
    target_configs: list[AuxTargetConfig],
    *,
    force: bool = False,
) -> dict[str, int]:
    """Precompute auxiliary GT targets for all volumes.

    For each volume and each target config, computes the auxiliary
    target from the label mask and saves as NIfTI.

    Args:
        volume_pairs: List of dicts with 'label' and 'volume_id' keys.
        output_dir: Directory to save precomputed NIfTI files.
        target_configs: List of AuxTargetConfig defining what to compute.
        force: If True, recompute even if file exists.

    Returns:
        Dict with 'computed' and 'skipped' counts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    computed = 0
    skipped = 0

    for pair in volume_pairs:
        volume_id = pair.get("volume_id", Path(pair["label"]).stem.split(".")[0])
        label_path = Path(pair["label"])
        label_img = nib.load(str(label_path))
        label_data: np.ndarray | None = None  # lazy load

        for config in target_configs:
            out_path = output_dir / f"{volume_id}_{config.suffix}.nii.gz"

            if out_path.exists() and not force:
                logger.debug("Skipping existing: %s", out_path)
                skipped += 1
                continue

            # Lazy load label data
            if label_data is None:
                label_data = np.asarray(label_img.dataobj)

            logger.info(
                "Computing %s for %s via %s",
                config.name,
                volume_id,
                config.compute_fn.__name__,
            )
            target = config.compute_fn(label_data)
            target = target.astype(np.float32)

            # Save with same affine as label
            nib.save(
                nib.Nifti1Image(target, label_img.affine),
                str(out_path),
            )
            computed += 1

    logger.info("Precompute complete: %d computed, %d skipped", computed, skipped)
    return {"computed": computed, "skipped": skipped}
