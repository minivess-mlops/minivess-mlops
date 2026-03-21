"""Uncertainty storage policy: debug vs production (#887).

Debug runs store only summary statistics (scalar metrics per volume).
Production runs store full 5D uncertainty maps (~50MB/vol) as artifacts.

This avoids bloating MLflow artifact storage during iterative debug runs
while retaining full spatial UQ data for production analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import (
    Path,  # noqa: TC003 — dataclass needs runtime access
)
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UncertaintyStoragePolicy:
    """Policy controlling what UQ artifacts are persisted.

    Parameters
    ----------
    debug:
        If True, store only scalar summaries.
        If False, store full 5D maps + summaries.
    output_dir:
        Directory for saved uncertainty artifacts.
    """

    debug: bool
    output_dir: Path

    def store(
        self,
        *,
        volume_id: str,
        total_uncertainty: Tensor,
        aleatoric_uncertainty: Tensor,
        epistemic_uncertainty: Tensor,
    ) -> dict[str, Any]:
        """Store uncertainty outputs according to the policy.

        Parameters
        ----------
        volume_id:
            Identifier for the volume (e.g., "vol_042").
        total_uncertainty:
            Total predictive entropy (B, 1, D, H, W).
        aleatoric_uncertainty:
            Aleatoric uncertainty (B, 1, D, H, W).
        epistemic_uncertainty:
            Epistemic / mutual information (B, 1, D, H, W).

        Returns
        -------
        Dict with scalar summaries and optionally file paths to saved maps.
        """
        # Always compute scalar summaries
        summaries = {
            "total_mean": float(total_uncertainty.mean()),
            "total_max": float(total_uncertainty.max()),
            "aleatoric_mean": float(aleatoric_uncertainty.mean()),
            "aleatoric_max": float(aleatoric_uncertainty.max()),
            "epistemic_mean": float(epistemic_uncertainty.mean()),
            "epistemic_max": float(epistemic_uncertainty.max()),
        }

        result: dict[str, Any] = {
            "volume_id": volume_id,
            "summaries": summaries,
            "maps_saved": False,
        }

        if not self.debug:
            # Production: save full 5D maps as .pt files
            maps_dir = self.output_dir / "uncertainty_maps"
            maps_dir.mkdir(parents=True, exist_ok=True)

            map_path = maps_dir / f"{volume_id}_uncertainty.pt"
            torch.save(
                {
                    "total": total_uncertainty.cpu(),
                    "aleatoric": aleatoric_uncertainty.cpu(),
                    "epistemic": epistemic_uncertainty.cpu(),
                },
                map_path,
            )
            result["maps_saved"] = True
            result["map_path"] = str(map_path)
            logger.info("Saved full UQ maps for %s (%s)", volume_id, map_path)
        else:
            logger.debug("Debug mode: storing only UQ summaries for %s", volume_id)

        return result
