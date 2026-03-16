"""WhyLogs continuous profiling service for 3D volumes (T-B2).

Profiles every volume using whylogs, supports profile merging across
batches, persistence to disk, and Prometheus text exposition export.

Usage:
    profiler = WhylogsVolumeProfiler()
    profile = profiler.profile_volume(volume, volume_id="vol_001")
    profiler.save_profile(profile, Path("profile.bin"))
    prom_text = format_whylogs_prometheus(profile, dataset="minivess")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import whylogs as why
from whylogs.core import DatasetProfile, DatasetProfileView

from minivess.data.feature_extraction import extract_volume_features

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

logger = logging.getLogger(__name__)

# Union type for profile or view — both support view-like operations
type ProfileLike = DatasetProfile | DatasetProfileView


def _to_view(profile: ProfileLike) -> DatasetProfileView:
    """Convert a DatasetProfile or DatasetProfileView to a view."""
    if isinstance(profile, DatasetProfileView):
        return profile
    return profile.view()


class WhylogsVolumeProfiler:
    """Profile 3D volumes using whylogs for continuous data monitoring.

    Extracts statistical features from each volume and logs them
    as whylogs profiles. Profiles are mergeable across batches
    and exportable to Prometheus format.
    """

    def profile_volume(
        self,
        volume: np.ndarray,
        volume_id: str,
        *,
        tags: dict[str, str] | None = None,
    ) -> DatasetProfile:
        """Profile a single 3D volume.

        Parameters
        ----------
        volume:
            3D numpy array (H, W, D) or (C, H, W, D).
        volume_id:
            Unique identifier for the volume.
        tags:
            Optional metadata tags (e.g., dataset, split).

        Returns
        -------
        whylogs DatasetProfile with extracted features.
        """
        # Squeeze channel dim if present
        vol = volume.squeeze() if volume.ndim > 3 else volume
        features = extract_volume_features(vol)

        # Log only numeric features (whylogs + numpy 2.0 has str type issues)
        numeric_features = {
            k: float(v) for k, v in features.items() if isinstance(v, int | float)
        }
        result = why.log(numeric_features)
        profile = result.profile()

        if tags:
            for key, value in tags.items():
                profile.metadata[key] = value

        logger.debug(
            "Profiled volume %s: %d features", volume_id, len(numeric_features)
        )
        return profile

    def profile_batch(
        self,
        volumes: list[np.ndarray],
        volume_ids: list[str] | None = None,
    ) -> list[DatasetProfile]:
        """Profile a batch of volumes.

        Parameters
        ----------
        volumes:
            List of 3D numpy arrays.
        volume_ids:
            Optional list of volume identifiers. Auto-generated if None.

        Returns
        -------
        List of whylogs DatasetProfiles, one per volume.
        """
        if volume_ids is None:
            volume_ids = [f"vol_{i:03d}" for i in range(len(volumes))]

        profiles = []
        for vol, vid in zip(volumes, volume_ids, strict=True):
            p = self.profile_volume(vol, volume_id=vid)
            profiles.append(p)

        logger.info("Profiled batch of %d volumes", len(volumes))
        return profiles

    @staticmethod
    def get_column_names(profile: ProfileLike) -> list[str]:
        """Get column names from a profile or view.

        Parameters
        ----------
        profile:
            whylogs DatasetProfile or DatasetProfileView.

        Returns
        -------
        List of column names in the profile.
        """
        view = _to_view(profile)
        return list(view.get_columns().keys())

    @staticmethod
    def merge_profiles(profiles: list[ProfileLike]) -> DatasetProfileView:
        """Merge multiple profiles into a single summary view.

        Merging happens at the DatasetProfileView level (whylogs 1.x API).

        Parameters
        ----------
        profiles:
            List of whylogs DatasetProfiles or DatasetProfileViews.

        Returns
        -------
        Merged DatasetProfileView.
        """
        if not profiles:
            msg = "Cannot merge empty list of profiles"
            raise ValueError(msg)

        merged_view = _to_view(profiles[0])
        for p in profiles[1:]:
            merged_view = merged_view.merge(_to_view(p))

        logger.debug("Merged %d profiles", len(profiles))
        return merged_view

    @staticmethod
    def save_profile(profile: ProfileLike, path: Path) -> Path:
        """Save a profile to disk as a binary file.

        Parameters
        ----------
        profile:
            whylogs DatasetProfile or DatasetProfileView.
        path:
            Output path for the serialized profile.

        Returns
        -------
        Path to the saved file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        _to_view(profile).write(str(path))
        logger.info("Profile saved to %s", path)
        return path

    @staticmethod
    def load_profile(path: Path) -> DatasetProfileView:
        """Load a profile from disk.

        Parameters
        ----------
        path:
            Path to a serialized whylogs profile.

        Returns
        -------
        Loaded DatasetProfileView.
        """
        view = DatasetProfileView.read(str(path))
        logger.info("Profile loaded from %s", path)
        return view

    def compare_profiles(
        self,
        reference: ProfileLike,
        current: ProfileLike,
    ) -> dict[str, Any]:
        """Compare two profiles for drift detection.

        Uses whylogs summary statistics to detect distribution shifts
        between reference and current profiles.

        Parameters
        ----------
        reference:
            Reference (baseline) profile or view.
        current:
            Current (production) profile or view.

        Returns
        -------
        Dict with drift_detected, drifted_columns, and per-column stats.
        """
        ref_view = _to_view(reference)
        cur_view = _to_view(current)

        ref_cols = ref_view.get_columns()
        cur_cols = cur_view.get_columns()

        drifted_columns: list[str] = []
        column_stats: dict[str, dict[str, float]] = {}

        for col_name in ref_cols:
            if col_name == "volume_id" or col_name not in cur_cols:
                continue

            ref_col = ref_cols[col_name]
            cur_col = cur_cols[col_name]

            ref_summary = _extract_column_summary(ref_col)
            cur_summary = _extract_column_summary(cur_col)

            if ref_summary is None or cur_summary is None:
                continue

            # Simple drift heuristic: mean shift > 2 * reference std
            ref_mean = ref_summary.get("mean", 0.0)
            ref_std = ref_summary.get("stddev", 1.0)
            cur_mean = cur_summary.get("mean", 0.0)

            if ref_std > 0 and abs(cur_mean - ref_mean) > 2.0 * ref_std:
                drifted_columns.append(col_name)

            column_stats[col_name] = {
                "ref_mean": ref_mean,
                "cur_mean": cur_mean,
                "ref_std": ref_std,
                "mean_shift": abs(cur_mean - ref_mean),
            }

        return {
            "drift_detected": len(drifted_columns) > 0,
            "n_drifted_columns": len(drifted_columns),
            "drifted_columns": drifted_columns,
            "column_stats": column_stats,
        }


def _extract_column_summary(column_profile: Any) -> dict[str, float] | None:
    """Extract summary stats from a whylogs column profile.

    Parameters
    ----------
    column_profile:
        whylogs column profile view.

    Returns
    -------
    Dict with mean, stddev, min, max or None if not available.
    """
    try:
        dist = column_profile.get_metric("distribution")
        if dist is None:
            return None

        # whylogs 1.x: mean is FractionalComponent (.value), others are plain floats
        mean_val = dist.mean.value if hasattr(dist.mean, "value") else float(dist.mean)
        return {
            "mean": float(mean_val),
            "stddev": float(dist.stddev),
            "min": float(dist.min),
            "max": float(dist.max),
        }
    except (AttributeError, TypeError):
        return None


def format_whylogs_prometheus(
    profile: ProfileLike,
    dataset: str = "default",
) -> str:
    """Format whylogs profile as Prometheus text exposition.

    Parameters
    ----------
    profile:
        whylogs DatasetProfile or DatasetProfileView.
    dataset:
        Dataset label for Prometheus metrics.

    Returns
    -------
    Multi-line Prometheus text format string.
    """
    view = _to_view(profile)
    columns = view.get_columns()
    lines: list[str] = []

    n_columns = len([c for c in columns if c != "volume_id"])
    lines.append(f'whylogs_profile_column_count{{dataset="{dataset}"}} {n_columns}')

    for col_name, col_profile in columns.items():
        if col_name == "volume_id":
            continue

        summary = _extract_column_summary(col_profile)
        if summary is None:
            continue

        safe_name = col_name.replace(" ", "_").replace("-", "_")
        for stat_name, stat_value in summary.items():
            lines.append(
                f'whylogs_column_{stat_name}{{dataset="{dataset}",column="{safe_name}"}} {stat_value}'
            )

    return "\n".join(lines) + "\n"
