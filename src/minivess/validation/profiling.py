"""whylogs profiling for data drift baselines.

Thin wrappers around whylogs for DataFrame profiling and comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ProfileDriftReport:
    """Result of comparing two whylogs profiles."""

    drifted_columns: list[str] = field(default_factory=list)
    column_summaries: dict[str, dict[str, float]] = field(default_factory=dict)


class DatasetProfileView:
    """Wrapper around a whylogs profile view for consistent API."""

    def __init__(self, view: object) -> None:
        self._view = view

    def get_columns(self) -> list[str]:
        """Return the column names in the profile."""
        return list(self._view.get_columns().keys())

    @property
    def raw(self) -> object:
        """Access the underlying whylogs view object."""
        return self._view


def profile_dataframe(df: pd.DataFrame) -> DatasetProfileView:
    """Profile a DataFrame with whylogs.

    Parameters
    ----------
    df:
        DataFrame to profile.

    Returns
    -------
    DatasetProfileView wrapping the whylogs result view.
    """
    import whylogs as why

    result = why.log(df)
    view = result.view()
    return DatasetProfileView(view)


def compare_profiles(
    reference: DatasetProfileView,
    current: DatasetProfileView,
    *,
    stddev_threshold: float = 2.0,
) -> ProfileDriftReport:
    """Compare two profiles for drift using mean/stddev heuristics.

    Parameters
    ----------
    reference:
        Reference profile view.
    current:
        Current profile view.
    stddev_threshold:
        Number of standard deviations for mean shift to flag drift.

    Returns
    -------
    ProfileDriftReport with drifted columns.
    """
    drifted: list[str] = []
    summaries: dict[str, dict[str, float]] = {}

    ref_cols = reference._view.get_columns()
    cur_cols = current._view.get_columns()

    for col_name in ref_cols:
        if col_name not in cur_cols:
            continue

        ref_col = ref_cols[col_name]
        cur_col = cur_cols[col_name]

        ref_summary = ref_col.to_summary_dict()
        cur_summary = cur_col.to_summary_dict()

        ref_mean = ref_summary.get("distribution/mean")
        cur_mean = cur_summary.get("distribution/mean")
        ref_stddev = ref_summary.get("distribution/stddev")

        if ref_mean is None or cur_mean is None or ref_stddev is None:
            continue

        summaries[col_name] = {
            "ref_mean": float(ref_mean),
            "cur_mean": float(cur_mean),
            "ref_stddev": float(ref_stddev),
        }

        # Drift if mean shift exceeds threshold * stddev
        if ref_stddev > 0 and abs(cur_mean - ref_mean) > stddev_threshold * ref_stddev:
            drifted.append(col_name)

    return ProfileDriftReport(
        drifted_columns=drifted,
        column_summaries=summaries,
    )
