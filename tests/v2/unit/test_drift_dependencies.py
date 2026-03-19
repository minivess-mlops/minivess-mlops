"""Tests that drift detection dependencies are importable (#574 T1, #599).

Verifies evidently and scipy permutation testing are available for the
three-tier drift detection pipeline.

Spec adaptation: alibi-detect requires numpy<2.0 (project needs numpy>=2.0).
T3 uses scipy.stats.permutation_test for MMD p-values instead of alibi-detect.
"""

from __future__ import annotations

import sys

import pytest

# evidently has unescaped regex in docstrings that causes SyntaxError on Python 3.13+.
try:
    import evidently as _evidently  # noqa: F401
except SyntaxError:
    pytest.skip(
        "evidently SyntaxError (invalid escape sequences on Python "
        f"{sys.version_info.major}.{sys.version_info.minor})",
        allow_module_level=True,
    )


class TestDriftDependencies:
    """Verify drift detection library imports."""

    def test_evidently_importable(self) -> None:
        """evidently package must be importable."""
        import evidently  # noqa: F401

    def test_evidently_data_drift_preset_available(self) -> None:
        """Evidently DataDriftPreset must be importable for Tier 1 reports."""
        from evidently.presets import DataDriftPreset  # noqa: F401

    def test_scipy_permutation_test_available(self) -> None:
        """scipy.stats.permutation_test must exist for Tier 2 MMD p-values.

        Replaces alibi-detect MMDDrift which requires numpy<2.0.
        """
        from scipy.stats import permutation_test  # noqa: F401

    def test_sklearn_rbf_kernel_available(self) -> None:
        """sklearn.metrics.pairwise.rbf_kernel must exist for kernel MMD."""
        from sklearn.metrics.pairwise import rbf_kernel  # noqa: F401
