"""Drift detection: feature-level KS tests (Tier 1) + kernel MMD (Tier 2).

Tier 1 — FeatureDriftDetector:
    Per-feature Kolmogorov-Smirnov tests on extracted image statistics.
    Interpretable, fast, good for acquisition shifts.

Tier 2 — EmbeddingDriftDetector:
    Kernel MMD test on model embeddings. Statistically rigorous for
    detecting semantic distribution shifts in high-dimensional spaces.
"""

from __future__ import annotations

import html
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import ks_2samp

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Result from a drift detection run."""

    drift_detected: bool
    dataset_drift_score: float
    feature_scores: dict[str, float] = field(default_factory=dict)
    drifted_features: list[str] = field(default_factory=list)
    n_features: int = 0
    n_drifted: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    evidently_html_path: Path | None = None


class FeatureDriftDetector:
    """Tier 1: Per-feature KS test drift detection.

    Uses two-sample Kolmogorov-Smirnov test per feature to detect
    distribution shifts in image statistics.  Dataset-level drift is
    flagged when more than ``drift_share`` of features individually
    drift (default 50 %).

    Parameters
    ----------
    reference_features:
        DataFrame of reference feature values (one row per sample).
    threshold:
        Significance level for per-feature drift (default 0.05).
    drift_share:
        Fraction of features that must drift to flag dataset drift
        (default 0.5).
    """

    def __init__(
        self,
        reference_features: pd.DataFrame,
        *,
        threshold: float = 0.05,
        drift_share: float = 0.5,
    ) -> None:
        self.reference = reference_features
        self.threshold = threshold
        self.drift_share = drift_share

    def detect(
        self,
        current_features: pd.DataFrame,
        *,
        evidently_html_path: Path | None = None,
    ) -> DriftResult:
        """Run drift detection on current features vs reference.

        Parameters
        ----------
        current_features:
            DataFrame of current feature values (same columns as reference).
        evidently_html_path:
            If provided, save an Evidently HTML report to this path.

        Returns
        -------
        DriftResult with per-feature KS p-values and overall drift assessment.
        """
        feature_scores: dict[str, float] = {}
        drifted_features: list[str] = []

        for col in self.reference.columns:
            ref_vals = self.reference[col].dropna().values
            cur_vals = current_features[col].dropna().values
            _stat, p_value = ks_2samp(ref_vals, cur_vals)
            feature_scores[col] = float(p_value)
            if p_value < self.threshold:
                drifted_features.append(col)

        n_features = len(feature_scores)
        n_drifted = len(drifted_features)
        dataset_drift_score = n_drifted / max(n_features, 1)
        dataset_drift = dataset_drift_score >= self.drift_share

        html_path = None
        if evidently_html_path is not None:
            html_path = self.save_evidently_html(
                current_features, output_path=evidently_html_path
            )

        return DriftResult(
            drift_detected=dataset_drift,
            dataset_drift_score=dataset_drift_score,
            feature_scores=feature_scores,
            drifted_features=drifted_features,
            n_features=n_features,
            n_drifted=n_drifted,
            evidently_html_path=html_path,
        )

    def generate_evidently_report(self, current_features: pd.DataFrame) -> Any:
        """Generate an Evidently DataDriftPreset report.

        Parameters
        ----------
        current_features:
            DataFrame of current feature values.

        Returns
        -------
        Evidently Snapshot object with drift analysis results.
        """
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report(metrics=[DataDriftPreset(drift_share=self.drift_share)])
        snapshot = report.run(
            reference_data=self.reference, current_data=current_features
        )
        return snapshot

    def save_evidently_html(
        self,
        current_features: pd.DataFrame,
        *,
        output_path: Path,
    ) -> Path:
        """Save an Evidently DataDriftPreset HTML report.

        Parameters
        ----------
        current_features:
            DataFrame of current feature values.
        output_path:
            Path to write the HTML report.

        Returns
        -------
        Path to the generated HTML file.
        """
        snapshot = self.generate_evidently_report(current_features)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(output_path))
        logger.info("Evidently drift report saved to %s", output_path)
        return output_path

    def get_evidently_dict(self, current_features: pd.DataFrame) -> dict[str, Any]:
        """Get Evidently report as a dictionary.

        Parameters
        ----------
        current_features:
            DataFrame of current feature values.

        Returns
        -------
        Dict with 'metrics' and 'tests' keys from Evidently Snapshot.
        """
        snapshot = self.generate_evidently_report(current_features)
        return snapshot.dict()  # type: ignore[no-any-return]

    def generate_html_report(
        self,
        current_features: pd.DataFrame,
        *,
        output_path: Path,
    ) -> Path:
        """Generate an HTML drift report.

        Parameters
        ----------
        current_features:
            DataFrame of current feature values.
        output_path:
            Path to write the HTML report.

        Returns
        -------
        Path to the generated HTML file.
        """
        result = self.detect(current_features)

        rows = []
        for col in sorted(result.feature_scores):
            p = result.feature_scores[col]
            drifted = col in result.drifted_features
            colour = "red" if drifted else "green"
            rows.append(
                f"<tr><td>{html.escape(col)}</td>"
                f"<td>{p:.4f}</td>"
                f'<td style="color:{colour}">{"DRIFT" if drifted else "OK"}</td></tr>'
            )

        body = "\n".join(rows)
        report_html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Drift Report</title>
<style>
  body {{ font-family: sans-serif; margin: 2em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
  th {{ background: #f5f5f5; }}
</style>
</head><body>
<h1>Feature Drift Report</h1>
<p>Generated: {result.timestamp:%Y-%m-%d %H:%M:%S UTC}</p>
<p>Dataset drift: <b>{"YES" if result.drift_detected else "NO"}</b>
   ({result.n_drifted}/{result.n_features} features drifted)</p>
<table>
<tr><th>Feature</th><th>KS p-value</th><th>Status</th></tr>
{body}
</table>
</body></html>"""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_html, encoding="utf-8")
        logger.info("Drift report saved to %s", output_path)
        return output_path


class EmbeddingDriftDetector:
    """Tier 2: Kernel MMD drift detection on embeddings.

    Uses a permutation-based Maximum Mean Discrepancy (MMD) test
    with RBF kernel to detect distribution shifts in embedding space.

    Parameters
    ----------
    reference_embeddings:
        Reference embedding array of shape (n_samples, embedding_dim).
    p_val_threshold:
        P-value threshold for drift detection (default 0.05).
    n_permutations:
        Number of permutations for the MMD test (default 100).
    """

    def __init__(
        self,
        reference_embeddings: NDArray[np.float32],
        *,
        p_val_threshold: float = 0.05,
        n_permutations: int = 100,
        kernel: str = "rbf",
    ) -> None:
        self.reference = reference_embeddings
        self.p_val_threshold = p_val_threshold
        self.n_permutations = n_permutations
        if kernel not in ("rbf", "linear"):
            msg = f"Unsupported kernel: {kernel!r}. Use 'rbf' or 'linear'."
            raise ValueError(msg)
        self.kernel = kernel

    def detect(self, current_embeddings: NDArray[np.float32]) -> DriftResult:
        """Run kernel MMD test on current embeddings vs reference.

        Parameters
        ----------
        current_embeddings:
            Current embedding array of shape (n_samples, embedding_dim).

        Returns
        -------
        DriftResult with MMD p-value as the drift score.
        """
        mmd_stat, p_value = self._permutation_mmd_test(
            self.reference, current_embeddings
        )

        return DriftResult(
            drift_detected=p_value < self.p_val_threshold,
            dataset_drift_score=float(p_value),
            feature_scores={"mmd_statistic": float(mmd_stat)},
            drifted_features=["embeddings"] if p_value < self.p_val_threshold else [],
            n_features=1,
            n_drifted=1 if p_value < self.p_val_threshold else 0,
        )

    def _permutation_mmd_test(
        self,
        x: NDArray[np.float32],
        y: NDArray[np.float32],
    ) -> tuple[float, float]:
        """Permutation-based kernel MMD test.

        Uses RBF kernel with median heuristic bandwidth.

        Parameters
        ----------
        x, y:
            Two samples to compare, shape (n, d).

        Returns
        -------
        (mmd_statistic, p_value)
        """
        n = len(x)
        m = len(y)
        combined = np.vstack([x, y])

        # Compute observed MMD
        observed_mmd = self._compute_mmd(x, y, kernel=self.kernel)

        # Permutation test
        rng = np.random.default_rng(42)
        count_ge = 0
        for _ in range(self.n_permutations):
            perm = rng.permutation(n + m)
            x_perm = combined[perm[:n]]
            y_perm = combined[perm[n:]]
            perm_mmd = self._compute_mmd(x_perm, y_perm, kernel=self.kernel)
            if perm_mmd >= observed_mmd:
                count_ge += 1

        p_value = (count_ge + 1) / (self.n_permutations + 1)
        return observed_mmd, p_value

    @staticmethod
    def _compute_mmd(
        x: NDArray[np.float32],
        y: NDArray[np.float32],
        *,
        kernel: str = "rbf",
    ) -> float:
        """Compute unbiased MMD^2 estimate with configurable kernel.

        Parameters
        ----------
        x, y:
            Two samples, shape (n, d) and (m, d).
        kernel:
            Kernel type: 'rbf' (Gaussian, median bandwidth) or 'linear'.

        Returns
        -------
        Unbiased MMD^2 estimate.
        """
        from sklearn.metrics.pairwise import linear_kernel, rbf_kernel

        if kernel == "rbf":
            # Median heuristic for bandwidth
            combined = np.vstack([x, y])
            dists = np.linalg.norm(combined[:, None] - combined[None, :], axis=-1)
            median_dist = float(np.median(dists[dists > 0]))
            gamma = 1.0 / (2.0 * max(median_dist, 1e-10) ** 2)
            k_xx = rbf_kernel(x, x, gamma=gamma)
            k_yy = rbf_kernel(y, y, gamma=gamma)
            k_xy = rbf_kernel(x, y, gamma=gamma)
        elif kernel == "linear":
            k_xx = linear_kernel(x, x)
            k_yy = linear_kernel(y, y)
            k_xy = linear_kernel(x, y)
        else:
            msg = f"Unsupported kernel: {kernel!r}"
            raise ValueError(msg)

        n = len(x)
        m = len(y)

        # Unbiased MMD^2 estimate
        np.fill_diagonal(k_xx, 0)
        np.fill_diagonal(k_yy, 0)

        mmd2 = (
            k_xx.sum() / max(n * (n - 1), 1)
            + k_yy.sum() / max(m * (m - 1), 1)
            - 2.0 * k_xy.sum() / max(n * m, 1)
        )
        return float(mmd2)


def persist_drift_reports(
    *,
    reference_features: pd.DataFrame,
    current_features: pd.DataFrame,
    reference_embeddings: NDArray[np.float32] | None = None,
    current_embeddings: NDArray[np.float32] | None = None,
    tmp_dir: Path,
    threshold: float = 0.05,
    drift_share: float = 0.5,
) -> DriftResult:
    """Run drift detection and persist reports as MLflow artifacts.

    Saves under ``drift_reports/`` artifact path:
    - ``tier1_evidently_report.html`` — Evidently DataDriftPreset HTML
    - ``tier1_drift_summary.json`` — KS test results per feature
    - ``tier2_mmd_summary.json`` — MMD test results (if embeddings provided)
    - ``drift_metadata.json`` — timestamps, thresholds, library versions

    Must be called inside an active ``mlflow.start_run()`` context.

    Parameters
    ----------
    reference_features:
        Reference feature DataFrame.
    current_features:
        Current feature DataFrame.
    reference_embeddings:
        Optional reference embeddings for Tier 2 MMD test.
    current_embeddings:
        Optional current embeddings for Tier 2 MMD test.
    tmp_dir:
        Temporary directory for writing artifacts before logging.
    threshold:
        P-value threshold for per-feature drift.
    drift_share:
        Fraction of features that must drift for dataset drift.

    Returns
    -------
    DriftResult from Tier 1 detection.
    """
    import mlflow

    tmp_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = "drift_reports"

    # --- Tier 1: Feature drift ---
    detector = FeatureDriftDetector(
        reference_features, threshold=threshold, drift_share=drift_share
    )
    html_path = tmp_dir / "tier1_evidently_report.html"
    tier1_result = detector.detect(current_features, evidently_html_path=html_path)

    # Save HTML report
    mlflow.log_artifact(str(html_path), artifact_path=artifact_path)

    # Save JSON summary
    tier1_json_path = tmp_dir / "tier1_drift_summary.json"
    tier1_summary = {
        "drift_detected": tier1_result.drift_detected,
        "dataset_drift_score": tier1_result.dataset_drift_score,
        "feature_scores": tier1_result.feature_scores,
        "drifted_features": tier1_result.drifted_features,
        "n_features": tier1_result.n_features,
        "n_drifted": tier1_result.n_drifted,
        "timestamp": tier1_result.timestamp.isoformat(),
    }
    tier1_json_path.write_text(json.dumps(tier1_summary, indent=2), encoding="utf-8")
    mlflow.log_artifact(str(tier1_json_path), artifact_path=artifact_path)

    # --- Tier 2: Embedding drift (optional) ---
    if reference_embeddings is not None and current_embeddings is not None:
        emb_detector = EmbeddingDriftDetector(
            reference_embeddings, p_val_threshold=threshold
        )
        tier2_result = emb_detector.detect(current_embeddings)
        tier2_json_path = tmp_dir / "tier2_mmd_summary.json"
        tier2_summary = {
            "drift_detected": tier2_result.drift_detected,
            "p_value": tier2_result.dataset_drift_score,
            "mmd_statistic": tier2_result.feature_scores.get("mmd_statistic"),
            "timestamp": tier2_result.timestamp.isoformat(),
        }
        tier2_json_path.write_text(
            json.dumps(tier2_summary, indent=2), encoding="utf-8"
        )
        mlflow.log_artifact(str(tier2_json_path), artifact_path=artifact_path)

    # --- Metadata ---
    import evidently

    metadata_path = tmp_dir / "drift_metadata.json"
    metadata = {
        "timestamp": datetime.now(UTC).isoformat(),
        "evidently_version": evidently.__version__,
        "threshold": threshold,
        "drift_share": drift_share,
        "n_reference_samples": len(reference_features),
        "n_current_samples": len(current_features),
        "has_tier2": reference_embeddings is not None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    mlflow.log_artifact(str(metadata_path), artifact_path=artifact_path)

    logger.info("Drift reports persisted to MLflow under %s/", artifact_path)
    return tier1_result
