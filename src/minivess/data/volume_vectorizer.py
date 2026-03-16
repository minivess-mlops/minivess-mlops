"""3D volume vectorization to 1D feature vectors for tabular drift tools.

Enables tools like NannyML, Deepchecks tabular, and other non-image drift
detectors to operate on 3D volumetric data by flattening to feature vectors.

Multiple strategies:
    - statistical: 9+ hand-crafted features (mean, std, SNR, entropy, etc.)
    - histogram: Normalized intensity histogram (configurable bins)
    - fft: Radially-averaged spatial frequency spectrum
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Strategy functions — each takes a 3-D volume and returns a 1-D vector
# ---------------------------------------------------------------------------


def _statistical_vector(
    volume: np.ndarray,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Extract statistical features (mirrors feature_extraction.py)."""
    flat = volume.ravel().astype(np.float64)
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    p5, p95 = float(np.percentile(flat, 5)), float(np.percentile(flat, 95))

    foreground = flat[flat > (mean + 0.5 * std)]
    snr = (
        float(np.mean(foreground) / (np.std(foreground) + 1e-8))
        if len(foreground) > 0
        else 0.0
    )

    # Histogram entropy
    hist, _ = np.histogram(flat, bins=64, density=True)
    hist = hist[hist > 0]
    entropy = float(scipy_stats.entropy(hist))

    features = np.array(
        [
            mean,
            std,
            float(np.min(flat)),
            float(np.max(flat)),
            p5,
            p95,
            snr,
            p95 - p5,  # contrast
            entropy,
        ],
        dtype=np.float64,
    )
    return features


_STATISTICAL_FEATURE_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "p5",
    "p95",
    "snr",
    "contrast",
    "entropy",
]


def _histogram_vector(
    volume: np.ndarray,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Normalized intensity histogram."""
    cfg = config or {}
    n_bins: int = cfg.get("n_bins", 64)
    hist, _ = np.histogram(volume.ravel(), bins=n_bins, density=True)
    # Normalize to sum=1 (probability distribution)
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist.astype(np.float64)


def _fft_vector(
    volume: np.ndarray,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Radially-averaged 3D FFT power spectrum.

    Produces a 1-D vector where each element is the average power
    at a given spatial frequency (radius in Fourier space).
    """
    cfg = config or {}
    n_bins: int = cfg.get("n_bins", 32)

    # 3D FFT → power spectrum
    fft = np.fft.fftn(volume.astype(np.float64))
    fft_shift = np.fft.fftshift(fft)
    power = np.abs(fft_shift) ** 2

    # Radial coordinates
    d, h, w = volume.shape
    center = np.array([d // 2, h // 2, w // 2])
    z_idx, y_idx, x_idx = np.mgrid[0:d, 0:h, 0:w]
    radius = np.sqrt(
        (z_idx - center[0]) ** 2 + (y_idx - center[1]) ** 2 + (x_idx - center[2]) ** 2
    )

    # Radial binning
    max_radius = min(d, h, w) // 2
    bin_edges = np.linspace(0, max_radius, n_bins + 1)
    spectrum = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_bins):
        mask = (radius >= bin_edges[i]) & (radius < bin_edges[i + 1])
        if mask.any():
            spectrum[i] = float(np.mean(power[mask]))

    # Normalize
    total = spectrum.sum()
    if total > 0:
        spectrum = spectrum / total

    return spectrum


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

_VectorFn = Callable[[np.ndarray, dict[str, Any] | None], np.ndarray]

VECTORIZATION_STRATEGIES: dict[str, _VectorFn] = {
    "statistical": _statistical_vector,
    "histogram": _histogram_vector,
    "fft": _fft_vector,
}


def vectorize_volume(
    volume: np.ndarray,
    strategy: str = "statistical",
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Vectorize a single 3D volume to a 1D feature vector.

    Args:
        volume: 3-D ``np.ndarray`` of shape ``(D, H, W)``.
        strategy: Vectorization strategy name.
        config: Optional strategy-specific configuration.

    Returns:
        1-D ``np.ndarray`` feature vector.

    Raises:
        KeyError: If *strategy* is not registered.
    """
    if strategy not in VECTORIZATION_STRATEGIES:
        available = ", ".join(sorted(VECTORIZATION_STRATEGIES))
        msg = f"Unknown vectorization strategy '{strategy}'. Available: {available}"
        raise KeyError(msg)
    return VECTORIZATION_STRATEGIES[strategy](volume, config)


def vectorize_batch(
    volumes: list[np.ndarray],
    strategy: str = "statistical",
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Vectorize a batch of 3D volumes to a 2D feature matrix.

    Args:
        volumes: List of 3-D ``np.ndarray`` volumes.
        strategy: Vectorization strategy name.
        config: Optional strategy-specific configuration.

    Returns:
        2-D ``np.ndarray`` of shape ``(n_volumes, n_features)``.
    """
    vectors = [vectorize_volume(v, strategy=strategy, config=config) for v in volumes]
    return np.stack(vectors, axis=0)


def vectorize_batch_to_dataframe(
    volumes: list[np.ndarray],
    strategy: str = "statistical",
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Vectorize a batch and return as a named DataFrame.

    Args:
        volumes: List of 3-D ``np.ndarray`` volumes.
        strategy: Vectorization strategy name.
        config: Optional strategy-specific configuration.

    Returns:
        ``pd.DataFrame`` with named columns (for statistical strategy)
        or generic ``f_{i}`` column names (for other strategies).
    """
    matrix = vectorize_batch(volumes, strategy=strategy, config=config)
    if strategy == "statistical":
        columns = _STATISTICAL_FEATURE_NAMES[: matrix.shape[1]]
    else:
        columns = [f"f_{i}" for i in range(matrix.shape[1])]
    return pd.DataFrame(matrix, columns=columns)
