"""SAM3 offline feature caching for 8GB VRAM workflow.

Extracts SAM3 ViT-32L features offline and caches them to disk as .pt
files. During training, cached features are loaded instead of running
the 648M-param encoder, enabling training on 8GB GPUs.

Disk budget: ~200MB per volume × 70 volumes ≈ 14GB SSD.

Usage::

    # Offline extraction (can use CPU or a larger GPU)
    extract_and_cache_features(config, volumes, cache_dir)

    # Training-time loading
    dataset = Sam3CachedFeatureDataset(cache_dir)
    features = load_cached_volume_features("vol_001", cache_dir)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.utils.data import Dataset

from minivess.adapters.sam3_backbone import Sam3Backbone

if TYPE_CHECKING:
    from pathlib import Path

    from minivess.config.models import ModelConfig

logger = logging.getLogger(__name__)


def extract_and_cache_features(
    config: ModelConfig,
    volumes: dict[str, Tensor],
    cache_dir: Path,
    *,
    use_stub: bool = False,
) -> None:
    """Extract SAM3 features for all volumes and cache to disk.

    Parameters
    ----------
    config:
        Model configuration.
    volumes:
        Mapping of volume_id → tensor of shape (B, C, D, H, W).
    cache_dir:
        Directory for cached .pt files.
    use_stub:
        If True, use stub encoder (for testing).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    backbone = Sam3Backbone(config=config, use_stub=use_stub, freeze=True)
    backbone.eval()

    for vol_id, volume in volumes.items():
        cache_path = cache_dir / f"{vol_id}.pt"
        if cache_path.exists():
            logger.info("Skipping %s — already cached", vol_id)
            continue

        logger.info("Extracting features for %s (shape=%s)", vol_id, volume.shape)
        with torch.no_grad():
            features = backbone.get_volume_embeddings(volume)

        torch.save(features, cache_path)
        size_mb = cache_path.stat().st_size / 1e6
        logger.info("Cached %s → %s (%.1f MB)", vol_id, cache_path, size_mb)

    logger.info(
        "Feature caching complete: %d volumes → %s",
        len(volumes),
        cache_dir,
    )


def load_cached_volume_features(volume_id: str, cache_dir: Path) -> Tensor:
    """Load cached features for a single volume.

    Parameters
    ----------
    volume_id:
        Volume identifier (filename stem without .pt).
    cache_dir:
        Directory containing cached .pt files.

    Returns
    -------
    Feature tensor of shape (B, embed_dim, D, H_feat, W_feat).

    Raises
    ------
    FileNotFoundError
        If cached file does not exist.
    """
    cache_path = cache_dir / f"{volume_id}.pt"
    if not cache_path.exists():
        msg = f"No cached features for {volume_id} at {cache_path}"
        raise FileNotFoundError(msg)

    result: Tensor = torch.load(cache_path, weights_only=True)
    return result


class Sam3CachedFeatureDataset(Dataset[tuple[str, Tensor]]):
    """PyTorch Dataset wrapping cached SAM3 features.

    Each item is a (volume_id, features) tuple where features has shape
    (B, embed_dim, D, H_feat, W_feat).

    Parameters
    ----------
    cache_dir:
        Directory containing .pt feature files.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._volume_ids = sorted(p.stem for p in cache_dir.glob("*.pt"))
        if not self._volume_ids:
            logger.warning("No cached features found in %s", cache_dir)

    def __len__(self) -> int:
        return len(self._volume_ids)

    def __getitem__(self, idx: int) -> tuple[str, Tensor]:
        vol_id = self._volume_ids[idx]
        features = load_cached_volume_features(vol_id, self._cache_dir)
        return vol_id, features

    @property
    def volume_ids(self) -> list[str]:
        """List of cached volume identifiers."""
        return list(self._volume_ids)
