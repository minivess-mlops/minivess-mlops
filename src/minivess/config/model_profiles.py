from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Default search directory for built-in profiles
_DEFAULT_PROFILE_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "configs" / "model_profiles"
)


@dataclass(frozen=True)
class ModelProfile:
    """Per-model memory and compute profile.

    Parameters
    ----------
    name:
        Model identifier (matches YAML filename stem).
    divisor:
        Spatial divisibility requirement (e.g., 8 for DynUNet, 16 for VISTA-3D).
    model_overhead_mb:
        Base model + optimizer memory in megabytes.
    bytes_per_voxel_amp:
        Per-voxel activation memory in bytes when using AMP.
    bytes_per_voxel_fp32:
        Per-voxel activation memory in bytes for full FP32.
    max_batch_size:
        Mapping of GPU tier to maximum batch size (e.g., ``{"gpu_low": 2}``).
    default_patch_xy:
        Default XY patch dimension in voxels.
    notes:
        Free-text empirical notes about the profile.
    """

    name: str
    divisor: int
    model_overhead_mb: int
    bytes_per_voxel_amp: int
    bytes_per_voxel_fp32: int
    max_batch_size: dict[str, int]
    default_patch_xy: int
    notes: str = ""


def load_model_profile(
    name: str, search_dirs: list[Path] | None = None
) -> ModelProfile:
    """Load a model profile from a YAML file.

    Searches each directory in ``search_dirs`` (or the built-in
    ``configs/model_profiles/`` directory) for a file named
    ``{name}.yaml``.

    Parameters
    ----------
    name:
        Model profile name (e.g., ``"dynunet"``, ``"vista3d"``).
    search_dirs:
        Optional list of directories to search before the default location.

    Returns
    -------
    ModelProfile
        Populated dataclass loaded from YAML.

    Raises
    ------
    FileNotFoundError
        If no matching YAML file is found in any search directory.
    """
    dirs = search_dirs if search_dirs is not None else [_DEFAULT_PROFILE_DIR]
    for d in dirs:
        yaml_path = Path(d) / f"{name}.yaml"
        if yaml_path.exists():
            logger.debug("Loading model profile %r from %s", name, yaml_path)
            with open(yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return ModelProfile(**data)
    raise FileNotFoundError(f"No profile found for model '{name}' in {dirs}")


def list_available_profiles(search_dirs: list[Path] | None = None) -> list[str]:
    """Return sorted list of available model profile names.

    Parameters
    ----------
    search_dirs:
        Optional list of directories to search. Defaults to the built-in
        ``configs/model_profiles/`` directory.

    Returns
    -------
    list[str]
        Sorted list of profile names (YAML filename stems).
    """
    dirs = search_dirs if search_dirs is not None else [_DEFAULT_PROFILE_DIR]
    names: set[str] = set()
    for d in dirs:
        resolved = Path(d)
        if resolved.exists():
            for p in resolved.glob("*.yaml"):
                names.add(p.stem)
    return sorted(names)


def estimate_vram_mb(
    profile: ModelProfile,
    batch_size: int,
    patch_size: tuple[int, int, int],
    mixed_precision: bool = True,
) -> float:
    """Estimate peak VRAM usage in megabytes.

    Uses a simple activation-volume model:
    ``total = model_overhead_mb + (voxels * bytes_per_voxel) / 2^20``

    Parameters
    ----------
    profile:
        Model profile containing memory coefficients.
    batch_size:
        Number of samples per forward pass.
    patch_size:
        3D patch dimensions ``(D, H, W)`` in voxels.
    mixed_precision:
        If ``True``, use AMP byte-per-voxel coefficient; otherwise FP32.

    Returns
    -------
    float
        Estimated VRAM in megabytes.
    """
    voxels = batch_size * patch_size[0] * patch_size[1] * patch_size[2]
    bpv = (
        profile.bytes_per_voxel_amp if mixed_precision else profile.bytes_per_voxel_fp32
    )
    activation_mb = (voxels * bpv) / (1024 * 1024)
    return profile.model_overhead_mb + activation_mb
