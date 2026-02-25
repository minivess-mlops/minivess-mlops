"""Patch validation guards for pre-training sanity checks.

Provides fast-fail validators that catch patch/dataset/memory mismatches
BEFORE training starts, rather than 30 minutes into epoch 5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minivess.data.profiler import DatasetProfile


class PatchValidationError(ValueError):
    """Raised when patch size is incompatible with dataset or model."""


class MemoryBudgetError(ValueError):
    """Raised when memory budget is exceeded."""


# VRAM overhead per model family in megabytes (model weights + framework overhead)
_MODEL_VRAM_OVERHEAD_MB: dict[str, int] = {
    "dynunet": 300,
    "segresnet": 200,
    "swinunetr": 500,
    "vista3d": 600,
    "vesselfm": 400,
    "comma_mamba": 400,
    "sam3_lora": 400,
}
_DEFAULT_MODEL_VRAM_OVERHEAD_MB = 400

# Activation memory bytes per voxel for 3D UNets in AMP mode.
# This accounts for the full forward + backward activation memory across all
# encoder/decoder scales: ~32 feature channels * 4 levels * 8 bytes (bf16 grad)
# Empirically calibrated to flag batch=8, patch=(128,128,64) on an 8 GB GPU.
_AMP_BYTES_PER_VOXEL = 1000


def validate_patch_fits_dataset(
    patch_size: tuple[int, int, int],
    profile: DatasetProfile,
) -> None:
    """Check patch_size <= min_shape for all dimensions.

    Parameters
    ----------
    patch_size:
        3D patch size as (x, y, z).
    profile:
        DatasetProfile with aggregated dataset statistics.

    Raises
    ------
    PatchValidationError
        If any patch dimension exceeds the corresponding minimum volume dimension.
    """
    dim_names = ("x", "y", "z")
    for i, (patch_dim, min_dim) in enumerate(
        zip(patch_size, profile.min_shape, strict=False)
    ):
        if patch_dim > min_dim:
            msg = (
                f"Patch {dim_names[i]}={patch_dim} exceeds smallest volume "
                f"{dim_names[i]}={min_dim}. Use SpatialPadd or reduce patch size."
            )
            raise PatchValidationError(msg)


def validate_patch_divisibility(
    patch_size: tuple[int, int, int],
    model_divisor: int,
) -> None:
    """Check all patch dimensions are divisible by model_divisor.

    Parameters
    ----------
    patch_size:
        3D patch size as (x, y, z).
    model_divisor:
        Required divisor for the model architecture (e.g. 8 for DynUNet
        with 4 pooling levels, 16 for SwinUNETR).

    Raises
    ------
    PatchValidationError
        If any dimension is not divisible by model_divisor, with the
        offending dimension name and value in the error message.
    """
    dim_names = ("x", "y", "z")
    offending: list[str] = []
    for i, dim_val in enumerate(patch_size):
        if dim_val % model_divisor != 0:
            offending.append(f"{dim_names[i]}={dim_val}")

    if offending:
        dims_str = ", ".join(offending)
        msg = (
            f"Patch dimensions [{dims_str}] are not divisible by {model_divisor}. "
            f"All patch dimensions must be divisible by {model_divisor} for this model."
        )
        raise PatchValidationError(msg)


def validate_cache_fits_ram(
    profile: DatasetProfile,
    cache_rate: float,
    available_ram_mb: int,
    max_fraction: float = 0.7,
) -> None:
    """Check that cached dataset fits in available RAM.

    Parameters
    ----------
    profile:
        DatasetProfile containing total_size_bytes for the dataset.
    cache_rate:
        Fraction of the dataset to cache in RAM (0.0 to 1.0).
    available_ram_mb:
        Total available RAM in megabytes.
    max_fraction:
        Maximum fraction of available RAM the cache may consume (default 0.7).

    Raises
    ------
    MemoryBudgetError
        If the estimated cache size exceeds max_fraction of available RAM,
        with actionable guidance in the error message.
    """
    cached_mb = (profile.total_size_bytes * cache_rate) / (1024 * 1024)
    budget_mb = available_ram_mb * max_fraction

    if cached_mb > budget_mb:
        suggested_rate = budget_mb / (profile.total_size_bytes / (1024 * 1024))
        suggested_rate = max(0.0, min(1.0, suggested_rate))
        msg = (
            f"cache size estimate {cached_mb:.0f} MB (cache_rate={cache_rate}, "
            f"dataset={profile.total_size_bytes / (1024**3):.1f} GB) exceeds "
            f"{max_fraction * 100:.0f}% of available RAM ({available_ram_mb} MB). "
            f"Reduce cache_rate to ~{suggested_rate:.2f} or increase available RAM."
        )
        raise MemoryBudgetError(msg)


def validate_vram_budget(
    batch_size: int,
    patch_size: tuple[int, int, int],
    gpu_vram_mb: int,
    model_name: str = "dynunet",
) -> None:
    """Estimate VRAM usage and check against available GPU budget.

    Uses a model-specific heuristic:
    - Model overhead: fixed per model family (weights + framework)
    - Activation memory (AMP): batch_size * patch_volume * 8 bytes
      (float16 * 4 intermediate activation tensors)
    - Budget: 80% of gpu_vram_mb

    Parameters
    ----------
    batch_size:
        Training batch size.
    patch_size:
        3D patch size as (x, y, z).
    gpu_vram_mb:
        Total GPU VRAM in megabytes.
    model_name:
        Model architecture name for overhead lookup (default "dynunet").

    Raises
    ------
    MemoryBudgetError
        If estimated VRAM exceeds 80% of gpu_vram_mb, with the estimate
        and budget in the error message.
    """
    model_key = model_name.lower()
    overhead_mb = _MODEL_VRAM_OVERHEAD_MB.get(
        model_key, _DEFAULT_MODEL_VRAM_OVERHEAD_MB
    )

    patch_x, patch_y, patch_z = patch_size
    voxels = batch_size * patch_x * patch_y * patch_z
    activation_bytes = voxels * _AMP_BYTES_PER_VOXEL
    activation_mb = activation_bytes / (1024 * 1024)

    total_estimated_mb = overhead_mb + activation_mb
    budget_mb = gpu_vram_mb * 0.8

    if total_estimated_mb > budget_mb:
        msg = (
            f"VRAM estimate {total_estimated_mb:.0f} MB "
            f"(model_overhead={overhead_mb} MB + activations={activation_mb:.0f} MB) "
            f"exceeds 80% VRAM budget ({budget_mb:.0f} MB of {gpu_vram_mb} MB). "
            f"Reduce batch_size (current={batch_size}) or patch_size {patch_size}."
        )
        raise MemoryBudgetError(msg)


def validate_no_default_resampling(
    voxel_spacing: tuple[float, float, float],
    strict: bool = True,
) -> str | None:
    """Warn or error if non-zero voxel spacing (resampling) is configured.

    A voxel_spacing of (0, 0, 0) means native spacing — no resampling.
    Any non-zero spacing triggers this validator because resampling can
    silently change patch-to-physical-space relationships.

    Parameters
    ----------
    voxel_spacing:
        Target voxel spacing in mm. (0.0, 0.0, 0.0) means native/no resampling.
    strict:
        If True, raise PatchValidationError for non-zero spacing.
        If False, return a warning string instead of raising.

    Returns
    -------
    str | None
        None when spacing is (0, 0, 0) or strict=True and an error is raised.
        A warning string when strict=False and spacing is non-zero.

    Raises
    ------
    PatchValidationError
        If strict=True and any voxel_spacing component is non-zero.
    """
    is_resampling = any(s != 0.0 for s in voxel_spacing)
    if not is_resampling:
        return None

    warning_msg = (
        f"resampling is enabled: voxel_spacing={voxel_spacing}. "
        f"This changes physical patch size. Verify patch_size is still appropriate "
        f"after resampling. Use voxel_spacing=(0.0, 0.0, 0.0) for native spacing."
    )
    if strict:
        raise PatchValidationError(warning_msg)
    return warning_msg
