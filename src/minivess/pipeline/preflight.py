"""Preflight environment validation system.

Automated one-time pre-training checks that validate the environment
before an experiment starts. Replaces manual checks for GPU, RAM,
disk space, swap usage, and data availability.

This is intentionally distinct from ``scripts/system_monitor.py``,
which performs ongoing monitoring. Preflight runs once and gates
the experiment launch.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class CheckStatus(StrEnum):
    """Status of a single preflight check."""

    PASS = "pass"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PreflightCheck:
    """Result of a single preflight check.

    Parameters
    ----------
    name:
        Short identifier for the check (e.g. ``"gpu"``, ``"disk_space"``).
    status:
        Outcome of the check.
    message:
        Human-readable summary of the check result.
    details:
        Optional structured metadata (e.g. measured values, thresholds).
    """

    name: str
    status: CheckStatus
    message: str
    details: dict[str, object] | None = field(default=None)


@dataclass
class PreflightResult:
    """Aggregated result of all preflight checks.

    Parameters
    ----------
    checks:
        List of individual check results.
    environment:
        Detected environment string (``"local"``, ``"docker"``,
        ``"cloud"``, or ``"ci"``).
    """

    checks: list[PreflightCheck]
    environment: str

    @property
    def passed(self) -> bool:
        """Return ``True`` if no check has CRITICAL status."""
        return not any(c.status == CheckStatus.CRITICAL for c in self.checks)


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------


def check_gpu() -> PreflightCheck:
    """Detect NVIDIA GPU availability via ``nvidia-smi``.

    CPU-only machines are valid training environments, so a missing
    GPU is a WARNING, not a CRITICAL failure.

    Returns
    -------
    PreflightCheck
        Check result with GPU name and VRAM details when available.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return PreflightCheck(
            name="gpu",
            status=CheckStatus.WARNING,
            message="nvidia-smi not found; GPU unavailable — CPU training only.",
            details={"nvidia_smi_found": False},
        )

    try:
        proc = subprocess.run(  # noqa: S603
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return PreflightCheck(
            name="gpu",
            status=CheckStatus.WARNING,
            message=f"nvidia-smi query failed: {exc}",
            details={"nvidia_smi_found": True, "error": str(exc)},
        )

    if proc.returncode != 0:
        return PreflightCheck(
            name="gpu",
            status=CheckStatus.WARNING,
            message=f"nvidia-smi returned non-zero exit code {proc.returncode}.",
            details={"returncode": proc.returncode, "stderr": proc.stderr.strip()},
        )

    gpus: list[dict[str, object]] = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:  # noqa: PLR2004
            gpus.append(
                {
                    "name": parts[0],
                    "memory_total_mb": float(parts[1])
                    if parts[1].isdigit()
                    else parts[1],
                    "memory_free_mb": float(parts[2])
                    if parts[2].isdigit()
                    else parts[2],
                }
            )

    if not gpus:
        return PreflightCheck(
            name="gpu",
            status=CheckStatus.WARNING,
            message="nvidia-smi returned no GPU entries.",
            details={"nvidia_smi_found": True},
        )

    names = ", ".join(str(g["name"]) for g in gpus)
    return PreflightCheck(
        name="gpu",
        status=CheckStatus.PASS,
        message=f"Found {len(gpus)} GPU(s): {names}.",
        details={"gpus": gpus},
    )


def check_ram(min_gb: float = 16.0) -> PreflightCheck:
    """Check available system RAM against a minimum threshold.

    Reads ``/proc/meminfo`` on Linux. On non-Linux systems the check
    degrades to a WARNING with an informative message.

    Parameters
    ----------
    min_gb:
        Minimum acceptable available RAM in gibibytes.

    Returns
    -------
    PreflightCheck
    """
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.exists():
        return PreflightCheck(
            name="ram",
            status=CheckStatus.WARNING,
            message="/proc/meminfo not available; RAM check skipped.",
            details={"platform": os.name},
        )

    mem_available_kb: int | None = None
    mem_total_kb: int | None = None

    with meminfo_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                mem_available_kb = int(line.split()[1])
            elif line.startswith("MemTotal:"):
                mem_total_kb = int(line.split()[1])
            if mem_available_kb is not None and mem_total_kb is not None:
                break

    if mem_available_kb is None:
        return PreflightCheck(
            name="ram",
            status=CheckStatus.WARNING,
            message="Could not parse MemAvailable from /proc/meminfo.",
        )

    available_gb = mem_available_kb / (1024 * 1024)
    total_gb = (mem_total_kb / (1024 * 1024)) if mem_total_kb is not None else None

    details: dict[str, object] = {
        "available_gb": round(available_gb, 2),
        "min_required_gb": min_gb,
    }
    if total_gb is not None:
        details["total_gb"] = round(total_gb, 2)

    if available_gb < min_gb:
        return PreflightCheck(
            name="ram",
            status=CheckStatus.WARNING,
            message=(
                f"Available RAM {available_gb:.1f} GiB < recommended {min_gb:.1f} GiB."
            ),
            details=details,
        )

    return PreflightCheck(
        name="ram",
        status=CheckStatus.PASS,
        message=f"Available RAM {available_gb:.1f} GiB >= {min_gb:.1f} GiB.",
        details=details,
    )


def check_disk_space(path: Path, min_gb: float = 20.0) -> PreflightCheck:
    """Check free disk space at a given path.

    Parameters
    ----------
    path:
        Filesystem path to check (uses the mount-point of this path).
    min_gb:
        Minimum acceptable free disk space in gibibytes.

    Returns
    -------
    PreflightCheck
    """
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        return PreflightCheck(
            name="disk_space",
            status=CheckStatus.CRITICAL,
            message=f"Cannot query disk usage for {path}: {exc}",
            details={"path": str(path), "error": str(exc)},
        )

    free_gb = usage.free / (1024**3)
    total_gb = usage.total / (1024**3)

    details: dict[str, object] = {
        "path": str(path),
        "free_gb": round(free_gb, 2),
        "total_gb": round(total_gb, 2),
        "min_required_gb": min_gb,
    }

    if free_gb < min_gb:
        return PreflightCheck(
            name="disk_space",
            status=CheckStatus.WARNING,
            message=(
                f"Free disk space {free_gb:.1f} GiB < "
                f"recommended {min_gb:.1f} GiB at {path}."
            ),
            details=details,
        )

    return PreflightCheck(
        name="disk_space",
        status=CheckStatus.PASS,
        message=f"Free disk space {free_gb:.1f} GiB >= {min_gb:.1f} GiB at {path}.",
        details=details,
    )


def check_swap(warn_gb: float = 5.0) -> PreflightCheck:
    """Check swap usage against a warning threshold.

    High swap usage indicates memory pressure; it is a WARNING because
    training can still proceed (slowly), but it suggests the system
    is under memory stress.

    Parameters
    ----------
    warn_gb:
        Warn if swap usage exceeds this many gibibytes.

    Returns
    -------
    PreflightCheck
    """
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.exists():
        return PreflightCheck(
            name="swap",
            status=CheckStatus.WARNING,
            message="/proc/meminfo not available; swap check skipped.",
            details={"platform": os.name},
        )

    swap_total_kb: int | None = None
    swap_free_kb: int | None = None

    with meminfo_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("SwapTotal:"):
                swap_total_kb = int(line.split()[1])
            elif line.startswith("SwapFree:"):
                swap_free_kb = int(line.split()[1])
            if swap_total_kb is not None and swap_free_kb is not None:
                break

    if swap_total_kb is None or swap_free_kb is None:
        return PreflightCheck(
            name="swap",
            status=CheckStatus.WARNING,
            message="Could not parse SwapTotal/SwapFree from /proc/meminfo.",
        )

    swap_used_kb = swap_total_kb - swap_free_kb
    swap_used_gb = swap_used_kb / (1024 * 1024)
    swap_total_gb = swap_total_kb / (1024 * 1024)

    details: dict[str, object] = {
        "swap_used_gb": round(swap_used_gb, 2),
        "swap_total_gb": round(swap_total_gb, 2),
        "warn_threshold_gb": warn_gb,
    }

    if swap_used_gb >= warn_gb:
        return PreflightCheck(
            name="swap",
            status=CheckStatus.WARNING,
            message=(
                f"Swap usage {swap_used_gb:.1f} GiB >= "
                f"warning threshold {warn_gb:.1f} GiB. "
                "System may be under memory pressure."
            ),
            details=details,
        )

    return PreflightCheck(
        name="swap",
        status=CheckStatus.PASS,
        message=f"Swap usage {swap_used_gb:.1f} GiB < {warn_gb:.1f} GiB.",
        details=details,
    )


def check_data_exists(data_dir: Path) -> PreflightCheck:
    """Verify that the data directory exists and contains NIfTI files.

    Looks for ``imagesTr/`` or ``raw/`` subdirectory with at least one
    ``.nii.gz`` file, plus a corresponding ``labelsTr/`` directory
    when ``imagesTr/`` is used.

    Parameters
    ----------
    data_dir:
        Root data directory to inspect.

    Returns
    -------
    PreflightCheck
    """
    if not data_dir.exists():
        return PreflightCheck(
            name="data_exists",
            status=CheckStatus.CRITICAL,
            message=f"Data directory does not exist: {data_dir}",
            details={"data_dir": str(data_dir)},
        )

    if not data_dir.is_dir():
        return PreflightCheck(
            name="data_exists",
            status=CheckStatus.CRITICAL,
            message=f"Data path is not a directory: {data_dir}",
            details={"data_dir": str(data_dir)},
        )

    # Check for standard Medical Decathlon / MONAI layout (imagesTr + labelsTr)
    images_dir = data_dir / "imagesTr"
    labels_dir = data_dir / "labelsTr"
    raw_dir = data_dir / "raw"

    if images_dir.exists():
        nifti_images = list(images_dir.glob("*.nii.gz"))
        if not nifti_images:
            return PreflightCheck(
                name="data_exists",
                status=CheckStatus.CRITICAL,
                message=f"imagesTr/ exists but contains no .nii.gz files: {images_dir}",
                details={"data_dir": str(data_dir), "images_dir": str(images_dir)},
            )

        details: dict[str, object] = {
            "data_dir": str(data_dir),
            "layout": "decathlon",
            "num_images": len(nifti_images),
        }

        if labels_dir.exists():
            nifti_labels = list(labels_dir.glob("*.nii.gz"))
            details["num_labels"] = len(nifti_labels)
            if not nifti_labels:
                return PreflightCheck(
                    name="data_exists",
                    status=CheckStatus.WARNING,
                    message=(
                        f"labelsTr/ exists but contains no .nii.gz files: {labels_dir}. "
                        "Training without labels is not possible."
                    ),
                    details=details,
                )
        else:
            details["labels_dir_missing"] = True

        return PreflightCheck(
            name="data_exists",
            status=CheckStatus.PASS,
            message=(f"Found {len(nifti_images)} NIfTI image(s) in {images_dir}."),
            details=details,
        )

    if raw_dir.exists():
        nifti_raw = list(raw_dir.glob("**/*.nii.gz"))
        if nifti_raw:
            return PreflightCheck(
                name="data_exists",
                status=CheckStatus.PASS,
                message=f"Found {len(nifti_raw)} NIfTI file(s) in raw/ layout.",
                details={
                    "data_dir": str(data_dir),
                    "layout": "raw",
                    "num_files": len(nifti_raw),
                },
            )

    # Fall back: scan top-level for any NIfTI
    any_nifti = list(data_dir.glob("*.nii.gz"))
    if any_nifti:
        return PreflightCheck(
            name="data_exists",
            status=CheckStatus.PASS,
            message=f"Found {len(any_nifti)} NIfTI file(s) in {data_dir} (flat layout).",
            details={
                "data_dir": str(data_dir),
                "layout": "flat",
                "num_files": len(any_nifti),
            },
        )

    return PreflightCheck(
        name="data_exists",
        status=CheckStatus.CRITICAL,
        message=(
            f"No NIfTI files found in {data_dir}. "
            "Expected imagesTr/, raw/, or flat .nii.gz files."
        ),
        details={"data_dir": str(data_dir)},
    )


def detect_environment() -> str:
    """Detect the current execution environment.

    Checks, in order of priority:

    1. ``CI`` environment variable → ``"ci"``
    2. ``GITHUB_ACTIONS`` environment variable → ``"ci"``
    3. ``/.dockerenv`` file existence → ``"docker"``
    4. Cloud metadata endpoint hints (``AWS_EXECUTION_ENV``,
       ``GOOGLE_CLOUD_PROJECT``, ``AZURE_CLIENT_ID``) → ``"cloud"``
    5. Default → ``"local"``

    Returns
    -------
    str
        One of ``"local"``, ``"docker"``, ``"cloud"``, ``"ci"``.
    """
    # CI detection first (highest priority)
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        return "ci"

    # Docker detection via /.dockerenv sentinel file
    if Path("/.dockerenv").exists():
        return "docker"

    # Cloud provider detection via environment variables
    cloud_indicators = (
        "AWS_EXECUTION_ENV",
        "GOOGLE_CLOUD_PROJECT",
        "GCP_PROJECT",
        "AZURE_CLIENT_ID",
        "AZURE_SUBSCRIPTION_ID",
    )
    if any(os.environ.get(var) for var in cloud_indicators):
        return "cloud"

    return "local"


def run_preflight(
    data_dir: Path,
    *,
    min_ram_gb: float = 16.0,
    min_disk_gb: float = 20.0,
    warn_swap_gb: float = 5.0,
) -> PreflightResult:
    """Run all preflight checks and return the aggregated result.

    Parameters
    ----------
    data_dir:
        Root directory containing training data.
    min_ram_gb:
        Minimum available RAM in GiB (warning if below).
    min_disk_gb:
        Minimum free disk space in GiB at ``data_dir`` (warning if below).
    warn_swap_gb:
        Swap usage threshold in GiB above which a warning is emitted.

    Returns
    -------
    PreflightResult
        Aggregated preflight result. ``result.passed`` is ``True`` if
        no check has ``CRITICAL`` status.
    """
    environment = detect_environment()

    # Determine which directory to use for disk check — fall back to cwd
    disk_check_path = data_dir if data_dir.exists() else Path.cwd()

    checks: list[PreflightCheck] = [
        check_gpu(),
        check_ram(min_gb=min_ram_gb),
        check_disk_space(path=disk_check_path, min_gb=min_disk_gb),
        check_swap(warn_gb=warn_swap_gb),
        check_data_exists(data_dir),
    ]

    return PreflightResult(checks=checks, environment=environment)
