"""System info collection for MLflow reproducibility logging.

Collects Python, library, GPU, git, and OS metadata as string-valued dicts
suitable for ``mlflow.log_params()``.  Every function degrades gracefully
when dependencies are missing (CI, Docker, CPU-only environments).

Pattern reference: foundation-PLR ``src/log_helpers/system_utils.py``.
"""

from __future__ import annotations

import logging
import platform
import subprocess

logger = logging.getLogger(__name__)


def get_system_params() -> dict[str, str]:
    """Collect OS, Python, hostname, RAM, and CPU model.

    CPU model detection fallback chain:
    1. ``/proc/cpuinfo`` parse (Linux)
    2. ``platform.processor()``
    3. ``"unknown"``
    """
    params: dict[str, str] = {
        "sys/python_version": platform.python_version(),
        "sys/os": platform.system(),
        "sys/os_kernel": platform.release(),
        "sys/hostname": platform.node() or "unknown",
    }

    # RAM via psutil
    try:
        import psutil

        params["sys/total_ram_gb"] = f"{psutil.virtual_memory().total / (1024**3):.1f}"
    except ImportError:
        params["sys/total_ram_gb"] = "unknown"

    # CPU model
    params["sys/cpu_model"] = _get_cpu_model()

    return params


def _get_cpu_model() -> str:
    """Get CPU model name with fallback chain."""
    # Try /proc/cpuinfo first (Linux)
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (FileNotFoundError, OSError, PermissionError):
        pass

    # Fallback to platform.processor()
    proc = platform.processor()
    if proc:
        return proc

    return "unknown"


def get_library_versions() -> dict[str, str]:
    """Collect versions of key ML libraries.

    Each library is imported conditionally. Returns ``"not_installed"``
    for libraries that cannot be imported.
    """
    versions: dict[str, str] = {}

    # PyTorch
    try:
        import torch

        versions["sys/torch_version"] = torch.__version__
    except ImportError:
        versions["sys/torch_version"] = "not_installed"

    # CUDA + cuDNN (via PyTorch)
    try:
        import torch

        if torch.cuda.is_available():
            versions["sys/cuda_version"] = torch.version.cuda or "N/A"
            try:
                cudnn_ver = torch.backends.cudnn.version()  # type: ignore[no-untyped-call]
                versions["sys/cudnn_version"] = str(cudnn_ver) if cudnn_ver else "N/A"
            except (AttributeError, RuntimeError):
                versions["sys/cudnn_version"] = "N/A"
        else:
            versions["sys/cuda_version"] = "N/A"
            versions["sys/cudnn_version"] = "N/A"
    except ImportError:
        versions["sys/cuda_version"] = "not_installed"
        versions["sys/cudnn_version"] = "not_installed"

    # MONAI
    try:
        import monai

        versions["sys/monai_version"] = monai.__version__
    except ImportError:
        versions["sys/monai_version"] = "not_installed"

    # MLflow
    try:
        import mlflow

        versions["sys/mlflow_version"] = mlflow.__version__
    except ImportError:
        versions["sys/mlflow_version"] = "not_installed"

    # NumPy
    try:
        import numpy

        versions["sys/numpy_version"] = numpy.__version__
    except ImportError:
        versions["sys/numpy_version"] = "not_installed"

    return versions


def get_gpu_info() -> dict[str, str]:
    """Collect GPU model, VRAM, and count.

    Guarded by ``torch.cuda.is_available()``. Returns ``sys_gpu_count=0``
    and ``sys_gpu_model=N/A`` when no GPU is available.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return {
                "sys/gpu_count": "0",
                "sys/gpu_model": "N/A",
                "sys/gpu_vram_mb": "0",
            }

        count = torch.cuda.device_count()
        model = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)

        return {
            "sys/gpu_count": str(count),
            "sys/gpu_model": model,
            "sys/gpu_vram_mb": str(vram),
        }
    except (ImportError, RuntimeError):
        return {
            "sys/gpu_count": "0",
            "sys/gpu_model": "N/A",
            "sys/gpu_vram_mb": "0",
        }


def get_git_info() -> dict[str, str]:
    """Collect git commit, branch, and dirty state.

    Handles detached HEAD (returns descriptive string instead of "HEAD").
    Returns ``"unknown"`` for all fields if git is unavailable.
    """
    unknown = {
        "sys/git_commit": "unknown",
        "sys/git_commit_short": "unknown",
        "sys/git_branch": "unknown",
        "sys/git_dirty": "unknown",
    }

    try:
        # Full commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return unknown
        commit = result.stdout.strip()

        # Short commit hash
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        commit_short = result.stdout.strip() if result.returncode == 0 else commit[:7]

        # Branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        branch = result.stdout.strip() if result.returncode == 0 else "unknown"

        # Handle detached HEAD
        if branch == "HEAD":
            branch = f"HEAD (detached at {commit_short})"

        # Dirty state
        dirty_result = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True,
            timeout=5,
        )
        dirty = "true" if dirty_result.returncode != 0 else "false"

        return {
            "sys/git_commit": commit,
            "sys/git_commit_short": commit_short,
            "sys/git_branch": branch,
            "sys/git_dirty": dirty,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return unknown


def get_dvc_info() -> dict[str, str]:
    """Collect DVC version only.

    DVC data hash and nfiles are *identifiers*, not hyperparameters, and
    belong in MLflow tags (logged via ``log_dvc_provenance()``), not in
    ``mlflow.log_params()``.  See issue #108.
    """
    info: dict[str, str] = {}

    # DVC Python package version
    try:
        import dvc

        info["sys/dvc_version"] = dvc.__version__
    except (ImportError, AttributeError):
        info["sys/dvc_version"] = "not_installed"

    return info


def get_all_system_info() -> dict[str, str]:
    """Combine all system info into a single dict.

    All keys are prefixed with ``sys/``.  All values are strings.
    """
    info: dict[str, str] = {}
    info.update(get_system_params())
    info.update(get_library_versions())
    info.update(get_gpu_info())
    info.update(get_git_info())
    info.update(get_dvc_info())
    return info
