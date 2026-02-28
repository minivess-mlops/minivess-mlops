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
from pathlib import Path

logger = logging.getLogger(__name__)

# Default location of the DVC tracking file for the MiniVess dataset
_DVC_FILE_PATH = Path(__file__).resolve().parents[3] / "data" / "minivess.dvc"


def get_system_params() -> dict[str, str]:
    """Collect OS, Python, hostname, RAM, and CPU model.

    CPU model detection fallback chain:
    1. ``/proc/cpuinfo`` parse (Linux)
    2. ``platform.processor()``
    3. ``"unknown"``
    """
    params: dict[str, str] = {
        "sys_python_version": platform.python_version(),
        "sys_os": platform.system(),
        "sys_os_kernel": platform.release(),
        "sys_hostname": platform.node() or "unknown",
    }

    # RAM via psutil
    try:
        import psutil

        params["sys_total_ram_gb"] = f"{psutil.virtual_memory().total / (1024**3):.1f}"
    except ImportError:
        params["sys_total_ram_gb"] = "unknown"

    # CPU model
    params["sys_cpu_model"] = _get_cpu_model()

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

        versions["sys_torch_version"] = torch.__version__
    except ImportError:
        versions["sys_torch_version"] = "not_installed"

    # CUDA + cuDNN (via PyTorch)
    try:
        import torch

        if torch.cuda.is_available():
            versions["sys_cuda_version"] = torch.version.cuda or "N/A"
            try:
                cudnn_ver = torch.backends.cudnn.version()
                versions["sys_cudnn_version"] = str(cudnn_ver) if cudnn_ver else "N/A"
            except (AttributeError, RuntimeError):
                versions["sys_cudnn_version"] = "N/A"
        else:
            versions["sys_cuda_version"] = "N/A"
            versions["sys_cudnn_version"] = "N/A"
    except ImportError:
        versions["sys_cuda_version"] = "not_installed"
        versions["sys_cudnn_version"] = "not_installed"

    # MONAI
    try:
        import monai

        versions["sys_monai_version"] = monai.__version__
    except ImportError:
        versions["sys_monai_version"] = "not_installed"

    # MLflow
    try:
        import mlflow

        versions["sys_mlflow_version"] = mlflow.__version__
    except ImportError:
        versions["sys_mlflow_version"] = "not_installed"

    # NumPy
    try:
        import numpy

        versions["sys_numpy_version"] = numpy.__version__
    except ImportError:
        versions["sys_numpy_version"] = "not_installed"

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
                "sys_gpu_count": "0",
                "sys_gpu_model": "N/A",
                "sys_gpu_vram_mb": "0",
            }

        count = torch.cuda.device_count()
        model = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)

        return {
            "sys_gpu_count": str(count),
            "sys_gpu_model": model,
            "sys_gpu_vram_mb": str(vram),
        }
    except (ImportError, RuntimeError):
        return {
            "sys_gpu_count": "0",
            "sys_gpu_model": "N/A",
            "sys_gpu_vram_mb": "0",
        }


def get_git_info() -> dict[str, str]:
    """Collect git commit, branch, and dirty state.

    Handles detached HEAD (returns descriptive string instead of "HEAD").
    Returns ``"unknown"`` for all fields if git is unavailable.
    """
    unknown = {
        "sys_git_commit": "unknown",
        "sys_git_commit_short": "unknown",
        "sys_git_branch": "unknown",
        "sys_git_dirty": "unknown",
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
            "sys_git_commit": commit,
            "sys_git_commit_short": commit_short,
            "sys_git_branch": branch,
            "sys_git_dirty": dirty,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return unknown


def get_dvc_info() -> dict[str, str]:
    """Collect DVC version and dataset data hash.

    Reads the DVC tracking file (``data/minivess.dvc``) to extract the
    content-addressed hash and file count.  Returns ``"unknown"`` for
    data fields when the file is absent.
    """
    info: dict[str, str] = {}

    # DVC Python package version
    try:
        import dvc

        info["sys_dvc_version"] = dvc.__version__
    except (ImportError, AttributeError):
        info["sys_dvc_version"] = "not_installed"

    # Parse .dvc file for md5 hash and nfiles
    info["sys_dvc_data_hash"] = "unknown"
    info["sys_dvc_data_nfiles"] = "unknown"

    try:
        if _DVC_FILE_PATH.exists():
            text = _DVC_FILE_PATH.read_text(encoding="utf-8")
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("- md5:"):
                    info["sys_dvc_data_hash"] = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("nfiles:"):
                    info["sys_dvc_data_nfiles"] = stripped.split(":", 1)[1].strip()
    except (OSError, PermissionError):
        logger.warning("Could not read DVC file: %s", _DVC_FILE_PATH)

    return info


def get_all_system_info() -> dict[str, str]:
    """Combine all system info into a single dict.

    All keys are prefixed with ``sys_``.  All values are strings.
    """
    info: dict[str, str] = {}
    info.update(get_system_params())
    info.update(get_library_versions())
    info.update(get_gpu_info())
    info.update(get_git_info())
    info.update(get_dvc_info())
    return info
