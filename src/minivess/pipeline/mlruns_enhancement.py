"""Post-hoc enhancement of MLflow run metadata.

Adds missing tags and parameters to existing runs for academic
reproducibility (TRIPOD+AI compliance).

MLflow stores its local filesystem artifacts as:
    mlruns/<experiment_id>/<run_id>/
        tags/<key>          — plain text, one value per file
        metrics/<key>       — lines: "<timestamp> <value> <step>"
        params/<key>        — plain text, one value per file
        artifacts/          — arbitrary nested artifacts

All enhancement functions operate directly on the filesystem (no live
MLflow tracking server or client required).
"""

from __future__ import annotations

import logging
import platform
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def identify_production_runs(
    mlruns_dir: Path,
    experiment_id: str,
) -> list[str]:
    """Identify production runs by presence of eval_fold2 metrics.

    A run is classified as *production* when it has evaluation metrics for
    **all three folds** (fold0, fold1, and fold2), indicated by the presence
    of at least one metric file whose name starts with ``eval_fold2``.

    Incomplete runs (e.g. those interrupted after fold0/fold1 only) are
    excluded.

    Args:
        mlruns_dir: Root mlruns directory (e.g. ``repo_root / "mlruns"``).
        experiment_id: MLflow experiment ID string.

    Returns:
        Sorted list of run ID strings for runs that completed all folds.
    """
    experiment_dir = mlruns_dir / experiment_id
    if not experiment_dir.is_dir():
        return []

    production_runs: list[str] = []
    for run_dir in sorted(experiment_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "meta.yaml":
            continue
        metrics_dir = run_dir / "metrics"
        if not metrics_dir.is_dir():
            continue
        # Production discriminator: all 3 folds complete (fold2 present)
        has_fold2 = any(
            metric_file.name.startswith("eval_fold2")
            for metric_file in metrics_dir.iterdir()
            if metric_file.is_file()
        )
        if has_fold2:
            production_runs.append(run_dir.name)

    return sorted(production_runs)


def get_software_versions() -> dict[str, str]:
    """Get current software version info.

    Collects Python, PyTorch, MONAI, and CUDA versions.  Each library is
    imported conditionally so the function degrades gracefully when a
    dependency is not installed.

    Returns:
        Dict mapping version key to version string.  ``python_version`` is
        always present; other keys are present only when the library is
        importable.
    """
    versions: dict[str, str] = {
        "python_version": platform.python_version(),
    }
    try:
        import torch

        versions["pytorch_version"] = torch.__version__
        # CUDA version is only meaningful when torch is available
        try:
            if torch.cuda.is_available():
                versions["cuda_version"] = torch.version.cuda or "N/A"
            else:
                versions["cuda_version"] = "N/A"
        except (AttributeError, RuntimeError):
            versions["cuda_version"] = "N/A"
    except ImportError:
        pass

    try:
        import monai

        versions["monai_version"] = monai.__version__
    except ImportError:
        pass

    return versions


def get_hardware_spec() -> dict[str, str]:
    """Get hardware specification.

    Reads total system RAM from ``psutil`` and, when CUDA is available,
    collects GPU model name and VRAM capacity.

    Returns:
        Dict mapping hardware key to value string.  ``total_ram_gb`` is
        always present when psutil is available; GPU keys are present only
        when a CUDA-capable device is detected.
    """
    spec: dict[str, str] = {}

    try:
        import psutil

        spec["total_ram_gb"] = f"{psutil.virtual_memory().total / (1024**3):.1f}"
    except ImportError:
        spec["total_ram_gb"] = "N/A"

    try:
        import torch

        if torch.cuda.is_available():
            spec["gpu_model"] = torch.cuda.get_device_name(0)
            spec["gpu_vram_mb"] = str(
                torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            )
        else:
            spec["gpu_model"] = "N/A"
            spec["gpu_vram_mb"] = "N/A"
    except (ImportError, RuntimeError):
        spec["gpu_model"] = "N/A"
        spec["gpu_vram_mb"] = "N/A"

    return spec


def get_git_commit() -> str:
    """Get current git commit hash.

    Runs ``git rev-parse HEAD`` in the repository root (three levels above
    this source file).  Returns ``"unknown"`` if git is unavailable or the
    command fails (e.g. in a CI Docker image without git history).

    Returns:
        40-character hex commit SHA, or ``"unknown"`` on failure.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[3],
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def enhance_run_tags(
    mlruns_dir: Path,
    experiment_id: str,
    run_id: str,
) -> dict[str, str]:
    """Add backward-compat and reproducibility tags to a run.

    Adds the following tags when not already present:

    - ``loss_type`` — alias for ``loss_function`` (backward compat for
      EnsembleBuilder which reads ``loss_type``).
    - ``python_version``, ``pytorch_version``, ``monai_version``,
      ``cuda_version`` — software environment snapshot.
    - ``total_ram_gb``, ``gpu_model``, ``gpu_vram_mb`` — hardware spec.
    - ``git_commit`` — repository HEAD at enhancement time.

    Writes directly to the MLflow filesystem layout (no client needed).
    Existing tag files are **never** overwritten (idempotent).

    Args:
        mlruns_dir: Root mlruns directory.
        experiment_id: MLflow experiment ID string.
        run_id: 32-character hexadecimal run ID.

    Returns:
        Dict of ``{tag_name: tag_value}`` for tags that were **newly added**.
        Returns empty dict when the tags directory does not exist.
    """
    tags_dir = mlruns_dir / experiment_id / run_id / "tags"
    if not tags_dir.is_dir():
        return {}

    added: dict[str, str] = {}

    # --- loss_type alias (backward compat for EnsembleBuilder) ---
    loss_function_file = tags_dir / "loss_function"
    loss_type_file = tags_dir / "loss_type"
    if loss_function_file.exists() and not loss_type_file.exists():
        loss_fn = loss_function_file.read_text(encoding="utf-8").strip()
        loss_type_file.write_text(loss_fn, encoding="utf-8")
        added["loss_type"] = loss_fn

    # --- software versions ---
    for key, value in get_software_versions().items():
        tag_file = tags_dir / key
        if not tag_file.exists():
            tag_file.write_text(value, encoding="utf-8")
            added[key] = value

    # --- hardware spec ---
    for key, value in get_hardware_spec().items():
        tag_file = tags_dir / key
        if not tag_file.exists():
            tag_file.write_text(value, encoding="utf-8")
            added[key] = value

    # --- git commit hash ---
    git_file = tags_dir / "git_commit"
    if not git_file.exists():
        commit = get_git_commit()
        git_file.write_text(commit, encoding="utf-8")
        added["git_commit"] = commit

    return added


def enhance_all_production_runs(
    mlruns_dir: Path,
    experiment_id: str,
) -> dict[str, dict[str, str]]:
    """Enhance all production runs with missing metadata.

    Calls :func:`identify_production_runs` then :func:`enhance_run_tags` for
    each qualifying run.  Only runs where at least one tag was added are
    included in the returned dict.

    Args:
        mlruns_dir: Root mlruns directory.
        experiment_id: MLflow experiment ID string.

    Returns:
        Mapping of ``{run_id: {tag_name: tag_value}}`` for all newly added
        tags.  Runs where nothing was added are omitted.
    """
    production_runs = identify_production_runs(mlruns_dir, experiment_id)
    results: dict[str, dict[str, str]] = {}
    for run_id in production_runs:
        added = enhance_run_tags(mlruns_dir, experiment_id, run_id)
        if added:
            results[run_id] = added
            logger.info("Enhanced run %s with %d tags", run_id, len(added))
    return results
