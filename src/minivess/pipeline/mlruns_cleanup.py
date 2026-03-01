"""MLflow run cleanup: identify and trash incomplete runs.

Scans an MLflow experiment's filesystem layout and moves incomplete
runs (those that did not finish all folds x epochs) to a ``.trash/``
directory under the mlruns root.  Trash-based deletion is reversible —
runs can be restored by moving them back.

MLflow filesystem layout:
    mlruns/<experiment_id>/<run_id>/
        tags/<key>          — plain text, one value per file
        metrics/<key>       — lines: "<timestamp> <value> <step>"
        params/<key>        — plain text, one value per file
        artifacts/          — arbitrary nested artifacts

Pattern reference: ``mlruns_enhancement.py`` (filesystem-first operations).
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunStatus:
    """Status of a single MLflow run's completeness."""

    run_id: str
    run_name: str
    num_entries: int
    expected_entries: int
    is_complete: bool


@dataclass
class CleanupResult:
    """Audit record from a cleanup operation."""

    experiment_id: str
    total_runs: int
    complete_runs: int
    identified: list[RunStatus] = field(default_factory=list)
    moved: int = 0

    def summary(self) -> str:
        """Return a human-readable summary of the cleanup result."""
        lines = [
            f"Experiment: {self.experiment_id}",
            f"Total runs: {self.total_runs}",
            f"Complete: {self.complete_runs}",
            f"Incomplete: {len(self.identified)}",
            f"Moved to trash: {self.moved}",
        ]
        if self.identified:
            lines.append("")
            lines.append("Incomplete runs:")
            for status in self.identified:
                lines.append(
                    f"  {status.run_id} ({status.run_name}): "
                    f"{status.num_entries}/{status.expected_entries} entries"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_run(
    mlruns_dir: Path,
    experiment_id: str,
    run_id: str,
    *,
    expected_entries: int = 300,
) -> RunStatus:
    """Classify a single run as complete or incomplete.

    A run is complete when its ``metrics/train_loss`` file has at least
    ``expected_entries`` lines (e.g. 300 = 3 folds x 100 epochs).

    Parameters
    ----------
    mlruns_dir:
        Root mlruns directory.
    experiment_id:
        MLflow experiment ID string.
    run_id:
        32-character hexadecimal run ID.
    expected_entries:
        Number of train_loss entries required for a complete run.

    Returns
    -------
    :class:`RunStatus` with completeness classification.

    Raises
    ------
    FileNotFoundError:
        If the run directory does not exist.
    """
    run_dir = mlruns_dir / experiment_id / run_id
    if not run_dir.is_dir():
        msg = f"Run directory does not exist: {run_dir}"
        raise FileNotFoundError(msg)

    # Read run name
    run_name_file = run_dir / "tags" / "mlflow.runName"
    run_name = ""
    if run_name_file.is_file():
        run_name = run_name_file.read_text(encoding="utf-8").strip()

    # Count train_loss entries
    train_loss_file = run_dir / "metrics" / "train_loss"
    num_entries = 0
    if train_loss_file.is_file():
        content = train_loss_file.read_text(encoding="utf-8").strip()
        if content:
            num_entries = len(content.splitlines())

    is_complete = num_entries >= expected_entries

    return RunStatus(
        run_id=run_id,
        run_name=run_name,
        num_entries=num_entries,
        expected_entries=expected_entries,
        is_complete=is_complete,
    )


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------


def identify_incomplete_runs(
    mlruns_dir: Path,
    experiment_id: str,
    *,
    expected_entries: int = 300,
) -> list[RunStatus]:
    """Scan an experiment and return statuses for all incomplete runs.

    Parameters
    ----------
    mlruns_dir:
        Root mlruns directory.
    experiment_id:
        MLflow experiment ID string.
    expected_entries:
        Number of train_loss entries required for completeness.

    Returns
    -------
    Sorted list of :class:`RunStatus` for runs that are NOT complete.
    Returns empty list if experiment does not exist.
    """
    experiment_dir = mlruns_dir / experiment_id
    if not experiment_dir.is_dir():
        return []

    incomplete: list[RunStatus] = []
    for run_dir in sorted(experiment_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_dir.name == "meta.yaml":
            continue

        status = classify_run(
            mlruns_dir,
            experiment_id,
            run_dir.name,
            expected_entries=expected_entries,
        )
        if not status.is_complete:
            incomplete.append(status)

    return incomplete


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def cleanup_incomplete_runs(
    mlruns_dir: Path,
    experiment_id: str,
    *,
    expected_entries: int = 300,
    dry_run: bool = True,
) -> CleanupResult:
    """Identify and optionally trash incomplete runs.

    Moves incomplete runs to ``mlruns_dir/.trash/<experiment_id>/<run_id>/``
    rather than permanently deleting them, so they can be restored.

    Parameters
    ----------
    mlruns_dir:
        Root mlruns directory.
    experiment_id:
        MLflow experiment ID string.
    expected_entries:
        Number of train_loss entries required for completeness.
    dry_run:
        When ``True`` (default), only identify — do not move anything.

    Returns
    -------
    :class:`CleanupResult` with audit information.
    """
    experiment_dir = mlruns_dir / experiment_id
    if not experiment_dir.is_dir():
        return CleanupResult(
            experiment_id=experiment_id,
            total_runs=0,
            complete_runs=0,
        )

    # Classify all runs
    all_statuses: list[RunStatus] = []
    for run_dir in sorted(experiment_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_dir.name == "meta.yaml":
            continue

        status = classify_run(
            mlruns_dir,
            experiment_id,
            run_dir.name,
            expected_entries=expected_entries,
        )
        all_statuses.append(status)

    incomplete = [s for s in all_statuses if not s.is_complete]
    complete_count = sum(1 for s in all_statuses if s.is_complete)

    result = CleanupResult(
        experiment_id=experiment_id,
        total_runs=len(all_statuses),
        complete_runs=complete_count,
        identified=incomplete,
        moved=0,
    )

    if dry_run or not incomplete:
        if incomplete:
            logger.info(
                "[DRY RUN] Would move %d incomplete run(s) from experiment %s",
                len(incomplete),
                experiment_id,
            )
        return result

    # Move incomplete runs to trash
    trash_base = mlruns_dir / ".trash" / experiment_id
    trash_base.mkdir(parents=True, exist_ok=True)

    for status in incomplete:
        src = experiment_dir / status.run_id
        dst = trash_base / status.run_id

        shutil.move(str(src), str(dst))
        result.moved += 1
        logger.info(
            "Moved incomplete run %s (%s, %d entries) to %s",
            status.run_id,
            status.run_name,
            status.num_entries,
            dst,
        )

    logger.info(
        "Cleanup complete: %d/%d incomplete runs moved to trash for experiment %s",
        result.moved,
        len(incomplete),
        experiment_id,
    )
    return result
