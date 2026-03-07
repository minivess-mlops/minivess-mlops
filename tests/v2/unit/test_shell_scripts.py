"""Tests for T-19: shell wrapper scripts calling prefect deployment run.

Verifies that scripts/run_training.sh, run_pipeline.sh, and run_analysis.sh:
- Exist and are executable
- Do NOT invoke python scripts/*.py
- Use prefect deployment run

NO subprocess invocation in tests — reads files directly.
"""

from __future__ import annotations

import os
from pathlib import Path

_SCRIPTS_DIR = Path("scripts")


class TestRunTrainingSh:
    def test_run_training_sh_exists(self) -> None:
        """scripts/run_training.sh must exist."""
        assert (_SCRIPTS_DIR / "run_training.sh").exists(), (
            "scripts/run_training.sh does not exist. "
            "Create it as a prefect deployment run wrapper."
        )

    def test_run_training_sh_executable(self) -> None:
        """scripts/run_training.sh must be executable."""
        p = _SCRIPTS_DIR / "run_training.sh"
        if not p.exists():
            return
        assert os.access(str(p), os.X_OK), (
            "scripts/run_training.sh is not executable. Run: chmod +x scripts/run_training.sh"
        )

    def test_run_training_sh_has_prefect_run(self) -> None:
        """scripts/run_training.sh must call prefect deployment run."""
        p = _SCRIPTS_DIR / "run_training.sh"
        if not p.exists():
            return
        content = p.read_text(encoding="utf-8")
        assert "prefect deployment run" in content, (
            "scripts/run_training.sh must call 'prefect deployment run'. "
            "Do not call python scripts/*.py directly."
        )

    def test_run_training_sh_no_python_script(self) -> None:
        """scripts/run_training.sh must NOT invoke python scripts/."""
        p = _SCRIPTS_DIR / "run_training.sh"
        if not p.exists():
            return
        content = p.read_text(encoding="utf-8")
        assert "python scripts/" not in content, (
            "scripts/run_training.sh must NOT invoke python scripts/. "
            "Use 'prefect deployment run' instead."
        )

    def test_run_training_sh_has_shebang(self) -> None:
        """scripts/run_training.sh must have #!/usr/bin/env bash shebang."""
        p = _SCRIPTS_DIR / "run_training.sh"
        if not p.exists():
            return
        content = p.read_text(encoding="utf-8")
        assert content.startswith("#!/usr/bin/env bash"), (
            "scripts/run_training.sh must start with #!/usr/bin/env bash"
        )


class TestRunPipelineSh:
    def test_run_pipeline_sh_exists(self) -> None:
        """scripts/run_pipeline.sh must exist."""
        assert (_SCRIPTS_DIR / "run_pipeline.sh").exists(), (
            "scripts/run_pipeline.sh does not exist. "
            "Create it as a pipeline trigger wrapper."
        )

    def test_run_pipeline_sh_no_python_script(self) -> None:
        """scripts/run_pipeline.sh must NOT invoke python scripts/."""
        p = _SCRIPTS_DIR / "run_pipeline.sh"
        if not p.exists():
            return
        content = p.read_text(encoding="utf-8")
        assert "python scripts/" not in content, (
            "scripts/run_pipeline.sh must NOT invoke python scripts/."
        )

    def test_run_pipeline_sh_executable(self) -> None:
        """scripts/run_pipeline.sh must be executable."""
        p = _SCRIPTS_DIR / "run_pipeline.sh"
        if not p.exists():
            return
        assert os.access(str(p), os.X_OK), "scripts/run_pipeline.sh is not executable."


class TestRunAnalysisSh:
    def test_run_analysis_sh_exists(self) -> None:
        """scripts/run_analysis.sh must exist."""
        assert (_SCRIPTS_DIR / "run_analysis.sh").exists(), (
            "scripts/run_analysis.sh does not exist. "
            "Create it as an analysis flow wrapper."
        )

    def test_run_analysis_sh_no_python_script(self) -> None:
        """scripts/run_analysis.sh must NOT invoke python scripts/."""
        p = _SCRIPTS_DIR / "run_analysis.sh"
        if not p.exists():
            return
        content = p.read_text(encoding="utf-8")
        assert "python scripts/" not in content, (
            "scripts/run_analysis.sh must NOT invoke python scripts/."
        )

    def test_run_analysis_sh_executable(self) -> None:
        """scripts/run_analysis.sh must be executable."""
        p = _SCRIPTS_DIR / "run_analysis.sh"
        if not p.exists():
            return
        assert os.access(str(p), os.X_OK), "scripts/run_analysis.sh is not executable."
