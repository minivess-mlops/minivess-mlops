"""Version pin consistency checker — L1 pre-commit deterministic gate.

Parses .env.example for *_VERSION keys, then verifies every Dockerfile,
docker-compose.yml, and Pulumi code uses the same version.

Catches the MLFLOW_SERVER_VERSION drift class: a version pinned in
.env.example hardcoded in 5+ files that can diverge independently.

See: .claude/metalearning/2026-03-14-mlflow-version-mismatch-fuckup.md
See: .claude/skills/qa-single-source-of-truth-yaml-quality-scanner-plan.md
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _parse_env_versions(env_path: Path) -> dict[str, str]:
    """Parse .env.example for keys ending in _VERSION."""
    versions: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key.endswith("_VERSION") and value:
            versions[key] = value
    return versions


def _check_mlflow_version(version: str) -> list[str]:
    """Check MLFLOW_SERVER_VERSION consistency across all files."""
    violations: list[str] = []
    deploy = REPO_ROOT / "deployment"

    # Dockerfiles: FROM lines should use ARG or match exactly
    for dockerfile_name in ("Dockerfile.mlflow", "Dockerfile.mlflow-gcp"):
        dockerfile = deploy / "docker" / dockerfile_name
        if not dockerfile.exists():
            continue
        content = dockerfile.read_text(encoding="utf-8")
        for i, line in enumerate(content.splitlines(), 1):
            is_mlflow_from = line.startswith("FROM") and "mlflow" in line.lower()
            if (
                is_mlflow_from
                and f"v{version}" not in line
                and f":{version}" not in line
                and "${" not in line
                and "ARG" not in line
            ):
                violations.append(
                    f"{dockerfile_name}:{i}: FROM uses different version than "
                    f"MLFLOW_SERVER_VERSION={version}"
                )

    # docker-compose.yml: image tags
    compose = deploy / "docker-compose.yml"
    if compose.exists():
        for i, line in enumerate(compose.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if "image:" in stripped and "mlflow" in stripped.lower():
                # Allow ${MLFLOW_SERVER_VERSION} substitution
                if "${" in stripped:
                    continue
                if f"v{version}" not in stripped and f":{version}" not in stripped:
                    violations.append(
                        f"docker-compose.yml:{i}: mlflow image uses different version "
                        f"than MLFLOW_SERVER_VERSION={version}"
                    )

    # Pulumi __main__.py: MLFLOW_SERVER_VERSION constant
    pulumi_main = deploy / "pulumi" / "gcp" / "__main__.py"
    if pulumi_main.exists():
        for i, line in enumerate(pulumi_main.read_text(encoding="utf-8").splitlines(), 1):
            is_version_line = "MLFLOW_SERVER_VERSION" in line and "=" in line and not line.strip().startswith("#")
            if is_version_line and f'"{version}"' not in line and f"'{version}'" not in line:
                violations.append(
                    f"pulumi/gcp/__main__.py:{i}: MLFLOW_SERVER_VERSION disagrees "
                        f"with .env.example value {version}"
                    )

    return violations


def main() -> int:
    """Run version pin consistency checks."""
    env_path = REPO_ROOT / ".env.example"
    if not env_path.exists():
        print("WARNING: .env.example not found — skipping version pin check")
        return 0

    versions = _parse_env_versions(env_path)
    if not versions:
        print("No *_VERSION keys found in .env.example")
        return 0

    all_violations: list[str] = []

    # Check MLFLOW_SERVER_VERSION specifically (highest ROI)
    mlflow_ver = versions.get("MLFLOW_SERVER_VERSION")
    if mlflow_ver:
        all_violations.extend(_check_mlflow_version(mlflow_ver))

    if all_violations:
        print("=" * 60)
        print("VERSION PIN CONSISTENCY — VIOLATIONS FOUND")
        print("=" * 60)
        for v in all_violations:
            print(f"  ✗ {v}")
        print()
        print(f"Fix: ensure all files use MLFLOW_SERVER_VERSION={mlflow_ver}")
        print("Source of truth: .env.example")
        return 1

    print(f"Version pins consistent: {', '.join(f'{k}={v}' for k, v in versions.items())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
