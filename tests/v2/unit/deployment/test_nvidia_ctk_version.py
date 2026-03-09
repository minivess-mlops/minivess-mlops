"""Tests for NVIDIA Container Toolkit version — CVE-2025-23266 mitigation.

CVE-2025-23266 (CVSS 9.0 Critical, NVIDIAScape): container-to-host escape.
Fixed in Container Toolkit v1.17.8+.

Rule #16: No regex. Parse version with str.split().
"""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent.parent.parent

_SAFE_MIN = (1, 17, 8)


def _ctk_installed() -> bool:
    try:
        subprocess.run(
            ["nvidia-ctk", "--version"],
            capture_output=True,
            check=False,
            timeout=5,
        )
        return True
    except FileNotFoundError:
        return False


_skip_no_ctk = pytest.mark.skipif(
    not _ctk_installed(), reason="nvidia-ctk not installed"
)


@_skip_no_ctk
def test_nvidia_ctk_version_is_safe() -> None:
    """nvidia-ctk version must be >= 1.17.8 (CVE-2025-23266 fix)."""
    result = subprocess.run(
        ["nvidia-ctk", "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    # Output: "NVIDIA Container Toolkit CLI version 1.17.8"
    # Use str.split() — Rule #16 bans regex
    output = result.stdout.strip() or result.stderr.strip()
    parts = output.split()
    # Find the token that looks like a version (contains dots and digits)
    version_str = None
    for part in parts:
        if part.count(".") >= 1 and all(c.isdigit() or c == "." for c in part):
            version_str = part
            break

    assert version_str is not None, (
        f"Could not find version string in nvidia-ctk output: {output!r}"
    )

    version_parts = version_str.split(".")
    version_tuple = tuple(int(p) for p in version_parts[:3])

    assert version_tuple >= _SAFE_MIN, (
        f"nvidia-ctk {version_str} is BELOW safe minimum {'.'.join(str(x) for x in _SAFE_MIN)}. "
        f"CVE-2025-23266 (CVSS 9.0) is unpatched! "
        f"Fix: sudo apt-get install --only-upgrade nvidia-container-toolkit"
    )


def test_cuda_compat_hook_not_disabled_by_config() -> None:
    """If CTK config exists, compat hook must not be disabled."""
    config_path = Path("/etc/nvidia-container-toolkit/config.toml")
    if not config_path.exists():
        pytest.skip("CTK config.toml not found — skipping hook check")

    with config_path.open("rb") as f:
        cfg = tomllib.load(f)

    hook_disabled = cfg.get("nvidia-container-cli", {}).get(
        "disable-cuda-compat-lib-hook", False
    )
    assert not hook_disabled, (
        "disable-cuda-compat-lib-hook = true in CTK config.toml. "
        "This disables the compatibility hook — set to false or remove the key."
    )
