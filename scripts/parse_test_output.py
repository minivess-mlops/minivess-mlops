"""Parse pytest output and enforce zero-skip, zero-fail policy.

The SINGLE SOURCE OF TRUTH for test result extraction. Used by:
- Makefile targets (L1 enforcement)
- PostToolUse hook (L2 enforcement)
- Pre-commit hook (L3 enforcement)

Exit codes:
    0 = all passed, zero skips, zero failures
    1 = skips > 0 OR failures > 0 OR parse error

Usage:
    pytest tests/ | python scripts/parse_test_output.py
    python scripts/parse_test_output.py --input pytest_output.txt
    python scripts/parse_test_output.py --json tests/last_test_result.json

See: docs/planning/v0-2_archive/original_docs/silent-ignoring-and-kicking-the-can-down-the-road-problem.xml
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path


def parse_test_output(output: str, *, print_prompt: bool = False) -> dict:
    """Parse pytest output into structured result.

    Parameters
    ----------
    output:
        Raw pytest stdout/stderr text.
    print_prompt:
        If True and skips > 0, print mandatory investigation prompt.

    Returns
    -------
    Dict with keys: passed, failed, skipped, deselected, error,
    skip_reasons, raw_summary.
    """
    result: dict = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "deselected": 0,
        "error": 0,
        "skip_reasons": [],
        "raw_summary": "",
    }

    if not output or not output.strip():
        result["error"] = 1
        result["raw_summary"] = "EMPTY OUTPUT"
        return result

    lines = output.strip().split("\n")

    # Extract skip reasons from SKIPPED lines
    for line in lines:
        stripped = line.strip()
        if "SKIPPED" in stripped and ("[" in stripped or ":" in stripped):
            # Extract the reason part after the last colon
            parts = stripped.split(":", maxsplit=3)
            if len(parts) >= 4:
                reason = parts[3].strip()
                result["skip_reasons"].append(reason)
            elif len(parts) >= 3:
                reason = parts[2].strip()
                result["skip_reasons"].append(reason)

    # Find the summary line (e.g., "5362 passed, 2 skipped in 408.85s")
    summary_line = ""
    for line in reversed(lines):
        stripped = line.strip()
        if "passed" in stripped or "failed" in stripped or "error" in stripped:
            # Remove ANSI codes and leading = signs
            clean = stripped.strip("= ")
            if "passed" in clean or "failed" in clean:
                summary_line = clean
                break

    result["raw_summary"] = summary_line

    if not summary_line:
        result["error"] = 1
        return result

    # Parse counts from summary line using str.split (NO regex — Rule 16)
    # Format: "5362 passed, 2 failed, 3 skipped in 408.85s"
    # Split by comma first, then parse each segment
    # Remove the timing part ("in XXXs" or "(H:MM:SS)")
    summary_no_time = summary_line
    if " in " in summary_no_time:
        summary_no_time = summary_no_time.split(" in ")[0]

    segments = summary_no_time.split(",")
    for segment in segments:
        segment = segment.strip()
        parts = segment.split()
        if len(parts) >= 2:
            try:
                count = int(parts[0])
            except ValueError:
                continue
            label = parts[1].lower()
            if "passed" in label:
                result["passed"] = count
            elif "failed" in label:
                result["failed"] = count
            elif "skipped" in label:
                result["skipped"] = count
            elif "deselected" in label:
                result["deselected"] = count
            elif "error" in label:
                result["error"] = count

    # Print mandatory investigation prompt if skips detected
    if print_prompt and result["skipped"] > 0:
        print("\n" + "=" * 70)
        print("MANDATORY: ZERO-SKIP ENFORCEMENT (Rule 28)")
        print("=" * 70)
        print(f"  {result['skipped']} test(s) were SKIPPED. Skips are not allowed.")
        print("  SKIP reasons:")
        for i, reason in enumerate(result["skip_reasons"], 1):
            print(f"    {i}. {reason}")
        print("\n  Fix each skip by:")
        print("    - Install missing dependency (uv sync --all-extras)")
        print("    - Move hardware-gated test to tests/gpu_instance/")
        print("    - Delete deprecated test")
        print("    - Add to skip_allowlist.yaml with justification")
        print("=" * 70)

    if print_prompt and result["failed"] > 0:
        print(f"\nFATAL: {result['failed']} test(s) FAILED. Fix before proceeding.")

    return result


def get_exit_code(result: dict) -> int:
    """Determine exit code from parsed result.

    Returns 0 only if zero skips AND zero failures AND no parse error.
    """
    if result.get("skipped", 0) > 0:
        return 1
    if result.get("failed", 0) > 0:
        return 1
    if result.get("error", 0) > 0:
        return 1
    if result.get("passed", 0) == 0:
        return 1  # No tests passed = something is wrong
    return 0


def parse_and_save(output: str, json_path: Path) -> dict:
    """Parse test output and save result to JSON file.

    Parameters
    ----------
    output:
        Raw pytest output text.
    json_path:
        Path to write the JSON result.

    Returns
    -------
    The parsed result dict.
    """
    result = parse_test_output(output, print_prompt=True)
    result["timestamp"] = datetime.now(UTC).isoformat()
    result["exit_code"] = get_exit_code(result)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = json_path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp_path.rename(json_path)

    return result


def main() -> None:
    """CLI entry point — read from stdin or file, parse, save, exit."""
    import argparse

    parser = argparse.ArgumentParser(description="Parse pytest output for skip enforcement")
    parser.add_argument("--input", type=Path, help="Read from file instead of stdin")
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("tests/last_test_result.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    output = args.input.read_text(encoding="utf-8") if args.input else sys.stdin.read()

    result = parse_and_save(output, args.json)
    sys.exit(get_exit_code(result))


if __name__ == "__main__":
    main()
