"""Knowledge graph link checker — verifies all cross-references are valid.

Usage:
  uv run python scripts/review_knowledge_links.py          # full check
  uv run python scripts/review_knowledge_links.py --quick   # paths only, skip bibliography
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
KG_DIR = REPO_ROOT / "knowledge-graph"


def _check_navigator_paths() -> list[dict]:
    """Check 1: Every path in navigator.yaml exists on disk."""
    checks = []
    nav_path = KG_DIR / "navigator.yaml"
    if not nav_path.exists():
        return [
            {
                "check": "navigator_exists",
                "severity": "ERROR",
                "ok": False,
                "message": "knowledge-graph/navigator.yaml does not exist",
            }
        ]

    nav = yaml.safe_load(nav_path.read_text(encoding="utf-8"))
    for domain_name, domain_data in nav.get("domains", {}).items():
        nav_file = domain_data.get("navigator", "")
        full_path = REPO_ROOT / nav_file
        ok = full_path.exists()
        checks.append(
            {
                "check": f"navigator_path:{domain_name}",
                "severity": "ERROR",
                "ok": ok,
                "message": f"{'OK' if ok else 'MISSING'}: {nav_file}",
            }
        )
        for claude_md in domain_data.get("claude_md", []):
            claude_path = REPO_ROOT / claude_md
            ok = claude_path.exists()
            checks.append(
                {
                    "check": f"claude_md:{claude_md}",
                    "severity": "WARN",
                    "ok": ok,
                    "message": f"{'OK' if ok else 'MISSING'}: {claude_md}",
                }
            )
    return checks


def _check_domain_file_paths() -> list[dict]:
    """Check 2-3: Implementation and evidence paths in domain files exist."""
    checks = []
    domains_dir = KG_DIR / "domains"
    if not domains_dir.exists():
        return []

    for yaml_file in sorted(domains_dir.glob("*.yaml")):
        data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        if not data:
            continue
        for dec_name, dec_data in data.get("decisions", {}).items():
            if not isinstance(dec_data, dict):
                continue
            # Check implementation paths
            impl = dec_data.get("implementation", "")
            if isinstance(impl, str) and impl:
                impl_path = REPO_ROOT / impl
                ok = impl_path.exists()
                checks.append(
                    {
                        "check": f"impl:{dec_name}:{impl}",
                        "severity": "WARN",
                        "ok": ok,
                        "message": f"{'OK' if ok else 'MISSING'}: {impl}",
                    }
                )
            elif isinstance(impl, list):
                for p in impl:
                    impl_path = REPO_ROOT / p
                    ok = impl_path.exists()
                    checks.append(
                        {
                            "check": f"impl:{dec_name}:{p}",
                            "severity": "WARN",
                            "ok": ok,
                            "message": f"{'OK' if ok else 'MISSING'}: {p}",
                        }
                    )
            # Check evidence paths
            for ev in dec_data.get("evidence", []):
                if isinstance(ev, str):
                    ev_path = REPO_ROOT / ev
                    ok = ev_path.exists()
                    checks.append(
                        {
                            "check": f"evidence:{dec_name}:{ev}",
                            "severity": "WARN",
                            "ok": ok,
                            "message": f"{'OK' if ok else 'MISSING'}: {ev}",
                        }
                    )
            # Check prd_node path
            prd = dec_data.get("prd_node", "")
            if prd:
                prd_path = REPO_ROOT / prd
                ok = prd_path.exists()
                checks.append(
                    {
                        "check": f"prd_node:{dec_name}",
                        "severity": "ERROR",
                        "ok": ok,
                        "message": f"{'OK' if ok else 'MISSING'}: {prd}",
                    }
                )
    return checks


def _check_bibliography_keys() -> list[dict]:
    """Check 6: Every citation_key in decision files resolves in bibliography.yaml."""
    checks = []
    bib_path = KG_DIR / "bibliography.yaml"
    if not bib_path.exists():
        return [
            {
                "check": "bibliography_exists",
                "severity": "ERROR",
                "ok": False,
                "message": "knowledge-graph/bibliography.yaml does not exist",
            }
        ]

    bib = yaml.safe_load(bib_path.read_text(encoding="utf-8"))
    known_keys = set(bib.get("citations", {}).keys())

    decisions_dir = KG_DIR / "decisions"
    if not decisions_dir.exists():
        return checks

    for yaml_file in sorted(decisions_dir.rglob("*.yaml")):
        data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        if not data:
            continue
        for evidence_item in data.get("resolution_evidence", []):
            if not isinstance(evidence_item, dict):
                continue
            ckey = evidence_item.get("citation_key")
            if ckey:
                ok = ckey in known_keys
                checks.append(
                    {
                        "check": f"citation:{ckey}",
                        "severity": "ERROR",
                        "ok": ok,
                        "message": f"{'OK' if ok else 'UNKNOWN KEY'}: {ckey} in {yaml_file.name}",
                    }
                )
    return checks


def _check_memory_md_length() -> list[dict]:
    """Check 7: MEMORY.md is under 200 lines."""
    memory_path = (
        Path.home()
        / ".claude"
        / "projects"
        / "-home-petteri-Dropbox-github-personal-minivess-mlops"
        / "memory"
        / "MEMORY.md"
    )
    if not memory_path.exists():
        return [
            {
                "check": "memory_md_exists",
                "severity": "WARN",
                "ok": True,
                "message": "MEMORY.md not found at expected path — skipping",
            }
        ]

    lines = memory_path.read_text(encoding="utf-8").splitlines()
    ok = len(lines) <= 200
    return [
        {
            "check": "memory_md_length",
            "severity": "WARN",
            "ok": ok,
            "message": f"MEMORY.md is {len(lines)} lines ({'OK' if ok else 'OVER 200 LINE LIMIT'})",
        }
    ]


def _check_domain_dates() -> list[dict]:
    """Check 9: Every domain navigator last_reviewed date is parseable."""
    checks = []
    domains_dir = KG_DIR / "domains"
    if not domains_dir.exists():
        return []

    for yaml_file in sorted(domains_dir.glob("*.yaml")):
        data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        if not data:
            continue
        date_str = data.get("last_reviewed", "")
        if not date_str:
            checks.append(
                {
                    "check": f"date:{yaml_file.name}",
                    "severity": "ERROR",
                    "ok": False,
                    "message": f"MISSING last_reviewed in {yaml_file.name}",
                }
            )
            continue
        try:
            datetime.strptime(str(date_str), "%Y-%m-%d")
            checks.append(
                {
                    "check": f"date:{yaml_file.name}",
                    "severity": "ERROR",
                    "ok": True,
                    "message": f"OK: {yaml_file.name} last_reviewed={date_str}",
                }
            )
        except ValueError:
            checks.append(
                {
                    "check": f"date:{yaml_file.name}",
                    "severity": "ERROR",
                    "ok": False,
                    "message": f"UNPARSEABLE date '{date_str}' in {yaml_file.name}",
                }
            )
    return checks


def _check_network_node_files() -> list[dict]:
    """Check: Every node in _network.yaml has a corresponding decision file."""
    checks = []
    network_path = KG_DIR / "_network.yaml"
    if not network_path.exists():
        return []

    network = yaml.safe_load(network_path.read_text(encoding="utf-8"))
    for node in network.get("nodes", []):
        file_ref = node.get("file", "")
        if file_ref:
            full_path = KG_DIR / file_ref
            ok = full_path.exists()
            checks.append(
                {
                    "check": f"network_node:{node.get('id', 'unknown')}",
                    "severity": "ERROR",
                    "ok": ok,
                    "message": f"{'OK' if ok else 'MISSING'}: {file_ref}",
                }
            )
    return checks


def main(quick: bool = False) -> dict:
    """Run all link checks and return results."""
    all_checks: list[dict] = []

    all_checks.extend(_check_navigator_paths())
    all_checks.extend(_check_domain_file_paths())
    all_checks.extend(_check_domain_dates())
    all_checks.extend(_check_network_node_files())
    all_checks.extend(_check_memory_md_length())

    if not quick:
        all_checks.extend(_check_bibliography_keys())

    failures = sum(1 for c in all_checks if not c["ok"] and c["severity"] == "ERROR")
    warnings = sum(1 for c in all_checks if not c["ok"] and c["severity"] == "WARN")

    return {
        "agent_name": "link_checker",
        "failures": failures,
        "warnings": warnings,
        "total_checks": len(all_checks),
        "checks": all_checks,
    }


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    result = main(quick=quick)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Knowledge Graph Link Checker ({'quick' if quick else 'full'} mode)")
    print(f"{'=' * 60}")
    print(f"Total checks: {result['total_checks']}")
    print(f"Failures (ERROR): {result['failures']}")
    print(f"Warnings (WARN):  {result['warnings']}")

    # Print failures and warnings
    for check in result["checks"]:
        if not check["ok"]:
            severity = check["severity"]
            print(f"  [{severity}] {check['message']}")

    if result["failures"] > 0:
        print(
            f"\n{result['failures']} ERROR(s) found — knowledge graph has broken links"
        )
        sys.exit(1)
    elif result["warnings"] > 0:
        print(f"\n{result['warnings']} WARNING(s) — review recommended")
        sys.exit(0)
    else:
        print("\nAll checks passed!")
        sys.exit(0)
