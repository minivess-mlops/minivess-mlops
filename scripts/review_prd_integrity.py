"""PRD integrity auditor — validates the 52-node Bayesian decision network.

Usage:
  uv run python scripts/review_prd_integrity.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
KG_DIR = REPO_ROOT / "knowledge-graph"
DECISIONS_DIR = KG_DIR / "decisions"


def _load_network() -> dict | None:
    """Load _network.yaml."""
    path = KG_DIR / "_network.yaml"
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_all_decisions() -> dict[str, dict]:
    """Load all decision YAML files into a dict keyed by decision_id."""
    decisions = {}
    if not DECISIONS_DIR.exists():
        return decisions
    for yaml_file in DECISIONS_DIR.rglob("*.yaml"):
        data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        if data and "decision_id" in data:
            decisions[data["decision_id"]] = data
    return decisions


def _check_node_count() -> list[dict]:
    """Check: Network has expected 52 nodes."""
    network = _load_network()
    if not network:
        return [
            {
                "check": "network_exists",
                "severity": "ERROR",
                "ok": False,
                "message": "_network.yaml does not exist",
            }
        ]
    count = len(network.get("nodes", []))
    ok = count == 52
    return [
        {
            "check": "node_count",
            "severity": "ERROR",
            "ok": ok,
            "message": f"Node count: {count} ({'OK' if ok else 'EXPECTED 52'})",
        }
    ]


def _check_all_nodes_have_files() -> list[dict]:
    """Check: Every node in network has a matching decision YAML."""
    checks = []
    network = _load_network()
    if not network:
        return []
    for node in network.get("nodes", []):
        file_ref = node.get("file", "")
        path = KG_DIR / file_ref
        ok = path.exists()
        checks.append(
            {
                "check": f"node_file:{node['id']}",
                "severity": "ERROR",
                "ok": ok,
                "message": f"{'OK' if ok else 'MISSING'}: {file_ref}",
            }
        )
    return checks


def _check_decision_ids_match_network() -> list[dict]:
    """Check: decision_id in file matches node id in network."""
    checks = []
    network = _load_network()
    decisions = _load_all_decisions()
    if not network:
        return []

    network_ids = {n["id"] for n in network.get("nodes", [])}
    decision_ids = set(decisions.keys())

    # IDs in network but not in files
    for missing in network_ids - decision_ids:
        checks.append(
            {
                "check": f"id_match:missing_file:{missing}",
                "severity": "ERROR",
                "ok": False,
                "message": f"Node '{missing}' in network but no decision file with that ID",
            }
        )

    # IDs in files but not in network
    for extra in decision_ids - network_ids:
        checks.append(
            {
                "check": f"id_match:extra_file:{extra}",
                "severity": "WARN",
                "ok": False,
                "message": f"Decision file '{extra}' not referenced in network",
            }
        )

    if not checks:
        checks.append(
            {
                "check": "id_match",
                "severity": "ERROR",
                "ok": True,
                "message": f"All {len(network_ids)} node IDs match decision files",
            }
        )
    return checks


def _check_dag_acyclicity() -> list[dict]:
    """Check: Network is a DAG (no cycles) via Kahn's algorithm."""
    network = _load_network()
    if not network:
        return []

    # Build adjacency
    in_degree: dict[str, int] = defaultdict(int)
    adj: dict[str, list[str]] = defaultdict(list)
    all_nodes = {n["id"] for n in network.get("nodes", [])}

    for node_id in all_nodes:
        in_degree.setdefault(node_id, 0)

    for edge in network.get("edges", []):
        src, dst = edge["from"], edge["to"]
        adj[src].append(dst)
        in_degree[dst] += 1

    # Kahn's algorithm
    queue = [n for n in all_nodes if in_degree[n] == 0]
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for neighbor in adj.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    ok = visited == len(all_nodes)
    return [
        {
            "check": "dag_acyclic",
            "severity": "ERROR",
            "ok": ok,
            "message": f"DAG acyclicity: {'OK' if ok else f'CYCLE DETECTED ({visited}/{len(all_nodes)} visited)'}",
        }
    ]


def _check_edge_references() -> list[dict]:
    """Check: Every edge references existing nodes."""
    checks = []
    network = _load_network()
    if not network:
        return []

    node_ids = {n["id"] for n in network.get("nodes", [])}
    for edge in network.get("edges", []):
        for direction in ["from", "to"]:
            node_id = edge.get(direction, "")
            ok = node_id in node_ids
            if not ok:
                checks.append(
                    {
                        "check": f"edge_ref:{direction}:{node_id}",
                        "severity": "ERROR",
                        "ok": False,
                        "message": f"Edge '{edge.get('from')}' → '{edge.get('to')}': node '{node_id}' not in network",
                    }
                )

    if not checks:
        edge_count = len(network.get("edges", []))
        checks.append(
            {
                "check": "edge_references",
                "severity": "ERROR",
                "ok": True,
                "message": f"All {edge_count} edges reference valid nodes",
            }
        )
    return checks


def _check_level_ordering() -> list[dict]:
    """Check: Edges flow from lower to higher levels (or skip)."""
    checks = []
    network = _load_network()
    if not network:
        return []

    level_map = {}
    level_order = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}
    for node in network.get("nodes", []):
        level_map[node["id"]] = node.get("level", "")

    violations = 0
    for edge in network.get("edges", []):
        src_level = level_order.get(level_map.get(edge["from"], ""), 0)
        dst_level = level_order.get(level_map.get(edge["to"], ""), 0)
        if src_level > dst_level and not edge.get("skip"):
            violations += 1
            checks.append(
                {
                    "check": f"level_order:{edge['from']}→{edge['to']}",
                    "severity": "WARN",
                    "ok": False,
                    "message": f"Edge {edge['from']} (L{src_level}) → {edge['to']} (L{dst_level}) flows upward without skip=true",
                }
            )

    if violations == 0:
        checks.append(
            {
                "check": "level_ordering",
                "severity": "WARN",
                "ok": True,
                "message": "All edges flow downward or are marked as skip connections",
            }
        )
    return checks


def _check_probabilities_sum() -> list[dict]:
    """Check: Options in each decision sum to ~1.0."""
    checks = []
    decisions = _load_all_decisions()
    for dec_id, data in decisions.items():
        options = data.get("options", [])
        if not options:
            continue
        prior_sum = sum(opt.get("prior_probability", 0) for opt in options)
        ok = abs(prior_sum - 1.0) < 0.05
        if not ok:
            checks.append(
                {
                    "check": f"prob_sum:{dec_id}",
                    "severity": "ERROR",
                    "ok": False,
                    "message": f"Prior probabilities sum to {prior_sum:.3f} for {dec_id} (expected ~1.0)",
                }
            )

    if not checks:
        checks.append(
            {
                "check": "prob_sums",
                "severity": "ERROR",
                "ok": True,
                "message": f"All {len(decisions)} decisions have valid probability sums",
            }
        )
    return checks


def _check_resolved_have_winner() -> list[dict]:
    """Check: Every resolved decision has a resolved_option."""
    checks = []
    decisions = _load_all_decisions()
    for dec_id, data in decisions.items():
        if data.get("status") == "resolved":
            has_winner = bool(data.get("resolved_option"))
            if not has_winner:
                checks.append(
                    {
                        "check": f"resolved_winner:{dec_id}",
                        "severity": "ERROR",
                        "ok": False,
                        "message": f"Decision '{dec_id}' is resolved but has no resolved_option",
                    }
                )
    if not checks:
        resolved_count = sum(
            1 for d in decisions.values() if d.get("status") == "resolved"
        )
        checks.append(
            {
                "check": "resolved_winners",
                "severity": "ERROR",
                "ok": True,
                "message": f"All {resolved_count} resolved decisions have a resolved_option",
            }
        )
    return checks


def _check_status_distribution() -> list[dict]:
    """Info check: Report status distribution across decisions."""
    decisions = _load_all_decisions()
    status_counts: dict[str, int] = defaultdict(int)
    for data in decisions.values():
        status_counts[data.get("status", "unknown")] += 1

    return [
        {
            "check": "status_distribution",
            "severity": "INFO",
            "ok": True,
            "message": f"Status distribution: {dict(status_counts)}",
        }
    ]


def main() -> dict:
    """Run all PRD integrity checks."""
    all_checks: list[dict] = []
    all_checks.extend(_check_node_count())
    all_checks.extend(_check_all_nodes_have_files())
    all_checks.extend(_check_decision_ids_match_network())
    all_checks.extend(_check_dag_acyclicity())
    all_checks.extend(_check_edge_references())
    all_checks.extend(_check_level_ordering())
    all_checks.extend(_check_probabilities_sum())
    all_checks.extend(_check_resolved_have_winner())
    all_checks.extend(_check_status_distribution())

    failures = sum(1 for c in all_checks if not c["ok"] and c["severity"] == "ERROR")
    warnings = sum(1 for c in all_checks if not c["ok"] and c["severity"] == "WARN")

    return {
        "agent_name": "prd_auditor",
        "failures": failures,
        "warnings": warnings,
        "total_checks": len(all_checks),
        "checks": all_checks,
    }


if __name__ == "__main__":
    result = main()
    print(f"\n{'=' * 60}")
    print("PRD Integrity Auditor")
    print(f"{'=' * 60}")
    print(f"Total checks: {result['total_checks']}")
    print(f"Failures (ERROR): {result['failures']}")
    print(f"Warnings (WARN):  {result['warnings']}")
    for check in result["checks"]:
        if not check["ok"]:
            print(f"  [{check['severity']}] {check['message']}")
        elif check["severity"] == "INFO":
            print(f"  [INFO] {check['message']}")
    if result["failures"] > 0:
        print(f"\n{result['failures']} ERROR(s) — PRD integrity violated")
        sys.exit(1)
    else:
        print("\nPRD integrity OK!")
        sys.exit(0)
