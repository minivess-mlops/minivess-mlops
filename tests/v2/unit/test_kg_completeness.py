"""Knowledge Graph structural completeness tests.

These tests validate the structural integrity of the knowledge graph:
- All decision node files conform to the schema
- _network.yaml is consistent with the decision files
- navigator.yaml references valid domains and decisions
- DAG is acyclic and referentially complete
- Probability invariants hold

Staging tier: no model loading, no slow, no integration.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest
import yaml

# Root of the repository
REPO_ROOT = Path(__file__).resolve().parents[3]
KG_ROOT = REPO_ROOT / "knowledge-graph"
DECISIONS_DIR = KG_ROOT / "decisions"
DOMAINS_DIR = KG_ROOT / "domains"
NETWORK_FILE = KG_ROOT / "_network.yaml"
SCHEMA_FILE = KG_ROOT / "_schema.yaml"
NAVIGATOR_FILE = KG_ROOT / "navigator.yaml"


def _load_yaml(path: Path) -> dict:
    """Load a YAML file safely."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@pytest.fixture(scope="module")
def network() -> dict:
    """Load _network.yaml once per module."""
    return _load_yaml(NETWORK_FILE)


@pytest.fixture(scope="module")
def navigator() -> dict:
    """Load navigator.yaml once per module."""
    return _load_yaml(NAVIGATOR_FILE)


@pytest.fixture(scope="module")
def schema() -> dict:
    """Load _schema.yaml once per module."""
    return _load_yaml(SCHEMA_FILE)


@pytest.fixture(scope="module")
def all_decision_files() -> list[Path]:
    """Collect all decision YAML files."""
    return sorted(DECISIONS_DIR.rglob("*.yaml"))


@pytest.fixture(scope="module")
def all_decisions(all_decision_files: list[Path]) -> dict[str, dict]:
    """Load all decision files keyed by decision_id."""
    decisions = {}
    for path in all_decision_files:
        data = _load_yaml(path)
        if "decision_id" in data:
            decisions[data["decision_id"]] = data
    return decisions


@pytest.fixture(scope="module")
def network_node_ids(network: dict) -> set[str]:
    """Extract all node IDs from _network.yaml."""
    return {n["id"] for n in network.get("nodes", [])}


# ─── Test 1: All node files in _network.yaml exist on disk ───────────────────


class TestNetworkFileConsistency:
    """Validate _network.yaml references existing files."""

    def test_all_node_files_exist(self, network: dict) -> None:
        """Every node in _network.yaml must reference an existing file."""
        missing = []
        for node in network.get("nodes", []):
            fpath = KG_ROOT / node["file"]
            if not fpath.exists():
                missing.append(f"{node['id']} -> {node['file']}")
        assert not missing, f"Missing decision files: {missing}"

    def test_all_decision_files_in_network(
        self, network: dict, all_decision_files: list[Path]
    ) -> None:
        """Every decision YAML file on disk must be registered in _network.yaml."""
        network_files = {
            (KG_ROOT / n["file"]).resolve() for n in network.get("nodes", [])
        }
        orphan = []
        for path in all_decision_files:
            if path.resolve() not in network_files:
                orphan.append(str(path.relative_to(KG_ROOT)))
        assert not orphan, f"Orphan decision files not in _network.yaml: {orphan}"


# ─── Test 2: Schema conformance ──────────────────────────────────────────────


class TestSchemaConformance:
    """Validate all decision files have required fields from _schema.yaml."""

    def test_required_fields_present(
        self, all_decisions: dict[str, dict], schema: dict
    ) -> None:
        """Every decision file must have all required fields."""
        required = schema.get("required_fields", [])
        violations = []
        for decision_id, data in all_decisions.items():
            for field in required:
                if field not in data:
                    violations.append(f"{decision_id}: missing {field}")
        assert not violations, f"Schema violations: {violations}"

    def test_valid_status_values(
        self, all_decisions: dict[str, dict], schema: dict
    ) -> None:
        """Status must be one of the allowed values."""
        valid_statuses = set(schema.get("status_values", {}).keys())
        violations = []
        for decision_id, data in all_decisions.items():
            status = data.get("status", "")
            if status not in valid_statuses:
                violations.append(f"{decision_id}: invalid status '{status}'")
        assert not violations, f"Invalid statuses: {violations}"

    def test_valid_level_values(
        self, all_decisions: dict[str, dict], schema: dict
    ) -> None:
        """Level must be one of the allowed values."""
        valid_levels = set(schema.get("level_values", {}).keys())
        violations = []
        for decision_id, data in all_decisions.items():
            level = data.get("level", "")
            if level not in valid_levels:
                violations.append(f"{decision_id}: invalid level '{level}'")
        assert not violations, f"Invalid levels: {violations}"


# ─── Test 3: DAG acyclicity ──────────────────────────────────────────────────


class TestDAGIntegrity:
    """Validate the decision network is a valid DAG."""

    def test_no_cycles(self, network: dict, network_node_ids: set[str]) -> None:
        """The decision network must be acyclic."""
        adj: dict[str, list[str]] = defaultdict(list)
        for edge in network.get("edges", []):
            adj[edge["from"]].append(edge["to"])

        visited: set[str] = set()
        in_stack: set[str] = set()
        cycle_node = None

        def dfs(node: str) -> bool:
            nonlocal cycle_node
            if node in in_stack:
                cycle_node = node
                return True
            if node in visited:
                return False
            visited.add(node)
            in_stack.add(node)
            for neighbor in adj[node]:
                if dfs(neighbor):
                    return True
            in_stack.remove(node)
            return False

        for nid in network_node_ids:
            if dfs(nid):
                pytest.fail(f"Cycle detected involving node: {cycle_node}")

    def test_edges_reference_valid_nodes(
        self, network: dict, network_node_ids: set[str]
    ) -> None:
        """All edge endpoints must reference nodes in the network."""
        bad = []
        for edge in network.get("edges", []):
            if edge["from"] not in network_node_ids:
                bad.append(f"from={edge['from']}")
            if edge["to"] not in network_node_ids:
                bad.append(f"to={edge['to']}")
        assert not bad, f"Edges reference non-existent nodes: {bad}"

    def test_propagation_references_valid_nodes(
        self, network: dict, network_node_ids: set[str]
    ) -> None:
        """All propagation edges must reference valid nodes."""
        bad = []
        for prop in network.get("propagation", []):
            if prop["source"] not in network_node_ids:
                bad.append(f"source={prop['source']}")
            if prop["target"] not in network_node_ids:
                bad.append(f"target={prop['target']}")
        assert not bad, f"Propagation references non-existent nodes: {bad}"


# ─── Test 4: Probability invariants ──────────────────────────────────────────


class TestProbabilityInvariants:
    """Validate probability constraints on decision options."""

    def test_prior_probabilities_sum_to_one(
        self, all_decisions: dict[str, dict]
    ) -> None:
        """Prior probabilities must sum to approximately 1.0."""
        violations = []
        for decision_id, data in all_decisions.items():
            options = data.get("options", [])
            if not options:
                continue
            total = sum(opt.get("prior_probability", 0.0) for opt in options)
            if abs(total - 1.0) > 0.05:
                violations.append(f"{decision_id}: prior sum = {total:.3f}")
        assert not violations, f"Prior probability violations: {violations}"

    def test_posterior_probabilities_sum_to_one(
        self, all_decisions: dict[str, dict]
    ) -> None:
        """Posterior probabilities must sum to approximately 1.0 when present."""
        violations = []
        for decision_id, data in all_decisions.items():
            options = data.get("options", [])
            if not options:
                continue
            posteriors = [
                opt.get("posterior_probability")
                for opt in options
                if opt.get("posterior_probability") is not None
            ]
            if not posteriors:
                continue
            total = sum(posteriors)
            if abs(total - 1.0) > 0.05:
                violations.append(f"{decision_id}: posterior sum = {total:.3f}")
        assert not violations, f"Posterior probability violations: {violations}"

    def test_resolved_has_winning_option(self, all_decisions: dict[str, dict]) -> None:
        """Resolved decisions must have a resolved_option field."""
        violations = []
        for decision_id, data in all_decisions.items():
            if data.get("status") == "resolved" and not data.get("resolved_option"):
                violations.append(decision_id)
        assert not violations, (
            f"Resolved decisions without resolved_option: {violations}"
        )

    def test_resolved_option_exists_in_options(
        self, all_decisions: dict[str, dict]
    ) -> None:
        """The resolved_option must match an existing option id."""
        violations = []
        for decision_id, data in all_decisions.items():
            resolved = data.get("resolved_option")
            if not resolved:
                continue
            option_ids = {opt.get("id") for opt in data.get("options", [])}
            if resolved not in option_ids:
                violations.append(
                    f"{decision_id}: resolved_option '{resolved}' not in options {option_ids}"
                )
        assert not violations, f"Invalid resolved_option references: {violations}"


# ─── Test 5: Navigator consistency ────────────────────────────────────────────


class TestNavigatorConsistency:
    """Validate navigator.yaml references valid domains and decisions."""

    def test_all_domains_have_yaml_files(self, navigator: dict) -> None:
        """Every domain in navigator.yaml must have a YAML file."""
        missing = []
        for domain_name, domain_data in navigator.get("domains", {}).items():
            nav_file = domain_data.get("navigator")
            if nav_file and not (REPO_ROOT / nav_file).exists():
                missing.append(f"{domain_name}: {nav_file}")
        assert not missing, f"Missing domain YAML files: {missing}"

    def test_decision_ids_in_navigator_exist(
        self, navigator: dict, network_node_ids: set[str]
    ) -> None:
        """All decision IDs listed in navigator.yaml must exist in _network.yaml."""
        bad = []
        for domain_name, domain_data in navigator.get("domains", {}).items():
            for did in domain_data.get("decisions", []):
                if did not in network_node_ids:
                    bad.append(f"{domain_name}: {did}")
        assert not bad, f"Navigator references non-existent decision IDs: {bad}"


# ─── Test 6: Level ordering ──────────────────────────────────────────────────


class TestLevelOrdering:
    """Validate edges respect level ordering (higher -> lower)."""

    def test_edges_flow_downward(self, network: dict) -> None:
        """Non-skip edges must flow from a higher level to a lower or equal level."""
        node_levels = {n["id"]: n["level"] for n in network.get("nodes", [])}
        level_order = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}

        violations = []
        for edge in network.get("edges", []):
            if edge.get("skip"):
                continue  # Skip connections are exempt
            src_level = level_order.get(node_levels.get(edge["from"], ""), 0)
            tgt_level = level_order.get(node_levels.get(edge["to"], ""), 0)
            if src_level > tgt_level:
                violations.append(
                    f"{edge['from']} (L{src_level}) -> {edge['to']} (L{tgt_level})"
                )
        assert not violations, (
            f"Non-skip edges flow upward (violates level ordering): {violations}"
        )


# ─── Test 7: Domain-decision cross-reference ─────────────────────────────────


class TestDomainDecisionXref:
    """Validate KG domain files cross-reference PRD decision nodes."""

    def test_domain_prd_node_references_exist(self) -> None:
        """Every prd_node reference in domain YAML must point to existing file."""
        bad = []
        for domain_file in DOMAINS_DIR.glob("*.yaml"):
            data = _load_yaml(domain_file)
            decisions = data.get("decisions", {})
            if not isinstance(decisions, dict):
                continue
            for did, ddata in decisions.items():
                if not isinstance(ddata, dict):
                    continue
                prd_node = ddata.get("prd_node")
                if prd_node and not (REPO_ROOT / prd_node).exists():
                    bad.append(f"{domain_file.stem}/{did}: {prd_node}")
        assert not bad, f"Domain prd_node references to non-existent files: {bad}"


# ─── Test 8: Node count sanity ────────────────────────────────────────────────


class TestNodeCountSanity:
    """Validate expected node counts per level."""

    def test_minimum_node_count(self, network: dict) -> None:
        """Network must have at least 60 nodes (current: 65)."""
        count = len(network.get("nodes", []))
        assert count >= 60, f"Expected >=60 nodes, got {count}"

    def test_all_levels_populated(self, network: dict) -> None:
        """All 5 levels (L1-L5) must have at least one node."""
        levels = {n["level"] for n in network.get("nodes", [])}
        for level in ["L1", "L2", "L3", "L4", "L5"]:
            assert level in levels, f"Level {level} has no nodes"

    def test_minimum_edge_count(self, network: dict) -> None:
        """Network must have at least 80 edges."""
        count = len(network.get("edges", []))
        assert count >= 80, f"Expected >=80 edges, got {count}"
