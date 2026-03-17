"""Tests for Grafana infrastructure-cost.json dashboard (issue #747).

Validates JSON structure, panel count, metric names, and tags.
"""

from __future__ import annotations

import json
from pathlib import Path

DASHBOARD_PATH = (
    Path(__file__).resolve().parents[4]
    / "deployment"
    / "grafana"
    / "dashboards"
    / "infrastructure-cost.json"
)


class TestDashboardStructure:
    """Validate Grafana dashboard JSON structure."""

    def test_dashboard_json_is_valid(self) -> None:
        """File loads as valid JSON with expected top-level structure."""
        text = DASHBOARD_PATH.read_text(encoding="utf-8")
        data = json.loads(text)
        assert "dashboard" in data
        assert "panels" in data["dashboard"]
        assert isinstance(data["dashboard"]["panels"], list)
        assert data["dashboard"]["uid"] == "minivess-infrastructure-cost"

    def test_dashboard_has_required_panels(self) -> None:
        """At least 6 panels exist with expected keyword-containing titles."""
        text = DASHBOARD_PATH.read_text(encoding="utf-8")
        data = json.loads(text)
        panels = data["dashboard"]["panels"]
        assert len(panels) >= 6

        # Check panel titles cover the required metric areas
        titles = [p["title"].lower() for p in panels]
        all_titles = " ".join(titles)
        required_keywords = [
            "cost",
            "gpu utilization",
            "setup",
            "rate",
            "estimated",
        ]
        for kw in required_keywords:
            assert kw in all_titles, f"No panel title contains '{kw}'"

    def test_panel_queries_reference_correct_metrics(self) -> None:
        """Every panel target expr references a minivess_training_ metric."""
        text = DASHBOARD_PATH.read_text(encoding="utf-8")
        data = json.loads(text)
        panels = data["dashboard"]["panels"]

        for panel in panels:
            for target in panel.get("targets", []):
                expr = target.get("expr", "")
                assert "minivess_training_" in expr, (
                    f"Panel '{panel['title']}' target does not reference "
                    f"a minivess_training_ metric: {expr}"
                )

    def test_dashboard_tags_include_finops(self) -> None:
        """Dashboard tags include both 'minivess' and 'finops'."""
        text = DASHBOARD_PATH.read_text(encoding="utf-8")
        data = json.loads(text)
        tags = data["dashboard"]["tags"]
        assert "minivess" in tags
        assert "finops" in tags
