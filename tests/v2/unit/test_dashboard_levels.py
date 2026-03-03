"""Tests for progressive disclosure dashboard (#284).

Covers:
- Dashboard level definitions (PI, Colleague, Researcher)
- Dashboard config validation
- Content filtering by level
"""

from __future__ import annotations


class TestDashboardLevels:
    """Test dashboard level definitions."""

    def test_level_enum_values(self) -> None:
        from minivess.orchestration.flows.dashboard_config import DashboardLevel

        assert DashboardLevel.PI.value == 1
        assert DashboardLevel.COLLEAGUE.value == 2
        assert DashboardLevel.RESEARCHER.value == 3

    def test_level_names(self) -> None:
        from minivess.orchestration.flows.dashboard_config import DashboardLevel

        assert DashboardLevel.PI.name == "PI"
        assert DashboardLevel.COLLEAGUE.name == "COLLEAGUE"
        assert DashboardLevel.RESEARCHER.name == "RESEARCHER"


class TestDashboardConfig:
    """Test dashboard configuration."""

    def test_config_creation(self) -> None:
        from minivess.orchestration.flows.dashboard_config import DashboardConfig

        config = DashboardConfig(level="PI")
        assert config.level == "PI"

    def test_config_default_level(self) -> None:
        from minivess.orchestration.flows.dashboard_config import DashboardConfig

        config = DashboardConfig()
        assert config.level == "COLLEAGUE"


class TestDashboardContent:
    """Test content filtering by dashboard level."""

    def test_get_sections_pi(self) -> None:
        from minivess.orchestration.flows.dashboard_config import get_sections_for_level

        sections = get_sections_for_level("PI")
        assert "executive_summary" in sections
        # PI level should NOT include per-fold details
        assert "per_fold_details" not in sections

    def test_get_sections_colleague(self) -> None:
        from minivess.orchestration.flows.dashboard_config import get_sections_for_level

        sections = get_sections_for_level("COLLEAGUE")
        assert "executive_summary" in sections
        assert "experiment_comparison" in sections

    def test_get_sections_researcher(self) -> None:
        from minivess.orchestration.flows.dashboard_config import get_sections_for_level

        sections = get_sections_for_level("RESEARCHER")
        assert "executive_summary" in sections
        assert "experiment_comparison" in sections
        assert "per_fold_details" in sections
        assert "grafana_links" in sections
