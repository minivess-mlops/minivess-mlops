"""T08 — RED phase: KG adapters.yaml + bibliography.yaml update tests."""

from __future__ import annotations

from pathlib import Path


class TestAdaptersYamlUpdated:
    """T08: adapters.yaml mambavesselnet entry must be updated to experimental."""

    def _load_adapters(self) -> dict:
        import yaml

        return yaml.safe_load(
            Path("knowledge-graph/code-structure/adapters.yaml").read_text(
                encoding="utf-8"
            )
        )

    def _find_mambavesselnet(self, data: dict) -> dict | None:
        for adapter in data.get("adapters", []):
            if adapter.get("id") == "mambavesselnet":
                return adapter
        return None

    def test_mambavesselnet_entry_exists(self) -> None:
        data = self._load_adapters()
        node = self._find_mambavesselnet(data)
        assert node is not None

    def test_status_is_experimental(self) -> None:
        data = self._load_adapters()
        node = self._find_mambavesselnet(data)
        assert node is not None
        assert node.get("status") == "experimental"

    def test_file_path_has_no_not_yet_created_comment(self) -> None:
        """The file path must not say 'NOT YET CREATED'."""
        content = Path("knowledge-graph/code-structure/adapters.yaml").read_text(
            encoding="utf-8"
        )
        # Check that NOT YET CREATED comment is gone from mambavesselnet section
        # (The archived entry can still have its own notes)
        lines = content.splitlines()
        in_mambavesselnet = False
        for line in lines:
            if "id: mambavesselnet" in line and "archived" not in line:
                in_mambavesselnet = True
            elif in_mambavesselnet and line.strip().startswith("- id:"):
                break
            if in_mambavesselnet and "NOT YET CREATED" in line:
                raise AssertionError(
                    "mambavesselnet entry still has 'NOT YET CREATED' comment"
                )

    def test_citation_is_tomm_2025(self) -> None:
        data = self._load_adapters()
        node = self._find_mambavesselnet(data)
        assert node is not None
        assert node.get("citation") == "xu_2025_mambavesselnet_plus"


class TestBibliographyUpdated:
    """T08: bibliography.yaml must have xu_2025_mambavesselnet_plus entry."""

    def _load_bibliography(self) -> dict:
        import yaml

        data = yaml.safe_load(
            Path("knowledge-graph/bibliography.yaml").read_text(encoding="utf-8")
        )
        # bibliography.yaml has top-level 'citations:' key
        return data.get("citations", data)

    def test_xu_2025_entry_exists(self) -> None:
        bib = self._load_bibliography()
        assert "xu_2025_mambavesselnet_plus" in bib

    def test_xu_2025_has_doi_url(self) -> None:
        bib = self._load_bibliography()
        entry = bib["xu_2025_mambavesselnet_plus"]
        assert "url" in entry
        assert "doi.org" in entry["url"]

    def test_xu_2025_year_is_2025(self) -> None:
        bib = self._load_bibliography()
        entry = bib["xu_2025_mambavesselnet_plus"]
        assert entry["year"] == 2025

    def test_xu_2025_venue_is_tomm(self) -> None:
        bib = self._load_bibliography()
        entry = bib["xu_2025_mambavesselnet_plus"]
        assert "TOMM" in entry.get("venue", "")
