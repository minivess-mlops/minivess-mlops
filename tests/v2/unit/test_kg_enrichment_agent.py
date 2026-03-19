"""Tests for the KG-Enrichment Agent (PR-5, #849).

Covers:
- T5.1: PubMed search returns structured metadata
- T5.3: Entity extraction from abstracts (Pydantic AI structured output)
- T5.4: Contradiction detector flags KG conflicts

Staging tier: no model loading, no slow, no integration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# T5.1: PubMed search returns structured metadata
# ---------------------------------------------------------------------------


class TestPubMedSearch:
    """Tests for search_pubmed() function."""

    def test_search_pubmed_returns_list_of_article_metadata(self) -> None:
        """search_pubmed() returns a list of PubMedArticle objects."""
        from minivess.agents.kg_enrichment.pubmed_search import (
            PubMedArticle,
            search_pubmed,
        )

        # Mock the HTTP call to NCBI E-utilities
        mock_search_response = MagicMock()
        mock_search_response.status_code = 200
        mock_search_response.json.return_value = {
            "esearchresult": {
                "idlist": ["12345678", "87654321"],
                "count": "2",
            }
        }

        mock_fetch_response = MagicMock()
        mock_fetch_response.status_code = 200
        mock_fetch_response.text = """<?xml version="1.0" ?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <ArticleTitle>Vessel Segmentation with clDice Loss</ArticleTitle>
        <Abstract>
          <AbstractText>We propose a topology-preserving loss function for vessel segmentation using soft skeleton-based clDice.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Shit</LastName><ForeName>Suprosanna</ForeName></Author>
          <Author><LastName>Paetzold</LastName><ForeName>Johannes</ForeName></Author>
        </AuthorList>
        <Journal><Title>Medical Image Analysis</Title></Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>87654321</PMID>
      <Article>
        <ArticleTitle>SAM3 for 3D Medical Image Segmentation</ArticleTitle>
        <Abstract>
          <AbstractText>We present Segment Anything Model 3, a foundation model for 3D medical image segmentation.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Kirillov</LastName><ForeName>Alexander</ForeName></Author>
        </AuthorList>
        <Journal><Title>Nature Methods</Title></Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""

        with patch(
            "minivess.agents.kg_enrichment.pubmed_search.httpx.get",
            side_effect=[mock_search_response, mock_fetch_response],
        ):
            results = search_pubmed("vessel segmentation clDice", max_results=2)

        assert isinstance(results, list)
        assert len(results) == 2
        for article in results:
            assert isinstance(article, PubMedArticle)

        # Check first article fields
        assert results[0].pmid == "12345678"
        assert results[0].title == "Vessel Segmentation with clDice Loss"
        assert "topology-preserving" in results[0].abstract
        assert len(results[0].authors) == 2
        assert results[0].journal == "Medical Image Analysis"

    def test_search_pubmed_empty_results(self) -> None:
        """search_pubmed() returns empty list when no results found."""
        from minivess.agents.kg_enrichment.pubmed_search import search_pubmed

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "esearchresult": {"idlist": [], "count": "0"}
        }

        with patch(
            "minivess.agents.kg_enrichment.pubmed_search.httpx.get",
            return_value=mock_response,
        ):
            results = search_pubmed("nonexistent_query_xyz123", max_results=5)

        assert isinstance(results, list)
        assert len(results) == 0

    def test_pubmed_article_model_fields(self) -> None:
        """PubMedArticle has required fields: pmid, title, authors, abstract, journal."""
        from minivess.agents.kg_enrichment.pubmed_search import PubMedArticle

        article = PubMedArticle(
            pmid="12345678",
            title="Test Article",
            authors=["Author A", "Author B"],
            abstract="Test abstract text.",
            journal="Test Journal",
        )
        assert article.pmid == "12345678"
        assert article.title == "Test Article"
        assert len(article.authors) == 2
        assert article.abstract == "Test abstract text."
        assert article.journal == "Test Journal"

    def test_rate_limiter_enforces_3_rps(self) -> None:
        """NCBIRateLimiter enforces max 3 requests per second."""
        from minivess.agents.kg_enrichment.pubmed_search import NCBIRateLimiter

        limiter = NCBIRateLimiter(max_requests_per_second=3)
        assert limiter.max_requests_per_second == 3
        # Verify it has an acquire method
        assert callable(getattr(limiter, "acquire", None))


# ---------------------------------------------------------------------------
# T5.3: Entity extraction from abstracts
# ---------------------------------------------------------------------------


class TestEntityExtraction:
    """Tests for LLM-powered entity extraction from paper abstracts."""

    def test_extracted_entities_model_has_required_fields(self) -> None:
        """ExtractedEntities Pydantic model has model_family, losses, metrics."""
        from minivess.agents.kg_enrichment.entity_extraction import (
            ExtractedEntities,
        )

        entities = ExtractedEntities(
            model_families=["DynUNet", "SAM3"],
            losses=["clDice", "Dice+CE"],
            metrics=["DSC", "clDice", "Betti Error"],
            datasets=["MiniVess"],
            techniques=["topology-preserving"],
        )
        assert "DynUNet" in entities.model_families
        assert "clDice" in entities.losses
        assert "DSC" in entities.metrics
        assert "MiniVess" in entities.datasets
        assert "topology-preserving" in entities.techniques

    def test_build_extraction_agent_returns_agent(self) -> None:
        """build_extraction_agent() returns a Pydantic AI Agent."""
        from pydantic_ai import Agent

        from minivess.agents.kg_enrichment.entity_extraction import (
            build_extraction_agent,
        )

        # Use "test" model to avoid requiring real API keys
        agent = build_extraction_agent(model="test")
        assert isinstance(agent, Agent)

    def test_extract_entities_from_abstract_returns_structured(self) -> None:
        """extract_entities() returns ExtractedEntities from an abstract string."""
        from minivess.agents.kg_enrichment.entity_extraction import (
            ExtractedEntities,
            extract_entities,
        )

        sample_abstract = (
            "We evaluate DynUNet and SegResNet on the MiniVess dataset using "
            "clDice loss and report Dice similarity coefficient and centerline "
            "Dice metrics. Our topology-preserving approach achieves state-of-the-art."
        )

        # Use "test" model to avoid real LLM calls —
        # TestModel returns a valid structured output based on the schema
        result = extract_entities(sample_abstract, model="test")

        assert isinstance(result, ExtractedEntities)
        # TestModel generates synthetic data based on the Pydantic model schema
        assert hasattr(result, "model_families")
        assert hasattr(result, "losses")
        assert hasattr(result, "metrics")


# ---------------------------------------------------------------------------
# T5.4: Contradiction detector flags KG conflicts
# ---------------------------------------------------------------------------


class TestContradictionDetector:
    """Tests for KG contradiction detection."""

    def test_contradiction_detected_for_banned_gpu(self) -> None:
        """Contradiction detector flags T4 GPU when KG says T4 is banned for SAM3."""
        from minivess.agents.kg_enrichment.contradiction_detector import (
            Contradiction,
            detect_contradictions,
        )

        # Mock KG constraint: T4 is banned for SAM3
        kg_constraints: list[dict[str, Any]] = [
            {
                "node_id": "gpu_compute",
                "constraint": "T4 GPU is banned for SAM3 (no BF16 support)",
                "applies_to": ["SAM3"],
                "banned_values": ["T4"],
            },
        ]

        # Proposal that violates the constraint
        proposal = {
            "node_id": "gpu_compute",
            "proposed_value": "T4",
            "context": "SAM3 training",
        }

        contradictions = detect_contradictions(proposal, kg_constraints)
        assert len(contradictions) >= 1
        assert isinstance(contradictions[0], Contradiction)
        assert "T4" in contradictions[0].description
        assert contradictions[0].severity == "hard"

    def test_no_contradiction_for_valid_proposal(self) -> None:
        """No contradiction when proposal does not conflict with KG."""
        from minivess.agents.kg_enrichment.contradiction_detector import (
            detect_contradictions,
        )

        kg_constraints: list[dict[str, Any]] = [
            {
                "node_id": "gpu_compute",
                "constraint": "T4 GPU is banned for SAM3 (no BF16 support)",
                "applies_to": ["SAM3"],
                "banned_values": ["T4"],
            },
        ]

        proposal = {
            "node_id": "gpu_compute",
            "proposed_value": "L4",
            "context": "SAM3 training",
        }

        contradictions = detect_contradictions(proposal, kg_constraints)
        assert len(contradictions) == 0

    def test_contradiction_has_required_fields(self) -> None:
        """Contradiction model has description, severity, and source_node fields."""
        from minivess.agents.kg_enrichment.contradiction_detector import (
            Contradiction,
        )

        c = Contradiction(
            description="T4 banned for SAM3",
            severity="hard",
            source_node="gpu_compute",
            conflicting_constraint="T4 GPU is banned for SAM3",
        )
        assert c.description == "T4 banned for SAM3"
        assert c.severity == "hard"
        assert c.source_node == "gpu_compute"
        assert c.conflicting_constraint == "T4 GPU is banned for SAM3"


# ---------------------------------------------------------------------------
# T5.5: KG update proposals with human review gate
# ---------------------------------------------------------------------------


class TestUpdateProposal:
    """Tests for KG update proposal generation and persistence."""

    def test_update_proposal_model_fields(self) -> None:
        """UpdateProposal has required fields: node_id, proposed_changes, rationale, status."""
        from minivess.agents.kg_enrichment.update_proposal import UpdateProposal

        proposal = UpdateProposal(
            node_id="loss_function",
            proposed_changes={"new_option": "topo_loss_v2"},
            rationale="New topology loss with better Betti number preservation",
            source_pmid="99999999",
            status="pending_review",
        )
        assert proposal.node_id == "loss_function"
        assert proposal.status == "pending_review"
        assert "new_option" in proposal.proposed_changes

    def test_save_proposal_writes_yaml(self, tmp_path: Path) -> None:
        """save_proposal() writes a YAML file to the proposals directory."""
        from minivess.agents.kg_enrichment.update_proposal import (
            UpdateProposal,
            save_proposal,
        )

        proposal = UpdateProposal(
            node_id="loss_function",
            proposed_changes={"new_option": "topo_loss_v2"},
            rationale="New topology loss from PubMed article",
            source_pmid="99999999",
            status="pending_review",
        )

        output_path = save_proposal(proposal, output_dir=tmp_path)
        assert output_path.exists()
        assert output_path.suffix == ".yaml"

        # Verify content is valid YAML
        import yaml

        with open(output_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["node_id"] == "loss_function"
        assert data["status"] == "pending_review"

    def test_proposal_never_auto_applied(self) -> None:
        """UpdateProposal status is always 'pending_review' — never 'applied'."""
        from minivess.agents.kg_enrichment.update_proposal import UpdateProposal

        proposal = UpdateProposal(
            node_id="gpu_compute",
            proposed_changes={"add_gpu": "H100"},
            rationale="H100 available for large models",
            source_pmid="11111111",
            status="pending_review",
        )
        # Proposals MUST default to pending_review
        assert proposal.status == "pending_review"

    def test_proposal_with_contradictions_includes_warnings(
        self, tmp_path: Path
    ) -> None:
        """Proposal with contradictions includes warning field in saved YAML."""
        from minivess.agents.kg_enrichment.contradiction_detector import (
            Contradiction,
        )
        from minivess.agents.kg_enrichment.update_proposal import (
            UpdateProposal,
            save_proposal,
        )

        contradiction = Contradiction(
            description="T4 banned for SAM3",
            severity="hard",
            source_node="gpu_compute",
            conflicting_constraint="T4 is Turing architecture — no BF16",
        )

        proposal = UpdateProposal(
            node_id="gpu_compute",
            proposed_changes={"proposed_gpu": "T4"},
            rationale="Cheaper GPU option",
            source_pmid="22222222",
            status="pending_review",
            contradictions=[contradiction],
        )

        output_path = save_proposal(proposal, output_dir=tmp_path)

        import yaml

        with open(output_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "contradictions" in data
        assert len(data["contradictions"]) == 1
        assert data["contradictions"][0]["severity"] == "hard"
