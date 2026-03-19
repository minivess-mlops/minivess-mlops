"""KG-Enrichment Agent — automated knowledge graph enrichment from PubMed.

Modules:
  - pubmed_search: NCBI E-utilities PubMed search with rate limiting
  - entity_extraction: Pydantic AI structured entity extraction from abstracts
  - contradiction_detector: Detect conflicts between proposals and existing KG
  - update_proposal: Generate and persist human-reviewable KG update proposals
"""

from __future__ import annotations
