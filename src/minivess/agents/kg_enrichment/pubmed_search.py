"""PubMed search via NCBI E-utilities with rate limiting.

NCBI policy: max 3 requests/second without API key, 10 req/s with API key.
We enforce 3 req/s as the safe default.

References:
  - https://www.ncbi.nlm.nih.gov/books/NBK25497/
  - https://www.ncbi.nlm.nih.gov/books/NBK25499/
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET  # noqa: N817 — stdlib XML parser, not regex
from typing import Any

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# NCBI E-utilities base URLs
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


class PubMedArticle(BaseModel):
    """Structured metadata for a PubMed article."""

    pmid: str = Field(description="PubMed ID")
    title: str = Field(description="Article title")
    authors: list[str] = Field(default_factory=list, description="Author names")
    abstract: str = Field(default="", description="Abstract text")
    journal: str = Field(default="", description="Journal name")


class NCBIRateLimiter:
    """Rate limiter for NCBI E-utilities (3 req/s default).

    NCBI policy requires max 3 requests/second for unauthenticated access.
    This limiter uses a simple token bucket approach.
    """

    def __init__(self, max_requests_per_second: int = 3) -> None:
        self.max_requests_per_second = max_requests_per_second
        self._min_interval = 1.0 / max_requests_per_second
        self._last_request_time: float = 0.0

    def acquire(self) -> None:
        """Block until a request slot is available."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.monotonic()


# Module-level rate limiter instance
_rate_limiter = NCBIRateLimiter(max_requests_per_second=3)


def _parse_articles_xml(xml_text: str) -> list[PubMedArticle]:
    """Parse PubMed XML efetch response into PubMedArticle objects.

    Uses stdlib xml.etree.ElementTree (NOT regex — Rule #16).
    """
    articles: list[PubMedArticle] = []

    root = ET.fromstring(xml_text)
    for article_elem in root.findall(".//PubmedArticle"):
        citation = article_elem.find(".//MedlineCitation")
        if citation is None:
            continue

        pmid_elem = citation.find("PMID")
        pmid = pmid_elem.text if pmid_elem is not None and pmid_elem.text else ""

        article_data = citation.find("Article")
        if article_data is None:
            continue

        title_elem = article_data.find("ArticleTitle")
        title = title_elem.text if title_elem is not None and title_elem.text else ""

        abstract_elem = article_data.find(".//AbstractText")
        abstract = (
            abstract_elem.text
            if abstract_elem is not None and abstract_elem.text
            else ""
        )

        journal_elem = article_data.find(".//Journal/Title")
        journal = (
            journal_elem.text if journal_elem is not None and journal_elem.text else ""
        )

        authors: list[str] = []
        author_list = article_data.find("AuthorList")
        if author_list is not None:
            for author in author_list.findall("Author"):
                last = author.find("LastName")
                fore = author.find("ForeName")
                name_parts = []
                if fore is not None and fore.text:
                    name_parts.append(fore.text)
                if last is not None and last.text:
                    name_parts.append(last.text)
                if name_parts:
                    authors.append(" ".join(name_parts))

        articles.append(
            PubMedArticle(
                pmid=pmid,
                title=title,
                authors=authors,
                abstract=abstract,
                journal=journal,
            )
        )

    return articles


def search_pubmed(
    query: str,
    *,
    max_results: int = 10,
    rate_limiter: NCBIRateLimiter | None = None,
) -> list[PubMedArticle]:
    """Search PubMed and return structured article metadata.

    Parameters
    ----------
    query:
        PubMed search query string.
    max_results:
        Maximum number of results to return.
    rate_limiter:
        Optional rate limiter override (uses module default if None).

    Returns
    -------
    List of PubMedArticle objects with structured metadata.
    """
    limiter = rate_limiter or _rate_limiter

    # Step 1: ESearch to get PMIDs
    limiter.acquire()
    search_response = httpx.get(
        ESEARCH_URL,
        params={
            "db": "pubmed",
            "term": query,
            "retmax": str(max_results),
            "retmode": "json",
        },
        timeout=30.0,
    )
    search_response.raise_for_status()

    search_data: dict[str, Any] = search_response.json()
    pmids: list[str] = search_data.get("esearchresult", {}).get("idlist", [])

    if not pmids:
        logger.info("PubMed search returned 0 results for query: %s", query)
        return []

    # Step 2: EFetch to get full article metadata
    limiter.acquire()
    fetch_response = httpx.get(
        EFETCH_URL,
        params={
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "xml",
            "retmode": "xml",
        },
        timeout=30.0,
    )
    fetch_response.raise_for_status()

    articles = _parse_articles_xml(fetch_response.text)
    logger.info(
        "PubMed search returned %d articles for query: %s",
        len(articles),
        query,
    )
    return articles
