# Metalearning: Citation Hallucination in Literature Research Report

**Date:** 2026-03-17
**Severity:** P0 — Academic integrity violation (7 errors in 38 citations)
**Predecessor:** 16 citation failure docs in sci-llm-writer/.claude/docs/meta-learnings/

---

## What Happened

1. Claude generated a missing citations HTML report with 38 entries
2. Used Google Scholar search URLs as a "safe" shortcut — LAZY, not safe
3. Guessed DOI links from memory (e.g., `doi.org/10.1038/533452a`) without verification
4. Guessed author names from domain reports without verifying against actual papers
5. Fabricated paper #38 entirely ("OLIF audit metric" by Marino D. & Lane M.)

## 7 Errors Found by Verification Agent

| # | Error Type | What Was Wrong |
|---|-----------|---------------|
| 27 | Wrong author | "Protschky et al." → actually Leest et al. |
| 30 | Wrong title | "ComplOps: Compliance-as-Code" → actually "TechOps: Technical Documentation Templates for the AI Act" |
| 34 | Wrong author initial | "Luo S." → actually Yuyu Luo |
| 35 | Wrong author | "Carbonaro et al." → actually Marfoglia, Jhee, Coulet |
| 36 | Wrong author | "Batista et al." → actually Bill Marino et al. |
| 37 | Wrong author | "Warnett et al." → actually Biswas, Bhatt, Vaidhyanathan |
| 38 | FABRICATED | "OLIF audit metric" does not exist. Conflation of real authors from different paper |

## Root Cause

**Exactly the same failure as the 16 prior metalearning docs:**
1. Pattern-matching from memory instead of tool-verified lookup
2. Trusting domain research report summaries as ground truth (they had errors too)
3. Using search URLs to avoid the work of finding real paper pages
4. DOI guessing — DOIs from memory are ALWAYS suspect

## Rule (Non-Negotiable)

1. **NEVER guess DOIs.** Every DOI must be verified via WebSearch/WebFetch to confirm it resolves to the correct paper.
2. **NEVER trust author names from secondary sources** (domain reports, summaries). Verify against the actual paper.
3. **NEVER use Google Scholar search URLs** as paper links. Every link must go DIRECTLY to the paper (publisher page, arXiv, PubMed).
4. **If you cannot find a verified URL, the citation must be marked "CANNOT VERIFY — REMOVE"** — never include unverified entries.
5. **Every citation HTML report must be verified by a separate agent** that web-searches each entry before the report is delivered.

## The Deeper Pattern

This is the 17th citation-related failure across two repos. The pattern:
- Claude is confident about papers it "knows" from training data
- Training data DOIs, author lists, and titles are frequently wrong (especially for 2024-2026 papers)
- The correct behavior is to treat ALL citation metadata as "probably wrong until verified"
- Google Scholar search links are a cop-out, not a solution

---

## Checklist

- [x] Metalearning doc written (this file)
- [ ] HTML report regenerated with ONLY verified URLs from web search
- [ ] Literature report v3.0 author/title corrections applied
