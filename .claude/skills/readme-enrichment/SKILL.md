---
name: readme-enrichment
version: 1.0.0
description: >
  Enrich README.md with verified hyperlinks for all libraries, papers, models,
  and standards. Zero-tolerance hallucination policy — every URL is web-verified
  before insertion. Use when: "update README links", "enrich README", "add
  hyperlinks to README", "link papers in README", "fix missing links".
  Do NOT use for: code implementation (use self-learning-iterative-coder),
  literature reports (use create-literature-report), knowledge graph sync (use kg-sync).
last_updated: 2026-03-25
activation: manual
invocation: /readme-enrichment
metadata:
  category: documentation
  tags: [readme, hyperlinks, citations, verification, documentation]
  relations:
    compose_with: [create-literature-report]
    depend_on: []
    similar_to: []
    belong_to: []
---

# README Enrichment Skill

Systematic enrichment of README.md with verified hyperlinks. Covers 4 categories:

1. **Libraries/Tools** — homepage URLs (e.g., `[Prefect](https://prefect.io/)`)
2. **Papers/Models** — DOI or ArXiv links (e.g., `[SAM3](https://arxiv.org/abs/...)`)
3. **Standards/Regulations** — official pages (e.g., `[IEC 62304](https://...)`)
4. **Issues/Planned** — GitHub issue cross-references (e.g., `[#894](link)`)

## Core Rules

### R1: Zero Hallucination Policy (NON-NEGOTIABLE)
Every URL inserted into the README MUST be verified via `WebFetch` or `WebSearch`
before insertion. "I'm fairly confident this is the URL" is NOT acceptable.
The cost of a broken link in a public README is higher than the cost of verification.

### R2: Bibliography First
Before web-searching for a paper URL, check these local sources:
1. `knowledge-graph/bibliography.yaml` — canonical citation database
2. `/home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/` — literature files

### R3: Link Format Convention
- **Libraries**: `[ToolName](homepage)` — name only, no version
- **Papers**: `[ModelName](DOI_or_ArXiv)` — model/method name as anchor, NOT "Author et al."
  This keeps the README clean: `[SAM3](https://arxiv.org/abs/2408.12166)` not
  `[Ravi et al. (2025)](https://arxiv.org/abs/2408.12166)`
- **Standards**: `[IEC 62304](URL)` — standard number as anchor
- **Issues**: `[#NNN](https://github.com/petteriTeikari/vascadia/issues/NNN)`

### R4: Preserve Existing Correct Links
Never modify a link that is already correct. Only ADD missing links or FIX broken ones.

### R5: Science Issues as Grouped Link
In the "Planned" section, group science-tagged open issues as a bullet with
individual links: `Science backlog: [#894](url), [#895](url), ...`

## 4-Phase Pipeline

### Phase 1: EXTRACT (read-only)
Parse README.md and identify every unlinked reference:
- Tools/libraries mentioned without `[Name](URL)` syntax
- Papers/models cited without DOI/ArXiv
- Standards without official page links
- Planned items without issue cross-refs

### Phase 2: RESOLVE (research)
For each unlinked reference:
1. Check `bibliography.yaml` first
2. Check `sci-llm-writer/biblio/biblio-vascular/` second
3. Web search ONLY if not found locally
4. Record: `{name, url, source, verified}`

### Phase 3: VERIFY (zero-tolerance)
For every resolved URL:
- `WebFetch` the URL and confirm it returns 200 + correct content
- For ArXiv: verify the paper title matches
- For GitHub: verify the repo exists
- For DOI: verify it resolves

### Phase 4: APPLY (edit README)
Apply all verified links to README.md:
- Use Edit tool for surgical replacements
- Preserve existing correct links (R4)
- Add science issue group to Planned section
- Run `ruff` format check after edits

## Anti-Patterns (BANNED)

- **Memory-based URLs**: "I remember PyTorch is at pytorch.org" → VERIFY FIRST
- **Guessed DOIs**: constructing DOIs from patterns → ALWAYS web-verify
- **Batch insertion without verification**: inserting 20 links then checking → verify EACH
- **Removing existing content**: enrichment is ADDITIVE, never destructive
