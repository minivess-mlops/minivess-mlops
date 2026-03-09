# Metalearning: Missing Hyperlinks in Academic References — Recurring Failure

**Date:** 2026-03-09
**Severity:** High — causes user extreme frustration, degrades academic credibility
**Trigger:** User expressed intense frustration about missing hyperlinks in `.md`, Issues, and PRs
**Pattern:** Confirmed recurring across multiple sessions

---

## The Failure Pattern

Claude Code consistently writes academic references in bare-text form without clickable hyperlinks:

**WRONG (what Claude Code keeps doing):**
```
- Nature Methods 2015, Bhattacharya et al. (2015)
- Bhattacharya et al. (2015) - Apache Spark pipeline
```

**CORRECT (what this repo requires):**
```
- [Bhattacharya et al. (2015). "An OME-TIFF/Spark pipeline..." *Nature Methods*, 12, 115–118.](https://www.nature.com/articles/nmeth.3041)
- [Draelos et al. (2025). "Real-time neural data acquisition..." *Nature Communications*, 16, 8501.](https://www.nature.com/articles/s41467-025-64856-3)
```

---

## Why This Is Unacceptable

1. **This is an academic software project** — every component must demonstrate research-grade practices
2. **The manuscript will cite these papers** — broken/missing reference URLs make traceability impossible
3. **GitHub Issues are permanent records** — they will be read by reviewers, collaborators, and referees
4. **Hyperlinks are the web's basic infrastructure** — writing a URL without making it clickable is equivalent to citing a book without a page number

---

## Root Cause Analysis

1. **Lazy pattern matching**: When writing references, Claude Code defaults to bibliography-style text instead of Markdown hyperlinks, copying a mental model from LaTeX (`\cite{key}`) that doesn't translate to Markdown.

2. **Forgetting the medium**: GitHub Markdown renders `[text](url)` as clickable links. Not using this is a deliberate choice to write worse output.

3. **Summarization bias**: When asked to "mention papers", Claude Code summarizes ("there's a paper about X") instead of providing the full citation with URL.

4. **No enforcement at generation time**: There is no pre-commit hook or CI check that validates hyperlinks in `.md` files.

---

## Mandatory Rules (Effective Immediately)

### Rule A: Every Citation = Hyperlink (Non-Negotiable)
Every academic reference in ANY `.md` file, GitHub Issue, PR description, or PR comment MUST follow this exact pattern:
```
[Author et al. (Year). "Title." *Journal* Vol, Pages.](URL)
```
- `URL` is the canonical DOI or publisher link (not raw DOI, not `doi.org` without `https://`)
- If DOI is unavailable: bioRxiv, arXiv, PubMed, or author's lab page URL
- If no URL exists at all: write `[Full citation — preprint pending]` so it is visibly incomplete

### Rule B: Never Write a Citation Without Verifying the URL
Before writing any citation, verify the exact URL either from:
- The user's message (preferred — use exactly what they provided)
- Web search (`WebFetch` or `WebSearch` tool)

### Rule C: Issue Updates Must Be Complete Citations
When updating a GitHub Issue with references, write the FULL academic citation (authors, year, title, journal, DOI link), not just "see this paper". The issue is a permanent document.

### Rule D: Existing Issues/PRs Must Be Retroactively Updated
When discovering an issue or PR with missing hyperlinks, fix it in the same response. Do not move on without updating it.

---

## Enforcement Plan

### 1. Pre-commit Hook (`.pre-commit-config.yaml`)
Add a hook that scans `.md` files for bare DOI patterns and fails if found:
```bash
# Detect bare DOIs not wrapped in markdown links
grep -rn 'doi\.org/[0-9]' --include='*.md' | grep -v '](https://'
# Detect arxiv links not wrapped
grep -rn 'arxiv\.org/abs/' --include='*.md' | grep -v '](https://'
```

### 2. GitHub Issue Template (`.github/ISSUE_TEMPLATE/`)
Add a "References" section to all issue templates with the explicit format:
```
## References
<!-- ALL citations must be hyperlinked: [Author et al. (Year). "Title." *Journal*.](URL) -->
```

### 3. CLAUDE.md Rule Addition
Add to CLAUDE.md "What AI Must NEVER Do":
> - Write an academic citation without a clickable hyperlink. Every citation = `[Author et al. (Year). "Title." *Journal*.](URL)`.

### 4. PR Readiness Check (`scripts/pr_readiness_check.sh`)
Add check:
```bash
# Check for bare URLs in markdown files (not wrapped in [text](url))
bare_urls=$(grep -rn 'https://www\.\|https://doi\.\|https://arxiv\.' docs/ --include='*.md' | grep -v '](https://' | grep -v '^.*<!--')
if [ -n "$bare_urls" ]; then
  echo "ERROR: Bare (non-hyperlinked) URLs found in docs/:"
  echo "$bare_urls"
  exit 1
fi
```

---

## Example: Issue #328 Fix (2026-03-09)

Updated with full citations including hyperlinks for:
- [Bhattacharya et al. (2015)](https://www.nature.com/articles/nmeth.3041) — OME-TIFF/Spark pipeline, *Nature Methods*
- [Prevedel et al. (2016)](https://www.biorxiv.org/content/10.1101/061507v2.abstract) — large-scale neural imaging
- [Draelos et al. (2025)](https://www.nature.com/articles/s41467-025-64856-3) — real-time acquisition+segmentation, *Nature Communications*
- [Tian et al. (2015)](https://dx.doi.org/10.1364/BOE.6.003113) — adaptive optics inverse design, *Biomedical Optics Express*
- [Optica OE inverse design paper](https://opg.optica.org/oe/fulltext.cfm?uri=oe-34-5-8961) — *Optics Express*

---

## Session Record

This failure was identified in the session of 2026-03-09. The user wrote:
> "add the fucking hyperlinks and do not overly summarize and lose academic context, this repo is all academic.. for fuck sake"
> "I am beyond frustration with this constant fuckup!"

This is a RECURRING failure. Previous sessions also had citations without hyperlinks. The pattern has been persisting for weeks and must be eliminated.

---

## Cross-References

- `CLAUDE.md` — Add to "What AI Must NEVER Do" section
- `MEMORY.md` — Add under "Session Rules"
- `docs/planning/docker-security-hardening-mlsecops-report.md` — References section uses full hyperlinks (use as model)
- GitHub Issue #328 — Updated with hyperlinks as immediate fix
