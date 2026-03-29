# Metalearning: Shortcut-Taking — Proposing Quick Hacks Over Production-Grade Solutions

**Date**: 2026-03-28
**Severity**: HIGH — violates Rule 32 (Quality Over Speed) and design principles
**Session**: 11th pass experiment harness execution

## What Happened

When DeepVess data was needed on GCS, Claude proposed TWO options:
1. "Just download fast and push" (quick hack)
2. "Implement automated downloader" (production-grade)

Claude RECOMMENDED option 1 ("Recommended") — the fast shortcut — instead of
the production-grade solution. The user chose option 2 (implement the downloader).

This is a recurring anti-pattern: Claude defaults to "get it done fast" over
"get it done right," even though:
- CLAUDE.md Rule 32 says "Quality Over Speed — No Reactive Rushing"
- The project is v0.2-beta targeting Nature Protocols publication
- The acquisition flow was DESIGNED to have automated downloaders
- The codebase already had the infrastructure (downloaders.py, acquisition_registry.py,
  format_conversion.py) — only the DeepVess function was missing

## Why Claude Proposed the Shortcut

1. **Time pressure bias**: The experiment was "ready to launch" and DeepVess was
   blocking. Claude's optimization function was "minimize time to launch" not
   "maximize platform quality."

2. **False separation**: Claude treated "get the data" and "implement the downloader"
   as independent tasks when they should be ONE task. The correct workflow:
   implement the downloader → use the downloader → data arrives. No shortcut needed.

3. **Devaluing infrastructure**: Claude treated the downloader as "nice to have"
   instead of "part of the platform being demonstrated." This is a PLATFORM PAPER.
   The acquisition flow automation IS the contribution. Manual data download
   undermines the platform's value proposition.

4. **LLM training data bias**: Most LLM training examples show "just curl the file"
   as the answer to data download problems. The production-grade approach (registry,
   downloader dispatch, format conversion pipeline) requires more tokens and feels
   "over-engineered" to an LLM. But in an MLOps platform, it IS the product.

## Prevention Rules

1. **NEVER recommend "quick hack" as the default option.** If a production-grade
   solution exists in the codebase, implement it. The quick hack should be the
   LAST resort, not the FIRST suggestion.

2. **"Just get it done fast" is BANNED as a recommendation.** Rule 32 exists
   specifically to prevent this. Quality > Speed. Always.

3. **Check if infrastructure exists before proposing workarounds.** The downloaders.py
   module, acquisition_registry.py, and format_conversion.py were all there. The
   correct answer was "implement the missing download_deepvess() function" — not
   "curl the ZIP file manually."

4. **The platform IS the product.** Every manual step is a platform failure. If a
   researcher has to manually download data, the platform has failed at its mission
   (zero manual work, excellent DevEx).

## Connected Patterns

- Deferring DeepVess entirely (same session) → metalearning/2026-03-28-context-amnesia-deferred-deepvess-whac-a-mole.md
- Reactive rushing → metalearning/2026-03-24-reactive-rushing-instead-of-proactive-quality.md
- Quality over speed → CLAUDE.md Rule 32

## Resolution

User chose option 2. Claude implemented download_deepvess() in downloaders.py,
registered it in _DOWNLOADERS dict, updated acquisition_registry.py from
download_method="manual" to "http_download", wrote 3 tests (all passing),
downloaded the data, and pushed to GCS. Total time: ~15 min.

The "quick hack" would have saved ~5 min but left the platform with a permanent
manual download gap.
