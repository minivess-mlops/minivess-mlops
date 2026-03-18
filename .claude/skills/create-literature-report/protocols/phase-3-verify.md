# Phase 3: VERIFY — Citation Verification (Zero Tolerance)

**Time budget**: 10 minutes
**Exit criteria**: 0 HALLUCINATED, 0 URL_BROKEN remaining in report

## Step 3.1: Batch Citations

Read the reference list from the report. Split into batches of ~25 citations.

## Step 3.2: Launch Parallel Verification Agents

Spawn 2-3 agents simultaneously using `run_in_background: true`.
Each agent gets the prompt from `prompts/verify-citation.md` with its batch.

**Agent type**: general-purpose (needs WebFetch and WebSearch)

## Step 3.3: Wait for All Agents

Synchronization barrier: ALL verification agents must complete before proceeding.
If an agent takes >10 minutes, check its output file for partial results.

## Step 3.4: Compile Verification Results

Merge all agent results into a unified corrections list:

```json
{
  "verified": 58,
  "title_mismatch": 4,
  "author_mismatch": 2,
  "year_mismatch": 1,
  "url_broken": 0,
  "hallucinated": 3,
  "unverifiable": 2,
  "corrections": [
    {"ref_num": 33, "type": "AUTHOR_MISMATCH", "old": "Fang Y.", "new": "Fang J."},
    {"ref_num": 39, "type": "YEAR_MISMATCH", "old": "2025", "new": "2024"},
    {"ref_num": 57, "type": "AUTHOR_MISMATCH", "old": "Seifrid M.", "new": "Tom G."}
  ],
  "removals": [1, 9, 25]
}
```

## Step 3.5: Apply Corrections

### For HALLUCINATED entries:
1. Delete from reference list (mark with ~~strikethrough~~ and "REMOVED" reason)
2. Search body text for any inline citations to this paper
3. Remove or replace orphaned citations
4. If a claim loses its only citation, either find a replacement or remove the claim

### For TITLE_MISMATCH:
Replace with the actual verified title.

### For AUTHOR_MISMATCH:
Replace with the actual first author.

### For YEAR_MISMATCH:
Replace with the actual publication year.

### For UNVERIFIABLE:
Keep in report but add a note: "(URL verified but content not parseable)"
Only if there is independent evidence the paper exists (e.g., Semantic Scholar entry).

## Step 3.6: Verify Corrections Were Applied

Re-read the reference list. Check:
- [ ] 0 entries marked HALLUCINATED remain
- [ ] 0 entries with TITLE_MISMATCH remain
- [ ] 0 entries with AUTHOR_MISMATCH remain
- [ ] 0 entries with YEAR_MISMATCH remain
- [ ] No orphaned body text citations (claims citing removed papers)
- [ ] Paper count in header matches actual reference count

## Step 3.7: Update State

```json
{
  "phase": "VERIFY",
  "substep": "corrections_applied",
  "verified_count": 64,
  "hallucinated_count": 3,
  "corrections_applied": 6,
  "agents_completed": {
    "verify_batch_1": true,
    "verify_batch_2": true,
    "verify_batch_3": true
  }
}
```

## Step 3.8: CHECKPOINT

Git commit: `fix: citation verification — remove {N} hallucinated, fix {M} errors`

## FORCE_STOP Triggers

- Hallucination rate > 20%: STOP. The research agents may have fabricated papers.
  Re-run Phase 1 with stricter prompts (add "NEVER fabricate" emphasis).
- All verification agents failed: STOP. WebFetch may be rate-limited.
  Wait 5 minutes and retry, or switch to manual verification.
- >50% UNVERIFIABLE: Many paywalled papers. Consider using Semantic Scholar
  API as primary verification instead of WebFetch.

## Verification Quality Metrics

After Phase 3, the state must show:
- `hallucinated_count == 0` (after removals)
- `url_broken_count == 0` (after removals)
- All remaining citations have verdict VERIFIED or UNVERIFIABLE-with-evidence
