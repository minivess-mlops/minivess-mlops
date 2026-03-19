# Eval Checklist — Fetch Docs

Tier C skill. Binary pass/fail criteria.

## Structural Criteria

1. **Max 3 chub calls**
   - The session used 3 or fewer `chub` CLI invocations.
   - Pass: ≤3 chub calls.
   - Fail: >3 chub calls (token budget exceeded).

2. **Local registry checked first**
   - The first `chub` call used `--source minivess` (local custom registry).
   - Pass: Local checked before community/web.
   - Fail: Jumped straight to community or web search.

3. **Correct library returned**
   - The fetched docs match the library the user asked about.
   - Pass: Docs are for the requested library.
   - Fail: Wrong library docs returned.

## Trigger Tests

**Should trigger:**
- "fetch docs for MONAI CacheDataset"
- "get documentation for skypilot task YAML"
- "what's the API for torch.nn.functional.grid_sample?"

**Should NOT trigger:**
- "create a GitHub issue for this bug"
- "run the training pipeline"
