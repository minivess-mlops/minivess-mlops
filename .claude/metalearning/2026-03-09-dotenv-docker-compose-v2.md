# 2026-03-09 — .env Not Loaded by Docker Compose V2 from Working Directory

## What Happened

HF_TOKEN was set in `.env` at repo root. Training SAM3 models failed with:
```
RuntimeError: HuggingFace token required to download 'facebook/sam3'.
```

Claude incorrectly labeled this "expected (SAM3 needs HF_TOKEN)". This was WRONG.
HF_TOKEN **was** in `.env`. Claude failed to diagnose that Docker Compose wasn't loading it.

## Root Cause

**Docker Compose V2 (v2.x) loads `.env` from the compose file's project directory, NOT the working directory.**

When running:
```bash
docker compose -f deployment/docker-compose.flows.yml run train
```
Docker Compose looks for `.env` in `deployment/` (compose file dir), not in the repo root.

The `.env` at repo root → INVISIBLE to Docker Compose V2 by default.

## Fix

Explicitly pass `--env-file` to every `docker compose` invocation:
```bash
docker compose --env-file /path/to/repo/.env -f deployment/docker-compose.flows.yml run train
```

In `run_debug.sh`:
```bash
ENV_FILE_ARG=""
if [ -f "$REPO_ROOT/.env" ]; then
  ENV_FILE_ARG="--env-file $REPO_ROOT/.env"
fi
# Then: docker compose $ENV_FILE_ARG -f "$FLOWS_COMPOSE" run ...
```

## Required Guardrails (MUST IMPLEMENT)

1. **`deployment/CLAUDE.md`**: Document Docker Compose V2 `.env` behavior — always pass `--env-file`.
2. **`scripts/run_debug.sh`**: Always use `--env-file $REPO_ROOT/.env` ✓ (fixed in commit).
3. **`hf_auth.py`**: The `require_hf_token()` error should:
   - Log at `loguru.error` level (not just raise RuntimeError)
   - Tell the user WHERE to set HF_TOKEN (`.env` file, not shell env var)
   - Never say "set HF_TOKEN environment variable" — always say "set HF_TOKEN in .env"

## This Repo Uses .env, Not System Environment Variables

**ABSOLUTE RULE: ALL secrets and configuration go in `.env`. Never in shell `export VAR=...`.**

- `.env` is the single source of secrets for this project
- `docker compose --env-file .env` is how secrets reach containers
- Telling users to `export HF_TOKEN` is WRONG — they should add it to `.env`
- The error message in `hf_auth.py` must say "Set HF_TOKEN in your `.env` file" not "set the HF_TOKEN environment variable"

## Pattern to Detect Future Violations

If you see code that:
1. Uses `os.environ.get("HF_TOKEN")` without fallback advice pointing to `.env` → ADD `.env` instruction to error
2. Documentation saying `export HF_TOKEN=...` → CHANGE TO `echo "HF_TOKEN=..." >> .env`
3. `docker compose -f X run Y` WITHOUT `--env-file` → ADD `--env-file .env`
