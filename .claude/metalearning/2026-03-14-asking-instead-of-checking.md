# Metafailure: Asking the User Instead of Checking Programmatically

**Date**: 2026-03-14
**Severity**: P0 — Direct violation of TOP-2 (Zero Manual Work)
**Rule violated**: TOP-2 (Automate everything), Design Goal #1 (DevEx)

## The Anti-Pattern

When Claude needs to know infrastructure state (is the server running? is data pushed?
is the image in the registry?), Claude ASKS THE USER instead of running commands to
check. This is the antithesis of automation.

## Examples from This Session

| Question Claude asked | What Claude should have done |
|----------------------|------------------------------|
| "Is UpCloud infrastructure running?" | `cd deployment/pulumi && pulumi stack output` |
| "Has DVC data been pushed to UpCloud?" | `dvc status -r upcloud` or `aws s3 ls` with UpCloud endpoint |
| "Has Docker image been pushed to GHCR?" | `docker manifest inspect ghcr.io/petteriteikari/minivess-base:latest` or `skopeo inspect` |
| "Is MLflow server accessible?" | `curl -s -o /dev/null -w '%{http_code}' http://<ip>:5000/health` |

## Why This Is Especially Egregious

The user's TOP-2 principle literally says: **"Nobody should ever manually launch pods,
VMs, or instances."** And: **"Automate everything."**

Asking the user to manually check infrastructure status violates the same principle.
If the user wanted to manually check, they wouldn't need Claude.

## Root Cause

Claude treats information gathering as a conversation task rather than a systems task.
Instead of running diagnostic commands (which are available via Bash tool), Claude
defaults to the social interaction pattern of asking.

## Corrective Protocol

1. **NEVER ask the user about infrastructure state** — run the command to check
2. **Build diagnostic checks into the workflow** — before any deployment step,
   verify prerequisites programmatically
3. **Report findings, don't ask questions** — "MLflow server is DOWN at <ip>:5000,
   need to run pulumi up" is better than "Is MLflow running?"
4. **If a check command doesn't exist, create one** — write a preflight script
   rather than asking the user

## Connection to make smoke-test-preflight

The `make smoke-test-preflight` target and `scripts/validate_smoke_test_env.py` already
exist for this EXACT purpose. Claude should have run this instead of asking.
