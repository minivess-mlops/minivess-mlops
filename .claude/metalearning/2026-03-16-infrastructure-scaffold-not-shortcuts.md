# Metalearning: Infrastructure Scaffold — Integrate ALL Methods, Never Collapse to "Simple"

**Date**: 2026-03-16
**Severity**: HIGH — fundamental misunderstanding of project purpose
**Pattern**: Claude repeatedly tries to minimize scope by suggesting "simpler" approaches

## The Failure Pattern

When presented with multiple implementation options (e.g., 3 synthetic generators),
Claude offers them as mutually exclusive choices and biases toward the simplest one.
This is exactly wrong for this project.

## Why This Is Wrong

MinIVess MLOps is an **infrastructure scaffold** — a platform that wraps and integrates
standalone approaches into a system with excellent DevEx. The repo is:

1. **NOT our product** — it's an academic paper published for the research community
2. **NOT for our lab only** — external researchers download it and use different methods
3. **NOT constrained to one approach** — flexibility is the POINT (see TOP-1)

## The Correct Approach

When there are 3 candidate methods (VascuSynth, VQ-VAE, VesselVAE):
- **Implement ALL THREE** as adapters behind a common ABC/interface
- **YAML config selects** which one to use: `method: vascusynth` / `method: vqvae` / `method: vesselVAE`
- **End-user API**: `generate_stack(method='vqvae')` — simple, clean, DevEx-first
- **New Prefect Flow** for synthetic generation — same pattern as all other flows

## When to Implement vs. Defer

- **Existing GitHub repo with code + finetuning recipe** → INTEGRATE IT (wrap in adapter)
- **ArXiv paper with NO code** → TOO COMPLEX for this PR, open issue
- **The bar is: does a working implementation exist that we can wrap?**

## The Rule

**NEVER offer "defer" or "simpler option" as the default recommendation.**
**NEVER resist engineering comprehensive solutions — that IS the point of this repo.**
**ALWAYS implement ALL viable options behind config-driven adapters.**
**The phrase "this is too complex" is BANNED unless there's genuinely no existing code to wrap.**

The user is building a scaffold. The scaffold supports many tools. Suggesting to use
only one tool defeats the entire purpose. A scaffold that only holds one thing is not
a scaffold — it's a pole.

## How to Apply

- When evaluating methods: check if GitHub repo exists with working code
- If yes → wrap it in an adapter, add YAML config
- If no → open issue, document the method, defer implementation
- Default to implementing ALL viable options, not choosing one
- The researcher using our code picks their method via config — we provide ALL options
