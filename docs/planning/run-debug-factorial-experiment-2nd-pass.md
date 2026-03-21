# Cold-Start Prompt: Debug Factorial Run — 2nd Pass

## Date: 2026-03-20
## Branch: test/debug-factorial-run (continuing from 1st pass)
## Execution Mode: INTERACTIVE (user available for decisions)
## Predecessor: docs/planning/run-debug-factorial-experiment-report.md (1st pass report)

---

## To run in a new session:

```bash
claude -p "Read and execute the plan at:
/home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/run-debug-factorial-experiment-2nd-pass.md
Branch: test/debug-factorial-run." --dangerously-skip-permissions
```

---

## PURPOSE

The 1st pass of the debug factorial run (2026-03-20) launched 26 conditions.
**12 SUCCEEDED, 14 FAILED.** This 2nd pass re-runs ONLY the failed conditions
after fixing the root causes, plus 1 checkpoint-persistence validation probe.

**This is NOT a production run.** The purpose is to validate pipeline mechanics:
- Can all 6 model families construct, train, and save checkpoints?
- Does the checkpoint persistence path work end-to-end?
- Does the zero-shot evaluation path work?

Production results go to `sci-llm-writer/manuscripts/vasculature-mlops`.
After this 2nd pass, a 3rd pass with all 26 conditions is planned.

---

## MANDATORY READING BEFORE ANY ACTION

1. `docs/planning/run-debug-factorial-experiment-report.md` — 1st pass report (12 glitches)
2. `docs/planning/run-debug-factorial-experiment-report-2nd-pass-fix-plan.xml` — Fix plan
3. `CLAUDE.md` — All rules
4. `.claude/skills/factorial-monitor/SKILL.md` — Monitoring skill

---

## WHAT FAILED IN 1ST PASS AND WHY

| Conditions | Count | Root Cause | Glitch # | Fix Required |
|-----------|-------|------------|----------|-------------|
| sam3_topolora × 3 losses × 2 calib | 6 | LoRA applied to Conv2d | #9 | Skip Conv2d in `_apply_lora_to_encoder()` |
| mambavesselnet × 3 losses × 2 calib | 6 | mamba-ssm not compiled | #10 | Rebuild Docker with `INSTALL_MAMBA=1` |
| sam3_vanilla zero-shot | 1 | max_epochs=0 validation | #12 | Allow ge=0 in TrainingConfig |
| vesselfm zero-shot | 1 | max_epochs=0 validation | #12 | Allow ge=0 in TrainingConfig |
| **Checkpoint persistence (P0)** | **ALL** | Cloud Run 32MB limit | **#8** | `--no-serve-artifacts` + GCS mount |

---

## PRE-REQUISITES (must be done BEFORE this 2nd pass)

All fixes from the fix-plan XML must be implemented and verified:

### Fix 1: P0 #878 — MLflow checkpoint persistence (Glitch #8)
```yaml
# In deployment/skypilot/train_factorial.yaml:
# CHANGE line 57:
MLFLOW_TRACKING_URI: ${MLFLOW_TRACKING_URI}  # was: /app/mlruns

# ADD file_mounts section:
file_mounts:
  /app/checkpoints:
    source: gs://minivess-mlops-checkpoints
    mode: MOUNT

# In scripts/run_factorial.sh:
# REMOVE --env MLFLOW_TRACKING_URI=/app/mlruns (line 193)
```

### Fix 2: Glitch #9 — sam3_topolora LoRA Conv2d
```python
# In src/minivess/adapters/sam3_topolora.py _apply_lora_to_encoder():
# CHANGE: isinstance(module, nn.Linear | nn.Conv2d)
# TO:     isinstance(module, nn.Linear)
```

### Fix 3: Glitch #10 — mamba-ssm Docker image
```bash
DOCKER_BUILDKIT=1 docker build --build-arg INSTALL_MAMBA=1 \
  -t minivess-base:latest -f deployment/docker/Dockerfile.base .
docker tag minivess-base:latest europe-north1-docker.pkg.dev/minivess-mlops/minivess/base:latest
docker push europe-north1-docker.pkg.dev/minivess-mlops/minivess/base:latest
```

### Fix 4: Glitch #12 — Zero-shot max_epochs=0
```python
# In src/minivess/config/models.py:
# CHANGE: max_epochs: int = Field(default=100, ge=1)
# TO:     max_epochs: int = Field(default=100, ge=0)
# AND add zero-shot handling in train_flow.py
```

### Verification before launch
```bash
# 1. Tests pass
make test-staging

# 2. Docker image rebuilt with INSTALL_MAMBA=1 and all code fixes
docker manifest inspect europe-north1-docker.pkg.dev/minivess-mlops/minivess/base:latest

# 3. GCP still accessible
uv run sky check gcp

# 4. Dry-run the 2nd pass script
./scripts/run_factorial_2nd_pass.sh --dry-run
```

---

## 2ND PASS CONDITIONS (15 total)

### 14 re-runs of failed conditions:
```
sam3_topolora × cbdice_cldice × calib=true  × fold=0
sam3_topolora × cbdice_cldice × calib=false × fold=0
sam3_topolora × dice_ce       × calib=true  × fold=0
sam3_topolora × dice_ce       × calib=false × fold=0
sam3_topolora × dice_ce_cldice× calib=true  × fold=0
sam3_topolora × dice_ce_cldice× calib=false × fold=0
mambavesselnet× cbdice_cldice × calib=true  × fold=0
mambavesselnet× cbdice_cldice × calib=false × fold=0
mambavesselnet× dice_ce       × calib=true  × fold=0
mambavesselnet× dice_ce       × calib=false × fold=0
mambavesselnet× dice_ce_cldice× calib=true  × fold=0
mambavesselnet× dice_ce_cldice× calib=false × fold=0
sam3_vanilla  × zero_shot     × fold=0      (eval: minivess)
vesselfm      × zero_shot     × fold=0      (eval: deepvess)
```

### 1 checkpoint-persistence probe:
```
sam3_hybrid   × cbdice_cldice × calib=true  × fold=0  (validates P0 #878 fix)
```

### Expected outcomes:
- sam3_hybrid probe: SUCCEEDED with checkpoint on GCS (validates #878)
- sam3_topolora (6): SUCCEEDED if #9 fixed
- mambavesselnet (6): SUCCEEDED if #10 fixed
- Zero-shot (2): SUCCEEDED if #12 fixed
- **Total: 15/15 SUCCEEDED**

---

## EXECUTION

```bash
# Launch all 15 conditions
./scripts/run_factorial_2nd_pass.sh

# Monitor
uv run sky jobs queue

# After all terminal → /factorial-monitor for diagnosis if any fail
```

---

## AFTER 2ND PASS

If 15/15 SUCCEEDED:
1. Verify checkpoint persistence: `gsutil ls gs://minivess-mlops-checkpoints/`
2. Verify MLflow runs: check Cloud Run MLflow UI for all 15 runs
3. Plan 3rd pass: full 26 conditions with all fixes applied
4. Update `docs/planning/run-debug-factorial-experiment-report.md` with 2nd pass results

If any FAILED:
1. Diagnose with `/factorial-monitor`
2. Fix (max 2 cycles per Rule F4)
3. Re-run only failed conditions

---

## CRITICAL RULES

Same as 1st pass. See `docs/planning/cold-start-prompt-debug-factorial-run.md`.

---

## APPENDIX: Decision-Making Record

### Initial User Prompt (verbatim)

```
After that, create /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/run-debug-factorial-experiment-2nd-pass that will
only run those models/losses combinations that had model issues! As in which creates a temporary run-debug-factorial-experimenl-2nd-pass.sh
just for this particular run to fix those failed "  Results: 11 SUCCEEDED / 14 FAILED / 1 STUCK
" which is A LOT then! and quite disappointing the level of issues that we still have :( And then after that let's create a plan how to
address the issues found in /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/run-debug-factorial-experiment-report.md to
/home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/run-debug-factorial-experiment-report-2nd-pass-fix-plab.xml with reviewer
agents working on the model-specific issues and the general infrastructure glitches. Optimize with reviewer agents then how to run this
fix-plan with our self-learning TDD Skill followed by code review for any glitches that we might have missed in the plan, and then run full
PROD test suite, in preparation for the 2nd attempt of the debug run, so that we eventually can run the full production test run on GCP with
zero failures. Use as many planning iterations rounds as needed and do not take any shortcuts as even the debug training time, and it is a
lot more efficient to try to identify issues locally before extensive cloud runs. Plan also how to improve our test suite for CI to be done
without live infrastructure to better catch those errors. Can we for example build some units tests for all those different models that could
 be run locally with my 8 GB VRAM GPU? Can we go around the the VRAM requirements if the tests are testing only parts of those models, like
the ones that you noticed crashing? As this seems for example very stupid issue only to be caught on live infrastructure: "      161 +###
Glitch #9: sam3_topolora — LoRA applied to Conv2d instead of Linear                                       "? Could we come up with ways to
test LoRA better? Try to run inference locally as part of the "local GPU test suite" (parametric limit then what runs on 8 GB, 16GB, and 24
GB VRAM GPUs as these are the typical consumer VRAM capacities? Or LoRA test would not be ever caught in that scenario anyway as that is
training-only method. But run your open-ended multi-decision test improvement plan for a "model-specific test suite"? See e.g.
https://github.com/clarahoffmann/pycon-2023-honey-i-broke-the-pytorch-model https://docs.pytorch.org/docs/stable/testing.html
https://discuss.pytorch.org/t/how-to-confirm-that-pytorch-code-is-working-as-intended-unit-testing-a-pytorch-code/16508
https://github.com/pytorch/pytorch/wiki/Running-and-writing-tests https://github.com/Lightning-AI/pytorch-lightning/issues/9674
https://circleci.com/blog/testing-pytorch-model-with-pytest/ https://pytorch.org/blog/pt-multidevice-integration/
https://www.lenovo.com/nz/en/knowledgebase/understanding-the-pytorch-model-structure-workflow-and-applications/?orgRef=https%253A%252F%252Fww
w.google.com%252F&srsltid=AfmBOorm8WTYnAPBVsEW5tcBm2sleull91anf0yuJqOtnwz_ZkF5TGUY https://github.com/pytorch/pytorch/issues/176833 . Create
a separate /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/test-suite-improvement-report.md which can also inform other
types of types beyond unit tests (e.g. /home/petteri/Dropbox/KnowledgeBase/MLOps/MLOps - Testing.md
/home/petteri/Dropbox/KnowledgeBase/MLOps/MLOps - Code - Notebooks.md), if you feel like? Do comprehensive web research on best practices,
and after this test report optimized with reviewer agents, take the insights then to this .xml plan that is all about executable steps based
on the research reports! Is this clear, ask me interactive questions with multiple answers presented to make sure that we are on same page
with the plan goal!
``


### Q&A Session

**Q1: For the 2nd-pass .sh script, should it ONLY re-run the 14 failed conditions, or also re-run the 11 'succeeded' conditions that lost checkpoints due to P0 #878?**

Options presented:
- A) All 26 conditions (Recommended)
- B) Only 14+1 failed
- C) Failed models + 1 probe per succeeded model

**User answer**: "I don't need the checkpoint per se for anything as this is a debugging
run and this is not the full production run to be reported in
`sci-llm-writer/manuscripts/vasculature-mlops`! This debug training run must ensure that
we get the artifacts needed through the pipeline mechanics that I eventually need for the
manuscript, and make sure that the pipeline is saving all the checkpoints correctly. Is
this clear? So only '14+1' to test that we fixed all models probably, and this subset tests
the checkpoint mechanics properly as it failed with too large weights whereas the succeeded
dynunet used the smallest .pth so it should work as well. And after this 2nd pass, I think
we need to run another 3rd pass with all the 26 conditions for another double-check.
Remember to save my answers and your questions verbatim, along with my initial prompt (As
an appendix to .xml?) for this 2nd pass planning as we will repeating this debugging experiment!"

**Q2: For the fix-plan XML, should the LoRA Conv2d fix implement Conv2d LoRA support, or simply skip Conv2d layers?**

Options presented:
- A) Skip Conv2d layers (Recommended)
- B) Implement LoRAConv2d
- C) Both — skip for now, issue for later

**User answer**: "Skip Conv2d layers now, but create a P2 priority issue on the possible
LoRAConv2d if desired at some later point, so that we remember that this is an option."

**Q3: For the local GPU test suite, your RTX 2070 Super has 8 GB VRAM. What test depth do you want for model-specific tests?**

Options presented:
- A) Component tests only (Recommended)
- B) Component + mini-training
- C) Component + mini-training + inference
- D) Full parametric VRAM tiers

**User answer**: "Couldn't we use a distinct group of the model-specific tests that only
get triggered when we have a change in models? or is manually triggered if the folder
structure is so that we cannot easily have conditional test filter based on file changes
in some folder? No need to run all the extensive tests everytime for model-specific changes
if the models never change, right?"

**Q4: For the test-suite-improvement-report.md, how deep should the web research go?**

Options presented:
- A) Focused on our models (Recommended)
- B) Comprehensive MLOps testing
- C) Minimal — skip research

**User answer**: "Comprehensive MLOps testing"
