# Experiment Harness — Failure History

Why each rule exists, grounded in real incidents.

## H1: REPORT-BEFORE-LAUNCH

**Incident (5th pass, 2026-03-23)**: Claude loaded `/factorial-monitor`, launched jobs,
polled `sky jobs queue` 6 times, but never created the report file. 4 DynUNet jobs
succeeded with timing data (SWAG 2.2x overhead, setup 47% of job time, ±1.5%
reproducibility) — all lost to conversation context.

## H2: UPDATE-EVERY-POLL

**Incident (5th pass)**: Report was only created after user noticed it was missing,
4+ hours into the run. By then, the per-poll observations were reconstructed from
memory, losing granularity.

## H3: COMPOUND-OR-FAIL

**Incident (4th pass, 2026-03-22)**: 8 FAILED_SETUP jobs went undetected for 2+ hours.
No new tests were written from the DVC bare-pull failure pattern. The same failure
recurred in the 5th pass preparation because no test existed.

## H4: REFERENCE-PRIOR-PASS

**Incident (5th pass)**: Claude launched 34 jobs KNOWING 16 would fail from 4th pass
root causes (mamba-ssm missing, --post-training-method unrecognized). The 4th pass
report was available but never read before launching.

## H5: NO-AD-HOC-POLLING

**Incident (5th pass)**: Instead of following the /factorial-monitor 6-phase protocol,
Claude ran raw `sky jobs queue` in sleep loops. No structured diagnosis, no manifest
tracking, no kill-switch evaluation.

## H6: REASON-BEFORE-TEMPLATE

**Root cause of H1-H5**: All five failures stem from the same meta-pattern — executing
templates mechanically without genuine reasoning about what could go wrong. The harness
exists to force thinking, not just checklist-filling.
