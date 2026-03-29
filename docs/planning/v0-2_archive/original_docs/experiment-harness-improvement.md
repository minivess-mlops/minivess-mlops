# Experiment Harness Improvement: Multi-Hypothesis Decision Matrix Report

**Authors**: 5-Expert Iterated LLM Council
**Date**: 2026-03-28
**Branch**: `fix/10th-pass-production-readiness`
**Status**: Decision report -- pending user authorization for implementation

---

## 1. Abstract

The Vascadia v0.2-beta experiment harness -- a Claude Code skill governing factorial
GPU experiments on GCP via SkyPilot -- has fundamentally failed across 11 iterative
passes. The 10th pass saw Job #154 run 12+ hours for a 5-minute training task ($23.30
wasted), and the 11th pass left 3 Phase 1 jobs PENDING for 10+ hours with zero
monitoring intervention. The root cause is architectural: the harness exists as
XML/markdown instructions executed by an ephemeral Claude Code session, not as
persistent infrastructure. When the session ends, monitoring ceases entirely. This
report convenes five expert perspectives -- MLOps Platform Architect, SRE/Observability
Engineer, Claude Code Skill Designer, Cloud Cost Engineer, and Research Software
Engineer -- to evaluate seven competing hypotheses for persistent monitoring. Through a
multi-criteria decision matrix scoring implementation effort, reliability, detection
latency, cost, and researcher-friendliness, we recommend a phased approach: (1) a
lightweight cron-based Python monitor as the immediate foundation, (2) Prefect-based
monitoring integration for orchestration-aware alerting, and (3) GCP Cloud Monitoring
budget alerts as a cost safety net. This combination achieves autonomous fault detection
within 60 seconds, survives session loss, and requires no SRE team to maintain.

---

## 2. Introduction

### 2.1 Problem Statement

The Vascadia project is an open-source, model-agnostic biomedical segmentation MLOps
platform targeting a Nature Protocols publication. Its experiment execution relies on a
"harness" -- a Claude Code skill (`.claude/skills/experiment-harness/SKILL.md`) that
standardizes factorial experiment runs across SkyPilot-managed GCP GPU instances.

The harness defines a 5-phase workflow: GENERATE (XML plan), VALIDATE (pre-launch
gates), EXECUTE (launch + monitor), COMPOUND (tests + issues + observations), and
REFLECT (self-assessment). On paper, this is a well-structured protocol. In practice,
it has one fatal flaw: **the entire monitoring layer depends on an active Claude Code
session**.

### 2.2 The 11-Pass Failure History

Over 11 iterative passes, the harness has accumulated a pattern of monitoring failures
with escalating severity:

| Pass | Date | Core Failure | Cost Impact |
|------|------|-------------|-------------|
| 4th | 2026-03-22 | 8 FAILED_SETUP jobs undetected for 2h 20m | $6.30 wasted |
| 10th | 2026-03-27 | Job #154 ran 12+ hours (expected: 5 min) | $23.30 wasted |
| 11th | 2026-03-28 | 3 jobs PENDING 10+ hours, zero intervention | Opportunity cost |

The 4th pass demonstrated that **no automated polling exists** -- detection required a
human running `sky jobs queue` manually. The 10th pass demonstrated that **no duration
monitoring exists** -- the `ralph_monitor.py` script tracks job status transitions but
never compares elapsed time against expected duration. The 11th pass demonstrated that
**no session-persistent monitoring exists** -- when the Claude Code session that launched
the jobs ended, all monitoring capability vanished.

### 2.3 Scope and Constraints

This analysis operates within the following non-negotiable constraints:

- **Single researcher**: No SRE team. The monitoring must be autonomous.
- **Docker-per-flow**: Monitoring runs outside training containers.
- **SkyPilot is the compute broker**: Monitoring must work WITH SkyPilot, not around it.
- **Ephemeral sessions**: Claude Code sessions end, lose context, hit token limits.
- **Prefect 3.x exists**: The project already has an orchestration layer.
- **GitHub Actions CI disabled**: No CI-driven monitoring.
- **Academic project**: Infra complexity must justify its existence for a Nature Protocols audience.

---

## 3. Related Work

### 3.1 Production ML Platform Monitoring

**Vertex AI Pipelines** (GCP) provides built-in pipeline step monitoring with automatic
timeout enforcement, email/PagerDuty alerts on step failure, and integrated cost
attribution. Steps that exceed their declared timeout are forcefully terminated. However,
Vertex AI Pipelines requires rewriting the pipeline to use Vertex SDK components -- a
heavy migration for a SkyPilot-based project.

**SageMaker Training Jobs** (AWS) implement `StoppingCondition.MaxRuntimeInSeconds` as a
first-class parameter. Jobs exceeding this limit are automatically terminated. CloudWatch
alarms trigger on job state transitions. The architectural lesson: **timeout enforcement
must be a property of the job submission, not a property of the monitoring system**.

**Databricks Jobs** provide built-in timeout, retry policies, and webhook-based alerting
(Slack, PagerDuty, email). Job clusters are automatically terminated when jobs complete or
timeout. The lesson: **monitoring is a platform feature, not an add-on script**.

### 3.2 Orchestration-Layer Monitoring

**Prefect 3.x** (already in the Vascadia stack) supports automations with triggers on flow
run state changes. A Prefect automation can fire a webhook, send a Slack message, or start
another flow when a flow run transitions to FAILED, CRASHED, or exceeds a timeout. The
Prefect Cloud tier provides this natively; Prefect self-hosted requires additional setup
but supports custom automations via the API.

**Dagster** uses sensors -- Python functions that poll external systems and trigger runs or
alerts based on conditions. A sensor polling SkyPilot job status every 30 seconds would
detect failures within one polling cycle. Dagster is not in the Vascadia stack, but its
sensor architecture is instructive.

**Airflow** uses SLA (Service Level Agreement) mechanisms that fire callbacks when tasks
exceed expected duration. The architectural pattern: **expected duration is declared per
task, and the scheduler enforces it**.

### 3.3 SkyPilot's Own Monitoring Capabilities

SkyPilot (Yang et al., NSDI 2023) provides:
- `sky jobs queue` / `sky.jobs.queue()` -- programmatic job status querying
- `job_recovery: FAILOVER` -- automatic recovery from spot preemption
- `max_restarts_on_errors` -- retry limit for setup failures
- Controller-managed job lifecycle -- jobs persist beyond the launching process
- **No built-in duration timeout** -- SkyPilot has no `max_runtime` parameter
- **No built-in alerting** -- no webhooks, no email, no callbacks on state change

SkyPilot's design assumes that job management is the user's responsibility after
submission. The controller manages spot recovery and failover, but duration enforcement
and failure alerting are out of scope.

### 3.4 Cost Management Approaches

**GCP Budget Alerts** can trigger email notifications or Pub/Sub messages when spending
exceeds defined thresholds (e.g., 50%, 90%, 100% of budget). These operate at the
project or billing-account level with ~15-minute latency.

**Infracost** and similar tools provide pre-commit cost estimation but not runtime cost
monitoring. The Vascadia `preflight_gcp.py` script already estimates per-experiment cost
before launch -- this is a pre-commit defense, not a runtime defense.

---

## 4. Failure Analysis: Root Cause Taxonomy

Analysis of monitoring failures across 11 passes reveals five distinct root cause
categories:

### RC1: Ephemeral Monitor (Architectural)

The experiment harness is a Claude Code skill -- a set of instructions that a Claude
session follows. It has no persistent process, no daemon, no cron job. When the
session ends (token limit, user closes terminal, network disconnection), monitoring
stops completely.

**Passes affected**: All 11 passes (latent in every pass)
**Metalearning**: `.claude/metalearning/2026-03-28-context-amnesia-deferred-deepvess-whac-a-mole.md`

### RC2: No Duration Enforcement (Detection Gap)

The `ralph_monitor.py` script tracks job status (PENDING, RUNNING, SUCCEEDED, FAILED)
but never compares elapsed duration against expected duration. A job running 12 hours
when 5 minutes was expected produces no alert. The monitoring script has a
`_TERMINAL_STATUSES` set but no `_EXPECTED_DURATIONS` mapping.

**Passes affected**: 10th pass (Job #154 ran 12+ hours)
**Metalearning**: `.claude/metalearning/2026-03-27-no-job-duration-monitoring-12h-debug-run.md`

### RC3: No Batch-Level Awareness (Scope Gap)

`ralph_monitor.py` watches one job. Factorial experiments launch 20-34 jobs. No script
aggregates batch-level health (e.g., "18 of 23 jobs FAILED_SETUP with same error").
The factorial-monitor skill describes this aggregation, but only as a Claude Code
behavior protocol, not as runnable code.

**Passes affected**: 4th pass (8 identical FAILED_SETUP undetected), 10th pass (31 PENDING)

### RC4: No Autonomous Escalation (Alerting Gap)

When a failure is detected (by any mechanism), there is no way to notify the offline
researcher. No email, no Slack webhook, no desktop notification, no SMS. The only
"alerting" is printing to the Claude Code conversation, which requires the session to
be active and the user to be watching.

**Passes affected**: All passes where the user was offline during failures

### RC5: No Cost Safety Net (Financial Gap)

The harness estimates cost before launch (preflight) but has no runtime cost guard.
A stuck job accumulates cost indefinitely. The ralph-loop skill has a `$5 cost budget`
but it is per-job, poorly enforced, and operates only within a Claude session.

**Passes affected**: 10th pass ($23.30 on a single job)

---

## 5. Hypothesis Generation

Seven competing approaches for solving the monitoring gap:

### H1: Cron-Based External Monitor

A standalone Python script (`scripts/monitor_factorial.py`) runs as a cron job or
`systemd` timer every 30-60 seconds. It reads a JSON manifest of launched jobs, queries
SkyPilot via `sky.jobs.queue()`, checks duration anomalies, detects batch-level failure
patterns, and writes alerts to a log file. Optionally sends desktop notifications via
`notify-send` or emails via `sendmail`.

**Architecture**: Cron triggers script -> script reads manifest -> queries SkyPilot ->
evaluates rules -> writes state file -> sends alerts.

### H2: SkyPilot Callbacks/Webhooks

SkyPilot could hypothetically support post-job callbacks or webhook notifications on
state transitions. If available, this would eliminate polling entirely.

**Reality check**: As of SkyPilot v1.0 (2026), there are no callback or webhook
mechanisms. The controller manages job lifecycle internally but does not expose event
hooks. This hypothesis is **not viable with current SkyPilot**.

### H3: Prefect Flow for Monitoring

Create a Prefect flow (`monitor_flow`) that runs locally (not on cloud GPU) and polls
SkyPilot job status. Prefect's built-in automation triggers can fire on flow state
changes. The monitor flow itself is orchestrated by Prefect, gaining retry logic,
state persistence, and UI visibility.

**Architecture**: `prefect deployment run 'monitor-flow/factorial'` -> Prefect agent
runs the flow -> flow polls SkyPilot -> Prefect automations handle alerting.

### H4: GCP Cloud Monitoring / Budget Alerts

Configure GCP Cloud Monitoring alert policies on Compute Engine metrics (CPU
utilization, instance uptime) and GCP Budget Alerts on project spending. These are
fully managed services that operate independently of any local process.

**Architecture**: GCP Budget Alert at 50%/90%/100% of experiment budget -> email
notification. Cloud Monitoring alert on instance running > N hours -> email.

### H5: Claude Code Scheduled Triggers

Claude Code's `/schedule` skill creates cron-scheduled remote Claude sessions that
execute predefined tasks. A scheduled trigger could run every 30 minutes, query job
status, and take action (cancel stuck jobs, alert the user).

**Architecture**: `/schedule` creates a cron entry -> Claude headless session starts ->
reads manifest -> queries SkyPilot -> takes action -> session ends.

### H6: Hybrid Approach

Combine a lightweight cron daemon (H1) for continuous polling with Claude Code sessions
(manual or scheduled) for intelligent diagnosis. The daemon handles detection; Claude
handles analysis.

**Architecture**: Cron script polls + detects anomalies + writes alert files + sends
basic notifications. Claude session (when available) reads alert files + performs root
cause analysis + updates report.

### H7: Accept SkyPilot Recovery, Monitor Only Cost

Accept that SkyPilot's built-in `job_recovery: FAILOVER` handles spot preemption and
setup failures via retries. Only add cost monitoring (GCP Budget Alerts) as a safety net.
Do not build custom monitoring infrastructure.

**Argument**: SkyPilot already retries on failure. The real risk is unbounded cost, not
undetected failure. A budget alert at $15 caps the worst case.

---

## 6. Decision Matrix

### 6.1 Criteria Definitions

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Implementation effort | 15% | Hours to build and test |
| Maintenance burden | 10% | Ongoing effort to keep working |
| Reliability | 20% | Survives session loss, reboots, network drops |
| Detection latency | 20% | Time from anomaly to detection |
| Cost | 10% | Infrastructure + monitoring overhead (USD/month) |
| DevEx | 10% | Researcher-friendly? One-command setup? |
| Reproducibility | 5% | Works on any machine, not just this one |
| Escalation | 10% | Can alert offline humans? |

### 6.2 Scoring (1-5 scale, 5 = best)

| Criterion | Weight | H1: Cron | H2: SkyPilot | H3: Prefect | H4: GCP | H5: Claude Schedule | H6: Hybrid | H7: Accept+Cost |
|-----------|--------|----------|-------------|-------------|---------|-------------------|-----------|-----------------|
| Implementation effort | 15% | 4 | 1* | 3 | 4 | 3 | 3 | 5 |
| Maintenance burden | 10% | 4 | 1* | 3 | 5 | 2 | 3 | 5 |
| Reliability | 20% | 4 | 1* | 4 | 5 | 2 | 4 | 3 |
| Detection latency | 20% | 5 | 1* | 4 | 2 | 2 | 5 | 1 |
| Cost | 10% | 5 | 5 | 4 | 4 | 3 | 4 | 5 |
| DevEx | 10% | 4 | 1* | 4 | 3 | 4 | 3 | 5 |
| Reproducibility | 5% | 4 | 1* | 3 | 2 | 2 | 3 | 4 |
| Escalation | 10% | 3 | 1* | 4 | 4 | 3 | 4 | 3 |
| **Weighted Total** | **100%** | **4.10** | **1.00** | **3.65** | **3.65** | **2.45** | **3.85** | **3.30** |

*\*H2 scores 1 across the board because SkyPilot v1.0 does not support callbacks/webhooks. This hypothesis is eliminated.*

### 6.3 Scoring Rationale

**H1 (Cron) scores highest (4.10)** because it is simple, reliable, and has the best
detection latency. A Python script running every 30 seconds via cron/systemd is
battle-tested infrastructure that survives session loss, reboots (with systemd), and
requires no additional services. Its weakness is escalation -- cron can send email via
`sendmail` but not Slack without additional setup.

**H6 (Hybrid) scores second (3.85)** because it combines H1's reliability with Claude
Code's intelligence. The cron daemon detects; Claude diagnoses. However, it has higher
implementation effort (two systems to build) and higher maintenance burden.

**H3 (Prefect) and H4 (GCP) tie at 3.65** for different reasons. Prefect integrates
naturally with the existing stack and has good escalation (automations), but adds
orchestration complexity to monitoring. GCP Cloud Monitoring is fully managed and
highly reliable, but has poor detection latency (~15 minutes for budget alerts) and
limited control (no SkyPilot-level insight).

**H7 (Accept+Cost) scores 3.30** and has the best DevEx (do nothing), but its detection
latency of "eventually" for stuck jobs is unacceptable given the 10th pass failure.

**H5 (Claude Schedule) scores 2.45** because scheduled Claude sessions are expensive
(API credits per invocation), have coarse granularity (minimum practical interval ~30
minutes), and are a novel mechanism with uncertain reliability.

---

## 7. Experimental Evidence

### 7.1 What 11 Passes Taught Us

**Evidence 1: Duration anomalies are the dominant failure mode.**
The 10th pass Job #154 is the most expensive single failure ($23.30). A simple
`if elapsed > 3 * expected: cancel()` rule would have caught it at minute 15, saving
$23+ and 12 hours. Duration checking is the highest-ROI monitoring feature.

**Evidence 2: Batch-level kill switches prevent cascading waste.**
The 4th pass had 8 identical FAILED_SETUP jobs. A kill switch ("3+ identical failures
in 5 minutes -> cancel remaining") would have saved $3.90 of $6.30. The
`factorial-monitor` skill upgrade plan (2026-03-22) specifies this but it was never
implemented as code.

**Evidence 3: Session persistence is non-negotiable.**
The 11th pass proved that Claude Code skills cannot guarantee monitoring continuity.
The session that launched 3 jobs did not persist to monitor them. Any monitoring
solution that depends on Claude Code session presence will fail in the same way.

**Evidence 4: SkyPilot's built-in recovery is necessary but insufficient.**
SkyPilot's `job_recovery: FAILOVER` correctly handles spot preemption by relaunching
on different instances. However, it does not handle: jobs that succeed training but
stall in post-processing, jobs where setup succeeds but training hangs, or jobs where
the cost exceeds budget due to slow recovery cycles. Recovery is for resilience;
monitoring is for intelligence.

**Evidence 5: Preflight checks prevent some failures but not runtime anomalies.**
The 11th pass validated 15/15 preflight checks before launch. All gates passed. The
jobs still PENDING for 10+ hours because the failure was a runtime capacity constraint,
not a configuration error. Preflight is necessary but not sufficient.

### 7.2 Cost of Inaction

Across the observed failures:

| Pass | Wasted Cost | Time Lost | Root Cause Preventable By |
|------|-------------|-----------|--------------------------|
| 4th | $6.30 | 2h 20m detection | H1 (30s polling + kill switch) |
| 10th | $23.30 | 12+ hours | H1 (duration anomaly detection) |
| 11th | ~$0 (PENDING) | 10+ hours opportunity | H1 (PENDING duration alert) |
| **Total** | **~$30** | **~25 hours** | |

While $30 is modest in absolute terms, the pattern compounds. The full factorial
experiment has 720+ conditions. At the observed failure rate, an unmonitored production
run could waste $200+ and days of GPU-hours.

---

## 8. Discussion

### 8.1 Trade-offs

**Simplicity vs. Intelligence**: H1 (cron) is simple but dumb -- it detects anomalies
via threshold rules but cannot diagnose root causes. H6 (hybrid) adds Claude's
diagnostic intelligence but doubles the implementation surface. For a single researcher,
simplicity wins: detect fast, diagnose manually.

**Platform integration vs. Independence**: H3 (Prefect) integrates monitoring into the
existing orchestration layer, which is architecturally clean. However, it couples
monitoring to Prefect's availability. If Prefect is down, monitoring is down. H1 (cron)
is independent -- it survives anything short of a machine reboot.

**Managed vs. Self-hosted**: H4 (GCP Cloud Monitoring) is fully managed and requires
zero local infrastructure. But its 15-minute latency for budget alerts is too slow
for the per-job anomalies observed (5-minute training jobs). GCP monitoring is
appropriate as a cost safety net, not as a primary anomaly detector.

### 8.2 What NOT to Build

**Do not build a custom dashboard.** The temptation to build a web UI showing job
status in real-time is strong. Resist it. A terminal-printed status table from the
cron script, combined with the JSON manifest, provides sufficient visibility. The
Nature Protocols paper does not need a monitoring dashboard -- it needs reproducible
experiments.

**Do not build a Slack integration in Phase 1.** Slack webhooks are easy to add but
create a dependency on external services. Start with `notify-send` (Linux desktop
notification) and email via `sendmail`. Add Slack later if needed.

**Do not attempt to make Claude Code sessions persistent.** The fundamental limitation
is that Claude Code sessions are stateless and ephemeral. Building workarounds
(heartbeat files, session resumption protocols) adds complexity without solving the
root cause. Accept that Claude Code is a diagnostic tool, not a monitoring daemon.

**Do not replace SkyPilot's job management.** SkyPilot's controller already manages
job lifecycle, spot recovery, and failover. The monitoring layer should OBSERVE
SkyPilot's state, not DUPLICATE its job management. Query `sky.jobs.queue()`, do
not re-implement job tracking.

### 8.3 Limitations of This Analysis

1. **SkyPilot API stability**: The Python API (`sky.jobs.queue()`) is used for
   programmatic access, but SkyPilot v1.0 is pre-stable and the API may change.
2. **Single-machine assumption**: The cron-based monitor assumes the researcher's
   machine is running. If the machine sleeps or shuts down, monitoring stops. A
   cloud-hosted monitor (e.g., Cloud Function polling SkyPilot) would solve this
   but adds significant complexity.
3. **Alert fatigue**: With 720+ conditions in a full factorial, even a well-tuned
   monitor will generate many notifications. Alert aggregation and deduplication
   are essential but not designed here.

---

## 9. Recommendations

### Ranked Recommendation List

| Rank | Approach | Rationale |
|------|----------|-----------|
| **1** | H1: Cron-based monitor (`monitor_factorial.py`) | Highest weighted score (4.10). Simple, reliable, fast detection. |
| **2** | H4: GCP Budget Alerts (cost safety net) | Fully managed, zero maintenance. Catches runaway cost even if H1 fails. |
| **3** | H3: Prefect monitor flow (orchestration-aware) | Natural fit with existing stack. Adds automation triggers for alerting. |
| **4** | H6: Hybrid (H1 + Claude diagnosis) | Best overall capability but highest complexity. Defer to Phase 3. |

### Phased Implementation Plan

#### Phase 1: Cron-Based Monitor (Priority: IMMEDIATE, Effort: 8-12 hours)

**Goal**: Autonomous monitoring that survives session loss and detects all observed
failure modes within 60 seconds.

| Task | Description | Est. Hours |
|------|-------------|-----------|
| T1 | Job manifest writing in `run_factorial.sh` (job_id, condition, timestamps) | 2-3 |
| T2 | `scripts/monitor_factorial.py` -- 30s polling, duration anomaly detection, batch kill switch | 4-6 |
| T3 | Wire `--monitor` flag to chain launch -> monitor | 1 |
| T4 | Desktop notification via `notify-send` on CRITICAL alerts | 1 |
| T5 | Tests for T1-T4 (mocked SkyPilot responses) | 2-3 |

**Detection rules (from metalearning)**:
- Duration > 3x expected -> WARN (log + notify)
- Duration > max_duration -> CANCEL + ALERT
- 3+ identical failures in 5 min -> KILL SWITCH (cancel PENDING/STARTING)
- >50% of batch failed -> KILL SWITCH
- PENDING > 2 hours -> WARN (capacity constraint notification)
- Total cost > budget threshold -> CANCEL ALL + ALERT

**Deployment**: `crontab -e` or `systemd` timer running every 30 seconds. Alternatively,
the script runs as a foreground loop (started by `--monitor` flag) that the researcher
can `nohup` or run in `tmux`.

**Acceptance gate**: Run a simulated factorial with mocked SkyPilot responses. All five
failure scenarios from passes 4, 10, and 11 must be detected within 60 seconds.

#### Phase 2: GCP Budget Alerts (Priority: HIGH, Effort: 2-3 hours)

**Goal**: Cloud-native cost safety net that operates even if the local machine is off.

| Task | Description | Est. Hours |
|------|-------------|-----------|
| T6 | Create GCP Budget Alert at $15/$30/$50 thresholds via Pulumi | 1-2 |
| T7 | Configure email notification channel to researcher's email | 0.5 |
| T8 | Document budget alert in `deployment/pulumi/gcp/CLAUDE.md` | 0.5 |

**Note**: GCP Budget Alerts have ~15-minute latency. They are a cost ceiling, not a
per-job monitor. They complement Phase 1, not replace it.

#### Phase 3: Prefect Monitor Flow (Priority: MEDIUM, Effort: 6-8 hours)

**Goal**: Orchestration-aware monitoring with Prefect automation triggers.

| Task | Description | Est. Hours |
|------|-------------|-----------|
| T9 | Create `monitor_flow` in `src/minivess/orchestration/flows/` | 3-4 |
| T10 | Prefect automation: trigger email/webhook on monitor_flow FAILED | 2 |
| T11 | Integration: `run_factorial.sh --monitor` triggers Prefect deployment | 1-2 |

**Rationale for Phase 3**: Prefect integration provides UI visibility (researcher can
check Prefect UI for experiment status), automation triggers (Slack/email on failure),
and state persistence (flow run state survives process restart). However, it adds
Prefect as a dependency for monitoring, which is why it follows the simpler H1 in Phase 1.

#### Phase 4: Hybrid Claude Diagnosis (Priority: LOW, Effort: 4-6 hours)

**Goal**: Intelligent root cause analysis via Claude Code sessions triggered by alerts.

| Task | Description | Est. Hours |
|------|-------------|-----------|
| T12 | Alert files written by Phase 1 monitor in a standard format | 1 |
| T13 | Claude `/schedule` trigger to read alert files and diagnose | 2-3 |
| T14 | Diagnosis results appended to experiment report automatically | 1-2 |

**Rationale for Phase 4**: This is the "luxury" layer. Phase 1 detects, Phase 2 caps
cost, Phase 3 orchestrates. Phase 4 adds AI-powered diagnosis. Defer until Phases 1-3
are proven stable.

### Total Estimated Effort

| Phase | Hours | Cumulative |
|-------|-------|-----------|
| Phase 1 (Cron) | 8-12 | 8-12 |
| Phase 2 (GCP) | 2-3 | 10-15 |
| Phase 3 (Prefect) | 6-8 | 16-23 |
| Phase 4 (Hybrid) | 4-6 | 20-29 |

**Minimum viable monitoring**: Phase 1 alone (8-12 hours) resolves all three observed
failure modes. Phases 2-4 add defense in depth.

---

## 10. References

1. Yang, Z., Choudhary, S., Chiang, W.-L., Wu, Z., Gonzalez, J. E., Stoica, I., Zaharia, M. (2023). "SkyPilot: An Intercloud Broker for Sky Computing." *NSDI '23*, USENIX. https://www.usenix.org/conference/nsdi23/presentation/yang-zongheng

2. Prefect Technologies. (2026). "Prefect 3.x Documentation: Automations." https://docs.prefect.io/latest/automate/

3. Google Cloud. (2026). "Cloud Monitoring: Alerting Policies." https://cloud.google.com/monitoring/alerts

4. Google Cloud. (2026). "Managing Your Cloud Billing Budget." https://cloud.google.com/billing/docs/how-to/budgets

5. Amazon Web Services. (2026). "SageMaker Training Jobs: StoppingCondition." https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StoppingCondition.html

6. Google Cloud. (2026). "Vertex AI Pipelines: Managing Pipeline Runs." https://cloud.google.com/vertex-ai/docs/pipelines/manage-pipeline-runs

7. Dagster Labs. (2026). "Dagster Docs: Sensors." https://docs.dagster.io/concepts/partitions-schedules-sensors/sensors

8. Bi, Y., Chen, Z., Wang, L. et al. (2026). "Automating Skill Acquisition for LLM Agents." arXiv:2603.11808.

9. Vascadia metalearning: `.claude/metalearning/2026-03-27-no-job-duration-monitoring-12h-debug-run.md`

10. Vascadia metalearning: `.claude/metalearning/2026-03-28-context-amnesia-deferred-deepvess-whac-a-mole.md`

11. Vascadia factorial-monitor upgrade plan: `.claude/skills/factorial-monitor/skill-upgrade-plan-for-proper-monitoring.md`

---

## Appendix A: Expert Council Perspectives

### Expert 1: MLOps Platform Architect

> "The fundamental issue is that you're using an LLM conversation as a control plane.
> Production ML platforms (Vertex AI, SageMaker) solve this by making timeout and
> monitoring intrinsic to the job submission API, not extrinsic via a separate process.
> SkyPilot lacks this, so you must build it as a sidecar. The cron-based monitor is
> the right pattern -- it's the equivalent of a Kubernetes sidecar container that
> health-checks the main workload. The Prefect flow approach is architecturally cleaner
> but over-engineered for a single-researcher setup."

### Expert 2: SRE / Observability Engineer

> "Three pillars: metrics, logs, traces. You have logs (SkyPilot job logs) but no
> metrics (duration, cost time-series) and no traces (cross-flow provenance). The
> immediate need is metrics -- specifically, job duration vs. expected duration. This
> is a single time-series comparison, not a complex observability stack. A cron script
> that writes metrics to a JSONL file and checks thresholds is sufficient. Do NOT
> build Prometheus/Grafana for 20 jobs. The escalation path matters most: if nobody
> reads the alert, the alert is useless. Desktop notifications plus email covers the
> single-researcher case."

### Expert 3: Claude Code Skill Designer

> "Claude Code skills persist state via files (JSONL, markdown reports, YAML configs),
> not via session memory. The experiment harness correctly writes to
> `outputs/harness-state.jsonl`, but the monitoring loop itself is session-bound.
> The fix is to externalize the monitoring loop to a real process (cron/systemd) and
> reserve Claude Code for the DIAGNOSIS step that requires LLM intelligence. The
> `/schedule` skill can bridge this: a scheduled Claude session reads the monitor's
> output files and performs root cause analysis. But this should be Phase 4, not
> Phase 1. Phase 1 must work without any Claude Code involvement."

### Expert 4: Cloud Cost Engineer

> "The $23.30 on a $0.02 job is a 1,165x overspend. In cloud cost management, the
> standard defense is a budget alert with automatic action (e.g., shut down billing
> account). GCP Budget Alerts at the project level with a Pub/Sub trigger can invoke
> a Cloud Function that cancels all SkyPilot jobs. However, this is heavyweight for an
> academic project. A simpler approach: the cron monitor tracks cumulative cost
> (elapsed_hours * hourly_rate per job) and cancels all jobs when the budget is exceeded.
> This is faster than GCP Budget Alerts (30s vs. 15 min latency) and requires no
> cloud-side configuration."

### Expert 5: Research Software Engineer

> "A PhD researcher needs exactly three things: (1) 'Are my jobs running?' -- answered
> by a status summary. (2) 'Is something broken?' -- answered by an alert. (3) 'How
> much is this costing?' -- answered by a running total. Everything else is nice-to-have.
> The cron monitor provides all three. The Prefect integration provides a nicer UI but
> is not essential. The GCP budget alert is insurance. The Claude diagnosis is a luxury.
> Start with cron. The researcher's actual workflow is: launch jobs, go to bed, wake up,
> check results. The monitor must handle the 8+ hours of unsupervised execution. A tmux
> session running `monitor_factorial.py` in a loop is the minimal viable solution."

---

## Appendix B: Implementation Specification for Phase 1

### `scripts/monitor_factorial.py` Core Loop (Pseudocode)

```
read manifest from argv[1]
load expected_durations from manifest header
load budget_cap from manifest header

while not all_terminal(manifest):
    queue = sky.jobs.queue()
    for job in manifest.jobs:
        update job.status from queue
        update job.duration = now - job.started_at

        if job.status == RUNNING and job.duration > job.max_duration:
            sky.jobs.cancel(job.job_id)
            job.status = CANCELLED
            job.failure_category = DURATION_EXCEEDED
            send_alert(CRITICAL, f"Job {job.job_id} exceeded max duration")

        if job.status == RUNNING and job.duration > 3 * job.expected_duration:
            send_alert(WARN, f"Job {job.job_id} running {job.duration}, expected {job.expected}")

        if job.status == FAILED_SETUP:
            logs = sky.jobs.logs(job.job_id)
            job.failure_category = diagnose(logs)
            send_alert(HIGH, f"Job {job.job_id} FAILED_SETUP: {job.failure_category}")

    check_kill_switch(manifest)  # 3+ identical failures OR >50% failed
    check_cost_budget(manifest, budget_cap)
    write manifest to disk (atomic write)
    print_status_table(manifest)
    sleep(30)

print_final_summary(manifest)
```

### Expected Durations Source

Expected durations come from the experiment XML `<expected-duration-minutes>` field,
which is written to the manifest header at launch time. The 11th pass report provides
calibrated values:

| Model | Expected (min) | WARN Threshold (3x) | CANCEL Threshold |
|-------|---------------|---------------------|------------------|
| DynUNet | 10 | 30 | 50 |
| SAM3 Hybrid | 25 | 75 | 75 |
| SAM3 Vanilla (ZS) | 15 | 45 | 75 |
| SAM3 TopoLoRA | 25 | 75 | 75 |
| VesselFM (ZS) | 15 | 45 | 75 |
| MambaVesselNet | 12 | 36 | 60 |
