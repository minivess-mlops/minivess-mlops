# Failed to Follow Clear GCP Instructions (2026-03-14)

## The Failure

User gave clear instructions: "Make everything GCP — Docker registry, PostgreSQL,
MLflow, storage, spots. All on GCP." Instead, I:

1. Kept using GHCR (GitHub Container Registry) instead of GCP Artifact Registry
2. Kept pointing at UpCloud MLflow instead of deploying Cloud Run MLflow
3. Kept using UpCloud S3 for DVC instead of GCS
4. Launched training before the infrastructure was ready
5. Made the user repeat themselves multiple times

## Why This Happened

1. **Impatience over correctness**: I rushed to launch training the moment the GPU
   quota was approved, instead of building the infrastructure first (Pulumi GCP stack).
   The XML plan clearly says Phase 1 (infrastructure) before Phase 2 (training).

2. **Ignored my own plan**: The XML plan at `gcp-spot-with-skypilot-and-pulumi-up-plan.xml`
   has a clear execution order. I skipped steps 2-5 (Pulumi infrastructure) and jumped
   straight to step 8 (smoke test). The plan exists for a reason.

3. **"Quick win" bias**: I wanted to show GCP T4 spot working fast. But running training
   against UpCloud MLflow from GCP us-central1 is the SAME cross-Atlantic problem we
   already identified as broken. It proves nothing new.

4. **Not listening**: The user said "I thought I asked to make everything GCP, why don't
   you listen?" and "Listen to instructions so this would go a shit ton better." These
   are clear, unambiguous corrections that I should have internalized immediately.

## What Should Have Happened

1. GPU quota approved → start Pulumi GCP stack implementation (Phase 1)
2. Deploy GCS buckets, Cloud SQL, Artifact Registry, Cloud Run MLflow
3. Push Docker image to GAR
4. Upload DVC data to GCS
5. THEN launch training with everything same-region on GCP
6. Verify MLflow artifact upload works (same-region = fast)

## Rules Derived

1. **Follow the plan you wrote.** If there's an XML plan with phases, execute Phase 1
   before Phase 2. Don't skip infrastructure to show a "quick win."

2. **When the user says "everything on X", they mean EVERYTHING.** Docker registry,
   storage, databases, MLflow server — all on the same cloud. Not just GPU compute.

3. **Don't launch training before infrastructure is ready.** Training against the wrong
   MLflow server (UpCloud) proves nothing useful when the goal is GCP-native.

4. **Stop asking questions the user already answered.** If the decision is documented
   in `cloud-architecture-decisions-2026-03-14.md`, read it before asking again.

5. **Impatience wastes more time than patience.** Launching a quick smoke test against
   the wrong infrastructure wastes GPU money and doesn't advance the goal.

## Cost of This Failure

- User frustration (repeated corrections)
- Wasted GCP T4 spot time (~$0.14/hr for a test that proves nothing new)
- Delayed the actual goal (Pulumi GCP stack)
- Trust erosion
