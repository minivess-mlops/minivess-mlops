# Cold-Start Prompt: MLflow Checkpoint Architecture Implementation

Branch: `test/run-debug-gcp-5th-pass`

## CONTEXT

The GCS checkpoint mount (`file_mounts` in `train_factorial.yaml`) violates the
KG invariant `mlflow_only_artifact_contract`. MLflow must be the ONLY persistence
mechanism. Research completed — H2 (Local SSD + ThreadPoolExecutor + mlflow.log_artifact)
is the recommended approach.

## RESEARCH COMPLETED

Full research: `docs/planning/mlflow-async-checkpoint-architecture-research.md`

Key findings:
- `mlflow.log_artifact()` has NO native async variant (FR #14153 closed)
- BUT it IS thread-safe in MLflow 3.x (contextvars, #9235)
- ThreadPoolExecutor wrapping is the correct pattern
- GCP L4 local NVMe: 375 GiB, 90K write IOPS, 62μs latency
- DynUNet checkpoint: ~50MB, 50ms local write, 0.5s GCS upload
- SAM3 checkpoint: ~500MB, 400ms local write, 5s GCS upload

## EXECUTION PLAN

Use `/self-learning-iterative-coder` with the Batch 2 XML plan:
`docs/planning/v0-2_archive/original_docs/run-debug-factorial-experiment-report-6th-pass-post-run-fix-2.xml`

### Phase 1: Implement AsyncCheckpointUploader (~2 hrs TDD)

1. **RED**: Write tests for `AsyncCheckpointUploader`:
   - `test_upload_submits_to_threadpool` — verify non-blocking
   - `test_flush_waits_for_all_uploads` — verify flush blocks until done
   - `test_upload_calls_mlflow_log_artifact` — verify MLflow integration
   - `test_shutdown_flushes_and_closes` — verify clean shutdown
   File: `tests/v2/unit/pipeline/test_async_checkpoint_uploader.py`

2. **GREEN**: Implement `AsyncCheckpointUploader`:
   ```python
   class AsyncCheckpointUploader:
       def __init__(self, max_workers: int = 1):
           self._executor = ThreadPoolExecutor(max_workers=max_workers)
           self._futures: list[Future] = []
       def upload(self, local_path: Path, artifact_subdir: str = "checkpoints") -> None
       def flush(self) -> None
       def shutdown(self) -> None
   ```
   File: `src/minivess/pipeline/checkpoint_manager.py`

3. **Wire into train_flow.py**: Replace checkpoint writes to use uploader
4. **Remove file_mounts** from `train_factorial.yaml`
5. **Update check_resume_state_task()**: Discover checkpoints from MLflow artifacts

### Phase 2: Verify Remaining Batch 1 Tasks (~1 hr)

Phases 2-5 of Batch 1 may already be done by other session. Check and verify:
- Config wiring (test_infra_params_wired.py)
- Metric key consistency (test_metric_key_consistency.py)
- Cross-flow contracts (test_cross_flow_contracts.py)
- Error path coverage (test_error_path_coverage.py)

### Phase 3: 4-Flow Chain Validation (~1 hr)

When GCP jobs complete, validate analysis + biostatistics flows discover
upstream training results correctly.

## MANDATORY: READ BEFORE IMPLEMENTING

- MLflow artifact logging: https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_artifact
- MLflow thread safety: https://github.com/mlflow/mlflow/issues/9235
- MLflow async FR: https://github.com/mlflow/mlflow/issues/14153
- SkyPilot MOUNT_CACHED: https://docs.skypilot.co/en/latest/reference/storage.html
- KG invariant: `knowledge-graph/navigator.yaml` line 191-200

## GCP JOB STATUS

Jobs 3-7+ PENDING (L4 spot queuing). Launch script submitted with resilience
(job_recovery, retry, resume). Check: `uv run sky jobs queue`

## TEST STATUS

Staging: **5990 passed, 0 skipped, 0 failed**

## KEY FILES

```
docs/planning/mlflow-async-checkpoint-architecture-research.md  — full research
docs/planning/v0-2_archive/.../6th-pass-post-run-fix-2.xml     — execution plan
deployment/skypilot/train_factorial.yaml                         — has file_mounts TO REMOVE
src/minivess/orchestration/flows/train_flow.py                  — checkpoint write code
src/minivess/orchestration/flows/train_flow.py:check_resume_state_task  — spot recovery
.claude/metalearning/2026-03-24-competing-checkpoint-mount-still-exists.md
knowledge-graph/navigator.yaml                                   — invariant definition
```
