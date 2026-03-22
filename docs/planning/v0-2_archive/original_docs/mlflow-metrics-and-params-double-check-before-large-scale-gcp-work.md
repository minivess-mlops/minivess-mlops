# MLflow Metrics & Params Double-Check Before Large-Scale GCP Training

**Created**: 2026-03-17
**Priority**: P0 — Blocker before GCP L4 factorial experiment runs
**Branch**: `fix/mlflow-metrics-and-params`

## Original User Prompt (verbatim)

> Let's create a P1 Issue on this biostatistic plan and leave this branch open, and let's then later execute it with the self-learning TDD Skill as it is not on the critical path, and as you should be able to figure out from the kg (and if not, it needs more work!) that the Flows are now decoupled so that we can do all the training flows and train all the models even though our post-training and evals flows are under development. Obviously the critical part is that we are saving all the information needed during this training Flows so that we can later develop the post-training and Evals flow and the biostatistics after that. So we had just the PR for improving benchmarking/profiling logging, and we should have all the possible params and metrics (https://apxml.com/courses/data-versioning-experiment-tracking/chapter-3-tracking-experiments-mlflow/logging-params-metrics-mlflow) logged to MLflow server. So parameters such as CPU, GPU, amount of RAM, and the experiment params from hydra conf with all the hyperparameter, and the timing/profiling/benchmarking could be considered as a metric with some prefix to separate easily from the "scientific metrics" such as MASD and clDice, or what do you think, all the metrics could be semantically grouped with some prefix? and then the CI from ensembles or some UQ method could be encoded as a suffix such as _ci-lo and _ci-hi so that the metrics and params are easy to parse from the flat structure as mlflow I guess does not support nested metrics and params like in a nested dict? has this been properly explored? Let's run a planning session with multiple reviewer agents again optimizing a plan and report to improve this as a P0 Issue and a blocker before we start large-scale training on GCP L4! See e.g. the MLflow practices from my previous repo that had this MLflow as contract design, but not totally sure if the MLflow metric and param structure is optimal (/home/petteri/mlruns) but it is decent and not abysmal. See especially the classification experiment there in which I had used bootstrapping which gave me AUROC, AUROC_ci_lo and AUROC_ci_hi type of names for metrics, but the semantic grouping could have been nicer. Create another branch on top of the biostatistic branch called fix/mlflow-metrics-and-params and let's plan to .md file in /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/mlflow-metrics-and-params-double-check-before-large-scale-gcp-work! And start saving my prompt there verbatim and also updating the kg for relevant mlflow and general experiment flow insights with reviewer agents

---

## Q&A Session Log

### Round 1

**Q1: Slash-prefix convention migration?**
- **Answer**: Migrate to slash-prefix. `val_dice` -> `val/dice`, `sys_gpu_model` -> `sys/gpu_model`.
  MLflow 2.11+ auto-groups by slash prefix in the UI.

**Q2: Logging gaps?**
- **Answer**: Fill ALL 10 gaps before large-scale runs. Training is one-shot -- can't re-run to add missing metrics.

---

## 1. Current State Audit

**113+ logged items across 6 categories.** The current underscore convention (e.g. `sys_gpu_model`, `val_loss`, `cost_total_usd`) does NOT auto-group in the MLflow UI. MLflow 2.11+ supports slash-delimited metric grouping (e.g. `sys/gpu_model`, `val/loss`, `cost/total_usd`) which collapses into expandable sections in both the Experiment table and Run detail pages.

### 1.1 Complete Inventory Table

Items are grouped by their current prefix convention. Source files verified by reading every `mlflow.log_param`, `mlflow.log_metric`, `mlflow.set_tag`, and `mlflow.log_artifact` call in the codebase.

#### A. Params (logged once at run start)

| # | Current Key | Value Example | Source File | Logged By |
|---|-------------|---------------|-------------|-----------|
| 1 | `model_family` | `"dynunet"` | `tracking.py` | `_log_config()` |
| 2 | `model_name` | `"dynunet"` | `tracking.py` | `_log_config()` |
| 3 | `in_channels` | `1` | `tracking.py` | `_log_config()` |
| 4 | `out_channels` | `2` | `tracking.py` | `_log_config()` |
| 5 | `batch_size` | `2` | `tracking.py` | `_log_config()` |
| 6 | `learning_rate` | `0.001` | `tracking.py` | `_log_config()` |
| 7 | `max_epochs` | `100` | `tracking.py` | `_log_config()` |
| 8 | `optimizer` | `"adamw"` | `tracking.py` | `_log_config()` |
| 9 | `scheduler` | `"cosine"` | `tracking.py` | `_log_config()` |
| 10 | `seed` | `42` | `tracking.py` | `_log_config()` |
| 11 | `num_folds` | `3` | `tracking.py` | `_log_config()` |
| 12 | `mixed_precision` | `True` | `tracking.py` | `_log_config()` |
| 13 | `weight_decay` | `0.0001` | `tracking.py` | `_log_config()` |
| 14 | `warmup_epochs` | `5` | `tracking.py` | `_log_config()` |
| 15 | `gradient_clip_val` | `1.0` | `tracking.py` | `_log_config()` |
| 16 | `gradient_checkpointing` | `False` | `tracking.py` | `_log_config()` |
| 17 | `early_stopping_patience` | `20` | `tracking.py` | `_log_config()` |
| 18 | `arch_{key}` (N keys) | `"[32,64,128]"` | `tracking.py` | `_log_config()` |
| 19 | `model_{key}` (N keys) | varies | `tracking.py` | `log_model_info()` |
| 20 | `trainable_parameters` | `4200000` | `tracking.py` | `log_model_info()` |
| 21 | `split_mode` | `"file"` | `tracking.py` | `log_fold_splits()` |
| 22 | `sys_python_version` | `"3.12.8"` | `system_info.py` | `get_system_params()` |
| 23 | `sys_os` | `"Linux"` | `system_info.py` | `get_system_params()` |
| 24 | `sys_os_kernel` | `"6.8.0-101"` | `system_info.py` | `get_system_params()` |
| 25 | `sys_hostname` | `"gpu-node-01"` | `system_info.py` | `get_system_params()` |
| 26 | `sys_total_ram_gb` | `"31.2"` | `system_info.py` | `get_system_params()` |
| 27 | `sys_cpu_model` | `"AMD Ryzen 9"` | `system_info.py` | `get_system_params()` |
| 28 | `sys_torch_version` | `"2.5.1"` | `system_info.py` | `get_library_versions()` |
| 29 | `sys_cuda_version` | `"12.4"` | `system_info.py` | `get_library_versions()` |
| 30 | `sys_cudnn_version` | `"90100"` | `system_info.py` | `get_library_versions()` |
| 31 | `sys_monai_version` | `"1.4.0"` | `system_info.py` | `get_library_versions()` |
| 32 | `sys_mlflow_version` | `"2.19.0"` | `system_info.py` | `get_library_versions()` |
| 33 | `sys_numpy_version` | `"2.1.3"` | `system_info.py` | `get_library_versions()` |
| 34 | `sys_gpu_count` | `"1"` | `system_info.py` | `get_gpu_info()` |
| 35 | `sys_gpu_model` | `"RTX 4090"` | `system_info.py` | `get_gpu_info()` |
| 36 | `sys_gpu_vram_mb` | `"24564"` | `system_info.py` | `get_gpu_info()` |
| 37 | `sys_git_commit` | `"abc123..."` | `system_info.py` | `get_git_info()` |
| 38 | `sys_git_commit_short` | `"abc123"` | `system_info.py` | `get_git_info()` |
| 39 | `sys_git_branch` | `"main"` | `system_info.py` | `get_git_info()` |
| 40 | `sys_git_dirty` | `"false"` | `system_info.py` | `get_git_info()` |
| 41 | `sys_dvc_version` | `"3.59.1"` | `system_info.py` | `get_dvc_info()` |
| 42 | `data_n_volumes` | `"70"` | `profiler.py` | `DatasetProfile.to_mlflow_params()` |
| 43 | `data_total_size_gb` | `"1.23"` | `profiler.py` | `DatasetProfile.to_mlflow_params()` |
| 44 | `data_min_shape` | `"(512,512,10)"` | `profiler.py` | `DatasetProfile.to_mlflow_params()` |
| 45 | `data_max_shape` | `"(512,512,54)"` | `profiler.py` | `DatasetProfile.to_mlflow_params()` |
| 46 | `data_median_shape` | `"(512,512,28)"` | `profiler.py` | `DatasetProfile.to_mlflow_params()` |
| 47 | `data_min_spacing` | `"(0.31,0.31,1.0)"` | `profiler.py` | `DatasetProfile.to_mlflow_params()` |
| 48 | `data_max_spacing` | `"(0.62,0.62,4.97)"` | `profiler.py` | `DatasetProfile.to_mlflow_params()` |
| 49 | `data_median_spacing` | `"(0.31,0.31,1.0)"` | `profiler.py` | `DatasetProfile.to_mlflow_params()` |
| 50 | `data_n_outlier_volumes` | `"1"` | `profiler.py` | `DatasetProfile.to_mlflow_params()` |
| 51 | `cfg_project_name` | `"minivess"` | `tracking.py` | `log_dynaconf_config()` |
| 52 | `cfg_data_dir` | `"/app/data"` | `tracking.py` | `log_dynaconf_config()` |
| 53 | `cfg_dvc_remote` | `"gcs"` | `tracking.py` | `log_dynaconf_config()` |
| 54 | `cfg_mlflow_tracking_uri` | `"mlruns"` | `tracking.py` | `log_dynaconf_config()` |
| 55 | `setup_python_install_seconds` | `35.2` | `infrastructure_timing.py` | `log_infrastructure_timing()` |
| 56 | `setup_uv_install_seconds` | `12.5` | `infrastructure_timing.py` | `log_infrastructure_timing()` |
| 57 | `setup_uv_sync_seconds` | `28.3` | `infrastructure_timing.py` | `log_infrastructure_timing()` |
| 58 | `setup_dvc_config_seconds` | `1.2` | `infrastructure_timing.py` | `log_infrastructure_timing()` |
| 59 | `setup_dvc_pull_seconds` | `45.0` | `infrastructure_timing.py` | `log_infrastructure_timing()` |
| 60 | `setup_model_weights_seconds` | `15.3` | `infrastructure_timing.py` | `log_infrastructure_timing()` |
| 61 | `setup_verification_seconds` | `3.1` | `infrastructure_timing.py` | `log_infrastructure_timing()` |
| 62 | `setup_total_seconds` | `140.6` | `infrastructure_timing.py` | `log_infrastructure_timing()` |
| 63 | `prof_cfg_{key}` (N keys) | varies | `tracking.py` | `log_profiling_artifacts()` |
| 64 | `sys_bench_gpu_model` | `"RTX 4090"` | `gpu_profile.py` | `load_benchmark_params()` |
| 65 | `sys_bench_total_vram_mb` | `24564` | `gpu_profile.py` | `load_benchmark_params()` |
| 66 | `sys_bench_{model}_vram_peak_mb` | `3200` | `gpu_profile.py` | `load_benchmark_params()` |
| 67 | `sys_bench_{model}_throughput` | `12.5` | `gpu_profile.py` | `load_benchmark_params()` |
| 68 | `sys_bench_{model}_forward_ms` | `80.0` | `gpu_profile.py` | `load_benchmark_params()` |
| 69 | `sys_bench_{model}_feasible` | `True` | `gpu_profile.py` | `load_benchmark_params()` |
| 70 | `data_n_folds` (data_flow) | `3` | `data_flow.py` | `run_data_flow()` |
| 71 | `data_hash` (data_flow) | `"abc123..."` | `data_flow.py` | `run_data_flow()` |
| 72 | `data_dvc_commit` (data_flow) | `"def456..."` | `data_flow.py` | `run_data_flow()` |
| 73 | `lineage_summary` (biostats) | `"{...}"` | `biostatistics_mlflow.py` | `log_biostatistics_run()` |
| 74 | `acq_n_datasets` (acquisition) | `4` | `acquisition_flow.py` | `run_acquisition_flow()` |
| 75 | `acq_total_volumes` (acquisition) | `70` | `acquisition_flow.py` | `run_acquisition_flow()` |
| 76 | `volume_id` (annotation) | `"mv01"` | `approval.py` | `approve_annotation()` |
| 77 | `label_path` (annotation) | `"/data/..."` | `approval.py` | `approve_annotation()` |
| 78 | `dvc_tag` (annotation) | `"annotation/v..."` | `approval.py` | `approve_annotation()` |
| 79 | `annotation_timestamp` | ISO timestamp | `approval.py` | `approve_annotation()` |

#### B. Per-Epoch Metrics (logged with `step=epoch`)

| # | Current Key | Type | Source File | Logged By |
|---|-------------|------|-------------|-----------|
| 1 | `train_loss` | float | `trainer.py` | `fit()` -> `tracker.log_epoch_metrics()` |
| 2 | `val_loss` | float | `trainer.py` | `fit()` -> `tracker.log_epoch_metrics()` |
| 3 | `learning_rate` | float | `trainer.py` | `fit()` -> `tracker.log_epoch_metrics()` |
| 4 | `train_dice` | float | `trainer.py` | via `SegmentationMetrics.compute()` |
| 5 | `train_f1_foreground` | float | `trainer.py` | via `SegmentationMetrics.compute()` |
| 6 | `val_dice` | float | `trainer.py` | via `SegmentationMetrics.compute()` |
| 7 | `val_f1_foreground` | float | `trainer.py` | via `SegmentationMetrics.compute()` |
| 8 | `val_cldice` | float (every 5 epochs) | `trainer.py` | via `_compute_extended_metrics()` |
| 9 | `val_masd` | float (every 5 epochs) | `trainer.py` | via `_compute_extended_metrics()` |
| 10 | `val_compound_masd_cldice` | float (every 5 epochs) | `trainer.py` | via `_compute_extended_metrics()` |
| 11 | `sys_gpu_{key}` | float | `trainer.py` | via `system_monitor.epoch_summary()` |
| 12 | `prof_first_epoch_seconds` | float (epoch 0 only) | `trainer.py` | `fit()` |
| 13 | `prof_steady_epoch_seconds` | float (epoch 1 only) | `trainer.py` | `fit()` |

#### C. Fold-Level Metrics (logged once per fold to parent run)

| # | Current Key | Source File | Logged By |
|---|-------------|-------------|-----------|
| 1 | `fold_{id}_best_val_loss` | `train_flow.py` | `log_fold_results_task()` |
| 2 | `fold_{id}_final_epoch` | `train_flow.py` | `log_fold_results_task()` |
| 3 | `fold_{id}_val_loss` (per step) | `train_flow.py` | `log_fold_results_task()` |
| 4 | `val_loss` (cross-fold, per step) | `train_flow.py` | `log_fold_results_task()` |
| 5 | `vram_peak_mb` | `train_flow.py` | `log_fold_results_task()` |
| 6 | `n_folds_completed` | `train_flow.py` | `training_flow()` |

#### D. Flow-Level / Cost Metrics (logged once at flow end)

| # | Current Key | Source File | Logged By |
|---|-------------|-------------|-----------|
| 1 | `cost_total_wall_seconds` | `infrastructure_timing.py` | `log_cost_analysis()` |
| 2 | `cost_total_usd` | `infrastructure_timing.py` | `log_cost_analysis()` |
| 3 | `cost_setup_usd` | `infrastructure_timing.py` | `log_cost_analysis()` |
| 4 | `cost_training_usd` | `infrastructure_timing.py` | `log_cost_analysis()` |
| 5 | `cost_effective_gpu_rate` | `infrastructure_timing.py` | `log_cost_analysis()` |
| 6 | `cost_setup_fraction` | `infrastructure_timing.py` | `log_cost_analysis()` |
| 7 | `cost_gpu_utilization_fraction` | `infrastructure_timing.py` | `log_cost_analysis()` |
| 8 | `cost_epochs_to_amortize_setup` | `infrastructure_timing.py` | `log_cost_analysis()` |
| 9 | `cost_break_even_epochs` | `infrastructure_timing.py` | `log_cost_analysis()` |
| 10 | `estimated_total_cost` | `infrastructure_timing.py` | `estimate_cost_from_first_epoch()` |
| 11 | `estimated_total_hours` | `infrastructure_timing.py` | `estimate_cost_from_first_epoch()` |
| 12 | `cost_per_epoch` | `infrastructure_timing.py` | `estimate_cost_from_first_epoch()` |
| 13 | `epoch_seconds` | `infrastructure_timing.py` | `estimate_cost_from_first_epoch()` |

#### E. Profiling Metrics (logged once after training)

| # | Current Key | Source File | Logged By |
|---|-------------|-------------|-----------|
| 1 | `prof_overhead_pct` | `tracking.py` | `log_profiling_artifacts()` |
| 2 | `prof_trace_size_mb` | `tracking.py` | `log_profiling_artifacts()` |
| 3 | `prof_data_to_device_fraction` | `tracking.py` | `log_profiling_artifacts()` |
| 4 | `prof_forward_fraction` | `tracking.py` | `log_profiling_artifacts()` |
| 5 | `prof_backward_fraction` | `tracking.py` | `log_profiling_artifacts()` |
| 6 | `trainable_parameters` (metric) | `tracking.py` | `log_model_info()` |

#### F. Tags (string metadata, not filterable as numeric)

| # | Current Key | Source File | Logged By |
|---|-------------|-------------|-----------|
| 1 | `model_family` | `tracking.py` | `start_run()` |
| 2 | `model_name` | `tracking.py` | `start_run()` |
| 3 | `started_at` | `tracking.py` | `start_run()` |
| 4 | `error_type` (on failure) | `tracking.py` | `start_run()` |
| 5 | `error_message` (on failure) | `tracking.py` | `start_run()` |
| 6 | `git_commit` | `tracking.py` | `log_git_hash()` |
| 7 | `test_set_sha256` | `tracking.py` | `log_test_set_hash()` |
| 8 | `test_set_locked_at` | `tracking.py` | `log_test_set_hash()` |
| 9 | `dvc_data_hash` | `tracking.py` | `log_dvc_provenance()` |
| 10 | `dvc_data_nfiles` | `tracking.py` | `log_dvc_provenance()` |
| 11 | `dvc_data_path` | `tracking.py` | `log_dvc_provenance()` |
| 12 | `dvc_remote_url` | `tracking.py` | `log_dvc_provenance()` |
| 13 | `cfg_environment` | `tracking.py` | `log_dynaconf_config()` |
| 14 | `splits_file` | `tracking.py` | `log_fold_splits()` |
| 15 | `fold_{id}_train` | `tracking.py` | `log_fold_splits()` |
| 16 | `fold_{id}_val` | `tracking.py` | `log_fold_splits()` |
| 17 | `best_{metric_name}` | `tracking.py` | `log_post_training_tags()` |
| 18 | `fold_id` | `tracking.py` / `train_flow.py` | `log_post_training_tags()` / child run tags |
| 19 | `loss_type` | `tracking.py` | `log_post_training_tags()` |
| 20 | `loss_function` | `train_flow.py` | child run tags + parent run tags |
| 21 | `config_fingerprint` | `train_flow.py` | child run tags |
| 22 | `flow_name` | `train_flow.py` / all flows | parent run tags |
| 23 | `parent_run_id` | `train_flow.py` | `training_flow()` |
| 24 | `upstream_data_run_id` | `train_flow.py` | `training_flow()` |
| 25 | `checkpoint_dir_fold_{id}` | `train_flow.py` | `log_fold_results_task()` |
| 26 | `flow_status` | `flow_contract.py` | `log_flow_completion()` |
| 27 | `flow_artifacts` | `flow_contract.py` | `log_flow_completion()` |
| 28 | `checkpoint_dir` | `flow_contract.py` | `log_flow_completion()` |
| 29 | `termination_signal` | `ghost_cleanup.py` | signal handler |
| 30 | `ghost_cleanup` | `ghost_cleanup.py` | `cleanup_ghost_runs()` |
| 31 | `upstream_fingerprint` (biostats) | `biostatistics_mlflow.py` | `log_biostatistics_run()` |
| 32 | `n_source_runs` (biostats) | `biostatistics_mlflow.py` | `log_biostatistics_run()` |
| 33 | `n_conditions` (biostats) | `biostatistics_mlflow.py` | `log_biostatistics_run()` |
| 34 | `n_figures` (biostats) | `biostatistics_mlflow.py` | `log_biostatistics_run()` |
| 35 | `n_tables` (biostats) | `biostatistics_mlflow.py` | `log_biostatistics_run()` |
| 36 | `splits_path` (data_flow) | `data_flow.py` | `run_data_flow()` |
| 37 | `upstream_training_run_id` (post) | `post_training_flow.py` | `post_training_flow()` |

#### G. Artifacts (files logged to MLflow run)

| # | Artifact Path | Format | Source File | Logged By |
|---|---------------|--------|-------------|-----------|
| 1 | `profiling/*.json.gz` | Chrome trace | `tracking.py` | `log_profiling_artifacts()` |
| 2 | `profiling/key_averages.txt` | Text | `tracking.py` | `log_profiling_artifacts()` |
| 3 | `profiling/summary.json` | JSON | `tracking.py` | `log_profiling_artifacts()` |
| 4 | `environment/frozen_deps_*.txt` | Text | `tracking.py` | `log_frozen_deps()` |
| 5 | `config/resolved_config.yaml` | YAML | `tracking.py` | `log_hydra_config()` |
| 6 | `config/settings.toml` | TOML | `tracking.py` | `log_dynaconf_config()` |
| 7 | `config/dvc.yaml` | YAML | `tracking.py` | `log_dvc_provenance()` |
| 8 | `config/dvc.lock` | YAML | `tracking.py` | `log_dvc_provenance()` |
| 9 | `dataset/dataset_profile_*.json` | JSON | `tracking.py` | `log_dataset_profile()` |
| 10 | `splits/splits.json` | JSON | `tracking.py` | `log_fold_splits()` |
| 11 | `per_volume_metrics/fold{id}_{loss}.json` | JSON | `tracking.py` | `log_per_volume_metrics()` |
| 12 | `checkpoints/best_*.pth` | PyTorch | `trainer.py` | `fit()` -> `tracker.log_artifact()` |
| 13 | `checkpoints/last.pth` | PyTorch | `trainer.py` | `fit()` -> `tracker.log_artifact()` |
| 14 | `history/metric_history.json` | JSON | `trainer.py` | `fit()` -> `tracker.log_artifact()` |
| 15 | `timing/timing_setup.txt` | Text | `infrastructure_timing.py` | `log_infrastructure_timing()` |
| 16 | `timing/timing_report.jsonl` | JSONL | `train_flow.py` | `log_timing_jsonl_artifact()` |
| 17 | `model/` (pyfunc) | MLflow model | `tracking.py` | `log_pyfunc_model()` |
| 18 | `figures/*` (biostats) | PNG/PDF | `biostatistics_mlflow.py` | `log_biostatistics_run()` |
| 19 | `tables/*` (biostats) | CSV | `biostatistics_mlflow.py` | `log_biostatistics_run()` |
| 20 | `duckdb/*.duckdb` (biostats) | DuckDB | `biostatistics_mlflow.py` | `log_biostatistics_run()` |
| 21 | `sidecars/*` (biostats) | JSON | `biostatistics_mlflow.py` | `log_biostatistics_run()` |

#### H. Evaluation Metrics (logged by evaluation_runner in analysis_flow)

| # | Current Key Pattern | Source File |
|---|---------------------|-------------|
| 1 | `eval_fold{id}_{metric}` (point) | `tracking.py` via `log_evaluation_results()` |
| 2 | `eval_fold{id}_{metric}_ci_lower` | `tracking.py` via `log_evaluation_results()` |
| 3 | `eval_fold{id}_{metric}_ci_upper` | `tracking.py` via `log_evaluation_results()` |
| 4 | `eval_{dataset}_{subset}_{metric}` | `evaluation_runner.py` via `log_results_to_mlflow()` |
| 5 | `eval_{dataset}_{subset}_{metric}_ci_lower` | `evaluation_runner.py` |
| 6 | `eval_{dataset}_{subset}_{metric}_ci_upper` | `evaluation_runner.py` |
| 7 | `eval_{dataset}_{subset}_compound_masd_cldice` | `evaluation_runner.py` |
| 8 | `post_{plugin}_{metric}` | `post_training_flow.py` |
| 9 | `n_plugins_run` | `post_training_flow.py` |

**Total: ~79 fixed params + N dynamic arch/model params, ~13 per-epoch metrics, ~6 fold-level metrics, ~13 cost/flow metrics, ~6 profiling metrics, ~37 tags, ~21 artifact types, ~9+ evaluation metric patterns.**

---

## 2. Proposed Slash-Prefix Namespace

MLflow 2.11+ (released 2024-03) introduced auto-grouping in the Experiment Runs table and the Run detail Metrics tab based on slash (`/`) delimiters. Metrics named `val/dice` and `val/loss` are automatically grouped under a collapsible `val/` section. This removes the need for manual visual scanning of flat underscore-delimited lists.

### 2.1 Semantic Groups

| Group | Prefix | Examples | When Logged | Type |
|-------|--------|----------|-------------|------|
| Training metrics | `train/` | `train/loss`, `train/dice`, `train/f1_foreground` | Per epoch | metric |
| Validation metrics | `val/` | `val/loss`, `val/dice`, `val/cldice`, `val/masd`, `val/compound_masd_cldice` | Per val_interval | metric |
| Learning rate | `optim/` | `optim/lr` | Per epoch | metric |
| System info | `sys/` | `sys/gpu_model`, `sys/torch_version`, `sys/total_ram_gb` | Once at start | param |
| Cost/FinOps | `cost/` | `cost/total_usd`, `cost/setup_fraction`, `cost/gpu_utilization_fraction` | Once at end | metric |
| Setup timing | `setup/` | `setup/uv_sync_seconds`, `setup/total_seconds` | Once at start | param |
| Profiling | `prof/` | `prof/overhead_pct`, `prof/forward_fraction`, `prof/first_epoch_seconds` | Once at end | metric |
| Architecture | `arch/` | `arch/filters`, `arch/strides` | Once at start | param |
| Data | `data/` | `data/n_volumes`, `data/patch_size`, `data/total_size_gb` | Once at start | param |
| Fold-specific | `fold/` | `fold/0/best_val_loss`, `fold/1/best_val_loss` | Once per fold | metric |
| Gradient | `grad/` | `grad/norm_mean`, `grad/clip_count` | Per epoch (NEW) | metric |
| Inference | `infer/` | `infer/latency_ms`, `infer/throughput_vps` | Once at end (NEW) | metric |
| Evaluation | `eval/` | `eval/fold0/dice`, `eval/minivess/all/cldice` | Once per eval run | metric |
| Config (Dynaconf) | `cfg/` | `cfg/project_name`, `cfg/dvc_remote` | Once at start | param |
| GPU monitoring | `gpu/` | `gpu/utilization_pct`, `gpu/mem_bw_pct`, `gpu/temp_c` | Per epoch | metric |
| Model info | `model/` | `model/trainable_params`, `model/family`, `model/name` | Once at start | param |
| Benchmark | `bench/` | `bench/gpu_model`, `bench/{model}/vram_peak_mb` | Once at start | param |
| Estimate | `est/` | `est/total_cost`, `est/total_hours` | Once (epoch 0) | metric |
| VRAM | `vram/` | `vram/peak_mb`, `vram/peak_gb` | Once at end | metric |
| Post-training | `post/` | `post/swa/val_loss`, `post/calibration/ece` | Once per plugin | metric |

### 2.2 Complete Migration Mapping (old -> new)

**Params:**
```
model_family         -> model/family
model_name           -> model/name
in_channels          -> model/in_channels
out_channels         -> model/out_channels
batch_size           -> train/batch_size
learning_rate        -> train/learning_rate
max_epochs           -> train/max_epochs
optimizer            -> train/optimizer
scheduler            -> train/scheduler
seed                 -> train/seed
num_folds            -> train/num_folds
mixed_precision      -> train/mixed_precision
weight_decay         -> train/weight_decay
warmup_epochs        -> train/warmup_epochs
gradient_clip_val    -> train/gradient_clip_val
gradient_checkpointing -> train/gradient_checkpointing
early_stopping_patience -> train/early_stopping_patience
arch_{key}           -> arch/{key}
model_{key}          -> model/{key}
trainable_parameters -> model/trainable_params
split_mode           -> data/split_mode
sys_*                -> sys/*
data_*               -> data/*
cfg_*                -> cfg/*
setup_*              -> setup/*
prof_cfg_*           -> prof/cfg/*
sys_bench_*          -> bench/*
```

**Per-epoch metrics:**
```
train_loss           -> train/loss
val_loss             -> val/loss
learning_rate        -> optim/lr
train_dice           -> train/dice
train_f1_foreground  -> train/f1_foreground
val_dice             -> val/dice
val_f1_foreground    -> val/f1_foreground
val_cldice           -> val/cldice
val_masd             -> val/masd
val_compound_masd_cldice -> val/compound_masd_cldice
sys_gpu_{key}        -> gpu/{key}
prof_first_epoch_seconds -> prof/first_epoch_seconds
prof_steady_epoch_seconds -> prof/steady_epoch_seconds
```

**Fold-level metrics:**
```
fold_{id}_best_val_loss -> fold/{id}/best_val_loss
fold_{id}_final_epoch   -> fold/{id}/final_epoch
fold_{id}_val_loss      -> fold/{id}/val_loss
vram_peak_mb            -> vram/peak_mb
n_folds_completed       -> fold/n_completed
```

**Cost metrics:**
```
cost_total_wall_seconds -> cost/total_wall_seconds
cost_total_usd          -> cost/total_usd
(etc. — all cost_ -> cost/)
estimated_total_cost    -> est/total_cost
estimated_total_hours   -> est/total_hours
cost_per_epoch          -> est/cost_per_epoch
epoch_seconds           -> est/epoch_seconds
```

---

## 3. CI/UQ Encoding Convention

MLflow's flat key-value structure does not support nested dicts. We encode confidence intervals and cross-fold aggregates using slash-delimited suffixes:

```
val/dice           # point estimate (per-epoch, logged with step=epoch)
val/dice/mean      # cross-fold mean (post-training, logged once)
val/dice/std       # cross-fold std
val/dice/ci95_lo   # 95% CI lower bound
val/dice/ci95_hi   # 95% CI upper bound
val/dice/fold_0    # per-fold value
val/dice/fold_1
val/dice/fold_2
```

**Evaluation metrics (per-dataset, per-subset):**
```
eval/minivess/all/dice            # point estimate
eval/minivess/all/dice/ci95_lo    # 95% CI lower
eval/minivess/all/dice/ci95_hi    # 95% CI upper
eval/minivess/thin_vessels/masd   # per-subset
eval/deepvess/all/cldice          # external dataset
```

**Benefits:**
- MLflow UI groups all `val/dice/*` together
- DuckDB queries: `WHERE key LIKE 'val/dice/%'` extracts all CI variants
- Foundation-PLR convention: `AUROC_ci_lo` -> `eval/{metric}/ci95_lo` (more semantic)
- Suffix convention is consistent: `/mean`, `/std`, `/ci95_lo`, `/ci95_hi`, `/fold_{id}`

---

## 4. 10 Logging Gaps -- Implementation Plan

### Gap 1: Data Augmentation Config (P0 -- blocking)

Training is one-shot. If augmentation config is not logged, we cannot attribute metric differences to augmentation choices in biostatistics analysis.

- **Keys to add (params):** `data/augmentation_pipeline` (string summary), `data/num_samples` (RandCropByPosNegLabeld num_samples), `data/cache_rate`
- **Where:** `tracking.py::_log_config()` -- extract from the Hydra config dict that is already passed to `log_hydra_config()`
- **Priority:** P0

### Gap 2: Gradient Norms (P1 -- important for debugging)

Gradient norm tracking is essential for diagnosing training instability, especially with BF16/AMP and the SAM3 ViT-32L encoder.

- **Keys to add (metrics):** `grad/norm_mean`, `grad/norm_max`, `grad/clip_count` (per epoch)
- **Where:** `trainer.py::train_epoch()` -- after `clip_grad_norm_()`, capture the returned total norm. Accumulate clip count when norm exceeds threshold.
- **Priority:** P1

### Gap 3: Per-Class Metrics (SKIP)

MiniVess is binary segmentation (vessel vs. background). Per-class metrics are N/A. If multi-class tasks are added later, the `SegmentationMetrics` class already supports `num_classes > 2`. No action needed now.

- **Priority:** N/A (skip)

### Gap 4: Inference Latency (P1 -- needed for deployment decisions)

Post-training inference speed determines serving strategy (BentoML batch vs. real-time). Currently not logged anywhere.

- **Keys to add (metrics):** `infer/latency_ms_per_volume`, `infer/throughput_volumes_per_sec`, `infer/sliding_window_patches`
- **Where:** `trainer.py::validate_epoch()` -- wrap the validation loop with `time.perf_counter()` and divide by number of volumes. Alternatively, add a dedicated `benchmark_inference()` method.
- **Priority:** P1

### Gap 5: Batch-Level Metrics (P2 -- optional, high cardinality)

Per-batch loss is useful for debugging data issues (outlier volumes) but creates high-cardinality metric series.

- **Keys to add (metrics):** `train/batch_loss` (optionally, with batch_idx as step)
- **Where:** `trainer.py::train_epoch()` -- log inside the batch loop
- **Decision:** Log only when `debug=True` or when a `log_batch_metrics` config flag is set. Default: OFF.
- **Priority:** P2 (defer to post-GCP-runs)

### Gap 6: Optimizer State (P0 -- blocking)

Learning rate is already logged. Momentum and other optimizer state are not.

- **Keys to add (metrics):** `optim/lr` (already exists as `learning_rate`, rename only), `optim/weight_decay` (static, already a param -- no action), `optim/grad_scale` (AMP scaler scale factor)
- **Where:** `trainer.py::fit()` epoch loop -- `self.scaler.get_scale()` for AMP scale
- **Priority:** P0 (the `optim/lr` rename is part of the migration; `optim/grad_scale` is new)

### Gap 7: Validation Timing (P1 -- needed for cost modeling)

We log `prof_first_epoch_seconds` and `prof_steady_epoch_seconds` but not validation time specifically. For SAM3, validation can be 10x slower than training due to sliding-window inference.

- **Keys to add (metrics):** `prof/val_seconds` (per val epoch), `prof/train_seconds` (per train epoch)
- **Where:** `trainer.py::fit()` -- wrap `train_epoch()` and `validate_epoch()` with `time.perf_counter()`
- **Priority:** P1

### Gap 8: Early Stopping Details (P1 -- needed for understanding convergence)

We log `early_stopping_patience` as a param but not the running patience counter. Biostatistics analysis needs to know whether runs were patience-limited or epoch-limited.

- **Keys to add (metrics):** `train/patience_counter` (per epoch, from MultiMetricTracker), `train/stopped_early` (boolean, logged once at end)
- **Where:** `trainer.py::fit()` -- query `self._multi_tracker` for counter state; log once at end whether the loop terminated by early stopping or max_epochs.
- **Priority:** P1

### Gap 9: Checkpoint Metadata (P1 -- needed for post-training flow)

Checkpoint file size is not logged. Post-training flow needs to know checkpoint sizes for storage cost estimation.

- **Keys to add (metrics):** `checkpoint/size_mb` (per save), `checkpoint/epoch` (per save -- already in metric_history.json but not as MLflow metric)
- **Where:** `trainer.py::fit()` -- after `save_metric_checkpoint()`, stat the file and log size
- **Priority:** P1

### Gap 10: Data Pipeline Metrics (P2 -- nice to have)

Data loading time and cache hit rate are not explicitly logged. MONAI's CacheDataset provides cache metrics but they are not surfaced.

- **Keys to add (metrics):** `data/load_time_seconds` (per epoch), `data/cache_hit_rate` (if using CacheDataset)
- **Where:** `trainer.py::train_epoch()` -- wrap the DataLoader iteration with timing. Cache hit rate requires instrumentation of MONAI CacheDataset.
- **Decision:** Defer cache_hit_rate. Log `data/load_time_seconds` by timing the `for batch in loader` loop separately from the forward/backward pass.
- **Priority:** P2

---

## 5. Migration Strategy

### Step 1: Create migration mapping dict

Create `src/minivess/observability/metric_keys.py` with:
```python
"""Canonical metric/param key definitions and migration mapping.

Single source of truth for ALL MLflow key names.
Used by ExperimentTracker, biostatistics_flow, analytics.py.
"""

# Old key -> New key mapping for backward compatibility
MIGRATION_MAP: dict[str, str] = {
    "train_loss": "train/loss",
    "val_loss": "val/loss",
    "learning_rate": "optim/lr",
    "train_dice": "train/dice",
    "val_dice": "val/dice",
    # ... complete mapping
}
```

### Step 2: Update ExperimentTracker (tracking.py)

- Replace all hardcoded underscore keys with slash-prefix keys
- Update `_log_config()` to use `model/family`, `train/batch_size`, etc.
- Update `_log_system_info_safe()` call site (system_info.py keys change)
- Update `log_epoch_metrics()` -- no change needed (it uses the keys passed to it)
- Update `log_profiling_artifacts()` to use `prof/` prefix

### Step 3: Update system_info.py

- `sys_python_version` -> `sys/python_version`
- `sys_gpu_model` -> `sys/gpu_model`
- All `sys_` -> `sys/` (same dict keys, different prefix convention)

### Step 4: Update infrastructure_timing.py

- `setup_*_seconds` -> `setup/*_seconds`
- `cost_*` -> `cost/*`
- `estimated_*` -> `est/*`

### Step 5: Update train_flow.py metric logging

- `fold_{id}_best_val_loss` -> `fold/{id}/best_val_loss`
- `vram_peak_mb` -> `vram/peak_mb`
- `n_folds_completed` -> `fold/n_completed`

### Step 6: Update trainer.py

- `train_loss` -> `train/loss` (in the `epoch_log` dict)
- `val_loss` -> `val/loss`
- `train_{metric}` -> `train/{metric}`
- `val_{metric}` -> `val/{metric}`
- `sys_gpu_{key}` -> `gpu/{key}`
- Add new gap metrics (gradient norms, validation timing, etc.)

### Step 7: Update biostatistics_flow and analytics.py

- `biostatistics_flow.py` must handle BOTH conventions for backward compatibility with existing MLflow runs
- `analytics.py` DuckDB queries must be updated to use new key patterns
- Add a `normalize_metric_key()` function that maps old -> new for reading legacy runs

### Step 8: Update evaluation_runner.py

- `eval_fold{id}_{metric}` -> `eval/fold{id}/{metric}`
- `eval_{dataset}_{subset}_{metric}` -> `eval/{dataset}/{subset}/{metric}`
- `_ci_lower` -> `/ci95_lo`, `_ci_upper` -> `/ci95_hi`

### Step 9: Add migration test

- `tests/v2/unit/test_metric_key_convention.py`
- Assert ALL keys in `metric_keys.py` use slash prefix
- Assert NO code paths produce underscore-delimited metric keys
- Assert `MIGRATION_MAP` is complete (covers all legacy keys)

### Step 10: Update CLAUDE.md observability section

- Replace the `CLAUDE.md` note about `sys_` prefix
- Update `src/minivess/observability/CLAUDE.md` param prefix documentation
- Document the slash-prefix convention as the new standard

---

## 6. KG Updates

### 6.1 Update `knowledge-graph/domains/observability.yaml`

Add a new decision node for the metric naming convention:

```yaml
  metric_naming_convention:
    status: resolved
    winner: slash_prefix
    rationale: "MLflow 2.11+ auto-groups metrics by slash prefix in UI.
      Migrating from underscore to slash for semantic grouping."
    implementation:
      - src/minivess/observability/metric_keys.py
      - src/minivess/observability/tracking.py
    prd_node: knowledge-graph/decisions/L3-technology/metric_naming.yaml
```

### 6.2 Update `experiment_tracker` decision node

Add notes about the naming convention dependency:

```yaml
  experiment_tracker:
    status: resolved
    winner: mlflow_local
    rationale: "MLflow local filesystem backend + DuckDB analytics.
      Metric keys use slash-prefix convention (MLflow 2.11+)."
    conventions:
      metric_prefix: "slash (e.g. val/dice, sys/gpu_model)"
      ci_suffix: "/ci95_lo, /ci95_hi"
      fold_suffix: "/fold_{id}"
```

### 6.3 Add planning doc reference

```yaml
planning_docs:
  - file: docs/planning/mlflow-metrics-and-params-double-check-before-large-scale-gcp-work.md
    status: planned
```

---

## 7. Files to Modify

| File | Est. Lines Changed | Description |
|------|-------------------|-------------|
| `src/minivess/observability/metric_keys.py` | +120 (NEW) | Canonical key definitions + migration map |
| `src/minivess/observability/tracking.py` | ~80 | Slash-prefix keys in all log methods |
| `src/minivess/observability/system_info.py` | ~30 | `sys_` -> `sys/` prefix |
| `src/minivess/observability/infrastructure_timing.py` | ~40 | `setup_`/`cost_` -> `setup/`/`cost/` |
| `src/minivess/pipeline/trainer.py` | ~60 | Key migration + new gap metrics (grad norms, timing) |
| `src/minivess/orchestration/flows/train_flow.py` | ~30 | Fold-level key migration |
| `src/minivess/pipeline/evaluation_runner.py` | ~20 | Eval key migration |
| `src/minivess/pipeline/biostatistics_mlflow.py` | ~15 | Backward compat key normalization |
| `src/minivess/observability/analytics.py` | ~20 | DuckDB query key updates |
| `src/minivess/data/profiler.py` | ~10 | `data_` -> `data/` in `to_mlflow_params()` |
| `src/minivess/compute/gpu_profile.py` | ~15 | `sys_bench_` -> `bench/` |
| `src/minivess/pipeline/backfill_metadata.py` | ~10 | Key-aware backfill |
| `src/minivess/observability/CLAUDE.md` | ~10 | Update convention docs |
| `CLAUDE.md` (observability section) | ~5 | Update convention reference |
| `knowledge-graph/domains/observability.yaml` | ~20 | New decision node |
| `tests/v2/unit/test_metric_key_convention.py` | +80 (NEW) | Convention enforcement test |
| `tests/v2/unit/test_tracking.py` | ~40 | Update expected keys |
| `tests/v2/unit/test_system_info.py` | ~20 | Update expected keys |
| `tests/v2/unit/test_infrastructure_timing.py` | ~15 | Update expected keys |
| `tests/v2/unit/test_train_flow.py` | ~20 | Update expected keys |

**Total: ~2 new files, ~18 modified files, ~630 lines changed**

---

## 8. TDD Task Breakdown

All tasks follow the self-learning-iterative-coder skill: RED (failing test) -> GREEN (implement) -> VERIFY (run all) -> FIX -> CHECKPOINT.

### T1: Canonical Key Module + ExperimentTracker Migration
- **RED:** Write `test_metric_key_convention.py` asserting all keys in `metric_keys.py` use `/` separator, no `_` separator in metric group prefix position.
- **GREEN:** Create `metric_keys.py` with complete migration map. Update `tracking.py::_log_config()` to use new keys.
- **VERIFY:** `make test-staging`
- **Est. effort:** 2h

### T2: train_flow.py Metric Key Migration
- **RED:** Update `test_train_flow.py` to assert fold metrics use `fold/{id}/` prefix.
- **GREEN:** Update `log_fold_results_task()` and `training_flow()` to use slash-prefix keys.
- **VERIFY:** `make test-staging`
- **Est. effort:** 1h

### T3: Gradient Norm Logging (New Feature)
- **RED:** Write test asserting `grad/norm_mean` and `grad/clip_count` appear in epoch metrics.
- **GREEN:** In `trainer.py::train_epoch()`, capture return value of `clip_grad_norm_()`, accumulate norms and clip counts, add to `EpochResult.metrics`.
- **VERIFY:** `make test-staging`
- **Est. effort:** 1.5h

### T4: Inference Latency Logging (New Feature)
- **RED:** Write test asserting `infer/latency_ms_per_volume` appears in fit() return dict.
- **GREEN:** In `trainer.py::validate_epoch()`, time the validation loop and compute per-volume latency. Report in fit() return dict. Log in train_flow.py.
- **VERIFY:** `make test-staging`
- **Est. effort:** 1h

### T5: Data Augmentation Config Logging
- **RED:** Write test asserting `data/augmentation_pipeline` param is logged.
- **GREEN:** In `tracking.py::_log_config()` or in `train_one_fold_task()`, extract augmentation config from Hydra config dict and log as params.
- **VERIFY:** `make test-staging`
- **Est. effort:** 1h

### T6: Optimizer State + Validation Timing + Early Stopping Logging
- **RED:** Write test asserting `optim/grad_scale`, `prof/val_seconds`, `train/patience_counter` metrics exist.
- **GREEN:** In `trainer.py::fit()`, add AMP scaler scale logging, wrap train/val epochs with timers, expose MultiMetricTracker patience counter.
- **VERIFY:** `make test-staging`
- **Est. effort:** 2h

### T7: Checkpoint Metadata Logging
- **RED:** Write test asserting `checkpoint/size_mb` metric is logged after checkpoint save.
- **GREEN:** In `trainer.py::fit()`, stat the saved checkpoint file and log size to tracker.
- **VERIFY:** `make test-staging`
- **Est. effort:** 0.5h

### T8: Backward Compatibility in Biostatistics Flow
- **RED:** Write test that feeds old-convention metric keys to biostatistics and verifies they are handled.
- **GREEN:** Add `normalize_metric_key()` in `metric_keys.py` that maps old -> new. Use in `biostatistics_mlflow.py` and `analytics.py` when reading existing runs.
- **VERIFY:** `make test-staging`
- **Est. effort:** 1.5h

### T9: KG Updates + CLAUDE.md Convention Documentation
- **RED:** (No test -- documentation task)
- **GREEN:** Update `observability.yaml` with new decision node. Update `CLAUDE.md` and `src/minivess/observability/CLAUDE.md` with slash-prefix convention.
- **VERIFY:** Manual review
- **Est. effort:** 0.5h

### T10: Integration Test -- Verify All Keys Follow Slash Convention
- **RED:** Write AST-based test that imports all source files, finds string literals matching metric key patterns, and asserts none use underscore-prefix convention.
- **GREEN:** Fix any remaining underscore keys discovered by the test.
- **VERIFY:** `make test-staging`
- **Est. effort:** 2h

**Total estimated effort: ~13 hours across 10 tasks.**

---

## 9. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Existing MLflow runs become unreadable | High | `normalize_metric_key()` backward compat layer in all read paths |
| Tests break due to hardcoded key expectations | Medium | Update all test assertions in the same PR |
| MLflow UI regression (slash rendering) | Low | Verified: MLflow 2.11+ renders slashes as group headers |
| Biostatistics flow breaks on old runs | High | T8 explicitly handles backward compat |
| DuckDB queries break | Medium | Update queries in T8, test with both conventions |

---

## 10. References

- [MLflow 2.11.0 Release Notes](https://github.com/mlflow/mlflow/releases/tag/v2.11.0) -- Metric grouping by slash prefix
- [MLflow Docs: Logging Params, Metrics](https://mlflow.org/docs/latest/tracking/tracking-api.html) -- Official logging API reference
- [APxml: Logging Params & Metrics MLflow](https://apxml.com/courses/data-versioning-experiment-tracking/chapter-3-tracking-experiments-mlflow/logging-params-metrics-mlflow) -- Community best practices (from user prompt)
- [MLflow Best Practices: Metric Naming](https://mlflow.org/docs/latest/tracking/tracking-api.html#logging-metrics) -- Slash convention recommendation
- [Foundation-PLR Classification Experiment](file:///home/petteri/mlruns) -- User's previous repo with `AUROC_ci_lo`/`AUROC_ci_hi` convention (predecessor to this plan's `/ci95_lo`/`/ci95_hi` design)
- [MLflow as Contract Design Pattern](https://mlflow.org/docs/latest/tracking.html#organizing-runs-in-experiments) -- Inter-flow communication via MLflow artifacts
- Maier-Hein et al. (2024). "Metrics Reloaded." *Nature Methods*. -- MetricsReloaded metric definitions used in evaluation

---

## 11. Decision Log

| Decision | Chosen | Alternatives Considered | Rationale |
|----------|--------|------------------------|-----------|
| Slash vs. underscore prefix | Slash (`/`) | Underscore (`_`), dot (`.`) | MLflow 2.11+ auto-groups by slash; underscore is flat; dots conflict with MLflow internal keys |
| CI encoding | `/ci95_lo`, `/ci95_hi` | `_ci_lo`, `_ci_hi`, `[lo]`, `[hi]` | Slash groups CIs with parent metric; underscore doesn't group in UI |
| Fold encoding | `/fold_{id}` | `_fold{id}`, `_f{id}` | Slash groups fold variants under parent metric |
| Gradient norm scope | Per-epoch mean/max | Per-batch, per-layer | Per-epoch is sufficient for convergence monitoring; per-batch creates too many data points; per-layer can be added later as artifacts |
| Batch-level metrics | OFF by default | Always ON, config-gated | High cardinality; only useful for debugging |
| Migration approach | All-at-once PR | Gradual (alias period) | No existing GCP production runs to break; alias period adds complexity without benefit |
