# User Prompt: Profiler & Benchmarking Plan (Verbatim)

**Date**: 2026-03-13
**Context**: During RunPod/SkyPilot integration testing implementation (T0.1-T0.2 complete)
**Related Issue**: #564 (dockerized GPU benchmark plan)

## Original User Message

> When we start running the Runpod finetuning, for " VesselFM: configs/model_profiles/vesselfm.yaml says ~6 GB inference, ~10 GB fine-tuning. The reviewer agent "corrected" 10→6 by confusing inference with fine-tuning. The smoke test does FINE-TUNING. The original 10 GB was correct.
>
> All VesselFM numbers are ESTIMATES — zero measured data." we should actually then implement benchmarking, so that in the code we can do some pytorch profiling (https://docs.pytorch.org/docs/stable/profiler.html) to see where time is spent on each different parts of the computation. There should be a parameter that should come our defaults.yaml that for example does the profiling for 5 first epochs during "real fine-tuning" and then for the 2 epochs on our debug mode? (e.g. https://docs.pytorch.org/tutorials/beginner/profiler.html, https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html, https://github.com/Project-MONAI/MONAI/blob/dev/monai/utils/profiling.py https://www.runpod.io/articles/guides/monitoring-and-debugging-ai-model-deployments https://www.atlantic.net/gpu-server-hosting/how-to-profile-and-debug-gpu-performance-for-machine-learning-models-on-ubuntu-24-04-gpu-server/ https://medium.com/@serverwalainfra/debugging-gpu-issues-with-ai-how-grok-simplifies-troubleshooting-0e7ccaadb109 https://www.reddit.com/r/CUDA/comments/180996l/what_is_your_experience_developing_on_a_cloud/ https://eunomia.dev/blog/2025/04/21/gpu-profiling-under-the-hood-an-implementation-focused-survey-of-modern-accelerator-tracing-tools/ https://docs.cloud.google.com/vertex-ai/docs/training/tensorboard-profiler https://www.spheron.network/blog/gpu-cost-optimization-playbook/ https://massedcompute.com/faq-answers/?question=How%20to%20use%20NVIDIA%20tools%20to%20debug%20and%20profile%20GPU%20memory%20usage%20in%20cloud-based%20deep%20learning%20environments? https://blog.neevcloud.com/how-to-monitor-cloud-gpu-use-for-model-training-and-inference https://towardsdatascience.com/remote-development-and-debugging-on-the-cloud-aws-azure-gcp-for-deep-learning-computer-vision-5333fc698769/ https://docs.jax.dev/en/latest/profiling.html https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-profiling-report-walkthrough.html https://lobehub.com/skills/gpu-cli-gpu-gpu). We should know if the GPU use is optimal, time is not wasted on CPU, no bottlenecks exist between data and GPU (everything can be cached to RAM or not, is the hard drive fast enough), etc. And all these should be then logged to MLflow routinely so we can always go back if it seems that the training does not work well? Plan how to close this issue before actually starting any debug runs on Runpod via Skypilot with reviewer agents optimizing the plan and research to /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/profiler-benchmarking-plan.md. And obviously there should be an "enable" switch on the YAML config, but by default this profiling should run on every model that we implement, both locally and on cloud (As it is possible that we don't get the performance upgrade on cloud over local GPu even though in theory we thought that we would get it). We have to improve our "pytorch / monai" evaluation suite as well! See for example this Honey, I broke the PyTorch model >.< - Debugging custom PyTorch models in a structured manner : [full talk transcript included]. , and this plan should aim for closing of this issue then https://github.com/petteriTeikari/minivess-mlops/issues/564 . Optimize with reviewer agents until converging into a perfect plan, and start by saving my prompt verbatim. Remember academic format for the report and save all the links that I now gave to you to the reference list!

## Key Requirements Extracted

1. **PyTorch Profiler integration** — profile computation time per component
2. **Configurable via YAML** — `profiling.enabled: true`, `profiling.epochs: 5` (real) / `2` (debug)
3. **MLflow logging** — all profiling results logged routinely for retrospective analysis
4. **GPU utilization monitoring** — GPU vs CPU time, data loading bottlenecks, memory fragmentation
5. **Cloud vs local comparison** — verify cloud GPUs actually outperform local (not always true)
6. **Default ON** — profiling runs on every model, both local and cloud
7. **Enable switch** — can be disabled via YAML config
8. **Close issue #564** — dockerized GPU benchmark plan
9. **Pre-training checks** (from talk) — WeightWatcher, Cleanlab, synthetic data validation
10. **Post-training checks** — model quality metrics beyond loss curves

## Reference Links (User-Provided)

1. [PyTorch Profiler — Official Docs](https://docs.pytorch.org/docs/stable/profiler.html)
2. [PyTorch Profiler Tutorial — Beginner](https://docs.pytorch.org/tutorials/beginner/profiler.html)
3. [PyTorch Profiler Recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
4. [MONAI Profiling Utilities](https://github.com/Project-MONAI/MONAI/blob/dev/monai/utils/profiling.py)
5. [RunPod — Monitoring and Debugging AI Model Deployments](https://www.runpod.io/articles/guides/monitoring-and-debugging-ai-model-deployments)
6. [Atlantic.net — Profile and Debug GPU Performance on Ubuntu 24.04](https://www.atlantic.net/gpu-server-hosting/how-to-profile-and-debug-gpu-performance-for-machine-learning-models-on-ubuntu-24-04-gpu-server/)
7. [Debugging GPU Issues with AI](https://medium.com/@serverwalainfra/debugging-gpu-issues-with-ai-how-grok-simplifies-troubleshooting-0e7ccaadb109)
8. [Reddit — Cloud GPU Development Experience](https://www.reddit.com/r/CUDA/comments/180996l/what_is_your_experience_developing_on_a_cloud/)
9. [Eunomia — GPU Profiling Under the Hood: Survey of Modern Tracing Tools (2025)](https://eunomia.dev/blog/2025/04/21/gpu-profiling-under-the-hood-an-implementation-focused-survey-of-modern-accelerator-tracing-tools/)
10. [Google Cloud — TensorBoard Profiler](https://docs.cloud.google.com/vertex-ai/docs/training/tensorboard-profiler)
11. [Spheron — GPU Cost Optimization Playbook](https://www.spheron.network/blog/gpu-cost-optimization-playbook/)
12. [Massed Compute — NVIDIA Tools for GPU Memory Profiling](https://massedcompute.com/faq-answers/?question=How%20to%20use%20NVIDIA%20tools%20to%20debug%20and%20profile%20GPU%20memory%20usage%20in%20cloud-based%20deep%20learning%20environments?)
13. [NeevCloud — Monitor Cloud GPU Use for Training and Inference](https://blog.neevcloud.com/how-to-monitor-cloud-gpu-use-for-model-training-and-inference)
14. [Towards Data Science — Remote Development and Debugging on Cloud](https://towardsdatascience.com/remote-development-and-debugging-on-the-cloud-aws-azure-gcp-for-deep-learning-computer-vision-5333fc698769/)
15. [JAX Profiling Documentation](https://docs.jax.dev/en/latest/profiling.html)
16. [AWS SageMaker Debugger — Profiling Report](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-profiling-report-walkthrough.html)
17. [GPU CLI Tools](https://lobehub.com/skills/gpu-cli-gpu-gpu)

## Talk Reference

- **Talk**: "Honey, I broke the PyTorch model >.< - Debugging custom PyTorch models in a structured manner"
- **Speaker**: Clara Hoffman, PhD candidate at KleineaLab, Research Center for Trustworthy Data Science and Security, University Alliance Ruhr
- **Key packages mentioned**: torch-test, Weight Watcher, Cleanlab, PlaitPy, Zumo Lab ZPy
- **Key concept**: Project-transferable ML test suites (pre-train checks + post-train checks)
- **Repository**: slides available on speaker's GitHub
