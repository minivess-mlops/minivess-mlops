---
title: "User Prompt: Advanced Segmentation Double-Check"
status: reference
created: "2026-03-04"
---

# User Prompt — Advanced Segmentation Double-Check (2026-03-04)

> this got merged and deleted, so fetch the latest remote and create branch feat/advanced-segmentation-double-check and let's first comment away the unit tests on Github Actions as we are now running these locally to save time, and bring the unit tests back to CI when we have implemented the prod and staging split in practice. This branch should be about double-checking that we have all the relevant state-of-art (SOTA) methods implemented for our MLOps repo which does not have the focus on the model SOTA, but building an architecture / platform helping researchers to come up with new SOTA a lot faster! So read these /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/albarqouni-2025-prosona-personalization-segmentation.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/borges-2024-selective-prediction-segmentation.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/borges-2024-soft-dice-confidence.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/conti-2025-large-multimodal.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/dionelis-2024-improving-foundation.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/gao-2023-mixture-stochastic-uncertainty.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/gao-2024-robust-segmentation-distribution-shifts.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/giron-2026-prefect-workflows-slurm.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/guimard-2025-classifier-bias-detection.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/hu-2025-weakly-supervised-icl-segmentation.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/huang-2024-p2sam-probabilistic-segmentation.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/huynh-2025-diffusion-model.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/hwang-2025-open-vocabulary-continual-learning.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/kassapis-2024-calibrated-adversarial-segmentation.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/kervadec-2023-dice-loss-gradient.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/keser-2025-benchmarking-vision.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/koleilat-2026-medclipseg-vision-language.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/kumari-2025-annotation-ambiguity-segmentation.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/li-2024-gmm-stochastic-segmentation.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/li-2024-sam-ambiguous-segmentation.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/li-2026-dacal-cross-domain-calibration.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/liu-2025-adaptive-conformal.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/lu-2025-steerable-pyramid.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/mossina-2024-conformal-semantic-segmentation.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/ngoc-2025-latentfm-latent.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/ramos-2025-clip-processing-acquisition-traces.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/ren-2026-reliable-medical-llm-confidence.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/ribeiro-2025-flow-stochastic-networks.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/specktor-fadida-2025-segqc-quality-control.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/su-2025-router-uncertainty.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/vakalopoulou-2026-class-adaptive.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/volpi-2023-reliability-semantic-segmentation.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/wakasugi-2021-diffusion-discrete-state.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/xu-2020-domain-division-generalization.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/yang-2025-hsrdiff-stochastic-segmentation.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/yang-2025-medsamix-training.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/yu-2025-crisp-sam2-multi-organ.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/zbinden-2023-conditional-categorical-diffusion.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/zhang-2022-pixelseg-stochastic-segmentation.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/zhao-2025-confagents-multi-agent-diagnosis.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/zhou-2024-image-segmentation-foundation-survey-github.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/zuluaga-2026-trustworthy-ai-medical-imaging.md and let's especially focus on the uncerainty methods for foundation models (SAMv3, and vesselfm), how conformal prediction is connected to distribution shifts and our drift detection (e.g. Evidently, and synthetic data generation to simulate the data drift, e.g. /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/gao-2024-robust-segmentation-distribution-shifts.md Badkul, Amitesh, and Lei Xie. 2025. "Adaptive Individual Uncertainty under Out-Of-Distribution Shift with Expert-Routed Conformal Prediction." arXiv:2510.15233. Preprint, arXiv, October 17. https://doi.org/10.48550/arXiv.2510.15233.
> Lin, Zhexiao, Yuanyuan Li, Neeraj Sarna, Yuanyuan Gao, and Michael von Gablenz. 2025. "Domain-Shift-Aware Conformal Prediction for Large Language Models." arXiv:2510.05566. Preprint, arXiv, October 7. https://doi.org/10.48550/arXiv.2510.05566.
> Ryan J. Tibshirani, Rina Foygel Barber, Emmanuel J. Candes, and Aaditya Ramdas. 2019. "Conformal Prediction Under Covariate Shift." 1904.06019. Preprint, arXiv. https://arxiv.org/abs/1904.06019.
> Shaer, Shalev, Yarin Bar, Drew Prinster, and Yaniv Romano. 2026. "Testing For Distribution Shifts with Conditional Conformal Test Martingales." arXiv:2602.13848. Preprint, arXiv, February 14. https://doi.org/10.48550/arXiv.2602.13848.
> Xu, Rui, Yue Sun, Chao Chen, Parv Venkitasubramaniam, and Sihong Xie. 2024. "Robust Conformal Prediction under Distribution Shift via Physics-Informed Structural Causal Model." arXiv:2403.15025. Preprint, arXiv, March 22. https://doi.org/10.48550/arXiv.2403.15025.
> ), implement some model merging method to our architecture so this could be done (/home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/yang-2025-medsamix-training.md /home/petteri/Dropbox/KnowledgeBase/LLM/LLM - Ensembles and Merging.md), should we implement some 3D Mamba model with uncertainty (e.g. /home/petteri/Dropbox/KnowledgeBase/LLM/LLM - Intro - State Space Models and Long-Context.md http://arxiv.org/abs/2502.02024 https://github.com/uakhan17/Mamba3D-MedSeg https://github.com/Ruixxxx/Awesome-Vision-Mamba-Models /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/wang-2025-mamba-3d-medical-segmentation.md), do we need some other alternative for synthetic data generation (e.g. https://jefferyzhifeng.github.io/projects/VasTSD/ /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/wang-2025-vastsd-vascular-tree-diffusion.md), how to improve data quality and annotator pipeline (/home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/specktor-fadida-2025-segqc-quality-control.md), can we implement some latent diffusion model for both segmentation and synthetic data generation (as in is there some code available for 3D medical segmentation) as that would present a novel category of model that we are not capturing yet (/home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/ngoc-2025-latentfm-latent.md
> /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/ribeiro-2025-flow-stochastic-networks.md)? ANd how about using VLMs or MLLMs for 3D medical segmentation? Is there some that we could use in zero-shot case for reference? something that has been used for 3D medical segmentation, e.g. /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/koleilat-2026-medclipseg-vision-language.md . Again quite comprehensive task so let's optimize the academic literature research report with reviewer agents for factual accuracy and breadth to /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/advanced-segmentation-double-check-plan.md. After this background literature report, let's update our probabilistic PRD and then create the Issue that an executable xml plan will close then with our self-learning TDD skill. Does it seem clear? Save my prompt verbatim first and let's start this!
