# Everything Dashboard

**Generated:** 2026-03-02T03:32:10.212589+00:00

## Data

- **Volumes:** 70
- **External datasets:** 0
- **Quality gate:** PASSED

## Configuration

- **Environment:** local
- **Experiment config:** dynunet_loss_variation_v2
- **Model profile:** dynunet

### Dynaconf Parameters

- debug: True
- project_name: minivess-mlops-v2

## Model

- **Architecture:** MONAI DynUNet (128 filters)
- **Parameters:** 5,604,866
- **Loss:** cbdice_cldice
- **ONNX exported:** True
- **Champion category:** balanced

## Pipeline

- **Trigger source:** test/quasi-e2e-debugging
- **Data version:** v1.0
- **Last training run:**

### Flow Results

| Flow | Status |
|------|--------|
| data_flow | PASSED |
| analysis_flow | PASSED |
| deploy_flow | PASSED |
| dashboard_flow | RUNNING |
