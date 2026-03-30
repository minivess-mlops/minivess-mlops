# Everything Dashboard

**Generated:** 2026-03-02T03:24:33.055608+00:00

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

- **Trigger source:** scripts/run_dashboard_real.py
- **Data version:** v1.0
- **Last training run:** 01d904c61b1043a6b4d4630ec1506992

### Flow Results

| Flow | Status |
|------|--------|
| data_flow | PASSED |
| analysis_flow | PASSED |
| deploy_flow | PASSED |
| dashboard_flow | RUNNING |
