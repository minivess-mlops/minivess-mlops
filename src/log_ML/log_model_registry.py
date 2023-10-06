import mlflow
import torch
import wandb
from loguru import logger


def log_ensembles_to_MLflow(ensemble_models_flat: dict, config: dict):
    """
    See these for example for logging ensemble of models to Model Registry in MLflow:
        https://www.databricks.com/blog/2021/09/21/managing-model-ensembles-with-mlflow.html
        https://medium.com/@pennyqxr/how-to-train-and-track-ensemble-models-with-mlflow-a1d2695e784b
        https://python.plainenglish.io/how-to-create-meta-model-using-mlflow-166aeb8666a8
    """
    no_ensembles = len(ensemble_models_flat)
    for i, ensemble_name in enumerate(ensemble_models_flat):
        no_submodels_per_ensemble = len(ensemble_models_flat[ensemble_name])

        for j, submodel_name in enumerate(ensemble_models_flat[ensemble_name]):

            artifact_name = '{}__{}'.format(ensemble_name, submodel_name)
            model_path = ensemble_models_flat[ensemble_name][submodel_name]
            logger.info('MLflow | Model file logged to registry: {}'.format(artifact_name))

            # Load the model
            model_dict = torch.load(model_path)
            model = model_dict['model']

            # Log model
            mlflow.pytorch.log_model(model, "model")

            # Register model
            mlflow.register_model(model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
                                  name = artifact_name,
                                  tags={'ensemble_name': ensemble_name, 'submodel_name': submodel_name})



def log_ensembles_to_WANDB(ensemble_models_flat: dict, config: dict,
                            wandb_run: wandb.sdk.wandb_run.Run):

    # TO-FIGURE-OUT: HOW TO DO ENSEMBLES THE MOST EFFICIENTLY FOR WANDB
    # This is "reserved name" in WANDB, and if you want to use Model Registry, keep this naming
    artifact_type = 'model'

    for i, ensemble_name in enumerate(ensemble_models_flat):
        for j, submodel_name in enumerate(ensemble_models_flat[ensemble_name]):
            artifact_name = '{}__{}'.format(ensemble_name, submodel_name)
            artifact_model = wandb.Artifact(name=artifact_name, type=artifact_type)
            model_path = ensemble_models_flat[ensemble_name][submodel_name]
            artifact_model.add_file(model_path)
            logger.info('WANDB | Model file logged to registry: {}'.format(artifact_name))
            wandb_run.log_artifact(artifact_model)


class ModelEnsemble(mlflow.pyfunc.PythonModel):
  def __init__(self, ensemble_model_paths: dict):
      self.ensemble_model_paths = ensemble_model_paths


# ensemble = ModelEnsemble(ensemble_model_paths=ensemble_models_flat[ensemble_name])

