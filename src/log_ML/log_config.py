import os
import shutil

import wandb
import mlflow
from loguru import logger
from omegaconf import DictConfig

from src.inference.ensemble_utils import get_ensemble_name, get_submodel_name
from src.log_ML.log_model_registry import (
    log_ensembles_to_mlflow,
    log_ensembles_to_wandb,
)
from src.log_ML.log_utils import write_config_as_yaml
from src.utils.dict_utils import cfg_key


def log_config_artifacts(
    log_name: str,
    cv_dir_out: str,
    output_dir: str,
    cfg: DictConfig,
    fold_results: dict,
    ensembled_results: dict,
    cv_ensemble_results: dict,
    experim_dataloaders: dict,
    logging_services: list,
    wandb_run: wandb.sdk.wandb_run.Run,
):
    """
    After Cross-validation of the all the folds, you are done with the config, and you can
    write the "final artifacts" then disk.
    """

    # Log the artifact dir
    logger.info(
        "{} | ENSEMBLED Cross-Validation results | Artifacts directory".format(
            logging_services
        )
    )
    if "WANDB" in logging_services:
        artifact_dir = wandb.Artifact(name=log_name, type="artifacts")
        artifact_dir.add_dir(local_path=cv_dir_out, name="CV-ENSEMBLE_artifacts")
        wandb_run.log_artifact(artifact_dir)
    if "MLflow" in logging_services:
        mlflow.log_artifact(cv_dir_out)

    # Log all the models from all the folds and all the repeats to the Artifact Store
    # and these are accessible to Model Registry as well
    logger.info(
        "{} | ENSEMBLED Cross-Validation results | Models to Model Registry".format(
            logging_services
        )
    )
    model_paths = log_model_ensemble_to_model_registry(
        fold_results=fold_results,
        ensembled_results=ensembled_results,
        cv_ensemble_results=cv_ensemble_results,
        experim_dataloaders=experim_dataloaders,
        log_name=log_name,
        wandb_run=wandb_run,
        logging_services=logging_services,
        cfg=cfg,
        test_loading=True,
    )  # GET FROM CONFIG

    # The log_model produced the "MLflow model", and as a backup, we could save all the .pth models
    # to a local folder, and then log that folder as an artifact to MLflow
    log_models_of_ensemble_to_folder(model_paths=model_paths, output_dir=output_dir)

    # HERE, log the config as .yaml file back to disk
    logger.info(
        "{} | ENSEMBLED Cross-Validation results | Config as YAML".format(
            logging_services
        )
    )
    path_out_cfg, loaded_cfg = write_config_as_yaml(
        config=cfg["hydra_cfg"], dir_out=output_dir, fname_out=get_cfg_yaml_fname()
    )
    path_out_run, loaded_run = write_config_as_yaml(
        config=cfg["run"], dir_out=output_dir, fname_out=get_run_params_yaml_fname()
    )

    # Log requirements?
    # cfg['run']['PARAMS']['requirements-txt_path']

    if "WANDB" in logging_services:
        artifact_cfg = wandb.Artifact(name="config", type="config")
        artifact_cfg.add_file(path_out_cfg)
        wandb_run.log_artifact(artifact_cfg)
        artifact_run = wandb.Artifact(name="run_params", type="config")
        artifact_run.add_file(path_out_run)
        wandb_run.log_artifact(artifact_run)

    if "MLflow" in logging_services:
        mlflow.log_dict(loaded_cfg, artifact_file=os.path.split(path_out_cfg)[1])
        mlflow.log_dict(loaded_run, artifact_file=os.path.split(path_out_run)[1])

    logger.info(
        "{} | ENSEMBLED Cross-Validation results | Loguru log saved as .txt".format(
            logging_services
        )
    )
    log_path_out = cfg_key(cfg, "run", "PARAMS", "output_log_path")
    std_path_out = cfg_key(cfg, "run", "PARAMS", "stdout_log_path")

    if "WANDB" in logging_services:
        artifact_log = wandb.Artifact(name="log", type="log")
        artifact_log.add_file(log_path_out)
        wandb_run.log_artifact(artifact_log)

        artifact_stdout = wandb.Artifact(name="stdout", type="log")
        artifact_stdout.add_file(std_path_out)
        wandb_run.log_artifact(artifact_stdout)

    if "MLflow" in logging_services:
        mlflow.log_artifact(log_path_out)
        mlflow.log_artifact(std_path_out)

    return model_paths


def define_ensemble_submodels_dir_name():
    return "ensemble_submodels"


def log_models_of_ensemble_to_folder(
    model_paths: dict, output_dir: str, move_files: bool = True
):
    logger.info("Log submodels of the ensemble(s) in a folder to MLflow Artifact Store")
    submodels_dir = os.path.join(output_dir, define_ensemble_submodels_dir_name())
    os.makedirs(submodels_dir, exist_ok=True)
    for ensemble_name in model_paths["ensemble_models_flat"]:
        model_paths_ensemble = model_paths["ensemble_models_flat"][ensemble_name]
        ensemble_dir = os.path.join(submodels_dir, ensemble_name)
        os.makedirs(ensemble_dir, exist_ok=True)
        for submodel_name in model_paths_ensemble:
            model_path = model_paths_ensemble[submodel_name]
            fname_in = os.path.split(model_path)[1]
            _, ext = os.path.splitext(fname_in)
            fname_out = f"{submodel_name}.{ext}"
            path_out = os.path.join(ensemble_dir, fname_out)
            if move_files:
                logger.debug('Move file "{}" to "{}"'.format(fname_in, path_out))
                shutil.move(model_path, os.path.join(ensemble_dir, fname_out))
            else:
                logger.debug('Copy file "{}" to "{}"'.format(fname_in, path_out))
                shutil.copy(model_path, os.path.join(ensemble_dir, fname_out))

    mlflow.log_artifact(submodels_dir)


def log_model_ensemble_to_model_registry(
    fold_results: dict,
    ensembled_results: dict,
    cv_ensemble_results: dict,
    experim_dataloaders: dict,
    log_name: str,
    wandb_run: wandb.sdk.wandb_run.Run,
    logging_services: list,
    cfg: DictConfig,
    test_loading: bool = False,
):
    # Collect and simplify the submodel structure of the ensemble(s)
    model_paths, ensemble_models_flat = collect_submodels_of_the_ensemble(fold_results)

    log_outputs = {}
    if "WANDB" in logging_services:
        log_ensembles_to_wandb(
            ensemble_models_flat=ensemble_models_flat,
            cfg=cfg,
            wandb_run=wandb_run,
            test_loading=test_loading,
        )

    if "MLflow" in logging_services:
        log_outputs["MLflow"] = log_ensembles_to_mlflow(
            ensemble_models_flat=ensemble_models_flat,
            experim_dataloaders=experim_dataloaders,
            ensembled_results=ensembled_results,
            cv_ensemble_results=cv_ensemble_results,
            cfg=cfg,
            test_loading=test_loading,
        )

    return {
        "model_paths": model_paths,
        "ensemble_models_flat": ensemble_models_flat,
        "log_outputs": log_outputs,
    }


def collect_submodels_of_the_ensemble(fold_results: dict):
    # Two ways of organizing the same saved models, deprecate the other later
    model_paths = {}  # original nesting notation
    ensemble_models_flat = (
        {}
    )  # more intuitive maybe, grouping models under the same ensemble name
    n_submodels = 0
    n_ensembles = 0

    for fold_name in fold_results:
        model_paths[fold_name] = {}
        for archi_name in fold_results[fold_name]:
            for repeat_name in fold_results[fold_name][archi_name]:
                model_paths[fold_name][repeat_name] = {}
                best_dict = fold_results[fold_name][archi_name][repeat_name][
                    "best_dict"
                ]
                for ds in best_dict:
                    model_paths[fold_name][repeat_name][ds] = {}
                    for tracked_metric in best_dict[ds]:
                        model_path = best_dict[ds][tracked_metric]["model"][
                            "model_path"
                        ]
                        model_paths[fold_name][repeat_name][ds][
                            tracked_metric
                        ] = model_path

                        ensemble_name = get_ensemble_name(
                            dataset_validated=ds, metric_to_track=tracked_metric
                        )

                        if ensemble_name not in ensemble_models_flat:
                            ensemble_models_flat[ensemble_name] = {}
                            n_ensembles += 1

                        submodel_name = get_submodel_name(
                            archi_name=archi_name,
                            fold_name=fold_name,
                            repeat_name=repeat_name,
                        )

                        if submodel_name not in ensemble_models_flat[ensemble_name]:
                            ensemble_models_flat[ensemble_name][
                                submodel_name
                            ] = model_path
                            n_submodels += 1

    # Remember that you get more distinct ensembles if use more tracking metrics (e.g. track for best loss and for
    # best Hausdorff distance), and if you validate for more subsets of the data instead of just having one
    # "vanilla" validation splöit
    logger.info(
        "Collected a total of {} models, and {} distinct ensembles to be logged to Model Registry".format(
            n_submodels, n_ensembles
        )
    )

    return model_paths, ensemble_models_flat


def get_cfg_yaml_fname():
    return "config.yaml"


def get_run_params_yaml_fname():
    return "run_params.yaml"
