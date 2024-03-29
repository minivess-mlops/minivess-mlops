import time
from copy import deepcopy
from loguru import logger
import numpy as np
import torch
from monai.losses import DiceFocalLoss, DiceCELoss
from monai.optimizers import Novograd
from omegaconf import DictConfig

from src.utils.general_utils import check_if_key_in_dict


def check_for_params_dict(cfg_tmp, name):
    params = cfg_tmp.get(name)
    if params is None:
        return {}
    else:
        return params


def choose_loss_function(training_config: dict, loss_config: dict):
    # TODO! Make this automagic for all possible MONAI losses without the extra if-else
    try:
        loss_name = loss_config["NAME"]
        loss_params = check_for_params_dict(cfg_tmp=loss_config, name=loss_name)
        if loss_name == "DiceFocalLoss":
            loss_function = DiceFocalLoss(**loss_params)
        elif loss_name == "DiceCELoss":
            loss_function = DiceCELoss(**loss_params)
        else:
            raise NotImplementedError('Unsupported loss_name = "{}"'.format(loss_name))
    except Exception as e:
        logger.error("Problem getting loss function, error = {}".format(e))
        raise IOError("Problem getting loss function, error = {}".format(e))

    return loss_function


def choose_optimizer(model, training_config: dict, optimizer_config: dict):
    try:
        optimizer_name = optimizer_config["NAME"]
        optimizer_params = check_for_params_dict(
            cfg_tmp=optimizer_config, name=optimizer_name
        )
        if optimizer_name == "Novograd":
            optimizer = Novograd(model.parameters(), lr=training_config["LR"])
        else:
            raise NotImplementedError(
                'Unsupported optimizer_name = "{}"'.format(optimizer_name)
            )
    except Exception as e:
        logger.error("Problem getting optimizer, error = {}".format(e))
        raise IOError("Problem getting optimizer, error = {}".format(e))

    return optimizer


def choose_lr_scheduler(optimizer, training_config: dict, scheduler_config: dict):
    try:
        scheduler_name = scheduler_config["NAME"]
        scheduler_params = check_for_params_dict(
            cfg_tmp=scheduler_config, name=scheduler_name
        )
        if scheduler_name == "CosineAnnealingLR":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=training_config["NUM_EPOCHS"]
            )
        else:
            raise NotImplementedError(
                'Unsupported scheduler_name = "{}"'.format(scheduler_name)
            )
    except Exception as e:
        logger.error("Problem getting scheduler, error = {}".format(e))
        raise IOError("Problem getting scheduler, error = {}".format(e))

    return lr_scheduler


def set_model_training_params(model, device, scaler, training_config: dict, cfg: dict):
    loss_function = choose_loss_function(
        training_config=training_config, loss_config=training_config["LOSS"]
    )

    optimizer = choose_optimizer(
        model=model,
        training_config=training_config,
        optimizer_config=training_config["OPTIMIZER"],
    )

    lr_scheduler = choose_lr_scheduler(
        optimizer=optimizer,
        training_config=training_config,
        scheduler_config=training_config["SCHEDULER"],
    )

    return loss_function, optimizer, lr_scheduler


def init_epoch_dict(
    epoch: int, loaders, split_name: str, subsplit_name: str = "MINIVESS"
) -> dict:
    # TOADD You could make this a class actually?
    results_out = {}
    for loader_key in loaders:
        epoch_dict = {
            "scalars": {},  # e.g. Dice per epoch (this could have "metric_" prefix?)
            "arrays": {},  # e.g. batch-wise losses per epoch
            "metadata_scalars": {},  # e.g. learning rate, timing info, metadata for the training itself
            # 'figures': {},  # e.g. .png files on disk or figure handle
            # 'dataframes': {},  # e.g. Polars dataframes
        }
        results_out[loader_key] = epoch_dict

    return results_out


def collect_epoch_results(
    train_epoch_results: dict,
    eval_epoch_results: dict,
    train_results: dict,
    eval_results: dict,
    epoch: int,
):
    eval_results = combine_epoch_and_experiment_dicts(
        epoch_results=deepcopy(eval_epoch_results),
        experiment_results=deepcopy(eval_results),
        loop_type="eval",
        epoch=epoch,
    )

    train_results = combine_epoch_and_experiment_dicts(
        epoch_results=deepcopy({"TRAIN": train_epoch_results}),
        experiment_results=deepcopy(train_results),
        loop_type="train",
        epoch=epoch,
    )

    return train_results, eval_results


def combine_epoch_and_experiment_dicts(
    epoch_results: dict, experiment_results: dict, loop_type: str, epoch: int
) -> dict:
    if len(experiment_results) == 0:
        # i.e. the first epoch
        experiment_results = deepcopy(epoch_results)
        experiment_results = convert_scalars_to_arrays(experiment_results)
    else:
        for split_name in epoch_results.keys():
            for dataset_name in epoch_results[split_name].keys():
                epoch_res = epoch_results[split_name][dataset_name]
                experim_res = experiment_results[split_name][dataset_name]
                experiment_results[split_name][
                    dataset_name
                ] = add_epoch_results_to_experiment_results(
                    experim_res, epoch_res, split_name, dataset_name, epoch
                )

    return experiment_results


def convert_scalars_to_arrays(experiment_results: dict):
    logger.debug("On first epoch, convert the scalars to arrays")
    for split in experiment_results:
        for dataset in experiment_results[split]:
            for var_type in experiment_results[split][dataset]:
                for scalar_key in experiment_results[split][dataset][var_type]:
                    # print(split, dataset, var_type, scalar_key) # e.g. TRAIN MINIVESS scalars loss

                    if isinstance(
                        experiment_results[split][dataset][var_type][scalar_key],
                        (float, int),
                    ):
                        # on the 2nd epoch, we convert the float into a numpy array
                        # so "scalars" will become an array if this is confusing, but "arrays" would contain variables
                        # that would have multiple values at epoch level, e.g. like a 1D histogram you could collect to
                        # experiment-level 2D array with 1D histogram per each epoch
                        experiment_results[split][dataset][var_type][
                            scalar_key
                        ] = np.array(
                            experiment_results[split][dataset][var_type][scalar_key]
                        )[
                            np.newaxis
                        ]

    return experiment_results


def add_epoch_results_to_experiment_results(
    experim_res: dict, epoch_res: dict, split_name: str, dataset_name: str, epoch: int
):
    for results_type in epoch_res.keys():
        # print(results_type)
        if results_type == "scalars" or "metadata_scalars":
            experim_res[results_type] = combine_epoch_and_experiment_scalars(
                epoch_scalars=deepcopy(epoch_res[results_type]),
                experim_scalars=deepcopy(experim_res[results_type]),
            )
        elif results_type == "arrays":
            experim_res[results_type] = combine_epoch_and_experiment_arrays(
                epoch_arrays=deepcopy(epoch_res[results_type]),
                experim_arrays=deepcopy(experim_res[results_type]),
            )
        else:
            if epoch == 1:
                logger.warning(
                    'Note if you have anything stored at results_type = "{}", '
                    "they do not get autocollected over epochs at the moment,\n"
                    'only the keys in "arrays" and "scalars" subdictionaries are correctly implemented'.format(
                        results_type
                    )
                )
            to_be = "implemented the other types if you get them"

    return experim_res


def combine_epoch_and_experiment_scalars(epoch_scalars, experim_scalars):
    for scalar_key in epoch_scalars.keys():
        epoch_scalar_per_key = np.array(epoch_scalars[scalar_key]).copy()
        experim_scalars[scalar_key] = np.hstack(
            [experim_scalars[scalar_key], epoch_scalar_per_key]
        )

    return experim_scalars


def combine_epoch_and_experiment_arrays(epoch_arrays, experim_arrays):
    for scalar_key in epoch_arrays.keys():
        epoch_array_per_key = np.array(epoch_arrays[scalar_key]).copy()
        if len(epoch_array_per_key.shape) == 1:
            experim_arrays[scalar_key] = np.concatenate(
                (experim_arrays[scalar_key], epoch_array_per_key)
            )

    return experim_arrays


def get_timings_per_epoch(
    metadata_dict: dict, epoch_start: float, no_batches: int, mean_batch_sz: float
) -> dict:
    metadata_dict["metadata_scalars"]["time_epoch"] = time.time() - epoch_start
    metadata_dict["metadata_scalars"]["time_batch"] = (
        metadata_dict["metadata_scalars"]["time_epoch"] / no_batches
    )
    # i.e you have 6 batches with 8, 8, 8, 8, 8 and 4 samples per batch (with the last one being smaller)
    metadata_dict["metadata_scalars"]["time_sample_ms"] = (
        metadata_dict["metadata_scalars"]["time_batch"] / mean_batch_sz
    ) * 1000

    return metadata_dict


def get_first_batch_from_dataloaders_dict(
    experim_dataloaders: dict, split: str = "TRAIN", return_batch_dict: bool = True
):
    first_fold = list(experim_dataloaders.keys())[0]
    dataloader = experim_dataloaders[first_fold][split]
    batch_dict = next(iter(dataloader))
    # images, labels = batch_dict['image'], batch_dict['label']
    # logger.info('MLflow | Get first batch of "{}" dataloader, batch sz = {}'.format(split, images.shape))
    return batch_dict
