import time
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger

from src.eval import evaluate_datasets_per_epoch
from src.log_ML.logging_main import log_epoch_results, save_models_if_improved, init_best_dict_with_criteria, \
    log_n_epochs_results
from src.utils.model_utils import import_segmentation_model
from src.utils.train_utils import init_epoch_dict, collect_epoch_results, init_training, set_model_training_params


def training_script(experim_dataloaders: dict,
                    config: dict,
                    training_config: dict,
                    model_config: dict,
                    machine_config: dict):

    # Cross-validation loop (if used), i.e. when train/val splits change for each execution
    fold_results = {}
    for f, fold_name in enumerate(list(experim_dataloaders.keys())):
        logger.info('Training fold #{}/{}: {}'.format(f+1, len(experim_dataloaders.keys()), fold_name))
        fold_results[fold_name] = \
            train_model_for_single_fold(fold_dataloaders=experim_dataloaders[fold_name],
                                        config=config,
                                        training_config=training_config,
                                        model_config=model_config,
                                        machine_config=machine_config)

    return fold_results


def train_model_for_single_fold(fold_dataloaders: dict,
                                config: dict,
                                training_config: dict,
                                model_config: dict,
                                machine_config: dict):

    # Repeat n times the same data fold (i.e. you get n submodels for an ensemble)
    no_repeats = training_config['NO_REPEATS']
    repeat_results = {}
    for repeat_idx in range(no_repeats):
        logger.info('Training repeat #{}/{}'.format(repeat_idx + 1, no_repeats))
        repeat_results['repeat_{}'.format(str(repeat_idx+1).zfill(2))] = \
            train_single_model(dataloaders=fold_dataloaders,
                               config=config,
                               training_config=training_config,
                               model_config=model_config,
                               machine_config=machine_config,
                               repeat_idx=repeat_idx,
                               device=machine_config['IN_USE']['device'])

    return repeat_results


def train_single_model(dataloaders: dict,
                       config: dict,
                       training_config: dict,
                       model_config: dict,
                       machine_config: dict,
                       repeat_idx: int,
                       device):


    if training_config['AMP']:
        scaler = torch.cuda.amp.GradScaler()
    else:
        raise NotImplementedError('Check the train loop also for non-AMP operation')

    # Define the model to be used
    model = import_segmentation_model(model_config, device)

    # Model training params
    loss_function, optimizer, lr_scheduler = \
        set_model_training_params(model, device, scaler, training_config, config)

    # Train script
    train_results, eval_results, best_dict = \
        train_n_epochs_script(model, dataloaders,
                              device, scaler,
                              loss_function, optimizer, lr_scheduler,
                              training_config, config)

    # When training is done, you con for example log the repeat/experiment/n_epochs level results
    log_n_epochs_results(train_results, eval_results, best_dict, config)

    results_out = {
                   'train_results': train_results,
                   'eval_results': eval_results,
                   'best_dict': best_dict,
                   }

    return results_out


def train_n_epochs_script(model, dataloaders,
                          device, scaler,
                          loss_function, optimizer, lr_scheduler,
                          training_config: dict, config: dict,
                          start_epoch: int = 0):

    # FIXME: get this from config
    metric_dict = {'roi_size': (64, 64, 8), 'sw_batch_size': 4, 'predictor': model, 'overlap': 0.6}

    train_results = {}
    eval_results = {}
    best_dict = init_best_dict_with_criteria()

    # https://github.com/Project-MONAI/tutorials/blob/2183d45f48c53924b291a16d72f8f0e0b29179f2/acceleration/distributed_training/brats_training_ddp.py#L285
    print(' ')
    for epoch in tqdm(range(start_epoch, training_config['NUM_EPOCHS']), desc = 'Training the network'):

        eval_epoch_results = {}

        # Train
        train_epoch_results, eval_epoch_results['TRAIN'] = \
            train_1_epoch(model, device, epoch, loss_function, optimizer, lr_scheduler, scaler, training_config,
                          train_loader = dataloaders['TRAIN'], metric_dict = metric_dict,
                          dataset_dummy_key = 'MINIVESS')

        # Validate (as in decide whether the model improved or not)
        split_name = 'VAL'
        eval_epoch_results[split_name] = evaluate_datasets_per_epoch(model, device, epoch, dataloaders,
                                                                     training_config, metric_dict, split_name)

        # Evaluate (check generalization for held-out datasets)
        split_name = 'TEST'
        eval_epoch_results[split_name] = evaluate_datasets_per_epoch(model, device, epoch, dataloaders,
                                                                     training_config, metric_dict, split_name)

        # Collect results to a dictionary and avoid having multiple lists for each metric
        train_results, eval_results = collect_epoch_results(train_epoch_results, eval_epoch_results,
                                                            train_results, eval_results, epoch)

        # Log epoch-level result
        log_epoch_results(train_epoch_results, eval_epoch_results, epoch, config)

        # Save model(s) if model has improved
        save_models_if_improved(best_dict, train_epoch_results, eval_epoch_results, config)


    return train_results, eval_results, best_dict


def train_1_epoch(model, device, epoch, loss_function, optimizer, lr_scheduler, scaler, training_config,
                 train_loader, metric_dict, dataset_dummy_key: str = 'MINIVESS'):

    # https://github.com/Project-MONAI/tutorials/blob/2183d45f48c53924b291a16d72f8f0e0b29179f2/acceleration/distributed_training/brats_training_ddp.py#L317
    model.train()
    epoch_eval_res = init_epoch_dict(epoch, loaders = {dataset_dummy_key: train_loader}, split_name='TRAIN')
    epoch_trn_res = init_epoch_dict(epoch, loaders = {dataset_dummy_key: train_loader}, split_name='TRAIN')
    batch_losses = []
    batch_szs = []

    step_start = time.time()
    for batch_idx, batch_data in enumerate(train_loader):
        optimizer.zero_grad()
        loss = train_1_batch(model, device, batch_data, loss_function,
                             amp_on= training_config['AMP'])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_losses.append(loss.item())
        batch_szs.append(batch_data['image'].shape[0])

    lr_scheduler.step()

    # Average loss per epoch (no_of_steps = batch_idx+1)
    epoch_eval_res[dataset_dummy_key]['scalars']['loss'] = np.mean(batch_losses)

    # Batch-level losses (depending on what you want to see, maybe both on MLflow/Wandb/etc.)
    epoch_trn_res[dataset_dummy_key]['arrays']['batch_loss'] = np.array(batch_losses)

    return epoch_trn_res, epoch_eval_res


def train_1_batch(model, device, batch_data, loss_function, amp_on: bool = True):

    if amp_on:
        with torch.cuda.amp.autocast():
            outputs = model(batch_data["image"].to(device)) # .to(device) here instead of "ToDeviced"
            # "ToDeviced" lead to GPU memory glitches, inspect this later?
            loss = loss_function(outputs, batch_data["label"].to(device)) # .to(device) here instead of "ToDeviced"
    else:
        outputs = model(batch_data["image"].to(device))  # .to(device) here instead of "ToDeviced"
        # "ToDeviced" lead to GPU memory glitches, inspect this later?
        loss = loss_function(outputs, batch_data["label"].to(device))  # .to(device) here instead of "ToDeviced"

    return loss


