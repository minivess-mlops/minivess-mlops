import os
import warnings
from copy import deepcopy
from datetime import datetime, timezone

from typing import Dict, Any
import hashlib
import json

import torch
import yaml
from loguru import logger
import collections.abc
import torch.distributed as dist

from omegaconf import OmegaConf

from src.log_ML.json_log import to_serializable
from src.log_ML.mlflow_log import init_mlflow_logging

CONFIG_DIR = os.path.join(os.getcwd(), 'configs')
if not os.path.exists(CONFIG_DIR):
    raise IOError('Cannot find the directory for (task) .yaml config files from "{}"'.format(CONFIG_DIR))
BASE_CONFIG_DIR = os.path.join(CONFIG_DIR, 'base')
if not os.path.exists(BASE_CONFIG_DIR):
    raise IOError('Cannot find the directory for (base) .yaml config files from "{}"'.format(BASE_CONFIG_DIR))


def import_config(args, task_config_file: str, base_config_file: str = 'base_config.yaml',
                  hyperparam_name: str = None, log_level: str = "INFO"):

    base_config = import_config_from_yaml(config_file = base_config_file,
                                          config_dir = BASE_CONFIG_DIR,
                                          config_type = 'base')

    config = update_base_with_task_config(task_config_file = task_config_file,
                                          config_dir = CONFIG_DIR,
                                          base_config = base_config)

    # Add the input arguments as an extra subdict to the config
    # config['config']['ARGS'] = args
    config['ARGS'] = args

    # Setup the computing environment
    config['config']['MACHINE'] = set_up_environment(machine_config=config['config']['MACHINE'],
                                                     local_rank=config['ARGS']['local_rank'])

    config_hash = dict_hash(dictionary=config['config'])
    start_time = get_datetime_string()
    output_experiments_base_dir = os.path.join(config['ARGS']['output_dir'], 'experiments')
    config['run'] = {
        'hyperparam_name': hyperparam_name,
        'hyperparam_base_name': hyperparam_name,
        'output_base_dir': config['ARGS']['output_dir'],
        'output_experiments_base_dir': output_experiments_base_dir,
        'output_experiment_dir': os.path.join(output_experiments_base_dir, hyperparam_name),
        # these sit outside the experiments and are not "hyperparameter run"-specific
        'output_wandb_dir': os.path.join(config['ARGS']['output_dir'], 'WANDB'),
        'output_mlflow_dir': os.path.join(config['ARGS']['output_dir'], 'MLflow'),
        'config_hash': config_hash,
        'start_time': start_time
    }

    # Init variables for 'run'
    config['run']['repeat_artifacts'] = {}
    config['run']['ensemble_artifacts'] = {}
    config['run']['fold_dir'] = {}

    # Get a predefined smaller subset to be logged as MLflow/WANDB columns/hyperparameters
    # to make the dashboards cleaner, or alternatively you can just dump the whole config['config']
    config['hyperparameters'] = define_hyperparam_run_params(config)
    config['hyperparameters_flat'] = flatten_nested_dictionary(dict_in=config['hyperparameters'])

    if config['config']['LOGGING']['unique_hyperparam_name_with_hash'] == 1:
        # i.e. whether you want a tiny change in dictionary content make this training to be grouped with
        # another existing (this is FALSE), or to be a new hyperparam name (TRUE) if you forgot for example to change
        # the hyperparam run name after some config changes. In some cases you would like to run the same experiment
        # over and over again and to be grouped under the same run if you want to know how reproducible your run is
        # and don't want to add for example date to the hyperparam name
        # e.g. from "'hyperparam_example_name'" ->
        #     "'hyperparam_example_name_9e0c146a68ec606442a6ec91265b11c3'"
        config['run']['hyperparam_name'] += '_{}'.format(config_hash)

    log_format = ("<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | "
                  "<yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>")
    try:
        logger.add(os.path.join(config['run']['output_experiment_dir'], 'log_{}.txt'.format(hyperparam_name)),
                   level=log_level, format=log_format, colorize=False, backtrace=True, diagnose=True)
    except Exception as e:
        raise IOError('Problem initializing the log file to the artifacts output, have you created one? '
                      'do you have the permissions correct? See README.md for the "minivess_mlops_artifacts" creation'
                      'with symlink to /mnt \n error msg = {}'.format(e))

    logger.info('Log will be saved to disk to "{}"'.format(config['run']['output_experiment_dir']))

    # Initialize ML logging (experiment tracking)
    config['run']['mlflow'] = init_mlflow_logging(config=config,
                                                  mlflow_config=config['config']['LOGGING']['MLFLOW'],
                                                  experiment_name = config['ARGS']['project_name'],
                                                  run_name = config['run']['hyperparam_name'])

    return config


def update_base_with_task_config(task_config_file: str, config_dir: str, base_config: dict):

    # https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    def update_config_dictionary(d, u):
        no_of_updates = 0
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k], _ = update_config_dictionary(d.get(k, {}), v)
                no_of_updates += 1
            else:
                d[k] = v
        return d, no_of_updates

    # Task config now contains only subset of keys (of the base config), i.e. the parameters
    # that you change (no need to redefine all possible parameters)
    task_config = import_config_from_yaml(config_file = task_config_file,
                                          config_dir = CONFIG_DIR,
                                          config_type = 'task')

    # update the base config now with the task config (i.e. the keys that have changed)
    config, no_of_updates = update_config_dictionary(d = base_config, u = task_config)
    logger.info('Updated the base config with a total of {} changed keys from the task config', no_of_updates)

    # TOADD: Hydra
    # https://www.sscardapane.it/tutorials/hydra-tutorial/#first-steps-manipulating-a-yaml-file
    # https://medium.com/pytorch/hydra-a-fresh-look-at-configuration-for-machine-learning-projects-50583186b710
    # https://github.com/khuyentran1401/Machine-learning-pipeline

    return config


def import_config_from_yaml(config_file: str = 'base_config.yaml',
                            config_dir: str = CONFIG_DIR,
                            config_type: str = 'base',
                            load_method: str = 'OmegaConf'):

    config_path = os.path.join(config_dir, config_file)
    if os.path.exists(config_path):
        logger.info('Import {} config from "{}", method = "{}"', config_type, config_path, load_method)
        if load_method == 'OmegaConf':
            # https://www.sscardapane.it/tutorials/hydra-tutorial/#first-steps-manipulating-a-yaml-file
            # https://omegaconf.readthedocs.io/en/2.3_branch/
            config = OmegaConf.load(config_path)
        else:
            raise IOError('Unknown method for handling configs? load_method = {}'.format(load_method))
    else:
        raise IOError('Cannot find {} config from = {}'.format(config_type, config_path))

    return config


def set_up_environment(machine_config: dict, local_rank: int = 0):

    if machine_config['DISTRIBUTED']:
        # initialize the distributed training process, every GPU runs in a process
        # see e.g.
        # https://github.com/Project-MONAI/tutorials/blob/main/acceleration/fast_model_training_guide.md
        # https://github.com/Project-MONAI/tutorials/blob/main/acceleration/distributed_training/brats_training_ddp.py
        dist.init_process_group(backend="nccl", init_method="env://")

    available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]

    if len(available_gpus) > 0:
        device = torch.device(f"cuda:{local_rank}")
        try:
            torch.cuda.set_device(device)
        except Exception as e:
            # e.g. "CUDA unknown error - this may be due to an incorrectly set up environment"
            raise EnvironmentError('Problem setting up the CUDA device'.format(e))
        torch.backends.cudnn.benchmark = True
    else:
        device = 'cpu'
        logger.warning('No Nvidia CUDA GPU found, training on CPU instead!')

    # see if this is actually the best way to do things, as "parsed" things start to be added to a static config dict
    machine_config['IN_USE'] = {'device': device,
                                'local_rank': local_rank}

    return machine_config


def hash_config_dictionary(dict_in: dict):
    """
    To check whether dictionary has changed
    https://www.doc.ic.ac.uk/~nuric/coding/how-to-hash-a-dictionary-in-python.html
    :param dict_in:
    :return:
    """



def dict_hash(dictionary: Dict[str, Any]) -> str:
    """
    MD5 hash of a dictionary.
    https://www.doc.ic.ac.uk/~nuric/coding/how-to-hash-a-dictionary-in-python.html
    """
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    # Fix this with the "to_serializable" TypeError: Object of type int64 is not JSON serializable
    try:
        encoded = json.dumps(dictionary, sort_keys=True, default=to_serializable).encode()
        dhash.update(encoded)
        hash_out = dhash.hexdigest()
    except Exception as e:
        logger.warning('Problem getting the hash of the config dictionary, error = {}'.format(e))
    return hash_out


def get_datetime_string():

    # Use GMT time if you have coworkers across the world running the jobs
    now = datetime.now(timezone.utc)
    date = now.strftime("%Y%m%d-%H%MGMT")

    return date


def define_hyperparam_run_params(config: dict) -> dict:
    """
    To be optimized later? You could read these from the config as well, which subdicts count as
    experiment hyperparameters. Not just dumping all possibly settings that maybe not have much impact
    on the "ML science" itself, e.g. number of workers for dataloaders or something
    :param config:
    :return:
    """

    hyperparams = {}
    cfg = config['config']

    logger.info('Hand-picking the keys/subdicts from "config" that are logged as hyperparameters for MLflow/WANDB')

    # What datasets were you used for training the model
    hyperparams['datasets'] = cfg['DATA']['DATA_SOURCE']['DATASET_NAME']

    # What model and architecture hyperparams you used
    hyperparams['model'] = {}
    # Well maybe you would not want all these to be logged either?
    model_name = cfg['MODEL']['MODEL_NAME']
    hyperparams['model'] = cfg['MODEL'][model_name]   # these are not all maybe wanted/needed
    hyperparams['model']['name'] = model_name

    # Training params
    training_tmp = deepcopy(cfg['TRAINING'])
    training_tmp.pop('METRICS', 'training_tmp')
    hyperparams['training'] = training_tmp

    setting_name = 'LOSS'
    hyperparams['training'][setting_name] = parse_settings_by_name(cfg=cfg, setting_name=setting_name,
                                                                   settings_key='LOSS_FUNCTIONS')
    setting_name = 'OPTIMIZER'
    hyperparams['training'][setting_name] = parse_settings_by_name(cfg=cfg, setting_name=setting_name,
                                                                   settings_key='OPTIMIZERS')
    setting_name = 'SCHEDULER'
    hyperparams['training'][setting_name] = parse_settings_by_name(cfg=cfg, setting_name=setting_name,
                                                                   settings_key='SCHEDULERS')

    return hyperparams


def parse_settings_by_name(cfg: dict, setting_name: str, settings_key: str) -> dict:
    settings_tmp = cfg[settings_key]
    # remove one nesting level
    name = list(settings_tmp.keys())[0]
    settings = settings_tmp[name]  # similarly here, you would like to have manual LUT for "main params"
    settings['name'] = name
    return settings


def flatten_nested_dictionary(dict_in: dict, delim: str = '__') -> dict:

    def parse_non_dict(var_in):
        # placeholder if you for example have lists that you would like to convert to strings?
        return var_in

    dict_out = {}
    for key1 in dict_in:
        subentry = dict_in[key1]
        if isinstance(subentry, dict):
            for key2 in subentry:
                subentry2 = subentry[key2]
                key_out2 = '{}{}{}'.format(key1, delim, key2)
                if isinstance(subentry2, dict):
                    for key3 in subentry2:
                        subentry3 = subentry2[key3]
                        key_out3 = '{}{}{}{}{}'.format(key1, delim, key2, delim, key3)
                        dict_out[key_out3] = parse_non_dict(subentry3)
                else:
                    dict_out[key_out2] = parse_non_dict(subentry2)
        else:
            dict_out[key1] = parse_non_dict(subentry)

    return dict_out