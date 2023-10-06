from copy import deepcopy

import monai.data
import numpy as np
import torch
from monai.data import MetaTensor
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete
from tqdm import tqdm

from src.inference.ensemble_utils import merge_nested_dicts, get_metadata_for_sample_metrics, \
    add_sample_metrics_to_split_results, compute_split_metric_stats
from src.inference.metrics import get_sample_metrics_from_np_masks, get_sample_uq_metrics_from_ensemble_stats
from src.log_ML.model_saving import import_model_from_path


def inference_sample(batch_data,
                     model,
                     metric_dict: dict,
                     device: str,
                     precision: str = 'AMP'):
    """
    See e.g. https://docs.monai.io/en/stable/inferers.html
             https://github.com/davidiommi/Ensemble-Segmentation/blob/main/predict_single_image.py
             https://github.com/Project-MONAI/tutorials/blob/main/modules/cross_validation_models_ensemble.ipynb
    :param model:
    :param dataloader:
    :param config:
    :param device:
    :param auto_mixedprec:
    :return:
    """
    if precision == 'AMP':  ## AMP
        with torch.cuda.amp.autocast():
            output = sliding_window_inference(inputs=batch_data["image"].to(device), **metric_dict)
    else:
        output = sliding_window_inference(inputs=batch_data["image"].to(device), **metric_dict)

    # logits -> probabilities
    activ = Activations(sigmoid=True)
    probability_mask = activ(output)

    # probabilities -> binary mask
    discret = AsDiscrete(threshold=0.5)  ## FIXME: get this threshold from config so you can set it more dynamically
    binary_mask = discret(probability_mask)

    inference_output = {'scalars':
                            {
                                'dummy_scalar': np.array([0.5])
                            },
                        'arrays':
                            {
                                'logits': conv_metatensor_to_numpy(output),
                                'probs': conv_metatensor_to_numpy(probability_mask),
                                'binary_mask': conv_metatensor_to_numpy(binary_mask)
                            },
                        }

    return inference_output


def conv_metatensor_to_numpy(metatensor_in: MetaTensor) -> np.ndarray:
    # https://github.com/Project-MONAI/MONAI/issues/4649#issuecomment-1177319399
    try:
        numpy_array = metatensor_in.cpu().detach().numpy()
    except Exception as e:
        # CHECK LATER IF NEEDED when you don't have a NVIDIA GPU for training
        numpy_array = metatensor_in.detach().numpy()
    assert len(numpy_array.shape) == 5, ('Code tested only for 5D tensors at the moment, '
                                         'now your metatensor is {}D').format(len(numpy_array.shape))
    return numpy_array[0,0,:,:,:]  # quick and dirty for single-channel and batch_sz = 1 tensors




def inference_best_repeat(dataloader: monai.data.dataloader.DataLoader,
                          split: str,
                          best_repeat_dicts: dict,
                          config: dict,
                          device):

    def get_model_path_from_repeat_best_dict(metric_dict_in: dict,
                                             dataset: str,
                                             tracked_metric: str):
        return metric_dict_in['repeat_best_dict'][dataset][tracked_metric]['model']['model_path']


    no_samples = len(dataloader.sampler)
    split_metrics, split_metrics_stat = {}, {}

    for dataset in best_repeat_dicts:
        split_metrics[dataset] = {}
        split_metrics_stat[dataset] = {}
        for tracked_metric in best_repeat_dicts[dataset]:  # what metric is used to save model
            split_metrics[dataset][tracked_metric] = {}
            split_metrics_stat[dataset][tracked_metric] = {}
            for dataset_eval in best_repeat_dicts[dataset][tracked_metric][split]:
                split_metrics[dataset][tracked_metric][dataset_eval] = {}
                split_metrics_stat[dataset][tracked_metric][dataset_eval] = {}

                metric_dict_in = best_repeat_dicts[dataset][tracked_metric][split][dataset_eval][tracked_metric]
                # double-check which dataset we actually had here
                model_path = get_model_path_from_repeat_best_dict(metric_dict_in, dataset, tracked_metric)
                model, _, _, _ = import_model_from_path(model_path=model_path,
                                                        validation_config=config['config']['VALIDATION'])

                # Here you can then compute additional metrics, like e.g. you never saved best model based on
                # Hausdorff Distance (HD), but here you can get the HD for the best model based on Dice,
                # and the Dice that you compute here should be the same as the tracked Dice (if this is confusing)
                metrics, metrics_stat = inference_dataloader(dataloader, model, device, split, config)

                split_metrics[dataset][tracked_metric][dataset_eval] = metrics
                split_metrics_stat[dataset][tracked_metric][dataset_eval] = metrics_stat

    return {'split_metrics': split_metrics, 'split_metrics_stat': split_metrics_stat}





def get_inference_metrics(y_pred: np.ndarray,
                          eval_config: dict,
                          batch_data: dict,
                          ensemble_stat_results: dict = None) -> dict:

    if 'label' in batch_data:
        # cannot compute any supervised metrics, if the label (segmentation mask) does not come with the image data
        x = conv_metatensor_to_numpy(batch_data['image'])
        y = ground_truth = conv_metatensor_to_numpy(batch_data['label'])
        ensemble_metrics = get_sample_metrics_from_np_masks(x, y, y_pred,
                                                            eval_config=eval_config)

        if ensemble_stat_results is not None:
            # if you have inferenced multiple repeats (or you have done MC Dropout or something)
            # you have some variance of each pixel and not just a y_pred mask:
            ensemble_uq_metrics = get_sample_uq_metrics_from_ensemble_stats(ensemble_stat_results)
            ensemble_metrics = merge_nested_dicts(ensemble_metrics, ensemble_uq_metrics) # {**ensemble_metrics, **ensemble_uq_metrics}

        ensemble_metrics['metadata'] = get_metadata_for_sample_metrics(metadata=batch_data['metadata'][0])

    else:
        ensemble_metrics = None

    return ensemble_metrics