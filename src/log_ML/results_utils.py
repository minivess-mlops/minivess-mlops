import numpy as np
from loguru import logger

from src.inference.ensemble_utils import compute_stats_of_array_in_dict
from src.log_ML.log_utils import convert_value_to_numpy_array, compute_numpy_stats


def average_repeat_results(repeat_results):
    # exploit the same function written for cv results, and a dummy fold to the repeats
    dummy_fold_results = {"dummy_fold": {"dummy_architecture": repeat_results}}
    reordered_dummy_fold_results = reorder_crossvalidation_results(dummy_fold_results)
    averaged_results = compute_crossval_stats(reordered_dummy_fold_results)

    return averaged_results


def get_best_repeat_result(repeat_results):
    # exploit the same function written for cv results, and a dummy fold to the repeats
    dummy_fold_results = {"dummy_fold": {"dummy_architecture": repeat_results}}
    reordered_dummy_fold_results = reorder_crossvalidation_results(dummy_fold_results)

    best_repeat_dicts = find_best_repeat(
        reordered_results=reordered_dummy_fold_results, repeat_results=repeat_results
    )

    return best_repeat_dicts


def find_best_repeat(
    reordered_results: dict, repeat_results: dict, var_type_for_best: str = "scalars"
):
    best_repeat_dicts = {}
    repeat_names = list(repeat_results.keys())

    def pick_best_metric_value(var_name: str, values: float):
        """
        This now breaks easily as we should have some LUT for each possible metric, or
        define these already in the config. i.e. is smaller or larger value better.
        You could use the "METRICS_OPERATORS" in config['VALIDATION']
        :return:
        """
        if "loss" in var_name:
            best_direction = "lower"
            idx = int(np.argmin(values))
            best_value = np.min(values)
        else:
            best_direction = "higher"
            idx = np.argmin(values)
            best_value = np.min(values)

        return best_direction, idx, best_value

    for dataset in reordered_results:  # that it is trained on?
        best_repeat_dicts[dataset] = {}
        for tracked_metric in reordered_results[dataset]:
            best_repeat_dicts[dataset][tracked_metric] = {}
            for split in reordered_results[dataset][tracked_metric]:
                best_repeat_dicts[dataset][tracked_metric][split] = {}
                for dataset2 in reordered_results[dataset][tracked_metric][
                    split
                ]:  # that it is evaluated on?
                    best_repeat_dicts[dataset][tracked_metric][split][dataset2] = {}
                    for var_type in reordered_results[dataset][tracked_metric][split][
                        dataset2
                    ]:
                        if var_type_for_best == var_type:
                            for var_name in reordered_results[dataset][tracked_metric][
                                split
                            ][dataset2][var_type]:
                                values = reordered_results[dataset][tracked_metric][
                                    split
                                ][dataset2][var_type][var_name]
                                (
                                    best_direction,
                                    idx,
                                    best_value,
                                ) = pick_best_metric_value(var_name, values)
                                logger.info(
                                    '{}: Best repeat idx = {} ("{}", value = {:.3f}) | '
                                    " ({}, {}, {}, {})".format(
                                        var_name,
                                        idx,
                                        best_direction,
                                        best_value,
                                        dataset,
                                        tracked_metric,
                                        split,
                                        dataset2,
                                    )
                                )

                                result_dict = {
                                    "best_value": best_value,
                                    "best_idx": idx,
                                    "best_name": repeat_names[idx],
                                    "best_direction": best_direction,
                                    "repeat_best_dict": repeat_results[
                                        repeat_names[idx]
                                    ]["best_dict"],
                                }

                                best_repeat_dicts[dataset][tracked_metric][split][
                                    dataset2
                                ][var_name] = result_dict

    return best_repeat_dicts


def reorder_crossvalidation_results(fold_results: dict):
    res_out = {}

    no_of_folds = len(fold_results)
    for f, fold_key in enumerate(fold_results):
        fold_result = fold_results[fold_key]
        no_of_architectures = len(
            fold_result
        )  # same as number of submodels in an inference

        for a, archi_key in enumerate(fold_result):
            archi_result = fold_result[archi_key]
            no_of_repeats = len(archi_result)

            for r, repeat_key in enumerate(archi_result):
                repeat_results = archi_result[repeat_key]
                repeat_best = repeat_results["best_dict"]

                for d, ds in enumerate(repeat_best):
                    if ds not in res_out.keys():
                        res_out[ds] = {}

                    for m, metric in enumerate(repeat_best[ds]):
                        if metric not in res_out[ds].keys():
                            res_out[ds][metric] = {}
                        repeat_best_eval = repeat_best[ds][metric]["eval_epoch_results"]

                        for s, split in enumerate(repeat_best_eval):
                            if split not in res_out[ds][metric].keys():
                                res_out[ds][metric][split] = {}

                            for d_e, ds_eval in enumerate(repeat_best_eval[split]):
                                if ds_eval not in res_out[ds][metric][split].keys():
                                    res_out[ds][metric][split][ds_eval] = {}
                                eval_metrics = repeat_best_eval[split][ds_eval]

                                for t, var_type in enumerate(eval_metrics):
                                    if (
                                        var_type
                                        not in res_out[ds][metric][split][
                                            ds_eval
                                        ].keys()
                                    ):
                                        res_out[ds][metric][split][ds_eval][
                                            var_type
                                        ] = {}

                                    for v, var_name in enumerate(
                                        eval_metrics[var_type]
                                    ):
                                        value_in = convert_value_to_numpy_array(
                                            eval_metrics[var_type][var_name]
                                        )
                                        # value_in will have a shape of (1,) for scalars and will be aggregating them
                                        # so that you will have (no_folds, no_repeats) np.arrays in the rearranged dict
                                        if (
                                            var_name
                                            not in res_out[ds][metric][split][ds_eval][
                                                var_type
                                            ].keys()
                                        ):
                                            value_array = np.zeros(
                                                (
                                                    no_of_folds,
                                                    no_of_architectures,
                                                    no_of_repeats,
                                                )
                                            )
                                            res_out[ds][metric][split][ds_eval][
                                                var_type
                                            ][var_name] = value_array

                                        # res_out_tmp =
                                        # res_out[ds][metric][split][ds_eval][var_type][var_name]
                                        res_out[ds][metric][split][ds_eval][var_type][
                                            var_name
                                        ][f, a, r] = value_in

    return res_out


def compute_crossval_stats(fold_results_reordered: dict):
    res_out = {}

    for d, ds in enumerate(fold_results_reordered):
        if ds not in res_out.keys():
            res_out[ds] = {}

        for m, metric in enumerate(fold_results_reordered[ds]):
            if metric not in res_out[ds].keys():
                res_out[ds][metric] = {}
            best_metric = fold_results_reordered[ds][metric]

            for s, split in enumerate(best_metric):
                if split not in res_out[ds][metric].keys():
                    res_out[ds][metric][split] = {}

                for d_e, ds_eval in enumerate(best_metric[split]):
                    if ds_eval not in res_out[ds][metric][split].keys():
                        res_out[ds][metric][split][ds_eval] = {}
                    eval_metrics = best_metric[split][ds_eval]

                    for t, var_type in enumerate(eval_metrics):
                        if var_type not in res_out[ds][metric][split][ds_eval].keys():
                            res_out[ds][metric][split][ds_eval][var_type] = {}

                        for v, var_name in enumerate(eval_metrics[var_type]):
                            value_array_in = eval_metrics[var_type][var_name]
                            res_out[ds][metric][split][ds_eval][var_type][
                                var_name
                            ] = compute_numpy_stats(value_array_in)

    return res_out


def compute_crossval_ensemble_stats(ensembled_results_reordered: dict):
    stats_out = {}
    for s, split in enumerate(ensembled_results_reordered):
        stats_out[split] = {}
        for ens, ensemble_name in enumerate(ensembled_results_reordered[split]):
            stats_out[split][ensemble_name] = {}
            for m_e, metric in enumerate(
                ensembled_results_reordered[split][ensemble_name]
            ):
                stats_out[split][ensemble_name][metric] = {}
                for sk, stat in enumerate(
                    ensembled_results_reordered[split][ensemble_name][metric]
                ):
                    # the results might be confusing at first, as easiest just to loop with the same operations
                    # you get mean of n for example that might be quite useless. And mean of mean with dice just
                    # means that "1st mean" came from all the samples in the dataloader, and the "2nd mean" is
                    # the mean of all the folds
                    values = ensembled_results_reordered[split][ensemble_name][metric][
                        stat
                    ]
                    stats_dict = compute_stats_of_array_in_dict(values)
                    stats_out[split][ensemble_name][metric][stat] = stats_dict

    return stats_out


def reorder_ensemble_crossvalidation_results(
    ensembled_results, ensemble_stats_key: str = "stats"
):
    stats_out = {}
    for f, fold_key in enumerate(ensembled_results):
        if ensembled_results[fold_key] is not None:
            for s, split_name in enumerate(ensembled_results[fold_key]):
                if split_name not in stats_out:
                    stats_out[split_name] = {}

                for ens, ensemble_name in enumerate(
                    ensembled_results[fold_key][split_name]
                ):
                    if ensemble_name not in stats_out[split_name]:
                        stats_out[split_name][ensemble_name] = {}

                    ensemble_stats = ensembled_results[fold_key][split_name][
                        ensemble_name
                    ][ensemble_stats_key]
                    ensemble_metrics = ensemble_stats["metrics"]

                    for m_e, metric in enumerate(ensemble_metrics):
                        if metric not in stats_out[split_name][ensemble_name]:
                            stats_out[split_name][ensemble_name][metric] = {}
                        metric_stats_dict = ensemble_metrics[metric]

                        for sk, stat_key in enumerate(metric_stats_dict):
                            stat_value = metric_stats_dict[stat_key]
                            stat_array = np.expand_dims(np.array(stat_value), axis=0)
                            if (
                                stat_key
                                not in stats_out[split_name][ensemble_name][metric]
                            ):
                                stats_out[split_name][ensemble_name][metric][
                                    stat_key
                                ] = stat_array
                            else:
                                stats_out[split_name][ensemble_name][metric][
                                    stat_key
                                ] = np.concatenate(
                                    (
                                        stats_out[split_name][ensemble_name][metric][
                                            stat_key
                                        ],
                                        stat_array,
                                    ),
                                    axis=0,
                                )

        else:
            stats_out = None

    return stats_out


def get_cv_sample_stats_from_ensemble(
    ensembled_results, sample_key: str = "samples", split_key: str = "TEST"
):
    """
    We have here sample-level performance per fold.
    If you want to see how specific sample in "TEST" split performs across the folds (you could later do for VAL/TRAIN,
    but these would require more code as you don't have the same data then on these splits across the folds)

    :param ensembled_results:
    :param sample_key:
    :param split_key:
    :return:
    """

    no_of_folds = len(ensembled_results)
    stats_out = {}

    for f, fold_key in enumerate(ensembled_results):
        if split_key in ensembled_results[fold_key]:
            if split_key not in stats_out:
                stats_out[split_key] = {}
            sample_stats_per_fold = ensembled_results[fold_key][split_key][sample_key]
            # TO BE CONTINUED FROM HERE

    stats_out = "not_implemented_yet"

    return stats_out
