from typing import List, Dict, Any

import numpy as np


def _flatten2d(list_of_lists):
    result = []
    for inner_list in list_of_lists:
        if type(inner_list) == list or type(inner_list) == np.ndarray:
            result.extend(inner_list)
        else:
            result.append(inner_list)
    return result


def _prettify_key_name(key_name):
    key_name = key_name.replace("_", " ")
    key_name = key_name.title()
    return key_name


def aggregate_diagnostics(diagnostics_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """ Takes in a list of dictionaries containing the diagnostic data.

    :param diagnostics_data List of Dictionaries containing arrays of numbers as their keys. These
                                are all aggregated for each key, flattened and summary statistics calculated
                                on them.
    """
    if len(diagnostics_data) == 0:
        return {}

    as_dict = {}
    for item in diagnostics_data:
        for k, v in item.items():
            if k in as_dict:
                as_dict[k].append(v)
            else:
                as_dict[k] = [v]

    # at this point, each of the keys can, through the aggregation, have a list of lists structure with different
    #  sized inner lists resulting from different rollout assignments to different workers, this needs to be
    #  flattened first
    for key in as_dict:
        as_dict[key] = np.array(_flatten2d(as_dict[key]))

    result = {}
    for key in as_dict:
        if len(as_dict[key]) == 0:
            continue

        result[_prettify_key_name(key) + "/ Mean"] = np.mean(as_dict[key])
        result[_prettify_key_name(key) + "/ Min"] = np.min(as_dict[key])
        result[_prettify_key_name(key) + "/ Max"] = np.max(as_dict[key])
        result[_prettify_key_name(key) + "/ STDev"] = np.std(as_dict[key])
        result[_prettify_key_name(key) + "/ Count"] = len(as_dict[key])
        result[_prettify_key_name(key) + "/ Sum"] = np.sum(as_dict[key])

    return result
