from typing import Any, Dict, Union
import numpy as np


def normalize_summary_format(data: Dict[str, Any]) -> Dict[str, Union[float, np.ndarray, str]]:
    """
    Takes a Dict[str, Any] and returns a dict whose entries are either:
        * str: float
        * str: np.ndarray
        * str: str
    Anything that is not either a float or an array is stringified
    """
    results = {}
    for key, value in data.items():
        if np.isscalar(value) or isinstance(value, np.ndarray):
            # this also captures builtin strings, but we want to keep str: str mappings intact so this is desired
            results[key] = value
        elif isinstance(value, list):
            results[key] = np.asarray(value)
        else:
            results[key] = str(value)
    return results


def aggregates_of_summary(data: Dict[str, Union[float, np.ndarray, str]], include_nonaggregatable=False) \
        -> Dict[str, Union[float, str]]:
    """
    Returns a new dictionary where np.ndarray keys have been instead turned into multiple summary keys for
    mean and std (named 'key_mean' and 'key_std'). These are computed on completely flattened versions of the input
    keys.
    :param data:
    :param include_nonaggregatable If a key has another type, include it as is, or not?
    """
    results = {}
    for key, value in data.items():
        if np.isscalar(value):
            if include_nonaggregatable:
                results[key] = value
            continue
        results[f"{key} Mean"] = np.mean(np.asarray(value).flatten())
        results[f"{key} STDev"] = np.std(np.asarray(value).flatten())
    return results
