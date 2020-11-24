from roerld.config.experiment_config import ExperimentConfigError


def merge_configs(config_1: dict, config_2: dict):
    """ Merges two configurations.

    Two configurations may be merged if they do not contain any overlapping keys.

    The exception to this is the key `experiment_tag` which is concatenated with an underscore inbetween.
    """
    result_config = {}

    keys_1 = list(config_1.keys())
    keys_2 = list(config_2.keys())
    keys = keys_1 + keys_2

    for key in keys:
        if key == "experiment_tag" and key in keys_1 and key in keys_2:
            result_config[key] = f"{config_1[key]}_{config_2[key]}"
            continue

        if key in keys_1 and key not in keys_2:
            result_config[key] = config_1[key]
        elif key not in keys_1 and key in keys_2:
            result_config[key] = config_2[key]
        else:
            if type(config_1[key]) == dict and type(config_2[key]) == dict:
                result_config[key] = merge_configs(config_1[key], config_2[key])
            else:
                raise ExperimentConfigError(f"Cannot merge two configs who provide conflicting values for "
                                            f"the same key. ({key})")
    return result_config
