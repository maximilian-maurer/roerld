import json
from typing import Iterable

from roerld.config.merge_configs import merge_configs


def load_and_merge_configs(paths: Iterable[str]):
    """ Loads the configuration files from the given paths.

    If there is more than a single configuration file in the list, the configuration files are merged per the rules
    in :func:`roerld.config.merge_config.merge_configs`

    :param paths A list of paths to the configuration files.
    """
    configuration_contents = []
    for config in paths:
        with open(config, "r") as config_file:
            configuration_contents.append(json.load(config_file))

    combined_config = {}
    for config in configuration_contents:
        combined_config = merge_configs(config, combined_config)

    return combined_config
