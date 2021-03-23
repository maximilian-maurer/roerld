from enum import Enum

import os


class PathKind(Enum):
    NewLog = 1,
    NewBootstrap = 2,
    ConfigDirectory = 3,
    BootstrapDataConfigDirectory = 4,


def resolve_path(experiment_config, path_kind, categories=None):
    from roerld.config.experiment_config import ExperimentConfig

    if categories is None:
        categories = []

    experiment_config = ExperimentConfig.view(experiment_config)

    environment_scope = experiment_config.key("environment.scope")
    environment_id = experiment_config.key("environment.name")

    categories = sorted(categories)
    categories_folder = "_".join(categories)

    environment_id_parts = "/".join(environment_id.split("-"))
    environment_folder = f"{environment_scope}_{environment_id_parts}"

    subfolders = [environment_folder]
    if categories_folder != "":
        subfolders.append(categories_folder)
    base_path_with_category = os.path.join(*subfolders)
    base_path_without_category = environment_folder

    if path_kind == PathKind.NewLog:
        return os.path.join(base_path_with_category)
    if path_kind == PathKind.NewBootstrap:
        data_root = os.path.join("../Datasets/", base_path_with_category)
        if not os.path.exists(data_root):
            return os.path.join(data_root, "data_0")

        sub_folders = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]

        for i in range(10000):
            name = f"data_{i}"
            if name in sub_folders:
                continue
            return os.path.join(data_root, name)

        raise ValueError("Cannot resolve boostrap folder path.")
    if path_kind == PathKind.ConfigDirectory:
        return os.path.join("configs", base_path_without_category)
    if path_kind == PathKind.BootstrapDataConfigDirectory:
        parts = ["configs", base_path_without_category, "data", "bootstrap"]
        if categories_folder != "":
            parts.append(categories_folder)
        return os.path.join(*parts)

    raise ValueError("Cannot resolve path.")


