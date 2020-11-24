import argparse
import os
import sys

import ray

from roerld.bootstrapping.bootstrap import bootstrap
from roerld.cli.config_files import load_and_merge_configs
from roerld.config import make_environment
from roerld.config.experiment_config import ExperimentConfig
from roerld.config.registry import make_data_source


def bootstrap_cli(command_line_args, worker_setup_function=None):
    """Provides a bootstrapping CLI.
    
    :param command_line_args the command line arguments (not including the executable path)
    :param worker_setup_function Function that will be run on each ray actor started. Use this to set up registration
                                    and state that needs to be set for each python process (such as environment
                                    registration).
    """
    parser = argparse.ArgumentParser(
        description="Run bootstrapping for the experiment as described by its configuration files.")
    parser.add_argument("-c", "--config-file",
                        required=False,
                        action="append",
                        help="Path(s) to the experiment configuration file(s).")
    parser.add_argument("-s", "--samples",
                        required=True,
                        type=int,
                        help="Number of samples to collect.")
    parser.add_argument("-j", "--jobs",
                        required=True,
                        type=int,
                        help="Number of parallel instances of the environment to use for sampling.")
    parser.add_argument("-o", "--out",
                        required=False,
                        type=str,
                        help="Output Path.")
    parser.add_argument("-cat", "--category",
                        required=False,
                        type=str,
                        action="append",
                        help="Category.")
    parser.add_argument("--display_render_mode",
                        required=False,
                        type=str,
                        help="Render mode for live display. If set, it will be passed to the environment's render "
                             "method.")
    parser.add_argument("configs", nargs=argparse.REMAINDER)

    args = parser.parse_args(command_line_args)
    _bootstrap_cli(args, worker_setup_function)


def env_registration_curry(env_function, worker_setup_function):
    def register_and_make():
        if worker_setup_function is not None:
            worker_setup_function()
        return env_function()

    return register_and_make


def _bootstrap_cli(args, actor_setup_function):
    """ :see bootstrap_cli"""
    from roerld.cli.paths import resolve_path, PathKind

    configs = []
    if args.config_file is not None:
        configs.extend(args.config_file)
    if args.configs is not None:
        configs.extend(args.configs)

    config_dict = load_and_merge_configs(configs)
    experiment_config = ExperimentConfig.view(config_dict)

    def _make_environment():
        return make_environment(experiment_config.section("environment"))

    def _log_function(*args, **kwargs):
        print(*args, **kwargs)

    actor_setup_function()

    env_factory = _make_environment

    ray.init(ignore_reinit_error=True)

    data_folder = args.out
    if data_folder is None:
        try:
            data_folder = resolve_path(experiment_config=experiment_config,
                                       path_kind=PathKind.NewBootstrap,
                                       categories=args.category)
            os.makedirs(data_folder)
        except ValueError:
            print("No output directory was given and it was not possible to automatically generate one. "
                  "Use --out to provide an output directory.")
            return

    print("Outputting to ", data_folder)

    # also create a config preset to save time when frequently changing environments
    desired_config_folder = resolve_path(experiment_config=experiment_config,
                                         path_kind=PathKind.BootstrapDataConfigDirectory,
                                         categories=args.category)
    os.makedirs(desired_config_folder, exist_ok=True)

    config_template = """{
    "workers": {
        "log_replay_workers": {
            "name": "worker_groups",
            "workers": [
                {
                    "name": "worker_group",
                    "count": 1,
                    "ray_kwargs": {
                        "num_cpus": 1,
                        "num_gpus": 0,
                        "resources": {}
                    },
                    "worker_kwargs": {
                        "max_new_samples_per_epoch": 10000,
                        "data_folder":  "DATA_FOLDER"
                    }
                }
            ]
        }
    }
    }"""
    config_template = config_template.replace("DATA_FOLDER", data_folder)

    pregen_path = os.path.join(desired_config_folder,
                           os.path.basename(os.path.normpath(data_folder))+"_l1.json")
    if not os.path.exists(pregen_path):
        with open(pregen_path, "w") as pregenerated_config:
            pregenerated_config.write(config_template)
    else:
        print(f"Warning: There is already a configuration file {pregen_path}")

    dataset = make_data_source(experiment_config.section("io"), [data_folder])
    writer = dataset.writer()

    bootstrap(experiment_config=experiment_config,
              environment_factory=env_factory,
              writer=writer,
              actor_setup_function=actor_setup_function,
              log_function=_log_function,
              num_jobs=args.jobs,
              num_samples=args.samples,
              display_render_mode=args.display_render_mode)


if __name__ == "__main__":
    ray.init()
    bootstrap_cli(sys.argv[1:])
