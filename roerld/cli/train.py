import argparse
import os
import sys

import ray
from roerld.cli.config_files import load_and_merge_configs
from roerld.config import make_distributed_update_step_runner
from roerld.config.experiment_config import ExperimentConfig, ExperimentConfigView


def _determine_training_length(args, experiment_config: ExperimentConfigView):
    if args.epochs is not None:
        return args.epochs
    return experiment_config.optional_key("general_config.epochs", 1e20)


def train_cli(argv, actor_setup_function):
    """ Provides a training CLI."""

    parser = argparse.ArgumentParser(description="Run the experiment described by the configuration file.")
    parser.add_argument("--config-file",
                        required=False,
                        action="append",
                        help="Path(s) to the experiment configuration file(s).")
    parser.add_argument("--epochs",
                        required=False,
                        type=int,
                        default=None,
                        help="Number of epochs to train for.")
    parser.add_argument("--tag",
                        required=True,
                        type=str,
                        help="Tag to use (appended to config tag).")
    parser.add_argument("--restore-from-checkpoint",
                        required=False,
                        type=str,
                        default=None,
                        help="Path to a checkpoint to restore from.")
    parser.add_argument("configs", nargs=argparse.REMAINDER)

    args = parser.parse_args(argv)

    configs = []
    if args.config_file is not None:
        configs.extend(args.config_file)
    if args.configs is not None:
        configs.extend(args.configs)

    ray.init(ignore_reinit_error=True)

    if len(configs) > 0:
        if args.restore_from_checkpoint is not None:
            print("A configuration was given with --config-file, however --restore-from-checkpoint was also set."
                  "Only one of these flags can be used at a time.")
            return

        config_dict = load_and_merge_configs(configs)

        if args.tag is not None:
            config_dict["general_config"]["experiment_tag"] = config_dict["general_config"]["experiment_tag"] + args.tag

        experiment_config = ExperimentConfig.view(config_dict)
        epochs = _determine_training_length(args, experiment_config)

        pipeline = make_distributed_update_step_runner(
            experiment_config.section("pipeline"),
            experiment_config,
            actor_setup_function,
            epochs=epochs,
            restore_from_checkpoint_path=None,
        )
        pipeline.run()
    elif args.restore_from_checkpoint is not None:
        if len(configs) > 0:
            print("If resuming from a checkpoint, --config-file may not be used to give a configuration file.")
            return

        config_paths = [os.path.join(args.restore_from_checkpoint, "experiment_config.json")]

        config_dict = load_and_merge_configs(config_paths)

        if args.tag is not None:
            config_dict["general_config"]["experiment_tag"] = config_dict["general_config"]["experiment_tag"] + args.tag

        experiment_config = ExperimentConfig.view(config_dict)
        epochs = _determine_training_length(args, experiment_config)

        pipeline = make_distributed_update_step_runner(
            experiment_config.section("pipeline"),
            experiment_config,
            actor_setup_function,
            epochs=epochs,
            restore_from_checkpoint_path=args.restore_from_checkpoint,
        )
        pipeline.run()
    else:
        print("Either --config-file or --restore-from-checkpoint must be given.")
        return


if __name__ == "__main__":
    train_cli(sys.argv[1:], lambda: None)
