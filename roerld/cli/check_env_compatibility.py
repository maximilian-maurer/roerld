import argparse
import sys

import gym
from roerld.cli.config_files import load_and_merge_configs
from roerld.config import make_environment
from roerld.config.experiment_config import ExperimentConfig


def fail_message(reason):
    print("--" * 20)
    print("Failed.")
    print(f"Reason: {reason}")


def check_env_compatibility_cli(argv):
    parser = argparse.ArgumentParser(
        description="Check for the compatibility of this environment with this pipeline. This will instantiate "
                    "the environment.")
    parser.add_argument("--config-file",
                        required=True,
                        action="append",
                        help="Path(s) to the experiment configuration file(s).")
    args = parser.parse_args(argv)

    config_dict = load_and_merge_configs(args.config_file)
    experiment_config = ExperimentConfig.view(config_dict)

    env_fn = lambda: make_environment(experiment_config.section("environment"))

    confirm = input("Will create the environment now (and consequently connect to the devices controlled by the "
                    "environment. "
                    "Before proceeding, ensure that this is safe."
                    "Proceed? (y/n)")

    if confirm != "y":
        print("Aborting.")
        return

    env = env_fn()
    observation_space = env.observation_space
    action_space = env.action_space
    env.close()

    if type(observation_space) != gym.spaces.Dict:
        fail_message("All envs for use with this pipeline must have Dict-space observations. Please use an adapter "
                     "class for this environment.")
        exit(-1)

    if type(action_space) != gym.spaces.Box:
        fail_message("This pipeline currently only supports environments with continuous action spaces.")
        exit(-2)

    if len(action_space.shape) != 1:
        fail_message("This pipeline only supports environments with flat action spaces. Please use an adapter class "
                     "for this environment.")
        exit(-3)

    print("Success.")
    exit(0)


if __name__ == "__main__":
    check_env_compatibility_cli(sys.argv[1:])
