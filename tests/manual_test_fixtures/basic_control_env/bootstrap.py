import gym
import json
import ray

from roerld.bootstrapping.bootstrap import bootstrap
from roerld.config.experiment_config import ExperimentConfig
from roerld.data_handling.json_driven_data_source import JsonDrivenDataSource
from tests.environments.basic_control_test_env import register_basic_control_test_env


def bootstrap_basic_control_env():
    ray.init(ignore_reinit_error=True)

    def _actor_setup_function():
        register_basic_control_test_env()

    _actor_setup_function()

    def _environment_factory():
        return gym.make("BasicControlEnv-v1")

    data_source = JsonDrivenDataSource("../Datasets/basic-control-env-v1/bootstrap")
    writer = data_source.writer(image_keys=[],
                                max_episodes_per_file=100)

    config_dict = ExperimentConfig.view(_make_bootstrap_config())
    bootstrap(
        experiment_config=config_dict,
        environment_factory=_environment_factory,
        writer=writer,
        actor_setup_function=_actor_setup_function,
        log_function=lambda x: print(x),
        num_jobs=5,
        num_samples=500000,
        display_render_mode=None
    )


def _make_bootstrap_config():
    config_json = """
{
    "general_config": {
        "experiment_tag": "BasicControlEnv",
        "seed": 1066
    },
    "bootstrapping": {
        "actor": {
            "name": "random_from_action_space"
        },
        "max_episode_length": 100
    }
}
    """
    return json.loads(config_json)


if __name__ == "__main__":
    bootstrap_basic_control_env()
