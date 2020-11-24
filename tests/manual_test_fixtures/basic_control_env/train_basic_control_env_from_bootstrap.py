import ray

from roerld.config import make_distributed_update_step_runner
from roerld.config.experiment_config import ExperimentConfig
from tests.environments.basic_control_test_env import register_basic_control_test_env
from tests.manual_test_fixtures.basic_control_env.config import _make_config


def train_basic_control_env():
    ray.init(ignore_reinit_error=True)

    def _actor_setup_function():
        register_basic_control_test_env()

    config_dict = ExperimentConfig.view(_make_config("../Datasets/basic-control-env-v1/bootstrap"))

    pipeline = make_distributed_update_step_runner(
        config_dict.section("pipeline"),
        config_dict,
        _actor_setup_function,
        epochs=500,
        restore_from_checkpoint_path=None,
    )
    pipeline.run()


if __name__ == "__main__":
    train_basic_control_env()
