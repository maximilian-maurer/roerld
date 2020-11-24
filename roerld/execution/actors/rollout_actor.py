from typing import Callable, Tuple

import gym.spaces
import ray
import tensorflow as tf

from roerld.config.experiment_config import ExperimentConfig
from roerld.execution.distributed_update_step_algorithm import DistributedUpdateStepAlgorithm
from roerld.learning_actors.learning_actor import LearningActor


@ray.remote
class RolloutActor:
    def __init__(self,
                 environment_factory,
                 algorithm_factory: Callable[[gym.spaces.Space], Tuple[DistributedUpdateStepAlgorithm, LearningActor]],
                 rollout_config,
                 max_episode_length,
                 seed,
                 actor_setup_function: Callable[[], None],
                 coordinator_actor,
                 **kwargs):
        actor_setup_function()

        # todo seed handling

        self.rollout_config = ExperimentConfig.view(rollout_config)

        if "tf_inter_op_parallelism_threads" in kwargs:
            tf.config.threading.set_inter_op_parallelism_threads(kwargs["tf_inter_op_parallelism_threads"])
        if "tf_intra_op_parallelism_threads" in kwargs:
            tf.config.threading.set_intra_op_parallelism_threads(kwargs["tf_intra_op_parallelism_threads"])

        for gpu in tf.config.experimental.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)

        self.kwargs = kwargs
        self.coordinator_actor = coordinator_actor

        self.environment = environment_factory()
        self.algorithm, self.learning_actor = algorithm_factory(self.environment.action_space)
        self.rollout_worker_params = {
            "environment": self.environment,
            "max_episode_length": max_episode_length,
            "local_render_mode": None,
            "eval_video_height": self.rollout_config.optional_key("evaluation.video_height", None),
            "eval_video_width": self.rollout_config.optional_key("evaluation.video_width", None),
            "eval_video_render_mode": self.rollout_config.key("evaluation.video_render_mode"),
            "render_every_n_frames": self.rollout_config.optional_key("evaluation.render_every_n_frames", 1)
        }

        self.rollout_worker = None

    def initialize(self, worker_control):
        from roerld.execution.rollouts.rollout_worker import RolloutWorker

        self.algorithm.setup(worker_control=worker_control, worker_specific_kwargs=self.kwargs)
        self.rollout_worker = RolloutWorker(actor=self.learning_actor, **self.rollout_worker_params)

    def rollout(self, num_episodes, weights, info, is_evaluation, render_eval_videos):
        self.algorithm.update_weights(weights)

        epoch = ray.get(self.coordinator_actor.epoch.remote())
        extra_info = {
            "rollout.environment": self.environment,
            "rollout.epoch": epoch
        }

        # unpack manually here to maintain compatibility to lower python versions
        if is_evaluation:
            experience, videos, episode_starts, diagnostics = \
                self.rollout_worker.evaluation_rollout(num_episodes, render_eval_videos, extra_info)
        else:
            experience, videos, episode_starts, diagnostics = \
                self.rollout_worker.training_rollout(num_episodes, render_eval_videos, extra_info)

        return experience, videos, episode_starts, diagnostics, info

    def env_observation_space(self):
        return self.environment.observation_space

    def env_action_space(self):
        return self.environment.action_space

    def close(self):
        self.environment.close()
