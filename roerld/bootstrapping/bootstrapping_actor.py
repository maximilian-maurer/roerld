from typing import Callable

import gym.spaces
import numpy as np
import ray
import tensorflow as tf

from roerld.config.experiment_config import ExperimentConfig
from roerld.execution.rollouts.rollout_worker import RolloutWorker
from roerld.learning_actors.learning_actor import LearningActor


@ray.remote
class BootstrappingActor:
    def __init__(self,
                 environment_factory,
                 learning_actor_factory: Callable[[gym.spaces.Space], LearningActor],
                 rollout_config,
                 max_episode_length,
                 seed,
                 actor_setup_function: Callable[[], None],
                 **kwargs):
        actor_setup_function()

        tf.random.set_seed(seed)
        np.random.seed(seed)

        self.rollout_config = ExperimentConfig.view(rollout_config)
        self.kwargs = kwargs

        self.environment = environment_factory()
        self.learning_actor = learning_actor_factory(self.environment.action_space)
        self.rollout_worker_params = {
            "environment": self.environment,
            "max_episode_length": max_episode_length,
            "local_render_mode": None,
            "eval_video_height": self.rollout_config.key("evaluation.video_height"),
            "eval_video_width": self.rollout_config.key("evaluation.video_width"),
            "eval_video_render_mode": self.rollout_config.key("evaluation.video_render_mode")
        }

        self.rollout_worker = RolloutWorker(actor=self.learning_actor, **self.rollout_worker_params)

    def collect_samples(self, num_samples):
        collected_samples = 0

        experiences = []
        while collected_samples < num_samples:
            extra_info = {
                "rollout.environment": self.environment,
            }
            experience, videos, episode_starts, diagnostics = self.rollout_worker.training_rollout(1, False, passthrough_extra_info=extra_info)
            experiences.append(experience)
            collected_samples += len(experience[list(experience.keys())[0]])
        return experiences

    def env_observation_space(self):
        return self.environment.observation_space

    def env_action_space(self):
        return self.environment.action_space

    def close(self):
        self.environment.close()
