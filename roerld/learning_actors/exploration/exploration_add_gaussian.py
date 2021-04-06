from typing import Union, Dict

import gym
import numpy as np

from roerld.learning_actors import LearningActor


class AddGaussianRandomLearningActor(LearningActor):
    def __init__(self,
                 policy: LearningActor,
                 action_space: gym.spaces.Box,
                 mean: Union[float, np.ndarray],
                 std: Union[float, np.ndarray]):
        self.action_space = action_space
        self.mean = mean
        if not np.isscalar(self.mean):
            self.mean = np.asarray(self.mean)
        self.std = std
        if not np.isscalar(self.std):
            self.std = np.asarray(self.std)
        self.policy = policy

    def choose_action(self, previous_observation: Dict,
                      step_index: int,
                      act_deterministically: bool,
                      additional_info: Dict) -> np.ndarray:
        random_addition = None
        if np.isscalar(self.mean) and np.isscalar(self.std):
            random_addition = np.random.normal(loc=self.mean * np.ones(self.action_space.shape),
                                               scale=self.std * np.ones(self.action_space.shape))
        elif not np.isscalar(self.mean) and not np.isscalar(self.std):
            random_addition = np.random.normal(loc=self.mean, scale=self.std)

        random_action = self.policy.choose_action(previous_observation,
                                                  step_index,
                                                  act_deterministically,
                                                  additional_info) + random_addition

        return np.clip(random_action, a_min=self.action_space.low, a_max=self.action_space.high)

    def episode_started(self):
        pass

    def episode_ended(self):
        pass
