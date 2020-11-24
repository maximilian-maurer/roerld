from typing import Union, Dict

import numpy as np
import gym

from roerld.learning_actors.learning_actor import LearningActor


class GaussianRandomLearningActor(LearningActor):
    def __init__(self, action_space: gym.spaces.Box,
                 mean: Union[float, np.ndarray],
                 std: Union[float, np.ndarray]):
        self.action_space = action_space
        self.mean = mean
        self.std = std

    def choose_action(self, previous_observation: Dict,
                      step_index: int,
                      act_deterministically: bool,
                      additional_info: Dict) -> np.ndarray:
        random_action = np.random.normal(loc=self.mean, scale=self.std)
        assert self.action_space.shape == random_action.shape

        return np.clip(random_action, a_min=self.action_space.low, a_max=self.action_space.high)

    def episode_started(self):
        pass

    def episode_ended(self):
        pass
