import gym.spaces
import numpy as np
from typing import Dict

from roerld.learning_actors.learning_actor import LearningActor


class RandomFromActionSpaceLearningActor(LearningActor):
    def __init__(self, action_space: gym.spaces.Space, scaling: float = 1.0):
        self.action_space = action_space
        self.scaling = scaling

    def choose_action(self, previous_observation: Dict, step_index: int, act_deterministically: bool,
                      additional_info: Dict) -> np.ndarray:
        return self.action_space.sample() * self.scaling

    def episode_started(self):
        pass

    def episode_ended(self):
        pass

