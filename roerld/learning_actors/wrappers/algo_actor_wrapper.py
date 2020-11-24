import numpy as np
from typing import Dict

from roerld.execution.distributed_update_step_algorithm import DistributedUpdateStepAlgorithm
from roerld.learning_actors.learning_actor import LearningActor


class AlgoActorWrapper(LearningActor):
    def __init__(self, algorithm: DistributedUpdateStepAlgorithm):
        self.algorithm = algorithm

    def choose_action(self, previous_observation: Dict, step_index: int, act_deterministically: bool,
                      additional_info: Dict) -> np.ndarray:
        return self.algorithm.choose_action(previous_observation)

    def episode_started(self):
        pass

    def episode_ended(self):
        pass

