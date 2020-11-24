from abc import abstractmethod, ABC
from typing import Dict

import numpy as np


class LearningActor(ABC):
    """ The actor may take actions in the environment.

    (It is not to be confused with the actors in ray).

    todo This interface temporarily duplicates the BootstrappingLearningActor interface which is currently retained for
            compatibility reasons. Moving forward, this will be the unified actor interface for exploration
            and bootstrapping.
    """

    @abstractmethod
    def choose_action(self,
                      previous_observation: Dict,
                      step_index: int,
                      act_deterministically: bool,
                      additional_info: Dict) -> np.ndarray:
        """
        Chooses the next action.

        Args:
            previous_observation: The last observation.
            step_index: The timestep within the episode.
            act_deterministically: Whether the action should be selected deterministically (e.g. during an evaluation
                                        rollout).
            additional_info:

        Returns:
            The action to perform.
        """
        raise NotImplementedError()

    @abstractmethod
    def episode_started(self):
        pass

    @abstractmethod
    def episode_ended(self):
        pass
