from typing import Dict

import numpy as np

from roerld.learning_actors.learning_actor import LearningActor


class EpsilonGreedyLearningActor(LearningActor):
    def __init__(self, policy_actor: LearningActor, random_actor: LearningActor, epsilon: float):
        self.policy_actor = policy_actor
        self.random_actor = random_actor
        self.epsilon = epsilon

    def choose_action(self, previous_observation: Dict, step_index: int, act_deterministically: bool,
                      additional_info: Dict) -> np.ndarray:
        if act_deterministically or np.random.uniform(0, 1) >= self.epsilon:
            return self.policy_actor.choose_action(previous_observation, step_index,
                                                   act_deterministically, additional_info)
        return self.random_actor.choose_action(previous_observation,
                                               step_index,
                                               act_deterministically,
                                               additional_info)

    def episode_started(self):
        pass

    def episode_ended(self):
        pass


class DecayingEpsilonGreedyLearningActor(LearningActor):
    def __init__(self,
                 policy_actor: LearningActor,
                 random_actor: LearningActor,
                 epsilon_start,
                 epsilon_end,
                 decay_per_epoch):
        self.policy_actor = policy_actor
        self.random_actor = random_actor
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_per_epoch = decay_per_epoch

        self.first_onpolicy_epoch = -1

    def _epsilon(self, epoch):
        epsilon = max(self.epsilon_end, self.epsilon_start - self.decay_per_epoch * epoch)
        return epsilon

    def choose_action(self, previous_observation: Dict, step_index: int, act_deterministically: bool,
                      additional_info: Dict) -> np.ndarray:
        if not act_deterministically and self.first_onpolicy_epoch < 0:
            self.first_onpolicy_epoch = additional_info["rollout.epoch"]

        if not act_deterministically and \
                np.random.uniform(0, 1) < self._epsilon(additional_info["rollout.epoch"] - self.first_onpolicy_epoch):
            return self.random_actor.choose_action(previous_observation,
                                                   step_index,
                                                   act_deterministically,
                                                   additional_info)

        return self.policy_actor.choose_action(previous_observation, step_index,
                                               act_deterministically, additional_info)

    def episode_started(self):
        pass

    def episode_ended(self):
        pass
