from typing import Dict

import gym
import numpy as np

from roerld.learning_actors.learning_actor import LearningActor


class MockRolloutTestEnv(gym.Env):
    def __init__(self):
        self.num_close_called = 0
        self.num_reset_called = 0
        self.num_step_called = 0
        self.num_step_called_since_last_reset = 0
        self.num_render_called = 0

        self.action_log = []

        self.real_episode_length = self.max_episode_length

    @property
    def max_episode_length(self):
        return 101

    def set_episode_length(self, length):
        self.real_episode_length = length

    def _observe(self, action):
        return {
            "test1": action,
            "test2": 2. * action,
            "test3": [self.num_step_called_since_last_reset]
        }

    def step(self, action):
        self.num_step_called += 1
        self.num_step_called_since_last_reset += 1

        assert len(action) == 3
        assert type(action) == np.ndarray

        self.action_log.append(action)

        obs = self._observe(action)

        reward = self.num_step_called
        done = self.num_step_called_since_last_reset == self.max_episode_length or \
               self.num_step_called_since_last_reset >= self.real_episode_length
        info = {"num_step_called": self.num_step_called}
        return obs, reward, done, info

    def reset(self):
        self.num_reset_called += 1
        self.num_step_called_since_last_reset = 0

        return self._observe(np.asarray([-1, -1, -1]))

    def render(self, mode='human', width=48, height=56):
        self.num_render_called += 1

    def close(self):
        super().close()
        self.num_step_called += 1

    @property
    def action_space(self):
        return gym.spaces.Box(
            low=np.asarray([-100000, -100000, -100000]),
            high=np.asarray([100000, 100000, 100000]),
            dtype=np.float32
        )

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                "test1": gym.spaces.Box(low=np.asarray([-1000, -2000, -3000]),
                                        high=np.asarray([4666, 2666, 5878]),
                                        dtype=np.float32),
                "test2": gym.spaces.Box(low=np.asarray([-200, -2000, -2000]),
                                        high=np.asarray([2000, 2353, 2353]),
                                        dtype=np.float32),
                "test3": gym.spaces.Box(low=np.asarray([0]), high=np.asarray([100000]), dtype=np.float32),
            }
        )


class MockRolloutTestActor(LearningActor):
    def __init__(self):
        self.num_episode_start_called = 0
        self.num_choose_action_called = 0
        self.num_episode_ended_called = 0
        self.last_determinism = None

        self.actions_performed = []

    def choose_action(self, previous_observation: Dict, step_index: int, act_deterministically: bool,
                      additional_info: Dict) -> np.ndarray:
        self.num_choose_action_called += 1
        self.last_determinism = act_deterministically
        action = np.asarray([self.num_choose_action_called,
                             2 * self.num_choose_action_called, -
                             2 * self.num_choose_action_called])
        self.actions_performed.append(action)
        return action

    def episode_started(self):
        self.num_episode_start_called += 1

    def episode_ended(self):
        self.num_episode_ended_called += 1


class MockFixedActionActor(LearningActor):
    def __init__(self, value):
        self.value = value

    def choose_action(self, previous_observation: Dict, step_index: int, act_deterministically: bool,
                      additional_info: Dict) -> np.ndarray:
        return self.value

    def episode_started(self):
        pass

    def episode_ended(self):
        pass