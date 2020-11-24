from typing import Union, List, Any

from roerld.execution.transition_format import TransitionFormat
from roerld.preprocessing.streaming_episode_preprocessor import StreamingEpisodePreprocessor

import copy

import numpy as np


class SubsamplingResampler(StreamingEpisodePreprocessor):
    def __init__(self, sample_interval):
        self.buffer = []
        self.sample_interval = sample_interval
        self.input_format = None  # type: TransitionFormat

    def transform_format(self, input_format: TransitionFormat) -> TransitionFormat:
        self.input_format = input_format
        return copy.deepcopy(input_format)

    def episode_started(self):
        assert len(self.buffer) == 0
        pass

    def receive_samples(self, samples: List[Any]) -> Union[None, List[Any]]:
        self.buffer.extend(samples)

        if len(self.buffer) < self.sample_interval:
            return []

        emitted_transitions = []
        while len(self.buffer) >= self.sample_interval:
            this_transition_samples = self.buffer[:self.sample_interval]

            observation_keys = self.input_format.observation_in_transition_keys()
            next_observation_keys = self.input_format.next_observation_in_transition_keys()
            previous_observation = {k:  this_transition_samples[0][k] for k in observation_keys}
            next_observation = {k: this_transition_samples[-1][k] for k in next_observation_keys}
            reward = np.sum([this_transition_samples[i]["rewards"] for i in range(len(this_transition_samples))])
            done = any([s["dones"] for s in this_transition_samples])
            info = {}

            actions = [s["actions"] for s in this_transition_samples]

            # todo strict mode
            #assert np.allclose(actions[0], actions)

            emitted_transitions.append({
                **previous_observation,
                **next_observation,
                "rewards": reward,
                "dones": done,
                "infos": info,
                "actions": actions[0]
            })
            self.buffer = self.buffer[self.sample_interval:]
        return emitted_transitions

    def episode_ended(self) -> Union[None, List[Any]]:
        assert len(self.buffer) == 0
        return []
