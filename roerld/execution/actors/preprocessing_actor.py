import copy
from typing import Callable

import numpy as np
import ray

from roerld.execution.transition_format import TransitionFormat
from roerld.preprocessing.streaming_episode_preprocessor import StreamingEpisodePreprocessor


@ray.remote
class PreprocessingActor:
    def __init__(self,
                 chain_factory: Callable[[], StreamingEpisodePreprocessor],

                 target_episode_writer_actors,
                 target_replay_buffer_actors,

                 seed: int,
                 actor_setup_function: Callable[[], None]):
        self.target_replay_buffer_actors = target_replay_buffer_actors
        self.target_episode_writer_actors = target_episode_writer_actors

        actor_setup_function()
        np.random.seed(seed)

        self.preprocessing_chain = chain_factory()

    def transform_format(self, input_format: TransitionFormat) -> TransitionFormat:
        return self.preprocessing_chain.transform_format(input_format)

    def process_episode(self, episode, additional_metadata):
        processed_transitions = []

        episode_old = episode
        episode_new = copy.deepcopy(episode_old)
        del episode
        episode = episode_new

        self.preprocessing_chain.episode_started()
        episode_length = len(episode[list(episode.keys())[0]])
        for i in range(episode_length):
            transition = {
                k: v[i] for k, v in episode.items()
            }
            processed_transitions.extend(self.preprocessing_chain.receive_samples([transition]))
        processed_transitions.extend(self.preprocessing_chain.episode_ended())

        episode = {
            k: np.asarray([t[k] for t in processed_transitions]) for k in processed_transitions[0].keys()
        }

        for buffer in self.target_replay_buffer_actors:
            buffer.store_batch.remote(**episode)
        for episode_writer in self.target_episode_writer_actors:
            episode_writer.store_episode.remote(episode, additional_metadata)
        return None

    def process_episode_waitable(self, episode, additional_metadata):
        return self.process_episode(episode, additional_metadata)