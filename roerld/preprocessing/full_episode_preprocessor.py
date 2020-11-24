from abc import ABC
from typing import List, Any

from roerld.preprocessing.streaming_episode_preprocessor import StreamingEpisodePreprocessor


class FullEpisodePreprocessor(StreamingEpisodePreprocessor, ABC):
    def __init__(self):
        self.buffer = []

    def _process_episode(self, samples):
        raise NotImplementedError()

    def episode_started(self):
        if len(self.buffer) > 0:
            raise AssertionError("FullEpisodePreprocessor received new episode before the episode_ended callback"
                                 "was called for the previous episode")

    def receive_samples(self, samples) -> List[Any]:
        self.buffer.extend(samples)
        return []

    def episode_ended(self) -> List[Any]:
        result = self._process_episode(self.buffer)
        self.buffer = []
        return result
