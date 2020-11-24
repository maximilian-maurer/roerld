from abc import ABC
from typing import List, Any

from roerld.execution.transition_format import TransitionFormat


class StreamingEpisodePreprocessor(ABC):
    def transform_format(self, input_format: TransitionFormat) -> TransitionFormat:
        raise NotImplementedError()

    def episode_started(self):
        raise NotImplementedError()

    def receive_samples(self, samples: List[Any]) -> List[Any]:
        raise NotImplementedError()

    def episode_ended(self) -> List[Any]:
        raise NotImplementedError()
