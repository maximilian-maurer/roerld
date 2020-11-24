from typing import List, Any

from roerld.execution.transition_format import TransitionFormat
from roerld.preprocessing.streaming_episode_preprocessor import StreamingEpisodePreprocessor


class PreprocessingChain(StreamingEpisodePreprocessor):
    def __init__(self, preprocessors: List[StreamingEpisodePreprocessor]):
        self.preprocessors = preprocessors

    def transform_format(self, input_format: TransitionFormat) -> TransitionFormat:
        current_format = input_format
        for p in self.preprocessors:
            current_format = p.transform_format(current_format)
        return current_format

    def episode_started(self):
        for p in self.preprocessors:
            p.episode_started()

    def receive_samples(self, samples: List[Any]) -> List[Any]:
        return self._receive_samples(samples, 0)

    def _receive_samples(self, samples: List[Any], level: int) -> List[Any]:
        intermediate_samples = samples
        for i in range(level, len(self.preprocessors)):
            p = self.preprocessors[i]
            intermediate_samples = p.receive_samples(intermediate_samples)
        return intermediate_samples

    def episode_ended(self) -> List[Any]:
        output_samples = []
        for level, p in enumerate(self.preprocessors):
            collected_samples = p.episode_ended()
            if len(collected_samples) <= 0:
                continue

            if level + 1 < len(self.preprocessors):
                output_samples.extend(self._receive_samples(collected_samples, level + 1))
            else:
                output_samples.extend(collected_samples)
        return output_samples
