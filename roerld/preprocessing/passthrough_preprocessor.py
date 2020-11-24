import copy

from roerld.execution.transition_format import TransitionFormat
from roerld.preprocessing.full_episode_preprocessor import FullEpisodePreprocessor


class PassthroughPreprocessor(FullEpisodePreprocessor):
    def __init__(self):
        super().__init__()

    def _process_episode(self, samples):
        return samples

    def transform_format(self, input_format: TransitionFormat) -> TransitionFormat:
        return copy.deepcopy(input_format)
