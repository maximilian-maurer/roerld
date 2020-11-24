from typing import Callable

import ray

from roerld.execution.transition_format import TransitionFormat


@ray.remote
class DistributedPreprocessingActor:
    def __init__(self,
                 actor_setup_function: Callable[[], None]):
        self.round_robin_index = 0

        actor_setup_function()

    def set_workers(self, workers):
        self.workers = workers

    def transform_format(self, input_format: TransitionFormat) -> TransitionFormat:
        return ray.get(self.workers[0].transform_format.remote(input_format))

    def process_episode_waitable(self, episode, additional_metadata):
        result = self.workers[self.round_robin_index].process_episode.remote(episode, additional_metadata)
        self.round_robin_index = (self.round_robin_index + 1) % len(self.workers)
        return result

    def process_episode(self, episode, additional_metadata):
        result = self.workers[self.round_robin_index].process_episode.remote(episode, additional_metadata)
        self.round_robin_index = (self.round_robin_index + 1) % len(self.workers)
        return result
