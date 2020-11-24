from typing import Callable, List

import numpy as np
import ray

from roerld.execution.transition_format import TransitionFormat
from roerld.replay_buffers.replay_buffer import ReplayBuffer


@ray.remote
class SingleInstanceReplayBufferActor(ReplayBuffer):
    """
    As opposed to a front for a distributed replay buffer, this is a single worker in charge of a replay buffer
    local to that worker
    """

    def __init__(self,
                 buffer_factories: List[Callable[[TransitionFormat], ReplayBuffer]],
                 transition_format: TransitionFormat,
                 seed: int,
                 actor_setup_function: Callable[[], None]):
        actor_setup_function()
        np.random.seed(seed)

        assert len(buffer_factories) == 1
        self.replay_buffer = buffer_factories[0](transition_format)

    def store_batch(self, **kwargs):
        return self.replay_buffer.store_batch(**kwargs)

    def sample_batch(self, batch_size: int):
        return self.replay_buffer.sample_batch(batch_size)

    def sample_count(self):
        return self.replay_buffer.sample_count()

    def max_size(self):
        return self.replay_buffer.max_size()
