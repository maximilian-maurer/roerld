import ray
from roerld.execution.control.remote_replay_buffer import RemoteReplayBuffer
import copy


class RayRemoteReplayBuffer(RemoteReplayBuffer):
    def __init__(self, replay_buffer_actor):
        self.replay_buffer_actor = replay_buffer_actor

    def store_batch_blocking(self, **kwargs):
        return ray.get(self.store_batch(**kwargs))

    def sample_batch_blocking(self, batch_size: int):
        batch = ray.get(self.sample_batch(batch_size))
        batch_copy = copy.deepcopy(batch)
        del batch
        return batch_copy

    def store_batch(self, **kwargs):
        return self.replay_buffer_actor.store_batch.remote(**kwargs)

    def sample_batch(self, batch_size: int):
        return self.replay_buffer_actor.sample_batch.remote(batch_size)

    def get(self, future):
        return ray.get(future)

    def wait_first_ready_future(self, futures):
        ready, _ = ray.wait(futures, num_returns=1)
        return ready[0]
