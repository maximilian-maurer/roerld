from abc import ABC


class RemoteReplayBuffer(ABC):

    def store_batch_blocking(self, **kwargs):
        raise NotImplementedError()

    def sample_batch_blocking(self, batch_size: int):
        raise NotImplementedError()

    def store_batch(self, **kwargs):
        raise NotImplementedError()

    def sample_batch(self, batch_size: int):
        raise NotImplementedError()

    def get(self, future):
        raise NotImplementedError()

    def wait_first_ready_future(self, futures):
        raise NotImplementedError()
