from abc import ABC, abstractmethod

from roerld.execution.control.remote_replay_buffer import RemoteReplayBuffer


class WorkerControl(ABC):

    @abstractmethod
    def input_spec(self):
        raise NotImplementedError()

    @abstractmethod
    def replay_buffer(self, name) -> RemoteReplayBuffer:
        raise NotImplementedError()

    @abstractmethod
    def observation_space(self):
        raise NotImplementedError()

    @abstractmethod
    def action_space(self):
        raise NotImplementedError()

    @abstractmethod
    def epoch(self):
        raise NotImplementedError()

    @abstractmethod
    def create_temporary_directory(self):
        raise NotImplementedError()
