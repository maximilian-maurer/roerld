from roerld.execution.control.remote_replay_buffer import RemoteReplayBuffer
from roerld.execution.control.worker_control import WorkerControl


class LocalWorkerControl(WorkerControl):
    def __init__(self, input_spec, observation_space, action_space):
        self._input_spec = input_spec
        self._observation_space = observation_space
        self._action_space = action_space

    def input_spec(self):
        return self._input_spec

    def replay_buffer(self, name) -> RemoteReplayBuffer:
        raise NotImplementedError()

    def observation_space(self):
        return self._observation_space

    def action_space(self):
        return self._action_space

    def epoch(self):
        return 0

    def create_temporary_directory(self):
        raise NotImplementedError()