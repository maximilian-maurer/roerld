import os
from typing import Dict

import ray
from roerld.execution.control.remote_replay_buffer import RemoteReplayBuffer
from roerld.execution.control.worker_control import WorkerControl


class DistributedUpdateStepPipelineWorkerControl(WorkerControl):
    def __init__(self,
                 input_spec,
                 replay_buffers: Dict[str, RemoteReplayBuffer],
                 observation_space,
                 action_space,
                 coordinator_actor,
                 name):
        self.coordinator_actor = coordinator_actor
        self._action_space = action_space
        self._observation_space = observation_space
        self.replay_buffers = replay_buffers
        self._input_spec = input_spec
        self.name = name
        self.temporary_directory_index = 0

    def input_spec(self):
        return self._input_spec

    def replay_buffer(self, name) -> RemoteReplayBuffer:
        return self.replay_buffers[name]

    def observation_space(self):
        return self._observation_space

    def action_space(self):
        return self._action_space

    def epoch(self):
        return ray.get(self.coordinator_actor.epoch.remote())

    def create_temporary_directory(self):
        path = os.path.join("temporary_files", self.name, str(self.temporary_directory_index))

        os.makedirs(path)
        self.temporary_directory_index += 1
        return path