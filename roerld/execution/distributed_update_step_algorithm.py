from abc import abstractmethod, ABC
from typing import Tuple, Dict, Any

from roerld.execution.actor import Actor
from roerld.execution.control.driver_control import DriverControl
from roerld.execution.control.worker_control import WorkerControl


class DistributedUpdateStepAlgorithm(Actor, ABC):
    @abstractmethod
    def setup(self, worker_control: WorkerControl, worker_specific_kwargs):
        """
        Since the algorithm is created before the runner it runs on is created, the interface to it is not always
        available during construction time. As the concrete structure of the task (e.g. observation spaces) may be
        necessary for constructing the data structures of the algorithm, this function is called to setup everything
        at a point in time where these quantities are known (e.g. a rollout worker has been able to be placed on a
        suitable device).
        """
        raise NotImplementedError()

    @abstractmethod
    def update_weights(self, weights):
        raise NotImplementedError()

    @abstractmethod
    def get_weights(self):
        raise NotImplementedError()

    @abstractmethod
    def start_epoch(self, driver: DriverControl):
        raise NotImplementedError()

    @abstractmethod
    def end_epoch(self, gradient_update_results, distributed_update_results, driver: DriverControl):
        raise NotImplementedError()

    @abstractmethod
    def distributed_update_step(self, arguments, worker_control: WorkerControl) -> Tuple[Any, Dict]:
        raise NotImplementedError()

    @abstractmethod
    def gradient_update(self, arguments, worker_control: WorkerControl) -> Tuple[Any, Dict]:
        raise NotImplementedError()

    @abstractmethod
    def get_checkpoint_state(self, worker_control: WorkerControl):
        raise NotImplementedError()

    @abstractmethod
    def set_checkpoint_state(self, state, worker_control: WorkerControl):
        raise NotImplementedError()

    @abstractmethod
    def checkpoint(self, path, collected_checkpoint_states, driver: DriverControl):
        raise NotImplementedError()

    @abstractmethod
    def restore_checkpoint(self, path, driver: DriverControl):
        raise NotImplementedError()

    @abstractmethod
    def before_initial_buffer_fill(self, driver: DriverControl):
        raise NotImplementedError()

    @abstractmethod
    def receive_broadcast(self, data):
        raise NotImplementedError()
