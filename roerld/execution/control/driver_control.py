from abc import abstractmethod, ABC

from roerld.execution.control.worker_control import WorkerControl


class DriverControl(ABC):

    @abstractmethod
    def update_weights(self, weights):
        raise NotImplementedError()

    @abstractmethod
    def update_gradient_update_args(self, args):
        raise NotImplementedError()

    @abstractmethod
    def update_distributed_update_args(self, args):
        raise NotImplementedError()

    @abstractmethod
    def worker_control(self) -> WorkerControl:
        raise NotImplementedError()

    @abstractmethod
    def start_onpolicy_rollouts(self):
        raise NotImplementedError()

    @abstractmethod
    def set_all_worker_checkpoint_states(self, state):
        raise NotImplementedError()

    @abstractmethod
    def broadcast_to_gradient_workers(self, data):
        raise NotImplementedError()

    @abstractmethod
    def broadcast_to_distributed_update_workers(self, data):
        raise NotImplementedError()
