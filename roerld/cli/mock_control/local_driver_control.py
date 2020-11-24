from roerld.execution.control.driver_control import DriverControl
from roerld.execution.control.worker_control import WorkerControl
from roerld.execution.distributed_update_step_algorithm import DistributedUpdateStepAlgorithm


class LocalDriverControl(DriverControl):
    def broadcast_to_gradient_workers(self, data):
        self.algorithm.receive_broadcast(data)

    def broadcast_to_distributed_update_workers(self, data):
        self.algorithm.receive_broadcast(data)

    def __init__(self, algorithm: DistributedUpdateStepAlgorithm, worker_control):
        self.algorithm = algorithm
        self._worker_control = worker_control

    def update_weights(self, weights):
        self.algorithm.update_weights(weights)

    def update_gradient_update_args(self, args):
        raise NotImplementedError()

    def update_distributed_update_args(self, args):
        raise NotImplementedError()

    def worker_control(self) -> WorkerControl:
        return self._worker_control

    def start_onpolicy_rollouts(self):
        raise NotImplementedError()

    def set_all_worker_checkpoint_states(self, state):
        # intentionally left empty
        pass
