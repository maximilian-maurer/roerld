from typing import Callable

import numpy as np
import ray
import tensorflow as tf

from roerld.execution.control.worker_control import WorkerControl
from roerld.execution.distributed_update_step_algorithm import DistributedUpdateStepAlgorithm


@ray.remote
class AlgoExecutionActor:
    def __init__(self,
                 actor_setup_function,
                 algorithm_factory: Callable[[], DistributedUpdateStepAlgorithm],
                 worker_control: WorkerControl,
                 seed: int,
                 **kwargs):
        actor_setup_function()

        tf.random.set_seed(seed)
        np.random.seed(seed)

        if "tf_inter_op_parallelism_threads" in kwargs:
            tf.config.threading.set_inter_op_parallelism_threads(kwargs["tf_inter_op_parallelism_threads"])
        if "tf_intra_op_parallelism_threads" in kwargs:
            tf.config.threading.set_intra_op_parallelism_threads(kwargs["tf_intra_op_parallelism_threads"])

        for gpu in tf.config.experimental.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)

        self.worker_control = worker_control
        self.algorithm = algorithm_factory()
        self.algorithm.setup(self.worker_control, kwargs)

    def update_weights(self, weights):
        return self.algorithm.update_weights(weights)

    def update_step(self, args):
        return self.algorithm.distributed_update_step(arguments=args, worker_control=self.worker_control)

    def gradient_update_step(self, args):
        return self.algorithm.gradient_update(arguments=args, worker_control=self.worker_control)

    def get_checkpoint_state(self):
        return self.algorithm.get_checkpoint_state(worker_control=self.worker_control)

    def set_checkpoint_state(self, state):
        return self.algorithm.set_checkpoint_state(state=state, worker_control=self.worker_control)

    def receive_broadcast(self, data):
        return self.algorithm.receive_broadcast(data)
