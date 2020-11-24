from typing import Callable

import ray


@ray.remote
class CoordinatorActor:
    def __init__(self,
                 actor_setup_function: Callable[[], None]):
        actor_setup_function()
        self.shutdown = False
        self.epoch_index = -1

    def should_shutdown(self):
        return self.shutdown

    def set_should_shutdown(self):
        self.shutdown = True

    def set_epoch(self, epoch_index):
        self.epoch_index = epoch_index

    def epoch(self):
        return self.epoch_index
