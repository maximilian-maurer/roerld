import copy
from typing import Callable

import ray


@ray.remote
class EpisodeWriterActor:
    def __init__(self,
                 directory,
                 io_factory,

                 actor_setup_function: Callable[[], None],
                 **kwargs):
        actor_setup_function()

        if "tf_inter_op_parallelism_threads" in kwargs or "tf_intra_op_parallelism_threads" in kwargs:
            import tensorflow as tf
            if "tf_inter_op_parallelism_threads" in kwargs:
                tf.config.threading.set_inter_op_parallelism_threads(kwargs["tf_inter_op_parallelism_threads"])
            if "tf_intra_op_parallelism_threads" in kwargs:
                tf.config.threading.set_intra_op_parallelism_threads(kwargs["tf_intra_op_parallelism_threads"])

        self.directory = directory
        self.data_handler = io_factory([directory])

        self.writer = self.data_handler.writer()

    def store_episode(self, episode, additional_metadata=None):
        episode_copy = copy.deepcopy(episode)
        del episode
        additional_metadata_copy = copy.deepcopy(additional_metadata)
        del additional_metadata
        self.writer.write_episode(episode_copy, additional_metadata_copy)
