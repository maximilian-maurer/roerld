import time
from typing import Callable

import ray


@ray.remote
class LogReplayActor:
    def __init__(self,
                 data_folder,
                 target_preprocessors,
                 coordinator_actor,
                 drop_keys,
                 actor_setup_function: Callable[[], None],
                 io_factory,
                 max_new_samples_per_epoch=1e15,
                 **kwargs):
        actor_setup_function()

        if "tf_inter_op_parallelism_threads" in kwargs or "tf_intra_op_parallelism_threads" in kwargs:
            import tensorflow as tf
            if "tf_inter_op_parallelism_threads" in kwargs:
                tf.config.threading.set_inter_op_parallelism_threads(kwargs["tf_inter_op_parallelism_threads"])
            if "tf_intra_op_parallelism_threads" in kwargs:
                tf.config.threading.set_intra_op_parallelism_threads(kwargs["tf_intra_op_parallelism_threads"])

        self.max_new_samples_per_epoch = max_new_samples_per_epoch
        self.target_preprocessors = target_preprocessors
        self.coordinator = coordinator_actor
        self.data_path = data_folder
        self.max_episodes_queued = 5
        self.drop_keys = drop_keys
        self.check_done_interval = min(self.max_episodes_queued, 50)
        self.io_factory = io_factory

    def run(self):
        source = self.io_factory([self.data_path])

        reader = source.reader(infinite_repeat=True)
        enqueued_futures = []

        read_this_epoch = 0
        epoch = ray.get(self.coordinator.epoch.remote())

        with reader:
            episode = reader.next_episode()
            episodes_since_last_check = 0
            while episode is not None:
                if episodes_since_last_check > self.check_done_interval:
                    if ray.get(self.coordinator.should_shutdown.remote()):
                        break
                    epoch = ray.get(self.coordinator.epoch.remote())
                    episodes_since_last_check = 0

                for k in self.drop_keys:
                    if k in episode:
                        del episode[k]

                for processor, returns_future in self.target_preprocessors:
                    waitable_future = processor.process_episode.remote(episode, None)
                    if returns_future:
                        waitable_future = ray.get(waitable_future)
                    enqueued_futures.append(waitable_future)

                episode = reader.next_episode()

                # don't overload the buffer. Particularly since the queue takes up space in the shared object store.
                if len(enqueued_futures) > self.max_episodes_queued:
                    _, enqueued_futures = ray.wait(enqueued_futures,
                                                   num_returns=max(0, len(enqueued_futures) - self.max_episodes_queued))
                episodes_since_last_check = 1
                read_this_epoch += len(episode[list(episode.keys())[0]])

                if epoch < 0:
                    epoch = ray.get(self.coordinator.epoch.remote())

                # todo: this shouldnt be hardcoded
                if epoch >= 0 and read_this_epoch > self.max_new_samples_per_epoch:
                    # wait until the next epoch starts
                    # print("Log replay job waiting on new epoch")
                    current_epoch = epoch
                    while current_epoch == epoch:
                        current_epoch = ray.get(self.coordinator.epoch.remote())
                        time.sleep(1)
                    epoch = current_epoch
                    read_this_epoch = 0
