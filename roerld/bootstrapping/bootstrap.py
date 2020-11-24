import ray
from datetime import datetime

import uuid

from typing import Callable

from roerld.bootstrapping.bootstrapping_actor import BootstrappingActor
from roerld.config.experiment_config import ExperimentConfigView
from roerld.config.registry import make_bootstrapping_actor


def bootstrap(experiment_config: ExperimentConfigView,
              environment_factory,
              writer,
              actor_setup_function: Callable[[], None],
              log_function: Callable[[str], None],
              num_jobs: int,
              num_samples: int,
              display_render_mode: str):
    assert num_jobs > 0
    assert num_samples > 0

    run_metadata = {
        "run_uuid": str(uuid.uuid4()),
        "date": str(datetime.now())
    }

    max_episode_length = experiment_config.key("bootstrapping.max_episode_length")

    rollout_config = {
        "evaluation": {
            "video_render_mode": "bgr_array",
            "video_width": 128,
            "video_height": 128
        }
    }

    if display_render_mode is not None:
        rollout_config["evaluation"]["video_render_mode"] = display_render_mode

    def learning_actor_factory(action_space):
        return make_bootstrapping_actor(experiment_config.key("bootstrapping.actor"), action_space)

    workers = [BootstrappingActor.remote(
        environment_factory=environment_factory,
        learning_actor_factory=learning_actor_factory,
        rollout_config=rollout_config,
        max_episode_length=max_episode_length,
        seed=experiment_config.key("general_config.seed"),
        actor_setup_function=actor_setup_function)
        for _ in range(num_jobs)]

    total_samples = 0
    samples_per_worker = max_episode_length
    with writer:
        writer.set_additional_metadata(run_metadata)

        worker_tasks = {}
        idle_workers = [w for w in workers]

        while total_samples < num_samples:
            for worker in idle_workers:
                future = worker.collect_samples.remote(num_samples=samples_per_worker)
                worker_tasks[future] = worker
            idle_workers = []

            finished_tasks, _ = ray.wait(list(worker_tasks.keys()), num_returns=1)
            finished_task = finished_tasks[0]
            episodes = ray.get(finished_task)
            for episode in episodes:
                total_samples += len(episode[list(episode.keys())[0]])
                writer.write_episode(episode)

            idle_workers.append(worker_tasks[finished_task])
            del worker_tasks[finished_task]
            del finished_task

            log_function(f"Samples collected: {total_samples}")