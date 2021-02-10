import datetime
import os
import re
import time
from enum import Enum
from pathlib import Path
from typing import Dict, Callable, List, Tuple, Iterable

import gym
import numpy as np
import ray
import tensorflow as tf

from roerld.config.experiment_config import ExperimentConfigError, ExperimentConfig
from roerld.data_handling.data_source import DataSource
from roerld.execution.actors.algo_execution_actor import AlgoExecutionActor
from roerld.execution.actors.coordinator_actor import CoordinatorActor
from roerld.execution.actors.distributed_preprocessing_actor import DistributedPreprocessingActor
from roerld.execution.actors.episode_writer_actor import EpisodeWriterActor
from roerld.execution.actors.log_replay_actor import LogReplayActor
from roerld.execution.actors.preprocessing_actor import PreprocessingActor
from roerld.execution.actors.rollout_actor import RolloutActor
from roerld.execution.actors.single_instance_replay_buffer_actor import SingleInstanceReplayBufferActor
from roerld.execution.actors.video_writer_actor import VideoWriterActor
from roerld.execution.control.driver_control import DriverControl
from roerld.execution.control.ray_remote_replay_buffer import RayRemoteReplayBuffer
from roerld.execution.control.worker_control import WorkerControl
from roerld.execution.distributed_update_step_algorithm import DistributedUpdateStepAlgorithm
from roerld.execution.rollouts.static_rollout_manager import StaticRolloutManager
from roerld.execution.runners.distributed_update_step.worker_control import DistributedUpdateStepPipelineWorkerControl
from roerld.execution.transition_format import TransitionFormat
from roerld.execution.utils.diagnostics import aggregate_diagnostics
from roerld.execution.utils.experience import rollout_experience_into_episodes
from roerld.execution.utils.result_reorder_buffer import ResultReorderBuffer
from roerld.execution.utils.timings import TimingHelper
from roerld.execution.utils.waiting import wait_all
from roerld.learning_actors.random.random_from_action_space import RandomFromActionSpaceLearningActor
from roerld.learning_actors.wrappers.algo_actor_wrapper import AlgoActorWrapper
from roerld.learning_actors.wrappers.epsilon_greedy import EpsilonGreedyLearningActor
from roerld.replay_buffers.replay_buffer import ReplayBuffer


class DataSources(Enum):
    Rollouts = "rollouts"
    OnpolicyExperience = "onpolicy_experience"
    TrainingRollout = "training_rollout"
    EvaluationRollout = "evaluation_rollout"
    LogReplay = "log_replay"
    EpisodeWriter = "episode_writer"
    RestoredOnpolicyReplay = "restored_onpolicy_replay"


class DistributedUpdateStepPipelineRunner(DriverControl):
    def __init__(self,
                 epochs: int,
                 environment_factory: Callable[[], gym.Env],
                 algorithm_factory: Callable[[], DistributedUpdateStepAlgorithm],
                 actor_setup_function: Callable[[], None],

                 buffer_factories: Dict[str, List[Callable[[TransitionFormat], ReplayBuffer]]],
                 buffer_worker_configurations: Dict[str, List[Tuple[Dict, Dict]]],

                 io_factory: Callable[[Iterable[str]], DataSource],

                 bellman_update_worker_configurations: List[Tuple[Dict, Dict]],
                 gradient_update_worker_configurations: List[Tuple[Dict, Dict]],
                 video_writer_worker_configurations: List[Tuple[Dict, Dict]],
                 rollout_worker_configurations: List[Tuple[Dict, Dict]],
                 log_replay_worker_configurations: List[Tuple[Dict, Dict]],
                 coordinator_worker_configuration: Tuple[Dict, Dict],
                 episode_writer_worker_configurations: Tuple[Dict, Dict],

                 log_path: str,
                 store_onpolicy_experience: bool,

                 write_checkpoint_callback: Callable[[str], None],

                 max_episode_length: int,
                 checkpoint_interval: int,
                 min_bellman_update_batches_per_epoch: int,
                 max_bellman_update_batches_per_epoch: int,
                 seed: int,
                 num_eval_episodes_per_epoch: int,
                 save_eval_videos: bool,
                 save_training_videos: bool,
                 eval_video_save_interval: int,
                 eval_interval: int,
                 experiment_tag: str,

                 preprocessing_workers,
                 preprocessing_chains,

                 experiment_config: Dict,

                 drop_keys: List[str] = None,
                 restore_from_checkpoint_path=None,
                 omit_timing_data=False):
        """
        Issues pending refactoring:
            * There is a muddling of responsibility for the initial setup of the buffers between the algorithm
                and this class. Currently, this assumes there to be a structure of named replay buffers
                "train", "offline" and "online" such that the distributed update step will fill the
                "train" buffer from the others.

              This class will during setup run the distributed update step until there is an initial fill of the
              "train" buffer.
            * This currently still takes the entire experiment config for legacy reasons pending the completion of
                separating the configuration from this class. This is currently missing the following components:
                    * Interfaces for Data IO. In particular, the current episode writers are a suboptimal
                        mis-implementation of the scope patterns, and need to be reworked at some point to conform to a
                        better interface (ideally in conjunction with the implementation of better IO).
                    * Interfaces for the Video Writers.
                    * Interfaces for the Rollout Management, pending a larger pass on the rollout architecture in
                        general.
                To that end, the configuration is currently still taken as a parameter, but is only heeded for the
                    following entities (everything else is taken from the parameters from this class, and the
                    content of the experiment configuration ignored):
                    * the "rollout_manager" section
                    * the "exploration_section" (pending rework of how the actor used for exploration is created)
                    * the "episode_writer" section
            * In order to properly restore the onpolicy buffer from checkpoints where there is a log of stored
                experience available, this should on restoring replay the onpolicy experience linearly so that the
                buffer is in the state it would have had at saving. Otherwise, restoring will have adverse effects on
                the further progress on training until the buffer is filled up again. This behavior is currently still
                missing.
            * The gradient worker currently performs the entire training process on its end instead of using a
                parameter server architecture as intended. This effectively blocks gpus on other devices being used,
                and so should be reworked.

        Worker configuration:
            All the `worker_configurations` arguments below take a tuple of (ray_kwargs, worker_kwargs) or a list of
             such tuples which describe the keyword arguments to be passed to the ray actor instantiation and the
             actor class itself. There will be as many actors instantiated as there are elements in the list (or one
             if the argument is type-hinted as only a tuple).

        Args:
            epochs: The number of epochs to run training for.
            environment_factory: A callable that will instantiate a gym environment when called.
            algorithm_factory: A callable that will instantiate the algorithm when called.
            actor_setup_function: This callable is called once on every remote actor when it is set up. If any of the
                                    other callbacks and classes used rely on global state in the local python
                                    interpreter process (e.g. instantiating an environment that was added to the gym
                                    registry and is thus only registered in the local python process and not in any
                                    of the remote processes), then this function must establish a compatible global
                                    state in the remote process.
            buffer_factories: A key-value mapping of factories, such that every key, the name of a replay buffer, maps
                                    to one or more factories that will instantiate this buffer.
            buffer_worker_configurations: See worker configuration section above.
            bellman_update_worker_configurations: See worker configuration section above.
            gradient_update_worker_configurations: See worker configuration section above.
            video_writer_worker_configurations: See worker configuration section above.
            rollout_worker_configurations: See worker configuration section above.
            log_replay_worker_configurations: See worker configuration section above.
            coordinator_worker_configuration: See worker configuration section above.
            episode_writer_worker_configurations: See worker configuration section above.
            log_path: The base-directory in which to save the logs.
            store_onpolicy_experience: Whether to save transitions from on-policy experience during training and
                                            evaluation to disk.
            write_checkpoint_callback: A callback that is called when a checkpoint is created. It is given the
                                        directory path of the checkpoint folder.
            max_episode_length: The maximum length of an episode. If an episode has not returned done by that number of
                                steps it is not stepped any further.
            checkpoint_interval: The interval at which checkpoints are created.
            min_bellman_update_batches_per_epoch: The minimum number of distributed update steps to perform per
                                                    epoch.
            seed: The seed.
            num_eval_episodes_per_epoch: The number of evaluation rollouts for an evaluation.
            save_eval_videos: Whether to save videos of the evaluation rollouts to disk. See also
                                    `eval_video_save_interval`
            save_training_videos: Whether to save training videos to disk.
            eval_video_save_interval: If `save_eval_videos` is set to true, these videos are saved every n evaluations,
                                        where n is this parameter.
            eval_interval: Evaluate every this-parameter epochs.
            experiment_tag: The experiment tag to use.
            experiment_config: The experiment config, see note on this above.
            drop_keys: Elements of the observation space with these names are immediately dropped and not stored,
                        saved or otherwise retained.
            restore_from_checkpoint_path: If the state of this pipeline is to be restored from a checkpoint,
            omit_timing_data: Whether to save timing data in the tensorboard output. While this data is useful for
                                profiling, it will significantly bloat up the tensorboard output and cause lags, so
                                this flag can be used to disable it.
        """
        self.epochs = epochs
        self.algorithm_factory = algorithm_factory
        self.restore_from_checkpoint_path = restore_from_checkpoint_path
        self.actor_setup_function = actor_setup_function
        self.environment_factory = environment_factory
        self.buffer_factories = buffer_factories
        self.buffer_worker_configurations = buffer_worker_configurations
        self.bellman_update_worker_configurations = bellman_update_worker_configurations
        self.gradient_update_worker_configurations = gradient_update_worker_configurations
        self.video_writer_worker_configurations = video_writer_worker_configurations
        self.rollout_worker_configurations = rollout_worker_configurations
        self.log_replay_worker_configurations = log_replay_worker_configurations
        self.store_onpolicy_experience = store_onpolicy_experience
        self.write_checkpoint_callback = write_checkpoint_callback
        self.coordinator_worker_configuration = coordinator_worker_configuration
        self.max_episode_length = max_episode_length
        self.episode_writer_worker_configurations = episode_writer_worker_configurations
        self.io_factory = io_factory
        self.preprocessing_workers = preprocessing_workers
        self.preprocessing_chains = preprocessing_chains

        self.checkpoint_interval = checkpoint_interval
        self.min_bellman_update_batches_per_epoch = min_bellman_update_batches_per_epoch
        self.max_bellman_update_batches_per_epoch = max_bellman_update_batches_per_epoch if max_bellman_update_batches_per_epoch is not None else np.inf
        self.seed = seed
        self.num_eval_episodes_per_epoch = num_eval_episodes_per_epoch
        self.save_eval_videos = save_eval_videos
        self.save_training_videos = save_training_videos
        self.eval_video_save_interval = eval_video_save_interval
        self.eval_interval = eval_interval
        self.experiment_tag = experiment_tag

        self.experiment_config = ExperimentConfig.view(experiment_config)

        assert self.max_episode_length > 0
        assert checkpoint_interval > 0
        assert min_bellman_update_batches_per_epoch >= 0
        assert num_eval_episodes_per_epoch > 0
        assert eval_interval > 0
        assert self.max_bellman_update_batches_per_epoch >= self.min_bellman_update_batches_per_epoch

        # keys set via the configuration
        self.algorithm = None
        self.drop_keys = []
        if drop_keys is not None:
            # todo this presupposes that this is what the transition format will do, and should be integrated there
            #  once the preprocessing pipeline is implemented.
            self.drop_keys.extend(["observations_" + k for k in drop_keys])
            self.drop_keys.extend(["next_observations_" + k for k in drop_keys])
            self.drop_keys.extend([k for k in drop_keys])

        if self.seed is not None:
            tf.random.set_seed(self.seed)
            np.random.seed(self.seed)

        # For all other processes, ray will manage the gpu allocation. Since this is the driver process
        #  which may not use the GPU, forbid it manually here.
        tf.config.experimental.set_visible_devices([], 'GPU')

        base_path = log_path
        self.output_directory_name = f"{base_path}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}" \
                                     f"_{self.experiment_tag}_s{str(self.seed)}"
        self.output_path = os.path.join(self.output_directory_name)
        self.experience_directory = os.path.join(self.output_path, "experience")

        # variables which will be lazy initialized because they require parts of the pipeline to be started already
        self.transition_format = None  # type: TransitionFormat

        self.eval_buffer = ResultReorderBuffer()
        self.tb_writer = None
        self.rollout_diagnostics = []
        self.distributed_update_args = None
        self.gradient_update_args = None
        self.episode_writers = None
        self.first_epoch_with_onpolicy_data = False
        self.omit_timing_data = omit_timing_data

        self.weights = None
        self.replay_buffers = {}  # type: Dict[str, List]

    def _setup(self):
        self._ensure_output_folders_exist()
        self.tb_writer = tf.summary.create_file_writer(logdir=self.output_path)

        if os.path.islink("latest_run"):
            os.unlink("latest_run")
        os.symlink(self.output_path, "latest_run", target_is_directory=True)

        # Save config to the experiment folder

        self.coordinator = CoordinatorActor \
            .options(**self.coordinator_worker_configuration[0]) \
            .remote(actor_setup_function=self.actor_setup_function)

        # At this point it is necessary to know the properties of the environment (mainly the structure of the
        # spaces). To that end there needs to be an instance of the environment. As the pipeline runner may run on a
        # different device, the rollout actors are the only actors which have placement guaranteed somewhere where
        # the environments can be constructed (e.g. all devices they need are available and connected), hence they
        # need to be constructed here before anything else is done
        self._setup_rollout_actors(coordinator_actor=self.coordinator)
        first_rollout_actor = self.rollout_actors[0]
        observation_space = ray.get(first_rollout_actor.env_observation_space.remote())
        action_space = ray.get(first_rollout_actor.env_action_space.remote())
        self.observation_space, self.action_space = observation_space, action_space

        self.transition_format = TransitionFormat.from_gym_spaces(observation_space, action_space)

        # todo the drop keys mechanism is currently disabled pending the rework of the input pipeline. Instead of the
        # current implementation it should be transitioned into a normal preprocessing module.
        self._setup_replay_buffer_actors()

        # The rollout manager is the only class which interacts with the environment directly, as such it needs
        #  to have buffers for all the data (the original input specification) rather than only the data retained
        #  for processing here.
        wait_all([
            rollout_actor.initialize.remote(
                self._make_worker_control("rollout" + str(idx)))
            for idx, rollout_actor in enumerate(self.rollout_actors)
        ])

        assert self.experiment_config.key("rollout_manager.name") == "static_rollouts"
        self.rollout_manager = StaticRolloutManager(
            runner=self,
            workers=self.rollout_actors,
            min_training_rollouts_per_epoch=self.experiment_config.key(
                "rollout_manager.min_training_rollouts_per_epoch"),
            max_training_rollouts_per_epoch=self.experiment_config.key(
                "rollout_manager.max_training_rollouts_per_epoch"),
            render_every_n_training_rollouts=self.experiment_config.key(
                "rollout_manager.render_every_n_training_rollouts"
            )
        )

        #
        if self.store_onpolicy_experience:
            self._setup_episode_writer_actors()
        else:
            self.episode_writer = None

        self._setup_preprocessing_actors()

        # setup the driver process version of the algorithm
        self.local_worker_control = self._make_worker_control("local_driver")
        self.algorithm = self.algorithm_factory()

        # setup the remaining actors
        self._setup_bellman_update_actors(self.action_space.low, self.action_space.high)
        self._setup_video_writer_actors()
        self._setup_gradient_actors()
        self._setup_log_replay_actors(self.coordinator)

        self.algorithm.setup(worker_control=self.worker_control(), worker_specific_kwargs={})

    def _initial_data_processing(self):
        """
        In order to get samples for training the training buffer is queued. This buffer is initially empty, as no
        data has yet been read or processed. This function arranges-for and waits-on data having gone through the
        pipeline and end up in the training buffer, so that it is possible to start training from that buffer.
        """
        if len(self.log_replay_actors) == 0:
            max_buffer_size = ray.get(self.online_buffer_actor.max_size.remote())
            current_size = ray.get(self.online_buffer_actor.sample_count.remote())

            if float(current_size) / max_buffer_size < 0.1:
                # there are no offline replay actors, which means everything is done on-policy.
                print("Performing initial onpolicy buffer fill.")
                self.rollout_manager.start_of_epoch(0, self.weights)

                for _ in range(len(self.rollout_actors) * 100):
                    self.rollout_manager.manually_schedule_training_rollout(fully_random=True)
                pending_rollout_futures = self.rollout_manager.pending_futures()
                wait_all(pending_rollout_futures)
                for idx, f in enumerate(pending_rollout_futures):
                    print(f"Initial Rollout {idx}")
                    self.rollout_manager.future_arrived(f)
            else:
                print("Skipping initial onpolicy buffer fill.")

        if len(self.log_replay_actors) > 0:
            print("Populating offline replay buffer. Waiting until the buffer is at least 50% full.")

            max_buffer_size = ray.get(self.offline_buffer_actor.max_size.remote())

            # wait for the log replay jobs to start
            while ray.get(self.offline_buffer_actor.sample_count.remote()) < 0.5 * max_buffer_size:
                time.sleep(0.5)
                samples = ray.get(self.offline_buffer_actor.sample_count.remote())
                print(f"Offline buffer has {samples} samples in it ({round((samples / max_buffer_size) * 100, 2)}%)")

        print("Populating training buffer, waiting for initial batches to be processed")
        for i in range(50):
            print(f"Initial Buffer Fill #{i}")
            wait_all([worker.update_step.remote(self.distributed_update_args) for worker in self.bellman_update_actors])

    def run(self):
        self._setup()
        gradient_actor = self.gradient_actors[0]

        #ray_errors = ray.errors()
        #if len(ray_errors) > 0:
        #    print(f"Encountered errors, shutting down: {ray_errors}")
        #    return False

        for log_replay_actor in self.log_replay_actors:
            log_replay_actor.run.remote()

        if self.restore_from_checkpoint_path is not None:
            self.algorithm.restore_checkpoint(self.restore_from_checkpoint_path, self)

            parent_folder = Path(self.restore_from_checkpoint_path).parent
            experience_folder = parent_folder / "experience"
            if os.path.exists(experience_folder):
                print(f"Found stored experience in {str(experience_folder)}. Populating online buffer.")
                io = self.io_factory([experience_folder])
                reader = io.reader(False)
                with reader:
                    next_episode, next_metadata = reader.next_episode_with_metadata()

                    # don't overload the preprocessing pipeline and the object store buffers
                    # todo: this is a bad heuristic. this should resolve the number of workers we have for the
                    #           preprocessing of this specific type of experience
                    max_queued = len(self.preprocessing_actors)

                    pending_futures = []
                    loaded_episodes = 0
                    skipped = 0
                    while next_episode is not None:
                        if next_metadata["type"] != "onpolicy_training":
                            skipped += 1
                            next_episode, next_metadata = reader.next_episode_with_metadata()
                            continue

                        next_metadata["restore_flag"] = True
                        this_futures = self._propagate_episode_waitable(DataSources.RestoredOnpolicyReplay,
                                                                        next_episode, next_metadata)
                        pending_futures.extend(this_futures)

                        loaded_episodes += 1

                        if len(pending_futures) > max_queued:
                            _, pending_futures = ray.wait(pending_futures,
                                                          num_returns=len(pending_futures) - max_queued)

                        episode_data = reader.next_episode_with_metadata()
                        if episode_data is None:
                            break
                        next_episode, next_metadata = episode_data

                print(f"Loaded {loaded_episodes} episodes, skipped {skipped} because they were not training rollouts.")

        self.algorithm.before_initial_buffer_fill(self)
        self._initial_data_processing()

        # when there are exceptions during the above procedure, the errors will not be printed immediately in spite
        # of the wait, instead the training buffer being queued below will throw an error (when np rand is called
        # requesting 0 samples). In case that happens, here is a good breakpoint where the driver process can be
        # paused while the other processes run so as to wait the necessary time for the error messages to come in.
        print("Starting training.")

        try:
            epoch_index = 0
            task_assignments = {}

            while epoch_index <= self.epochs:
                # print("Epoch ", epoch_index)
                timer = TimingHelper("Time Driver Epoch Startup")

                self.coordinator.set_epoch.remote(epoch_index)
                self.algorithm.start_epoch(self)
                self.rollout_manager.start_of_epoch(epoch_index, self.weights)

                timer.time_stamp("Time Driver Queue Update")
                gradient_update_results = []
                distributed_update_results = []
                self.rollout_diagnostics = []

                # if this is the first epoch that has onpolicy rollouts (and in consequence, the algorithm an
                # expectation that the online replay buffer is not completely empty), wait for those manually
                if self.first_epoch_with_onpolicy_data:
                    print("First Epoch with On-Policy data collection.")
                    # make sure at least a single training rollout is scheduled so as to have one to wait on
                    self.rollout_manager.manually_schedule_training_rollout()
                    pending_rollout_futures = self.rollout_manager.pending_futures()
                    wait_all(pending_rollout_futures)
                    for f in pending_rollout_futures:
                        self.rollout_manager.future_arrived(f)

                    print("Waiting on online data to propagate")
                    while ray.get(self.online_buffer_actor.sample_count.remote()) == 0:
                        time.sleep(1)
                    self.first_epoch_with_onpolicy_data = False
                    print("Continuing training.")

                # Start the gradient update, and while that is running, process everything else
                gradient_future = gradient_actor.gradient_update_step.remote(self.gradient_update_args)
                task_assignments[gradient_future] = gradient_actor

                timer.time_stamp("Time Driver Idle Worker Assignment")

                idle_bellman_update_actors = [w for w in self.bellman_update_actors if
                                              w not in task_assignments.values()]
                for worker in idle_bellman_update_actors:
                    future = worker.update_step.remote(self.distributed_update_args)
                    task_assignments[future] = worker

                timer.time_stamp("Time Driver Epoch Loop")

                this_epoch_bellman_updates = 0
                bellman_updater_diagnostics = []
                gradient_done = False
                gradient_actor_diagnostics = None
                had_rollout_stall = False
                had_bellman_updater_stall = False
                rollout_stall_start = -1
                bellman_updater_stall_start = -1

                while not (gradient_done and this_epoch_bellman_updates >= self.min_bellman_update_batches_per_epoch
                           and not self.rollout_manager.needs_stall()):
                    # If the gradient is done then we are waiting on something else. Determine which it is.
                    if gradient_done:
                        if self.rollout_manager.needs_stall():
                            if had_rollout_stall is False:
                                rollout_stall_start = time.perf_counter()
                            had_rollout_stall = True
                        if this_epoch_bellman_updates < self.min_bellman_update_batches_per_epoch:
                            if had_bellman_updater_stall is False:
                                bellman_updater_stall_start = time.perf_counter()
                            had_bellman_updater_stall = True

                    pending_tasks_in_this_class = list(task_assignments.keys())
                    pending_tasks_in_rollout_manager = self.rollout_manager.pending_futures()

                    ready_futures, _ = ray.wait(pending_tasks_in_this_class + pending_tasks_in_rollout_manager,
                                                num_returns=1)
                    assert len(ready_futures) == 1
                    ready_future = ready_futures[0]

                    if ready_future in pending_tasks_in_rollout_manager:
                        self.rollout_manager.future_arrived(ready_future)
                    elif task_assignments[ready_future] == gradient_actor:
                        # this is a gradient update
                        assert ready_future == gradient_future
                        gradient_done = True
                        results, gradient_actor_diagnostics = ray.get(gradient_future)
                        gradient_update_results.append(results)
                        del task_assignments[gradient_future]
                    else:
                        # it is a bellman updater, give it a new task
                        bellman_updater = task_assignments[ready_future]
                        results, diagnostics = ray.get(ready_future)
                        distributed_update_results.append(results)
                        this_epoch_bellman_updates += 1
                        del task_assignments[ready_future]

                        if this_epoch_bellman_updates < self.max_bellman_update_batches_per_epoch:
                            # todo: this over-schedules by the worker count
                            new_future = bellman_updater.update_step.remote(self.distributed_update_args)
                            task_assignments[new_future] = bellman_updater
                            bellman_updater_diagnostics.append(diagnostics)

                time_bellman_updater_stall = time.perf_counter() - bellman_updater_stall_start \
                    if had_bellman_updater_stall else 0
                time_rollout_stall = time.perf_counter() - rollout_stall_start if had_rollout_stall else 0

                algo_info = self.algorithm.end_epoch(gradient_update_results, distributed_update_results, self)

                timer.time_stamp()

                diagnostics_data = [*self.rollout_diagnostics, *bellman_updater_diagnostics, gradient_actor_diagnostics]

                info = {
                    **algo_info,
                    "On-Policy Rollouts": self.rollout_manager.training_rollouts_started_this_epoch,
                    "Bellman Updater Batches": this_epoch_bellman_updates,

                    # todo move this over to the algorithm info
                    # "Bellman Updater Samples Updated": this_epoch_bellman_updates * self.bellman_update_worker_batch_size,
                    "Pipeline Stall Waiting on Rollouts": time_rollout_stall,
                    "Pipeline Stall Waiting on Bellman Updates": time_bellman_updater_stall,
                    **timer.result(),
                    **aggregate_diagnostics(diagnostics_data),
                    "epoch": epoch_index
                }

                if self.omit_timing_data:
                    info = {
                        k: v for k, v in info.items() if "time" not in k.lower()
                    }

                with self.tb_writer.as_default():
                    for key in info:
                        tf.summary.scalar(key, info[key], step=epoch_index)

                render_eval_videos_this_epoch = self.save_eval_videos and (
                        epoch_index % self.eval_video_save_interval) == 0

                if (epoch_index % self.eval_interval) == 0:
                    eval_futures = self.rollout_manager.schedule_evaluation_rollout(
                        self.num_eval_episodes_per_epoch,
                        self.weights,
                        {"epoch": epoch_index},
                        True,
                        render_eval_videos_this_epoch)
                    self.eval_buffer.associate_futures_with_epoch(eval_futures, epoch_index)

                if epoch_index != 0 and epoch_index % self.checkpoint_interval == 0:
                    self._checkpoint(self.output_path, epoch_index)

                epoch_index += 1

                if epoch_index % 100 == 0:
                    print(f"Epoch {epoch_index} done.")

            remaining_eval_futures = self.eval_buffer.pending_futures()
            ray.wait(remaining_eval_futures, num_returns=len(remaining_eval_futures))
            for f in remaining_eval_futures:
                self._handle_eval_future(self.tb_writer, f)

        except Exception as e:
            raise e
        finally:
            ray.get(self.coordinator.set_should_shutdown.remote())

            self.eval_buffer.clear()
            self.rollout_manager.close()

    def _checkpoint(self, path, epoch):
        checkpoint_dir = os.path.join(path, f"checkpoint_{epoch}")
        assert not os.path.exists(checkpoint_dir)
        os.mkdir(checkpoint_dir)

        collected_data = []
        for actor in self.gradient_actors:
            collected_data.append(ray.get(actor.get_checkpoint_state.remote()))
        self.algorithm.checkpoint(checkpoint_dir, collected_data, self)

        self.write_checkpoint_callback(checkpoint_dir)

    def _make_worker_control(self, name):
        # todo this doesn't deal correctly with having more than one replay buffer actor as part of the muddled
        #       responsibility between pipeline and algorithm as to preprocessing
        # todo this can only handle a single replay buffer worker to a single replay buffer
        remote_replay_buffers = {
            k: RayRemoteReplayBuffer(v[0]) for k, v in self.replay_buffers.items()
        }

        worker_control = DistributedUpdateStepPipelineWorkerControl(input_spec=self.transition_format,
                                                                    replay_buffers=remote_replay_buffers,
                                                                    observation_space=self.observation_space,
                                                                    action_space=self.action_space,
                                                                    coordinator_actor=self.coordinator,
                                                                    name=self.output_directory_name + "_" + name)
        return worker_control

    def _propagate_episode(self, source: DataSources, episode, episode_metadata):
        true_sources = [source]
        for source in true_sources:
            source_as_str = source.value
            for accepted_sources, actor, _ in self.preprocessing_actors:
                if source_as_str not in accepted_sources:
                    continue
                actor.process_episode.remote(episode, episode_metadata)

    def _propagate_episode_waitable(self, source: DataSources, episode, episode_metadata):
        futures = []
        true_sources = [source]
        for source in true_sources:
            source_as_str = source.value
            for accepted_sources, actor, is_distributed in self.preprocessing_actors:
                if source_as_str not in accepted_sources:
                    continue
                
                waitable_future = actor.process_episode_waitable.remote(episode, episode_metadata)
                if is_distributed:
                    waitable_future = ray.get(waitable_future)
                                                            
                futures.append(waitable_future)
        return futures

    def receive_evaluation_rollout(self, evaluation_rollout_future):
        self.rollout_diagnostics.extend(self._handle_eval_future(self.tb_writer, evaluation_rollout_future))

    def receive_training_rollout(self, training_rollout_future):
        # store onpolicy experience and sample new one
        onpolicy_experience, videos, episode_starts, diagnostics, info = ray.get(training_rollout_future)

        # print(f"Processed training rollout for epoch {info['epoch']}")

        self.rollout_diagnostics.append(diagnostics)

        into_episodes = rollout_experience_into_episodes(onpolicy_experience, episode_starts)
        for episode in into_episodes:
            self._propagate_episode(DataSources.TrainingRollout, episode,
                                    {"type": "onpolicy_training",
                                     "from_epoch": info["epoch"]})
        # store_future = self.online_buffer_actor.store_batch.remote(**experience_without_drop_keys)
        if videos is not None and self.save_training_videos:
            for i in range(len(videos)):
                self.evaluation_video_writer.write_video.remote(
                    path=os.path.join(self.output_path,
                                      "videos",
                                      f"episode_{info['epoch']}_training_r{i}.avi"),
                    frames=videos[i]
                )

    def _handle_eval_future(self, tb_writer, eval_future):
        self.eval_buffer.receive_future(eval_future)

        diagnostic_returns = []
        while self.eval_buffer.has_result():
            eval_experience, videos, episode_starts, diagnostics, info = self.eval_buffer.pop_result()
            diagnostic_returns.append(diagnostics)

            as_episodes = rollout_experience_into_episodes(eval_experience, episode_starts)
            additional_info = {
                "Episode Return (From Environment Reward)": [np.sum(e["rewards"]) for e in as_episodes],
                "Episode Length": [len(e["actions"]) for e in as_episodes]
            }

            if len(as_episodes) > 0 and len(as_episodes[0]["infos"]) > 0 and "is_success" in as_episodes[0]["infos"][0]:
                success_rate = np.sum([np.max([e["infos"][i]["is_success"] for i in range(len(e["infos"]))])
                                       for e in as_episodes]) / len(as_episodes)
                additional_info["Success Rate"] = success_rate

            info.update(aggregate_diagnostics([additional_info]))

            print(
                f"Epoch {info['epoch']} mean return was {additional_info['Episode Return (From Environment Reward)']}")

            with tb_writer.as_default():
                for key in info:
                    tf.summary.scalar(key, info[key], step=info["epoch"])

            # send off video
            if videos is not None and self.save_eval_videos:
                for i in range(len(videos)):
                    self.evaluation_video_writer.write_video.remote(
                        path=os.path.join(self.output_path,
                                          "videos",
                                          f"episode_{info['epoch']}_r{i}.avi"),
                        frames=videos[i]
                    )

            into_episodes = rollout_experience_into_episodes(eval_experience, episode_starts)
            for episode in into_episodes:
                self._propagate_episode(DataSources.EvaluationRollout, episode,
                                        {"type": "onpolicy_evaluation",
                                         "from_epoch": info["epoch"]})

            print(f"Processed evaluation rollout for epoch {info['epoch']}")

        tf.summary.flush()
        return diagnostic_returns

    def _ensure_output_folders_exist(self):
        os.makedirs("logs", exist_ok=True)

        assert not os.path.exists(self.output_path)
        os.makedirs(self.output_path, exist_ok=True)

        if self.save_eval_videos:
            os.makedirs(os.path.join(self.output_path, "videos"), exist_ok=True)

        assert not os.path.exists(self.experience_directory)
        os.makedirs(self.experience_directory, exist_ok=True)

    # region Actor Setup

    def _setup_rollout_actors(self, coordinator_actor):
        rollout_actor_configs = self.rollout_worker_configurations
        assert len(rollout_actor_configs) > 0

        algorithm_factory = self.algorithm_factory
        epsilon = self.experiment_config.key("exploration.epsilon")
        scale = self.experiment_config.key("exploration.scale")

        def _learning_actor_factory(action_space):
            algo = algorithm_factory()
            return algo, EpsilonGreedyLearningActor(
                policy_actor=AlgoActorWrapper(algo),
                random_actor=RandomFromActionSpaceLearningActor(action_space=action_space, scaling=scale),
                epsilon=epsilon
            )

        print(f"Starting {len(rollout_actor_configs)} rollout actors.")
        # @formatter:off
        self.rollout_actors = [
            RolloutActor
                .options(**ray_kwargs)
                .remote(environment_factory=self.environment_factory,
                        algorithm_factory=_learning_actor_factory,
                        max_episode_length=self.max_episode_length,
                        rollout_config=self.experiment_config.section("rollouts"),
                        actor_setup_function=self.actor_setup_function,
                        seed=self.seed,
                        coordinator_actor=coordinator_actor,
                        **worker_kwargs)
            for ray_kwargs, worker_kwargs in rollout_actor_configs
        ]
        # @formatter:on

    def _setup_replay_buffer_actors(self):
        buffer_actors = {}
        for buffer_key in self.buffer_worker_configurations:
            assert buffer_key in self.buffer_factories
            buffer_factories = self.buffer_factories[buffer_key]

            worker_configs = self.buffer_worker_configurations[buffer_key]
            assert len(worker_configs) > 0
            assert len(buffer_factories) > 0
            if len(worker_configs) > 1:
                raise NotImplementedError("The current implementation does not support using multiple actors for a "
                                          "single replay buffer transparently.")
            if len(buffer_factories) > 1:
                raise NotImplementedError("The current implementation does not support using multiple actors for a "
                                          "single replay buffer transparently.")

            # With pycharm 2020.1 there are oscillations in the auto-formatter that cause this to be re-formatted
            # continually.
            # @formatter:off
            buffer_actors[buffer_key] = [
                SingleInstanceReplayBufferActor
                    .options(**ray_kwargs)
                    .remote(buffer_factories=buffer_factories,
                            transition_format=self.transition_format,
                            seed=self.seed,
                            actor_setup_function=self.actor_setup_function,
                            **worker_kwargs)
                for ray_kwargs, worker_kwargs in worker_configs]
            # @formatter:on
        self.replay_buffers = buffer_actors

        # todo this temporarily makes stronger assumptions about the relation of the replay buffers and the
        #       algorithms update step than is strictly necessary, so this could potentially be migrated into
        #       the algorithm class instead.
        self.online_buffer_actor = self.replay_buffers["online"][0]
        self.offline_buffer_actor = self.replay_buffers["offline"][0]

    def _setup_bellman_update_actors(self, action_clip_low, action_clip_high):
        bellman_update_actor_configs = self.bellman_update_worker_configurations
        assert len(bellman_update_actor_configs) > 0

        print(f"Starting {len(bellman_update_actor_configs)} bellman update actors.")

        # @formatter:off
        self.bellman_update_actors = [
            AlgoExecutionActor
                .options(**ray_kwargs)
                .remote(
                actor_setup_function=self.actor_setup_function,
                algorithm_factory=self.algorithm_factory,
                worker_control=self._make_worker_control("distributed_updater_worker_" + str(i)),
                seed=self.seed,
                **actor_kwargs)
            for i, (ray_kwargs, actor_kwargs) in enumerate(bellman_update_actor_configs)]
        # @formatter:on

    def _setup_episode_writer_actors(self):
        episode_writer_configs = self.episode_writer_worker_configurations

        if len(episode_writer_configs) != 1:
            raise ExperimentConfigError("The current implementation only supports using exactly one "
                                        "episode writer actor.")

        # @formatter:off
        self.episode_writers = [
            EpisodeWriterActor
                .options(**ray_kwargs)
                .remote(
                directory=os.path.abspath(self.experience_directory),
                io_factory=self.io_factory,
                actor_setup_function=self.actor_setup_function,
                **worker_kwargs)
            for ray_kwargs, worker_kwargs in episode_writer_configs
        ]
        # @formatter:on

        self.episode_writer = self.episode_writers[0]

    def _setup_gradient_actors(self):
        gradient_actor_configs = self.gradient_update_worker_configurations
        if len(gradient_actor_configs) != 1:
            raise ExperimentConfigError("The current implementation only supports using exactly one gradient actor.")

        # @formatter:off
        self.gradient_actors = [
            AlgoExecutionActor
                .options(**ray_kwargs)
                .remote(
                algorithm_factory=self.algorithm_factory,
                worker_control=self._make_worker_control("gradient_actor" + str(i)),
                actor_setup_function=self.actor_setup_function,
                seed=self.seed,
                **worker_kwargs)
            for i, (ray_kwargs, worker_kwargs) in enumerate(gradient_actor_configs)
        ]
        # @formatter:on

    def _setup_video_writer_actors(self):
        video_actor_configs = self.video_writer_worker_configurations

        if len(video_actor_configs) != 1:
            raise ExperimentConfigError("The current implementation only supports using exactly one"
                                        " video writer actor.")

        # @formatter:off
        self.evaluation_video_writer_actors = [
            VideoWriterActor
                .options(**ray_kwargs)
                .remote(
                actor_setup_function=self.actor_setup_function,
                **worker_kwargs)
            for ray_kwargs, worker_kwargs in video_actor_configs
        ]
        # @formatter:on

        self.evaluation_video_writer = self.evaluation_video_writer_actors[0]

    def _setup_log_replay_actors(self, coordinator_actor):
        log_replay_actor_configs = self.log_replay_worker_configurations

        target_preprocessors = [(p[1], p[2]) for p in self.preprocessing_actors if DataSources.LogReplay.value in p[0]]

        # @formatter:off
        self.log_replay_actors = [
            LogReplayActor
                .options(**ray_kwargs)
                .remote(
                target_preprocessors=target_preprocessors,

                coordinator_actor=coordinator_actor,
                drop_keys=self.drop_keys,
                actor_setup_function=self.actor_setup_function,
                io_factory=self.io_factory,
                **worker_kwargs
            )
            for ray_kwargs, worker_kwargs in log_replay_actor_configs
        ]
        # @formatter:on

    def _setup_preprocessing_actors(self):

        episode_writer = None
        if self.episode_writers is not None:
            episode_writer = self.episode_writers[0]
        replay_buffers = self.replay_buffers

        def _resolve_target(sinks):
            target_episode_writers = []
            target_replay_buffers = []
            for sink in sinks:
                if sink == DataSources.EpisodeWriter.value:
                    if episode_writer is not None:
                        target_episode_writers.append(episode_writer)
                else:
                    target_replay_buffers.append(replay_buffers[sink][0])
            return {
                "target_episode_writer_actors": target_episode_writers,
                "target_replay_buffer_actors": target_replay_buffers
            }

        def _resolve_sources(sources):
            out = []
            for source in sources:
                if source == DataSources.Rollouts.value:
                    out.extend([DataSources.EvaluationRollout.value, DataSources.TrainingRollout.value])
                elif source == DataSources.OnpolicyExperience.value:
                    out.extend([DataSources.TrainingRollout.value, DataSources.RestoredOnpolicyReplay.value])
                else:
                    out.append(source)
            return out

        prep_chains = []
        worker_index = 0
        for sources, sink, factory, workers in self.preprocessing_chains:
            if workers == 1:
                ray_kwargs, worker_kwargs = self.preprocessing_workers[worker_index]

                actor = \
                    PreprocessingActor.options(**ray_kwargs).remote(
                        chain_factory=factory,
                        seed=self.seed,
                        actor_setup_function=self.actor_setup_function,
                        **_resolve_target(sink),
                        **worker_kwargs
                    )
                ray.get(actor.transform_format.remote(self.transition_format))

                prep_chains.append((_resolve_sources(sources), actor, False))
                worker_index += 1
            else:
                actors = []
                for _ in range(workers):
                    ray_kwargs, worker_kwargs = self.preprocessing_workers[worker_index]
                    actor = \
                        PreprocessingActor.options(**ray_kwargs).remote(
                            chain_factory=factory,
                            seed=self.seed,
                            actor_setup_function=self.actor_setup_function,
                            **_resolve_target(sink),
                            **worker_kwargs
                        )
                    actors.append(actor)
                    ray.get(actor.transform_format.remote(self.transition_format))
                    worker_index += 1
                ray_kwargs, worker_kwargs = self.preprocessing_workers[worker_index]
                actor = \
                    DistributedPreprocessingActor.options(**ray_kwargs).remote(
                        actor_setup_function=self.actor_setup_function,
                        **worker_kwargs
                    )
                ray.get(actor.set_workers.remote(workers=actors))
                prep_chains.append((_resolve_sources(sources), actor, True))
                worker_index += 1

        assert len(self.preprocessing_workers) == worker_index

        # @formatter:off
        self.preprocessing_actors = prep_chains
        # @formatter:on
        # check that the sources are not accidentially in the enum type here
        assert all([all([type(s) == str for s in p[0]]) for p in self.preprocessing_actors])

    # endregion

    # region Config Handling

    def _assert_config_values_valid(self):
        if self.save_eval_videos:
            assert self.eval_video_save_interval > 0

        # the experiment config name is part of the output directory name in order to make it easier to associate which
        # directory belonged to which run, so it has to be file-safe
        if re.search(r'[/\\:*?"<>|]', self.experiment_tag):
            raise ExperimentConfigError("The experiment_tag must be safe to use in filenames. It may not contain "
                                        "/,\\,:,*,?,\",<,> or |.")

        # model save frequency has to be whole number
        assert self.checkpoint_interval % 1 == 0

    # endregion

    # region Driver Control Interface

    def update_weights(self, weights):
        self.weights = weights
        self.algorithm.update_weights(weights)

        futures = []
        for worker in self.bellman_update_actors:
            futures.append(worker.update_weights.remote(self.weights))

        for worker in self.gradient_actors:
            futures.append(worker.update_weights.remote(self.weights))

        # ray.wait(futures, num_returns=len(futures))

    def update_gradient_update_args(self, args):
        self.gradient_update_args = args

    def update_distributed_update_args(self, args):
        self.distributed_update_args = args

    def worker_control(self) -> WorkerControl:
        return self.local_worker_control

    def start_onpolicy_rollouts(self):
        self.rollout_manager.enable_training_rollouts()
        self.first_epoch_with_onpolicy_data = True

    def set_all_worker_checkpoint_states(self, state):
        futures = []
        for actor in self.bellman_update_actors:
            futures.append(actor.set_checkpoint_state.remote(state))
        for actor in self.gradient_actors:
            futures.append(actor.set_checkpoint_state.remote(state))
        ray.wait(futures, num_returns=len(futures))

    def broadcast_to_gradient_workers(self, data):
        wait_all([a.receive_broadcast.remote(data) for a in self.gradient_actors])

    def broadcast_to_distributed_update_workers(self, data):
        wait_all([a.receive_broadcast.remote(data) for a in self.bellman_update_actors])

    # endregion
