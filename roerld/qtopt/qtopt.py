import os
import time

import numpy as np
import tensorflow as tf
from roerld.execution.control.driver_control import DriverControl
from roerld.execution.control.worker_control import WorkerControl
from roerld.execution.distributed_update_step_algorithm import DistributedUpdateStepAlgorithm
from roerld.execution.utils.timings import TimingHelper
from roerld.qtopt.cross_entropy_method import cross_entropy_method_normal
from roerld.qtopt.workers.batch_remapping import remap_keys_observations_in_batch_with_actions, \
    remap_keys_observations_in_batch_without_actions
from roerld.qtopt.workers.bellman_updater_worker import BellmanUpdaterWorker
from roerld.qtopt.workers.bellman_updater_worker_gpu import BellmanUpdaterWorkerGPU


class QtOpt(DistributedUpdateStepAlgorithm):
    def __init__(self,
                 q_network_factory,

                 gradient_updates_per_epoch,
                 polyak_factor,
                 q_t2_update_every,
                 gamma,
                 gradient_update_batch_size,

                 bellman_updater_batch_size,
                 max_bellman_updater_optimizer_batch_size,

                 cem_iterations,
                 cem_sample_count,
                 cem_elite_sample_count,
                 cem_initial_mean,
                 cem_initial_std,

                 optimizer: str,
                 optimizer_kwargs,

                 onpolicy_fraction_strategy,

                 clip_q_target_max=None,
                 clip_q_target_min=None,

                 full_prefetch=False
                 ):
        super().__init__()
        self.q_network_factory = q_network_factory
        self.gradient_updates_per_epoch = gradient_updates_per_epoch
        self.polyak_factor = polyak_factor
        self.q_t2_update_every = q_t2_update_every
        self.gamma = gamma
        self.gradient_update_batch_size = gradient_update_batch_size
        self.bellman_updater_batch_size = bellman_updater_batch_size
        self.cem_iterations = cem_iterations
        self.cem_sample_count = cem_sample_count
        self.cem_elite_sample_count = cem_elite_sample_count
        self.cem_initial_mean = cem_initial_mean
        self.cem_initial_std = cem_initial_std
        self.onpolicy_fraction_strategy = onpolicy_fraction_strategy
        self.current_onpolicy_fraction = 0
        self.last_onpolicy_fraction = 0
        self.gradient_update_count = 0
        self.max_bellman_updater_optimizer_batch_size = max_bellman_updater_optimizer_batch_size
        self.full_prefetch = full_prefetch
        self.clip_q_target_max = clip_q_target_max
        self.clip_q_target_min = clip_q_target_min
        self.epoch = 0

        self.restored = False

        self.optimizer_name = optimizer
        assert self.optimizer_name == "adam"
        self.optimizer_kwargs = optimizer_kwargs

        self.q_t0_optimizer = None
        self.worker_control = self.q_network_0 = self.q_network_1 = self.q_network_2 \
            = self.action_clip_low = self.action_clip_high = self.bellman_updater_worker = self.tf_train_step = None

    def setup(self, worker_control: WorkerControl, worker_specific_kwargs):
        self.worker_control = worker_control

        self.q_network_0 = self.q_network_factory(worker_control)
        self.q_network_1 = self.q_network_factory(worker_control)
        self.q_network_2 = self.q_network_factory(worker_control)

        self.action_clip_low = worker_control.action_space().low
        self.action_clip_high = worker_control.action_space().high

        gpus_available = len(tf.config.experimental.list_physical_devices("GPU")) > 0

        if gpus_available:
            print("Worker has GPU available.")

            self.bellman_updater_worker = BellmanUpdaterWorkerGPU(
                input_spec=worker_control.input_spec().to_legacy_format(),
                q_network_1=self.q_network_1,
                q_network_2=self.q_network_2,
                gamma=self.gamma,
                cem_iterations=self.cem_iterations,
                cem_sample_count=self.cem_sample_count,
                cem_elite_sample_count=self.cem_elite_sample_count,
                cem_initial_mean=self.cem_initial_mean,
                cem_initial_std=self.cem_initial_std,

                action_clip_low=self.action_clip_low,
                action_clip_high=self.action_clip_high,
                max_optimizer_batch_size=self.max_bellman_updater_optimizer_batch_size,

                clip_q_target_max=self.clip_q_target_max,
                clip_q_target_min=self.clip_q_target_min
            )
        else:
            self.bellman_updater_worker = BellmanUpdaterWorker(
                input_spec=worker_control.input_spec().to_legacy_format(),
                q_network_1=self.q_network_1,
                q_network_2=self.q_network_2,
                gamma=self.gamma,
                cem_iterations=self.cem_iterations,
                cem_sample_count=self.cem_sample_count,
                cem_elite_sample_count=self.cem_elite_sample_count,
                cem_initial_mean=self.cem_initial_mean,
                cem_initial_std=self.cem_initial_std,

                action_clip_low=self.action_clip_low,
                action_clip_high=self.action_clip_high,
                max_optimizer_batch_size=self.max_bellman_updater_optimizer_batch_size,

                clip_q_target_max=self.clip_q_target_max,
                clip_q_target_min=self.clip_q_target_min
            )

        if self.optimizer_name == "adam":
            self.q_t0_optimizer = tf.compat.v2.keras.optimizers.Adam(**self.optimizer_kwargs)
        else:
            raise NotImplementedError()

        self.loss = tf.keras.losses.MeanSquaredError()

        loss = self.loss
        q_t0_optimizer = self.q_t0_optimizer
        polyak_factor = self.polyak_factor

        q_network_0 = self.q_network_0
        q_network_1 = self.q_network_1
        q_network_2 = self.q_network_2

        @tf.function
        def _train_step(inputs, stored_targets, do_q2_update):
            with tf.GradientTape() as gradient_tape:
                predictions = q_network_0.tensor_predict(inputs, training=True)
                q_network_0_loss = loss(y_true=stored_targets, y_pred=predictions)
                q_network_0_loss = q_network_0_loss + tf.reduce_sum(q_network_0.get_regularization_losses())

            q_network_0_gradient = gradient_tape.gradient(q_network_0_loss,
                                                          q_network_0.get_trainable_vars())
            q_t0_optimizer.apply_gradients(zip(q_network_0_gradient, q_network_0.get_trainable_vars()))

            trainable_vars_q0 = q_network_0.model.trainable_variables
            trainable_vars_q1 = q_network_1.model.trainable_variables

            for i in range(len(trainable_vars_q0)):
                trainable_vars_q1[i].assign(
                    polyak_factor * trainable_vars_q1[i] + (1 - polyak_factor) * trainable_vars_q0[i])

            if do_q2_update:
                trainable_vars_q2 = q_network_2.model.trainable_variables

                for i in range(len(trainable_vars_q1)):
                    trainable_vars_q2[i].assign(trainable_vars_q1[i])

            # see https://datascience.stackexchange.com/questions/22163/why-does-q-learning-diverge/29742#29742
            gradient_magnitudes = tf.reduce_sum(
                [tf.reduce_sum(gradient ** 2) for gradient in q_network_0_gradient]) ** 0.5

            return q_network_0_loss, gradient_magnitudes

        self.tf_train_step = _train_step

    def _aggregate_training_data(self, offpolicy_transitions, onpolicy_transitions):
        bellman_update_batch = {}
        for key in offpolicy_transitions.keys():
            bellman_update_batch[key] = np.concatenate((offpolicy_transitions[key],
                                                        onpolicy_transitions[key]), axis=0)
        return bellman_update_batch

    def distributed_update_step(self, arguments, worker_control: WorkerControl):
        onpolicy_fraction = arguments["onpolicy_fraction"]

        assert onpolicy_fraction <= 1
        assert onpolicy_fraction >= 0

        timer = TimingHelper("Time Bellman Updater Get Data")

        offpolicy_samples = int(self.bellman_updater_batch_size * (1 - onpolicy_fraction))
        onpolicy_samples = self.bellman_updater_batch_size - offpolicy_samples

        bellman_update_batch_offpolicy = worker_control.replay_buffer("offline").sample_batch_blocking(offpolicy_samples)
        bellman_update_batch_onpolicy = worker_control.replay_buffer("online").sample_batch_blocking(onpolicy_samples)

        assert len(bellman_update_batch_offpolicy["rewards"]) == offpolicy_samples
        assert len(bellman_update_batch_onpolicy["rewards"]) == onpolicy_samples

        bellman_update_batch = self._aggregate_training_data(bellman_update_batch_offpolicy,
                                                             bellman_update_batch_onpolicy)

        timer.time_stamp("Time Bellman Updater Bellman Update")
        bellman_backups, q1, q2 = self.bellman_updater_worker.bellman_update_extended(bellman_update_batch)
        timer.time_stamp()

        observations = remap_keys_observations_in_batch_without_actions(bellman_update_batch)

        worker_control.replay_buffer("training").store_batch(
            **dict({"observations_"+k: v for k, v in observations.items()}),
            actions=bellman_update_batch["actions"],
            dones=bellman_update_batch["dones"],
            stored_targets=bellman_backups)

        diagnostics = {
            **timer.result(),
            "Bellman Update Q1's": q1,
            "Bellman Update Q2's": q2
        }

        return None, diagnostics

    def gradient_update(self, arguments, worker_control: WorkerControl):
        start_time = time.perf_counter()
        diagnostics = {}

        all_training_data = None
        if self.full_prefetch:
            all_training_data = worker_control.replay_buffer("training") \
                .sample_batch_blocking(self.gradient_updates_per_epoch * self.gradient_update_batch_size)

        gradient_magnitudes = []
        q0_losses = []
        for i in range(self.gradient_updates_per_epoch):
            # train q_theta_0 from training buffer
            if not self.full_prefetch:
                training_batch = worker_control.replay_buffer("training").sample_batch_blocking(
                    self.gradient_update_batch_size)
            else:
                training_batch = {k: v[i * self.gradient_update_batch_size:(i + 1) * self.gradient_update_batch_size]
                                  for k, v in all_training_data.items()}

            remapped = remap_keys_observations_in_batch_with_actions(training_batch)

            do_qt2_update = ((self.gradient_update_count+1) % self.q_t2_update_every) == 0
            q0_loss, gradient_magnitude = self.tf_train_step(inputs=remapped,
                                                             stored_targets=training_batch["stored_targets"],
                                                             do_q2_update=do_qt2_update)

            self.gradient_update_count = self.gradient_update_count + 1
            q0_losses.append(q0_loss)
            gradient_magnitudes.append(gradient_magnitude)

            if not self.full_prefetch:
                del training_batch

        if self.full_prefetch:
            del all_training_data

        time_taken = time.perf_counter() - start_time
        diagnostics.update({
            "q0_loss": q0_losses,
            "time_gradient_actor_train": time_taken,
            "gradient_magnitudes": gradient_magnitudes
        })

        # todo This should actually return gradients. For now this owns the optimizer and returns the
        #  updated model weights. Refactor as part of the gradient actor rework.
        new_weights = {
            "q_network_0": self.q_network_0.get_weights(),
            "q_network_1": self.q_network_1.get_weights(),
            "q_network_2": self.q_network_2.get_weights()
        }

        self.restored = True

        return new_weights, diagnostics

    def before_initial_buffer_fill(self, driver):
        if self.restored:
            return

        initial_weights = self.q_network_1.get_weights()
        self.q_network_0.set_weights(initial_weights)
        self.q_network_2.set_weights(initial_weights)

        driver.update_gradient_update_args(None)
        driver.update_distributed_update_args({
            "onpolicy_fraction": self.onpolicy_fraction_strategy.onpolicy_fraction(driver.worker_control().epoch())
        })
        driver.update_weights(self.get_weights())

    def start_epoch(self, driver: DriverControl):
        self.current_onpolicy_fraction = self.onpolicy_fraction_strategy.onpolicy_fraction(
            driver.worker_control().epoch())
        assert self.current_onpolicy_fraction >= 0.
        assert self.current_onpolicy_fraction <= 1.

        driver.update_distributed_update_args({
            "onpolicy_fraction": self.current_onpolicy_fraction  # todo load this from checkpoint
        })

        if self.current_onpolicy_fraction >= 1e-8 > self.last_onpolicy_fraction:
            # this is the first epoch where onpolicy data collection is required
            driver.start_onpolicy_rollouts()

        self.last_onpolicy_fraction = self.current_onpolicy_fraction

    def end_epoch(self, gradient_update_results, distributed_update_results, driver: DriverControl):
        assert len(gradient_update_results) == 1

        new_weights = gradient_update_results[0]

        self.q_network_0.set_weights(new_weights["q_network_0"])
        self.q_network_1.set_weights(new_weights["q_network_1"])
        self.q_network_2.set_weights(new_weights["q_network_2"])

        driver.update_weights(self.get_weights())

        return {
            "On-Policy Fraction": self.current_onpolicy_fraction,
        }

    def choose_action(self, observation):
        def eval_objective_fn(action_samples):
            inputs_repeated = dict([(k, np.repeat([v], len(action_samples), axis=0))
                                    for k, v in observation.items()])
            inputs_repeated["actions"] = action_samples
            q_values = self.q_network_1.predict(input_dict=inputs_repeated, training=False)

            return q_values

        action_dim = self.worker_control.action_space().shape[0]

        best_action = cross_entropy_method_normal(objective_function=eval_objective_fn,
                                                  initial_mean=self.cem_initial_mean * np.ones(shape=action_dim),
                                                  initial_std=self.cem_initial_std * np.ones(shape=action_dim),
                                                  max_num_iterations=self.cem_iterations,
                                                  sample_count=self.cem_sample_count,
                                                  elite_sample_count=self.cem_elite_sample_count)
        action_clip_low = self.worker_control.action_space().low
        action_clip_high = self.worker_control.action_space().high

        action = np.clip(best_action, a_min=action_clip_low, a_max=action_clip_high)

        return action

    def checkpoint(self, path, collected_checkpoint_states, driver: DriverControl):
        states_with_checkpoint_file = [s for s in collected_checkpoint_states if "checkpoint_file" in s]
        assert len(states_with_checkpoint_file) == 1

        for filename, data in states_with_checkpoint_file[0]["checkpoint_file"]:
            with open(os.path.join(path, filename), "wb") as file:
                file.write(data)

    def restore_checkpoint(self, path, driver: DriverControl):
        all_files = []
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), "rb") as file:
                data = file.read()
            all_files.append((filename, data))

        state = {
            "checkpoint_file": all_files
        }

        driver.set_all_worker_checkpoint_states(state)

        self.q_network_0.load(os.path.join(path, "q_network_0.h5"))
        self.q_network_1.load(os.path.join(path, "q_network_1.h5"))
        self.q_network_2.load(os.path.join(path, "q_network_2.h5"))

        driver.update_weights(self.get_weights())

    def get_checkpoint_state(self, worker_control: WorkerControl):
        checkpoint = tf.train.Checkpoint(optimizer=self.q_t0_optimizer,
                                         model_q0=self.q_network_0.model,
                                         model_q1=self.q_network_1.model,
                                         model_q2=self.q_network_2.model)
        path = worker_control.create_temporary_directory()

        checkpoint.save(os.path.join(path, "checkpoint_"))

        self.q_network_0.save(os.path.join(path, "q_network_0.h5"))
        self.q_network_1.save(os.path.join(path, "q_network_1.h5"))
        self.q_network_2.save(os.path.join(path, "q_network_2.h5"))

        # todo: this is not particularly efficient, and with larger models will significantly bloat up the
        #  object store, but is the easiest way to get the checkpoint to the driver process without a file abstraction
        #  for the workers
        all_files = []
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), "rb") as file:
                data = file.read()
            all_files.append((filename, data))

        return {
            "checkpoint_file": all_files
        }

    def set_checkpoint_state(self, state, worker_control: WorkerControl):
        checkpoint = tf.train.Checkpoint(optimizer=self.q_t0_optimizer,
                                         model_q0=self.q_network_0.model,
                                         model_q1=self.q_network_1.model,
                                         model_q2=self.q_network_2.model)
        path = worker_control.create_temporary_directory()

        for filename, data in state["checkpoint_file"]:
            with open(os.path.join(path, filename), "wb") as file:
                file.write(data)

        # cannot use assert_consumed here as this may be called before the optimizer has done an initial
        # training step (compare TF issue 33150)
        checkpoint.restore(os.path.join(path, "checkpoint_-1")).assert_existing_objects_matched()

        print("Restored Gradient Worker from checkpoint")

    def update_weights(self, weights):
        self.q_network_0.set_weights(weights["q_network_0"])
        self.q_network_1.set_weights(weights["q_network_1"])
        self.q_network_2.set_weights(weights["q_network_2"])

    def get_weights(self):
        return {
            "q_network_0": self.q_network_0.get_weights(),
            "q_network_1": self.q_network_1.get_weights(),
            "q_network_2": self.q_network_2.get_weights()
        }

    def receive_broadcast(self, data):
        pass
