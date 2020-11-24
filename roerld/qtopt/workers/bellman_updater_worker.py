import math

import numpy as np

from roerld.models.model import Model
from roerld.qtopt.cross_entropy_method import cross_entropy_method_normal_batched
from roerld.qtopt.workers.batch_remapping import remap_keys_next_observations_in_batch_without_actions


class BellmanUpdaterWorker:
    def __init__(self,
                 input_spec,
                 q_network_1: Model,
                 q_network_2: Model,

                 cem_iterations,
                 cem_initial_mean,
                 cem_initial_std,
                 cem_sample_count,
                 cem_elite_sample_count,

                 gamma,
                 action_clip_low,
                 action_clip_high,

                 max_optimizer_batch_size,

                 clip_q_target_max=None,
                 clip_q_target_min=None,
                 ):
        self.gamma = gamma
        self.q_network_1 = q_network_1
        self.q_network_2 = q_network_2
        self.action_dimension = input_spec["actions"][0][0]
        self.action_shape = input_spec["actions"][0]

        self.cem_iterations = cem_iterations
        self.cem_initial_mean = cem_initial_mean
        self.cem_initial_std = cem_initial_std
        self.cem_sample_count = cem_sample_count
        self.cem_elite_sample_count = cem_elite_sample_count

        self.action_clip_low = action_clip_low
        self.action_clip_high = action_clip_high

        self.clip_q_target_max = clip_q_target_max
        self.clip_q_target_min = clip_q_target_min

        self.optimizer_batch_size = max_optimizer_batch_size

        # the buffers used in the optimization are not reallocated until the batch size changes,
        #  setting last_batch_size will cause them to be initialized in _calculate_actions_parallel when the first
        #  batch arrives
        self.last_batch_size = -1
        self.opt_input_buffer = self.opt_input_action_buffer = None

    def _calculate_batch(self, remapped_next_observations, action_shape):
        random_obs_key = list(remapped_next_observations.keys())[0]

        samples_per_request = self.cem_sample_count
        batch_size = len(remapped_next_observations[random_obs_key])

        if batch_size > self.last_batch_size:
            self.opt_input_action_buffer = np.zeros(shape=(samples_per_request * batch_size,
                                                           *action_shape))

        self.opt_input_buffer = dict.fromkeys(remapped_next_observations)
        for key in self.opt_input_buffer:
            self.opt_input_buffer[key] = np.repeat(remapped_next_observations[key], samples_per_request, axis=0)

        def bellman_update_batched_objective_fn(parameters, action_sample_arrays):
            assert len(action_sample_arrays[0]) == samples_per_request
            assert len(parameters[random_obs_key]) == batch_size

            # care must be taken here: the update batch does contain an action key (which is later used to correctly
            #  insert the result into the training buffer, but we do not want this key as actions in the inputs below.

            # repeat the observation for each batch
            for i in range(batch_size):
                self.opt_input_action_buffer[(i * samples_per_request):((i + 1) * samples_per_request)] = \
                    action_sample_arrays[i]

            q_values = self.q_network_1.predict({
                **self.opt_input_buffer,
                "actions": self.opt_input_action_buffer
            })

            # reshape the q values into the original structure
            q_values = q_values.reshape((batch_size, samples_per_request))

            return q_values

        best_action = cross_entropy_method_normal_batched(
            objective_function=bellman_update_batched_objective_fn,
            objective_function_parameters=remapped_next_observations,
            num_items=len(remapped_next_observations[random_obs_key]),
            initial_mean=self.cem_initial_mean * np.ones(shape=self.action_dimension),
            initial_std=self.cem_initial_std * np.ones(shape=self.action_dimension),
            max_num_iterations=self.cem_iterations,
            sample_count=self.cem_sample_count,
            elite_sample_count=self.cem_elite_sample_count,
            clip_samples=False,
            clip_min=-1000000,
            clip_max=10000000)

        action = np.clip(best_action, self.action_clip_low, self.action_clip_high)
        return action

    def _calculate_actions_parallel(self, remapped_next_observations, action_shape):
        random_obs_key = list(remapped_next_observations.keys())[0]

        effective_batch_size = min(self.optimizer_batch_size, len(remapped_next_observations[random_obs_key]))
        batches = math.ceil(len(remapped_next_observations[random_obs_key]) / effective_batch_size)

        actions = []
        for batch_idx in range(batches):
            start_idx = batch_idx * effective_batch_size
            end_idx = min(len(remapped_next_observations[random_obs_key]), (batch_idx + 1) * effective_batch_size)
            actions.extend(self._calculate_batch(
                remapped_next_observations={k: v[start_idx:end_idx] for k, v in remapped_next_observations.items()},
                action_shape=action_shape
            ))

        return np.asarray(actions)

    def bellman_update_extended(self, bellman_update_batch):
        remapped_batch = remap_keys_next_observations_in_batch_without_actions(bellman_update_batch)
        remapped_batch["actions"] = self._calculate_actions_parallel(remapped_batch, self.action_shape)

        q_theta_1 = self.q_network_1.predict(input_dict=remapped_batch)
        q_theta_2 = self.q_network_2.predict(input_dict=remapped_batch)

        if self.clip_q_target_min is not None:
            q_theta_1 = np.maximum(self.clip_q_target_min, q_theta_1)
            q_theta_2 = np.maximum(self.clip_q_target_min, q_theta_2)
        if self.clip_q_target_max is not None:
            q_theta_1 = np.minimum(self.clip_q_target_max, q_theta_1)
            q_theta_2 = np.minimum(self.clip_q_target_max, q_theta_2)

        v_theta1_theta2 = np.minimum(q_theta_1, q_theta_2)

        assert q_theta_1.shape == q_theta_2.shape
        assert v_theta1_theta2.shape == bellman_update_batch["rewards"].shape

        bellman_backups = bellman_update_batch["rewards"] + self.gamma * (1.0 - bellman_update_batch["dones"]) * v_theta1_theta2

        self.last_batch_size = len(bellman_update_batch["rewards"])

        return bellman_backups, q_theta_1, q_theta_2

    def bellman_update(self, bellman_update_batch):
        return self.bellman_update_extended(bellman_update_batch)[0]
