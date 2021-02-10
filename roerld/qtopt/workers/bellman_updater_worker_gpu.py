import math

import numpy as np

from roerld.models.model import Model
from roerld.qtopt.cross_entropy_method import cross_entropy_method_normal_batched
from roerld.qtopt.workers.batch_remapping import remap_keys_next_observations_in_batch_without_actions
from roerld.qtopt.workers.bellman_updater_worker import BellmanUpdaterWorker

import tensorflow as tf


class BellmanUpdaterWorkerGPU(BellmanUpdaterWorker):
    def __init__(self, input_spec, q_network_1: Model, q_network_2: Model, cem_iterations, cem_initial_mean,
                 cem_initial_std, cem_sample_count, cem_elite_sample_count, gamma, action_clip_low, action_clip_high,
                 max_optimizer_batch_size, clip_q_target_max=None, clip_q_target_min=None):

        super().__init__(input_spec, q_network_1, q_network_2, cem_iterations, cem_initial_mean, cem_initial_std,
                         cem_sample_count, cem_elite_sample_count, gamma, action_clip_low, action_clip_high,
                         max_optimizer_batch_size, clip_q_target_max, clip_q_target_min)

        max_num_iterations = self.cem_iterations
        elite_sample_count = self.cem_elite_sample_count
        sample_count = self.cem_sample_count

        assert max_num_iterations > 0
        assert sample_count > 0
        assert elite_sample_count > 0
        assert elite_sample_count <= sample_count

        action_dimension = self.action_dimension
        initial_mean = tf.constant(tf.ones(shape=self.action_dimension) * self.cem_initial_mean)
        initial_std = tf.constant(tf.ones(shape=self.action_dimension) * self.cem_initial_std)

        q_network_1 = self.q_network_1

        @tf.function
        def gpu_bellman_update(num_items, inputs):
            mean = tf.repeat([initial_mean], num_items, axis=0)
            std = tf.repeat([initial_std], num_items, axis=0)

            for _ in range(max_num_iterations):
                mean = tf.reshape(tf.repeat(mean, repeats=sample_count, axis=0),
                                  shape=(num_items, sample_count, action_dimension))
                std = tf.reshape(tf.repeat(std, repeats=sample_count, axis=0),
                                 shape=(num_items, sample_count, action_dimension))
                sample_requests = tf.random.normal(mean=mean, stddev=std, shape=(num_items, sample_count, action_dimension))

                q_values = q_network_1.tensor_predict({
                    **inputs,
                    "actions": tf.reshape(sample_requests, shape=(num_items*sample_count, action_dimension))
                }, training=False)
                q_values = tf.reshape(q_values, shape=(num_items, cem_sample_count))

                _, indices = tf.math.top_k(q_values, k=elite_sample_count, sorted=False)
                all_elite_samples = tf.gather(sample_requests, indices, batch_dims=1)

                mean = tf.reduce_mean(all_elite_samples, axis=1)
                std = tf.math.reduce_std(all_elite_samples, axis=1)

            max_qs = tf.argmax(q_values, axis=1)
            best_samples = tf.gather(sample_requests, max_qs, batch_dims=1)

            return best_samples

        self.gpu_bellman_update = gpu_bellman_update

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

        best_action = self.gpu_bellman_update(
            num_items=len(remapped_next_observations[random_obs_key]),
            inputs={k: tf.convert_to_tensor(v) for k, v in self.opt_input_buffer.items()}
        ).numpy()

        action = np.clip(best_action, self.action_clip_low, self.action_clip_high)
        return action
