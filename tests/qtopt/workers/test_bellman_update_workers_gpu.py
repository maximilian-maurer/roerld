"""
    Tests the bellman update workers. They are supposed to calculate
        best_next_action = arg max_a  Q1(next_obs, a)
        targets = reward + gamma * (1-done) * min(Q1(next_obs, best_next_action), Q2(next_obs, best_next_action))

    In order ot test this all the test cases in this class use the same system:
        * The sample observations ascend by 2 for every observation value throughout the whole array,
            so that every value in the observation array is different
        * The sum prediction model mock is such that the optimal action by predicted Q value for an
            observation [o_0, o_1, ..., o_n] is sum_i o_i, at which point it predicts a Q value of 1. Values
            away from that optimum are penalized with squared difference
        * Hence, with a reward of 0, gamma of 1, done of 0, the optimal action for all samples has been found when the
            targets all equal 1.
    """

import numpy as np

from roerld.qtopt.workers.bellman_updater_worker import BellmanUpdaterWorker
from roerld.qtopt.workers.bellman_updater_worker_gpu import BellmanUpdaterWorkerGPU
from tests.mocks.constant_prediction_model_mock_gpu import ConstantPredictionModelMockGPU
from tests.mocks.sum_prediction_model_mock_gpu import SumPredictionModelMockGPU


def _generate_sample_data_pattern():
    # in order to be able to ascertain whether all actions were correctly chosen
    # and no data got intermixed, this pattern is used in the test-cases,
    # see the documentation of BellmanUpdateWorkerTests for more information
    obs = []
    number = 2
    for i in range(100):
        obs.append([number, number + 2, number + 4])
        number += 8

    input_spec = {
        "next_observations_obs": ((5,), np.float32),
        "actions": ((1,), np.float32)
    }

    return {
               "next_observations_obs": np.array(obs, dtype=np.float32),
               "actions": np.zeros(shape=(len(obs), 1), dtype=np.float32),
               "rewards": np.zeros(shape=len(obs), dtype=np.float32),
               "dones": np.zeros(shape=len(obs), dtype=np.float32),
           }, input_spec


def _generate_sample_data_pattern_multi_action():
    # in order to be able to ascertain whether all actions were correctly chosen
    # and no data got intermixed, this pattern is used in the test-cases,
    # see the documentation of BellmanUpdateWorkerTests for more information
    obs = []
    number = 2
    for i in range(100):
        obs.append([number, number + 2, number + 4])
        number += 8

    input_spec = {
        "next_observations_obs": ((5,), np.float32),
        "actions": ((2,), np.float32)
    }

    return {
               "next_observations_obs": np.array(obs, dtype=np.float32),
               "actions": np.zeros(shape=(len(obs), 1), dtype=np.float32),
               "rewards": np.zeros(shape=len(obs), dtype=np.float32),
               "dones": np.zeros(shape=len(obs), dtype=np.float32),
           }, input_spec


def test_action_selection():
    q1, q2 = SumPredictionModelMockGPU("obs", "actions"), ConstantPredictionModelMockGPU(1e9)
    input_dict, input_spec = _generate_sample_data_pattern()

    bellman_updater = BellmanUpdaterWorkerGPU(
        input_spec=input_spec,
        q_network_1=q1,
        q_network_2=q2,
        gamma=1,
        cem_iterations=10,
        cem_sample_count=100,
        cem_initial_mean=0,
        cem_initial_std=2000,
        cem_elite_sample_count=10,

        action_clip_low=-1e9,
        action_clip_high=1e9,
        max_optimizer_batch_size=1e4
    )

    results = bellman_updater.bellman_update(input_dict)

    # per definition of the sample data and the networks, if all the actions that were selected were chosen
    # optimally, then the predicted Q on them is 0, so with a gamma of one the overall q value for each entry
    # should be 0
    np.testing.assert_allclose(results,
                               np.ones(input_dict["next_observations_obs"].shape[0]),
                               atol=0.0001, rtol=0.001)


def test_action_selection_multi_action():
    q1, q2 = SumPredictionModelMockGPU("obs", "actions"), ConstantPredictionModelMockGPU(1e9)
    input_dict, input_spec = _generate_sample_data_pattern_multi_action()

    bellman_updater = BellmanUpdaterWorkerGPU(
        input_spec=input_spec,
        q_network_1=q1,
        q_network_2=q2,
        gamma=1,
        cem_iterations=60,
        cem_sample_count=100,
        cem_initial_mean=0,
        cem_initial_std=5000,
        cem_elite_sample_count=10,

        action_clip_low=-1e9,
        action_clip_high=1e9,
        max_optimizer_batch_size=59
    )

    results = bellman_updater.bellman_update(input_dict)

    # per definition of the sample data and the networks, if all the actions that were selected were chosen
    # optimally, then the predicted Q on them is 0, so with a gamma of one the overall q value for each entry
    # should be 0
    np.testing.assert_allclose(results,
                               np.ones(input_dict["next_observations_obs"].shape[0]),
                               atol=0.0001, rtol=0.001)


def test_action_selection_batching():
    q1, q2 = SumPredictionModelMockGPU("obs", "actions"), ConstantPredictionModelMockGPU(1e9)
    input_dict, input_spec = _generate_sample_data_pattern()

    bellman_updater = BellmanUpdaterWorkerGPU(
        input_spec=input_spec,
        q_network_1=q1,
        q_network_2=q2,
        gamma=1,
        cem_iterations=10,
        cem_sample_count=100,
        cem_initial_mean=0,
        cem_initial_std=2000,
        cem_elite_sample_count=10,

        action_clip_low=-1e9,
        action_clip_high=1e9,
        max_optimizer_batch_size=3
    )

    results = bellman_updater.bellman_update(input_dict)

    # per definition of the sample data and the networks, if all the actions that were selected were chosen
    # optimally, then the predicted Q on them is 0, so with a gamma of one the overall q value for each entry
    # should be 0
    np.testing.assert_allclose(results,
                               np.ones(input_dict["next_observations_obs"].shape[0]),
                               atol=0.0001, rtol=0.001)


def test_correct_q_network_chosen():
    q1, q2 = ConstantPredictionModelMockGPU(2), SumPredictionModelMockGPU("obs", "actions")
    input_dict, input_spec = _generate_sample_data_pattern()

    bellman_updater = BellmanUpdaterWorkerGPU(
        input_spec=input_spec,
        q_network_1=q1,
        q_network_2=q2,
        gamma=1,
        cem_iterations=10,
        cem_sample_count=100,
        cem_initial_mean=0,
        cem_initial_std=2000,
        cem_elite_sample_count=10,

        action_clip_low=-1e9,
        action_clip_high=1e9,
        max_optimizer_batch_size=1e4
    )

    results = bellman_updater.bellman_update(input_dict)

    # the correct q values cannot be arrived at with q1. This test will be passed if the task just outputs
    # 0 for all q values, but in that case, it will fail test_action_selection
    assert not (np.allclose(results,
                            np.zeros(input_dict["next_observations_obs"].shape[0]),
                            atol=0.0001, rtol=0.001))


def test_reward_values_used():
    q1, q2 = SumPredictionModelMockGPU("obs", "actions"), ConstantPredictionModelMockGPU(1e9)
    input_dict, input_spec = _generate_sample_data_pattern()
    input_dict["rewards"] = np.arange(0, len(input_dict["rewards"]), 1)

    bellman_updater = BellmanUpdaterWorkerGPU(
        input_spec=input_spec,
        q_network_1=q1,
        q_network_2=q2,
        # gamma = 0 -> use only the reward, nothing else
        gamma=0,
        cem_iterations=1,
        cem_sample_count=100,
        cem_initial_mean=0,
        cem_initial_std=2000,
        cem_elite_sample_count=10,

        action_clip_low=-1e9,
        action_clip_high=1e9,
        max_optimizer_batch_size=1e4
    )

    results = bellman_updater.bellman_update(input_dict)

    # since gamma is 0, Q_1 should be equal to the rewards
    np.testing.assert_allclose(results,
                               input_dict["rewards"],
                               atol=0.0001, rtol=0.001)


def test_dones_used():
    q1, q2 = SumPredictionModelMockGPU("obs", "actions"), ConstantPredictionModelMockGPU(1e9)
    input_dict, input_spec = _generate_sample_data_pattern()
    input_dict["dones"] = np.ones(len(input_dict["rewards"]))

    bellman_updater = BellmanUpdaterWorkerGPU(
        input_spec=input_spec,
        q_network_1=q1,
        q_network_2=q2,
        # gamma = 0 -> use only the reward, nothing else
        gamma=0,
        cem_iterations=1,
        cem_sample_count=100,
        cem_initial_mean=0,
        cem_initial_std=2000,
        cem_elite_sample_count=10,

        action_clip_low=-1e9,
        action_clip_high=1e9,
        max_optimizer_batch_size=1e4
    )

    results = bellman_updater.bellman_update(input_dict)

    # reward is 0, done is 1, hence the results should be 0
    np.testing.assert_allclose(results,
                               np.zeros(input_dict["next_observations_obs"].shape[0]),
                               atol=0.0001, rtol=0.001)


def test_double_clipped():
    q1, q2 = ConstantPredictionModelMockGPU(1), ConstantPredictionModelMockGPU(2)
    input_dict, input_spec = _generate_sample_data_pattern()

    bellman_updater = BellmanUpdaterWorkerGPU(
        input_spec=input_spec,
        q_network_1=q1,
        q_network_2=q2,
        gamma=1,
        cem_iterations=3,
        cem_sample_count=100,
        cem_initial_mean=0,
        cem_initial_std=1,
        cem_elite_sample_count=10,

        # actions are irrelevant for this test case
        action_clip_low=-1,
        action_clip_high=1,
        max_optimizer_batch_size=1e4
    )

    results = bellman_updater.bellman_update(input_dict)
    np.testing.assert_array_almost_equal(results, np.ones(input_dict["next_observations_obs"].shape[0]))

    q1, q2 = ConstantPredictionModelMockGPU(3), ConstantPredictionModelMockGPU(2)
    input_dict, input_spec = _generate_sample_data_pattern()

    bellman_updater = BellmanUpdaterWorkerGPU(
        input_spec=input_spec,
        q_network_1=q1,
        q_network_2=q2,
        gamma=1,
        cem_iterations=3,
        cem_sample_count=100,
        cem_initial_mean=0,
        cem_initial_std=1,
        cem_elite_sample_count=10,

        # actions are irrelevant for this test case
        action_clip_low=-1,
        action_clip_high=1,
        max_optimizer_batch_size=1e4
    )

    results = bellman_updater.bellman_update(input_dict)
    np.testing.assert_array_almost_equal(results, 2 * np.ones(input_dict["next_observations_obs"].shape[0]))
