import numpy as np

from roerld.execution.rollouts.rollout_worker import RolloutWorker

from tests.workers.rollout.rollout_test_mocks import MockRolloutTestEnv, MockRolloutTestActor, MockFixedActionActor

import pytest


@pytest.fixture()
def setup_fixture():
    actor = MockRolloutTestActor()
    env = MockRolloutTestEnv()
    rollout_worker = RolloutWorker(
        environment=env,
        actor=actor,
        max_episode_length=env.max_episode_length,
        local_render_mode=None,
        eval_video_render_mode=None,
        eval_video_width=0,
        eval_video_height=0,
        render_every_n_frames=1
    )

    return {
        "actor": actor,
        "env": env,
        "rollout_worker": rollout_worker
    }


def test_determinism(setup_fixture):
    setup_fixture["rollout_worker"].evaluation_rollout(1, False)
    assert setup_fixture["actor"].last_determinism
    setup_fixture["rollout_worker"].training_rollout(1, False)
    assert not setup_fixture["actor"].last_determinism


def test_basic_callbacks(setup_fixture):
    setup_fixture["rollout_worker"].evaluation_rollout(2, True)
    assert setup_fixture["env"].num_step_called == setup_fixture["env"].max_episode_length * 2
    assert setup_fixture["env"].num_reset_called == 2
    assert setup_fixture["env"].num_step_called_since_last_reset == setup_fixture["env"].max_episode_length
    assert setup_fixture["env"].num_close_called == 0
    assert setup_fixture["env"].num_render_called, setup_fixture["env"].max_episode_length * 2
    assert setup_fixture["actor"].num_episode_start_called == 2
    assert setup_fixture["actor"].num_episode_ended_called == 2
    assert setup_fixture["actor"].num_choose_action_called == 2 * setup_fixture["env"].max_episode_length


def test_actions_preserved(setup_fixture):
    experience, videos, episode_starts, diagnostics = setup_fixture["rollout_worker"].evaluation_rollout(3, False)
    np.testing.assert_allclose(experience["actions"], np.asarray(setup_fixture["actor"].actions_performed))
    setup_fixture["actor"].actions_performed = []
    experience, videos, episode_starts, diagnostics = setup_fixture["rollout_worker"].training_rollout(3, False)
    np.testing.assert_allclose(experience["actions"], np.asarray(setup_fixture["actor"].actions_performed))


def _assert_strictly_ascending_integer_sequence(x):
    """ Test for an increasing pattern. Requires an array of at least size 2"""
    assert len(x) >= 2
    for i in range(1, len(x)):
        if not (x[i] - 1 == x[i - 1]):
            assert False, "Sequence not ascending"


def test_rewards_correct(setup_fixture):
    experience, videos, episode_starts, diagnostics = setup_fixture["rollout_worker"].evaluation_rollout(3, False)
    _assert_strictly_ascending_integer_sequence(experience["rewards"])
    experience, videos, episode_starts, diagnostics = setup_fixture["rollout_worker"].training_rollout(3, False)
    _assert_strictly_ascending_integer_sequence(experience["rewards"])


def _test_observations_correct(training, setup_fixture):
    if training:
        experience, videos, episode_starts, diagnostics = setup_fixture["rollout_worker"].training_rollout(3, False)
    else:
        experience, videos, episode_starts, diagnostics = setup_fixture["rollout_worker"].evaluation_rollout(3, False)

    np.testing.assert_allclose(experience["next_observations_test1"],
                               np.asarray(setup_fixture["actor"].actions_performed))
    np.testing.assert_allclose(experience["next_observations_test2"],
                               2.0 * np.asarray(setup_fixture["actor"].actions_performed))

    np.testing.assert_allclose(np.asarray(experience["observations_test1"][0]), np.asarray([-1, -1, -1]))
    np.testing.assert_allclose(np.asarray(experience["observations_test2"][0]), np.asarray([-2, -2, -2]))
    np.testing.assert_allclose(np.asarray(experience["observations_test3"][0]), np.asarray([0]))

    np.testing.assert_allclose(np.asarray(experience["observations_test1"][setup_fixture["env"].max_episode_length]),
                               np.asarray([-1, -1, -1]))
    np.testing.assert_allclose(np.asarray(experience["observations_test2"][setup_fixture["env"].max_episode_length]),
                               np.asarray([-2, -2, -2]))
    np.testing.assert_allclose(np.asarray(experience["observations_test3"][setup_fixture["env"].max_episode_length]),
                               np.asarray([0]))

    np.testing.assert_allclose(
        np.asarray(experience["observations_test1"][2 * setup_fixture["env"].max_episode_length]),
        np.asarray([-1, -1, -1]))
    np.testing.assert_allclose(
        np.asarray(experience["observations_test2"][2 * setup_fixture["env"].max_episode_length]),
        np.asarray([-2, -2, -2]))
    np.testing.assert_allclose(
        np.asarray(experience["observations_test3"][2 * setup_fixture["env"].max_episode_length]),
        np.asarray([0]))

    np.testing.assert_allclose(experience["observations_test1"][1:setup_fixture["env"].max_episode_length],
                               np.asarray(setup_fixture["actor"].actions_performed)[
                               :setup_fixture["env"].max_episode_length - 1])
    np.testing.assert_allclose(experience["observations_test1"][setup_fixture["env"].max_episode_length + 1:
                                                                2 * setup_fixture["env"].max_episode_length],
                               np.asarray(setup_fixture["actor"].actions_performed)[
                               setup_fixture["env"].max_episode_length:
                               2 * setup_fixture["env"].max_episode_length - 1])
    np.testing.assert_allclose(experience["observations_test1"][2 * setup_fixture["env"].max_episode_length + 1:],
                               np.asarray(setup_fixture["actor"].actions_performed)[
                               2 * setup_fixture["env"].max_episode_length:-1])

    np.testing.assert_allclose(experience["observations_test2"][1:setup_fixture["env"].max_episode_length],
                               2.0 * np.asarray(setup_fixture["actor"].actions_performed)[
                                     :setup_fixture["env"].max_episode_length - 1])
    np.testing.assert_allclose(experience["observations_test2"][setup_fixture["env"].max_episode_length + 1:
                                                                2 * setup_fixture["env"].max_episode_length],
                               2.0 * np.asarray(setup_fixture["actor"].actions_performed)[
                                     setup_fixture["env"].max_episode_length:
                                     2 * setup_fixture["env"].max_episode_length - 1])
    np.testing.assert_allclose(experience["observations_test2"][2 * setup_fixture["env"].max_episode_length + 1:],
                               2.0 * np.asarray(setup_fixture["actor"].actions_performed)[
                                     2 * setup_fixture["env"].max_episode_length:-1])

    _assert_strictly_ascending_integer_sequence(
        experience["observations_test3"][1:setup_fixture["env"].max_episode_length])
    _assert_strictly_ascending_integer_sequence(
        experience["observations_test3"][
        setup_fixture["env"].max_episode_length + 1:2 * setup_fixture["env"].max_episode_length])
    _assert_strictly_ascending_integer_sequence(
        experience["observations_test3"][2 * setup_fixture["env"].max_episode_length + 1:])


def test_observations_correct_training(setup_fixture):
    _test_observations_correct(True, setup_fixture)


def test_observations_correct_evaluation(setup_fixture):
    _test_observations_correct(False, setup_fixture)


def test_episode_boundaries_correct(setup_fixture):
    experience, videos, episode_starts, diagnostics = setup_fixture["rollout_worker"].evaluation_rollout(3, False)

    np.testing.assert_equal(episode_starts,
                            np.asarray([0, setup_fixture["env"].max_episode_length,
                                        2 * setup_fixture["env"].max_episode_length]))


def test_respects_done(setup_fixture):
    length = 55
    setup_fixture["env"].set_episode_length(length)
    experience, videos, episode_starts, diagnostics = setup_fixture["rollout_worker"].evaluation_rollout(3, False)
    assert experience["actions"].shape == (3 * length, 3)
    assert setup_fixture["env"].num_step_called == 3 * length


def test_clips_action_correctly(setup_fixture):
    setup_fixture["actor"] = MockFixedActionActor(np.asarray([-1e9, -1e9, -1e9]))
    setup_fixture["rollout_worker"].actor = setup_fixture["actor"]
    setup_fixture["env"].set_episode_length(1)

    experience, _, _, _ = setup_fixture["rollout_worker"].evaluation_rollout(1, False)
    np.testing.assert_equal(setup_fixture["env"].action_log[0], setup_fixture["env"].action_space.low)

    setup_fixture["actor"].value = np.asarray([1e9, 1e9, 1e9])
    experience, _, _, _ = setup_fixture["rollout_worker"].evaluation_rollout(1, False)
    np.testing.assert_equal(setup_fixture["env"].action_log[1], setup_fixture["env"].action_space.high)

    # this should not be clipped
    good_action = setup_fixture["actor"].value = np.asarray([0, 0, 0])
    setup_fixture["actor"].value = good_action
    experience, _, _, _ = setup_fixture["rollout_worker"].evaluation_rollout(1, False)
    np.testing.assert_equal(setup_fixture["env"].action_log[2], good_action)

    # this should be partially clipped
    setup_fixture["actor"].value = np.asarray([-1e9, 1e9, 1])
    experience, _, _, _ = setup_fixture["rollout_worker"].evaluation_rollout(1, False)
    np.testing.assert_equal(setup_fixture["env"].action_log[3], np.asarray([-100000, 100000, 1]))
