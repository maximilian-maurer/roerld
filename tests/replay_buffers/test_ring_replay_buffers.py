import numpy as np

from roerld.replay_buffers.ring_replay_buffer import RingReplayBuffer


def test_setup():
    buffer = RingReplayBuffer(
        fields={"test1": (100, np.float32)},
        size=10)
    assert buffer.buffers["test1"].shape == (10, 100)
    buffer2 = RingReplayBuffer(
        fields={"test1": ((10, 20), np.float32), "test2": (5, np.uint8)},
        size=30)
    assert buffer2.buffers["test1"].shape == (30, 10, 20)
    assert buffer2.buffers["test1"].dtype == np.float32
    assert buffer2.buffers["test2"].shape == (30, 5)
    assert buffer2.buffers["test2"].dtype == np.uint8


def test_store():
    sample_row = {"test1": np.array([[1, 1]]), "test2": np.array([1, 2, 3, 4, 5])}
    buffer2 = RingReplayBuffer(
        fields={"test1": ((1, 2), np.uint8), "test2": (5, np.uint8)},
        size=30)
    buffer2.store_single(**sample_row)
    buffer2.store_single(**sample_row)
    taken_back = buffer2.sample_batch(1)
    assert (np.array_equal(sample_row["test1"], taken_back["test1"][0]))
    assert (np.array_equal(sample_row["test2"], taken_back["test2"][0]))


def test_batch_store():
    sample_row = {"test1": np.array([[[1, 1]], [[2, 2]]]),
                  "test2": np.array([[1, 2, 3, 4, 5], [7, 8, 9, 10, 11]])}
    buffer2 = RingReplayBuffer(
        fields={"test1": ((1, 2), np.uint8), "test2": (5, np.uint8)},
        size=30)
    buffer2.store_batch(**sample_row)
    taken_back = buffer2.sample_batch(1)
    assert (np.array_equal(sample_row["test1"][0], taken_back["test1"][0])
            or np.array_equal(sample_row["test1"][1], taken_back["test1"][0]))
    assert (np.array_equal(sample_row["test2"][0], taken_back["test2"][0])
            or np.array_equal(sample_row["test2"][1], taken_back["test2"][0]))
