import os

from roerld.data_handling.tf_data_source import TFDataSource

import numpy as np


def test_tf_record_serialization(tmp_path):
    dir = tmp_path / "tmp1"
    os.mkdir(dir)
    ds = TFDataSource(dir, [])
    writer = ds.writer()

    data = [
        {
            "img": np.asarray([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8),
            "f1": np.asarray([[1.2, 2.4, 3.6], [3, 4, 6]], dtype=np.float32),
            "f3": np.asarray([[1.2, 2.4], [1, 2]], dtype=np.float32),
            "rewards": np.asarray([1, 2], dtype=np.float32)
        },

        {
            "img": np.asarray([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8)*2,
            "f1": np.asarray([[1.2, 2.4, 3.6], [3, 4, 6]], dtype=np.float32)*2,
            "f3": np.asarray([[1.2, 2.4], [1, 2]], dtype=np.float32)*2,
            "rewards": np.asarray([1, 2], dtype=np.float32)*2
        },

        {
            "img": np.asarray([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8)*3,
            "f1": np.asarray([[1.2, 2.4, 3.6], [3, 4, 6]], dtype=np.float32)*3,
            "f3": np.asarray([[1.2, 2.4], [1, 2]], dtype=np.float32)*3,
            "rewards": np.asarray([1, 2], dtype=np.float32)*3
        }
    ]

    with writer:
        for e in data:
            writer.write_episode(e)

    reader = ds.reader(False)
    episodes = []
    with reader:
        episode = reader.next_episode()
        while episode is not None:
            episodes.append(episode)
            episode = reader.next_episode()

    for index, episode in enumerate(episodes):
        for key in episode.keys():
            np.testing.assert_allclose(episode[key], data[index][key])


def test_tf_record_serialization_multi_file(tmp_path):
    dir = tmp_path / "tmp1"
    os.mkdir(dir)
    ds = TFDataSource(dir, [], 5)
    writer = ds.writer()

    data = [{
        "q1": np.arange(1, 200)*i,
        "q2": 2 * np.arange(1,200)*-2*i
    } for i in range(23)]

    with writer:
        for e in data:
            writer.write_episode(e)

    reader = ds.reader(False)
    episodes = []
    with reader:
        episode = reader.next_episode()
        while episode is not None:
            episodes.append(episode)
            episode = reader.next_episode()

    for index, episode in enumerate(episodes):
        assert any([all(np.allclose(episode[key], data[i][key]) for key in episode.keys()) for i in range(len(data))])
    assert len(episodes) == len(data)
