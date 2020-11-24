import json
import os
import sys

import numpy as np
import tensorflow as tf

from roerld.data_handling.episode_reader import EpisodeReader
from roerld.data_handling.episode_writer import EpisodeWriter


class TFDataSource:
    def __init__(self, directory, image_keys, max_episodes_per_file=100, max_bytes_before_flush=100 * 1024 * 1024,
                 shuffle_files=False, shuffle_episodes_in_files=False):
        self.directory = directory
        assert os.path.exists(directory)

        self.shuffle_files = shuffle_files
        self.image_keys = image_keys
        self.max_episodes_per_file = max_episodes_per_file
        self.max_bytes_before_flush = max_bytes_before_flush
        self.shuffle_episodes_in_files = shuffle_episodes_in_files

    def writer(self):
        return TFDataSourceWriter(self.directory,
                                  self.image_keys,
                                  self.max_episodes_per_file,
                                  self.max_bytes_before_flush)

    def reader(self, infinite_repeat=False):
        return TFDataSourceReader(self.directory, infinite_repeat, self.shuffle_files,
                                  self.shuffle_episodes_in_files)


class TFDataSourceWriter(EpisodeWriter):

    def __init__(self, directory, image_keys, max_episodes_per_file, max_bytes_before_flush):
        """
        todo this does not currently serialize the info
        todo this should use the sequence formats
        Args:
            directory:
            image_keys:
            max_episodes_per_file:
            max_bytes_before_flush:
        """
        self.max_bytes_before_flush = max_bytes_before_flush
        self.max_episodes_per_file = max_episodes_per_file
        self.directory = directory
        self.image_keys = image_keys
        self.additional_metadata = {}

        self.in_memory_data = []
        self.in_memory_data_size_estimate = 0
        self.episode_metadata = {}

    @staticmethod
    def _estimate_size(obj):
        if hasattr(obj, "nbytes"):
            return obj.nbytes
        return sys.getsizeof(obj)

    def write_episode(self, episode, additional_episode_metadata=None):
        if "infos" in episode:
            # todo add separate info serialization
            del episode["infos"]

        self.in_memory_data.append(episode)
        self.in_memory_data_size_estimate += np.sum([
            TFDataSourceWriter._estimate_size(episode[key]) for key in episode
        ])

        if additional_episode_metadata is not None:
            self.episode_metadata[str(len(self.in_memory_data) - 1)] = additional_episode_metadata

        if len(self.in_memory_data) >= self.max_episodes_per_file or \
                self.in_memory_data_size_estimate >= self.max_bytes_before_flush:
            self._flush()

    def _convert_episode(self, data, episode_index):
        def _to_float_list(v):
            return tf.train.Feature(float_list=tf.train.FloatList(value=np.asarray(v).flatten()))

        def _to_int64_list(v):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=np.asarray(v).flatten()))

        record = {}
        for key in data.keys():
            if data[key].dtype == np.float or data[key].dtype == np.float32 or data[key].dtype == np.float64:
                feature = _to_float_list(data[key])
            elif data[key].dtype == np.uint8 or data[key].dtype == np.int64:
                feature = _to_int64_list(data[key])
            else:
                raise ValueError(f"Cannot serialize key {key}")
            record[key] = feature

        record["episode_index"] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=np.repeat(episode_index, len(data[list(data.keys())[0]]))))
        example = tf.train.Example(features=tf.train.Features(feature=record))

        records = [example.SerializeToString()]
        return records

    @staticmethod
    def dtype_to_str(dtype):
        if dtype == np.float32:
            return "np.float32"
        if dtype == np.float64:
            return "np.float64"
        elif dtype == np.uint8:
            return "np.uint8"
        elif dtype == np.int64:
            return "np.int64"
        raise ValueError(f"Cannot convert dtype {dtype}")

    @staticmethod
    def dtype_from_str(dtype):
        if dtype == "np.float32":
            return np.float32
        if dtype == "np.float64":
            return np.float64
        elif dtype == "np.uint8":
            return np.uint8
        elif dtype == "np.int64":
            return np.int64
        raise ValueError(f"Cannot convert dtype {dtype}")

    def _flush(self):
        if len(self.in_memory_data) == 0:
            return

        filename, filename_index = self._get_next_data_filename()
        data_file_path = os.path.join(self.directory, filename)
        metadata_file_path = os.path.join(self.directory, "meta_" + filename + ".json")
        assert not os.path.exists(data_file_path)

        # todo this is pending the preprocessing rework
        for e in self.in_memory_data:
            if "dones" in e:
                e["dones"] = np.asarray(e["dones"], dtype=np.float32)

        record_formats = {}
        first_episode = self.in_memory_data[0]
        first_transition = {k: v[0] for k, v in first_episode.items()}

        for key, value in first_transition.items():
            if type(value) == np.ndarray:
                shape = value.shape
                dtype = value.dtype
            elif type(value) == np.float32:
                shape = []
                dtype = np.float32
            elif type(value) == np.float64:
                shape = []
                dtype = np.float64
            elif type(value) == np.uint8:
                shape = []
                dtype = np.uint8
            elif type(value) == np.int64:
                shape = []
                dtype = np.int64
            else:
                raise ValueError(f"Cannot handle type {type(value)} of {key}")

            record_formats[key] = (shape, TFDataSourceWriter.dtype_to_str(dtype))
        record_formats["episode_index"] = ([], TFDataSourceWriter.dtype_to_str(np.int64))
        writer = tf.io.TFRecordWriter(data_file_path, options=tf.io.TFRecordOptions(compression_type="GZIP"))

        for episode_index, episode in enumerate(self.in_memory_data):
            converted = self._convert_episode(episode, episode_index)

            for record in converted:
                writer.write(record)

        print(f"Writing {data_file_path}")

        metadata_json = {
            "dataset_format_version": 2,
            "data_collection_run_metadata": {
                **self.additional_metadata
            },
            "metadata": {
            },
            "episode_metadata": self.episode_metadata,
            "record_format": record_formats,
        }
        with open(metadata_file_path, "w") as data_file:
            data_file.write(json.dumps(metadata_json, indent=4))

        del self.in_memory_data
        self.in_memory_data = []
        self.in_memory_data_size_estimate = 0
        self.episode_metadata = {}

    def __enter__(self):
        pass

    def __exit__(self, exception_type, exception_value, traceback):
        self._flush()
        self.additional_metadata = {}

    def _get_next_data_filename(self):
        for i in range(1, 1000000):
            filename = f"data_{i}.tfrecord"
            path = os.path.join(self.directory, filename)
            if os.path.exists(path):
                continue

            return filename, i
        raise ValueError(f"Could not find a free filename data_i.tfrecord in {self.directory}.")

    def set_additional_metadata(self, run_metadata):
        self.additional_metadata = run_metadata


class TFDataSourceReader(EpisodeReader):
    def __init__(self, directory, infinite_repeat, shuffle_files,
                 shuffle_episodes_in_files):
        self.directory = directory
        self.infinite_repeat = infinite_repeat
        self.shuffle_files = shuffle_files
        self.shuffle_episodes_in_files = shuffle_episodes_in_files

        self.current_batch = []
        self.current_episode_in_batch = 0
        self.current_batch_data = None

        self.had_at_least_one_episode = False

    def _batch_files(self):
        all_files = []
        for root, _, files in os.walk(self.directory):
            for file in files:
                if not file.endswith(".tfrecord"):
                    continue
                if not file.startswith("data"):
                    continue
                meta_file_name = "meta_" + os.path.basename(file) + ".json"

                all_files.append((os.path.join(root, file),
                                  os.path.join(root, os.path.dirname(file), meta_file_name)))

        if self.shuffle_files:
            np.random.shuffle(all_files)

        for f, m in all_files:
            yield f, m
        return None

    def _clean_last_batch(self):
        self.current_batch = []
        self.current_episode_in_batch = 0

    def _initialize_generator(self):
        self.generator = self._batch_files()
        self.had_at_least_one_episode = False

    def __enter__(self):
        self._initialize_generator()

    def __exit__(self, exception_type, exception_value, traceback):
        self.generator = None
        self._clean_last_batch()
        self.had_at_least_one_episode = False

    def _read_batch(self, file_tuple):
        self.had_at_least_one_episode = True
        self._clean_last_batch()

        file, metadata_file = file_tuple

        with open(metadata_file, "r") as meta_file:
            metadata = json.load(meta_file)

        tf_features = {}
        for key, value in metadata["record_format"].items():
            metadata["record_format"][key] = (value[0], TFDataSourceWriter.dtype_from_str(value[1]))
            dtype = tf.as_dtype(metadata["record_format"][key][1])
            # these can only be stored in the format as int64, and need to be loaded as such and converted afterwards
            if dtype == tf.uint8:
                dtype = tf.int64
            if dtype == tf.double or dtype == tf.float64:
                dtype = tf.float32

            tf_features[key] = tf.io.VarLenFeature(dtype)

        reader = tf.data.TFRecordDataset([file], compression_type="GZIP")

        records = list(reader.take(-1).as_numpy_iterator())
        parsed_records = tf.io.parse_example(records, features=tf_features)

        batch_dim = len(records)
        new_records = {}
        for key, value in parsed_records.items():
            new_records[key] = np.asarray(tf.sparse.to_dense(value).numpy(), dtype=metadata["record_format"][key][1])
            new_records[key] = new_records[key].reshape((batch_dim, -1, *metadata["record_format"][key][0]))

        episodes = []
        episode_indices = []
        for i in range(batch_dim):
            episode = dict.fromkeys(new_records)
            for key in episode.keys():
                episode[key] = new_records[key][i]
            episode_indices.append(episode["episode_index"][0])
            del episode["episode_index"]
            episodes.append(episode)

        order_sort = np.argsort(episode_indices)
        self.current_batch = [episodes[i] for i in order_sort]

        def _optional_metadata(i):
            return metadata["episode_metadata"][str(i)] if str(i) in metadata["episode_metadata"] else {}

        self.current_batch_metadata = [_optional_metadata(i) for i in range(len(self.current_batch))]

        if self.shuffle_episodes_in_files:
            shuffle = np.random.permutation(len(self.current_batch))
            self.current_batch = [self.current_batch[i] for i in shuffle]
            self.current_batch_metadata = [self.current_batch_metadata[i] for i in shuffle]

    def next_episode_with_metadata(self):
        if self.current_episode_in_batch < len(self.current_batch):
            # there are still episodes in the current batch. We have already read the entire batch worth of non-image
            #  data, now load the images.
            this_episode_data = self.current_batch[self.current_episode_in_batch]
            this_episode_metadata = self.current_batch_metadata[self.current_episode_in_batch]
            self.current_episode_in_batch += 1
            return this_episode_data, this_episode_metadata

        # try and load the next batch
        next_batch_file = next(self.generator, None)
        if next_batch_file is None:
            if self.infinite_repeat and self.had_at_least_one_episode:
                # prevent infinite recursion when the dataset is empty
                self._initialize_generator()
                return self.next_episode_with_metadata()

            # this is the end
            return None, None

        # we have a new batch file
        self._read_batch(next_batch_file)
        return self.next_episode_with_metadata()

    def next_episode(self):
        data, _ = self.next_episode_with_metadata()
        return data
