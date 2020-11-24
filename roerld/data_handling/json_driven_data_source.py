import json
import os
import sys
import zipfile

import cv2
import numpy as np

from roerld.data_handling.data_source import DataSource
from roerld.data_handling.episode_reader import EpisodeReader
from roerld.data_handling.episode_writer import EpisodeWriter


class JsonDrivenDataSource(DataSource):
    def __init__(self, directory, image_keys=[], max_episodes_per_file=100, max_bytes_before_flush=100 * 1024 * 1024):
        self.directory = directory

        if not os.path.exists(directory):
            print(f"Unable to find {directory}")
            assert os.path.exists(directory)
        self.image_keys = image_keys
        self.max_episodes_per_file = max_episodes_per_file
        self.max_bytes_before_flush = max_bytes_before_flush

    def writer(self):
        return JsonDrivenDataSourceWriter(self.directory, self.image_keys,
                                          self.max_episodes_per_file, self.max_bytes_before_flush)

    def reader(self, infinite_repeat=False):
        return JsonDrivenDataSourceReader(self.directory, infinite_repeat)


class JsonDrivenDataSourceWriter(EpisodeWriter):

    def __init__(self, directory, image_keys, max_episodes_per_file, max_bytes_before_flush):
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
        self.in_memory_data.append(episode)
        self.in_memory_data_size_estimate += np.sum([
            JsonDrivenDataSourceWriter._estimate_size(episode[key]) for key in episode
        ])

        if additional_episode_metadata is not None:
            self.episode_metadata[str(len(self.in_memory_data) - 1)] = additional_episode_metadata
        else:
            self.episode_metadata[str(len(self.in_memory_data) - 1)] = {}

        if len(self.in_memory_data) >= self.max_episodes_per_file or \
                self.in_memory_data_size_estimate >= self.max_bytes_before_flush:
            self._flush()

    def _nparrays_to_list(self, dictionary):
        subdicts = []
        convert_keys = []
        for key, value in dictionary.items():
            if type(value) == np.ndarray:
                convert_keys.append(key)
            if type(value) == dict:
                subdicts.append(key)
        for key in convert_keys:
            dictionary[key] = dictionary[key].tolist()
        for key in subdicts:
            dictionary[key] = self._nparrays_to_list(dictionary[key])
        return dictionary

    def _flush(self):
        if len(self.in_memory_data) == 0:
            return

        filename, filename_index = self._get_next_data_json_filename()
        data_file_path = os.path.join(self.directory, filename)
        assert not os.path.exists(data_file_path)

        image_folder_name = f"data_{filename_index}_images"
        os.mkdir(os.path.join(self.directory, image_folder_name))

        data_json = {
            "dataset_format_version": 1,
            "data_collection_run_metadata": {
                **self.additional_metadata
            },
            "metadata": {
                "image_keys": self.image_keys,
                "image_directory": image_folder_name,
            },
            "episode_metadata": self.episode_metadata,
            "episodes": {}
        }

        images_to_save = {}

        for episode_index, episode in enumerate(self.in_memory_data):
            image_filename_index = 0
            for key in self.image_keys:
                if key not in episode:
                    continue
                # since this is an image key, it actually contains the whole batch of images for the entire
                # episode
                image_filenames = []
                for image_index in range(len(episode[key])):
                    image_filename = f"image_e{episode_index}_i{image_filename_index}.png"
                    image_filename_index += 1
                    images_to_save[image_filename] = episode[key][image_index]
                    image_filenames.append(image_filename)
                episode[key] = image_filenames

            episode = self._nparrays_to_list(episode)
            data_json["episodes"][str(episode_index)] = episode

        print(f"Writing {data_file_path}")
        with open(data_file_path, "w") as data_file:
            data_file.write(json.dumps(data_json, indent=4))

        with zipfile.ZipFile(os.path.join(self.directory, image_folder_name, "images.zip"), "w") as zip_file:
            for key, image in images_to_save.items():
                is_success, byte_content = cv2.imencode(".png", image)
                # todo more graceful handling
                assert is_success
                zip_file.writestr(key, byte_content)

        del self.in_memory_data
        self.in_memory_data = []
        self.in_memory_data_size_estimate = 0
        self.episode_metadata = {}

    def __enter__(self):
        pass

    def __exit__(self, exception_type, exception_value, traceback):
        self._flush()
        self.additional_metadata = {}

    def _get_next_data_json_filename(self):
        for i in range(1, 1000000):
            filename = f"data_{i}.json"
            path = os.path.join(self.directory, filename)
            if os.path.exists(path):
                continue

            return filename, i
        raise ValueError(f"Could not find a free filename data_i.json in {self.directory}.")

    def set_additional_metadata(self, run_metadata):
        self.additional_metadata = run_metadata


class JsonDrivenDataSourceReader(EpisodeReader):
    def __init__(self, directory, infinite_repeat):
        self.directory = directory
        self.infinite_repeat = infinite_repeat

        self.current_batch = []
        self.current_episode_in_batch = 0
        self.current_batch_image_keys = []
        self.current_batch_zip_file = None
        self.current_batch_data = None

        self.had_at_least_one_episode = False

    def _batch_files(self):
        for root, _, files in os.walk(self.directory):
            for file in files:
                if not file.endswith(".json"):
                    continue
                if not file.startswith("data"):
                    continue
                yield os.path.join(root, file)
        return None

    def _clean_last_batch(self):
        if self.current_batch_zip_file is not None:
            self.current_batch_zip_file.close()

        self.current_batch = []
        self.current_episode_in_batch = 0
        self.current_batch_image_keys = []
        self.current_batch_zip_file = None

    def _initialize_generator(self):
        self.generator = self._batch_files()
        self.had_at_least_one_episode = False

    def __enter__(self):
        self._initialize_generator()

    def __exit__(self, exception_type, exception_value, traceback):
        self.generator = None
        self._clean_last_batch()
        self.had_at_least_one_episode = False

    def _read_batch(self, file):
        self.had_at_least_one_episode = True

        self._clean_last_batch()

        batch = None
        with open(file, "r") as json_file:
            batch = json.load(json_file)

        # load image keys
        self.current_batch_image_keys = batch["metadata"]["image_keys"]

        # open the image zip
        image_folder = batch["metadata"]["image_directory"]
        self.current_batch_zip_file = zipfile.ZipFile(os.path.join(os.path.dirname(file),
                                                                   image_folder, "images.zip"), "r")

        self.current_batch = list(batch["episodes"].values())

        self.current_batch_metadata = [batch["episode_metadata"][str(i)] for i in range(len(self.current_batch))]

        # convert to numpy arrays
        for idx in range(len(self.current_batch)):
            for key in self.current_batch[idx].keys():
                if key not in self.current_batch_image_keys:
                    self.current_batch[idx][key] = np.array(self.current_batch[idx][key])

    def next_episode_with_metadata(self):
        if self.current_episode_in_batch < len(self.current_batch):
            # there are still episodes in the current batch. We have already read the entire batch worth of non-image
            #  data, now load the images.
            this_episode_data = self.current_batch[self.current_episode_in_batch]
            this_episode_metadata = self.current_batch_metadata[self.current_episode_in_batch]
            for image_key in self.current_batch_image_keys:
                if image_key not in this_episode_data:
                    continue

                this_key_img_filenames = this_episode_data[image_key]
                images = [
                    cv2.imdecode(np.frombuffer(self.current_batch_zip_file.read(fn), dtype=np.uint8),
                                 cv2.IMREAD_UNCHANGED)
                    for fn
                    in this_key_img_filenames
                ]
                this_episode_data[image_key] = np.array(images)
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
            return None

        # we have a new batch file
        self._read_batch(next_batch_file)
        return self.next_episode_with_metadata()

    def next_episode(self):
        return self.next_episode_with_metadata()[0]
