from abc import ABC

from roerld.data_handling.episode_reader import EpisodeReader
from roerld.data_handling.episode_writer import EpisodeWriter


class DataSource(ABC):
    def writer(self) -> EpisodeWriter:
        raise NotImplementedError()

    def reader(self, infinite_repeat=False) -> EpisodeReader:
        raise NotImplementedError()
