from abc import abstractmethod


class EpisodeReader:
    @abstractmethod
    def next_episode(self):
        raise NotImplementedError()

    @abstractmethod
    def next_episode_with_metadata(self):
        raise NotImplementedError()

    @abstractmethod
    def __enter__(self):
        raise NotImplementedError

    @abstractmethod
    def __exit__(self, exception_type, exception_value, traceback):
        raise NotImplementedError()