from abc import abstractmethod


class EpisodeWriter:
    @abstractmethod
    def write_episode(self, episode, additional_episode_metadata=None):
        raise NotImplementedError()

    @abstractmethod
    def __enter__(self):
        raise NotImplementedError

    @abstractmethod
    def __exit__(self, exception_type, exception_value, traceback):
        raise NotImplementedError()

    @abstractmethod
    def set_additional_metadata(self, run_metadata):
        pass
