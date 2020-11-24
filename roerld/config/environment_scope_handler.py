import abc
from abc import abstractmethod

from roerld.config.experiment_config import ExperimentConfigView


class EnvironmentScopeHandler(abc.ABC):
    @abstractmethod
    def make_environment(self, config_section: ExperimentConfigView, **kwargs):
        raise NotImplementedError()
