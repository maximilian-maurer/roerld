import gym
from roerld.config.environment_scope_handler import EnvironmentScopeHandler
from roerld.config.experiment_config import ExperimentConfigView


class _GymScopeHandler(EnvironmentScopeHandler):
    def make_environment(self, config_section: ExperimentConfigView, **kwargs):
        return gym.make(config_section.key("name"), **config_section.key("kwargs"), **kwargs)
