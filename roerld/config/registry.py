import gym
from typing import Union, Callable, Any, Iterable

from roerld.config.environment_scope_handler import EnvironmentScopeHandler
from roerld.config.experiment_config import ExperimentConfigView, ExperimentConfig
from roerld.data_handling.data_source import DataSource
from roerld.execution.distributed_update_step_algorithm import DistributedUpdateStepAlgorithm
from roerld.execution.transition_format import TransitionFormat
from roerld.learning_actors.learning_actor import LearningActor
from roerld.models.model import Model
from roerld.preprocessing.streaming_episode_preprocessor import StreamingEpisodePreprocessor
from roerld.replay_buffers.replay_buffer import ReplayBuffer


class _ExperimentConfigRegistry:
    def __init__(self):
        self.models = {}
        self.distributed_update_step_runners = {}
        self.algorithms = {}
        self.env_scope_handlers = {}
        self.bootstrap_actors = {}
        self.replay_buffers = {}
        self.data_sources = {}
        self.preprocessors = {}
        self.exploration_actors = {}

    def make_environment(self, environment_config: Union[ExperimentConfigView, dict], **kwargs):
        view = ExperimentConfig.view(environment_config)
        if view.key("scope") not in self.env_scope_handlers:
            raise ValueError(f"There is no registered environment scope handler for scope {view.key('scope')}")

        return self.env_scope_handlers[view.key("scope")].make_environment(view, **kwargs)

    def make_distributed_update_step_runner(self,
                                            runner_config: Union[ExperimentConfigView, dict],
                                            experiment_config: Union[ExperimentConfigView, dict],
                                            actor_setup_function: Callable,
                                            **kwargs):
        view = ExperimentConfig.view(runner_config)
        name = view.key("name")
        if name not in self.distributed_update_step_runners:
            raise ValueError(f"There is no registered DistributedUpdateStepRunner with name {name}")
        return self.distributed_update_step_runners[name](
            view, ExperimentConfig.view(experiment_config), actor_setup_function, **kwargs)

    def make_distributed_update_step_algorithm(self,
                                               algorithm_config,
                                               **kwargs):
        view = ExperimentConfig.view(algorithm_config)
        name = view.key("name")
        if name not in self.algorithms:
            raise ValueError(f"There is no registered DistributedUpdateStepAlgorithm with name {name}."
                             f"Registered are: {','.join(self.algorithms.keys())}")
        return self.algorithms[name](view, **kwargs)

    def make_model(self, model_config: Union[ExperimentConfigView, dict], **kwargs):
        view = ExperimentConfig.view(model_config)
        name = view.key("name")
        if name not in self.models:
            raise ValueError(f"There is no registered Model with name {name}")
        return self.models[name](view, **kwargs)

    def make_bootstrapping_actor(self, actor_config: Union[ExperimentConfigView, dict],
                                 action_space: gym.spaces.Space, **kwargs):
        view = ExperimentConfig.view(actor_config)
        name = view.key("name")
        if name not in self.bootstrap_actors:
            raise ValueError(f"There is no registered Bootstrapping Actor with name {name}")
        return self.bootstrap_actors[name](view, action_space, **kwargs)

    def make_replay_buffer(self, replay_buffer_config, transition_format, **kwargs):
        view = ExperimentConfig.view(replay_buffer_config)
        name = view.key("name")
        if name not in self.replay_buffers:
            raise ValueError(f"There is no registered ReplayBuffer with name {name}")
        return self.replay_buffers[name](replay_buffer_config, transition_format, **kwargs)

    def make_data_source(self, data_source_config, paths, **kwargs):
        view = ExperimentConfig.view(data_source_config)
        name = view.key("name")
        if name not in self.data_sources:
            raise ValueError(f"There is no registered DataSource with name {name}")
        return self.data_sources[name](data_source_config, paths, **kwargs)

    def make_episode_preprocessor(self, preprocessor_config, **kwargs):
        view = ExperimentConfig.view(preprocessor_config)
        name = view.key("name")
        if name not in self.preprocessors:
            raise ValueError(f"There is no registered preprocessor with name {name}")
        return self.preprocessors[name](preprocessor_config, **kwargs)

    def make_exploration_actor(self, exploration_actor_config, policy_actor: LearningActor, action_space, **kwargs):
        view = ExperimentConfig.view(exploration_actor_config)
        name = view.key("name")
        if name not in self.exploration_actors:
            raise ValueError(f"There is no registered exploration actor with name {name}")
        return self.exploration_actors[name](exploration_actor_config, policy_actor, action_space, **kwargs)

    def register_distributed_update_step_runner(self, name, runner_factory: Callable[[ExperimentConfigView], Any]):
        if name in self.distributed_update_step_runners:
            raise ValueError(f"There already is a registered DistributedUpdateStepRunner with name {name}")
        self.distributed_update_step_runners[name] = runner_factory

    def register_environment_scope_handler(self, scope, handler: EnvironmentScopeHandler):
        if scope in self.env_scope_handlers:
            raise ValueError(f"There already is a registered EnvironmentScopeHandler for scope {scope}")
        self.env_scope_handlers[scope] = handler

    def register_distributed_update_step_algorithm(
            self, name, algorithm_factory: Callable[[ExperimentConfigView], DistributedUpdateStepAlgorithm]):
        if name in self.algorithms:
            raise ValueError(f"There already is a registered DistributedUpdateStepAlgorithm with name {name}")
        self.algorithms[name] = algorithm_factory

    def register_model(self, name, model_factory: Callable[[ExperimentConfigView], Model]):
        if name in self.models:
            raise ValueError(f"There already is a registered Model with name {name}")
        self.models[name] = model_factory

    def register_bootstrapping_actor(self, name, actor_factory):
        if name in self.bootstrap_actors:
            raise ValueError(f"There already is a registered Boostrapping Actor with name {name}")
        self.bootstrap_actors[name] = actor_factory

    def register_replay_buffer(self, name, replay_buffer_factory):
        if name in self.replay_buffers:
            raise ValueError(f"There already is a registered ReplayBuffer with the name {name}")
        self.replay_buffers[name] = replay_buffer_factory

    def register_data_source(self, name,
                             data_source_factory: Callable[[ExperimentConfigView, Iterable[str]], DataSource]):
        if name in self.replay_buffers:
            raise ValueError(f"There already is a registered DataSource with the name {name}")
        self.data_sources[name] = data_source_factory

    def register_episode_preprocessor(
            self, name, preprocessor_factory: Callable[[ExperimentConfigView], StreamingEpisodePreprocessor]):
        if name in self.preprocessors:
            raise ValueError(f"There already is a registered preprocessor with the name {name}")
        self.preprocessors[name] = preprocessor_factory

    def register_exploration_actor(self, name,
                                   exploration_actor_factory: Callable[[ExperimentConfigView, LearningActor],
                                                                       LearningActor]):
        if name in self.exploration_actors:
            raise ValueError(f"There already is a registered exploration actor with the name {name}")
        self.exploration_actors[name] = exploration_actor_factory


# global instance
_registry = _ExperimentConfigRegistry()


def make_environment(environment_config: Union[ExperimentConfigView, dict], **kwargs):
    return _registry.make_environment(environment_config, **kwargs)


def make_distributed_update_step_algorithm(algorithm_config, **kwargs):
    return _registry.make_distributed_update_step_algorithm(algorithm_config, **kwargs)


def make_distributed_update_step_runner(pipeline_config: Union[ExperimentConfigView, dict],
                                        experiment_config, actor_setup_function, **kwargs):
    return _registry.make_distributed_update_step_runner(pipeline_config, experiment_config, actor_setup_function,
                                                         **kwargs)


def make_model(model_config: Union[ExperimentConfigView, dict], **kwargs):
    return _registry.make_model(model_config, **kwargs)


def make_bootstrapping_actor(model_config: Union[ExperimentConfigView, dict],
                             action_space: gym.spaces.Space, **kwargs):
    return _registry.make_bootstrapping_actor(model_config, action_space, **kwargs)


def make_replay_buffer(replay_buffer_config, transition_format, **kwargs):
    return _registry.make_replay_buffer(replay_buffer_config, transition_format, **kwargs)


def make_data_source(data_source_config, paths, **kwargs):
    return _registry.make_data_source(data_source_config, paths, **kwargs)


def make_episode_preprocessor(preprocessor_config, **kwargs):
    return _registry.make_episode_preprocessor(preprocessor_config, **kwargs)


def make_exploration_actor(exploration_actor_config, policy_actor: LearningActor, action_space,  **kwargs):
    return _registry.make_exploration_actor(exploration_actor_config, policy_actor, action_space, **kwargs)


def register_distributed_update_step_runner(
        name,
        runner_factory: Callable[[ExperimentConfigView, ExperimentConfigView, Callable], Any]):
    return _registry.register_distributed_update_step_runner(name, runner_factory)


def register_environment_scope_handler(scope, handler: EnvironmentScopeHandler):
    return _registry.register_environment_scope_handler(scope, handler)


def register_distributed_update_step_algorithm(
        name,
        algorithm_factory: Callable[[], DistributedUpdateStepAlgorithm]):
    return _registry.register_distributed_update_step_algorithm(name, algorithm_factory)


def register_model(name, model_factory: Callable[[ExperimentConfigView], Model]):
    return _registry.register_model(name, model_factory)


def register_bootstrapping_actor(name, actor_factory: Callable[[gym.spaces.Space], LearningActor]):
    return _registry.register_bootstrapping_actor(name, actor_factory)


def register_replay_buffer(name,
                           replay_buffer_factory: Callable[[ExperimentConfigView, TransitionFormat], ReplayBuffer]):
    return _registry.register_replay_buffer(name, replay_buffer_factory)


def register_data_source(name,
                         data_source_factory: Callable[[ExperimentConfigView, Iterable[str]], DataSource]):
    return _registry.register_data_source(name, data_source_factory)


def register_episode_preprocessor(name,
                                  preprocessor_factory: Callable[[ExperimentConfigView], StreamingEpisodePreprocessor]):
    return _registry.register_episode_preprocessor(name, preprocessor_factory)


def register_exploration_actor(name, exploration_actor_factory: Callable[[ExperimentConfigView, LearningActor],
                                                                         LearningActor]):
    return _registry.register_exploration_actor(name, exploration_actor_factory)
