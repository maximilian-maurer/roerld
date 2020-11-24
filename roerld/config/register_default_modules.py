import copy
import os

from roerld.config.environment_scope_handlers import _GymScopeHandler
from roerld.config.experiment_config import ExperimentConfig, ExperimentConfigView
from roerld.config.registry import register_distributed_update_step_runner, \
    register_environment_scope_handler, make_environment, register_distributed_update_step_algorithm, \
    make_distributed_update_step_algorithm, register_model, make_model, register_bootstrapping_actor, \
    register_replay_buffer, make_replay_buffer, register_data_source, make_data_source, make_episode_preprocessor, \
    register_episode_preprocessor
from roerld.config.resolution.worker_group_resolution import resolve_worker_groups
from roerld.data_handling.json_driven_data_source import JsonDrivenDataSource
from roerld.data_handling.tf_data_source import TFDataSource
from roerld.execution.control.worker_control import WorkerControl
from roerld.execution.transition_format import TransitionFormat
from roerld.learning_actors.random.gaussian_random import GaussianRandomLearningActor
from roerld.learning_actors.random.random_from_action_space import RandomFromActionSpaceLearningActor
from roerld.models.cnn_with_mlp import CNNWithMLP
from roerld.models.mlp_model import MLPModel
from roerld.preprocessing.passthrough_preprocessor import PassthroughPreprocessor
from roerld.preprocessing.preprocessing_chain import PreprocessingChain
from roerld.preprocessing.preprocessors.data_interpretation.subsampling_resampler import SubsamplingResampler
from roerld.qtopt.qtopt import QtOpt
from roerld.replay_buffers.ring_replay_buffer import RingReplayBuffer
from roerld.utils.onpolicy_fraction_strategy.onpolicy_fraction_strategy_resolver import \
    resolve_onpolicy_fraction_strategy

import numpy as np


def _register_distributed_update_step_runners():
    from roerld.execution.runners.distributed_update_step_pipeline_runner import DistributedUpdateStepPipelineRunner

    def distributed_update_step_runner_factory(section, experiment_config, actor_setup_function, **kwargs):
        from roerld.cli.paths import PathKind, resolve_path

        def _env_function():
            return make_environment(experiment_config.section("environment"))

        def _algorithm_factory():
            config = ExperimentConfig.view(experiment_config.section("algorithm"))
            return make_distributed_update_step_algorithm(config)

        def _replay_buffer_factory_curry(config_section):
            def _replay_buffer_factory(transition_format):
                return make_replay_buffer(config_section, transition_format)

            return _replay_buffer_factory

        def _write_checkpoint_callback(checkpoint_dir):
            experiment_config.save_to_file(os.path.join(checkpoint_dir, "experiment_config.json"))

        experiment_config = ExperimentConfig.view(experiment_config)
        pipeline_section = experiment_config.section("pipeline")

        # todo this addresses aspects for which there is no proper resolution of what should be created yet
        assert pipeline_section.key("evaluation.name") == "basic"

        log_replay_workers = []
        if experiment_config.has_key("workers.log_replay_workers"):
            log_replay_workers = resolve_worker_groups(experiment_config.section("workers.log_replay_workers"))

        # preprocessing
        preprocessing_chains = experiment_config.key("preprocessing.chains")
        chains_out = []

        def _make_preprocessing_chain_curry(this_chain):
            local_chain = copy.deepcopy(this_chain)

            def _make_this_preprocessing_chain():
                preprocessor_configs = local_chain["preprocessors"]
                preprocessors = [make_episode_preprocessor(config) for config in preprocessor_configs]
    
                if len(preprocessors) == 0:
                    preprocessors = [PassthroughPreprocessor()]
    
                chain_instance = PreprocessingChain(preprocessors)
                return chain_instance
            return _make_this_preprocessing_chain

        for chain in preprocessing_chains:
            sources = chain["sources"]
            sink = chain["sink"]
            workers = chain["workers"] if "workers" in chain else 1
            factory = _make_preprocessing_chain_curry(chain)
            chains_out.append((sources, sink, factory, workers))

        preprocessing_workers = resolve_worker_groups(experiment_config.section("workers.preprocessing_workers"))

        runner = DistributedUpdateStepPipelineRunner(
            algorithm_factory=_algorithm_factory,
            environment_factory=_env_function,
            actor_setup_function=actor_setup_function,
            omit_timing_data=experiment_config.optional_key("pipeline.omit_timing_data", False),

            buffer_factories={
                k: [_replay_buffer_factory_curry(config_section=experiment_config.section(f"replay_buffers.{k}"))]
                for k in experiment_config.sections("replay_buffers")
            },
            buffer_worker_configurations={
                k: resolve_worker_groups(experiment_config.section(f"workers.{k}_buffer_workers"))
                for k in experiment_config.sections("replay_buffers")
            },
            gradient_update_worker_configurations=resolve_worker_groups(
                experiment_config.section("workers.gradient_update_workers")),
            bellman_update_worker_configurations=resolve_worker_groups(
                experiment_config.section("workers.bellman_update_workers")),
            coordinator_worker_configuration=(experiment_config.key("workers.coordinator_worker.ray_kwargs"), {}),

            video_writer_worker_configurations=resolve_worker_groups(
                experiment_config.section("workers.evaluation_video_writer_workers")),

            rollout_worker_configurations=resolve_worker_groups(experiment_config.section("workers.rollout_workers")),

            log_replay_worker_configurations=log_replay_workers,
            episode_writer_worker_configurations=resolve_worker_groups(
                experiment_config.section("workers.episode_writer_workers")),

            store_onpolicy_experience=experiment_config.key("pipeline.store_onpolicy_experience"),

            write_checkpoint_callback=_write_checkpoint_callback,

            log_path=resolve_path(experiment_config,
                                  path_kind=PathKind.NewLog,
                                  categories=[]),
            max_episode_length=pipeline_section.key("max_episode_length"),
            checkpoint_interval=pipeline_section.key("model_save_frequency"),
            min_bellman_update_batches_per_epoch=pipeline_section.key("min_bellman_update_batches_per_epoch"),
            seed=experiment_config.key("general_config.seed"),
            num_eval_episodes_per_epoch=pipeline_section.key("evaluation.num_eval_episodes_per_epoch"),
            save_eval_videos=pipeline_section.key("evaluation.save_videos"),
            save_training_videos=pipeline_section.key("save_training_videos"),
            eval_video_save_interval=pipeline_section.key("evaluation.save_videos_every_n_epochs"),
            eval_interval=pipeline_section.key("evaluation.evaluation_interval"),
            experiment_tag=experiment_config.key("general_config.experiment_tag"),

            io_factory=lambda paths: make_data_source(experiment_config.section("io"), paths),
            preprocessing_workers=preprocessing_workers,
            preprocessing_chains=chains_out,

            experiment_config=experiment_config,

            **kwargs
        )
        return runner

    register_distributed_update_step_runner("qtopt_style", distributed_update_step_runner_factory)


def _register_environments():
    gym_scope = _GymScopeHandler()
    register_environment_scope_handler("gym", gym_scope)


def _register_algorithms():
    def qt_opt_factory(algorithm_section: ExperimentConfigView, **kwargs):
        def model_factory(pipeline_info: WorkerControl):
            return make_model(algorithm_section.section("model"), input_spec=pipeline_info.input_spec())

        onpolicy_fraction_strategy = \
            resolve_onpolicy_fraction_strategy(algorithm_section.section("onpolicy_fraction_strategy"))

        qt_opt = QtOpt(
            q_network_factory=model_factory,
            gradient_updates_per_epoch=algorithm_section.key("gradient_updates_per_epoch"),
            polyak_factor=algorithm_section.key("polyak_factor"),
            q_t2_update_every=algorithm_section.key("q_t2_update_every"),
            gamma=algorithm_section.key("gamma"),
            gradient_update_batch_size=algorithm_section.key("gradient_update_batch_size"),
            bellman_updater_batch_size=algorithm_section.key("bellman_updater_batch_size"),
            cem_iterations=algorithm_section.key("cem_iterations"),
            cem_sample_count=algorithm_section.key("cem_sample_count"),
            cem_elite_sample_count=algorithm_section.key("cem_elite_sample_count"),
            cem_initial_mean=algorithm_section.key("cem_initial_mean"),
            cem_initial_std=algorithm_section.key("cem_initial_std"),
            onpolicy_fraction_strategy=onpolicy_fraction_strategy,
            optimizer=algorithm_section.key("optimizer.name"),
            optimizer_kwargs=algorithm_section.key("optimizer.kwargs"),
            max_bellman_updater_optimizer_batch_size=algorithm_section.key("max_bellman_updater_optimizer_batch_size"),
            full_prefetch=algorithm_section.optional_key("full_prefetch", False),
            clip_q_target_max=algorithm_section.optional_key("clip_q_target_max", None),
            clip_q_target_min=algorithm_section.optional_key("clip_q_target_min", None),
        )

        return qt_opt

    register_distributed_update_step_algorithm("qtopt", qt_opt_factory)


def _register_models():
    def mlp_factory(config_section, **kwargs):
        return MLPModel(config_section=config_section,
                        input_spec=kwargs["input_spec"].to_legacy_format())

    register_model("mlp_model", mlp_factory)

    def cnn_mlp_merge_sum_model_factory(config_section, **kwargs):
        return CNNWithMLP(config=config_section,
                          input_spec=kwargs["input_spec"].to_legacy_format(),
                          input_key_order=config_section.key("input_key_order"))

    register_model("cnn_mlp_merge_sum_model", cnn_mlp_merge_sum_model_factory)


def _register_bootstrapping_actors():
    def gaussian_random_actor_factory(config_section, action_space):
        config_section = ExperimentConfig.view(config_section)
        return GaussianRandomLearningActor(action_space,
                                           config_section.key("mean"),
                                           config_section.key("std"))

    register_bootstrapping_actor("gaussian_random", gaussian_random_actor_factory)

    register_bootstrapping_actor("random_from_action_space",
                                 lambda config, action_space: RandomFromActionSpaceLearningActor(action_space))


def _register_replay_buffers():
    def _make_ring_replay_buffer_from_transition_format(config_section, transition_format: TransitionFormat):
        config_section = ExperimentConfig.view(config_section)
        assert config_section.key("name") == "ring_replay_buffer"
        size = config_section.key("size")

        fields = {
            k: (transition_format.key_shape(k), transition_format.key_dtype(k))
            for k in transition_format.transition_keys()
        }

        return RingReplayBuffer(fields=fields, size=size)

    register_replay_buffer("ring_replay_buffer", _make_ring_replay_buffer_from_transition_format)

    def _make_q_targets_replay_buffer(config_section, transition_format: TransitionFormat):
        config_section = ExperimentConfig.view(config_section)
        assert config_section.key("name") == "q_targets"
        size = config_section.key("size")

        fields = {
            k: (transition_format.key_shape(k), transition_format.key_dtype(k))
            for k in transition_format.observation_in_transition_keys()
        }
        fields["dones"] = (transition_format.key_shape("dones"), transition_format.key_dtype("dones"))
        fields["actions"] = (transition_format.key_shape("actions"), transition_format.key_dtype("actions"))
        fields["stored_targets"] = (None, np.float32)

        return RingReplayBuffer(fields=fields, size=size)

    register_replay_buffer("q_targets", _make_q_targets_replay_buffer)


def _register_data_sources():
    def _make_json_data_source(config, paths):
        assert len(paths) == 1

        # get dataset handling
        config = ExperimentConfig.view(config)
        max_bytes_before_flush = config.key("max_bytes_before_flush")
        max_episodes_per_file = config.key("max_episodes_per_file")

        # todo: this is suboptimal
        image_keys = config.key("image_keys")

        actual_image_keys = []
        actual_image_keys.extend(["observations_" + key for key in image_keys])
        actual_image_keys.extend(["next_observations_" + key for key in image_keys])

        return JsonDrivenDataSource(paths[0],
                                    max_bytes_before_flush=max_bytes_before_flush,
                                    max_episodes_per_file=max_episodes_per_file,
                                    image_keys=actual_image_keys)

    register_data_source("json_driven", _make_json_data_source)

    def _make_tf_data_source(config, paths):
        assert len(paths) == 1

        config = ExperimentConfig.view(config)
        max_bytes_before_flush = config.key("max_bytes_before_flush")
        max_episodes_per_file = config.key("max_episodes_per_file")

        return TFDataSource(paths[0],
                            max_bytes_before_flush=max_bytes_before_flush,
                            max_episodes_per_file=max_episodes_per_file,
                            image_keys=[],
                            shuffle_files=config.optional_key("shuffle_files", False),
                            shuffle_episodes_in_files=config.optional_key("shuffle_episodes_in_files", False))

    register_data_source("tf_data_source", _make_tf_data_source)


def _register_preprocessors():
    def _make_subsampling_preprocessor(config, **kwargs):
        config = ExperimentConfig.view(config)
        return SubsamplingResampler(sample_interval=config.key("sample_interval"))

    register_episode_preprocessor("subsampling_resampler", _make_subsampling_preprocessor)


def _register_all():
    if _register_all.did_register:
        return

    _register_preprocessors()
    _register_data_sources()
    _register_replay_buffers()
    _register_environments()
    _register_models()
    _register_algorithms()
    _register_distributed_update_step_runners()
    _register_bootstrapping_actors()
    _register_all.did_register = True


_register_all.did_register = False
