import ray
import json

from roerld.config import make_distributed_update_step_runner
from roerld.config.experiment_config import ExperimentConfig
from tests.environments.basic_control_test_env import register_basic_control_test_env


def train_basic_control_env():
    ray.init(ignore_reinit_error=True)

    def _actor_setup_function():
        register_basic_control_test_env()

    config_dict = ExperimentConfig.view(_make_config("../Datasets/basic-control-env-v1/onpolicy"))

    pipeline = make_distributed_update_step_runner(
        config_dict.section("pipeline"),
        config_dict,
        _actor_setup_function,
        epochs=5000,
        restore_from_checkpoint_path=None,
    )
    pipeline.run()


def _make_config(data_path):
    config_json = """
{
    "general_config": {
        "experiment_tag": "BasicControlEnvFromOnPolicy",
        "seed": 1066
    },
    "algorithm": {
        "model": {
            "name": "mlp_model",
            "layers": [
                {
                    "name": "Dense",
                    "units": 32,
                    "activation": "relu"
                },
                {
                    "name": "Dense",
                    "units": 32,
                    "activation": "relu"
                },
                {
                    "name": "Dense",
                    "units": 16,
                    "activation": "relu"
                },
                {
                    "name": "Dense",
                    "units": 8,
                    "activation": "relu"
                },
                {
                    "name": "Dense",
                    "units": 1,
                    "activation": "none"
                }
            ],
            "input_key_order": [
                "observation",
                "actions"
            ]
        },
        "onpolicy_fraction_strategy": {
            "name": "linear_rampup",
            "start_epoch": 100,
            "increase_per_epoch": 0.01,
            "max": 0.5
        },
        "cem_iterations": 3,
        "cem_initial_mean": 0,
        "cem_initial_std": 0.4,
        "cem_sample_count": 64,
        "cem_elite_sample_count": 6,
        "name": "qtopt",
        "gamma": 0.98,
        "polyak_factor": 0.999,
        "q_t2_update_every": 1000,
        "gradient_updates_per_epoch": 200,
        "gradient_update_batch_size": 32,
        "bellman_updater_batch_size": 512,
        "max_bellman_updater_optimizer_batch_size": 512,
        "optimizer": {
            "name": "adam",
            "kwargs": {
                "learning_rate": 0.001,
                "beta_1": 0.9,
                "beta_2": 0.999,
                "epsilon": 1e-03
            }
        }
    },
    "workers": {
        "bellman_update_workers": {
            "name": "worker_groups",
            "workers": [
                {
                    "name": "worker_group",
                    "count": 6,
                    "ray_kwargs": {
                        "num_cpus": 0.001,
                        "num_gpus": 0.01
                    },
                    "worker_kwargs": {
                        "tf_inter_op_parallelism_threads": 1,
                        "tf_intra_op_parallelism_threads": 1
                    }
                }
            ]
        },
        "rollout_workers": {
            "name": "worker_groups",
            "workers": [
                {
                    "name": "worker_group",
                    "count": 3,
                    "ray_kwargs": {
                        "num_cpus": 0.1,
                        "num_gpus": 0,
                        "resources": {}
                    },
                    "worker_kwargs": {
                        "tf_inter_op_parallelism_threads": 1,
                        "tf_intra_op_parallelism_threads": 1
                    }
                }
            ]
        },
        "online_buffer_workers": {
            "name": "worker_groups",
            "workers": [
                {
                    "name": "worker_group",
                    "count": 1,
                    "ray_kwargs": {
                        "num_cpus": 0.1,
                        "num_gpus": 0,
                        "resources": {}
                    },
                    "worker_kwargs": {}
                }
            ]
        },
        "offline_buffer_workers": {
            "name": "worker_groups",
            "workers": [
                {
                    "name": "worker_group",
                    "count": 1,
                    "ray_kwargs": {
                        "num_cpus": 0.1,
                        "num_gpus": 0,
                        "resources": {}
                    },
                    "worker_kwargs": {}
                }
            ]
        },
        "training_buffer_workers": {
            "name": "worker_groups",
            "workers": [
                {
                    "name": "worker_group",
                    "count": 1,
                    "ray_kwargs": {
                        "num_cpus": 0.1,
                        "num_gpus": 0,
                        "resources": {}
                    },
                    "worker_kwargs": {}
                }
            ]
        },
        "episode_writer_workers": {
            "name": "worker_groups",
            "workers": [
                {
                    "name": "worker_group",
                    "count": 1,
                    "ray_kwargs": {
                        "num_cpus": 0.1,
                        "num_gpus": 0,
                        "resources": {}
                    },
                    "worker_kwargs": {}
                }
            ]
        },
        "gradient_update_workers": {
            "name": "worker_groups",
            "workers": [
                {
                    "name": "worker_group",
                    "count": 1,
                    "ray_kwargs": {
                        "num_cpus": 1,
                        "num_gpus": 0.1
                    },
                    "worker_kwargs": {
                        "tf_inter_op_parallelism_threads": 1,
                        "tf_intra_op_parallelism_threads": 1
                    }
                }
            ]
        },
        "evaluation_video_writer_workers": {
            "name": "worker_groups",
            "workers": [
                {
                    "name": "worker_group",
                    "count": 1,
                    "ray_kwargs": {
                        "num_cpus": 0.1,
                        "num_gpus": 0,
                        "resources": {}
                    },
                    "worker_kwargs": {
                        "frame_repeat": 1,
                        "end_frame_still_frames": 30
                    }
                }
            ]
        },
        "coordinator_worker": {
            "ray_kwargs": {
                "resources": {}
            }
        },
        "log_replay_workers": {
            "name": "worker_groups",
            "workers": [
                {
                    "name": "worker_group",
                    "count": 1,
                    "ray_kwargs": {
                        "num_cpus": 1,
                        "num_gpus": 0,
                        "resources": {}
                    },
                    "worker_kwargs": {
                        "max_new_samples_per_epoch": 10000,
                        "data_folder": "DATA_PLACEHOLDER"
                    }
                }
            ]
        }
    },
    "pipeline": {
        "name": "qtopt_style",
        "max_episode_length": 100,
        "min_bellman_update_batches_per_epoch": 24,
        "model_save_frequency": 100,
        "drop_keys": [],
        "store_onpolicy_experience": false,
        "save_training_videos": false,
        "evaluation": {
            "name": "basic",
            "num_eval_episodes_per_epoch": 25,
            "evaluation_interval": 5,
            "save_videos": false,
            "save_videos_every_n_epochs": 1
        }
    },
    "rollout_manager": {
        "name": "static_rollouts",
        "max_training_rollouts_per_epoch": 10,
        "min_training_rollouts_per_epoch": 10,
        "render_every_n_training_rollouts": 0
    },
    "replay_buffers": {
        "offline": {
            "name": "ring_replay_buffer",
            "size": 400000
        },
        "online": {
            "name": "ring_replay_buffer",
            "size": 400000
        },
        "training": {
            "name": "q_targets",
            "size": 100000
        }
    },
    "rollouts": {
        "evaluation": {
            "video_render_mode": "bgr_array",
            "video_width": 128,
            "video_height": 128,
            "render_every_n_frames": 10
        }
    },
    "episode_writer": {
        "name": "json_driven",
        "max_bytes_before_flush": 1073741824,
        "max_episodes_per_file": 100,
        "image_keys": []
    },
    "environment": {
        "scope": "gym",
        "name": "BasicControlEnv-v1",
        "kwargs": {
        }
    },
    "exploration": {
        "name": "epsilon_greedy",
        "epsilon": 0.2,
        "scale": 1
    }
}
    """
    config_json = config_json.replace("DATA_PLACEHOLDER", data_path)
    return json.loads(config_json)


if __name__ == "__main__":
    train_basic_control_env()
