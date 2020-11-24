import argparse
import json
import os
import sys

from roerld.config import make_environment
from roerld.config.experiment_config import ExperimentConfig
from roerld.config.registry import make_distributed_update_step_algorithm
from roerld.data_handling.json_driven_data_source import JsonDrivenDataSource
from roerld.execution.control.driver_control import DriverControl
from roerld.execution.control.remote_replay_buffer import RemoteReplayBuffer
from roerld.execution.control.worker_control import WorkerControl
from roerld.execution.distributed_update_step_algorithm import DistributedUpdateStepAlgorithm

from roerld.execution.rollouts.rollout_worker import RolloutWorker

from pathlib import Path

from roerld.execution.transition_format import TransitionFormat
from roerld.execution.utils.experience import rollout_experience_into_episodes
from roerld.learning_actors.wrappers.algo_actor_wrapper import AlgoActorWrapper
import numpy as np


def find_checkpoint(args) -> str:
    if args.use_latest_experiment and args.checkpoint_path is not None:
        print("Experiment is ambiguous: Either use use_latest_experiment or provide the checkpoint path.")
        exit()
    if not args.use_latest_experiment and args.checkpoint_path is None:
        print("Please provide a checkpoint path.")
        exit()

    checkpoint_path = None
    if args.checkpoint_path is not None:
        checkpoint_path = args.checkpoint_path
    elif args.use_latest_experiment:
        experiment_dir = sorted(Path("logs").iterdir(), key=os.path.getmtime)[-1]

        if args.use_latest_checkpoint:
            sorted_checkpoints = sorted([p for p in Path(experiment_dir).iterdir() if os.path.isdir(str(p))],
                                        key=os.path.getmtime)
            if len(sorted_checkpoints) == 0:
                print(f"The latest experiment {experiment_dir} does not have any checkpoints.")
                exit()
            return str(sorted_checkpoints[-1])
        else:
            raise NotImplementedError()

    else:
        raise ValueError("Illegal State")

    return checkpoint_path


class _WorkerControl(WorkerControl):
    def __init__(self, input_spec, observation_space, action_space):
        self._input_spec = input_spec
        self._observation_space = observation_space
        self._action_space = action_space

    def input_spec(self):
        return self._input_spec

    def replay_buffer(self, name) -> RemoteReplayBuffer:
        raise NotImplementedError()

    def observation_space(self):
        return self._observation_space

    def action_space(self):
        return self._action_space

    def epoch(self):
        return 0

    def create_temporary_directory(self):
        raise NotImplementedError()


class _DriverControl(DriverControl):
    def broadcast_to_gradient_workers(self, data):
        self.algorithm.receive_broadcast(data)

    def broadcast_to_distributed_update_workers(self, data):
        self.algorithm.receive_broadcast(data)

    def __init__(self, algorithm: DistributedUpdateStepAlgorithm, worker_control):
        self.algorithm = algorithm
        self._worker_control = worker_control

    def update_weights(self, weights):
        self.algorithm.update_weights(weights)

    def update_gradient_update_args(self, args):
        raise NotImplementedError()

    def update_distributed_update_args(self, args):
        raise NotImplementedError()

    def worker_control(self) -> WorkerControl:
        return self._worker_control

    def start_onpolicy_rollouts(self):
        raise NotImplementedError()

    def set_all_worker_checkpoint_states(self, state):
        # intentionally left empty
        pass


def run_trained_policy_cli(argv):
    parser = argparse.ArgumentParser(description="Run the model from the given checkpoint.")
    parser.add_argument('--use-latest-experiment',
                        required=False,
                        action="store_true",
                        help="Use the latest experiment log directory by file creation date.")
    parser.add_argument('--save-data',
                        required=False,
                        action="store_true",
                        help="")
    parser.add_argument('--use-latest-checkpoint',
                        required=False,
                        action="store_true",
                        help="Use the latest checkpoint.")
    parser.add_argument('--checkpoint-path',
                        required=False,
                        type=str,
                        help="Use the latest checkpoint.")
    parser.add_argument('--out',
                        required=False,
                        type=str,
                        help="")
    parser.add_argument('--checkpoint-dir',
                        required=False,
                        type=str,
                        help="Path to the checkpoint directory.")
    parser.add_argument('--rollouts',
                        required=False,
                        type=int,
                        default=1,
                        help="Number of rollouts.")
    parser.add_argument('--render-mode',
                        required=False,
                        type=str,
                        default="human",
                        help="Render mode for the environment.")
    parser.add_argument('--seed',
                        required=False,
                        type=int,
                        help="Seed to use. If not provided, a random seed will be chosen.")

    args = parser.parse_args(argv)

    checkpoint_dir = find_checkpoint(args)

    config_path = os.path.join(checkpoint_dir, "experiment_config.json")

    with open(config_path, "r") as config_file:
        config_dict = json.load(config_file)
    config_dict["general_config"]["seed"] = args.seed
    experiment_config = ExperimentConfig.view(config_dict)

    env = make_environment(experiment_config.section("environment"))

    input_spec = TransitionFormat.from_gym_spaces(env.observation_space, env.action_space)
    render = args.render_mode is not None
    local_render_mode = args.render_mode if render else None

    algorithm = make_distributed_update_step_algorithm(experiment_config.section("algorithm"))
    rollout_worker = RolloutWorker(
        environment=env,
        actor=AlgoActorWrapper(algorithm),
        max_episode_length=experiment_config.key("pipeline.max_episode_length"),
        local_render_mode=local_render_mode,
        eval_video_render_mode=None,
        eval_video_height=None,
        eval_video_width=None,
        render_every_n_frames=1)

    worker_control = _WorkerControl(input_spec, observation_space=env.observation_space, action_space=env.action_space)
    driver_control = _DriverControl(algorithm, worker_control)
    algorithm.setup(worker_control, {})
    algorithm.restore_checkpoint(checkpoint_dir, driver_control)

    # output this here, otherwise it will get drowned out by initialization logging
    print(f"Running checkpoint {checkpoint_dir}")

    # get dataset handling
    if args.out is not None:
        max_bytes_before_flush = experiment_config.key("episode_writer.max_bytes_before_flush")
        max_episodes_per_file = experiment_config.key("episode_writer.max_episodes_per_file")

        # todo: this should be replaced by the normal IO factory instead of manually loading IO here
        image_keys = experiment_config.key("episode_writer.image_keys")

        actual_image_keys = []
        actual_image_keys.extend(["observations_" + key for key in image_keys])
        actual_image_keys.extend(["next_observations_" + key for key in image_keys])

        data_folder = args.out

        dataset = JsonDrivenDataSource(data_folder)
        writer = dataset.writer(actual_image_keys, max_episodes_per_file, max_bytes_before_flush)

        with writer:
            for i in range(args.rollouts):
                experience, videos, episode_starts, diagnostics = rollout_worker.evaluation_rollout(num_episodes=1,
                                                                                                    render_videos=render)
                episodes = rollout_experience_into_episodes(experience, episode_starts)
                for episode in episodes:
                    writer.write_episode(episode)
    else:
        rewards = []
        for i in range(args.rollouts):
            experience, videos, episode_starts, diagnostics = rollout_worker.evaluation_rollout(num_episodes=1,
                                                                                                render_videos=render)
            print("Return", np.sum(experience["rewards"], axis=0))
            rewards.append(np.sum(experience["rewards"]))
        print("All Returns", rewards)
        print(f"Return is {np.mean(rewards)} +- {np.std(rewards)}")

    return


if __name__ == "__main__":
    run_trained_policy_cli(sys.argv[1:])
