import time
from typing import Union

import gym
import numpy as np

from roerld.envs.features.subsampling_env import SubsamplingEnv
from roerld.learning_actors.learning_actor import LearningActor


class RolloutWorker:
    def __init__(self,
                 environment: Union[gym.Env, SubsamplingEnv],
                 actor: LearningActor,
                 max_episode_length,
                 local_render_mode,
                 eval_video_render_mode,
                 eval_video_height,
                 eval_video_width,
                 render_every_n_frames=1):
        self.env = environment
        self.action_dim = self.env.action_space.shape[0]
        self.action_clip_low = self.env.action_space.low
        self.action_clip_high = self.env.action_space.high
        self.max_episode_length = max_episode_length

        if self.max_episode_length is None:
            # query the environment
            if hasattr(self.env, "spec") and hasattr(self.env.spec, "max_episode_steps") \
                    and self.env.spec.max_episode_steps is not None:
                self.max_episode_length = self.env.spec.max_episode_steps
                print("Using max episode length of ", self.max_episode_length)
            else:
                raise ValueError("No max_episode_steps was provided, and it could not be deduced form the environment")


        self.local_render_mode = local_render_mode
        self.eval_video_height = eval_video_height
        self.eval_video_width = eval_video_width
        self.eval_video_render_mode = eval_video_render_mode
        self.actor = actor
        self.render_every_n_frames = render_every_n_frames

        self.env_supports_variable_render = hasattr(self.env, "render_image")

        self.is_subsampling = issubclass(type(self.env), SubsamplingEnv)
        if self.is_subsampling:
            print("Rollout worker is subsampling the environment.")

    def training_rollout(self, num_episodes, render_videos, passthrough_extra_info=None, fully_random=False):
        if passthrough_extra_info is None:
            passthrough_extra_info = {}
        #print("Starting Training Rollout")
        return self._rollout(num_episodes, False, render_videos, passthrough_extra_info, fully_random)

    def evaluation_rollout(self, num_episodes, render_videos, passthrough_extra_info=None):
        if passthrough_extra_info is None:
            passthrough_extra_info = {}
        #print("Starting Evaluation Rollout")
        return self._rollout(num_episodes, True, render_videos, passthrough_extra_info)

    def _rollout(self, num_episodes, is_evaluation, render_videos, extra_info=None, fully_random=False):
        #print(f"Rollout Perform:  ne={num_episodes}, info={extra_info}, is_eval={is_evaluation}, "
        #      f"render_videos={render_videos}, fully_random={fully_random}")
        if extra_info is None:
            extra_info = {}
        start_time = time.perf_counter()

        max_num_transitions = num_episodes * self.max_episode_length
        if self.is_subsampling:
            max_num_transitions *= self.env.max_samples_per_step()

        observation_dict = {
            "observations_" + name: np.zeros(
                shape=(max_num_transitions, *self.env.observation_space.spaces[name].shape))
            for name in self.env.observation_space.spaces.keys()
        }
        next_observation_dict = {
            "next_observations_" + name: np.zeros(
                shape=(max_num_transitions, *self.env.observation_space.spaces[name].shape))
            for name in self.env.observation_space.spaces.keys()
        }

        experience = {
            **observation_dict,
            **next_observation_dict,
            "actions": [],
            "rewards": [],
            "dones": [],
            "infos": []
        }

        if self.is_subsampling:
            experience["subsample_index"] = []

        videos = []

        did_render_in_epoch = self.local_render_mode is not None or render_videos
        key_start = "time_rollout" if not did_render_in_epoch else "time_rollout_with_render"
        diagnostics = {
            key_start + "_episode": [],
            key_start + "_sampling": [],
            key_start + "_rendering": [],
            key_start + "_action_selection": [],
            key_start + "_perSample_sampling": [],
            key_start + "_perSample_action_selection": [],
            key_start + "_perSample_rendering": [],
        }

        episode_starts = []
        transition_index = 0

        for eval_episode_index in range(num_episodes):
            time_episode_start = time.perf_counter()

            episode_return = 0
            episode_length = 0
            sampling_time = 0
            rendering_time = 0
            action_selection_time = 0
            episode_starts.append(len(experience["rewards"]))

            reset_start = time.perf_counter()
            observation = self.env.reset()
            sampling_time += reset_start - time.perf_counter()

            episode_video = []

            self.actor.episode_started()

            for i in range(self.max_episode_length):
                previous_observation = observation

                action_sel = time.perf_counter()
                if not fully_random:
                    action = self.actor.choose_action(observation, i, is_evaluation, extra_info)
                else:
                    action = self.env.action_space.sample()
                action = np.clip(action, a_min=self.action_clip_low, a_max=self.action_clip_high)
                action_selection_time += time.perf_counter() - action_sel

                sample = time.perf_counter()
                local_transitions = []
                if self.is_subsampling:
                    local_transitions.extend(self.env.subsample_step(action))
                else:
                    local_transitions.append(self.env.step(action))
                sampling_time += time.perf_counter() - sample

                for subsample_idx, transition in enumerate(local_transitions):
                    observation, reward, done, info = transition

                    episode_return += reward
                    episode_length += 1
                    for key in previous_observation:
                        experience["observations_" + key][transition_index] = previous_observation[key]
                    for key in observation:
                        experience["next_observations_" + key][transition_index] = observation[key]
                    experience["rewards"].append(reward)
                    experience["dones"].append(done)
                    experience["actions"].append(action)
                    experience["infos"].append(info)

                    if self.is_subsampling:
                        experience["subsample_index"].append(subsample_idx)
                    transition_index += 1

                    previous_observation = observation

                render_t = time.perf_counter()

                if self.local_render_mode is not None:
                    self.env.render(mode=self.local_render_mode)

                # always include the final result
                if render_videos and (i % self.render_every_n_frames == 0 or done):
                    additional_kwargs = {}
                    if self.eval_video_width is not None:
                        assert self.eval_video_height is not None
                        additional_kwargs.update({"width": self.eval_video_width,
                                                  "height": self.eval_video_height})
                    image = None
                    if self.env_supports_variable_render:
                        image = self.env.render_image(mode=self.eval_video_render_mode,
                                        **additional_kwargs)
                    else:
                        image = self.env.render(mode=self.eval_video_render_mode,
                                                         **additional_kwargs)
                    episode_video.append(image)
                rendering_time += time.perf_counter() - render_t
                if done:
                    break

            self.actor.episode_ended()

            if len(episode_video) > 0:
                videos.append(episode_video)

            diagnostics[key_start + "_sampling"].append(sampling_time)
            diagnostics[key_start + "_rendering"].append(rendering_time)
            diagnostics[key_start + "_action_selection"].append(action_selection_time)
            diagnostics[key_start + "_perSample_sampling"].append(np.array(sampling_time) / episode_length)
            diagnostics[key_start + "_perSample_rendering"].append(np.array(rendering_time) / episode_length)
            diagnostics[key_start + "_perSample_action_selection"].append(
                np.array(action_selection_time) / episode_length)

            diagnostics[key_start + "_episode"].append(time.perf_counter() - time_episode_start)

        videos = np.array(videos)

        end_time = time.perf_counter() - start_time
        diagnostics[key_start + "total_perEpisode_(fromBatch)"] = end_time / num_episodes

        for key in experience:
            if key in next_observation_dict or key in observation_dict:
                continue
            experience[key] = np.array(experience[key])

        for key in observation_dict:
            experience[key] = experience[key][:transition_index]

        for key in next_observation_dict:
            experience[key] = experience[key][:transition_index]

        return experience, videos, episode_starts, diagnostics
