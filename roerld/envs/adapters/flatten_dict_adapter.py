import gym


class FlattenDictAdapter(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        old_observation_space = self.observation_space

        result = {}
        for section_name, section in old_observation_space.spaces.items():
            for key, value in section.spaces.items():
                result[f"{section_name}.{key}"] = value
        self.observation_space = gym.spaces.Dict(result)

    def observation(self, observation):
        result = {}
        for section_name, section in observation.items():
            for key, value in section.items():
                result[f"{section_name}.{key}"] = value
        return result
