import gym


class DictEnvAdapter:
    def __init__(self, inner_env: gym.Env):
        self._inner_env = inner_env

    def step(self, action):
        observation, reward, done, info = self.inner_env.step(action)
        return {"observation": observation}, reward, done, info

    def reset(self):
        observation = self.inner_env.reset()
        return {"observation": observation}

    @property
    def inner_env(self):
        return self._inner_env

    @property
    def observation_space(self):
        return gym.spaces.Dict({"observation": self.inner_env.observation_space})

    def __getattr__(self, attribute):
        return getattr(self._inner_env, attribute)
