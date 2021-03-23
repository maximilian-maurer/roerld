import gym


class DictEnvAdapter(gym.Wrapper):
    def __init__(self, inner_env: gym.Env):
        super().__init__(inner_env)
        self.observation_space = gym.spaces.Dict({"observation": inner_env.observation_space})

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return {"observation": observation}, reward, done, info

    def reset(self):
        observation = self.env.reset()
        return {"observation": observation}
