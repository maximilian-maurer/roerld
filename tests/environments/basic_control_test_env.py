import gym
import numpy as np


class BasicControlEnvV0(gym.Env):
    def __init__(self):
        self.position = -5
        self.goal = self.velocity = 0
        self.velocity_low = -10
        self.velocity_high = 10
        self.position_low = -10
        self.position_high = 10
        self.time = 0

        self.action_low = -1
        self.action_high = 1

        self.reset()

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            "observation": gym.spaces.Box(low=np.asarray([-1, -1, -1]),
                                          high=np.asarray([1, 1, 1]),
                                          dtype=np.float32)
        })

    @property
    def action_space(self):
        return gym.spaces.Box(low=np.asarray([self.action_low]),
                              high=np.asarray([self.action_high]),
                              dtype=np.float32)

    def reset(self):
        self.goal = np.random.uniform(low=0, high=self.position_high)
        self.position = -5
        self.velocity = 0
        self.time = 0

        return self._observe()

    def render(self, mode="human", width=128, height=128):
        if mode == "bgr_array":
            # this dependency for this file is solely for the rendering output which is not desired in many use-cases
            # so it is only loaded in case the functionality is actually used
            import cv2
            obs = self._observe()
            img = np.zeros(shape=(50, 200, 3))

            x_on_img = int(obs["observation"][0] * 100 + 100)
            x_future_on_img = int((obs["observation"][0] + 0.1 * 10 * obs["observation"][1]) * 100 + 100)
            goal_on_img = int(obs["observation"][2] * 100 + 100)
            img = cv2.rectangle(img, (goal_on_img-1, 0), (goal_on_img+1, 50), (0, 0, 255), 2)
            img = cv2.rectangle(img, (x_on_img, 0), (x_on_img, 50), (255, 255, 255), 2)
            img = cv2.rectangle(img, (x_future_on_img, 0), (x_future_on_img, 50), (128, 64, 64), 2)
            return img

        raise NotImplementedError()

    def _observe(self):
        obs = np.asarray([self.position, self.velocity, self.goal])
        obs -= np.asarray([self.position_low, self.velocity_low, self.position_low])
        obs = obs / (np.asarray([self.position_high, self.velocity_high, self.position_high])
                     - np.asarray([self.position_low, self.velocity_low, self.position_low]))
        obs -= 0.5
        obs *= 2
        return {"observation": obs}

    def step(self, action):
        assert len(action) == 1
        action = np.clip(action, a_min=self.action_low, a_max=self.action_high)

        self.velocity += action[0]
        self.position = self.position + 0.1 * self.velocity
        self.time = self.time + 1
        self.position = np.clip(self.position, self.position_low, self.position_high)
        self.velocity = np.clip(self.velocity, self.velocity_low, self.velocity_high)

        reward = -(abs(self.position - self.goal) ** 2) - abs(self.velocity) * 0.1

        done = False

        if abs(self.position-self.goal) < 0.1 and abs(self.velocity) < 0.1:
            done = True
            reward = 500

        return self._observe(), reward, done, None


def register_basic_control_test_env():
    gym.register("BasicControlEnv-v1",
                 entry_point=BasicControlEnvV0)


if __name__ == "__main__":
    env = BasicControlEnvV0()
    env.reset()
    for i in range(100):
        action = [0]
        if i < 25:
            action = [0.5]
        elif i < 20:
            action = [-0.5]
        else:
            action = [0]
        o, r, d, info = env.step(np.asarray(action))

        print(o, r, d)

