import amrl
import gymnasium as gym
import random
import numpy as np


if __name__ == "__main__":
    amrl.register_envs()
    env = gym.make("AMRL-MiniGrid-Unlock-2by2-v0", render_mode="human")
    observation = env.reset()
    for _ in range(1000):
        action = env.action_space.sample(np.array(env.action_mask(), dtype="int8"))
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()

    env.close()