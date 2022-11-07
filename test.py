import amrl
import gymnasium as gym
import random


if __name__ == "__main__":
    amrl.register_envs()
    env = gym.make("AMRL-MiniGrid-Unlock-2by2-v0", render_mode="human")
    observation = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()

    env.close()