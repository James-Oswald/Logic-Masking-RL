
import minigrid
import torch
import gymnasium as gym

from DQN import observationToTensor, MinigridDQN, device

if __name__ == "__main__":
    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")
    agent = MinigridDQN(env).to(device)
    agent.load_state_dict(torch.load("trainedDQN"))

    observation, _ = env.reset()
    for _ in range(1000):
        action = torch.argmax(agent(observationToTensor(env, observation))).cpu().numpy()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, _ = env.reset()

    env.close()