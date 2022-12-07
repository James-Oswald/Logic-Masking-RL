
import math
import random

import minigrid
import gymnasium as gym

import torch
import torch.optim as optim
from DQN import device, MinigridDQN, ReplayMemory, observationToTensor

#Hyperparams
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
num_episodes = 50

#Main objects
env = gym.make("MiniGrid-Dynamic-Obstacles-8x8-v0", render_mode="human")    #Environment 
memory = ReplayMemory(10000)

policy_net = MinigridDQN(env).to(device)    #Agent Policy Net
target_net = MinigridDQN(env).to(device)    
target_net.load_state_dict(policy_net.state_dict()) #initialize the policy network weights to the target network
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
 
if __name__ == "__main__":
    
    stepCounter = 0 #Counts the number of times we've selected actions while training
    for _ in range(num_episodes):
        observation, _ = env.reset()
        observationTensor = observationToTensor(observation)
        episodeLength = 0
        while True:
            #Action Selection 
            epsilonThreshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * stepCounter / EPS_DECAY)
            stepCounter += 1
            if random.random() > epsilonThreshold:  
                with torch.no_grad():  
                    action = torch.argmax(policy_net(observationTensor)) #Select Best action according to policy net
            else:
                action = torch.tensor(env.action_space.sample(), device=device, dtype=torch.long) #Select random action
            
            #Perform the action
            nextObservation, reward, terminated, truncated, _ = env.step(action.item())
            nextObservationTensor = None if terminated or truncated else observationToTensor(nextObservation)
            #Add the action to our memory
            memory.append(observationTensor, action, nextObservation, reward)
            observationTensor = nextObservationTensor #Progress a step

            #Optimize the policy network
            if len(memory) > BATCH_SIZE: #Only perform an optimization step if we have enough memory for a batch
                transitions = memory.sample(BATCH_SIZE)
            

            if terminated or truncated:
                break
            episodeLength += 1


    env.close()