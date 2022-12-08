
import math
import random

import minigrid
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

#My Code
from DQN import device, MinigridDQN, ReplayMemory, observationToTensor, Transition

#Hyperparameters
num_episodes = 2000  # Number of episodes
BATCH_SIZE = 1024   # Number of experiences to train on
MEMORY_SIZE = 10000 #Size of the replay memory
GAMMA = 0.999       # Discount in (0, 1): 0 -> Prioritize short term rewards, 1 -> Prioritize Long Term
TARGET_UPDATE = 10  # Number of optimization steps before updating the target net

#epsilon greedy policy Hyperparameters
EPS_START = 0.9     # Starting epsilon (random action chance)
EPS_END = 0.2      # Ending epsilon (random action chance)
EPS_DECAY = 200     # Rate at which epsilon decays during training

optimizerType = optim.RMSprop #Type of optimizer


#Main objects
env = gym.make("MiniGrid-Empty-5x5-v0")    #Environment 
memory = ReplayMemory(MEMORY_SIZE)

policy_net = MinigridDQN(env).to(device)    #Agent Policy Net
target_net = MinigridDQN(env).to(device)    
target_net.load_state_dict(policy_net.state_dict()) #initialize the policy network weights to the target network
target_net.eval()

optimizer = optimizerType(policy_net.parameters())
 
if __name__ == "__main__":
    
    episodeLengths = []

    stepCounter = 0 #Counts the number of times we've selected actions while training
    for episode in range(num_episodes):
        print(episode, end=" ")
        observation, _ = env.reset()
        observationTensor = observationToTensor(env, observation)
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
            nextObservationTensor = None if terminated or truncated else observationToTensor(env, nextObservation)
            #Add the action to our memory
            rewardTensor = torch.tensor(reward, device=device, dtype=torch.float)
            memory.append(observationTensor, action, nextObservationTensor, rewardTensor)
            observationTensor = nextObservationTensor #Progress a step

            #Optimize the policy network
            if len(memory) > BATCH_SIZE: #Only perform an optimization step if we have enough memory for a batch
                transitions = memory.sample(BATCH_SIZE)
                
                # Convert list of transition tuples into a transition tuple of lists for the whole batch
                # [(o_1, no_1, ...), ..., (o_n, no_n, ...)] => ([o_1, ..., o_n], [no_1, ..., no_n], ...)
                batch = Transition(*zip(*transitions)) 

                #Mask for if the next state is a final state
                endStateMaskTuple = tuple(map(lambda s: s is not None, batch.nextObservationTensor))
                endStateMaskTensor = torch.tensor(endStateMaskTuple, device=device, dtype=torch.bool)

                #observations that aren't the end
                nonEndNextObservationTensors = torch.stack([s for s in batch.nextObservationTensor if s is not None])
                
                observationTensorBatch = torch.stack(batch.observationTensor)
                actionBatch = torch.stack(batch.action)
                rewardBatch = torch.stack(batch.reward)
                
                #predictedQs are the Q(s,a) for each state
                #policy net computes an array [Q(s, a1), Q(s, a2), ...] for all actions
                #When batched .gather(1, actionBatch) selects the Q value corresponding actual action taken 
                #  torch.gather([[Q(s1, a1), Q(s1, a2), ...],        1,  [a1, a2, ...]) = [Q(s1, a1), Q(s2, a2), ...] 
                #                [Q(s2, a1), Q(s2, a2), ...], ...],    
                predictedQs = policy_net(observationTensorBatch).gather(1, actionBatch.unsqueeze(1))
                
                #predictedVs are the V(s+1) = max_a(Q(s+1, a))
                predictedVs = torch.zeros(BATCH_SIZE, device=device)

                # [[Q(s1, a1), Q(s1, a2), ...],        .max(1) = 
                #  [Q(s2, a1), Q(s2, a2), ...], ...],
                predictedVs[endStateMaskTensor] = target_net(nonEndNextObservationTensors).max(1)[0].detach()
                
                #The expected total reward
                expectedQs = ((predictedVs * GAMMA) + rewardBatch).squeeze()

                #Temporal Difference Loss
                loss = torch.nn.SmoothL1Loss()(predictedQs, expectedQs)

                #Back propagate on policy net
                optimizer.zero_grad()
                loss.backward()
                for param in policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

            #We've lost or won
            if terminated or truncated:
                episodeLengths.append(episodeLength)
                break
            
            if episodeLength % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            episodeLength += 1
    env.close()
    
    torch.save(target_net.state_dict(), "trainedDQN")
    plt.plot(episodeLengths)
    plt.show()