
#Python Libs
import math 
import random
from collections import deque

#RL Libs
import torch                #Main ML lib
import gymnasium as gym     #RL environment

#Yea CUDA lets go FAST
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def observationToTensor(observation):
    directionTensor = torch.tensor(observation["direction"], device=device)
    directionOHE = torch.nn.functional.one_hot(directionTensor, num_classes=observation["direction"].n)
    flatImage = torch.flatten(torch.tensor(observation["image"], device=device))
    return torch.cat([directionOHE, flatImage])

class ReplayMemory:
    def __init__(self, size):
        self.memory = deque(maxlen=1000)

    def __len__(self):
        return len(self.memory)
        
    def append(self, observationTensor, action, nextObservationTensor, reward):
        self.append((observationTensor, action, nextObservationTensor, reward))

    def sample(self, size):
        return random.sample(self.memory, size)

class MinigridDQN:
    def __init__(self, env:gym.Env, hiddenSize:int = 100) -> None:
        self.directionShape = env.observation_space["direction"].n       #Number of directions (flattened into a one hot encoding)
        self.viewShape = math.prod(env.observation_space["image"].shape) #Number of items in observation space
        inputShape = self.directionShape + self.viewShape                #Combined Direction + Image Input Tensor size
        outputShape = env.action_space.n                                 #The Number of actions
        
        #Layers
        self.hiddenLayer = torch.nn.Linear(inputShape, hiddenSize)          
        self.outputLayer = torch.nn.Linear(hiddenSize, outputShape)
    
    def forward(self, tensorObservation):
        x = torch.nn.functional.relu(self.hiddenLayer(tensorObservation))
        x = torch.nn.functional.sigmoid(self.outputLayer(x))
        return x

        
        


