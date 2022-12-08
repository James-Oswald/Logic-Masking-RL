
#Python Libs
import math 
import random
from collections import namedtuple, deque

#RL Libs
import torch                #Main ML lib
import gymnasium as gym     #RL environment

#Yea CUDA lets go FAST
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def observationToTensor(env, observation):
    directionTensor = torch.tensor(observation["direction"], device=device)
    directionOHE = torch.nn.functional.one_hot(directionTensor, num_classes=env.observation_space["direction"].n)
    flatImage = torch.flatten(torch.tensor(observation["image"], device=device))
    return torch.cat([directionOHE, flatImage]).type(torch.float)

Transition = namedtuple('Transition', ('observationTensor', 'action', 'nextObservationTensor', 'reward'))

class ReplayMemory:
    def __init__(self, size):
        self.memory = deque(maxlen=1000)

    def __len__(self):
        return len(self.memory)
        
    def append(self, observationTensor, action, nextObservationTensor, reward):
        self.memory.append(Transition(observationTensor, action, nextObservationTensor, reward))

    def sample(self, size):
        return random.sample(self.memory, size)

class MinigridDQN(torch.nn.Module):
    '''
        Computes Q values for all actions, returning an array
        [Q(s, action1), Q(s, action2), Q(s, action3)]
    '''

    def __init__(self, env:gym.Env, hiddenSize:int = 100) -> None:
        super(MinigridDQN, self).__init__()
        self.directionShape = env.observation_space["direction"].n       #Number of directions (flattened into a one hot encoding)
        self.viewShape = math.prod(env.observation_space["image"].shape) #Number of items in observation space
        inputShape = self.directionShape + self.viewShape                #Combined Direction + Image Input Tensor size
        outputShape = env.action_space.n                                 #The Number of actions
        
        #Layers
        self.hiddenLayer = torch.nn.Linear(inputShape, hiddenSize)          
        self.outputLayer = torch.nn.Linear(hiddenSize, outputShape)
    
    def forward(self, tensorObservation):
        x = torch.nn.functional.sigmoid(self.hiddenLayer(tensorObservation))
        x = self.outputLayer(x)
        return x

        
        


