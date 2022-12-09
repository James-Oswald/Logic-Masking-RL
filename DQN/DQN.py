
#Python Libs
import math 
import random
from collections import namedtuple, deque

#RL Libs
import torch                #Main ML lib
import gymnasium as gym     #RL environment

#Yea CUDA lets go FAST
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

def observationToTensor(env, observation):
    directionTensor = torch.tensor(observation["direction"], device=device)
    directionOHE = torch.nn.functional.one_hot(directionTensor, num_classes=env.observation_space["direction"].n)
    flatImage = torch.flatten(torch.tensor(observation["image"], device=device))
    return torch.cat([directionOHE, flatImage]).type(torch.float)
    #return flatImage.type(torch.float)

Transition = namedtuple('Transition', ('observationTensor', 'action', 'nextObservationTensor', 'reward'))

class ReplayMemory:
    def __init__(self, size):
        self.memory = deque(maxlen=size)

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

    def __init__(self, env:gym.Env, hiddenSizes:"list[int]" = [256, 256, 100]) -> None:
        super(MinigridDQN, self).__init__()
        self.directionShape = env.observation_space["direction"].n       #Number of directions (flattened into a one hot encoding)
        self.viewShape = math.prod(env.observation_space["image"].shape) #Number of items in observation space
        inputShape = self.directionShape + self.viewShape                #Combined Direction + Image Input Tensor size
        #inputShape = self.viewShape 
        outputShape = env.action_space.n                                 #The Number of actions
        
        #Layers
        self.hiddenLayers = torch.nn.ModuleList()
        inputCatHiddenSizes = [inputShape] + hiddenSizes
        for i in range(1, len(inputCatHiddenSizes)):
            self.hiddenLayers.append(torch.nn.Linear(inputCatHiddenSizes[i-1], inputCatHiddenSizes[i]))
        self.outputLayer = torch.nn.Linear(hiddenSizes[-1], outputShape)
    
    def forward(self, x):
        for layer in self.hiddenLayers:
            x = torch.relu(layer(x))
        x = self.outputLayer(x)
        return x

        
        


